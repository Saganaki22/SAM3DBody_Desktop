cmake_minimum_required(VERSION 3.18)
project(fast_sam_3dbody CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(WIN32)
    option(SAM3D_BUILD_TOOLS "Build CLI/offline tools" OFF)
    option(SAM3D_BUILD_RENDERER "Build the Linux GLX OpenGL renderer" OFF)
else()
    option(SAM3D_BUILD_TOOLS "Build CLI/offline tools" ON)
    option(SAM3D_BUILD_RENDERER "Build the Linux GLX OpenGL renderer" ON)
endif()

set(SAM3D_PLATFORM_LIBS "")
if(NOT WIN32)
    list(APPEND SAM3D_PLATFORM_LIBS m)
endif()

# ---------------------------------------------------------------------------
# Model files check
# ---------------------------------------------------------------------------
set(_ONNX_DIR  "${CMAKE_CURRENT_SOURCE_DIR}/onnx")
set(_ONNX_ZIP  "${CMAKE_CURRENT_SOURCE_DIR}/SAM3DBody-cpp-onnx-models.zip")
set(_SENTINEL  "${_ONNX_DIR}/pipeline.gguf")   # representative file

if(NOT EXISTS "${_SENTINEL}")
    if(EXISTS "${_ONNX_ZIP}")
        message(STATUS "")
        message(STATUS "  *** Model zip found: ${_ONNX_ZIP}")
        message(STATUS "  *** Extract it before running: unzip SAM3DBody-cpp-onnx-models.zip")
        message(STATUS "")
    else()
        message(WARNING
            "\n"
            "  ┌─────────────────────────────────────────────────────────────────┐\n"
            "  │  Model files not found!                                         │\n"
            "  │                                                                 │\n"
            "  │  Download SAM3DBody-cpp-onnx-models.zip from HuggingFace:       │\n"
            "  │  https://huggingface.co/AmmarkoV/SAM3DBody-cpp-onnx-models      │\n"
            "  │                                                                 │\n"
            "  │  Then extract it at the repo root:                              │\n"
            "  │    unzip SAM3DBody-cpp-onnx-models.zip                          │\n"
            "  │                                                                 │\n"
            "  │  The build will succeed but the runtime will fail without them. │\n"
            "  └─────────────────────────────────────────────────────────────────┘\n"
        )
    endif()
else()
    message(STATUS "Model files found in ${_ONNX_DIR}")
endif()

# ---------------------------------------------------------------------------
# Backbone quantization  (build-time default; overridable at runtime via --backbone)
# ---------------------------------------------------------------------------
# After running tools/quantize_backbone.py you can switch the baked-in default
# filename from backbone.onnx to backbone_int8.onnx so all three binaries pick
# up the quantized model without needing the --backbone flag every invocation.
# The original backbone.onnx is never removed — both coexist in onnx/.
#
# Usage:  cmake -DSAM3D_BACKBONE_QUANT=ON ..
# ---------------------------------------------------------------------------
option(SAM3D_BACKBONE_QUANT
    "Default to backbone_int8.onnx instead of backbone.onnx (run tools/quantize_backbone.py first)"
    OFF)
if(SAM3D_BACKBONE_QUANT)
    message(STATUS "SAM3D_BACKBONE_QUANT=ON — default backbone: backbone_int8.onnx")
endif()

# ---------------------------------------------------------------------------
# CUDA  (optional – enables ONNX Runtime CUDA EP and ggml CUDA backend)
# ---------------------------------------------------------------------------
option(SAM3D_USE_CUDA "Build CUDA acceleration when the CUDA Toolkit is available" ON)
option(SAM3D_ALLOW_UNSUPPORTED_CUDA_HOST_COMPILER
    "Pass -allow-unsupported-compiler to nvcc when the installed MSVC is newer than CUDA officially recognizes"
    ON)

if(SAM3D_USE_CUDA)
    find_package(CUDAToolkit QUIET)
else()
    set(CUDAToolkit_FOUND OFF)
endif()

if(CUDAToolkit_FOUND)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
        if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.8")
            set(CMAKE_CUDA_ARCHITECTURES "120" CACHE STRING "CUDA architectures")
        else()
            set(CMAKE_CUDA_ARCHITECTURES "86" CACHE STRING "CUDA architectures")
        endif()
    endif()
    if(MSVC AND SAM3D_ALLOW_UNSUPPORTED_CUDA_HOST_COMPILER)
        string(APPEND CMAKE_CUDA_FLAGS " -allow-unsupported-compiler")
    endif()
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(GGML_CUDA     ON  CACHE BOOL "" FORCE)
    set(GGML_CUDA_F16 ON  CACHE BOOL "" FORCE)
    set(WITH_CUDA     ON)
    message(STATUS "CUDA ${CUDAToolkit_VERSION} found (sm_${CMAKE_CUDA_ARCHITECTURES})")
else()
    set(GGML_CUDA OFF CACHE BOOL "" FORCE)
    set(WITH_CUDA OFF)
    message(STATUS "CUDA not found – CPU-only build")
endif()

# ---------------------------------------------------------------------------
# ggml  (provides GGUF loading + small FFN inference for MHR/camera heads)
# ---------------------------------------------------------------------------
include(FetchContent)
FetchContent_Declare(
    ggml
    GIT_REPOSITORY https://github.com/ggml-org/ggml.git
    GIT_TAG        master
    GIT_SHALLOW    TRUE
    GIT_PROGRESS   TRUE
)
set(GGML_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GGML_BLAS           OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP         OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(ggml)

# ---------------------------------------------------------------------------
# ONNX Runtime  (pre-built binaries from GitHub releases)
# ONNX_RUNTIME_DIR can be set via -DONNX_RUNTIME_DIR=/path/to/onnxruntime
# Otherwise we try to find it in common locations or download it.
# ---------------------------------------------------------------------------
if(NOT DEFINED ONNX_RUNTIME_DIR OR ONNX_RUNTIME_DIR STREQUAL "")
    # Try pkg-config / system install first
    find_package(onnxruntime QUIET)
    if(onnxruntime_FOUND)
        message(STATUS "Found system ONNX Runtime via CMake config")
        set(ORT_LIBS onnxruntime::onnxruntime)
        set(ORT_INCLUDE_DIRS "")
    else()
        # Try common install paths
        foreach(_try
            "/usr/local/lib/cmake/onnxruntime"
            "/opt/onnxruntime"
            "$ENV{HOME}/.local"
        )
            if(EXISTS "${_try}/include/onnxruntime_cxx_api.h")
                set(ONNX_RUNTIME_DIR "${_try}")
                break()
            endif()
        endforeach()

        if(ONNX_RUNTIME_DIR STREQUAL "" OR NOT DEFINED ONNX_RUNTIME_DIR)
            # Download pre-built GPU package
            set(SAM3D_ORT_VERSION "" CACHE STRING "ONNX Runtime version to download when ONNX_RUNTIME_DIR is unset")
            if(SAM3D_ORT_VERSION)
                set(_ORT_VERSION "${SAM3D_ORT_VERSION}")
            elseif(WITH_CUDA AND CUDAToolkit_VERSION VERSION_GREATER_EQUAL "13.0")
                set(_ORT_VERSION "1.26.0")
            else()
                set(_ORT_VERSION "1.20.1")
            endif()
            if(WIN32)
                if(WITH_CUDA)
                    if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "13.0")
                        set(_ORT_ARCHIVE "onnxruntime-win-x64-gpu_cuda13-${_ORT_VERSION}.zip")
                        set(_ORT_EXTRACTED "onnxruntime-win-x64-gpu_cuda13-${_ORT_VERSION}")
                    else()
                        set(_ORT_ARCHIVE "onnxruntime-win-x64-gpu-${_ORT_VERSION}.zip")
                        set(_ORT_EXTRACTED "onnxruntime-win-x64-gpu-${_ORT_VERSION}")
                    endif()
                else()
                    set(_ORT_ARCHIVE "onnxruntime-win-x64-${_ORT_VERSION}.zip")
                    set(_ORT_EXTRACTED "onnxruntime-win-x64-${_ORT_VERSION}")
                endif()
            else()
                if(WITH_CUDA)
                    if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "13.0")
                        set(_ORT_ARCHIVE "onnxruntime-linux-x64-gpu_cuda13-${_ORT_VERSION}.tgz")
                        set(_ORT_EXTRACTED "onnxruntime-linux-x64-gpu_cuda13-${_ORT_VERSION}")
                    else()
                        set(_ORT_ARCHIVE "onnxruntime-linux-x64-gpu-${_ORT_VERSION}.tgz")
                        set(_ORT_EXTRACTED "onnxruntime-linux-x64-gpu-${_ORT_VERSION}")
                    endif()
                else()
                    set(_ORT_ARCHIVE "onnxruntime-linux-x64-${_ORT_VERSION}.tgz")
                    set(_ORT_EXTRACTED "onnxruntime-linux-x64-${_ORT_VERSION}")
                endif()
            endif()
            set(_ORT_URL "https://github.com/microsoft/onnxruntime/releases/download/v${_ORT_VERSION}/${_ORT_ARCHIVE}")
            set(_ORT_DOWNLOAD_DIR "${CMAKE_BINARY_DIR}/onnxruntime_dl")
            # After extraction + rename, contents live directly in _ORT_DOWNLOAD_DIR
            set(ONNX_RUNTIME_DIR "${_ORT_DOWNLOAD_DIR}")

            if(NOT EXISTS "${_ORT_DOWNLOAD_DIR}")
                message(STATUS "Downloading ONNX Runtime ${_ORT_VERSION} …")
                file(DOWNLOAD "${_ORT_URL}" "${CMAKE_BINARY_DIR}/${_ORT_ARCHIVE}" SHOW_PROGRESS)
                execute_process(
                    COMMAND ${CMAKE_COMMAND} -E tar xf "${CMAKE_BINARY_DIR}/${_ORT_ARCHIVE}"
                    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
                )
                set(_ORT_EXTRACTED_DIR "${CMAKE_BINARY_DIR}/${_ORT_EXTRACTED}")
                if(NOT EXISTS "${_ORT_EXTRACTED_DIR}")
                    file(GLOB _ORT_CANDIDATES
                        LIST_DIRECTORIES true
                        "${CMAKE_BINARY_DIR}/onnxruntime-*-${_ORT_VERSION}")
                    foreach(_ORT_CANDIDATE IN LISTS _ORT_CANDIDATES)
                        if(IS_DIRECTORY "${_ORT_CANDIDATE}" AND EXISTS "${_ORT_CANDIDATE}/include/onnxruntime_cxx_api.h")
                            set(_ORT_EXTRACTED_DIR "${_ORT_CANDIDATE}")
                            break()
                        endif()
                    endforeach()
                endif()
                if(NOT EXISTS "${_ORT_EXTRACTED_DIR}")
                    message(FATAL_ERROR "Could not find extracted ONNX Runtime directory from ${_ORT_ARCHIVE}")
                endif()
                file(RENAME
                    "${_ORT_EXTRACTED_DIR}"
                    "${_ORT_DOWNLOAD_DIR}"
                )
            endif()
        endif()
    endif()
endif()

if(NOT TARGET onnxruntime::onnxruntime)
    set(ORT_INCLUDE_DIRS "${ONNX_RUNTIME_DIR}/include")
    set(ORT_LIB_DIR      "${ONNX_RUNTIME_DIR}/lib")
    add_library(onnxruntime::onnxruntime SHARED IMPORTED)
    if(WIN32)
        find_library(ORT_IMPLIB onnxruntime PATHS "${ORT_LIB_DIR}" NO_DEFAULT_PATH REQUIRED)
        find_file(ORT_DLL onnxruntime.dll PATHS "${ORT_LIB_DIR}" NO_DEFAULT_PATH)
        file(GLOB ORT_RUNTIME_DLLS CONFIGURE_DEPENDS "${ORT_LIB_DIR}/onnxruntime*.dll")
        set_target_properties(onnxruntime::onnxruntime PROPERTIES
            IMPORTED_IMPLIB "${ORT_IMPLIB}"
            INTERFACE_INCLUDE_DIRECTORIES "${ORT_INCLUDE_DIRS}"
        )
        if(ORT_DLL)
            set_target_properties(onnxruntime::onnxruntime PROPERTIES
                IMPORTED_LOCATION "${ORT_DLL}"
            )
        endif()
    else()
        find_library(ORT_LIB onnxruntime PATHS "${ORT_LIB_DIR}" NO_DEFAULT_PATH REQUIRED)
        set_target_properties(onnxruntime::onnxruntime PROPERTIES
            IMPORTED_LOCATION "${ORT_LIB}"
            INTERFACE_INCLUDE_DIRECTORIES "${ORT_INCLUDE_DIRS}"
        )
    endif()
    set(ORT_LIBS onnxruntime::onnxruntime)
    message(STATUS "ONNX Runtime at ${ONNX_RUNTIME_DIR}")
endif()

set(SAM3D_CUDA_RUNTIME_DLLS)
if(WIN32 AND WITH_CUDA)
    option(SAM3D_BUNDLE_CUDA_RUNTIME_DLLS
        "Copy CUDA/cuDNN runtime DLLs next to fast_sam_3dbody.dll for app bundling"
        ON)
    set(SAM3D_CUDNN_BIN_DIR "" CACHE PATH "Directory containing cuDNN cudnn*.dll files to bundle")

    if(SAM3D_BUNDLE_CUDA_RUNTIME_DLLS)
        foreach(_SAM3D_CUDA_DLL_PATTERN
                cublas64_*.dll
                cublasLt64_*.dll
                cufft64_*.dll
                cudart64_*.dll)
            file(GLOB _SAM3D_CUDA_DLLS
                "${CUDAToolkit_BIN_DIR}/${_SAM3D_CUDA_DLL_PATTERN}"
                "${CUDAToolkit_BIN_DIR}/x64/${_SAM3D_CUDA_DLL_PATTERN}")
            list(APPEND SAM3D_CUDA_RUNTIME_DLLS ${_SAM3D_CUDA_DLLS})
        endforeach()

        if(SAM3D_CUDNN_BIN_DIR)
            file(GLOB _SAM3D_CUDNN_DLLS "${SAM3D_CUDNN_BIN_DIR}/cudnn*.dll")
            list(APPEND SAM3D_CUDA_RUNTIME_DLLS ${_SAM3D_CUDNN_DLLS})
        endif()

        if(SAM3D_CUDA_RUNTIME_DLLS)
            list(REMOVE_DUPLICATES SAM3D_CUDA_RUNTIME_DLLS)
        endif()
    endif()
endif()

# ---------------------------------------------------------------------------
# OpenCV
# ---------------------------------------------------------------------------
set(SAM3D_OPENCV_COMPONENTS core imgproc dnn videoio)
if(SAM3D_BUILD_TOOLS OR SAM3D_BUILD_RENDERER)
    list(APPEND SAM3D_OPENCV_COMPONENTS videoio highgui video imgcodecs)
endif()
find_package(OpenCV REQUIRED COMPONENTS ${SAM3D_OPENCV_COMPONENTS})
message(STATUS "OpenCV ${OpenCV_VERSION}")

set(GRAPHICS_ENGINE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/GraphicsEngine)

# ---------------------------------------------------------------------------
# AmMatrix  (built as subdirectory)
# ---------------------------------------------------------------------------
add_subdirectory(AmMatrix)

# ---------------------------------------------------------------------------
# fast_sam_3dbody  library
# ---------------------------------------------------------------------------
set(MOCAP_DIR ${GRAPHICS_ENGINE_DIR}/MotionCaptureLoader)
set(TRAJ_DIR  ${GRAPHICS_ENGINE_DIR}/TrajectoryParser)

add_library(fast_sam_3dbody SHARED
    src/fast_sam_3dbody.cpp
    src/fast_sam_3dbody_capi.cpp
    src/bvh_writer.cpp
    ${GRAPHICS_ENGINE_DIR}/ModelLoader/model_loader_tri.c
    ${GRAPHICS_ENGINE_DIR}/ModelLoader/model_loader_transform_joints.c
    # MotionCaptureLoader (BVH I/O) ------------------------------------------------
    ${MOCAP_DIR}/bvh_loader.c
    ${MOCAP_DIR}/calculate/bvh_project.c
    ${MOCAP_DIR}/calculate/bvh_transform.c
    ${MOCAP_DIR}/calculate/bvh_to_tri_pose.c
    ${MOCAP_DIR}/calculate/smoothing.c
    ${MOCAP_DIR}/export/bvh_to_bvh.c
    ${MOCAP_DIR}/edit/bvh_cut_paste.c
    ${MOCAP_DIR}/edit/bvh_rename.c
    ${MOCAP_DIR}/import/fromBVH.c
    ${TRAJ_DIR}/InputParser_C.c
)

target_include_directories(fast_sam_3dbody PUBLIC
    src
    ${GRAPHICS_ENGINE_DIR}
    ${MOCAP_DIR}
    ${TRAJ_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/AmMatrix
    ${ggml_SOURCE_DIR}/include
    ${ORT_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(fast_sam_3dbody PUBLIC
    ggml
    ${ORT_LIBS}
    ${OpenCV_LIBS}
    AmMatrix
    ${SAM3D_PLATFORM_LIBS}
)

target_compile_definitions(fast_sam_3dbody PRIVATE FSB_BUILDING_DLL)

if(WIN32 AND ORT_DLL)
    add_custom_command(TARGET fast_sam_3dbody POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${ORT_DLL}"
                "$<TARGET_FILE_DIR:fast_sam_3dbody>"
        COMMENT "Copying ONNX Runtime DLL next to fast_sam_3dbody.dll")
endif()

if(WIN32 AND ORT_RUNTIME_DLLS)
    add_custom_command(TARGET fast_sam_3dbody POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${ORT_RUNTIME_DLLS}
                "$<TARGET_FILE_DIR:fast_sam_3dbody>"
        COMMAND_EXPAND_LISTS
        COMMENT "Copying ONNX Runtime DLLs next to fast_sam_3dbody.dll")
endif()

if(WIN32 AND SAM3D_CUDA_RUNTIME_DLLS)
    add_custom_command(TARGET fast_sam_3dbody POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${SAM3D_CUDA_RUNTIME_DLLS}
                "$<TARGET_FILE_DIR:fast_sam_3dbody>"
        COMMAND_EXPAND_LISTS
        COMMENT "Copying CUDA/cuDNN runtime DLLs next to fast_sam_3dbody.dll")
endif()

if(WIN32)
    set(SAM3D_OPENCV_PLUGIN_DLLS)
    foreach(_SAM3D_OPENCV_BIN_DIR
            "${OpenCV_DIR}/bin"
            "${OpenCV_DIR}/${OpenCV_ARCH}/${OpenCV_RUNTIME}/bin")
        file(GLOB _SAM3D_OPENCV_PLUGIN_DLLS
            "${_SAM3D_OPENCV_BIN_DIR}/opencv_videoio*.dll")
        list(APPEND SAM3D_OPENCV_PLUGIN_DLLS ${_SAM3D_OPENCV_PLUGIN_DLLS})
    endforeach()
    if(SAM3D_OPENCV_PLUGIN_DLLS)
        list(REMOVE_DUPLICATES SAM3D_OPENCV_PLUGIN_DLLS)
        add_custom_command(TARGET fast_sam_3dbody POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    ${SAM3D_OPENCV_PLUGIN_DLLS}
                    $<TARGET_FILE_DIR:fast_sam_3dbody>
            COMMAND_EXPAND_LISTS
            COMMENT "Copying OpenCV video plugins next to fast_sam_3dbody.dll")
    endif()

    add_custom_command(TARGET fast_sam_3dbody POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_RUNTIME_DLLS:fast_sam_3dbody>
                $<TARGET_FILE_DIR:fast_sam_3dbody>
        COMMAND_EXPAND_LISTS
        COMMENT "Copying runtime DLL dependencies next to fast_sam_3dbody.dll")
endif()

if(WITH_CUDA)
    target_sources(fast_sam_3dbody PRIVATE src/mhr_lbs_cuda.cu)
    target_compile_definitions(fast_sam_3dbody PUBLIC GGML_USE_CUDA FSB_CUDA)
    target_link_libraries(fast_sam_3dbody PUBLIC CUDA::cudart)
endif()

if(SAM3D_BACKBONE_QUANT)
    target_compile_definitions(fast_sam_3dbody PUBLIC SAM3D_BACKBONE_QUANT)
endif()

target_compile_options(fast_sam_3dbody PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wno-unused-parameter -O3 -march=native>
)

# ---------------------------------------------------------------------------
# fast_sam_3dbody_run  executable
# ---------------------------------------------------------------------------
if(SAM3D_BUILD_TOOLS)
add_executable(fast_sam_3dbody_run src/main.cpp)

target_link_libraries(fast_sam_3dbody_run PRIVATE fast_sam_3dbody)

target_compile_options(fast_sam_3dbody_run PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wno-unused-parameter -O3 -march=native>
)
endif()

# ---------------------------------------------------------------------------
# fast_sam_3dbody_render  executable  (OpenGL overlay renderer)
# ---------------------------------------------------------------------------
if(SAM3D_BUILD_RENDERER)
find_package(OpenGL REQUIRED)
find_package(GLEW   REQUIRED)

add_executable(fast_sam_3dbody_render
    render/fast_sam_3dbody_render.cpp
    ${GRAPHICS_ENGINE_DIR}/System/glx3.c
    ${GRAPHICS_ENGINE_DIR}/ModelLoader/model_loader_tri.c
    ${GRAPHICS_ENGINE_DIR}/ModelLoader/model_loader_transform_joints.c
)

target_compile_definitions(fast_sam_3dbody_render PRIVATE USE_GLEW)

target_include_directories(fast_sam_3dbody_render PRIVATE
    src
    ${GRAPHICS_ENGINE_DIR}
    ${ORT_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${GLEW_INCLUDE_DIRS}
)

target_link_libraries(fast_sam_3dbody_render PRIVATE
    fast_sam_3dbody
    OpenGL::GL
    GLEW::GLEW
    ${OpenCV_LIBS}
    X11
    rt
    m
    AmMatrix
)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(fast_sam_3dbody_render PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wno-unused-parameter -g -O0>
        $<$<COMPILE_LANGUAGE:C>:-Wall -Wno-unused-parameter -g -O0>
    )
else()
    target_compile_options(fast_sam_3dbody_render PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wno-unused-parameter -O3 -march=native>
        $<$<COMPILE_LANGUAGE:C>:-Wall -Wno-unused-parameter -O2>
    )
endif()
endif()

# ---------------------------------------------------------------------------
# offline_sam_3dbody_render  executable
#
# Multi-pass offline BVH extractor.  Same fsb::Pipeline + BVHWriter as the
# live binaries; no OpenGL dependency (this tool never opens a window).
# Only the per-pass orchestration code is unique to this target.
# ---------------------------------------------------------------------------
if(SAM3D_BUILD_TOOLS)
add_executable(offline_sam_3dbody_render
    render/offline_sam_3dbody_render.cpp
)

target_include_directories(offline_sam_3dbody_render PRIVATE
    src
    ${GRAPHICS_ENGINE_DIR}
    ${ORT_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(offline_sam_3dbody_render PRIVATE
    fast_sam_3dbody
    ${OpenCV_LIBS}
    ${SAM3D_PLATFORM_LIBS}
)

target_compile_options(offline_sam_3dbody_render PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wno-unused-parameter -O3 -march=native>
)
endif()

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
message(STATUS "")
message(STATUS "=== fast_sam_3dbody build ===")
message(STATUS "  CUDA support  : ${WITH_CUDA}")
message(STATUS "  ggml CUDA     : ${GGML_CUDA}")
message(STATUS "  OpenCV        : ${OpenCV_VERSION}")
message(STATUS "  ORT include   : ${ORT_INCLUDE_DIRS}")
message(STATUS "  Build tools   : ${SAM3D_BUILD_TOOLS}")
message(STATUS "  Build renderer: ${SAM3D_BUILD_RENDERER}")
message(STATUS "  Targets: fast_sam_3dbody (lib)")
