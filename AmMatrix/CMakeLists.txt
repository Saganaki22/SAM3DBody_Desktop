cmake_minimum_required( VERSION 3.5 )
project( AmMatrix )
 
#Make fast, lean and platform independent binaries..
if(NOT MSVC)
set(CMAKE_CXX_FLAGS "-s -O3 -fPIC -march=native -mtune=native")
set(CMAKE_C_FLAGS "-s -O3 -fPIC -march=native -mtune=native")
endif()

set(AMMATRIX_PLATFORM_LIBS "")
if(NOT WIN32)
    list(APPEND AMMATRIX_PLATFORM_LIBS m rt)
endif()

OPTION(INTEL_OPTIMIZATIONS OFF)

if (INTEL_OPTIMIZATIONS)
add_definitions(-DINTEL_OPTIMIZATIONS)
endif(INTEL_OPTIMIZATIONS)

add_library(
            AmMatrix STATIC
            collisions.c
            quaternions.c
            matrix3x3Tools.c
            matrix4x4Tools.c
            matrixCalculations.c
            matrixCalculations.h
            matrixMultiplicationOptimization.c
            matrixMultiplicationOptimization.h
            matrixOpenGL.c
            matrixOpenGL.h 
            ocvStaging.c
            ocvStaging.h
            solveHomography.c
            solveLinearSystemGJ.c 
            simpleRenderer.c
            solids.c
           )
target_link_libraries(AmMatrix ${AMMATRIX_PLATFORM_LIBS} )

if(NOT WIN32)
add_subdirectory(TestCPUOptimizedInstructionSet/)
endif()

set_target_properties(AmMatrix PROPERTIES
                       DEBUG_POSTFIX "D"
                       ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
                       LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
                       RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
                      )
