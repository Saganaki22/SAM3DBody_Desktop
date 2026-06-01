# SAM3DBody Desktop

![Platform](https://img.shields.io/badge/platform-Windows%20x64-0078D4)
![GPU](https://img.shields.io/badge/GPU-NVIDIA%20CUDA%2013-76B900)
![UI](https://img.shields.io/badge/UI-Tauri%202%20%2B%20Three.js-24C8DB)
![Engine](https://img.shields.io/badge/engine-C%2B%2B%20DLL%20via%20C%20API-00599C)
![Build](https://img.shields.io/badge/build-MSVC%202022-success)

SAM3DBody Desktop `0.1.1` is a Windows Tauri app for running the `SAM3DBody-cpp`
native engine in-process. It does not launch the CLI as a sidecar. The UI calls
Rust commands, Rust loads `fast_sam_3dbody.dll` with `libloading`, and the DLL
executes the C++/CUDA inference pipeline through a plain C ABI.

## What It Does

- Loads the bundled CUDA engine DLL, or a custom `fast_sam_3dbody.dll`.
- Finds bundled or manually downloaded model files without requiring an HF token.
- Supports Browse buttons and glowing drag/drop targets for image, image-result
  JSON, and video inputs.
- Processes images and shows the source image with aligned boxes, optional
  landmark points, and an optional MHR-70 skeleton overlay.
- Provides preview zoom, pan, and reset for inspecting image overlays.
- Shows all detected image meshes in the image 3D view. Multi-person image
  results keep image left-right order and use `pred_cam_t` plus bbox size to
  place nearer people forward and smaller background detections farther back.
- Exports a static one-frame image BVH through the native BVH writer when the
  original image is available. This is useful for a rest/static pose skeleton;
  it is not temporal motion unless the input is a video.
- Toggles image bbox, image landmarks, Image Skeleton, 3D Skeleton, 3D Body,
  mesh wireframe, floor grid, and auto-rotation in the UI.
- Provides a grid size slider from 8 to 32 divisions for image and video 3D
  previews.
- Preserves video mesh motion in 3D space using the extended video-preview C
  callback. When `pred_cam_t` is available, the preview uses it as the same root
  translation source that upstream writes into BVH; detector bboxes are only a
  fallback and identity aid.
- Exports the current image mesh as `.obj` or `.glb` with a save dialog and
  success toast.
- Saves image run JSON with real `mesh_vertices`, so that JSON can be imported
  later to recreate the image 3D result without running inference again. Older
  JSON that only says `"18439 vertices"` cannot recreate the mesh.
- Processes videos through OpenCV's FFmpeg backend in-process.
- Shows the input video beside the 3D preview.
- Optional video `3D Body Overlay` renders the generated body mesh directly over
  the video preview. New bundled DLLs stream projected 70-point `kps_2d` data so
  the overlay is fitted to the model projection instead of only to the detector
  bbox; older custom DLLs fall back to bbox fitting.
- Keeps separate retained image and video 3D preview states so switching tabs
  does not wipe the previous result.
- Streams video mesh frames while export is running, then allows playback with a
  seek bar, loop toggle, and 0.25x, 0.5x, 1x, 1.25x, and 2x viewer-only speeds.
- Generates video motion into an internal cache first, then enables Extract
  Motion so users can choose when and where to save BVH files.
- Multi-person BVH export writes one BVH file per subject automatically, using
  the upstream `BVHWriter` naming convention: `name_0.bvh`, `name_1.bvh`, etc.
- Supports full body plus hands from the native MHR-70 output. The diagnostic
  CSV writes all 70 3D keypoints per person and frame.
- Includes upstream Butterworth temporal smoothing controls for video motion:
  keypoints, body pose, hand pose, and root translation can be smoothed before
  BVH/preview output; root rotation can optionally use the quaternion filter.
- Provides an Abort button for video exports. The UI stops accepting late
  preview/result updates immediately; the current native CUDA/OpenCV frame may
  still finish because the upstream C API has no hard cancellation hook yet.
- Optionally writes a diagnostic 70-keypoint CSV for video runs.
- Includes collapsible hardware telemetry with 250 ms, 500 ms, 1000 ms, and
  1500 ms polling options.
- Keeps result JSON/text collapsed by default, with themed copy and save buttons.
  Large JSON output is preview-truncated when expanded so the UI remains fast;
  Copy and Save still use the complete text.
- Opens model download links, the footer GitHub link, and the build release link
  in the user's default browser.
- Disables the browser/WebView context menu inside the app window.
- Includes a settings popover with persistent Dark/Light theme buttons, accent
  color choices, and a UI scale slider. The popover itself is not scaled while
  the main UI is. The settings popover also links to the Apache-2.0 license.

## Supported Systems

Primary target:

- Windows 11 x64
- Minimum app window size: 1180 x 720
- Visual Studio 2022 MSVC toolchain
- NVIDIA GPU with a driver new enough for CUDA 13 runtime DLLs
- At least 8 GB VRAM for the CUDA path; observed peak usage is about 7.2 GB
  during model load/inference with the bundled model set.
- RTX 50-series tested path, including RTX 5090 / `sm_120`
- Tauri 2 desktop runtime / WebView2

Likely compatible:

- Windows 10 x64 with current WebView2 and NVIDIA drivers
- Other CUDA-capable NVIDIA GPUs if the native build includes their architecture

Limited / not packaged here:

- CPU-only Windows can work only with the separate `backbone_fp32.onnx` model and
  `cuda_device = -1`, but it is slow and video is not practical.
- Linux can build the upstream engine, but this desktop bundle is Windows-first.
- macOS is not a practical target for the CUDA build.

## Architecture

```text
Tauri Web UI
  -> Rust command layer
    -> libloading + Windows DLL search path setup
      -> fast_sam_3dbody.dll C API
        -> ONNX Runtime CUDA / ggml / OpenCV / native LBS
          -> image mesh, OBJ export, video BVH export
```

The important boundary is the C API in `src/fast_sam_3dbody_capi.h`. The UI
never depends on C++ symbols directly, and the app avoids a CLI sidecar process.
Long-running image and video jobs are executed on Rust blocking worker threads so
the WebView should stay responsive while CUDA/OpenCV work is running.
For video previews, newer DLLs expose `fsb_process_video_mesh_overlay`, which
adds frame size, bbox, camera translation, 70 projected 2D keypoints, 70 3D
keypoints, and mesh vertices to each streamed person. The desktop app uses this
for the video overlay and falls back through `fsb_process_video_mesh_tracked` and
`fsb_process_video_mesh_ex` when a custom older DLL is loaded.

Motion export semantics:

- Video BVH uses the upstream `BVHWriter` directly. Its root translation comes
  from `pred_cam_t`, the same native field used by the original repo.
- The 3D viewer is a diagnostic preview, not a calibrated scene reconstruction.
  Monocular video does not provide a true floor plane or metric multi-person
  spacing. The desktop viewer uses stable native track IDs plus image-space bbox
  separation to keep distinct people visually separated while preserving the
  per-person native mesh pose.
- Multi-person video and static-image BVH export both write one file per subject
  automatically. Pick one base filename in the UI; the writer creates numbered
  subject files beside it.
- Static image BVH is a one-frame BVH. It can carry a pose/skeleton for DCC
  tools, but it cannot invent temporal motion because there is only one frame.
- Butterworth smoothing is only meaningful for video. In the desktop DLL path it
  is applied before CSV/BVH writing and before streamed 3D preview callbacks, so
  saved motion and preview motion match.
- The 70-keypoint diagnostic CSV is for inspection and downstream tooling. The
  actual BVH is generated from the native pose/body/hand parameters, not by
  guessing a skeleton from 2D overlay marks.

Image overlay semantics:

- The bbox is only the detector rectangle.
- The image skeleton overlay comes from model keypoints when `kps_2d` is
  available. If the body/keypoint model path is skipped or unavailable, the UI
  can fall back to the 17-point YOLO detector rig.
- The 3D skeleton overlay comes from `kps_3d` when the body model and
  `keypoint_mapping.bin` are loaded.
- If a source image crops out legs or feet, the model may hallucinate lower-body
  pose; the viewer intentionally does not "fix" that because it would corrupt
  normal full-body results.

## Runtime Files

The app expects these resources either bundled beside the release executable or
discoverable from the repo/app data directory:

```text
resources/engine/fast_sam_3dbody.dll
resources/engine/onnxruntime.dll
resources/engine/onnxruntime_providers_cuda.dll
resources/engine/opencv_world4100.dll
resources/engine/opencv_videoio_ffmpeg4100_64.dll
resources/engine/cuda + cudnn runtime DLLs
resources/templates/body.bvh
resources/templates/body_mesh.tri
onnx/backbone.onnx
onnx/backbone.onnx.data
onnx/decoder.onnx
onnx/yolo.onnx
onnx/pipeline.gguf
onnx/body_model.lbs
onnx/correctives.bin
onnx/keypoint_mapping.bin
```

Place an already extracted `onnx/` folder next to the portable app or repo root.
The UI can also browse to a custom ONNX folder, GGUF file, YOLO file, backbone,
LBS body model, and BVH template. The links below are included for manual model
setup.

Manual downloads:

| File | Link |
| --- | --- |
| All-in-one CUDA zip | [SAM3DBody-cpp-onnx-models.zip](https://huggingface.co/AmmarkoV/SAM3DBody-cpp-onnx-models/resolve/main/SAM3DBody-cpp-onnx-models.zip) |
| `backbone.onnx` | [download](https://huggingface.co/AmmarkoV/SAM3DBody-cpp-onnx-models/resolve/main/backbone.onnx) |
| `backbone.onnx.data` | [download](https://huggingface.co/AmmarkoV/SAM3DBody-cpp-onnx-models/resolve/main/backbone.onnx.data) |
| `decoder.onnx` | [download](https://huggingface.co/AmmarkoV/SAM3DBody-cpp-onnx-models/resolve/main/decoder.onnx) |
| `yolo.onnx` | [download](https://huggingface.co/AmmarkoV/SAM3DBody-cpp-onnx-models/resolve/main/yolo.onnx) |
| `pipeline.gguf` | [download](https://huggingface.co/AmmarkoV/SAM3DBody-cpp-onnx-models/resolve/main/pipeline.gguf) |
| `body_model.lbs` | [download](https://huggingface.co/AmmarkoV/SAM3DBody-cpp-onnx-models/resolve/main/body_model.lbs) |
| `correctives.bin` | [download](https://huggingface.co/AmmarkoV/SAM3DBody-cpp-onnx-models/resolve/main/correctives.bin) |
| `keypoint_mapping.bin` | [download](https://huggingface.co/AmmarkoV/SAM3DBody-cpp-onnx-models/resolve/main/keypoint_mapping.bin) |
| `backbone_fp32.onnx` | [download](https://huggingface.co/AmmarkoV/SAM3DBody-cpp-onnx-models/resolve/main/backbone_fp32.onnx) |
| `backbone_fp32.onnx.data` | [download](https://huggingface.co/AmmarkoV/SAM3DBody-cpp-onnx-models/resolve/main/backbone_fp32.onnx.data) |

## Build Requirements

Install:

- Visual Studio 2022 Build Tools with MSVC C++ workload
- CMake
- Ninja
- Node.js 20+
- Rust stable
- OpenCV 4.10 Windows build, `x64/vc16`
- NVIDIA driver plus CUDA/cuDNN runtime matching the configured CMake build

## Build Native Engine

From the repository root:

```powershell
$opencv = "$env:USERPROFILE\.codex\deps\opencv-4.10.0\opencv\build"
$cudnn = "$env:APPDATA\Python\Python314\site-packages\nvidia\cudnn\bin"
$vsdev = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat"
$ninja = "$env:APPDATA\Python\Python314\Scripts\ninja.exe"

cmd /c "`"$vsdev`" -arch=x64 && cmake -S . -B build\windows-cuda -G Ninja `
  -DCMAKE_MAKE_PROGRAM=`"$ninja`" `
  -DCMAKE_BUILD_TYPE=Release `
  -DSAM3D_USE_CUDA=ON `
  -DCMAKE_CUDA_ARCHITECTURES=120 `
  -DSAM3D_BUILD_TOOLS=OFF `
  -DSAM3D_BUILD_RENDERER=OFF `
  -DOpenCV_DIR=`"$opencv`" `
  -DOpenCV_ARCH=x64 `
  -DOpenCV_RUNTIME=vc16 `
  -DSAM3D_CUDNN_BIN_DIR=`"$cudnn`""

cmd /c "`"$vsdev`" -arch=x64 && cmake --build build\windows-cuda -j"
```

The output DLL and native dependencies land in `build/windows-cuda/`.

## Build Desktop App

```powershell
cd desktop
npm install
npm run build:vite

cd src-tauri
cargo check

cd ..
npm run build
```

`npm run build` creates:

```text
desktop/src-tauri/target/release/sam3dbody-desktop.exe
desktop/src-tauri/target/release/bundle/nsis/SAM3DBody Desktop_0.1.1_x64-setup.exe
desktop/src-tauri/target/release/bundle/msi/SAM3DBody Desktop_0.1.1_x64_en-US.msi
```

For a portable folder, copy the release exe, `target/release/resources/`, and
the extracted `onnx/` model folder into one directory.

For fast local release iteration without building MSI/NSIS installers:

```powershell
cd desktop
npm run build:vite
npx tauri build --no-bundle
```

The no-bundle executable is:

```text
desktop/src-tauri/target/release/sam3dbody-desktop.exe
```

`--no-bundle` only builds the raw release executable. It does not create the MSI
or NSIS installer, and it does not package a distributable installer layout. Use
the full `npm run build` / `npx tauri build` path when you need a bundled
installer.

To refresh the portable CUDA folder after a no-bundle build:

```powershell
$root = (Resolve-Path '..').Path
$portable = Join-Path $root 'desktop\portable-cuda'
Copy-Item "$root\desktop\src-tauri\target\release\sam3dbody-desktop.exe" `
  "$portable\sam3dbody-desktop.exe" -Force
robocopy "$root\desktop\src-tauri\target\release\resources" `
  "$portable\resources" /E /NFL /NDL /NJH /NJS /NP
robocopy "$root\onnx" "$portable\onnx" /E /NFL /NDL /NJH /NJS /NP
```

## Verification

Useful checks:

```powershell
cd desktop
npm run build:vite
cd src-tauri
cargo check

cd ..\..
.\build\windows-cuda\fsb_process_probe.exe
.\build\windows-cuda\fsb_mesh_probe.exe
.\build\windows-cuda\fsb_video_probe.exe
```

Expected behavior:

- Image probe detects a person and returns keypoints.
- Mesh probe returns `18439` vertices.
- Video probe writes `sample_motion_0.bvh`.
- The desktop app loads without a `127.0.0.1` page only when built through
  Tauri (`npx tauri build --no-bundle` or `npm run build`), not plain
  `cargo build --release`.
- `npm run build:vite` may warn about a roughly 500 KB JavaScript chunk because
  Three.js is bundled. That warning is not a failed build.

## Notes

- The blank console window is removed in release builds by using the Windows GUI
  subsystem, and child hardware polling commands are launched with no console
  window.
- OpenCV is used for video decode because it gives a stable in-process FFmpeg
  backend without shipping an external `ffmpeg.exe`.
- Tauri's asset protocol is enabled for user media previews, so selected videos
  can be shown in the WebView while processing/export continues through the
  native engine.
- BVH is the main motion export path from the upstream repo. OBJ export is for a
  single reconstructed image mesh; a per-frame mesh sequence export would be a
  separate feature.
- The UI's video People count reports the strongest per-frame estimate rather
  than the accumulated per-frame detection count. The accumulated count remains
  visible in the run output for debugging.
- Video mesh preview events are grouped by source frame and person index, so the
  3D video preview can show multiple people when the native engine emits them.
  The UI also applies a lightweight bbox-IoU tracker matching the upstream live
  BVH export rule, because the current C preview callback streams detections but
  not the writer's final internal track IDs.

## Upstream And Model Credits

SAM3DBody Desktop builds on the native C++ inference work from
[AmmarkoV/SAM3DBody-cpp](https://github.com/AmmarkoV/SAM3DBody-cpp).

The underlying SAM 3D Body research/model family is from
[facebookresearch/sam-3d-body](https://github.com/facebookresearch/sam-3d-body)
and the DINOv3 body model release at
[facebook/sam-3d-body-dinov3](https://huggingface.co/facebook/sam-3d-body-dinov3).

Additional motion-capture lineage and related prior work:
[FORTH-ModelBasedTracker/MocapNET](https://github.com/FORTH-ModelBasedTracker/MocapNET).
