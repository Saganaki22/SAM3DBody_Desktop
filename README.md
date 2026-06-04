<div align="center">

# SAM3DBody Desktop

<img
  width="80"
  height="80"
  alt="SAM3DBody Logo"
  src="https://github.com/user-attachments/assets/73df9249-2c53-4bdf-aca2-e86da48bc4c6"
/>

![Platform](https://img.shields.io/badge/platform-Windows%20x64-0078D4)
![GPU](https://img.shields.io/badge/GPU-NVIDIA%20CUDA%2013-76B900)
![UI](https://img.shields.io/badge/UI-Tauri%202%20%2B%20Three.js-24C8DB)
![Engine](https://img.shields.io/badge/engine-C%2B%2B%20DLL%20via%20C%20API-00599C)
![Build](https://img.shields.io/badge/build-MSVC%202022-success)

SAM3DBody Desktop `0.1.1` is a Windows Tauri app for running the `SAM3DBody-cpp`
native engine in-process. It does not launch the CLI as a sidecar. The UI calls
Rust commands, Rust loads `fast_sam_3dbody.dll` with `libloading`, and the DLL
executes the C++/CUDA inference pipeline through a plain C ABI.

</div>

https://github.com/user-attachments/assets/b0d01615-d000-4dbc-821e-d822b9ed754b

## Pre-Built Executables 

<details>
<summary><strong>SAM3DBody Desktop v0.1.1 Downloads (click to expand)</strong></summary>

<br>

### Recommended: Portable CUDA Build

Fastest way to get up and running.

- No installer required
- Extract and run
- Recommended for most users

**Download:**

- [portable-cuda.zip](https://github.com/Saganaki22/SAM3DBody_Desktop/releases/download/v0.1.1/portable-cuda.zip)

---

### Windows Installer

Standard Windows installation.

**Download:**

- [SAM3DBody.Desktop_0.1.1_x64-setup.exe](https://github.com/Saganaki22/SAM3DBody_Desktop/releases/download/v0.1.1/SAM3DBody.Desktop_0.1.1_x64-setup.exe)

---

### MSI Package

For enterprise deployment and MSI-based installation workflows.

**Download:**

- [SAM3DBody.Desktop_0.1.1_x64_en-US.msi](https://github.com/Saganaki22/SAM3DBody_Desktop/releases/download/v0.1.1/SAM3DBody.Desktop_0.1.1_x64_en-US.msi)

---

### Required Models

All release packages require the SAM3DBody model bundle from Hugging Face.

**Download:**

- [SAM3DBody-cpp-onnx-models.zip](https://huggingface.co/AmmarkoV/SAM3DBody-cpp-onnx-models/resolve/main/SAM3DBody-cpp-onnx-models.zip)

After extraction, place the resulting `onnx/` folder beside the portable application, or configure the model paths from within the application.

---

### Quick Start

1. Download **portable-cuda.zip**.
2. Download **SAM3DBody-cpp-onnx-models.zip**.
3. Extract both archives.
4. Place the extracted `onnx/` folder beside `sam3dbody-desktop.exe`.
5. Launch the application.
6. Load an image or video and start processing.

</details>

## What It Does

- Runs the native SAM3DBody CUDA engine inside a Tauri desktop app.
- Supports image, image-result JSON, and video inputs with browse and drag/drop.
- Shows 2D previews with bbox, landmarks, and skeleton overlays.
- Shows image and video results in interactive Three.js 3D viewers.
- Toggles body, skeleton, wireframe, grid, rotation, and grid size in the UI.
- Exports image meshes as OBJ or GLB.
- Exports static image BVH poses and multi-person video BVH motion files.
- Uses the native MHR-70 full-body-plus-hands output.
- Includes video smoothing controls, diagnostic 70-keypoint CSV output, video
  playback controls, and abort for video generation.
- Keeps image and video results separate when switching tabs.
- Includes hardware telemetry, themed output copy/save, dark/light themes,
  accent colors, UI scaling, and default-browser links.

## Supported Systems

Primary target:

- Windows 11 x64
- Minimum app window size: 1180 x 720
- Visual Studio 2022 MSVC toolchain
- NVIDIA GPU with a driver new enough for CUDA 13 runtime DLLs
- Ampere, Lovelace, or Blackwell RTX GPU with 8 GB+ VRAM
  (RTX 30-series, 40-series, or 50-series).
- Observed peak VRAM usage is about 7.2 GB
  during model load/inference with the bundled model set.
- The v0.1.2 bundled CUDA runtime targets `sm_86`, `sm_89`, and `sm_120`
  for RTX 30xx, 40xx, and 50xx cards.
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

The Windows installers and `portable-cuda.zip` already include the native DLLs.
If you are building the app from source and do not want to rebuild/copy the
native CUDA runtime files yourself, download the runtime DLL pack:

[SAM3DBody-Desktop_engine-runtime-cuda13-win64-dlls.zip](https://github.com/Saganaki22/SAM3DBody_Desktop/releases/download/v0.1.1/SAM3DBody-Desktop_engine-runtime-cuda13-win64-dlls.zip)

Extract it into `desktop/src-tauri/` so it creates
`desktop/src-tauri/resources/engine/`.

Place an already extracted `onnx/` folder next to the portable app or repo root.
The UI can also browse to a custom ONNX folder, GGUF file, YOLO file, backbone,
LBS body model, and BVH template. The links below are included for manual model
setup.

Manual model download:

| File | Link |
| --- | --- |
| All-in-one model zip | [SAM3DBody-cpp-onnx-models.zip](https://huggingface.co/AmmarkoV/SAM3DBody-cpp-onnx-models/resolve/main/SAM3DBody-cpp-onnx-models.zip) |
| Hugging Face model repo | [AmmarkoV/SAM3DBody-cpp-onnx-models](https://huggingface.co/AmmarkoV/SAM3DBody-cpp-onnx-models) |

The all-in-one zip is hosted on HuggingFace

## Build Requirements

Known-good Windows build inputs:

| Dependency | Version / Notes |
| --- | --- |
| Visual Studio Build Tools | 2022, MSVC C++ workload |
| CMake | 3.18+ |
| Ninja | 1.11+ |
| Node.js | 20+ |
| Rust | Stable MSVC toolchain |
| OpenCV | 4.10 Windows build, `x64/vc16` |
| CUDA | CUDA 13 runtime build for Ampere/Lovelace/Blackwell RTX GPUs: `sm_86;89;120` |
| cuDNN | cuDNN 9 DLL folder for runtime bundling |
| WebView2 | Required by Tauri on Windows |

The source repo does not track the large native runtime DLLs or model files. If
you only want to build the desktop installer and do not want to rebuild the
native CUDA DLLs yourself, use the runtime DLL pack linked above.

## Build Native Engine

From the repository root:

```powershell
# Set these for your machine before configuring CMake.
$env:OpenCV_DIR = "C:\opencv\build"
$env:SAM3D_CUDNN_BIN_DIR = "C:\path\to\cudnn\bin"

$opencv = $env:OpenCV_DIR
$cudnn = $env:SAM3D_CUDNN_BIN_DIR
$vsdev = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat"
$ninja = (Get-Command ninja).Source
$cudaArch = if ($env:SAM3D_CUDA_ARCH) { $env:SAM3D_CUDA_ARCH } else { "86;89;120" }

if (-not $opencv) { throw "Set OpenCV_DIR to your OpenCV build folder" }
if (-not $cudnn) { throw "Set SAM3D_CUDNN_BIN_DIR to your cuDNN bin folder" }

cmd /c "`"$vsdev`" -arch=x64 && cmake -S . -B build\windows-cuda -G Ninja `
  -DCMAKE_MAKE_PROGRAM=`"$ninja`" `
  -DCMAKE_BUILD_TYPE=Release `
  -DSAM3D_USE_CUDA=ON `
  -DCMAKE_CUDA_ARCHITECTURES=`"$cudaArch`" `
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
cargo test --lib
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
- GitHub Actions CI runs frontend, Rust, and source hygiene checks. It does not
  run CUDA inference because hosted runners do not provide the release GPU/model
  environment.

## Upstream And Model Credits

SAM3DBody Desktop builds on the native C++ inference work from
[AmmarkoV/SAM3DBody-cpp](https://github.com/AmmarkoV/SAM3DBody-cpp).

The underlying SAM 3D Body research/model family is from
[facebookresearch/sam-3d-body](https://github.com/facebookresearch/sam-3d-body)
and the DINOv3 body model release at
[facebook/sam-3d-body-dinov3](https://huggingface.co/facebook/sam-3d-body-dinov3).

Additional motion-capture lineage and related prior work:
[FORTH-ModelBasedTracker/MocapNET](https://github.com/FORTH-ModelBasedTracker/MocapNET).
