# Contributing

Thanks for helping improve SAM3DBody Desktop.

## Before Opening A Pull Request

- Keep changes focused and small.
- Do not commit generated output, model files, installer files, portable builds,
  or runtime DLLs.
- Run the lightweight checks:

```powershell
cd desktop
npm ci
npm run build:vite
cargo test --manifest-path src-tauri/Cargo.toml --lib
cargo check --manifest-path src-tauri/Cargo.toml
```

## Native Engine Changes

The native engine depends on CUDA, ONNX Runtime, OpenCV, ggml, and the model
pack. If a change touches native inference, BVH export, or mesh output, include
the hardware/runtime setup used for manual verification.

## Release Assets

Large runtime files are distributed through GitHub Releases or Hugging Face, not
through Git:

- installer / MSI / portable CUDA zip: GitHub Releases
- native runtime DLL pack: GitHub Releases
- model zip: Hugging Face

## Style

- Prefer existing local code patterns over broad refactors.
- Avoid unrelated cleanup in feature or bug-fix PRs.
- Keep user-facing errors clear; runtime failures should mention which file or
  model path failed when possible.
