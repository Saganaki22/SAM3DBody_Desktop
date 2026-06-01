use std::env;
use std::fs;
use std::path::Path;

fn main() {
    copy_engine_resources();
    copy_template_resources();
    tauri_build::build();
}

fn copy_engine_resources() {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR missing");
    let manifest_dir = Path::new(&manifest_dir);
    let native_candidates = [
        manifest_dir.join("../../build/windows-cuda"),
        manifest_dir.join("../../build/windows-cuda/Release"),
        manifest_dir.join("../../build/windows/Release"),
    ];
    let resource_dir = manifest_dir.join("resources/engine");

    for native_dir in &native_candidates {
        println!("cargo:rerun-if-changed={}", native_dir.display());
    }

    let Some(native_dir) = native_candidates
        .iter()
        .find(|dir| dir.join("fast_sam_3dbody.dll").is_file())
    else {
        return;
    };

    if fs::create_dir_all(&resource_dir).is_err() || !native_dir.is_dir() {
        return;
    }

    if let Ok(entries) = fs::read_dir(&native_dir) {
        for entry in entries.flatten() {
            let src = entry.path();
            let is_dll = src
                .extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| ext.eq_ignore_ascii_case("dll"))
                .unwrap_or(false);
            if is_dll {
                let _ = fs::copy(&src, resource_dir.join(entry.file_name()));
            }
        }
    }
}

fn copy_template_resources() {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR missing");
    let manifest_dir = Path::new(&manifest_dir);
    let dst_dir = manifest_dir.join("resources/templates");

    for file_name in ["body.bvh", "body_mesh.tri"] {
        let src = manifest_dir.join("../..").join(file_name);
        println!("cargo:rerun-if-changed={}", src.display());

        if src.is_file() && fs::create_dir_all(&dst_dir).is_ok() {
            let _ = fs::copy(src, dst_dir.join(file_name));
        }
    }
}
