//! Build script for `cuda-accel`: compiles SM 8.9 PTX kernels into `$OUT_DIR/victory_kernels.ptx`.
//!
//! Dual-track PTX discovery:
//! 1. Track A (`cuda-oxide` pure-Rust PTX): uses `VICTORY_CUDA_OXIDE_PTX` or prebuilt
//!    `kernels_oxide/target/nvptx64-nvidia-cuda/release/victory_kernels_oxide.ptx` when present.
//! 2. Track B (`nvcc` SM 8.9 PTX twin): invokes the local CUDA Toolkit `nvcc --ptx -arch=sm_89 -O3`
//!    on `kernels/victory_kernels.cu` so the RTX 4080 executes real PTX today without requiring
//!    a nightly `cargo-oxide` / LLVM 21 compiler setup.
//!
//! Never emits `.ptx` binaries into the repository tree (output lives strictly inside Cargo's `OUT_DIR`).

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rustc-check-cfg=cfg(victory_cuda_ptx_ready)");
    println!("cargo:rerun-if-changed=kernels/victory_kernels.cu");
    println!("cargo:rerun-if-env-changed=VICTORY_CUDA_OXIDE_PTX");
    println!("cargo:rerun-if-env-changed=VICTORY_CUDA_ARCH");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));
    let out_ptx = out_dir.join("victory_kernels.ptx");
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set"));

    // Track A: prebuilt cuda-oxide PTX (explicit env var or kernels_oxide build output).
    if let Some(oxide_ptx) = find_oxide_ptx(&manifest_dir) {
        println!("cargo:rerun-if-changed={}", oxide_ptx.display());
        if let Err(err) = fs::copy(&oxide_ptx, &out_ptx) {
            println!("cargo:warning=failed to copy cuda-oxide PTX {}: {err}", oxide_ptx.display());
        } else {
            println!("cargo:rustc-cfg=victory_cuda_ptx_ready");
            return;
        }
    }

    // Track B: compile kernels/victory_kernels.cu via local nvcc into OUT_DIR.
    let cu_src = manifest_dir.join("kernels").join("victory_kernels.cu");
    if !cu_src.is_file() {
        println!("cargo:warning=CUDA kernel source missing at {}", cu_src.display());
        return;
    }

    let Some(nvcc) = find_nvcc() else {
        println!(
            "cargo:warning=nvcc not found (set CUDA_PATH or VICTORY_CUDA_OXIDE_PTX); \
             cuda-accel will build without embedded PTX and fall back to CPU/DX12 at runtime"
        );
        return;
    };

    let arch = env::var("VICTORY_CUDA_ARCH").unwrap_or_else(|_| "sm_89".to_string());
    let mut cmd = Command::new(&nvcc);
    cmd.arg("--ptx")
        .arg(format!("-arch={arch}"))
        .arg("-O3")
        .arg("--allow-unsupported-compiler");

    if cfg!(windows) {
        if let Some(cl_dir) = find_msvc_cl_dir() {
            cmd.arg("-ccbin").arg(cl_dir);
        }
    }

    cmd.arg(&cu_src).arg("-o").arg(&out_ptx);

    match cmd.output() {
        Ok(output) if output.status.success() => {
            println!("cargo:rustc-cfg=victory_cuda_ptx_ready");
        }
        Ok(output) => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            panic!(
                "nvcc failed compiling {} (status={}):\nstdout:\n{}\nstderr:\n{}",
                cu_src.display(),
                output.status,
                stdout,
                stderr
            );
        }
        Err(err) => {
            println!(
                "cargo:warning=failed to launch nvcc ({}): {err}; building without embedded PTX",
                nvcc.display()
            );
        }
    }
}

fn find_oxide_ptx(manifest_dir: &Path) -> Option<PathBuf> {
    if let Ok(explicit) = env::var("VICTORY_CUDA_OXIDE_PTX") {
        let p = PathBuf::from(explicit);
        if p.is_file() {
            return Some(p);
        }
    }
    let default_oxide = manifest_dir
        .join("kernels_oxide")
        .join("target")
        .join("nvptx64-nvidia-cuda")
        .join("release")
        .join("victory_kernels_oxide.ptx");
    default_oxide.is_file().then_some(default_oxide)
}

fn find_nvcc() -> Option<PathBuf> {
    let exe = if cfg!(windows) { "nvcc.exe" } else { "nvcc" };
    for var in ["CUDA_PATH", "CUDA_HOME"] {
        if let Ok(base) = env::var(var) {
            let candidate = PathBuf::from(base).join("bin").join(exe);
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    if cfg!(windows) {
        let root = Path::new(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA");
        if let Ok(entries) = fs::read_dir(root) {
            let mut versions: Vec<PathBuf> = entries
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.join("bin").join(exe).is_file())
                .collect();
            versions.sort();
            if let Some(latest) = versions.pop() {
                return Some(latest.join("bin").join(exe));
            }
        }
    }
    if Command::new(exe).arg("--version").output().is_ok() {
        return Some(PathBuf::from(exe));
    }
    None
}

fn find_msvc_cl_dir() -> Option<PathBuf> {
    // Check if cl.exe is already in PATH.
    if let Ok(path_var) = env::var("PATH") {
        for dir in env::split_paths(&path_var) {
            if dir.join("cl.exe").is_file() {
                return Some(dir);
            }
        }
    }

    // Query vswhere.exe for the latest MSVC x64 host toolchain.
    let vswhere_candidates = [
        r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe",
        r"C:\Program Files\Microsoft Visual Studio\Installer\vswhere.exe",
    ];
    for vswhere in vswhere_candidates {
        if !Path::new(vswhere).is_file() {
            continue;
        }
        let Ok(out) = Command::new(vswhere)
            .args([
                "-latest",
                "-products",
                "*",
                "-requires",
                "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                "-property",
                "installationPath",
            ])
            .output()
        else {
            continue;
        };
        if !out.status.success() {
            continue;
        }
        let install_path = String::from_utf8_lossy(&out.stdout).trim().to_string();
        if install_path.is_empty() {
            continue;
        }
        let msvc_root = PathBuf::from(install_path).join("VC").join("Tools").join("MSVC");
        let Ok(entries) = fs::read_dir(&msvc_root) else {
            continue;
        };
        let mut versions: Vec<PathBuf> = entries
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.join("bin").join("Hostx64").join("x64").join("cl.exe").is_file())
            .collect();
        versions.sort();
        if let Some(latest) = versions.pop() {
            return Some(latest.join("bin").join("Hostx64").join("x64"));
        }
    }
    None
}
