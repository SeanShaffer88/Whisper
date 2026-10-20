//! Build integration for the vendored whisper.cpp engine.
//!
//! Modes:
//! - vendor/src-whisper ABSENT: build the pure-Rust crate only. The FFI
//!   surface compiles and unit-tests; engine calls report
//!   WS_ERR_NOT_SUPPORTED. This keeps the host gate hermetic on a fresh
//!   checkout (no ~90 MB of sources/model needed to lint and test).
//! - vendor/src-whisper PRESENT and host == target: configure + build
//!   whisper.cpp/ggml as static libraries with CMake (Release, shared
//!   OFF, examples/tests/server OFF — only the library), link them plus
//!   the C++ runtime, and set `cfg(ws_engine)`.
//! - Cross builds (host != target) never try to run a host-compiler
//!   CMake build for a foreign target; they take the no-engine path.
//!   On Windows CI host == target, so the real link happens there.
//!
//! The CMake build directory lives under target/whisper-cmake (stable
//! across cargo invocations, unlike OUT_DIR) so repeat builds are
//! incremental no-ops.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rustc-check-cfg=cfg(ws_engine)");
    println!("cargo:rerun-if-changed=build.rs");

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let vendor_src = manifest_dir.join("vendor").join("src-whisper");
    println!("cargo:rerun-if-changed={}", vendor_src.display());
    println!(
        "cargo:rerun-if-changed={}",
        vendor_src.join("include").join("whisper.h").display()
    );

    if !vendor_src.join("CMakeLists.txt").is_file() {
        println!(
            "cargo:warning=whisper: vendored whisper.cpp sources not found — building WITHOUT \
             the engine. Run fetch.sh / fetch.ps1, then rebuild. ws_context_create will \
             report WS_ERR_NOT_SUPPORTED."
        );
        return;
    }

    let host = env::var("HOST").expect("HOST");
    let target = env::var("TARGET").expect("TARGET");
    if host != target {
        println!(
            "cargo:warning=whisper: cross build ({host} -> {target}) — engine not built by \
             build.rs for foreign targets; building without the engine."
        );
        return;
    }

    println!("cargo:rustc-cfg=ws_engine");
    build_engine(&vendor_src, &target);
}

fn run(cmd: &mut Command, what: &str) {
    let status = cmd
        .status()
        .unwrap_or_else(|e| panic!("failed to spawn {what} (is CMake installed?): {e}"));
    if !status.success() {
        panic!("{what} failed with {status}");
    }
}

fn build_engine(src: &Path, target: &str) {
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    // OUT_DIR = <target>/<profile>/build/<pkg>-<hash>/out — walk up to
    // the target dir so the CMake tree survives across cargo commands.
    let build_dir = out_dir
        .ancestors()
        .nth(4)
        .map(|t| t.join("whisper-cmake"))
        .unwrap_or_else(|| out_dir.join("whisper-cmake"));
    fs::create_dir_all(&build_dir).expect("create cmake build dir");

    if !build_dir.join("CMakeCache.txt").is_file() {
        let mut cfg = Command::new("cmake");
        cfg.arg("-S").arg(src).arg("-B").arg(&build_dir);
        cfg.args([
            "-DCMAKE_BUILD_TYPE=Release",
            "-DBUILD_SHARED_LIBS=OFF",
            "-DWHISPER_BUILD_TESTS=OFF",
            "-DWHISPER_BUILD_EXAMPLES=OFF",
            "-DWHISPER_BUILD_SERVER=OFF",
            // No -march=native: the DLL must run on machines other than
            // the one that compiled it.
            "-DGGML_NATIVE=OFF",
            // No OpenMP: it would drag a libgomp/vcomp runtime
            // dependency into the DLL. ggml's own threadpool is the
            // standard path on platforms without OpenMP.
            "-DGGML_OPENMP=OFF",
            // Static libs get linked into a cdylib — need PIC.
            "-DCMAKE_POSITION_INDEPENDENT_CODE=ON",
        ]);
        run(&mut cfg, "cmake configure (whisper.cpp)");
    }

    let mut build = Command::new("cmake");
    build
        .arg("--build")
        .arg(&build_dir)
        .args(["--config", "Release", "--parallel"]);
    run(&mut build, "cmake build (whisper.cpp)");

    // Collect the static libraries CMake produced (names/paths differ
    // between the Make/Ninja layout and the Visual Studio layout, where
    // libs land in per-config subdirectories).
    let mut libs: Vec<(i32, String, PathBuf)> = Vec::new(); // (order, name, dir)
    collect_libs(&build_dir, &mut libs);
    if !libs.iter().any(|(_, n, _)| n == "whisper") {
        panic!(
            "whisper static library not found under {} after CMake build",
            build_dir.display()
        );
    }
    libs.sort();
    let mut dirs: Vec<PathBuf> = Vec::new();
    for (_, _, dir) in &libs {
        if !dirs.contains(dir) {
            dirs.push(dir.clone());
        }
    }
    for d in &dirs {
        println!("cargo:rustc-link-search=native={}", d.display());
    }
    for (_, name, _) in &libs {
        println!("cargo:rustc-link-lib=static={name}");
    }

    // C++ runtime + platform libs for the ggml CPU backend.
    if target.contains("windows-msvc") {
        // MSVC: the C++ runtime is part of the default link set.
    } else if target.contains("apple") {
        println!("cargo:rustc-link-lib=c++");
    } else {
        // Linux and other ELF targets (incl. windows-gnu via MinGW).
        println!("cargo:rustc-link-lib=stdc++");
        println!("cargo:rustc-link-lib=pthread");
        println!("cargo:rustc-link-lib=dl");
        println!("cargo:rustc-link-lib=m");
    }
    if target.contains("windows") {
        // ggml's platform code references a few Win32 basics.
        println!("cargo:rustc-link-lib=advapi32");
    }
}

/// Find `whisper` / `ggml*` static libraries under the CMake build tree.
/// Order key: whisper first, then umbrella ggml, cpu backend, base —
/// dependents before dependencies, as static linkers require.
fn collect_libs(dir: &Path, out: &mut Vec<(i32, String, PathBuf)>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_libs(&path, out);
            continue;
        }
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        if ext != "a" && ext != "lib" {
            continue;
        }
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_string();
        let name = stem.strip_prefix("lib").unwrap_or(&stem).to_string();
        let order = if name == "whisper" {
            0
        } else if name == "ggml" {
            1
        } else if name.starts_with("ggml-cpu") {
            2
        } else if name == "ggml-base" {
            3
        } else if name.starts_with("ggml") {
            4
        } else {
            continue;
        };
        out.push((order, name, dir.to_path_buf()));
    }
}
