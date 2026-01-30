# Technology Stack

**Analysis Date:** 2026-01-30

## Languages

**Primary:**
- Rust 1.70+ (Edition 2024) - Core application, binaries for all platforms

**Secondary:**
- WebAssembly (WASM) - Browser-based version compiled from Rust
- HTML/CSS/JavaScript - Web interface wrapper for WASM module

## Runtime

**Environment:**
- Rust standard library + async-std for async runtime
- Single-threaded event loop via `async-std::main` macro

**Platform Targets:**
- Linux (x86_64-unknown-linux-musl, x86_64-unknown-linux-gnu, aarch64, armv7)
- Windows (x86_64-pc-windows-msvc, x86_64-pc-windows-gnu)
- macOS (x86_64-apple-darwin, aarch64-apple-darwin via GitHub Actions)
- FreeBSD (x86_64-unknown-freebsd)
- WebAssembly (wasm32-unknown-unknown) for browser deployment
- Cross-compilation via `cross` tool for non-native targets

**Package Manager:**
- Cargo (Rust package manager)
- Lockfile: `Cargo.lock` (committed, version 4 format) - 236 total dependencies

## Frameworks

**Core Runtime:**
- `async-std` 1.x - Async runtime with attributes macro support
- `async-trait` 0.1 - Trait support for async functions

**CLI/Argument Parsing:**
- `clap` 4.5 - Command-line argument parsing with derive macros, cargo metadata, help wrapping
- `clap_complete` 4.x - Shell completion generation (bash, fish, zsh, powershell, elvish)
- `clap_mangen` 0.2 - Man page generation

**Data Generation & Randomization:**
- `fake` 4.3 - Fake data generation with chrono feature
- `rand` 0.9 - Random number generation
- `rand_distr` 0.5 - Probability distributions (ChiSquared, Exp for version numbers)
- `getrandom` 0.3 - Cryptographic randomness with wasm_js backend

**Time & Duration:**
- `chrono` 0.4 - Date/time handling with WASM support (clock, wasmbind features)
- `humantime` 2.x - Human-readable time duration parsing
- `instant` 0.1 - Cross-platform instant time with WASM support

**Terminal & Display:**
- `yansi` 1.x - ANSI color output for terminal
- `colorgrad` 0.8 - Color gradients with presets
- `progress_string` 0.2 - Progress bar string generation
- `humansize` 2.x - Human-readable file sizes
- `terminal_size` 0.4 - Terminal width detection (Linux/Windows/macOS only)
- `ctrlc` 3.5 - CTRL-C signal handling with termination feature

**WASM-Specific Bindings:**
- `wasm-bindgen` 0.2 - WebAssembly bindings to JavaScript
- `wasm-bindgen-futures` 0.4 - JavaScript Promise integration
- `js-sys` 0.3 - Raw JavaScript bindings
- `web-sys` 0.3 - Web API bindings (Window, Location, console)
- `console_error_panic_hook` 0.1 - Better panic messages in WASM

**Platform-Specific (Linux/macOS/Windows only):**
- `keepawake` 0.6 - System idle inhibition/display wake-lock control
  - Gated to: `cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))`

**Utilities:**
- `anyhow` 1.x - Error handling and context
- `lazy_static` 1.5 - Lazy static initialization
- `sha2` 0.10.9 - SHA-256 hashing
- `regex` 1.12 - Regular expression support
- `url` 2.5 - URL parsing (WASM target only)

## Configuration

**Environment:**
- No `.env` file usage detected
- Configuration via command-line arguments (clap)
- Speed factor: `--speed-factor` (float, default 1.0, minimum 0.01)
- Module selection: `--modules` (default: all modules)
- Exit conditions: `--exit-after-time` and `--exit-after-modules`
- WASM configuration: query string parameters (`?module=`, `?speed-factor=`)

**Build Configuration:**
- `Cargo.toml` edition = 2024
- Release profile: LTO enabled, opt-level 'z' (size), single codegen-unit, strip debuginfo
- Cross-compilation toolchain: RUSTFLAGS via GitHub Actions

**Web Build:**
- `Trunk.toml` - WASM bundler configuration
  - Public URL: `/genact/` for GitHub Pages deployment
- WASM-specific flag: `--cfg getrandom_backend=\"wasm_js\"`

## Build Tools

**Build System:**
- `cargo build` - Standard Rust compilation
- `trunk` - WASM bundler and development server
- `upx` - Binary compression (used in Makefile targets)
- GitHub Actions - CI/CD and multi-platform binary releases

**Testing:**
- `cargo test` - Built-in Rust test framework
- No external test framework detected (uses standard library)

**Linting & Formatting:**
- `cargo fmt` - Rust code formatter (enforced in CI)
- `cargo clippy` - Linter with `-D warnings` (fail on warnings in CI)
- `rustfmt` and `clippy` installed as components via `dtolnay/rust-toolchain@stable`

**Release Automation:**
- `cargo-release` - Version management and release automation
- `parse-changelog` - CHANGELOG entry extraction for releases

## Platform Requirements

**Development:**
- Rust stable toolchain (installed via dtolnay/rust-toolchain action)
- `rustfmt` component for formatting
- `clippy` component for linting
- `trunk` (installed via GitHub Actions or manual install)
- `wasm-bindgen-cli` (installed for WASM target)
- `musl-tools` (for Linux musl targets)
- `cross` tool (for cross-compilation)

**Production:**
- **Binaries:** No runtime dependencies (fully static linked for musl targets)
- **WASM:** Modern browser with WebAssembly support (Chrome, Firefox, Safari, Edge)
- **Web Deployment:** GitHub Pages (configured in build-release.yml)
- **Container:** OCI-compatible runtime (Docker, Podman) for container images
- **Distribution:**
  - crates.io (Rust package registry)
  - Docker Hub: `svenstaro/genact` (multi-arch images: amd64, aarch64, armv7)
  - FreeBSD packages: `pkg install genact`
  - Homebrew (macOS)
  - MacPorts (macOS)
  - Scoop (Windows)
  - GitHub Releases (pre-built binaries with UPX compression)

---

*Stack analysis: 2026-01-30*
