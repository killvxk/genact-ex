# Codebase Structure

**Analysis Date:** 2026-01-30

## Directory Layout

```
genact-ex/
├── src/                    # Rust source code
│   ├── main.rs            # Entry point (dual-target: native + WASM)
│   ├── lib.rs             # Core run loop and public API
│   ├── args.rs            # Argument parsing (platform-specific)
│   ├── modules/           # Activity generator modules
│   │   ├── mod.rs         # Module trait definition and registry
│   │   ├── ansible.rs     # Ansible playbook simulation
│   │   ├── bootlog.rs     # System boot sequence
│   │   ├── botnet.rs      # Botnet activity
│   │   ├── bruteforce.rs  # Password brute-force attack
│   │   ├── cargo.rs       # Rust build process
│   │   ├── cc.rs          # C compiler output
│   │   ├── composer.rs    # PHP composer install
│   │   ├── cryptomining.rs # Crypto mining simulation
│   │   ├── docker_build.rs # Docker image build
│   │   ├── docker_image_rm.rs # Docker image removal
│   │   ├── download.rs    # File download simulation
│   │   ├── julia.rs       # Julia package manager
│   │   ├── kernel_compile.rs # Linux kernel build
│   │   ├── memdump.rs     # Memory dump hex output
│   │   ├── mkinitcpio.rs  # Arch Linux initramfs
│   │   ├── rkhunter.rs    # Rootkit hunter scan
│   │   ├── simcity.rs     # SimCity game simulation
│   │   ├── terraform.rs   # Terraform IaC provisioning
│   │   ├── weblog.rs      # Web server access logs
│   │   └── wpt.rs         # Web Platform Tests
│   ├── data.rs            # Static data loading and lazy initialization
│   ├── generators.rs      # Procedural content generation utilities
│   └── io.rs              # Platform-abstracted I/O (terminal + WASM)
├── data/                  # Static data files (included at compile-time)
│   ├── ansible_roles.txt
│   ├── ansible_tasks.txt
│   ├── boot_hooks.txt
│   ├── bootlog.txt        # Sample boot messages
│   ├── cfiles.txt         # ~20k C source file paths
│   ├── composer.txt       # PHP packages
│   ├── css_properties.txt # CSS property names
│   ├── docker_packages.txt # Linux packages for Docker
│   ├── docker_tags.txt    # Docker image tags
│   ├── julia.csv          # Julia package registry
│   ├── os_releases.txt    # OS identifiers
│   ├── packages.txt       # ~10k generic package names
│   ├── rkhunter_*.txt     # Rootkit hunter data
│   ├── simcity.txt        # SimCity messages
│   ├── terraform_*_resources.txt # AWS/Azure/GCP resources
│   ├── terraform_ids.txt  # Resource identifiers
│   ├── web_apis.txt       # Web API names
│   └── wpt_categories.txt # Web test categories
├── static/                # Web assets
│   └── styles.css         # Minimal styling for web UI
├── Cargo.toml             # Rust package manifest and dependencies
├── Cargo.lock             # Locked dependency versions
├── Trunk.toml             # WASM bundler configuration
├── README.md              # Project documentation
├── CHANGELOG.md           # Version history
├── CONTRIBUTING.md        # Contribution guidelines
├── LICENSE                # MIT license
├── Makefile               # Build shortcuts
├── index.html             # Web UI entry point
└── Containerfile          # OCI container definition
```

## Directory Purposes

**src/:**
- Purpose: All Rust source code (libraries and binaries)
- Contains: Module implementations, utilities, and orchestration logic
- Key files: `main.rs` (entry), `lib.rs` (core), `modules/mod.rs` (registry)

**src/modules/:**
- Purpose: Individual activity simulation modules
- Contains: 20 structs implementing `Module` trait, each with distinctive fake output
- Key files: `mod.rs` defines trait and static registry; each `.rs` file is one self-contained module

**data/:**
- Purpose: Static reference data for realistic-looking simulations
- Contains: Plain text lists of real/plausible package names, file paths, resource identifiers
- Key files: `packages.txt` (used by many modules), `cfiles.txt` (C compiler), `terraform_*.txt` (infrastructure)

**static/:**
- Purpose: Web UI assets (minimal; main UI provided by trunk build output)
- Contains: CSS stylesheet
- Key files: `styles.css` (xterm styling)

## Key File Locations

**Entry Points:**
- `src/main.rs`: Platform-specific main() - handles native CLI or WASM initialization
- `index.html`: Web UI root - loads WASM binary and xterm terminal

**Configuration:**
- `Cargo.toml`: Rust dependencies, feature flags, optimization profile
- `Trunk.toml`: WASM build configuration (asset bundling, optimization)
- `.editorconfig`: Editor formatting consistency

**Core Logic:**
- `src/lib.rs`: `run()` function (module selection loop), `exit_handler()`, global state
- `src/args.rs`: `AppConfig` struct and `parse_args()` (dual implementation)
- `src/modules/mod.rs`: `Module` trait, `ALL_MODULES` registry

**Data Loading:**
- `src/data.rs`: Compile-time includes of all `data/*.txt` files, lazy-static Vec conversions

**Utilities:**
- `src/generators.rs`: Random content functions (strings, hex, paths, versions)
- `src/io.rs`: Cross-platform output (`print`, `dprint`, `csleep`, `newline`)

**Testing:**
- None detected. Project uses no test framework or `tests/` directory.

## Naming Conventions

**Files:**
- Rust modules: snake_case (e.g., `docker_build.rs`, `kernel_compile.rs`)
- Data files: snake_case with domain (e.g., `ansible_roles.txt`, `terraform_aws_resources.txt`)
- Compiled output: Lowercase with OS suffix (e.g., `genact-linux`, `genact-osx`, `genact-win.exe`)

**Directories:**
- Flat structure for modules (no subdirectories)
- No feature-based grouping (all modules at `src/modules/*.rs`)

**Code:**
- Structs implementing `Module`: PascalCase (e.g., `Cc`, `Cargo`, `Bootlog`, `DockerBuild`)
- Functions: snake_case (e.g., `gen_hex_string()`, `parse_args()`, `generate_includes()`)
- Constants: SCREAMING_SNAKE_CASE (e.g., `COMPILERS`, `FLAGS_OPT` in cc.rs)
- Global statics: SCREAMING_SNAKE_CASE (e.g., `CTRLC_PRESSED`, `SPEED_FACTOR`)

## Where to Add New Code

**New Activity Module (Simulation):**
- Create: `src/modules/newname.rs` with struct implementing `Module` trait
- Register: Add `pub mod newname;` to `src/modules/mod.rs` and insert into `ALL_MODULES` BTreeMap
- Data: Add reference data file `data/newname.txt` if needed
- Import patterns: Follow `src/modules/cargo.rs` or `src/modules/cc.rs` as template

**New Generator Function:**
- Location: `src/generators.rs`
- Pattern: Pure functions taking `&mut ThreadRng` and returning `String` or `T`
- Example: `gen_package_version()` uses statistical distributions for realism

**New Data Source:**
- Location: Create `data/newname.txt` with line-delimited entries
- Usage: In `src/data.rs`, add `static NEWNAME: &str = include_str!("../data/newname.txt");`
- Access: Create lazy-static `Vec<&'static str>` in same file, use in modules

**Platform-Specific Feature:**
- Conditional compilation: Use `#[cfg(target_arch = "wasm32")]` for WASM, `#[cfg(not(target_arch = "wasm32"))]` for native
- Examples: `src/io.rs` has dual implementations of `csleep()` and `print()`; `src/args.rs` has dual `AppConfig` struct definitions

**Shared Utilities:**
- Location: `src/io.rs` for output/timing, `src/generators.rs` for content generation
- Do NOT add cross-platform logic to individual modules

## Special Directories

**target/:**
- Purpose: Cargo build artifacts (binaries, dependencies, intermediate files)
- Generated: Yes (by cargo build)
- Committed: No (.gitignore)

**.planning/codebase/:**
- Purpose: GSD codebase documentation (this directory)
- Generated: Yes (created by mapper)
- Committed: Yes (but not for long-term; reference docs)

**.git/:**
- Purpose: Git version control metadata
- Generated: N/A
- Committed: N/A

**gifs/:**
- Purpose: README screenshot animations
- Generated: No (manually created for docs)
- Committed: Yes (content files)

## File Modification Patterns

**When adding a new module:**
1. Create `src/modules/newname.rs`
2. Add to `src/modules/mod.rs` with `pub mod newname;`
3. Register in `ALL_MODULES` static
4. Optional: Create `data/newname.txt` if simulation needs realistic data

**When changing output behavior:**
- Modify module's `async fn run()` implementation
- Use `src/io::print()`, `dprint()`, `newline()`, `csleep()` for output
- Import data from `src::data` (e.g., `use crate::data::PACKAGES_LIST;`)

**When modifying configuration:**
- Add field to `AppConfig` struct in `src/args.rs` (both native and WASM variants)
- Add parsing logic in platform-specific `parse_args()` function
- Access in modules via `appconfig` parameter to `Module::run()`

**When optimizing binary size:**
- Cargo profile in `Cargo.toml` already configured with LTO and size optimization
- Review dependencies for essential ones only (currently using minimal set)

---

*Structure analysis: 2026-01-30*
