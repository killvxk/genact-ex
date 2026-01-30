# Coding Conventions

**Analysis Date:** 2026-01-30

## Naming Patterns

**Files:**
- Rust modules follow snake_case: `bootlog.rs`, `docker_build.rs`, `kernel_compile.rs`
- Module files named after functionality they implement
- Struct names in PascalCase: `Bootlog`, `Cargo`, `Ansible`, `RkHunter`, `Terraform`
- Example: `src/modules/cargo.rs` contains `struct Cargo`

**Functions:**
- Async functions use snake_case: `dprint()`, `csleep()`, `newline()`, `print()`
- Helper functions in snake_case: `gen_string_with_chars()`, `gen_hex_string()`, `gen_file_path()`
- Private helper functions prefixed with underscore if needed
- Example: `async fn run(&self, app_config: &AppConfig)` in trait implementation

**Variables:**
- snake_case for local variables: `num_packages`, `sleep_length`, `chosen_names`
- Mutable variables indicated with `mut` keyword
- Descriptive names over abbreviations: `num_packages` not `n_pkgs`

**Types:**
- PascalCase for structs: `AppConfig`, `Cargo`, `Bootlog`, `Module`
- PascalCase for traits: `Module` (async trait with `#[async_trait(?Send)]`)
- Static constants in UPPER_SNAKE_CASE: `INSTANT_PRINT_LINES`, `SPEED_FACTOR`, `CTRLC_PRESSED`
- Example: `pub static ref SPEED_FACTOR: Mutex<f32> = Mutex::new(1.0);`

## Code Style

**Formatting:**
- Tool: `cargo fmt` (Rust standard formatter)
- Indentation: 4 spaces (enforced via .editorconfig)
- End of line: LF only
- Final newline required on all files except Markdown
- Trim trailing whitespace

**Linting:**
- Tool: `cargo clippy`
- Strict warnings: `-D warnings` flag in CI (all warnings treated as errors)
- No unused imports allowed
- Consistent error handling patterns required

**Configuration Files:**
- `.editorconfig`: Controls formatting defaults
- `.github/workflows/ci.yml`: Runs `cargo fmt --all -- --check` and `cargo clippy -- -D warnings`

## Import Organization

**Order:**
1. Standard library imports (`std::`, `async_std::`)
2. External crate imports (alphabetically: `anyhow`, `async_trait`, `clap`, `instant`, `rand`, `yansi`)
3. Local crate imports (`crate::`)
4. Conditional imports with `#[cfg(...)]`

**Path Aliases:**
- Direct imports used, no wildcard imports
- Example from `src/modules/cargo.rs`:
  ```rust
  use async_trait::async_trait;
  use instant::Instant;
  use rand::seq::IndexedRandom;
  use rand::{Rng, rng};
  use yansi::Paint;

  use crate::args::AppConfig;
  use crate::data::PACKAGES_LIST;
  use crate::generators::gen_package_version;
  use crate::io::{csleep, dprint, newline, print};
  use crate::modules::Module;
  ```

**Platform-Specific Imports:**
- Use conditional compilation: `#[cfg(target_arch = "wasm32")]` and `#[cfg(not(target_arch = "wasm32"))]`
- Example: WASM imports in `src/io.rs` vs native IO imports

## Error Handling

**Patterns:**
- Use `anyhow::Result<T>` for fallible operations
- Early returns with `?` operator for error propagation
- Example from `src/main.rs`:
  ```rust
  #[async_std::main]
  async fn main() -> Result<()> {
      let appconfig = parse_args();
      if let Some(shell) = appconfig.print_completions {
          let mut clap_app = AppConfig::command();
          clap_complete::generate(shell, &mut clap_app, app_name, &mut std::io::stdout());
          return Ok(());
      }
      run(appconfig).await;
      Ok(())
  }
  ```
- Custom validators return `Result<T, String>` for clap value parsers
- `inspect_err()` for logging warnings without failing: `.inspect_err(|err| println!("WARN: {err}"))`

## Logging

**Framework:** `println!()` for stdout, direct stderr for warnings

**Patterns:**
- Use `println!()` for user-facing output
- Warnings formatted as: `println!("WARN: {description}")`
- Example from `src/lib.rs`: `println!("WARN: failed to set up idle inhibition: {err}")`
- Module output uses helper functions: `log_action()` in `src/modules/julia.rs`
- Progress tracking via `progress_string` crate for visual feedback

## Comments

**When to Comment:**
- Doc comments (`///`) on public module items and trait methods
- Inline comments (`//`) for complex logic or non-obvious decisions
- Example from `src/modules/bootlog.rs`:
  ```rust
  /// Pretend to boot a system
  ```

**JSDoc/Rust Doc:**
- Module-level doc comments using `//!` at file start
- Example: `//! Module containing random utilities.` in `src/generators.rs`
- Function doc comments describe purpose and parameters
- Example from `src/generators.rs`:
  ```rust
  /// Return a String containing `n` random concatenated elements from `list`.
  ///
  /// If `n` >= `list.len()` then `list.len()` will be used instead of `n`.
  pub fn gen_random_n_from_list_into_string(rng: &mut ThreadRng, list: &[&str], n: u64) -> String {
  ```

## Function Design

**Size:**
- Prefer focused functions with single responsibility
- Large modules split into helper functions
- Example: `terraform.rs` (153 lines) uses helper `async fn bold()` for repeated pattern

**Parameters:**
- Use references where ownership not needed: `&AppConfig`, `&mut ThreadRng`
- Trait bounds kept simple: trait functions rarely exceed 3 parameters
- Mutable parameters explicit: `&mut rng`
- Example: `async fn gen_file_path<T: Clone + AsRef<str> + AsRef<Path>>()`

**Return Values:**
- Async functions return concrete types or `Result<T>`
- No generic return types in module implementations
- Trait methods return concrete types per implementation
- Early returns used for error/exit conditions

## Module Design

**Exports:**
- Trait definition (`Module`) exported from `src/modules/mod.rs`
- Individual module structs exported: `pub mod ansible`, `pub mod cargo`, etc.
- Data lists exported from `src/data.rs` via `lazy_static!` statics
- Generators exported from `src/generators.rs` as public functions

**Barrel Files:**
- `src/lib.rs` re-exports: `pub mod args`, `pub mod modules`
- `src/modules/mod.rs` re-exports all module implementations
- `src/modules/` structures: one file per module with public struct and trait impl

**Module Organization:**
- Each module is self-contained in one file under `src/modules/`
- Modules implement `#[async_trait(?Send)]` trait
- Consistent trait methods: `name()`, `signature()`, `run()`
- Direct access to global statics: `PACKAGES_LIST`, `BOOTLOG_LIST`, etc.

## Architecture Patterns

**Trait-Based Module System:**
- Dynamic dispatch via `Box<dyn Module>` in `BTreeMap<&str, Box<dyn Module>>`
- Runtime module selection by name
- Uniform interface across all activity simulators

**Configuration-Driven Behavior:**
- `AppConfig` passed to all module `run()` methods
- `should_exit()` method checks exit conditions globally
- `speed_factor` and `instant_print_lines` modify timing behavior

**Async/Await with async-std:**
- All user-facing IO is async: `dprint()`, `print()`, `newline()`, `csleep()`
- `#[async_std::main]` for entry point
- `#[async_trait(?Send)]` for trait methods allowing async

---

*Convention analysis: 2026-01-30*
