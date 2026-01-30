# Testing Patterns

**Analysis Date:** 2026-01-30

## Test Framework

**Runner:**
- Rust built-in test framework via `cargo test`
- Config: `Cargo.toml` (no separate config file needed)
- Async runtime: `async-std` with async test attribute support

**Assertion Library:**
- Standard Rust assertions: `assert!`, `assert_eq!`, `assert_ne!`
- No external assertion library used

**Run Commands:**
```bash
cargo test              # Run all tests
cargo test --lib       # Run library tests only
cargo test --release   # Release mode tests (faster iteration)
cargo build             # Build project
cargo fmt --all -- --check  # Check formatting
cargo clippy -- -D warnings  # Run linter with strict warnings
```

## Test File Organization

**Current State:**
- **No unit tests present** in the codebase (0 tests reported by `cargo test --lib`)
- Test infrastructure available but not implemented
- Test discovery location: Would use standard Rust patterns

**Location Pattern (if tests were added):**
- Inline module tests: `#[cfg(test)]` at bottom of implementation files
- Integration tests would go in `tests/` directory (currently missing)
- Separate test files not used currently

**Naming Convention (if tests added):**
- Module tests: `mod tests { #[test] fn test_name() { } }`
- Test function names: snake_case describing behavior tested
- Macro tests: `#[test]` attribute

## Test Structure

**Module Test Pattern (standard Rust):**
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = some_function();
        assert_eq!(result, expected);
    }
}
```

**Async Test Pattern (if needed):**
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[async_std::test]
    async fn test_async_function() {
        let result = async_function().await;
        assert_eq!(result, expected);
    }
}
```

## Mocking

**Framework:** None currently used
- No mock framework dependencies in `Cargo.toml`
- Module system allows stubbing for integration tests via trait bounds

**If Mocking Needed:**
- Use `mockall` crate (would need to add to dependencies)
- Or implement alternative implementations of `Module` trait
- Example approach: Create test-only module structs implementing `Module` trait with fixed output

**What to Mock (guidance for future tests):**
- External IO operations (`print`, `dprint`, `csleep`)
- Random number generation (inject `ThreadRng` instead of using `rng()`)
- CLI argument parsing (test `AppConfig` directly)

**What NOT to Mock:**
- Core logic in `async fn run()` implementations
- Randomization distributions (test with fixed RNG seed if needed)
- Time measurements (`Instant::now()` is fine to test directly)

## Fixtures and Factories

**Test Data:**
- Data is currently embedded in `src/data.rs` using `include_str!()` macro
- Example:
  ```rust
  static BOOTLOG: &str = include_str!("../data/bootlog.txt");
  lazy_static::lazy_static! {
      pub static ref BOOTLOG_LIST: Vec<&'static str> = BOOTLOG.lines().collect();
  }
  ```

**Location:**
- Data files in `data/` directory (bootlog.txt, packages.txt, ansible_roles.txt, etc.)
- Compiled into binary at build time via `include_str!()`
- Tests would directly reference these via lazy_static bindings

**Builder Pattern (if needed):**
- `AppConfig` could be constructed directly in tests
- Example:
  ```rust
  let config = AppConfig {
      modules: vec!["cargo".to_string()],
      speed_factor: 1.0,
      instant_print_lines: 0,
  };
  ```

## Coverage

**Requirements:** Not enforced
- No code coverage tools configured
- No coverage reporting in CI
- No minimum coverage target

**View Coverage (if needed):**
```bash
# Install tarpaulin
cargo install cargo-tarpaulin

# Generate coverage report
cargo tarpaulin --out Html
```

## Test Types

**Unit Tests:**
- Scope: Individual functions and simple logic paths
- Approach: Test generator functions (`gen_string_with_chars`, `gen_hex_string`, `gen_file_path`)
- Example target: `src/generators.rs` utilities (currently untested)

**Integration Tests:**
- Scope: Module trait implementations, config parsing, module selection
- Approach: Create AppConfig and run module `run()` methods
- Would verify: Output generation, exit conditions, speed factor application
- Location: `tests/integration_tests.rs` (would need to be created)

**E2E Tests:**
- Framework: Not used
- Alternative: Manual testing via `cargo run`
- CI-level testing: Only format and clippy checks, no functional E2E

## Common Patterns

**Async Testing:**
```rust
#[async_std::test]
async fn test_async_function() {
    let result = csleep(100).await;
    // assertions
}
```

**Error Testing (if implemented):**
```rust
#[test]
fn test_parse_speed_factor_validation() {
    let result = parse_speed_factor("-1");
    assert!(result.is_err());
}
```

**Trait Trait Testing (Module implementations):**
```rust
#[async_std::test]
async fn test_module_implementation() {
    let module = Bootlog;
    assert_eq!(module.name(), "bootlog");

    let config = AppConfig { /* ... */ };
    module.run(&config).await;
    // Verify output (would require capturing stdout)
}
```

## CI/CD Testing

**Pipeline Location:** `.github/workflows/ci.yml`

**Test Steps in CI:**
```yaml
- name: cargo build
  run: cargo build

- name: cargo test
  run: cargo test

- name: cargo fmt
  run: cargo fmt --all -- --check

- name: cargo clippy
  run: cargo clippy -- -D warnings
```

**Test Matrix:**
- Runs on: ubuntu-latest, windows-latest, macos-latest
- Rust: stable (latest stable toolchain)
- Components: rustfmt, clippy
- No platform-specific test differences

## Test Quality Assessment

**Current Gaps:**
- No unit tests for generator functions
- No validation tests for `AppConfig` parsing
- No integration tests for module implementations
- No test for timing-related functions (`csleep`, speed factor application)

**High-Priority Test Targets:**
1. `src/args.rs` - `parse_speed_factor()`, `parse_min_1()` value validators
2. `src/generators.rs` - `gen_string_with_chars()`, `gen_file_path()`, `gen_package_version()`
3. `src/lib.rs` - `AppConfig::should_exit()` exit condition logic
4. Module trait implementations - verify `signature()` returns non-empty strings

**Testing Challenges:**
- Output is direct to stdout/stderr (would need capture for integration tests)
- Async runtime complexity (requires `#[async_std::test]`)
- Global state (`SPEED_FACTOR`, `INSTANT_PRINT_LINES` mutexes) makes unit testing difficult

---

*Testing analysis: 2026-01-30*
