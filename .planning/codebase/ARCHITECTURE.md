# Architecture

**Analysis Date:** 2026-01-30

## Pattern Overview

**Overall:** Plugin/Module-Based Simulation Engine

**Key Characteristics:**
- Multiple independent simulation modules (activity generators) loaded into a registry
- Trait-based polymorphism via the `Module` trait for extensibility
- Asynchronous execution model using `async_std` for both native and WebAssembly targets
- Dual-target architecture (native binary + WebAssembly) with conditional compilation
- Global state management via `lazy_static` for configuration and runtime metrics

## Layers

**Module Layer (Simulation):**
- Purpose: Encapsulates fake activity scenarios (cargo builds, Docker operations, boot logs, etc.)
- Location: `src/modules/`
- Contains: 20+ module implementations (each in its own file)
- Depends on: `Module` trait, data generators, I/O utilities, application config
- Used by: Main run loop in `lib.rs`

**Data Layer:**
- Purpose: Provides pre-loaded static data lists (package names, C files, boot messages, Terraform resources)
- Location: `src/data.rs`, `data/` directory
- Contains: Static strings included at compile-time, lazy-loaded `Vec<&'static str>` collections
- Depends on: File includes from `data/` directory
- Used by: All module implementations for realistic content generation

**Generator Layer:**
- Purpose: Utility functions for procedurally generating realistic fake data (file paths, version strings, random flags)
- Location: `src/generators.rs`
- Contains: Pure functions for randomized content synthesis (strings, hex, paths, versions)
- Depends on: `rand` and `rand_distr` crates for statistical distributions
- Used by: Module implementations to construct fake system outputs

**I/O Layer:**
- Purpose: Platform-abstracted output and timing for both native terminals and WebAssembly xterm
- Location: `src/io.rs`
- Contains: Print, character-delay print, newline, sleep functions with dual implementations
- Depends on: `async_std`, `yansi` for colors (native); `wasm-bindgen` for Web target
- Used by: All module implementations for controlled output with timing

**Configuration & Orchestration Layer:**
- Purpose: Parses command-line arguments (native) or URL parameters (Web), manages global state
- Location: `src/args.rs`, `src/lib.rs`, `src/main.rs`
- Contains: `AppConfig` struct, argument parsing, exit condition checks, main run loop
- Depends on: `clap` for CLI (native only), module registry
- Used by: Entry point (`main.rs`) and modules for configuration values

## Data Flow

**Native Binary Execution:**

1. User invokes binary with arguments
2. `main.rs` entry point calls `parse_args()`
3. `args.rs` validates arguments via clap, defaults to all modules if none specified
4. Global state (`SPEED_FACTOR`, `INSTANT_PRINT_LINES`) initialized
5. `lib.rs::run()` begins infinite loop
6. Loop randomly selects a module from `ALL_MODULES` registry (weighted by user selection)
7. Selected module executes `Module::run()` with app config
8. Module uses data layer to access lists, generators to create output, I/O layer for controlled printing
9. Loop checks exit conditions (`should_exit()`) after each module run
10. On exit (CTRL-C or time/count limit), `exit_handler()` prints message and exits

**Web (WASM) Execution:**

1. Browser loads `index.html`
2. WASM module instantiates, panic hook registered
3. `parse_args()` reads URL query parameters (`?module=cc&module=memdump&speed-factor=2`)
4. Config initialized with parsed values
5. `lib.rs::run()` begins loop (infinite for Web, no exit conditions)
6. Same module selection and execution as native path
7. Output written to embedded xterm terminal via `write_to_xterm()` JS interop

**State Management:**

- `SPEED_FACTOR`: Mutable global controlling output delay (milliseconds per character)
- `INSTANT_PRINT_LINES`: Atomic counter for lines to print immediately (bypassing delays)
- `CTRLC_PRESSED`: Atomic flag for native CTRL-C handler
- `STARTED_AT`: Instant snapshot for measuring elapsed time
- `MODULES_RAN`: Atomic counter for total modules executed (native only)
- `COUNTER`: Static atomic tracking output lines for instant printing logic

All global state uses thread-safe primitives: `Mutex` for async operations, `AtomicBool/AtomicU32` for flags and counters.

## Key Abstractions

**Module Trait:**
- Purpose: Defines contract for any activity generator
- File: `src/modules/mod.rs`
- Methods: `name()` → static name, `signature()` → command being simulated, `run(AppConfig)` → async execution
- Examples: `src/modules/cargo.rs` (simulates Rust build), `src/modules/cc.rs` (C compiler), `src/modules/bootlog.rs` (system boot)
- Pattern: `#[async_trait(?Send)]` for non-Send async in single-threaded context

**ALL_MODULES Registry:**
- Purpose: Static BTreeMap storing all available modules as boxed trait objects
- Location: `src/modules/mod.rs` in `lazy_static!` block
- Pattern: Key = module name string, Value = `Box<dyn Module>`
- Initialization: Inserts 20 module instances at startup

**AppConfig:**
- Purpose: Single source of truth for runtime configuration
- Location: `src/args.rs`
- Fields (native): `modules` (Vec of selected modules), `speed_factor` (f32), `instant_print_lines` (u32), `inhibit` (bool for keepawake), `exit_after_time` (Option<Duration>), `exit_after_modules` (Option<u32>)
- Fields (WASM): Simplified version without CLI-specific fields
- Method: `should_exit()` checks termination conditions

## Entry Points

**Native Binary Entry:**
- Location: `src/main.rs` (non-WASM target)
- Triggers: Execution of compiled binary, user provides CLI arguments
- Responsibilities: Argument parsing, shell completion/man page generation, CTRL-C handler setup, calls `lib::run()`

**WebAssembly Entry:**
- Location: `src/main.rs` (WASM target)
- Triggers: WASM module instantiation by browser
- Responsibilities: Panic hook registration, URL parameter parsing, calls `lib::run()`

**Core Run Loop:**
- Location: `src/lib.rs::run()`
- Triggers: Called from entry point after config initialization
- Responsibilities: Infinite loop, random module selection, module execution, keepawake management, exit condition checking

## Error Handling

**Strategy:** Minimal error handling, graceful degradation

**Patterns:**
- Fallback defaults: Generator functions use `.unwrap_or()` to default to empty strings or placeholder values if data unavailable
- WASM-specific: Speed factor parsing defaults to 1.0 if invalid URL parameter
- I/O: Platform-specific implementations silently handle missing terminal capabilities (e.g., ignored in WASM)
- Module execution: No try/catch around module runs; panic would show console error hook message
- Keepawake: Non-fatal, logs warning if setup fails, continues execution

## Cross-Cutting Concerns

**Logging:** None. All output is simulated terminal activity via `print()`, `dprint()`, or module-specific formatting with `yansi` colors.

**Validation:** Lightweight - clap handles CLI validation (speed factor > 0.01, module names), URL params parsed with `.unwrap_or()` defaults, data consistency not validated (assumes data files are correct).

**Authentication:** None. Genact is a standalone utility with no external services or auth requirements.

**Platform Abstraction:** Achieved via `#[cfg(target_arch = "wasm32")]` conditional compilation:
- I/O redirection: stdout → xterm JS function or native stdout
- Timing: `async_std::task::sleep()` → `window.setTimeout()` promise wrapper
- Keepawake: Conditional compilation of `keepawake` crate (not available on WASM)
- Exit handling: Native only (WASM runs indefinitely)

---

*Architecture analysis: 2026-01-30*
