# External Integrations

**Analysis Date:** 2026-01-30

## APIs & External Services

**None Detected**

This is a standalone application with no external API integrations. All data is generated internally using embedded datasets and random generation algorithms.

## Data Storage

**Databases:**
- Not used

**File Storage:**
- Local filesystem only - No external storage integration
- Static data embedded in binary via `include_str!()` macros

**Embedded Data Files:**
- Location: `data/` directory (embedded at compile time)
- Files loaded statically:
  - `bootlog.txt` - Boot log messages
  - `cfiles.txt` - C source file names
  - `packages.txt` - Package names
  - `composer.txt` - Composer packages
  - `simcity.txt` - SimCity output
  - `boot_hooks.txt` - Boot hook names
  - `os_releases.txt` - OS release names
  - `docker_packages.txt` - Docker packages
  - `docker_tags.txt` - Docker image tags
  - `ansible_roles.txt` - Ansible roles
  - `ansible_tasks.txt` - Ansible tasks
  - `rkhunter_checks.txt` - RootKit Hunter checks
  - `rkhunter_rootkits.txt` - RootKit Hunter rootkit names
  - `rkhunter_tasks.txt` - RootKit Hunter tasks
  - `julia.csv` - Julia packages (CSV format with name, id, versions)
  - `terraform_aws_resources.txt` - AWS Terraform resources
  - `terraform_azure_resources.txt` - Azure Terraform resources
  - `terraform_gcp_resources.txt` - GCP Terraform resources
  - `terraform_ids.txt` - Terraform resource IDs
  - `css_properties.txt` - CSS properties
  - `web_apis.txt` - Web API names
  - `wpt_categories.txt` - Web Platform Test categories

**Caching:**
- None detected

## Authentication & Identity

**Auth Provider:**
- Not applicable - No authentication system

## Monitoring & Observability

**Error Tracking:**
- Not detected - No error reporting service integration

**Logs:**
- Standard output (stdout) only
- WASM: Browser console via `web_sys::console`
- Sample output: "WARN: failed to set up idle inhibition: {err}" (keepawake errors)
- Exit message: "Saving work to disk..." (simulated exit sequence)

**Metrics:**
- None detected

## CI/CD & Deployment

**Hosting:**
- GitHub Pages (WASM web version): `https://svenstaro.github.io/genact/`
- Docker Hub (container images): `docker.io/svenstaro/genact`
- GitHub Releases (binary artifacts)
- crates.io (Rust package registry)
- Package managers: Homebrew, MacPorts, FreeBSD pkg, Scoop

**CI Pipeline:**
- GitHub Actions (workflows in `.github/workflows/`)

**CI/CD Services Used:**
- `dtolnay/rust-toolchain@stable` - Rust toolchain setup
- `houseabsolute/actions-rust-cross@v0` - Cross-compilation support
- `svenstaro/upx-action@v2` - Binary compression
- `actions/upload-artifact@v4` - Artifact storage
- `actions/download-artifact@v4` - Artifact retrieval
- `taiki-e/install-action@parse-changelog` - Changelog parsing
- `juliangruber/read-file-action@v1` - File reading in workflow
- `svenstaro/upload-release-action@v2` - GitHub release uploads
- `peaceiris/actions-gh-pages@v4` - GitHub Pages deployment
- `podman` - Container image building and pushing

**Release Automation:**
- GitHub Actions trigger on version tags (e.g., `v1.5.1`)
- Automated release creation with changelog entries
- Multi-platform binary compilation and upload
- Container image builds (amd64, aarch64, armv7) pushed to Docker Hub
- WASM web version deployed to GitHub Pages

**Webhook Configuration:**
- None - No incoming webhooks

## Environment Configuration

**Required Environment Variables:**
- `DOCKERHUB_USERNAME` (GitHub secret) - Docker Hub authentication
- `DOCKERHUB_TOKEN` (GitHub secret) - Docker Hub authentication
- `GITHUB_TOKEN` (auto-provided by GitHub Actions) - Release uploads and Pages deployment

**Configuration Method:**
- Command-line arguments (via `clap`)
- Query string parameters for web version (URL parsing)
- No `.env` file support

**Web Version URL Parameters:**
- `?module=MODULE_NAME` (repeatable) - Select specific modules
- `?speed-factor=N` - Global speed multiplier (float)
- `?instant-print-lines=N` - Lines to print instantly (integer)

## Webhooks & Callbacks

**Incoming Webhooks:**
- None detected

**Outgoing Webhooks:**
- None detected

## Third-Party Services & Dependencies

**Package Sources:**
- crates.io (Rust package registry) - All direct dependencies
- GitHub (source control) - Repository hosted on GitHub
- Docker Hub - Container distribution

**Dependency Management:**
- Cargo.lock committed (reproducible builds)
- Dependabot configured (`.github/dependabot.yml`) for automated dependency updates
- 236 total transitive dependencies

**No External Service Integrations:**
- No cloud storage (AWS S3, Azure Storage, GCP Cloud Storage)
- No database services (PostgreSQL, MongoDB, MySQL)
- No message queues (RabbitMQ, Kafka)
- No cache services (Redis, Memcached)
- No authentication services (OAuth, Auth0, Cognito)
- No analytics (Google Analytics, Mixpanel)
- No error tracking (Sentry, Rollbar)
- No logging services (ELK, Datadog, Splunk)
- No CDN
- No payment processing

---

*Integration audit: 2026-01-30*
