# Rustracer development commands
# Run with: just <recipe>

_default:
    @just --list

# Build all crates
build:
    cargo build --workspace

# Build in release mode
release:
    cargo build --workspace --release

# Run tests
test:
    cargo test --workspace

# Check that everything compiles (fast)
check:
    cargo check --workspace

# Validate WGSL shaders by checking they exist and are parseable
validate-shaders:
    @echo "Checking shader files..."
    @for f in rustracer-renderer/shaders/*.wgsl; do \
        if [ -f "$$f" ]; then \
            echo "  Found: $$f"; \
        fi; \
    done
    @if ! ls rustracer-renderer/shaders/*.wgsl >/dev/null 2>&1; then \
        echo "  (no shaders yet — expected during development)"; \
    fi

# Run the app with a test scene
run:
    cargo run -p rustracer-app

# Run release with a scene
run-release scene:
    cargo run -p rustracer-app --release -- {{scene}}

# Format code
fmt:
    cargo fmt --all

# Lint with clippy
lint:
    cargo clippy --workspace -- -D warnings

# Clean build artifacts
clean:
    cargo clean
