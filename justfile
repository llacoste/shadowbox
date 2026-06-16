image := "shadowbox:dev"
release_image := "ghcr.io/llacoste/shadowbox:latest"

default: build

# Build the Docker image used for everything else.
docker-build:
    docker build -t {{image}} .

# Alias kept for consistency with other repos' default recipe.
build: docker-build

# Run the locally-built image's CLI against the given args.
# Example: just slice tests/fixtures/mountains.png --layers 5 --output tmp/
slice *args: docker-build
    docker run --rm -v "$(pwd):/work" {{image}} slice {{args}}

# Boot the web UI on http://localhost:8000.
serve: docker-build
    docker run --rm -p 8000:8000 -v "$(pwd):/work" {{image}} serve --host 0.0.0.0

# Run the latest published release from GHCR. No local build needed.
# Example: just run slice mountains.png --layers 5
run *args:
    docker run --rm -v "$(pwd):/work" {{release_image}} {{args}}

# Run the test suite inside the container. Source/tests are baked at /app.
test: docker-build
    docker run --rm --workdir /app --entrypoint pytest {{image}} --color=yes

# Lint with ruff.
lint: docker-build
    docker run --rm --workdir /app --entrypoint ruff {{image}} check src tests

# Format check with ruff.
format-check: docker-build
    docker run --rm --workdir /app --entrypoint ruff {{image}} format --check src tests

# Apply ruff formatting in-place (mounts the host workspace so edits land outside the container).
format: docker-build
    docker run --rm -v "$(pwd):/host" --workdir /host --entrypoint ruff {{image}} format src tests

# Type-check with mypy.
typecheck: docker-build
    docker run --rm --workdir /app --entrypoint mypy {{image}} src

# Drop into a shell inside the image. Mounts the host workspace at /work.
shell: docker-build
    docker run --rm -it -v "$(pwd):/work" --entrypoint /bin/bash {{image}}

# Remove generated outputs.
clean:
    rm -rf tmp out *.zip

# Remove all caches and generated artifacts.
clean-all: clean
    rm -rf .pytest_cache .mypy_cache .ruff_cache __pycache__ src/*.egg-info build dist
