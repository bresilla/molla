SHELL := /bin/bash

PROJECT_NAME := $(shell sed -n '/^[[:space:]]*[^#\[[:space:]]/p' PROJECT | head -1 | tr -d '[:space:]')
PROJECT_VERSION := $(shell sed -n '/^[[:space:]]*[^#\[[:space:]]/p' PROJECT | sed -n '2p' | tr -d '[:space:]')
ifeq ($(PROJECT_NAME),)
    $(error Error: PROJECT file not found or invalid)
endif

CARGO := nixVulkan cargo
# DISPLAY=:1 is required so winit can find the host X server when
# cargo is invoked from contexts (nix shell, systemd user unit, etc.)
# that don't inherit the desktop session's DISPLAY env var. Only
# matters for targets that actually open a window (`run`).
# WAYLAND_DISPLAY= forces winit onto the X11 path: if both env vars
# are set, winit prefers Wayland and we want it pinned to X11.
RUN_ENV := DISPLAY=:1 WAYLAND_DISPLAY=

$(info ------------------------------------------)
$(info Project: $(PROJECT_NAME) v$(PROJECT_VERSION))
$(info ------------------------------------------)

.PHONY: build b compile c run r run-profile test t test-gpu test-featherstone-gpu test-xpbd-gpu test-raycast-gpu test-imu-gpu test-contact-gpu test-frame-transform-gpu test-oracle bench-cpu bench-gpu bench-featherstone check fmt clean help h

build:
	@$(CARGO) build --workspace --all-targets

b: build

compile:
	@$(CARGO) clean
	@$(MAKE) build

c: compile

# Launches `molla_viewer` (the Bevy app). `BIN` overrides the bin name.
# `RELEASE=1` builds with optimisations (recommended — debug builds are
# noticeably laggy under Bevy).
RELEASE_FLAG := $(if $(RELEASE),--release,)
BIN := molla_viewer
run:
	@$(RUN_ENV) $(CARGO) run $(RELEASE_FLAG) -p $(BIN) -- $(ARGS)

# Run with Tracy instrumentation. Build with `make run-profile RELEASE=1`,
# then in another terminal launch the Tracy GUI (`nix-shell -p tracy &&
# tracy` or download from https://github.com/wolfpld/tracy/releases) and
# click "Connect". Every Bevy schedule + system + render pass appears
# on the timeline.
run-profile:
	@$(RUN_ENV) $(CARGO) run $(RELEASE_FLAG) -p $(BIN) --features profile -- $(ARGS)

r: run

test:
	@$(CARGO) test --workspace

t: test

# GPU tests are gated behind `--ignored` because the wgpu adapter is
# only reachable through nixVulkan. `make test-gpu` runs only the
# GPU-specific parity test (skips the Newton-oracle tests, which need a
# venv'd Python — those run via `make test-oracle`).
test-gpu:
	@$(RUN_ENV) $(CARGO) test -p molla --features gpu --test cpu_gpu_parity -- --ignored

test-featherstone-gpu:
	@$(RUN_ENV) $(CARGO) test -p molla --features gpu --test featherstone_gpu_parity -- --ignored

test-xpbd-gpu:
	@$(RUN_ENV) $(CARGO) test -p molla --features gpu --test xpbd_gpu_parity -- --ignored

test-raycast-gpu:
	@$(RUN_ENV) $(CARGO) test -p molla --features gpu --test sensor_raycast_gpu -- --ignored

test-imu-gpu:
	@$(RUN_ENV) $(CARGO) test -p molla --features gpu --test sensor_imu_gpu -- --ignored

test-contact-gpu:
	@$(RUN_ENV) $(CARGO) test -p molla --features gpu --test sensor_contact_gpu -- --ignored

test-frame-transform-gpu:
	@$(RUN_ENV) $(CARGO) test -p molla --features gpu --test sensor_frame_transform_gpu -- --ignored

test-oracle:
	@$(RUN_ENV) $(CARGO) test -p molla --test oracle -- --ignored

bench-cpu:
	@$(CARGO) bench -p molla --bench step_cpu

bench-gpu:
	@$(RUN_ENV) $(CARGO) bench -p molla --bench step_gpu --features gpu

bench-featherstone:
	@$(RUN_ENV) $(CARGO) bench -p molla --bench featherstone_cpu_vs_gpu --features gpu

check:
	@$(CARGO) check --workspace --all-targets

fmt:
	@$(CARGO) fmt --all

clean:
	@$(CARGO) clean

help:
	@echo
	@echo "Usage: make [target]"
	@echo
	@echo "Available targets:"
	@echo "  build        Build the workspace (all targets)"
	@echo "  compile      Clean and rebuild"
	@echo "  run          Launch molla_viewer (RELEASE=1 recommended; BIN=name overrides)"
	@echo "  run-profile  Launch with Tracy instrumentation; connect Tracy GUI separately"
	@echo "  test         Run the test suite"
	@echo "  test-gpu     Run GPU-only tests via nixVulkan (CPU/GPU parity, etc.)"
	@echo "  bench-cpu    cargo bench the CPU integrator"
	@echo "  bench-gpu    cargo bench the GPU integrator via nixVulkan"
	@echo "  check        cargo check the workspace"
	@echo "  fmt          Format the workspace"
	@echo "  clean        Remove Cargo build artifacts"
	@echo

h: help
