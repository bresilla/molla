SHELL := /bin/bash

ifneq ($(wildcard PROJECT),)
PROJECT_NAME := $(shell sed -n '/^[[:space:]]*[^#\[[:space:]]/p' PROJECT | head -1 | tr -d '[:space:]')
PROJECT_VERSION := $(shell sed -n '/^[[:space:]]*[^#\[[:space:]]/p' PROJECT | sed -n '2p' | tr -d '[:space:]')
else
PROJECT_NAME := molla
PROJECT_VERSION := $(shell sed -n 's/^version = "\([^"]*\)"/\1/p' crates/molla/Cargo.toml | head -1)
endif
ifeq ($(PROJECT_NAME),)
    $(error Error: PROJECT file not found or invalid)
endif

TOP_DIR := $(CURDIR)
DOCS_DIR ?= $(TOP_DIR)/docs
CARGO := cargo
BACKEND ?= x11
DISPLAY ?= :1
APP_PKG ?= molla_viewer
APP_TARGET := -p $(APP_PKG)
RUN_WITH ?= nixVulkan
BENCH_ARGS ?=
SENSOR_BENCH ?= bench_scene_queries
SENSOR_BENCH_PKG ?= molla-sensors
LATENCY_BENCH ?= bench_sensor_latency
NVIDIA_SMI ?= nvidia-smi
FMT_ARGS ?=
TYPE ?= patch
HAS_REL := $(shell command -v git-rel 2>/dev/null)

# GPU test lane (see `test-gpu`): package P + test file T, extra args A.
P ?= molla-solvers
T ?= cpu_gpu_parity
REL := $(if $(RELEASE),--release,)

$(info ------------------------------------------)
$(info Project: $(PROJECT_NAME) v$(PROJECT_VERSION))
$(info Display: $(BACKEND) backend)
$(info ------------------------------------------)

.PHONY: build b compile c run r test t test-gpu test-sensors test-sensor-resources test-viewer-sensors test-all check check-all fmt-files harden bench bench-sensors bench-sensors-latency sensor-bench-info clean docs release help h

build:
	@$(CARGO) build $(APP_TARGET)

b: build

compile:
	@$(CARGO) clean
	@$(MAKE) build

c: compile

run:
	@$(RUN_WITH) $(CARGO) run $(APP_TARGET)

r: run

test:
	@$(CARGO) test $(APP_TARGET)

t: test

# GPU tests need a real wgpu adapter (only via nixVulkan) and are
# #[ignore]-marked. Pick the package P and test file T; pass extra args
# via A and add RELEASE=1 for perf lanes:
#   make test-gpu                                  # CPU/GPU parity lane
#   make test-gpu P=molla-geometry T=gjk_gpu A=--nocapture
#   make test-gpu P=molla-fem T=fem_ns_gpu RELEASE=1 A='--skip gpu_ns_perf...'
test-gpu:
	@$(RUN_WITH) $(CARGO) test $(REL) -p $(P) --features gpu --test $(T) -- --ignored $(A)

test-sensors:
	@$(MAKE) test APP_TARGET='-p molla-compute --features gpu --lib'
	@$(MAKE) test APP_TARGET='-p molla-geometry --no-default-features --lib mesh_bvh::tests'
	@$(MAKE) test APP_TARGET='-p molla-geometry --features gpu --lib validation_tests'
	@$(MAKE) test APP_TARGET='-p molla-sensors --no-default-features'
	@$(MAKE) test APP_TARGET='-p molla-sensors --features gpu --lib'
	@$(MAKE) test APP_TARGET='-p molla-sensors --features gpu --test sensor_allocation_gpu'
	@$(MAKE) test APP_TARGET='-p molla-sensors --features gpu --test sensor_large_coordinates_gpu'
	@$(MAKE) test APP_TARGET='-p molla --no-default-features --test sensor_api'
	@$(MAKE) test APP_TARGET='-p molla --features gpu --test sensor_api'
	@$(MAKE) test-gpu P=molla T=sensor_physics_gpu A='--test-threads=1'
	@$(RUN_WITH) $(CARGO) test $(REL) -p molla-sensors --features gpu --tests -- --ignored --test-threads=1
	@$(MAKE) test-sensor-resources

# Warm-frame GPU object retention and heap-allocation attribution (wgpu counters).
test-sensor-resources:
	@$(RUN_WITH) $(CARGO) test $(REL) -p molla-sensors --features resource-counters --test sensor_resource_retention_gpu -- --ignored --test-threads=1 $(A)

test-viewer-sensors:
	@$(RUN_WITH) $(CARGO) test $(REL) -p molla_viewer --bin molla_viewer -- sensor_ --include-ignored --test-threads=1 $(A)

test-all:
	@$(CARGO) test --workspace --all-targets

check:
	@$(CARGO) check $(APP_TARGET)

check-all:
	@$(CARGO) check --workspace --all-targets

fmt-files:
	@test -n "$(FILES)" || { echo 'Set FILES to the Rust files to format'; exit 1; }
	@rustfmt --edition 2024 $(FMT_ARGS) $(FILES)

harden:
	@git diff --check
	@$(CARGO) fmt --all -- --check
	@$(CARGO) check --workspace --no-default-features
	@$(CARGO) test --workspace --no-default-features
	@$(CARGO) clippy --workspace --all-targets --all-features -- -D warnings
	@$(RUN_WITH) $(CARGO) test --workspace --all-targets --all-features

bench:
	@$(RUN_WITH) $(CARGO) bench $(BENCH_ARGS)

bench-sensors:
	@$(RUN_WITH) $(CARGO) bench -p $(SENSOR_BENCH_PKG) --features gpu --bench $(SENSOR_BENCH) -- $(A)

sensor-bench-info:
	@$(CARGO) --version
	@rustc --version
	@uname -srmo
	@git rev-parse HEAD
	@printf 'Dirty/untracked entries: '; git status --porcelain | wc -l
	@printf 'Crate source/manifests fingerprint: '; find crates -type f \( -name '*.rs' -o -name '*.wgsl' -o -name Cargo.toml \) -print0 | sort -z | xargs -0 sha256sum | sha256sum
	@sha256sum Cargo.toml Cargo.lock Makefile flake.nix flake.lock
	@if test -r /proc/driver/nvidia/version; then cat /proc/driver/nvidia/version; fi
	@printf 'NVIDIA telemetry command: %s\n' '$(NVIDIA_SMI)'
	@if command -v nvidia-smi >/dev/null; then $(NVIDIA_SMI) --query-gpu=name,driver_version,utilization.gpu,memory.used,memory.total --format=csv; fi

bench-sensors-latency: sensor-bench-info
	@$(MAKE) bench-sensors SENSOR_BENCH=$(LATENCY_BENCH) A='$(A)'

docs:
	@command -v mdbook >/dev/null 2>&1 || { echo "mdbook is not installed. Please install it first."; exit 1; }
	@mdbook build $(TOP_DIR)/book --dest-dir $(DOCS_DIR)

release:
	@if [ -z "$(HAS_REL)" ]; then \
		echo "git-rel is not installed. Please install it first."; \
		exit 1; \
	fi
	@if [ -z "$(TYPE)" ]; then \
		echo "Release type not specified. Use 'make release TYPE=[patch|minor|major|m.m.p]'"; \
		exit 1; \
	fi
	@git rel $(TYPE)

clean:
	@$(CARGO) clean

help:
	@echo
	@echo "Usage: make [target]"
	@echo
	@echo "Available targets:"
	@echo "  build        Build $(APP_PKG)"
	@echo "  compile      Clean and rebuild"
	@echo "  run          Run $(APP_PKG) ($(BACKEND) backend, $(RUN_WITH) wrapper)"
	@echo "  test         Test $(APP_PKG)"
	@echo "  test-gpu     Run a GPU test lane via $(RUN_WITH) (P= T= A= RELEASE=)"
	@echo "  test-sensors Run sensor host, feature and explicit GPU regression lanes"
	@echo "  test-viewer-sensors Run headless viewer sensor lifecycle tests on GPU (RELEASE= A=)"
	@echo "  test-all     Run the full workspace all-target test suite"
	@echo "  check        Check $(APP_PKG)"
	@echo "  check-all    Check the full workspace all-target suite"
	@echo "  fmt-files    Format explicit Rust paths (FILES= FMT_ARGS=--check)"
	@echo "  harden       Run diff whitespace check + fmt/check + strict clippy + all-feature tests"
	@echo "  bench        Run benchmarks (BENCH_ARGS=)"
	@echo "  bench-sensors Run GPU sensor benchmarks (SENSOR_BENCH=bench_scene_queries A=)"
	@echo "  bench-sensors-latency Record per-frame latency matrix with build/device metadata"
	@echo "  sensor-bench-info Print source fingerprint, toolchain and device metadata"
	@echo "  docs         Build documentation with mdbook (DOCS_DIR=)"
	@echo "  release      Create a new release (TYPE=patch|minor|major|m.m.p)"
	@echo "  clean        Remove Cargo build artifacts"
	@echo
	@echo "Examples:"
	@echo "  make run"
	@echo "  make run APP_PKG=other        # run a different package"
	@echo "  make run BACKEND=x11          # force X11 / XWayland (.envrc auto-detects)"
	@echo "  make run RUN_WITH=nixGL       # OpenGL wrapper instead of Vulkan"
	@echo "  make run RUN_WITH=            # no wrapper (native run)"
	@echo "  make test-gpu P=molla-geometry T=gjk_gpu A=--nocapture"
	@echo "  make bench-sensors A=scene_queries_1024_rays/hierarchy.*warm"
	@echo "  make bench-sensors SENSOR_BENCH=bench_sensor_lidar A=--test"
	@echo

h: help
