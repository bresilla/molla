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

.PHONY: build b compile c run r test t check fmt clean help h

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

r: run

test:
	@$(CARGO) test --workspace

t: test

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
	@echo "  test         Run the test suite"
	@echo "  check        cargo check the workspace"
	@echo "  fmt          Format the workspace"
	@echo "  clean        Remove Cargo build artifacts"
	@echo

h: help
