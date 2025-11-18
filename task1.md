# Task 1 – Project Catalog

## Root Documentation & Metadata
1. `README.md`
   - **Functionality:** Plain-language tour of CAF’s goals, commands, and layout.
   - **Purpose:** First stop for any student who needs setup or usage instructions.
   - **Relationships:** Mentions every major directory so readers know where to look next.
2. `pyproject.toml` (root)
   - **Functionality:** Central workspace configuration that defines project metadata and shared settings for tools like pytest and Ruff.
   - **Purpose:** Purpose: Ensures consistent tooling behavior across all subpackages, even though the root itself is not a Python package.
   - **Relationships:** pytest and Ruff load these settings regardless of which submodule you’re working in or testing.
   - **Insight:** Setting packages = [] explicitly marks the root as a workspace container, making it clear that actual Python packages live in subdirectories.
3. `.envrc`
   - **Functionality:** Sets `ENABLE_COVERAGE` when `direnv` enters the repo.
   - **Purpose:** Gives the Makefile and Docker scripts a single toggle for coverage runs.
   - **Relationships:** Targets in the Makefile read the same variable to decide whether to collect coverage.
   - **Insight:** Help us optimize the test, by checking that we cover all the code while testing.

## Tooling & Automation
4. `Makefile`
   - **Functionality:** Single entry point for building/running the Docker container, deploying both packages, toggling coverage, testing, and cleaning artifacts.
   - **Purpose:** Ensures every environment (local, CI, grading) runs the exact same commands.
   - **Relationships:** Calls the Dockerfile to build images, installs `libcaf` before `caf`, and forwards `ENABLE_COVERAGE` from `.envrc`.
   - **Insight:** Shourtcuts to easily build and verify the hybrid Python/C++ stack.
5. `deployment/Dockerfile`
   - **Functionality:** Produces the “caf-dev” image with compilers, CMake, pybind11, pytest/coverage, Ruff, and a ready-to-use virtualenv.
   - **Purpose:** Removes “works on my machine” issues when compiling `_libcaf` or running coverage-heavy tests.
   - **Relationships:** Used by both make build-container and CI. It also copies .envrc into the container so tests run with the same settings everywhere.
   - **Insight:** With every dependency pre-installed, students can go straight to `make run` instead of fighting local toolchains.


## Python CLI Package (`caf/`)
6. `caf/pyproject.toml`
    - **Functionality:** Declares the CLI package, console entry point, and optional test extras.
    - **Purpose:** Allows `pip install -e caf` so the `caf` command is globally available.
    - **Relationships:** Called by the Makefile after `libcaf` is built to ensure bindings resolve.
    - **Insight:** Separate packaging lets the CLI evolve independently from the core library.
7. `caf/caf/__main__.py`
    - **Functionality:** Tiny bridge that simply calls `cli()` when `python -m caf` or the `caf` script runs.
    - **Purpose:** Gives Python a predictable entry point without duplicating logic.
    - **Relationships:** Imports `cli` from `caf/cli.py`; setuptools points the console script here.
    - **Insight:** Because it only forwards calls, the real behavior (and tests) stay in `cli.py`.
8. `caf/caf/cli.py`
    - **Functionality:** Builds the `argparse` command tree, registers handlers, and exits with each command’s return code.
    - **Purpose:** Centralizes user interaction so every CLI verb works the same way.
    - **Relationships:** Pulls defaults from `libcaf.constants` and calls functions in `cli_commands`.
    - **Insight:** The user interaction with caf.
9. `caf/caf/cli_commands.py`
    - **Functionality:** Implements each CLI verb (init, commit, branch, diff, etc.), prints user-facing messages, and returns status codes.
    - **Purpose:** Turns user actions into repository operations while keeping errors readable.
    - **Relationships:** Heavily uses `libcaf.repository.Repository`, `libcaf.plumbing`, and `libcaf.ref`.
    - **Insight:** Doing the work here keeps the CLI layer thin and easy to extend.

## Python Library Layer (`libcaf/libcaf`)
10. `libcaf/pyproject.toml`
    - **Functionality:** Configures scikit-build-core so `pip install -e libcaf` compiles the C++ sources.
    - **Purpose:** Packages the native module plus its Python glue in one command.
    - **Relationships:** The Makefile passes `CMAKE_ARGS` through this file to toggle coverage.
    - **Insight:** Using scikit-build keeps the packaging story close to standard Python tooling.
11. `_libcaf.cpython-*.so` and `_libcaf.pyi`
    - **Functionality:** Compiled pybind11 extension plus matching type stubs.
    - **Purpose:** Provide fast hashing, IO, and data structures that Python code imports.
    - **Relationships:** Re-exported by `libcaf/__init__.py` and wrapped by `plumbing.py`.
    - **Insight:** Compiled merged file of all the C++ files.
12. `libcaf/libcaf/plumbing.py`
    - **Functionality:** Python-friendly wrappers around `_libcaf` for hashing, saving blobs/trees/commits, and opening file descriptors.
    - **Purpose:** Shields higher layers from low-level file-handle juggling.
    - **Relationships:** Called by `Repository`, CLI commands, and tests that need direct object IO.
    - **Insight:** Convert Path to string to send the function to the C++ implementations.
13. `libcaf/libcaf/ref.py`
    - **Functionality:** Defines `HashRef`, `SymRef`, parser/writer helpers, and validation logic.
    - **Purpose:** Read and write refs.
    - **Relationships:** Used across `Repository` and CLI branch commands.
    - **Insight:** Our tool that can understand whats written in a ref, and write a ref for a file.
14. `libcaf/libcaf/repository.py`
    - **Functionality:** High-level API for repos: init, branch management, commits, logs, diffs, and helper dataclasses.
    - **Purpose:** Serves as the “porcelain” layer every other component calls.
    - **Relationships:** Depends on constants, plumbing, refs, and the `_libcaf` data types.
    - **Insight:** The main file that all the CLI commands call, and that assign tasks for plumbing and refs, everything about the implementation of the caf functions starts here.

## Native Core (`libcaf/src`)
15. `caf.h` / `caf.cpp`
    - **Functionality:** Implement hashing, blob storage layout, file locking, and content read/write operations using OpenSSL and POSIX calls.
    - **Purpose:** Provide the high-speed, race-safe foundation for content-addressable storage.
    - **Relationships:** Wrapped by pybind11 so Python can call these routines, and used by `object_io`.
    - **Insight:** Responsable for hashing files, save, delete, read and write from the work files.
16. `hash_types.h` / `hash_types.cpp`
    - **Functionality:** Define overloaded `hash_object` helpers for blobs, trees, and commits.
    - **Purpose:** Ensure every object type has a consistent hashing recipe.
    - **Relationships:** Called by `object_io` and exposed to Python via `_libcaf`.
    - **Insight:** Hashing tree records deterministically is key for accurate diffs.
17. `object_io.h` / `object_io.cpp`
    - **Functionality:** Serialize and deserialize commits and trees with length-prefixed binary formats and strict locking.
    - **Purpose:** Turn rich objects into bytes stored under their hash and back again.
    - **Relationships:** Use the file helpers from `caf.cpp` and hashing from `hash_types.cpp`.
    - **Insight:** The `MAX_LENGTH` guardrails defend against corrupt inputs—important in a student repo.
18. `blob.h`, `commit.h`, `tree_record.h`, `tree.h`
    - **Functionality:** Define the in-memory C++ data structures used across the native core.
    - **Purpose:** Provide strong typing for pybind11 and internal helpers.
    - **Relationships:** Included by almost every native file and mirrored on the Python side.
    - **Insight:** Immutable members (const fields) emphasize that repository objects never mutate.
19. `bind.cpp`
    - **Functionality:** Pybind11 module that exposes hashing, IO helpers, and the data classes to Python.
    - **Purpose:** Acts as the bridge between C++ speed and Python ergonomics.
    - **Relationships:** Depends on every header above and compiles into `_libcaf`.
    - **Insight:** The bridge between python and C++.

## Test Suite
20. `tests/conftest.py`
    - **Functionality:** Pytest fixtures for temp repos, random files, and CLI output parsing.
    - **Purpose:** Gives every test an isolated workspace without repeated setup code.
    - **Relationships:** Shared by both CLI and library tests.
21. `tests/libcaf/*.py`
    - **Functionality:** Unit tests for repository internals (hashing, refs, diffs, object IO, etc.).
    - **Purpose:** Prove the Python layer around the native module behaves correctly.
    - **Relationships:** Hit `repository.py`, `ref.py`, `plumbing.py`, and the pybind11 classes directly.
22. `tests/caf/cli_commands/*.py`
    - **Functionality:** Tests for each CLI verb that assert outputs, exit codes, and repo effects.
    - **Purpose:** Ensure the user-facing commands stay stable and friendly.
    - **Relationships:** Depend on the fixtures above and call into CLI command functions.