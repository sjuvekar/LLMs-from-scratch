# Task: C++ Project Toolchain Initialization

## 1. Objective
Design and implement a modern, scalable C++ toolchain for a new project. You are responsible for researching the best build system for a project that targets high performance and cross-platform compatibility (Linux/Windows/macOS).

## 2. Phase 1: Research & Selection
Before writing any code, research and compare the following build systems in the context of 2026 standards:
* **CMake:** The industry standard. Evaluate current best practices (Target-based approach, FetchContent, etc.).
* **Meson:** Known for speed and a Python-like DSL.
* **Bazel:** For high-scale, hermetic builds and remote caching.
* **XMake / CForge:** Evaluate modern, lightweight alternatives that prioritize developer experience.

### Deliverable 1:
Provide a table comparing these systems based on:
1.  **Build Speed** (Incremental & Clean).
2.  **Dependency Management** (Ease of integrating vcpkg/Conan).
3.  **IDE Support** (VS Code, CLion, Visual Studio 2026).
4.  **Learning Curve.**

> **STOP:** Do not proceed to Phase 2 until I have selected a system from your research.

---

## 3. Phase 2: Implementation (Pending Selection)
Once I select a system, you will generate the following:
* **Root Build File:** (e.g., `CMakeLists.txt`, `meson.build`, or `BUILD.bazel`).
* **Toolchain Rules:** Compiler flags for Clang/GCC/MSVC (enforcing C++20/23/26 standards).
* **Structure:** Create a standard directory layout:
    * `/src` (Implementation)
    * `/include` (Public headers)
    * `/tests` (Unit tests)
    * `/scripts` (Build helper scripts)

## 4. Constraints
* **Modernity:** Use the latest stable C++ standard available.
* **Clarity:** Comments in build files must explain *why* a specific flag or rule is used.
* **Strictness:** Enable high warning levels (`-Wall -Wextra -Wpedantic` or `/W4`).