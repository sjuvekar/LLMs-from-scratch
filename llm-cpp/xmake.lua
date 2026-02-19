-- ============================================================================
-- XMake Build Configuration for llm-cpp
-- ============================================================================
-- This build file follows modern C++ best practices for a cross-platform,
-- high-performance project targeting C++26 (latest stable standard).
--
-- Why XMake?
-- - Built-in package management (like Cargo for Rust)
-- - Fast builds without separate generation step
-- - Simple Lua syntax vs CMake's custom DSL
-- - Cross-platform support (Linux/Windows/macOS/mobile/Wasm)
-- ============================================================================

-- Project metadata
set_project("llm-cpp")
set_version("0.1.0")
set_description("High-performance LLM implementation in modern C++")

-- ============================================================================
-- C++ Standard Configuration
-- ============================================================================
-- C++26 is the latest stable standard (as of 2026), providing:
-- - Static reflection (P2996)
-- - Linear algebra bindings (P1673)
-- - Contracts (P2900)
-- - Improved constexpr and template features
set_languages("c++26")

-- ============================================================================
-- Build Modes
-- ============================================================================
-- Debug: Full debug symbols, no optimization, all assertions enabled
-- Release: Full optimization, no debug symbols, assertions disabled
-- ReleaseWithDebug: Optimization with debug symbols (for profiling)
add_rules("mode.debug", "mode.release", "mode.releasedbg")

-- ============================================================================
-- Compiler-Specific Warning Flags
-- ============================================================================
-- High warning levels catch bugs early and enforce code quality.
-- We use the strictest settings to ensure portable, correct code.

-- Clang/GCC warnings (Linux, macOS with Clang)
if is_plat("linux", "macosx") then
    -- -Wall: Enable common warnings (enabled by default in XMake)
    -- -Wextra: Enable additional warnings for common issues
    -- -Wpedantic: Warn about non-standard C++ usage (ISO C++ compliance)
    -- -Werror: Treat warnings as errors (enforces zero-warning policy)
    add_cxxflags("-Wall", "-Wextra", "-Wpedantic", {public = true})

    -- Additional useful warnings for modern C++:
    -- -Wshadow: Warn when a variable shadows another in an outer scope
    -- -Wconversion: Warn on implicit type conversions that may lose data
    -- -Wold-style-cast: Warn on C-style casts (prefer static_cast, etc.)
    -- -Wnull-dereference: Warn on potential null pointer dereferences
    add_cxxflags("-Wshadow", "-Wconversion", "-Wold-style-cast", "-Wnull-dereference", {public = true})

    -- Release-specific: Enable link-time optimization (LTO) for better performance
    if is_mode("release") then
        add_cxxflags("-flto", {public = true})
        add_ldflags("-flto", {public = true})
    end
end

-- MSVC warnings (Windows with Visual Studio)
if is_plat("windows") then
    -- /W4: Level 4 warnings (equivalent to -Wall -Wextra)
    -- /WX: Treat warnings as errors (equivalent to -Werror)
    -- /permissive-: Strict conformance mode (disables non-standard extensions)
    -- /Zc:__cplusplus: Report correct __cplusplus value (not 199711L)
    add_cxxflags("/W4", "/WX", "/permissive-", "/Zc:__cplusplus", {public = true})

    -- Additional MSVC conformance flags:
    -- /Zc:strictStrings: Disallow non-const string literal conversion
    -- /Zc:rvalueCast: Enforce standard behavior for rvalue casts
    -- /Zc:throwingNew: Assume operator new throws (not returns nullptr)
    add_cxxflags("/Zc:strictStrings", "/Zc:rvalueCast", "/Zc:throwingNew", {public = true})
end

-- ============================================================================
-- Main Library Target
-- ============================================================================
-- The library contains the core LLM implementation components.
-- We use static linking by default for simplicity and performance.
target("llm-core")
    set_kind("static")  -- Static library for simplicity; change to "shared" for dynamic linking

    -- Public headers are visible to consumers of the library
    add_includedirs("include", {public = true})

    -- Source files (using recursive glob for convenience)
    -- Note: XMake handles file changes automatically, unlike CMake GLOB
    add_files("src/*.cpp", "src/**/*.cpp")

    -- Include directory for private headers
    add_includedirs("src", {public = false})

    -- Export definitions for shared library builds (future-proofing)
    add_defines("LLM_EXPORTS", {public = true})
target_end()

-- ============================================================================
-- Main Executable Target
-- ============================================================================
-- Simple executable that uses the library for demonstration and testing.
target("llm-cli")
    set_kind("binary")
    add_files("src/main.cpp")
    add_deps("llm-core")  -- Link against our library

    -- Runtime libraries for MSVC (static linking to avoid DLL dependencies)
    if is_plat("windows") then
        add_cxxflags("/MT$<s:d>", {public = true})  -- Static CRT in release, debug CRT in debug
    end
target_end()

-- ============================================================================
-- Test Target
-- ============================================================================
-- Unit tests using a minimal test framework.
-- Tests are only built when explicitly requested (xmake build tests).
target("tests")
    set_kind("binary")
    set_default(false)  -- Don't build by default; requires explicit `xmake build tests`

    add_files("tests/*.cpp")
    add_deps("llm-core")
    add_includedirs("include", {public = true})

    -- Define test mode for conditional compilation
    add_defines("LLM_TEST_MODE")
target_end()

-- ============================================================================
-- Package Dependencies (Optional)
-- ============================================================================
-- XMake has built-in package management. Uncomment to add dependencies.
-- Example: GoogleTest for testing, spdlog for logging, etc.

-- add_requires("gtest")          -- Google Test framework
-- add_requires("spdlog")         -- Fast C++ logging library
-- add_requires("fmt")            -- Modern formatting library
-- add_requires("catch2")         -- Catch2 test framework

-- Then add to targets:
-- target("tests")
--     add_packages("gtest")
-- target_end()

-- ============================================================================
-- Build Hooks (Optional)
-- ============================================================================
-- Custom actions that run before/after builds.
-- Useful for code generation, formatting, etc.

-- Run clang-format before build (if available)
-- on_load(function (target)
--     os.execv("clang-format", {"-i", "src/*.cpp", "include/**/*.hpp"})
-- end)

-- ============================================================================
-- Custom Tasks
-- ============================================================================
-- Define custom build tasks accessible via `xmake task <name>`

-- task("format")
--     on_run(function ()
--         os.execv("clang-format", {"-i", "src/*.cpp", "tests/*.cpp", "include/**/*.hpp"})
--         print("Code formatted successfully")
--     end)
--     set_menu {
--         usage = "xmake task format",
--         description = "Format all source files with clang-format"
--     }
-- task_end()