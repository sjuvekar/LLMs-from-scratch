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
--
-- Dependencies:
-- - libtorch: PyTorch C++ API for tensor operations
-- - cpp-tiktoken: BPE tokenizer for GPT-2/GPT-4 (manual install required)
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
add_rules("mode.debug", "mode.release", "mode.releasedbg")

-- ============================================================================
-- Compiler-Specific Warning Flags
-- ============================================================================
-- High warning levels catch bugs early and enforce code quality.

-- Clang/GCC warnings (Linux, macOS)
if is_plat("linux", "macosx") then
    add_cxxflags("-Wall", "-Wextra", "-Wpedantic", {public = true})
    add_cxxflags("-Wshadow", "-Wconversion", "-Wold-style-cast", "-Wnull-dereference", {public = true})

    -- Release-specific: Enable link-time optimization (LTO)
    if is_mode("release") then
        add_cxxflags("-flto", {public = true})
        add_ldflags("-flto", {public = true})
    end
end

-- MSVC warnings (Windows)
if is_plat("windows") then
    add_cxxflags("/W4", "/WX", "/permissive-", "/Zc:__cplusplus", {public = true})
    add_cxxflags("/Zc:strictStrings", "/Zc:rvalueCast", "/Zc:throwingNew", {public = true})
end

-- ============================================================================
-- Package Dependencies
-- ============================================================================
-- XMake has built-in package management for easy dependency handling.

-- LibTorch: PyTorch C++ API for tensor operations and neural network building
-- Provides: torch::Tensor, torch::nn modules, autograd, CUDA support
-- Docs: https://pytorch.org/cppdocs/
-- Install: xmake requires -y libtorch
add_requires("libtorch", {configs = {cuda = false}})  -- Set cuda=true for GPU support

-- cpp-tiktoken: BPE tokenizer compatible with GPT-2/GPT-4
-- Provides: GptEncoding, encode(), decode()
-- Repo: https://github.com/gh-markt/cpp-tiktoken
--
-- Manual installation required (not in xmake repo yet):
-- 1. Clone: git clone https://github.com/gh-markt/cpp-tiktoken thirdparty/cpp-tiktoken
-- 2. Build: cd thirdparty/cpp-tiktoken && mkdir build && cd build
--           cmake .. && make && make install
-- 3. Uncomment the add_includedirs and add_links below

-- Optional development packages
add_requires("gtest", {optional = true})  -- Google Test framework
add_requires("spdlog", {optional = true}) -- Fast logging library

-- ============================================================================
-- Main Library Target
-- ============================================================================
-- The library contains the core LLM implementation components:
-- - dataloader.h/cpp: GPT dataset and data loading (from ch02)
-- - Tokenization support via cpp-tiktoken
-- - Tensor operations via libtorch
target("llm-core")
    set_kind("static")

    -- Public headers
    add_includedirs("include", {public = true})

    -- Source files
    add_files("src/*.cpp")

    -- Private headers
    add_includedirs("src", {public = false})

    -- Dependencies
    add_packages("libtorch")

    -- Uncomment for cpp-tiktoken (after manual installation):
    add_includedirs("thirdparty", {public = true})
    -- add_links("tiktoken")

    -- Export definitions for shared library builds
    add_defines("LLM_EXPORTS", {public = true})

    -- LibTorch requires exceptions and RTTI
    if is_plat("linux", "macosx") then
        add_cxxflags("-fexceptions", "-frtti", {public = true})
    end
target_end()

-- ============================================================================
-- Main Executable Target
-- ============================================================================
target("llm-cli")
    set_kind("binary")
    add_files("src/main.cpp")
    add_deps("llm-core")
    add_packages("libtorch")

    -- Static CRT for MSVC
    if is_plat("windows") then
        add_cxxflags("/MT$<s:d>", {public = true})
    end
target_end()

-- ============================================================================
-- Test Target
-- ============================================================================
target("tests")
    set_kind("binary")
    set_default(false)  -- Build explicitly: xmake build tests

    add_files("tests/*.cpp")
    add_deps("llm-core")
    add_packages("libtorch")
    add_includedirs("include", {public = true})

    -- Use GoogleTest if available
    if has_package("gtest") then
        add_packages("gtest")
    end

    add_defines("LLM_TEST_MODE")
target_end()
