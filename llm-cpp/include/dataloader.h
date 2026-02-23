/**
 * @file dataloader.h
 * @brief C++ implementation of GPTDataset and create_dataloader from ch02
 *
 * This is a direct port of the Python implementation from the book
 * "Build a Large Language Model (From Scratch)" Chapter 2.
 *
 * Dependencies:
 * - libtorch: PyTorch C++ API for tensors and data loading
 * - cpp-tiktoken: BPE tokenizer compatible with GPT models
 *
 * @author Ported from Sebastian Raschka's Python implementation
 */

#pragma once

#include "dataset.h"

#include <torch/torch.h>
#include <tiktoken/encoding.h>
#include <string>
#include <vector>
#include <unordered_set>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <memory>

namespace llm {

/**
 * @brief Configuration options for the DataLoader
 */
struct DataLoaderConfig {
    int64_t batch_size = 4;       ///< Number of samples per batch
    int64_t max_length = 256;     ///< Context length (sequence length)
    int64_t stride = 128;         ///< Sliding window step (overlap = max_length - stride)
    bool shuffle = true;          ///< Whether to shuffle the data
    bool drop_last = true;        ///< Drop the last incomplete batch
    int64_t num_workers = 0;      ///< Number of worker threads (0 = main thread)
    LanguageModel language_model = LanguageModel::R50K_BASE;
};


/**
 * @brief Create a DataLoader for GPT training data
 *
 * This function creates a complete data pipeline:
 * 1. Initializes a GPT-2 tokenizer using tiktoken
 * 2. Creates a GPTDataset with sliding window sequences
 * 3. Wraps it in a PyTorch DataLoader for batching
 *
 * @param txt The raw text to process
 * @param config DataLoader configuration options
 * @return A unique_ptr to a DataLoader
 *
 * @note In C++, we return a unique_ptr because torch::data::DataLoader
 *       is not copyable and has a complex template type.
 *
 * Example usage:
 * @code
 * auto dataloader = create_dataloader(text, {
 *     .batch_size = 8,
 *     .max_length = 256,
 *     .stride = 256,  // No overlap between batches
 *     .shuffle = true
 * });
 *
 * for (auto& batch : *dataloader) {
 *     auto inputs = batch.data;
 *     auto targets = batch.target;
 *     // ... training loop
 * }
 * @endcode
 */
inline std::unique_ptr<
    torch::data::StatelessDataLoader<
        torch::data::datasets::MapDataset<
            GPTDataset,
            torch::data::transforms::Stack<torch::data::Example<>>>,
        torch::data::samplers::SequentialSampler
    >
>
create_dataloader(const std::string& txt, const DataLoaderConfig& config = {}) {
    // Create the dataset
    auto dataset = GPTDataset(txt, config.max_length, config.stride, config.language_model)
        .map(torch::data::transforms::Stack<>());

    // Create the dataloader
    auto sampler = torch::data::samplers::SequentialSampler(dataset.size().value());
    return torch::data::make_data_loader(
        std::move(dataset),
        std::move(sampler),
        torch::data::DataLoaderOptions()
            .batch_size(config.batch_size)
            .workers(config.num_workers)
            .enforce_ordering(!config.shuffle)
            .drop_last(config.drop_last)
    );
}


/**
 * @brief Helper function to read a text file
 *
 * @param filepath Path to the text file
 * @return The file contents as a string
 * @throws std::runtime_error if the file cannot be opened
 */
inline std::string read_text_file(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filepath);
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}


/**
 * @brief Create a DataLoader from a text file
 *
 * Convenience function that reads a file and creates the dataloader.
 *
 * @param filepath Path to the text file
 * @param config DataLoader configuration options
 * @return A unique_ptr to a DataLoader
 */
inline auto create_dataloader_from_file(
    const std::string& filepath,
    const DataLoaderConfig& config = {}
) {
    std::string text = read_text_file(filepath);
    return create_dataloader(text, config);
}

} // namespace llm