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
 * @class GPTDataset
 * @brief Dataset class that creates overlapping input/target sequences from text
 *
 * This dataset tokenizes the entire text using a BPE tokenizer, then creates
 * overlapping chunks using a sliding window approach. Each sample consists of:
 * - input_ids: A sequence of token IDs
 * - target_ids: The same sequence shifted by one position (next-token prediction)
 *
 * Example with max_length=4, stride=1:
 *   Text: "The quick brown fox"
 *   Token IDs: [1, 2, 3, 4, 5]
 *
 *   Sample 0: input=[1, 2, 3, 4], target=[2, 3, 4, 5]
 *   Sample 1: input=[2, 3, 4, 5], target=[3, 4, 5, 6]
 *   ...
 */
class GPTDataset : public torch::data::Dataset<GPTDataset> {
public:
    /**
     * @brief Construct a GPTDataset from text
     *
     * @param txt The raw text to tokenize and create sequences from
     * @param max_length The length of each sequence (context length)
     * @param stride The step size for the sliding window (overlap = max_length - stride)
     * @param gpt_encoding Shared pointer to a tiktoken encoder (GPT-2 encoding)
     * @param allowed_special allowed special characters during encoding
     *
     * @throws std::runtime_error if text is too short for max_length
     */
    GPTDataset(
        const std::string& txt,
        int64_t max_length,
        int64_t stride,
        std::shared_ptr<GptEncoding> gpt_encoding,
        const std::unordered_set<std::string>& allowed_special = {}
    );

    /**
     * @brief Construct a GPTDataset from token ids
     *
     * @param token_ids Integer vector of token ids
     * @param max_length The length of each sequence (context length)
     * @param stride The step size for the sliding window (overlap = max_length - stride)
     *
     * @throws std::runtime_error if text is too short for max_length
     */
    GPTDataset(
        const std::vector<int>& token_ids,
        int64_t max_length,
        int64_t stride
    );

    /**
     * @brief Get a single sample from the dataset
     *
     * @param index The sample index
     * @return A pair of (input_ids, target_ids) tensors
     */
    torch::data::Example<> get(size_t index) override {
        return {input_ids_[index], target_ids_[index]};
    }

    /**
     * @brief Get the number of samples in the dataset
     */
    torch::optional<size_t> size() const override {
        return input_ids_.size();
    }

    /**
     * @brief Get the raw token IDs (for debugging/inspection)
     */
    const std::vector<int>& get_token_ids() const {
        return token_ids_;
    }

private:
    void build_dataset(const std::vector<int>& token_ids);

    std::vector<int> token_ids_;           ///< All token IDs from the text
    std::vector<torch::Tensor> input_ids_; ///< Input sequences
    std::vector<torch::Tensor> target_ids_;///< Target sequences (shifted by 1)
    int64_t max_length_;                   ///< Context length
    int64_t stride_;                       ///< Sliding window step size
    std::shared_ptr<GptEncoding> gpt_encoding_;
};


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
        GPTDataset, torch::data::samplers::SequentialSampler
    >
>
create_dataloader(const std::string& txt, const DataLoaderConfig& config = {}) {
    // Initialize the tokenizer with GPT-2 encoding
    // This is equivalent to: tiktoken.get_encoding("gpt2")
    auto tokenizer = GptEncoding::get_encoding(config.language_model);

    // Create the dataset
    auto dataset = GPTDataset(txt, config.max_length, config.stride, tokenizer)
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