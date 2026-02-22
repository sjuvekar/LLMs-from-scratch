/**
 * @file dataloader.cpp
 * @brief Implementation of GPT dataset and data loader
 *
 * This implements the core data loading functionality for GPT training,
 * ported from Python's GPTDatasetV1 and create_dataloader_v1.
 */

#include "dataloader.h"
#include "tiktoken/encoding.h"

#include <algorithm>
#include <iostream>

namespace {
    using ::tiktoken::GptEncoding;
}

namespace llm {

// ============================================================================
// GPTDataset Implementation
// ============================================================================

GPTDataset::GPTDataset(
    const std::string& txt,
    std::shared_ptr<GptEncoder> gpt_encoder,
    int64_t max_length,
    int64_t stride,
    const std::vector<std::string>& allowed_special
)
    : gpt_encoder_(std::move(gpt_encoder))
    , max_length_(max_length)
    , stride_(stride)
{
    if (!gpt_encoder_) {
        // Default to GPT-2 tokenizer
        gpt_encoder_ = GptEncoding::get_encoding(LanguageModel::R50K_BASE);
    }

    // Tokenize the entire text
    auto token_ids = gpt_encoder_->encode(txt, allowed_special);

    // Build the dataset
    build_dataset(token_ids);
}

GPTDataset::GPTDataset(
    const std::vector<int64_t>& token_ids,
    int64_t max_length,
    int64_t stride
)
    : gpt_encoder_(nullptr)  // No tokenizer when constructed from tokens
    , max_length_(max_length)
    , stride_(stride)
{
    build_dataset(token_ids);
}

void GPTDataset::build_dataset(const std::vector<int64_t>& token_ids) {
    // Verify we have enough tokens
    // Maps to Python: assert len(token_ids) > max_length
    if (static_cast<int64_t>(token_ids.size()) <= max_length_) {
        throw InsufficientTextError(
            "Number of tokenized inputs (" + std::to_string(token_ids.size()) +
            ") must at least be equal to max_length + 1 (" +
            std::to_string(max_length_ + 1) + ")"
        );
    }

    input_ids_.clear();
    target_ids_.clear();

    for (int64_t i = 0; i < static_cast<int64_t>(token_ids.size()) - max_length_; i += stride_) {
        // Extract input chunk: tokens[i : i + max_length]
        std::vector<int64_t> input_chunk(
            token_ids.begin() + i,
            token_ids.begin() + i + max_length_
        );

        // Extract target chunk: tokens[i+1 : i + max_length + 1]
        // Note: target is shifted by 1 position for next-token prediction
        std::vector<int64_t> target_chunk(
            token_ids.begin() + i + 1,
            token_ids.begin() + i + max_length_ + 1
        );

        // Convert to tensors
        // Maps to Python: torch.tensor(input_chunk)
        input_ids_.push_back(torch::tensor(input_chunk, torch::kInt64));
        target_ids_.push_back(torch::tensor(target_chunk, torch::kInt64));
    }
}

std::pair<torch::Tensor, torch::Tensor> GPTDataset::get(int64_t index) const {
    if (index < 0 || index >= size()) {
        throw std::out_of_range(
            "Index " + std::to_string(index) + " out of range for dataset size " +
            std::to_string(size())
        );
    }
    return {input_ids_[index], target_ids_[index]};
}

torch::Tensor GPTDataset::all_inputs() const {
    if (input_ids_.empty()) {
        return torch::Tensor();
    }
    return torch::stack(input_ids_);
}

torch::Tensor GPTDataset::all_targets() const {
    if (target_ids_.empty()) {
        return torch::Tensor();
    }
    return torch::stack(target_ids_);
}

// ============================================================================
// Custom Dataset Loader Adapter
// ============================================================================

namespace detail {

/**
 * @brief Custom DataLoader that wraps GPTDataset
 *
 * libtorch's DataLoader requires a specific interface. This adapter
 * provides iteration over batches.
 */
class GPTDataLoader {
public:
    GPTDataLoader(
        std::shared_ptr<GPTDataset> dataset,
        int64_t batch_size,
        bool shuffle,
        bool drop_last
    )
        : dataset_(std::move(dataset))
        , batch_size_(batch_size)
        , shuffle_(shuffle)
        , drop_last_(drop_last)
        , current_index_(0)
    {
        // Initialize indices
        indices_.resize(dataset_->size());
        std::iota(indices_.begin(), indices_.end(), 0);

        if (shuffle_) {
            shuffle_indices();
        }
    }

    /// Reset for new epoch
    void reset() {
        current_index_ = 0;
        if (shuffle_) {
            shuffle_indices();
        }
    }

    /// Check if there are more batches
    [[nodiscard]] bool has_next() const {
        auto remaining = static_cast<int64_t>(indices_.size()) - current_index_;
        if (drop_last_) {
            return remaining >= batch_size_;
        }
        return remaining > 0;
    }

    /// Get next batch
    std::pair<torch::Tensor, torch::Tensor> next_batch() {
        auto remaining = static_cast<int64_t>(indices_.size()) - current_index_;
        int64_t actual_batch_size = drop_last_ ? batch_size_ : std::min(batch_size_, remaining);

        std::vector<torch::Tensor> inputs;
        std::vector<torch::Tensor> targets;

        for (int64_t i = 0; i < actual_batch_size && current_index_ < static_cast<int64_t>(indices_.size()); ++i, ++current_index_) {
            auto [input_tensor, target_tensor] = dataset_->get(indices_[current_index_]);
            inputs.push_back(input_tensor);
            targets.push_back(target_tensor);
        }

        // Stack into batched tensors: [batch_size, max_length]
        return {
            torch::stack(inputs, 0),   // Shape: [batch_size, max_length]
            torch::stack(targets, 0)   // Shape: [batch_size, max_length]
        };
    }

    /// Get total number of batches
    [[nodiscard]] int64_t num_batches() const {
        auto total = static_cast<int64_t>(indices_.size());
        if (drop_last_) {
            return total / batch_size_;
        }
        return (total + batch_size_ - 1) / batch_size_;  // Ceiling division
    }

    /// Get dataset size
    [[nodiscard]] int64_t size() const {
        return static_cast<int64_t>(indices_.size());
    }

private:
    void shuffle_indices() {
        torch::Tensor indices_tensor = torch::randperm(
            static_cast<int64_t>(indices_.size()),
            torch::TensorOptions().dtype(torch::kInt64)
        );
        for (int64_t i = 0; i < static_cast<int64_t>(indices_.size()); ++i) {
            indices_[i] = indices_tensor[i].item<int64_t>();
        }
    }

    std::shared_ptr<GPTDataset> dataset_;
    int64_t batch_size_;
    bool shuffle_;
    bool drop_last_;
    std::vector<int64_t> indices_;
    int64_t current_index_;
};

} // namespace detail

// ============================================================================
// Factory Functions
// ============================================================================

std::shared_ptr<torch::DataLoader<GPTDataset>> create_dataloader(
    const std::string& txt,
    const DataLoaderConfig& config,
    std::shared_ptr<Tokenizer> tokenizer
) {
    // Create tokenizer if not provided
    if (!tokenizer) {
        tokenizer = TikTokenTokenizer::gpt2();
    }

    // Create dataset
    auto dataset = std::make_shared<GPTDataset>(
        txt,
        tokenizer,
        config.max_length,
        config.stride
    );

    // Create dataloader using libtorch's built-in DataLoader
    // Note: libtorch DataLoader syntax differs slightly from PyTorch Python
    return std::make_shared<torch::DataLoader<GPTDataset>>(
        *dataset,
        torch::DataLoaderOptions()
            .batch_size(config.batch_size)
            .shuffle(config.shuffle)
            .drop_last(config.drop_last)
    );
}

std::shared_ptr<torch::DataLoader<GPTDataset>> create_dataloader(
    const std::vector<int64_t>& token_ids,
    const DataLoaderConfig& config
) {
    // Create dataset from pre-tokenized data
    auto dataset = std::make_shared<GPTDataset>(
        token_ids,
        config.max_length,
        config.stride
    );

    return std::make_shared<torch::DataLoader<GPTDataset>>(
        *dataset,
        torch::DataLoaderOptions()
            .batch_size(config.batch_size)
            .shuffle(config.shuffle)
            .drop_last(config.drop_last)
    );
}

// ============================================================================
// Utility Functions
// ============================================================================

std::string load_text_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);

    if (!file) {
        throw std::runtime_error("Failed to open file: " + path);
    }

    auto size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::string content;
    content.resize(size);
    file.read(content.data(), size);

    if (!file) {
        throw std::runtime_error("Failed to read file: " + path);
    }

    return content;
}

int64_t tokenize_and_save(
    const std::string& input_path,
    const std::string& output_path,
    std::shared_ptr<Tokenizer> tokenizer
) {
    // Load text
    auto text = load_text_file(input_path);

    // Tokenize
    auto tokens = tokenizer->encode(text);

    // Save to binary file
    std::ofstream out(output_path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Failed to create output file: " + output_path);
    }

    // Write size first
    int64_t size = static_cast<int64_t>(tokens.size());
    out.write(reinterpret_cast<const char*>(&size), sizeof(size));

    // Write tokens
    out.write(reinterpret_cast<const char*>(tokens.data()), tokens.size() * sizeof(int64_t));

    return size;
}

std::vector<int64_t> load_tokens(const std::string& path) {
    std::ifstream file(path, std::ios::binary);

    if (!file) {
        throw std::runtime_error("Failed to open token file: " + path);
    }

    // Read size
    int64_t size;
    file.read(reinterpret_cast<char*>(&size), sizeof(size));

    if (!file) {
        throw std::runtime_error("Failed to read token file size: " + path);
    }

    // Read tokens
    std::vector<int64_t> tokens(size);
    file.read(reinterpret_cast<char*>(tokens.data()), size * sizeof(int64_t));

    if (!file) {
        throw std::runtime_error("Failed to read tokens from: " + path);
    }

    return tokens;
}

} // namespace llm