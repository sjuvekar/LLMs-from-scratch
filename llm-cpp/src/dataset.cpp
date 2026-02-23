/**
* @file dataloader.cpp
 * @brief Implementation of GPT dataset
 *
 */

#include "dataset.h"

namespace llm {

GPTDataset::GPTDataset(
   const std::string& txt,
   int64_t window_length,
   int64_t stride,
   LanguageModel language_model,
   const std::unordered_set<std::string>& allowed_special
)
    : window_length_(window_length)
    , stride_(stride)
    , gpt_encoding_(GptEncoding::get_encoding(language_model))
    , token_ids_(gpt_encoding_->encode(txt, allowed_special))
{
    build_dataset();
}

void GPTDataset::build_dataset() {
    input_ids_.clear();
    target_ids_.clear();

    for (int64_t i = 0; i < static_cast<int64_t>(token_ids_.size()) - window_length_; i += stride_) {
        // Extract input chunk: tokens[i : i + max_length]
        std::vector<int64_t> input_chunk(
            token_ids_.begin() + i,
            token_ids_.begin() + i + window_length_
        );

        // Extract target chunk: tokens[i+1 : i + max_length + 1]
        // Note: target is shifted by 1 position for next-token prediction
        std::vector<int64_t> target_chunk(
            token_ids_.begin() + i + 1,
            token_ids_.begin() + i + window_length_ + 1
        );

        // Convert to tensors
        // Maps to Python: torch.tensor(input_chunk)
        input_ids_.push_back(torch::tensor(input_chunk, torch::kInt64));
        target_ids_.push_back(torch::tensor(target_chunk, torch::kInt64));
    }
}

}