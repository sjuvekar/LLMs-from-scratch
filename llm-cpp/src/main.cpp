#include "dataset.h"

#include <tiktoken/encoding.h>
#include <iostream>

int main() {
    auto gpt_dataset = llm::GPTDataset("hello world",
        /*window_length=*/4,
        /*strid*/1,
        /*language_model=*/LanguageModel::R50K_BASE
    );
}