#include "dataset.h"

#include <tiktoken/encoding.h>
#include <iostream>

int main() {
    auto gpt_dataset = llm::GPTDataset("hello world. This is a test of long sentence. A mouse took a stroll through a deep dark wood.",
        /*window_length=*/3,
        /*strid*/2,
        /*language_model=*/LanguageModel::R50K_BASE
    );
    std::cout << *gpt_dataset.size() << std::endl;
}