#include "llama.h"

#include "../src/llama-batch.h"
#include "../src/llama-vocab.h"

#include <cstddef>
#include <cstdint>

// an embd batch wider than n_embd (expanded hidden state, e.g. hc streams at a
// pipeline boundary) must reach the ubatch at its declared row width

static bool test_embedding_width_reaches_allocator() {
    constexpr int32_t n_tokens = 11;
    constexpr int32_t n_embd = 4096;
    constexpr int32_t flow_count = 4;

    for (const int32_t input_width : { n_embd, n_embd * flow_count }) {
        llama_batch batch = llama_batch_init(n_tokens, input_width, 1);
        if (batch.embd == nullptr || batch.n_embd != input_width) {
            return false;
        }
        batch.n_tokens = n_tokens;

        for (int32_t i = 0; i < n_tokens; ++i) {
            batch.pos[i] = i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = 1;
            batch.embd[(size_t) i * input_width] = 100.0f + i;
            batch.embd[(size_t) i * input_width + input_width - 1] = 200.0f + i;
        }

        // the decoder width is n_embd, the row width comes from llama_batch.n_embd
        llama_batch_ext batch_ext(n_tokens, n_embd, n_embd, 1, nullptr, 100, 1);
        llama_batch_compat::init(batch_ext, batch, batch.n_embd);

        llama_vocab vocab;
        llama_batch_allocr allocator(1);
        if (!allocator.init(batch_ext, vocab, false)) {
            llama_batch_free(batch);
            return false;
        }

        allocator.split_reset();
        const llama_ubatch ubatch = allocator.split_simple(n_tokens);
        if (ubatch.n_embd != (uint32_t) input_width ||
                ubatch.embd == nullptr ||
                ubatch.embd[input_width - 1] != 200.0f ||
                ubatch.embd[(size_t) (n_tokens - 1) * input_width] != 100.0f + n_tokens - 1) {
            llama_batch_free(batch);
            return false;
        }

        llama_batch_free(batch);
    }

    return true;
}

int main() {
    return test_embedding_width_reaches_allocator() ? 0 : 1;
}
