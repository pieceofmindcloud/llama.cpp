#include "ggml.h"
#include "llama.h"

#include "../src/llama-batch.h"
#include "../src/llama-vocab.h"

#include <cassert>
#include <cstddef>
#include <cstdio>

struct ggml_context;
struct ggml_tensor;

// The common MoE graph builder must normalize route tensors before reshaping
// them. Keep this test coupled to that internal helper so a future change
// cannot replace the regression with a test-only copy of the logic.
ggml_tensor * llama_moe_reshape_route_2d(
        ggml_context * ctx,
        ggml_tensor * tensor,
        int64_t       ne0,
        int64_t       ne1);

bool llama_moe_route_trace_layer_name(const char * name, int layer);

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

        llama_vocab vocab;
        llama_batch_allocr allocator(1);
        if (!allocator.init(batch, vocab, nullptr, input_width, 1, false)) {
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

static bool test_embedding_width_mismatch_is_rejected() {
    constexpr int32_t n_tokens = 11;
    constexpr int32_t n_embd = 4096;
    constexpr int32_t flow_count = 4;
    const int32_t input_width = n_embd * flow_count;

    llama_batch batch = llama_batch_init(n_tokens, input_width, 1);
    if (batch.embd == nullptr) {
        return false;
    }
    batch.n_tokens = n_tokens;

    llama_vocab vocab;
    llama_batch_allocr allocator(1);
    const bool accepted = allocator.init(batch, vocab, nullptr, n_embd, 1, false);
    llama_batch_free(batch);
    return !accepted;
}

int main() {
    if (!test_embedding_width_reaches_allocator() ||
            !test_embedding_width_mismatch_is_rejected()) {
        return 1;
    }

    assert(llama_moe_route_trace_layer_name("ffn_moe_topk-0", 0));
    assert(llama_moe_route_trace_layer_name("ffn_moe_weights-0 (reshaped)", 0));
    assert(llama_moe_route_trace_layer_name("ffn_moe_topk-1 (cont) (reshaped)", 1));
    assert(!llama_moe_route_trace_layer_name("ffn_moe_topk-1", 0));
    assert(!llama_moe_route_trace_layer_name("unrelated-0", 0));
    assert(!llama_moe_route_trace_layer_name(nullptr, 0));

    ggml_init_params params = {
        /*.mem_size   =*/ 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        return 1;
    }

    ggml_tensor * base = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 8, 4);
    ggml_tensor * reshaped_contiguous = llama_moe_reshape_route_2d(ctx, base, 4, 8);
    if (!ggml_is_contiguous(base) || !ggml_is_contiguous(reshaped_contiguous)) {
        std::fputs("route helper returned a non-contiguous tensor\n", stderr);
        return 1;
    }
    // Route tensors cross from the backend that computes top-k/gather to the
    // CPU custom op. Even a tensor whose strides look contiguous must have a
    // concrete producer so that scheduler copies cannot observe an
    // uninitialized allocation on the first route element.
    if (reshaped_contiguous->src[0] == base ||
        reshaped_contiguous->src[0]->op != GGML_OP_CONT ||
        reshaped_contiguous->src[0]->src[0] != base) {
        std::fputs("route helper did not materialize the contiguous route\n", stderr);
        return 1;
    }

    // A view with the source row stride is valid but not contiguous after
    // narrowing the first dimension. This is the layout produced by graph
    // route operations that triggered ggml_reshape_2d's assertion.
    ggml_tensor * non_contiguous = ggml_view_2d(
        ctx,
        base,
        4,
        4,
        base->nb[1],
        0);
    if (ggml_is_contiguous(non_contiguous)) {
        std::fputs("test view unexpectedly contiguous\n", stderr);
        return 1;
    }

    ggml_tensor * reshaped_non_contiguous =
        llama_moe_reshape_route_2d(ctx, non_contiguous, 4, 4);
    if (!ggml_is_contiguous(reshaped_non_contiguous) ||
        reshaped_non_contiguous->ne[0] != 4 ||
        reshaped_non_contiguous->ne[1] != 4 ||
        reshaped_non_contiguous->src[0] == non_contiguous ||
        !ggml_is_contiguous(reshaped_non_contiguous->src[0])) {
        std::fputs("route helper failed to materialize the strided route\n", stderr);
        return 1;
    }

    ggml_free(ctx);
    return 0;
}
