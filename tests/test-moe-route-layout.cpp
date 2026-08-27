#include "ggml.h"

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

bool llama_moe_route_trace_first_layer_name(const char * name);

int main() {
    assert(llama_moe_route_trace_first_layer_name("ffn_moe_topk-0"));
    assert(llama_moe_route_trace_first_layer_name("ffn_moe_weights-0 (reshaped)"));
    assert(!llama_moe_route_trace_first_layer_name("ffn_moe_topk-1"));
    assert(!llama_moe_route_trace_first_layer_name("unrelated-0"));
    assert(!llama_moe_route_trace_first_layer_name(nullptr));

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
