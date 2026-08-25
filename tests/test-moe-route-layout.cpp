#include "ggml.h"

#include <cassert>
#include <cstddef>

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

int main() {
    ggml_init_params params = {
        /*.mem_size   =*/ 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    assert(ctx != nullptr);

    ggml_tensor * base = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 8, 4);
    ggml_tensor * reshaped_contiguous = llama_moe_reshape_route_2d(ctx, base, 4, 8);
    assert(ggml_is_contiguous(base));
    assert(ggml_is_contiguous(reshaped_contiguous));
    assert(reshaped_contiguous->src[0] == base);

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
    assert(!ggml_is_contiguous(non_contiguous));

    ggml_tensor * reshaped_non_contiguous =
        llama_moe_reshape_route_2d(ctx, non_contiguous, 4, 4);
    assert(ggml_is_contiguous(reshaped_non_contiguous));
    assert(reshaped_non_contiguous->ne[0] == 4);
    assert(reshaped_non_contiguous->ne[1] == 4);
    assert(reshaped_non_contiguous->src[0] != non_contiguous);
    assert(ggml_is_contiguous(reshaped_non_contiguous->src[0]));

    ggml_free(ctx);
    return 0;
}
