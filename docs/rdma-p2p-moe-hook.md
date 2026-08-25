# rdma-p2p generic MoE replacement hook

This fork exposes an opt-in extension point for an external MoE executor.
The extension is in the common `llm_graph_context::build_moe_ffn` overload in
`src/llama-graph.cpp`. Model builders still resolve routing, activation, and
weight normalization there. When a callback is installed, that common point
builds one custom graph node and passes the resolved route plus a descriptor of
the expert tensors to the callback.

The descriptor is data, not an architecture or quantization branch. It carries
the architecture name, layer, activation, gating mode, expert counts, fused or
split gate/up tensors, down tensor, tensor names, GGML dtypes, ranks, and
dimensions. The callback therefore chooses its decoder from the tensor data.
The hook does not assume Gemma4, NVFP4, a fixed expert count, or one dtype for
the whole model.

The hook replaces the native expert matmuls only after llama.cpp has computed
the route. The callback output replaces the complete expert FFN result. If the
callback rejects the descriptor or its shape, the bridge records an error and
the graph node is zeroed; it does not fall back to native experts. This keeps a
failed external executor visible instead of performing the block twice or
silently changing engines.

The callback is opt-in per context. With no callback, `build_moe_ffn` follows
the upstream native path exactly. Dense models never install the callback and
therefore retain the native dense path. Attention, normalization, embeddings,
KV cache, tokenization, and sampling remain in llama.cpp in both modes.

The extension is intentionally kept at the common graph boundary so future
upstream changes can be reviewed against one point. Changes in model-specific
tensor conventions belong in the descriptor and in the external executor,
not in `arch == ...` branches in this fork.
