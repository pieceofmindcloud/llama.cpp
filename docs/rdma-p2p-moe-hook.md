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

The route tensors are normalized by `llama_moe_reshape_route_2d` before the
callback node is built. Top-k and gather operations may return a valid
non-contiguous view, while `ggml_reshape_2d` requires a contiguous source. The
helper therefore calls `ggml_cont` for every route, including a tensor whose
strides look contiguous. The callback is a CPU custom op and needs a concrete
producer so the scheduler can materialize the route before the backend
boundary; logical contiguity alone is not that guarantee. The test covers both
layouts and asserts that both produce a `GGML_OP_CONT` node.

This materialization adds one concrete route operation for the selected expert
IDs and one for their weights per externally executed layer. The current test
and local CPU build validate the graph contract, but they do not measure a
representative CUDA inference. The cost must therefore be measured on the
deployment hardware before tuning this boundary.

Gemma 4 remains an open validation question. Its router-logit construction is
different, but its external route still reaches this common materialization and
CPU callback path. No Gemma diagnostic execution has established that it
escapes the failure or that it is safe when the external executor is enabled.

## F5-E2: metadata-only routed expert loading

`llama_model_params::moe_external_executor` is the second, independent opt-in.
When it is false, the loader and model buffers are unchanged. When it is true,
the optional `moe_external_executor_layers` byte list selects ownership per
layer: byte `1` makes that layer metadata-only, while byte `0` leaves its
experts fully native. An empty list preserves the legacy global opt-in and
selects every layer. The common loader classifies routed expert tensors by the generic
`llm_tensor` semantic (`FFN_*_EXP`, `FFN_*_EXPS`, and the grouped expert
variants), regardless of architecture or GGML dtype. Those tensors are created
in a `no_alloc` metadata context only for externally owned layers and retain
their GGUF name, rank, shape and dtype, but they are not put in a backend buffer
and `load_all_data` never reads their payload into llama.cpp. Native layers keep
their upstream allocation and execution. The ownership list is copied into
`llama_model` so the caller may release its parameter storage after loading.

The routed expert bytes remain the responsibility of the Rust `MoeRuntime` and
its local cache. Shared-expert tensors (`FFN_*_SHEXP`) are intentionally not
classified as routed tensors: the common graph still executes those native
shared experts, and the current callback descriptor does not replace them.
This boundary is explicit rather than silently dropping a computation. A
future hook that takes ownership of shared experts must add their metadata and
execution contract before extending the classifier.

The external metadata context is not part of `llama_model`'s backend-buffer
map, so model memory breakdowns cannot report it as CUDA, CUDA_Host, or CPU
weight residency. The loader also excludes those names from its load-progress
byte total, so the final progress callback still reaches 100% even though
their GGUF records are intentionally not read. The opt-in is propagated from `EngineConfig` and is part of
the Rust `ModelCache` identity; a dense/native load and a metadata-only load of
the same GGUF can never share a cached `llama_model`. At runtime, an external
model without an installed callback is rejected at the decode boundary instead
of falling back to native experts.

The graph applies the same per-layer decision: an externally owned layer is
replaced by the callback, and a native layer follows the upstream expert graph.
If a model has no device bridge for its dtype, the Rust side passes zero for
every layer and records the concrete dtype as the reason; this is an explicit
native decision, not a silent fallback or a partially unloaded model.

The extension is intentionally kept at the common graph boundary so future
upstream changes can be reviewed against one point. Changes in model-specific
tensor conventions belong in the descriptor and in the external executor,
not in `arch == ...` branches in this fork.

## Route-copy trace

The optional boundary trace is controlled only by `POM_MOE_ROUTE_COPY_TRACE`.
When the variable is absent, the trace is disabled and adds no synchronization,
tensor reads, or log output. The legacy value `1` (and the boolean values
`true`, `yes`, and `on`) selects layer 0. A decimal value selects that layer,
with `0` selecting layer 0; because `1` is reserved for the legacy meaning,
layer 1 is selected explicitly with `layer=1` (or `layer:1`). Invalid values
disable the trace.

For the selected layer, the scheduler logs at most four elements from the
source and destination of one selected-expert copy and one weight copy. The
callback logs at most four elements received in each route source, once. The
trace synchronizes only while enabled and reads data without changing it; it
does not alter numerical execution. This bounded sampling is intended for a
single deployment diagnosis, not normal operation.
