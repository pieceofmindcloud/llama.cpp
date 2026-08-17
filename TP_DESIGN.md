# Tensor parallelism without weight transfer (pom path C)

## Goal

Split a model across N machines by tensor parallelism where each rank loads its
own tensor shard from a local GGUF copy. No weight travels over the network;
only activations do, via an all-reduce after each row-parallel projection.

## Why not ggml-rpc

ggml-rpc + split_mode=ROW already do tensor parallelism, but the primary ships
weight shards to the workers at load time. Path C keeps the "no weight on the
wire" invariant of the pom project: each worker slices its own copy locally.

## Changes

1. A reduce op in ggml that sums a tensor across the group during graph compute,
   driven by a callback registered by the caller (the network all-reduce lives
   outside llama.cpp). Identity when no callback is set, so a plain build is a
   no-op.
2. Insert that op in llama-graph.cpp after the row-parallel GEMMs, guarded by a
   tp_size > 1 flag:
   - attention output projection (wo) - the `build_attn_mha` callers in
     `build_attn` (around line 2517+);
   - MLP down projection - the `down` GEMM in `build_ffn` (line 1669+).
3. No loader changes needed: tensor dims come from the GGUF, so a row-split
   tensor (ne[0] / tp_size) already produces a partial GEMM result that the
   reduce sums.

## Shard layout (produced by the pom splitter, not this repo)

- column-parallel (split ne[1], no reduce): attn_q/k/v, ffn_gate/up
- row-parallel (split ne[0], needs reduce): attn_output, ffn_down, output
- replicated: norms, token_embd

## Integration

rdma-p2p builds this fork through a vendored llama-cpp-sys-2 (patch.crates-io)
whose `llama.cpp` is a symlink to this checkout. The collective
(`ReduceBarrier`) lives in the pom node at
`crates/node/src/inference/collective.rs`; the ggml op will call back into it
using the same C-shim pattern as `llama_mtp_shim.cpp`
(`crates/node/build.rs`).
