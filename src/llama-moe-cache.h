#pragma once

#include "ggml-backend.h"

#include <map>
#include <memory>

struct llama_model;

#include <vector>

// keeps the most recently used experts of host-resident MoE layers in a device buffer
// MUL_MAT_ID ops on these experts run on the device and only the cache misses are uploaded
// with several GPUs, each one gets its own cache of `size` bytes for the experts of the
// layers assigned to it, so every card keeps its share of the model in VRAM
class llama_moe_cache {
public:
    llama_moe_cache(const llama_model & model, const std::vector<ggml_backend_t> & backends,
            const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size);
    ~llama_moe_cache();

    ggml_backend_t backend() const;

    // false when no GPU holds a layer that keeps all of its experts in host memory
    // (the fit placed every expert layer in VRAM): there is nothing to cache
    bool active() const;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const;

    // ggml_backend_sched callbacks, user_data is the llama_moe_cache
    static bool sched_resolve(void * user_data, const ggml_tensor * node, ggml_backend_t backend, ggml_tensor ** cached_weight, void ** cache_entry);
    static void sched_begin(void * user_data);
    static bool sched_prepare(void * user_data, void * cache_entry, const int32_t * ids, size_t n_ids, const int32_t ** remapped_ids);

private:
    struct impl;
    std::vector<std::unique_ptr<impl>> caches; // one per GPU
};
