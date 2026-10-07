#include "arg.h"
#include "common.h"
#include "llama.h"

#include <algorithm>
#include <clocale>
#include <cstdio>

static llama_context * make_context(const common_params & params, llama_model * model) {
    auto cparams = common_context_params_to_llama(params);
    cparams.n_ctx = 750000;
    cparams.n_batch = 4096;
    cparams.n_ubatch = 512;
    // The engine reserves two physical sequences per session.  Six active
    // sessions therefore exercise the non-zero generation sequence ids used
    // by the production gateway instead of only ids 0..5.
    cparams.n_seq_max = 12;
    cparams.n_rs_seq = 4;
    cparams.kv_unified = true;
    return llama_init_from_model(model, cparams);
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.n_predict = 1;

    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    ggml_backend_load_all();
    common_init_result_ptr llama_init = common_init_from_params(params);
    llama_model * model = llama_init->model();
    if (model == nullptr) {
        fprintf(stderr, "%s: failed to init model\n", __func__);
        return 1;
    }
    if (!llama_model_is_hybrid(model)) {
        fprintf(stderr, "%s: skipping for non-hybrid model\n", __func__);
        return 0;
    }

    llama_context * ctx = make_context(params, model);
    if (ctx == nullptr) {
        fprintf(stderr, "%s: failed to init context\n", __func__);
        return 1;
    }

    constexpr int n_seqs = 6;
    constexpr int n_tokens_per_seq = 516;
    constexpr int n_rounds = 4;
    constexpr int n_rollback = 3;
    constexpr int first_seq = 6;
    constexpr llama_pos first_pos = 747500;
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    for (int round = 0; round < n_rounds; ++round) {
        llama_batch batch = llama_batch_init(n_seqs * n_tokens_per_seq, 0, n_seqs);
        for (int seq = 0; seq < n_seqs; ++seq) {
            const int seq_id = first_seq + seq;
            for (int offset = 0; offset < n_tokens_per_seq; ++offset) {
                const int row = seq * n_tokens_per_seq + offset;
                const llama_token token = 1 + (row + round) % std::max(1, n_vocab - 1);
                common_batch_add(batch, token,
                        first_pos + round * n_tokens_per_seq + offset,
                        { seq_id }, offset + 1 == n_tokens_per_seq);
            }
        }

        const int rc = llama_decode(ctx, batch);
        llama_batch_free(batch);
        if (rc != 0) {
            fprintf(stderr, "%s: llama_decode failed at round %d with rc=%d\n", __func__, round, rc);
            llama_free(ctx);
            return 1;
        }

        // Exercise the recurrent snapshot path after a concurrent decode. The
        // production gateway can invalidate the last few tokens independently
        // for every active sequence before replaying them.
        if (round == 0 || round == 3) {
            const llama_pos rollback_pos = first_pos + round * n_tokens_per_seq + n_tokens_per_seq - n_rollback;
            for (int seq = 0; seq < n_seqs; ++seq) {
                const int seq_id = first_seq + seq;
                if (!llama_memory_seq_rm(llama_get_memory(ctx), seq_id, rollback_pos, -1)) {
                    fprintf(stderr, "%s: rollback failed at round %d seq %d\n", __func__, round, seq);
                    llama_free(ctx);
                    return 1;
                }
            }

            llama_batch replay = llama_batch_init(n_seqs * n_rollback, 0, n_seqs);
            for (int seq = 0; seq < n_seqs; ++seq) {
                const int seq_id = first_seq + seq;
                for (int offset = 0; offset < n_rollback; ++offset) {
                    const int row = seq * n_rollback + offset;
                    const llama_token token = 1 + (row + round + 17) % std::max(1, n_vocab - 1);
                    common_batch_add(replay, token, rollback_pos + offset, { seq_id }, offset + 1 == n_rollback);
                }
            }

            const int replay_rc = llama_decode(ctx, replay);
            llama_batch_free(replay);
            if (replay_rc != 0) {
                fprintf(stderr, "%s: replay decode failed at round %d with rc=%d\n", __func__, round, replay_rc);
                llama_free(ctx);
                return 1;
            }
        }
    }

    fprintf(stderr, "%s: hybrid six-sequence batch completed\n", __func__);
    llama_free(ctx);
    return 0;
}
