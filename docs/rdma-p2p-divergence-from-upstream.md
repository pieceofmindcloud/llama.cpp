# pom-tensor-parallel vs upstream master: relatorio de divergencia

Este documento explica o que `pom-tensor-parallel` muda em relacao ao
`llama.cpp` de fabrica (branch `master` deste fork, que acompanha
`ggml-org/llama.cpp`) e por que cada mudanca existe. Gerado apos o merge de
546 commits do upstream (SHA `01818e4` -> `b49650a`) em `pom-tensor-parallel`
via merge commit `231ca1a28a95f5df1ef4feb480cdcdb9e519f00e`.

Escopo: `git diff --stat origin/master...origin/pom-tensor-parallel` no ponto
pos-merge - 42 arquivos, ~1600 linhas. Os 13 commits proprios do POM (fora o
merge) estao listados abaixo em ordem cronologica.

## 1. Tensor parallelism sem transferencia de peso

Commits: `ba39cec84` (design doc `TP_DESIGN.md`), `6e5d86726`, `a2d91edd2`.

Adiciona uma operacao de reduce de tensor-parallelism a `llama-graph.cpp` /
`llama-context.cpp` / `include/llama.h`, aplicada em toda sobrecarga de
`build_attn`. O objetivo e permitir que multiplas GPUs processem a mesma
camada em paralelo (tensor parallelism), sem replicar ou transferir pesos
entre elas - cada GPU mantem sua propria fatia de peso e so o resultado
parcial de ativacao e reduzido. Isto e uma capacidade que o upstream nao tem:
o llama.cpp de fabrica assume pipeline parallelism (uma GPU por faixa de
camadas) ou paralelismo de dados, nao TP dentro da mesma camada sem mover
pesos pela rede.

**Por que manter:** e a base do modo de paralelismo por camada do rdma-p2p
quando duas GPUs da mesma maquina servem a mesma fatia. Sem isto, o rdma-p2p
perderia a capacidade de dividir uma camada entre GPUs locais.

## 2. Hook generico de MoE externo

Commits: `ff59e560a`, `03902a5ba`, `4bb64b070`, `b7b163cdc`, `32d461112`,
`e822a565c`, `16c517a43`.

Documentado em detalhe em `docs/rdma-p2p-moe-hook.md` (mantido pelo proprio
commit `32d461112`). Resumo: expoe um ponto de extensao opt-in em
`llm_graph_context::build_moe_ffn` (`src/llama-graph.cpp`) que deixa um
executor de MoE externo (fora do llama.cpp) substituir o matmul nativo dos
experts depois que a rota ja foi resolvida. O descritor passado ao callback e
generico - carrega arquitetura, camada, contagem de experts, dtypes e formas
- e nao assume Gemma4/NVFP4/dtype fixo. O carregamento por camada
(`moe_external_executor_layers` em `llama_model_params`, `src/llama-model.cpp`
/ `.h`, `src/llama-model-loader.cpp` / `.h`) permite que so algumas camadas
sejam metadata-only (peso do expert nao entra em buffer de backend), mantendo
as demais 100% nativas.

**Por que manter:** e o mecanismo que permite ao rdma-p2p rotear experts de
MoE para um executor fora do processo llama.cpp (parte do trabalho de MoE
proprio do POM/FreeToken). Upstream nao tem ponto de extensao equivalente -
os experts sao sempre resolvidos e executados dentro do proprio llama.cpp.

Testes proprios que cobrem isto: `tests/test-moe-route-layout.cpp`,
`tests/test-batch-alloc.cpp` (novos, adicionados por `tests/CMakeLists.txt`).

## 3. Largura de embedding expandida preservada

Commit: `05efc320b`.

Carrega a largura explicita de embedding pelo batch publico, alocador e input
do grafo, para que estados ocultos expandidos nao sejam reduzidos de volta a
`n_embd_inp`. Usa `n_embd_out` relatado como a geometria expandida autoritativa
e rejeita larguras incompativeis. Toca `include/llama.h`, `src/llama-batch.cpp`
/ `.h`, `src/llama-context.cpp` / `.h`, `src/llama-graph.cpp` / `.h`,
`common/speculative.cpp`.

**Por que manter:** o rdma-p2p troca estado oculto (hidden state) em f16 entre
peers via `/v1/prefill` e `/v1/step` (ver `docs/subsystems/inferencia.md`).
Esse estado oculto pode ter largura diferente de `n_embd` nativo do modelo
quando passa por uma camada de projecao entre peers; sem esta mudanca, o
llama.cpp de fabrica trunca/reduz a largura de volta ao valor nativo do
modelo, corrompendo o wire de estado oculto entre nos.

## 4. Timing do snapshot de fit-memory

Commit: `b60af3833`.

Corrige `common/fit.cpp` para que o snapshot de memoria usado pelo
planejamento de fatiamento (`common_fit_params`) seja tirado no momento certo
do ciclo de vida, evitando medir memoria antes de alocacoes relevantes
acontecerem. Adiciona `tests/test-fit-memory.cpp`.

**Por que manter:** o planejador de fatiamento do rdma-p2p
(`docs/subsystems/fatiamento.md`) depende de medicoes de memoria corretas
para decidir quantas camadas cabem em cada peer. Um snapshot cedo demais
subestima o uso real e gera planos que estouram memoria em producao.

## 5. Rollback recorrente do Kimi-K3

Commit: `d9dd86175` (a mesma correcao documentada em progresso anterior como
"fix do Kimi-K3 recurrent rollback").

Ajusta `src/models/kimi-k3.cpp` e `src/llama-arch.cpp` para que o rollback de
estado recorrente (usado por `/v1/session/rewind`, ver a invariante de
KV-cache em `AGENTS.md`) funcione corretamente para a arquitetura hibrida
Kimi-K3. Upstream nao tem este ajuste porque nao testa rollback de sequencia
completa neste modelo especifico da forma que o rdma-p2p exige (remocao
**total** de sequencia, nunca parcial).

**Por que manter:** sem isto, sessoes Kimi-K3 no rdma-p2p quebrariam o
invariante "a geracao nunca acontece na sequencia de KV do prompt" descrito no
`AGENTS.md` da raiz.

## Arquivos tocados so por limpeza/ajuste local (sem commit dedicado)

Alguns arquivos no diff stat (`docs/backend/snapdragon/*.md`,
`ggml/src/ggml-vulkan/*`, `ggml/src/ggml-metal/kernels/quantize.metal`)
aparecem como remocoes de 1-2 linhas: sao vestigios de resolucao de conflito
do merge de 546 commits (o upstream reescreveu essas areas e a resolucao
manteve o lado upstream, descartando um comentario/linha morta que so existia
no fork antigo). Nao sao modificacoes intencionais do POM - se algo parecer
errado ali, e mais provavel ser um residuo de merge do que uma feature.

## Como manter este relatorio

Regenerar apos qualquer novo merge de upstream: rodar
`git diff --stat origin/master...origin/pom-tensor-parallel` no clone do fork
e conferir se a lista de arquivos ainda bate com as secoes acima. Commits
novos do POM devem ganhar uma secao curta aqui explicando **por que** (nao
so o que) a mudanca existe - o "o que" ja esta no `git show` do commit.
