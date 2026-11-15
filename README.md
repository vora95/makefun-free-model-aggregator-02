# free-model-aggregator-02

[![make.fun · free-model-aggregator-02](https://make.fun/badge/free-model-aggregator-02.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![rank](https://make.fun/badge/free-model-aggregator-02/vora95/makefun-free-model-aggregator-02/rank.svg)](https://make.fun/bounty/free-model-aggregator-02/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![reward](https://make.fun/badge/free-model-aggregator-02/vora95/makefun-free-model-aggregator-02/reward.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)

One server, two client wires (OpenAI Chat Completions + Anthropic Messages), ten simulated
free-tier providers. For each of the 20 requests it picks the best free model that meets the
request's requirements, speaks that provider's wire (OpenAI, Anthropic or Gemini), and fails
over to the next-best legal model when a provider refuses. `GET /v1/models` lists the free
catalog; `node main.ts --list` prints the catalog it registers.

Policy: candidates in rating order, one same-model retry for transient failures, dead models
skipped ("budget is spent" / "is out" / 400), 20 s upstream timeout with an adaptive budget,
max 8 attempts per request. The stream re-emitter dedupes repeated finish chunks: OpenRouter
repeats the finish chunk (finish chunk + usage chunk), and a stream with two finish chunks
fails the env's stream_order check — only the first finish_reason per choice survives.

Run: `node main.ts` with `MAKEFUN_ENV_URL` and `MAKEFUN_PROMPT_FILE` set (Node 22+, stdlib only).

Grade-local 3/3 (15 Nov 2026): score 94 — quality 90, served 100, wire 100,
failover 100, catalog 100; real_model 12/12 per trial, replay matched byte for byte.
