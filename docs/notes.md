# Build notes

## Findings so far

1. **Model access differs by account.** Claude 5.5 models returned AccessDenied; Claude Haiku 4.5, Sonnet 4.5,
   Nova (Lite/Pro/2-Lite), gpt-oss-120b, Llama 3.3 70B and Mistral Large 3 all worked. Probe with a tiny
   Converse call (src/step0b_probe_models.py) instead of trusting the model list.
2. **Claude models appeared only as inference profiles** (us./global. prefix) in the model listing, not as on-demand base models, so the profile ID is the one to call. Confirmed: calling the bare ID `anthropic.claude-haiku-4-5-20251001-v1:0` returns a ValidationException about invoking the model ID with on-demand throughput (message truncated in my terminal; the profile ID works).
3. **Guardrail PII ANONYMIZE did not mask prompts when configured with the legacy single `action` field.**
   It masked model _output_ only, while BLOCK worked on both. Setting `inputAction`/`outputAction`
   (and `inputEnabled`/`outputEnabled`) explicitly fixed it. Lesson: test input and output directions separately.
   One trace reported guardrailProcessingLatency=191 (unit assumed ms, single sample, input check only).
   TODO: measure real overhead = total latency with vs without the guardrail over ~20 runs.
4. **Guardrail results (version 2):** benign pass; name/email/policy number masked; SIN and card numbers blocked;
   legal and medical advice blocked as denied topics; prompt-attack blocked.
5. **Without retrieval the model answers generically** (a car-insurance deductible example) instead of the plan's $100 deductible.
   This is the motivation for RAG and contextual grounding.

## To measure next

- Retrieval hit rate and fact recall for four chunking strategies (step 3a)
- Hand-built RAG vs Bedrock Knowledge Base
- Latency, token cost per answer by model
- Agent loop: tool calls per question, failures

## Gotchas to mention

- Use a recent AWS CLI (older versions silently create classic OpenSearch Serverless collections).
- opensearch-py: AWSV4SignerAuth with service name `aoss`.
- OpenSearch search lags indexing by seconds, so add retries.

## Retrieval experiment (step 3a): 11 answerable questions, Titan v2 (512 dims), local cosine search
Metric note: my first fact-recall metric was too lenient. "$100" (deductible) also appears in the vision
document, so Q2 looked fine when the right document ranked 4th. Fixed: facts only count when found in a
retrieved chunk from an expected document.

| strategy | chunks | avg chars | doc@1 | doc@3 | fact@1 | fact@3 | fact@5 | MRR |
|---|---|---|---|---|---|---|---|---|
| whole_doc | 11 | 613 | 0.91 | 0.91 | 0.82 | 0.91 | 1.0 | 0.93 |
| line_with_title | 58 | 138 | 0.91 | 1.0 | 0.73 | 1.0 | 1.0 | 0.95 |
| fixed_200_ov40 | 46 | 176 | 0.91 | 1.0 | 0.64 | 0.91 | 0.91 | 0.95 |
| fixed_400_ov80 | 26 | 302 | 0.91 | 0.91 | 0.82 | 0.91 | 0.91 | 0.93 |

- n=11, so one question = 0.09. Differences between strategies are 1-2 questions, not conclusive.
- Q2 ("yearly deductible per person on Standard plan") is a real failure for whole_doc and fixed_400: the
  dental doc outranks the health overview (right doc ranked 4th). The word "deductible" is an exact term that
  vector search underweights -> candidate fix: hybrid (keyword + vector) search. Untested hypothesis.
- Q11 (two documents needed) is the other weak spot: fixed_200 misses a fact at k=3.
- Smaller chunks need k=3 to match larger chunks at k=1 (fact@1 lower), but cost less context per hit.

## Hybrid test (step 3c): vector vs BM25 vs hybrid, same 11 questions, cached embeddings
Hybrid = vector + hand-written BM25 (no stemming, tiny corpus), fused by RRF (k=60) or 50/50 min-max.
- BM25 alone is poor (fact@3 0.45-0.64): paraphrases miss ("braces" vs "orthodontics", "glasses" vs "eyeglasses").
- Hybrid DID fix the targeted failure: Q2 ("yearly deductible") now returns group_health_overview in the top 3
  for whole_doc and fixed_400 (BM25 ranks it #1).
- But hybrid did NOT improve overall: best vector-only (line_with_title) has fact@3 = 1.0; hybrid_rrf 0.91,
  hybrid_w50 0.73. doc@1 and fact@1 also dropped (vector 0.91/0.73 -> RRF 0.82/0.55 on line_with_title).
  New misses appear on Q10 (psychologist) and others, plausibly because common words ("plan", "enhanced") match many docs.
- Caveats: 11 questions; crude BM25; no tuning of fusion weights; OpenSearch's analyzers/stemming would differ.
- Decision: vector-only + line_with_title, top-3 for now. Candidate next experiment: rerank top-10 with Cohere Rerank 3.5
  (cohere.rerank-v3-5:0 appeared in the model list).

## OpenSearch Serverless NextGen (step 3b/3d)
- Setup worked first time with NextGen group (min 0 / max 2 OCU, standby replicas ENABLED), VECTORSEARCH collection,
  `knn_vector` dim 512, engine left at default. 58 chunks indexed and searchable within the polling window.
- Retrieval on OpenSearch matches the local numpy search exactly (line_with_title, vector-only):
  doc@1 0.91, doc@3 1.0, fact@1 0.73, fact@3 1.0, fact@5 1.0, MRR@10 0.95. No misses at k=3.
- Query latency (11 queries after 1 warm-up, from Toronto to us-east-1, question embeddings cached): median 133 ms,
  max 210 ms. This includes the network round trip, so it is not server-side time.
- TODO: record idle cost from Cost Explorer after a day (OpenSearch Serverless, NextGen scale-to-zero claim).
