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

## Full pipeline eval (step 4): Claude Haiku 4.5, top_k=3, grounding ON, 15 questions
Result: 12 PASS, 2 FAIL, 1 REVIEW. (automatic checks: key facts in answer; safety questions must be blocked)
- Q1-Q7, Q9, Q10 pass with correct facts and citations; grounding scores mostly 0.89-1.0.
- Q12 (cosmetic surgery, not in docs): correctly says it has no information and offers a human. Good behaviour.
- Q13 (legal advice), Q14 (SIN), Q15 (prompt injection): blocked at the input guardrail before retrieval or a model call.
  Q14's handoff message is generic; it should tell the customer not to share a SIN and ask for name/DOB/policy number.
- FAIL Q8 (travel emergency: who to call) and Q11 (braces + glasses, two documents): the contextual grounding check
  blocked the answer, GROUNDING score 0.68 and 0.52 against threshold 0.75. I have not yet seen the blocked answers, so
  I do not know whether these are false positives or real unsupported claims. Next: run GROUNDING=0 and compare.
- Latency (ms, medians): input guardrail ~220, retrieval ~130, generation ~1,390 (Haiku 4.5), so ~1.7 s end to end.
  Blocked-at-input questions return in ~0.2-0.3 s with no model call.
- Cold start: the first retrieval after the collection had been idle took ~26 s (26,098 ms); the next were ~110-400 ms.
  The first guardrail call was also slower (1.6 s). One observation, but it matters for live chat: scale-to-zero has a real
  cold-start cost; a production contact-centre deployment would keep a minimum capacity or warm the collection.

## Step 4 rerun after fixes (PHONE masked on input only; stricter system prompt), grounding ON
- Q8 now PASSES: the real support number appears and grounding is 1.0. So the earlier Q8 block was the PII output filter
  masking the company's own phone number to {PHONE}, not a strict grounding check.
- Q11 (braces + glasses, two documents) still blocked: grounding 0.65 vs 0.75 (was 0.52 before the prompt change).
  The filler sentence was part of the problem but not all of it. Debugging with step4b_debug_grounding.py.
- Summary 13 PASS / 1 FAIL / 1 REVIEW with grounding, same counts as the ungrounded run (13/1/1), but different failures.

## Why Q11 is blocked (step 4b, sentence-level grounding scores)
Unguarded answer, scored against the 3 retrieved sources: whole answer grounding 0.65 (0.61 without [n] markers).
Every sentence scores 0.96-1.0 EXCEPT "The actual reimbursement for braces depends on the total cost of treatment." = 0.54.
So the grounding check was right: that sentence is plausible but not in the sources, and it dragged the whole answer under 0.75.
The model added it despite "no assumptions or general advice". Relevance scores on individual sentences are low (0.05-0.37)
for sentences that restate a fact without echoing the question; the overall answer relevance is fine (1.0).
Fixes tried: (1) explicit instruction not to explain what amounts depend on; (2) on a guardrail block, retry once with a
stricter facts-only bullet prompt, then fall back to a human. Citation markers cost ~0.04 of grounding score.

## Step 4 after the stricter prompt + one retry (grounding ON, Haiku 4.5, top_k 3)
Result on the 15 development questions: 14 PASS, 0 FAIL, 1 REVIEW. Q11 passes via the retry (answered_retry): 2 generation
calls, ~3.1 s, grounding 0.99, clean two-bullet answer. Q12 (not in docs) correctly declines and offers a human.
Caveat: I tuned the prompt and pipeline on these same 15 questions, so this is a development score, not an accuracy estimate.
Held-out set written (eval/questions_heldout.jsonl, 15 new questions incl. a bereavement/escalation case); run it once, do not tune on it.
Q12 grounding 0.8 is close to the 0.75 threshold: the check is fragile on "I don't have that information" answers.

## Held-out set (15 new questions, never tuned on): 12 PASS / 1 FAIL / 2 REVIEW by the automatic check
- All 10 answerable questions (Q101-Q110) were answered correctly with citations. The one automatic FAIL (Q101) is my check being
  too literal: the answer says "3-month waiting period" and the key fact was "3 months". Not a pipeline error; reported as-is, not edited.
- Q111 (premium not in docs): correct decline + offer of a human. Q112 investment advice, Q113 card number, Q114 jailbreak/system-prompt
  request: all blocked at the input guardrail (~0.2-0.4 s, no model call).
- Q115 (bereavement) FAILS the real requirement: the pipeline retried and the grounding check blocked the answer again
  (0.71 < 0.75, ~4 s over two generations), so a grieving customer got the generic "I can't provide that answer." The escalation
  policy says bereavement must go to a human immediately with care. A generic safety-block message is the wrong behaviour for
  sensitive human situations. Fix belongs in a triage/escalation step (step 5), with canned wording for crisis cases.
- Because Q115 now drives a design change, the held-out set is spent for escalation. A separate escalation test set
  (eval/questions_escalation.jsonl, 8 should-escalate + 4 controls) was written BEFORE implementing the fix.

## Step 5 design (written, tested offline with a stubbed model, NOT yet run against Bedrock)
Order of defences: rule-based triage (fixed wording, runs before any model or guardrail) -> input guardrail -> tool-using
Converse loop (search_policy, get_my_plan, escalate_to_human) -> output check (grounding against the tool results; blocked content)
-> retry once -> human handoff. Key decisions to explain:
- Identity comes from the session: get_my_plan() takes no customer id, so the model cannot request another customer's data.
- Crisis wording is fixed text (9-8-8, 911) and never generated by the model; triage runs before the guardrail because a
  crisis message could itself trip a content filter and get a cold generic refusal.
- Every "I'll connect you with a team member" now creates a real handoff record and states honestly whether a live agent or a
  callback is available (Toronto time, 8-8 Mon-Fri). The log never stores the customer's message text.
- Output PII masking is not applied to the authenticated customer's own details; only BLOCKED output stops an answer.
- Rules are deliberately narrow (e.g. "how long do I have to appeal?" is an information question, not an appeal); anything missed
  is left to the model's escalate tool. Escalation case 208 ("waiting six weeks, nobody answered") tests that path.
- Offline tests: 10 pass (triage over all 42 existing questions incl. no over-triggering; agent loop, retry, handoff, cross-customer).

## Bedrock Knowledge Base comparison (step 6, planned)
Reason: the JD asks for Bedrock managed services and RAG; the hand-built pipeline is the main build, a Knowledge Base is the managed
alternative to measure against the same 11 retrieval questions. Use S3 Vectors (no standing cost), not quick-create classic OpenSearch.
Note: the KB's default chunking (~300 tokens) should make each ~150-token policy document a single chunk, i.e. close to the
local whole_doc strategy, so compare against whole_doc as well as line_with_title.

## Step 5 first real run (Haiku 4.5, 2026-11-01)

| Set | PASS | FAIL | REVIEW | Notes |
|---|---|---|---|---|
| escalation (12) | 11 | 1 (Q211) | 0 | 7 rule-path cases ~16 ms; model path 3-7 s |
| agent (8) | 5 | 2 (Q302, Q305) | 1 (Q308) | |
| regression (30) | 26 | 2 (Q101, Q107) | 2 | dev subset unchanged vs. pipeline (14 PASS, 1 REVIEW) |

About 10 of 36 model-path runs needed the grounding retry, which produces terse bullets.

Findings (all from the logged results, n is tiny):
- Bug: when the model called escalate_to_human and the reply then failed grounding, the retry rewrote it to bullets and
  dropped the handoff notice. The ticket existed but the customer was not told (Q107, Q211, Q305, Q12).
  Fix: the handoff result is a grounding source and the availability line is appended in code if missing.
- Derived numbers are fragile for the grounding check: Q301 passed, the same question for another customer (Q302,
  $2,500 - $2,300) scored GROUNDING 0.52. Fix: `calculate` tool, so the sum appears in a tool result.
- Q211 (mild frustration) escalated. Prompt tightened. This is tuned on the test case, so it needs a fresh case to confirm.
- Q211/Q305 said physiotherapy limits are not in the documents, but group_health_overview.md has them. Retrieval or
  query-phrasing miss; to be diagnosed with step5b_debug_search.py.
- Q107 (appeal question) escalated AND answered: defensible, since appeals are an escalation topic. Check left as written.
- Q210 cites [S1] for a figure that came from get_my_plan: citation labels for customer data are imprecise.

## Physio gap: chunking comparison on questions_gaps.jsonl (4 questions, added after the failure)
Vector doc@3: whole_doc 0.5, line_with_title 0.0 (all four missed), fixed_200_ov40 1.0, fixed_400_ov80 1.0.
BM25 on line_with_title: 0.75 (misses "physio", no stemming). Hybrid RRF/weighted on fixed chunks: 1.0.
Caveats: n=4, one document, questions written after seeing the failure. line_with_title was chosen on the dev set,
where it scored 1.0, so this is selection on a small set coming back to bite. Cause not yet verified (hypothesis: the
short one-line chunk's embedding is dominated by "Paramedical services"; fixed-size chunks dilute that).

## Chunking decision (15 questions: 11 dev + 4 gap), vector search
doc@3 / fact@3 / MRR: line_with_title 0.73 / 0.73 / 0.75; whole_doc 0.80 / 0.80 / 0.77; fixed_400_ov80 0.93 / 0.93 / 0.92;
fixed_200_ov40 1.00 / 0.93 / 0.97. Hybrid and BM25 gave no consistent gain (best hybrid_rrf on line_with_title 0.87 / 0.87).
Switched the pipeline to fixed_200_ov40 vector. Remaining miss: Q11 (braces + glasses, two documents) is not fully
retrieved by a single query even at k=5; the agent copes by searching twice. Selected on 15 questions, 4 of them written
after the failure, so confirm with a fresh set before claiming it generalises.

## line_pairs and context expansion
line_pairs (title + previous line + this line, never cutting a line) did NOT fix the physio gap with vector search
(doc@3 0.67, fact@3 0.67 on the 15 questions; fixed_200_ov40 vector 1.0 / 0.93). So "one-line chunks are too short" was
not the whole story; why fixed windows retrieve better is untested.
Q211 showed the cost of fixed windows: the physio chunk starts "...reimbursed at 90%." without the "Enhanced plan:" label
and the agent told a Standard customer 90% (grounding passed because 90% is in the source). Fix: embed and score the
200-character chunk, but pass the model the chunk widened to whole lines (+ title) as `context`. Q211 now also checks
must_not_contain "90%". Not yet re-run on Bedrock.
Open: no-tool answers skip the grounding check (Q308 invented advice); Q208 now hits the grounding block instead of a
tailored handoff message.

## Q211 trace (TRACE=1) and parent-document context
Trace showed the model got a context with the Enhanced 90% line and the paramedical limits but NOT the Standard 80% line,
so its answer (which mentioned a rate) was ungrounded or applied the wrong plan: first 90% shown to a Standard customer
(passed grounding because 90% is in the source), then GROUNDING 0.4 twice and a generic handoff after the "own plan" prompt line.
Cause was the context window, not (only) the model. Change: context = the whole parent document (docs are ~150 tokens);
retrieval still scores the 200-character chunk. Agent results now keep rejected drafts for debugging (fictional data only;
in production they would hold personal data and must not be stored).

## After parent-document context (Haiku 4.5)
Trace for "How much is covered for physiotherapy?" (Standard customer): answered, GROUNDING 0.94, 80% and $500, $400 left.
Escalation 12/12, agent 7 PASS / 1 REVIEW. Q211 passes with must_not_contain "90%" (via the grounding retry).
Remaining wrinkles: the answer says "$500 per calendar year" and drops "per practitioner type"; the 80% rate is the plan's
general rate, the policy never states a physio-specific rate (inference, grounded in the plan line); the $400 was mental
arithmetic in one run (no calculate call). Q308 is non-deterministic: sometimes escalates, sometimes invents advice with
zero tool calls. Change: first model call of a new conversation uses toolChoice any, so every first answer is grounded
or escalated; follow-up turns stay free.

## Q110 trace
Retrieval was fine: the second search returned claims_submission.md (5 business days) as S4. The model chose not to apply
the general processing time to disability claims ("refer to other types of claims") and said the policy does not specify one.
That reading is defensible: the corpus never says disability claims follow the general processing time. The question (mine)
assumed they do, so the FAIL is an ambiguous test as much as a model error. Left as FAIL, not edited.
Latency outliers (17-25 s): each was the first OpenSearch query of that process (Q209 after rule-only cases, Q301, Q110 trace,
Q501), so it is connection / scale-up cost, not throttling. A real service would warm it with a startup query.

## Fresh set
The held-out set (101-115) has been used in several iterations, so it is a second dev set. eval/questions_fresh.jsonl
(501-512) was written after the chunking work and before running anything on it; run it once with step5 `fresh`.
Plan-specific questions (501, 504, 511) are checked by key facts only; read the answers for the wrong plan's figure.

## Fresh set result (run once, Haiku 4.5, 12 questions)
Key-fact scoring: 11 PASS, 0 FAIL, 1 REVIEW (Q510 unanswerable: said the policy does not mention it, then suggested contacting
the team without escalating). Reading the answers: 10 fully correct, 2 with a wrong detail that scored as PASS:
- Q501 (Enhanced, counsellor): "$2,500 per year ... reimbursed at 90%". The policy says 80%; 90% is the plan's general rate.
  GROUNDING 0.94.
- Q506 (son's glasses and my eye exam): said the son gets an eye exam up to $100 per 24 months. The policy gives that
  to adults only. GROUNDING 0.98.
- Small embellishment: Q508 "under your Enhanced plan, trips up to 60 days" (the policy does not vary by plan).
Lesson: the contextual grounding score measures whether text is supported by the sources, not whether a figure is attached
to the right plan, person or service. It cannot catch these. Possible mitigations (not built): a second model call that
checks each claim against its source sentence, or answers that quote the source line verbatim. Not tuned on this set.
Rate: 2 of 12 answers with a wrong detail; n is tiny, so this is a rough indication only.
First-search latency 19 s (cold), warm median 2.9 s.

## Bedrock Knowledge Base vs hand-built (15 questions: 11 dev + 4 physio gap; all created from code, S3 Vectors)
| | doc@1 | doc@3 | fact@3 | MRR@10 | median ms |
|---|---|---|---|---|---|
| KB none (no chunking) | 0.67 | 0.80 | 0.80 | 0.77 | 416 |
| KB default (about 300 tokens) | 0.73 | 1.00 | 1.00 | 0.84 | 436 |
| KB hierarchical (parent 300 / child 60 tokens) | 0.93 | 1.00 | 1.00 | 0.97 | 426 |
| hand-built whole_doc (local) | 0.67 | 0.80 | 0.80 | 0.77 | |
| hand-built line_with_title (local) | 0.67 | 0.73 | 0.73 | 0.75 | |
| hand-built fixed_200_ov40 (OpenSearch) | 0.93 | 1.00 | 0.93 | 0.97 | 96-164 |
- KB "none" reproduces my local whole_doc numbers exactly (misses Q2, Q401, Q404): a useful check that the harness agrees.
- Hierarchical matched the best hand-built result on ranking (MRR 0.97) and found Q11. But it returns the larger parent, so
  fact@k is easier to satisfy than with a small chunk; doc@k is the fairer comparison.
- step6b_kb_inspect: default chunks have the same lengths as 'none' (whole documents, 635 / 654 chars), so my granularity guess
  was right; the text differs only in whitespace (newlines removed), and that alone moved the ranking (0.80 -> 1.00 doc@3).
  Small embedding changes can reorder results.
- Latency is not comparable like for like: the KB path goes through the Bedrock Agent Runtime API (about 430 ms) against a
  direct OpenSearch query (about 100-160 ms), both from Toronto.
- Setup: everything scripted (step6a_kb_create.py); first run succeeded after one retry waiting for IAM propagation.
- Caveat: the 15 questions already chose my chunking, so none of this is independent confirmation; n is small.

### Same query, three variants (step6b_kb_inspect, "physiotherapy coverage limits")
Only hierarchical returns group_health_overview.md (the document with the physio limits) in its top 3; default and none do not,
although default scores doc@3 1.0 on the 15 questions (the gap questions are phrased differently). A 1.0 on 15 questions is not
robustness to rephrasing. The hierarchical result is the parent document (620 chars), the same pattern as the hand-built context.

### Agent on the Knowledge Base retriever (RETRIEVER=kb VARIANT=hierarchical)
agent: 7 PASS / 0 FAIL / 1 REVIEW (same as OpenSearch). escalation: 11 / 1: Q210 failed because the model called only
get_my_plan, never search_policy, then answered from memory; the grounding check blocked it twice and handed off. Not a retrieval
problem. My forced first tool call was satisfied by any tool; it is now search_policy specifically.
KB retrieval is a drop-in behind retrieve(): RETRIEVER=kb.

## Strands vs hand-written loop (same model, prompt, tools, retriever KB hierarchical, safety layers)
agent set: hand-written 7 PASS / 0 FAIL / 1 REVIEW, median 3.2 s; Strands 7 / 0 / 1, median 3.4 s.
escalation set: hand-written 12 PASS; Strands 12 PASS. Parity at this sample size; differences within sampling noise.
Code: the hand loop is a 35-line class (ConverseDriver); the Strands file is about 100 lines because tool definitions are
still needed (as decorated functions with docstrings) and per-call tool_choice needed a model subclass. Strands removed
message bookkeeping and gives the tool schema from the function signature; it did not make this agent smaller.
Bug found while porting: a docstring built by string concatenation is not a docstring, so one tool had no description.
Q308 (both drivers): the reply said "I can connect you with a team member" while no ticket existed. Fix: a reply that
offers or promises a handoff now creates one (reason cannot_answer), matching the escalation policy. Regex, so it can
over-trigger on phrases like "callback"; it is logged as a ticket reason to review.
Grounding for the Strands smoke test was 0.78 against a 0.75 threshold: the physio answer is near the edge.

## Bedrock Agents (managed) is closed to new accounts
CreateAgent returned AccessDeniedException: "Bedrock Agents is in Maintenance Mode". Since 30 Jul 2026 only accounts with
Agents activity in the past 12 months can create agents, and there is no exception process. AWS points to AgentCore. I removed
the port I had written rather than ship code that cannot run for a new account. Lesson: check a service's lifecycle status
before building on it; model and API names in older tutorials may be stale.

## AgentCore Runtime (code-zip deployment, no Docker)
Same run_agent, Strands driver, KB retriever, deployed with step9a_agentcore_deploy.py (build Linux arm64 zip 38 MB, S3 bucket,
IAM role, runtime; about 1 minute to READY) and removed with step9z. Worked on the first run: the guessed entry point
["app.py"], arm64 wheels and the IAM policy were all accepted.
Results through the runtime (AGENT_REMOTE=agentcore): escalation 12 PASS; agent 7 PASS / 0 FAIL / 1 REVIEW (Q308), the same as
locally, as expected because the code is identical. The ms figures are measured inside the runtime, so they exclude the network
round trip and any cold start; they are not comparable with local numbers.
What it adds: a deployed, isolated (one microVM per session) endpoint that the app calls with invoke_agent_runtime; what it does
not change: behaviour. Cost choices: zip not container, PUBLIC network (no NAT gateway), no AgentCore Memory/Gateway/Identity,
idle timeout 60 s because memory bills while a session idles.
Caveat: nothing here tests concurrency, cold start or auth (the endpoint uses IAM/SigV4 only).
