# Build notes (raw material for the blog post and interview answers)

## Findings so far
1. **Model access differs by account.** Claude 5.5 models returned AccessDenied; Claude Haiku 4.5, Sonnet 4.5,
   Nova (Lite/Pro/2-Lite), gpt-oss-120b, Llama 3.3 70B and Mistral Large 3 all worked. Probe with a tiny
   Converse call (src/step0b_probe_models.py) instead of trusting the model list.
2. **Newer Claude models need an inference profile ID** (us./global. prefix), not the bare model ID.
3. **Guardrail PII ANONYMIZE did not mask prompts when configured with the legacy single `action` field.**
   It masked model *output* only, while BLOCK worked on both. Setting `inputAction`/`outputAction`
   (and `inputEnabled`/`outputEnabled`) explicitly fixed it. Lesson: test input and output directions separately.
   Guardrail overhead measured at ~191 ms per call.
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
