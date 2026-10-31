# Bedrock customer-support chatbot (fictional insurer demo)

Customer-service chatbot with RAG, tools, guardrails and human handoff on Amazon Bedrock. All data is fictional.

## Build order
0. List models  ->  src/step0_list_models.py
1. Converse API ->  src/step1_converse.py
2. Guardrails (PII, denied topics, grounding)
3. RAG (chunk, embed, OpenSearch Serverless NextGen), eval hit rate@k
4. Tools + agent loop (search_policy, get_customer, escalate_to_human)
5. Framework rebuild (Strands or LangGraph) + Bedrock Agents comparison
6. Handoff summary, chat UI (Streamlit), Terraform/CDK, architecture diagram

## Cost safety
Max OCU limit on the collection group; budget alert; delete collection at the end.
