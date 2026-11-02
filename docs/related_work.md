# Related-work survey for the grounding-checks post

Done 2 November 2026 with a web search tool and page fetcher; not a systematic review in the PRISMA sense. One search engine,
about 15 queries, only pages the tool could fetch. "Read" means I read the fetched page or the part of it that the fetcher
returned (long papers were summarised by the fetcher, so details may be missed); "abstract" means abstract-level only;
"title only" means I saw the title in search results and did not read it. Nothing marked "title only" is cited as support.

## Questions
1. Is checking an answer claim by claim against the sources established practice?
2. Is "supported by the sources is not the same as correct" already documented, and for what error types?
3. Have injected (synthetic) errors been compared with natural ones as a test of detectors?
4. Has anyone measured the Bedrock contextual grounding check, and its Classic and Standard tiers?
5. How reliable are model judges, and what biases apply when a model checks a model?

## Sources
| Source | Status | What it says | Relevance |
|---|---|---|---|
| AWS: contextual grounding check docs | read | Grounding and relevance scores; Classic = strict grounding, Standard = more consistent for reasoning and conversational text; several grounding sources evaluated together; Classic is the default tier | Explains the setup and the combined-source behaviour |
| AWS: Standard tier announcement (20 Oct 2026) | read | "More robust", more consistent scores for reworded answers, same price, needs cross-Region inference; **no accuracy numbers** | Q4: no vendor measurement |
| Azure AI Content Safety groundedness docs | read | Groundedness is defined relative to the supplied sources; corrections follow the sources; English only; "ensure your grounding sources are accurate" | Another managed check with the same definition |
| FActScore (Min et al.) | abstract | Atomic-fact precision against a knowledge source; automated estimator under 2% error vs humans on biographies | Q1: decomposition is established |
| MiniCheck (Tang et al.) | read (fetcher summary) | Small sentence-level fact-checkers trained on synthetic data reach GPT-4-level accuracy at far lower cost; LLM-AggreFact benchmark of 10 datasets; assumes each claim is verifiable against a single document; no separate atomic-fact step needed | Q1; shows a cheaper checker; its single-document assumption is exactly where attachment errors live |
| RAGTruth (Niu et al.) | read (fetcher summary) | About 18,000 human-annotated responses from six models; span-level labels (conflict, baseless information); response-level detection F1 ranged from 1.0% to 80.7% depending on method and model; "hallucination detection remains a significant challenge" | Q2, Q3: natural errors, hard to detect |
| Vectara leaderboards and FaithJudge | abstract | Hallucination leaderboards; judge built from human-annotated hallucinations; notes limits of existing detectors | Related benchmark, not compared |
| Arize LibreEval page (vendor) | read, self-reported | Synthetic hallucinations easier to detect than natural ones; base models' recall about 35% lower on natural examples; LLM judges disagree about four times more on natural data | Q3: same pattern as my injected-vs-real result, but vendor-reported and unchecked |
| Perturbation-based synthetic data for hallucination detection (2407.05474) | read | Detectors trained on LLM-made hallucinations; authors say synthetic errors may differ from unintentional ones; only dialogue studied | Q3: caveat on synthetic errors |
| ClashEval (Wu et al.) | read | 1,294 questions, numeric and categorical perturbations of retrieved documents; models adopt wrong retrieved content in a large share of cases (GPT-4o 60.8% when its prior was right); enriched error rate, so not real-world rates | A different failure (the document itself is wrong); relevant to conflicting documents |
| Langfuse, Traceloop, OneUptime posts | read | Claim-level faithfulness checks; faithfulness vs correctness; plan names stay inside claims; wrong-source attribution as a test; calibrate judges; no injected-error catch rates | Q1, Q2 |
| LLM-as-a-judge survey (2411.15594) | read (fetcher summary, last 5,000 characters not read) | Self-enhancement, concreteness, position and length biases; adversarial inflation; thin meta-evaluation of mitigations | Q5 |
| ALCE (Gao et al., 2305.14627) | read (fetcher summary) | Benchmark for answers with citations; citation recall = an NLI model checks that the cited passages entail each statement; citation precision flags citations that do not help; about half of ChatGPT and GPT-4 answers on ELI5 were not fully supported by their citations; NLI agreement with humans about 85% (recall) and 78% (precision); NLI cannot detect partial support | Per-statement check against only the passages the statement cites. Untested idea for my agent: my answers carry [S#] citations |
| VeriCite (2510.11394) | read (fetcher summary) | Verifies statements with an NLI model before writing the final answer, improving citation F1 but with mixed effects on correctness; extra cost not reported; metrics rely on NLI models that may share its blind spots | Citation quality, not attachment to the right entity; same-family-evaluator caveat |
| Other attribution work seen in search results (Learning to Plan and Generate Text with Citations, Model Internals-based Answer Attribution, CoTAR and others) | title only | Not read | Not cited |
| Bedrock Classic vs Standard comparisons by third parties | none found | | Q4 |

## What I did not find
A measured comparison of the Classic and Standard grounding tiers; an injected-error catch rate compared with a trivial
baseline for a managed grounding check; the same checks scored on injected and on real errors side by side (the closest is
the Arize page, self-reported). The absence of a hit in one search engine is not proof that none exists.
