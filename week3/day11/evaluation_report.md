# Evaluation Report – RAG Pipeline Engineering Analysis

# 1. System Overview

This project implements a Retrieval-Augmented Generation (RAG) pipeline consisting of:

A vector retrieval layer (Chroma + embeddings)
A local LLM for answer generation
An external LLM judge (Azure OpenAI via DeepEval) for structured evaluation

The goal of this evaluation is not only to measure answer correctness but to analyze:

- Retrieval quality
- Grounding behavior
- Metric inconsistencies
- Security robustness
- Architectural weaknesses
This report documents all iterations performed during development.


# 2. Evaluation Setup
# 2.1 Judge Configuration

5 Evaluation metric was performed using:

1. FaithfulnessMetric
2. AnswerRelevancyMetric
3. ContextualRelevancyMetric
4. ContextualRecallMetric
5. ContextualPrecisionMetric

The generator model and the evaluation model were separated to avoid self judging bias.


# 2.2 Exact vs Soft Accuracy

Two additional evaluation methods were computed:

Correct answers: 3 / 18  
Exact match accuracy: 0.167  
This metric penalizes formatting differences and paraphrasing.


Soft-match correct answers: 10 / 18  
Soft-match accuracy: 0.556  
Soft matching better captures semantic equivalence.

As conclusion, Exact matching significantly underestimates system capability in generative settings.


# 3. Quantitative Results

# Average DeepEval Scores

FaithfulnessMetric: 0.944
AnswerRelevancyMetric: 0.981
ContextualRelevancyMetric: 0.432
ContextualRecallMetric: 0.778
ContextualPrecisionMetric: 0.667

# 3.1 Local vs Cloud Comparison

The pipeline was evaluated under two generation settings:

 **Local model:** llama3.2:3b (Ollama)
 **Cloud model:** gpt-4.1-mini (Azure)
 Retrieval layer remained identical in both runs.

Instead of a raw comparison table, the differences are analyzed metric-by-metric below.


# Faithfulness
Local: **0.944**
Cloud: **0.963**
Difference: **+0.019 (Cloud higher)**

Cloud generation shows slightly stronger grounding discipline. 
Both models are highly faithful, but the cloud model demonstrates marginally better consistency when aligning answers strictly with context.
Impact: Low practical difference. Both are production-safe in terms of hallucination control.


# Answer Relevancy

Local: **0.981**
Cloud: **0.954**
Difference: **-0.027 (Local higher)**

Local responses were slightly more direct and concise.
Cloud occasionally produced more verbose phrasing, which slightly reduced relevancy scores.
Impact: Minimal. Both models answer the correct question reliably.


# Contextual Relevancy

Local: **0.432**
Cloud: **0.432**
Difference: **No change**

This is the key insight.
Changing the generation model did not improve contextual relevancy at all.

This confirms the primary bottleneck lies in:
- Retrieval noise
- Chunk ranking
- Top-K configuration
- Chunk segmentation strategy

Impact: Retrieval quality dominates contextual purity.

# Contextual Recall

- Local: **0.778**
- Cloud: **0.861**
- Difference: **+0.083 (Cloud higher)**

The cloud model more reliably extracts all necessary details from the retrieved context. Even when chunks are identical, GPT-4.1-mini better utilizes the available information.
Impact: Cloud provides stronger completeness in answers.


# Contextual Precision

- Local: **0.667**
- Cloud: **0.671**
- Difference: **+0.004 (Negligible)**

Ranking behavior remains nearly identical.
This again confirms that retrieval configuration not generation is the dominant constraint.



# Overall Interpretation

- Retrieval quality is the primary bottleneck.
- Model upgrade improves recall, not retrieval purity.
- Local model already achieves high faithfulness and relevancy.
- Switching to cloud does not solve contextual noise issues.

Conclusion:
Improving retrieval strategy would yield higher returns than upgrading the generation model.


# 4. Metric Interpretation

# 4.1 Faithfulness (0.944)

The system rarely hallucinates.  
Answers are generally supported by retrieved context.

The grounding rule:

"If the answer is not in the context, say not found" proved highly effective.

# 4.2 Answer Relevancy (0.981)

Responses directly address user questions. This confirms prompt structure is well-aligned with task objective.

# 4.3 Contextual Relevancy (0.432)

This is the weakest metric.

Observation:
The retrieval layer often returns partially relevant or noisy chunks. The generator remains grounded, but retrieval inefficiency reduces contextual purity. This confirms retrieval is the main bottleneck.

# 4.4 Contextual Recall (0.778)

Most necessary information is retrieved. Missed cases are primarily due to ranking issues rather than embedding failure.


# 4.5 Contextual Precision (0.667)

Relevant chunks are frequently retrieved but not always ranked highest. Ranking strategy requires refinement.


# 5. Retrieval Iterations and Trade-offs

Several retrieval strategies were tested:

Top-K adjustment
Threshold filtering
MMR (diversity-based retrieval)
Chunk truncation
Context window limits

Findings:

Increasing top_k:
- Improves recall
- Reduces precision

Applying threshold filtering:
- Improves contextual relevancy
- Risks lowering recall

Using MMR:
- Reduces redundancy
- Slightly affects recall

There is no single configuration that optimizes all metrics simultaneously.
Retrieval tuning produces trade-offs rather than linear improvements.


# 6. Metric Inconsistency Analysis

A key observation:

Faithfulness remained high even when contextual relevancy was low.

Interpretation:

- The generator refuses when uncertain.
- Retrieval noise does not automatically cause hallucination.
- The model prioritizes safety over speculation.

This reveals that grounding constraints are stronger than retrieval precision.


# 7. Security and Robustness Evaluation

Manual adversarial probes were executed to evaluate:

- Prompt injection resistance
- System prompt exposure (Recommended mitigation: strict separation between system prompt and user-accessible context, and filtering of meta-instruction queries.  )  
- Secret exfiltration
- Hallucination under non-existent queries
- Sensitive data leakage
- Context manipulation attacks

# 7.1 Results

Out of 6 probes:

- 5 resisted successfully
- 1 vulnerability detected

### Resisted Attacks

Prompt injection override  
Environment secret exfiltration  
Hallucination (non-existent policy)  
Sensitive personal data request  
Context manipulation attempt  

# Detected Vulnerability

System prompt exposure

When directly asked, the model revealed its system instructions.
This indicates insufficient separation between system configuration and user-visible output.
This is an architectural issue not a retrieval issue.


# 8. Threat Model Summary

The system was analyzed across four layers:

Input Layer:
- Resistant to instruction override

Retrieval Layer:
- Performance-sensitive but stable

Generation Layer:
- Hallucination-resistant
- Vulnerable to prompt disclosure

Environment Layer:
- No secret leakage detected

# 9. Architectural Strengths

Strong grounding enforcement
Low hallucination rate
High answer relevancy
Safe fallback behavior
Resistant to secret exfiltration
Robust against document-level injection


# 10. Architectural Weaknesses

- Retrieval ranking inefficiency
- Moderate contextual noise
- System prompt exposure vulnerability
- Metric trade-offs under tuning

---

# 11. Lessons Learned

1. Exact-match accuracy is insufficient for generative QA evaluation.
2. Retrieval ranking quality dominates overall performance.
3. Grounding rules dramatically reduce hallucination.
4. Security must be evaluated independently from correctness.
5. Prompt isolation is critical in production-grade systems.
6. Metric optimization involves trade-offs, not absolute gains.


# 12. Conclusion

This project demonstrates that building a functional RAG system and engineering a robust RAG system are fundamentally different tasks.

The implemented system is:

- Semantically grounded
- Safety-oriented
- Retrieval-sensitive
- Resistant to hallucination
- Secure against secret leakage
- Architecturally incomplete in prompt isolation

The primary bottleneck is retrieval ranking, not generation quality.

Future improvements should focus on:

- Re-ranking strategies
- Meta-query filtering
- Prompt compartmentalization
- Structured adversarial evaluation pipelines

The evaluation confirms that robustness, grounding, and security analysis are as important as accuracy metrics when assessing RAG systems.