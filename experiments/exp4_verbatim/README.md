# Experiment 4: Verbatim Copying vs. Semantic Paraphrasing

## Research Question
Does citation preservation correlate with literal string copying or semantic paraphrasing strategies?

## Hypothesis
**H1**: Citation preservation correlates MORE with verbatim overlap than semantic similarity  
**H2**: Gemini's superior preservation results from higher verbatim copying rates  
**H3**: Claude's total loss results from complete semantic reformulation with zero verbatim copying

## Method

### Design
3 length conditions × 3 models × 50 facts × 5 layers = 2,250 API calls

### Length Conditions
Control output length to test if preservation varies with constraint:

**1. Short (max_tokens=100)**
- Forces compression, may increase verbatim copying

**2. Medium (max_tokens=200)**
- Balanced condition, typical output length

**3. Long (max_tokens=400)**
- Allows elaboration, may decrease verbatim copying

### Stimuli
Same 50 verified scientific facts from Experiment 2 with citations

### Metrics

**1. Verbatim Overlap (Character-level)**
```python
# Longest common substring (LCS) with original fact
verbatim_overlap = len(longest_common_substring(original, interpretation))
verbatim_ratio = verbatim_overlap / len(original)
```

**2. Semantic Similarity**
```python
# Cosine similarity of sentence embeddings
semantic_similarity = cosine_sim(embed(original), embed(interpretation))
```

**3. Citation Preservation**
```python
# Binary: exact citation present/absent
citation_preserved = 1 if original_citation in interpretation else 0
```

**4. Citation Phrase Copying**
```python
# Check if exact citation phrase appears verbatim
citation_phrase_intact = check_exact_substring(original_citation, interpretation)
```

## Analysis Plan

### Primary Analysis: Logistic Regression
```
P(citation preserved) ~ verbatim_ratio + semantic_similarity + model + layer + length_condition
```

**Predictions**:
- Verbatim ratio: Positive coefficient (β > 0)
- Semantic similarity: Weaker or non-significant coefficient
- Model effects: Gemini > GPT-4 > Claude in verbatim ratio
- Layer: Negative effect (verbatim decreases with distance)

### Secondary Analyses

**1. Model Comparison**
```
# Compare verbatim ratios across models
ANOVA: verbatim_ratio ~ model
Post-hoc: Tukey HSD for pairwise comparisons
```

**2. Preservation Mechanism**
```
# Partial correlation controlling for semantic similarity
partial_corr(citation_preserved, verbatim_ratio | semantic_similarity)
```

**3. Length Effects**
```
# Does output length affect copying strategy?
ANOVA: verbatim_ratio ~ length_condition × model
```

### Visualization
1. Scatter: Verbatim ratio vs. Citation preservation (by model)
2. Scatter: Semantic similarity vs. Citation preservation (by model)
3. Heatmap: Model × Length condition → Preservation rate
4. Line plot: Layer → Verbatim ratio (by model)

## Predictions by Model

| Model | Verbatim Ratio | Semantic Sim | Citation Preservation | Mechanism |
|-------|---------------|--------------|----------------------|-----------|
| Claude | Low (< 0.1) | High (> 0.9) | 0% | Complete reformulation |
| GPT-4 | Medium (0.3) | High (> 0.9) | 15% | Selective copying |
| Gemini | High (> 0.5) | Low (0.7-0.8) | 24% | Liberal copying |

## Expected Outcomes

**Outcome 1**: Verbatim ratio predicts preservation (β > 0.5, p < 0.001)
- **Interpretation**: Citation preservation requires literal string copying
- **Implication**: Semantic paraphrasing inherently destroys citations

**Outcome 2**: Gemini has higher verbatim ratio than GPT-4
- **Interpretation**: Explains paradox (worse semantic, better attribution)
- **Implication**: Different models optimize for different objectives

**Outcome 3**: Claude has near-zero verbatim copying
- **Interpretation**: Elaboration = complete reformulation
- **Implication**: Architectural constraint, not fixable via prompting

## Scientific Value
- Identifies **mechanistic basis** of citation preservation
- Quantifies **trade-off** between paraphrasing and attribution
- Distinguishes **copying vs. reformulation** as processing strategies

## Practical Implications
- If verbatim copying required: Need explicit "copy citation phrases" instructions
- If models differ in copying propensity: Select model based on task requirements
- Inform design of citation-preserving AI systems

## Follow-up Questions
- Can we train models to copy citations while paraphrasing content?
- Is there an optimal verbatim ratio for balancing understanding and attribution?
- Do humans exhibit similar trade-offs in serial reproduction?