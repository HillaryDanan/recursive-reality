# Experiment 3: Elaboration and Citation Loss

## Research Question
Does elaboration level causally determine citation loss in LLMs?

## Hypothesis
**H1**: Claude's 100% citation loss is caused by elaboration process, not architectural constraint  
**H2**: Explicit instructions to preserve citations will recover attribution despite elaboration

## Method

### Design
Within-subjects: 3 prompt conditions × 3 models × 50 facts × 5 layers = 2,250 API calls

### Prompt Conditions

**1. Baseline (Replication)**
```
Please restate this scientific fact in your own words: [FACT]
```

**2. Preserve (Explicit Instruction)**
```
Please restate this scientific fact in your own words. IMPORTANT: Preserve all citations exactly as they appear in the original.

[FACT]
```

**3. Elaborate + Preserve (Test Compatibility)**
```
Please explain this scientific fact in detail, providing additional context and elaboration. IMPORTANT: Preserve all citations exactly as they appear in the original.

[FACT]
```

### Stimuli
Same 50 verified scientific facts from Experiment 2 with citations

### Models
- GPT-4
- Claude-3.5-Sonnet
- Gemini-2.5-Flash

### Parameters
- Temperature: 0.1
- Max tokens: 2000
- 5 interpretation layers per condition

## Predictions

### If Elaboration is the Cause
- Baseline: Replicates Exp 2 (Claude 0%, GPT-4 15%, Gemini 24%)
- Preserve: ALL models improve preservation
- Elaborate: Claude STILL loses citations (elaboration incompatible with preservation)

### If Architecture/Training is the Cause  
- Baseline: Replicates Exp 2
- Preserve: Claude MAY improve (can follow instructions)
- Elaborate: Claude CAN preserve while elaborating (instruction compliance)

## Analysis Plan

### Primary Analyses
1. **ANOVA**: Preservation rate ~ condition × model × layer
2. **Planned contrasts**:
   - Baseline vs. Preserve (instruction effect)
   - Preserve vs. Elaborate (elaboration cost)
   - Model × condition interaction (architecture differences)

### Secondary Analyses
1. **Text length**: Does "Elaborate" condition actually increase length?
2. **Semantic similarity**: Are elaborations meaningful or verbose?
3. **Citation accuracy**: Are preserved citations correct or hallucinated?

### Effect Sizes
- Cohen's d for condition differences
- η² for model × condition interactions
- Report all results (significant and non-significant)

## Timeline
- Setup: 1 day
- Data collection: ~2 hours (2,250 API calls with rate limiting)
- Analysis: 1 day
- Interpretation: 1 day

## Expected Outcomes

**Outcome 1**: Instructions improve preservation but elaboration still causes loss
- **Interpretation**: Elaboration process fundamentally incompatible with exact phrase preservation
- **Implication**: Architectural constraint, not training artifact

**Outcome 2**: Instructions fully recover preservation even with elaboration
- **Interpretation**: Claude CAN preserve citations when explicitly instructed
- **Implication**: Training/prompt engineering solution exists

**Outcome 3**: Mixed results across models
- **Interpretation**: Model-specific trade-offs between elaboration and preservation
- **Implication**: Need model-specific strategies for citation-rich tasks

## Scientific Value
- Tests **causal mechanism** of citation loss
- Distinguishes **architectural vs. behavioral** constraints
- Informs **practical interventions** for citation preservation

## Open Questions After Exp 3
- If instructions work: Why doesn't default behavior preserve citations?
- If instructions fail: What architectural changes would enable preservation?
- Can we quantify the elaboration-preservation trade-off curve?