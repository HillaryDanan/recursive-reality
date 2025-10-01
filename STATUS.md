# Recursive Reality - Current Status

**Last Updated**: September 30, 2025  
**Phase**: Experiments 1 & 2 COMPLETE, Paper DRAFT READY FOR SUBMISSION

---

## Quick Facts

- **Repository**: https://github.com/HillaryDanan/recursive-reality
- **Primary Finding**: Double dissociation - semantic preservation ≠ source attribution
- **Claude**: 0% citation preservation, 2.87× text expansion
- **GPT-4**: 15% citation preservation, 1.17× text expansion  
- **Gemini**: 24.2% citation preservation (best), 2.41× text expansion
- **Statistical Significance**: χ²=51.36 (model differences), χ²=91.22 (layer effects), both p<0.0001

---

## Completed Work ✅

### Experiment 1: Information Transformation
- **Status**: COMPLETE
- **Data**: 1,326 interpretations across 100 scientific facts
- **Completion**: 88.4%
- **Key Finding**: Three distinct processing strategies (elaboration, maintenance, degradation)
- **Models**: GPT-4 (100%), Claude (100%), Gemini (69%)

### Experiment 2: Source Attribution Preservation
- **Status**: COMPLETE
- **Data**: 660 interpretations across 48 facts with citations
- **Completion**: 88%
- **Key Finding**: Catastrophic citation loss (85.6% absent)
- **Models**: GPT-4 (100%), Claude (76%), Gemini (100%)

### Paper Draft
- **Status**: PUBLICATION READY
- **Location**: `docs/paper_draft.md`
- **Length**: 5,200 words
- **Target**: Nature Communications
- **Sections**: Complete (Abstract, Intro, Methods, Results, Discussion, Conclusions, References)

---

## Designed But Not Run 📋

### Experiment 3: Elaboration and Citation Loss
- **Purpose**: Test if explicit instructions can preserve citations during elaboration
- **Design**: 3 conditions × 3 models × 50 facts × 5 layers = 2,250 API calls
- **Conditions**: Baseline, Preserve, Elaborate+Preserve
- **Cost**: ~$50-75
- **Time**: 2-3 hours with rate limiting
- **Files**: `experiments/exp3_elaboration/`

### Experiment 4: Verbatim Copying vs. Paraphrasing
- **Purpose**: Test if citation preservation correlates with literal string copying
- **Design**: 3 length conditions × 3 models × 50 facts × 5 layers = 2,250 API calls
- **Conditions**: Short (100 tokens), Medium (200 tokens), Long (400 tokens)
- **Cost**: ~$50-75
- **Time**: 2-3 hours with rate limiting
- **Files**: `experiments/exp4_verbatim/`

---

## Repository Structure

```
recursive-reality/
├── experiments/
│   ├── exp1_degradation/
│   │   ├── results/              ✅ 1,326 data points
│   │   └── scripts/              ✅ Complete analysis
│   ├── exp2_attribution/
│   │   ├── results/              ✅ 660 interpretations
│   │   ├── scripts/              ✅ Comprehensive analysis
│   │   └── data/                 ✅ 50 verified facts
│   ├── exp3_elaboration/
│   │   ├── scripts/              📋 Ready to run
│   │   └── README.md             📋 Full design
│   └── exp4_verbatim/
│       ├── scripts/              📋 Ready to run
│       └── README.md             📋 Full design
├── docs/
│   └── paper_draft.md            ✅ 5,200 words, ready
├── utils/
│   └── api_handlers.py           ✅ LLMHandler (async)
├── STATUS.md                     📍 This file
├── CURRENT_STATUS.md             📚 Detailed project status
├── EXPERIMENT_1_SUMMARY.md       📚 Exp 1 full results
└── README.md                     📚 Project overview
```

---

## Key Results Summary

### Overall Citation Loss
- **14.2%** exact preservation
- **85.6%** absent (lost completely)
- **0.2%** partial preservation
- **0%** hallucinated citations

### Model-Specific Performance

| Model | Exact Preservation | Text Expansion | Semantic Loss |
|-------|-------------------|----------------|---------------|
| Claude | 0.0% ❌ | 2.87× | -0.3% (NS) ✅ |
| GPT-4 | 15.0% ~ | 1.17× | -7.7% (NS) ✅ |
| Gemini | 24.2% ✅ | 2.41× | -22.1%* ❌ |

*p=0.006

### Critical Findings
1. **Dissociation demonstrated**: Semantic preservation ≠ source attribution
2. **Claude's catastrophe**: 100% citation loss despite perfect semantic preservation
3. **Gemini's paradox**: Best citation preservation despite worst semantic preservation
4. **Negative correlation**: More text = fewer citations (ρ = -0.20 to -0.23, p<0.003)

---

## To Resume Work (For Fresh Claude)

### Quick Start
1. Clone/access repository: https://github.com/HillaryDanan/recursive-reality
2. Read this STATUS.md file (you are here)
3. Check relevant experiment README for details
4. Run experiments: `python3 experiments/expN/scripts/run_*.py`

### For New Experiments
```bash
# Experiment 3 (elaboration causality)
cd /Users/hillarylevinson/Desktop/recursive-reality
python3 experiments/exp3_elaboration/scripts/run_elaboration.py

# Experiment 4 (verbatim mechanism)
cd /Users/hillarylevinson/Desktop/recursive-reality
python3 experiments/exp4_verbatim/scripts/run_verbatim.py
```

### For Analysis
```bash
# Analyze Experiment 2 (comprehensive)
python3 experiments/exp2_attribution/scripts/comprehensive_analysis.py

# View visualizations
open experiments/exp2_attribution/results/exp2_comprehensive_analysis.png
```

---

## Decision Point: Next Steps

### Option A: Submit Paper Now ✅ RECOMMENDED
- **Action**: Prepare submission to Nature Communications
- **Timeline**: Review takes 3-6 months
- **During review**: Run Exps 3 & 4 at leisure
- **Benefit**: Fastest path to publication

### Option B: Run Pilot First
- **Action**: Run Exps 3 & 4 with n=5 facts (~$10-15)
- **Timeline**: 1 day
- **Purpose**: Test feasibility before full runs
- **Benefit**: Risk mitigation

### Option C: Full Experiments First
- **Action**: Run complete Exps 3 & 4 (n=50, ~$100)
- **Timeline**: 2-3 days
- **Purpose**: Stronger paper before submission
- **Benefit**: More comprehensive initial submission

---

## API Configuration

### Models Used
- **GPT-4**: `gpt-4` (OpenAI)
- **Claude**: `claude-3-5-sonnet-20241022` (Anthropic)
- **Gemini**: `gemini-2.5-flash` (Google)

### Standard Parameters
- **Temperature**: 0.1 (reproducibility)
- **Max tokens**: 2000 (default)
- **Rate limiting**: 50 calls/minute
- **Retry logic**: 3 attempts with exponential backoff

---

## Important Notes

### Scientific Rigor
- All experiments pre-registered with hypotheses
- Temperature=0.1 for reproducibility across runs
- Multiple metrics for robustness (cosine similarity, text length, citation extraction)
- Report ALL findings (significant and non-significant)
- Cite only peer-reviewed sources

### Data Integrity
- Ground truth: 100 scientific facts (Exp 1), 50 with verified citations (Exp 2)
- All citations independently verified against original sources
- Missing data documented with completion rates
- No p-hacking, no selective reporting

### File Formats
- **Results**: JSON with full metadata
- **Analysis**: Python scripts with comprehensive statistics
- **Visualizations**: PNG at 300 DPI
- **Reports**: Markdown for version control

---

## Contact & Collaboration

- **Repository**: https://github.com/HillaryDanan/recursive-reality
- **Issues**: Open GitHub issue for questions
- **License**: MIT
- **Citation**: See README.md for BibTeX

---

## Version History

- **v1.0** (2025-09-30): Experiments 1 & 2 complete, paper draft ready
- **v0.9** (2025-09-29): Experiment 1 complete, Experiment 2 in progress
- **v0.5** (2025-09-28): Pilot studies, ground truth verified

---

**STATUS**: Ready for submission 🚀