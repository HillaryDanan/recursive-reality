# Recursive Reality Implementation Summary

## Project Overview
Testing the hypothesis that consuming information through multiple layers of human interpretation (vs direct empirical observation) affects epistemic practices and reality perception. Using LLMs as controlled experimental systems.

## Current Status
- ✅ Repository structure created
- ✅ Ground truth dataset: 100 scientific facts (5 categories, 3 complexity levels)
- ✅ API handlers for OpenAI, Anthropic, Google working
- ✅ Power analysis completed
- 🔄 Ready to run main experiments

## Repository Structure
```
recursive-reality/
├── experiments/
│   ├── exp1_degradation/     # Information degradation through layers
│   ├── exp2_attribution/     # Source attribution decay
│   └── exp3_calibration/     # Confidence miscalibration
├── utils/
│   ├── api_handlers.py       # Unified LLM API interface
│   ├── ground_truth_generator.py  # Scientific facts dataset
│   └── metrics.py            # Accuracy/similarity metrics
├── analysis/
│   ├── statistical/          # Power analysis, effect sizes
│   └── visualization/        # Plotting results
├── data/
│   └── ground_truth/         # 100 scientific facts CSV/JSON
└── logs/                     # API call history
```

## Key Components

### 1. API Handler (`utils/api_handlers.py`)
- Unified interface for GPT-4, Claude-3, Gemini-Pro
- Rate limiting, retry logic, error handling
- Consistent temperature (0.1) for reproducibility
- Call history logging for cost tracking

### 2. Ground Truth (`data/ground_truth/scientific_facts.csv`)
- 100 facts from peer-reviewed sources
- Categories: physics, biology, chemistry, mathematics, computer_science
- Complexity levels: simple, moderate, complex
- Each fact has canonical "ground_truth" version

### 3. Experiments Design
**Exp 1: Degradation Cascade**
- Pass facts through 5 interpretation layers
- Measure accuracy decay per layer
- n=100 facts × 3 models × 5 layers = 1,500 data points

**Exp 2: Source Attribution**
- Mix primary/secondary sources at ratios
- Test source identification accuracy
- n=50 claims × 4 ratios × 3 models = 600 data points

**Exp 3: Confidence Calibration**
- Force confidence scores on claims
- Measure confidence-accuracy correlation
- n=200 claims × 3 models = 600 calibration points

## Statistical Framework
- Power: 80% at α=0.05, effect size d=0.5
- Mixed-effects models for degradation curves
- Bayesian analysis for uncertainty quantification
- Inter-rater reliability (Krippendorff's α > 0.8)

## Key Findings Expected
Based on information theory (Shannon, 1948):
1. Exponential accuracy decay through layers
2. Source attribution failure after 3+ layers
3. Overconfidence on multiply-mediated information

## Running Experiments

```bash
# Setup environment
python3 -m pip install -r requirements.txt

# Configure API keys in .env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AIza...

# Test setup
python3 test_setup.py

# Run main experiments
python3 experiments/exp1_degradation/scripts/run_degradation.py

# Analyze results
python3 analysis/statistical/power_analysis.py
```

## Cost Estimates
- ~3,000 total API calls across experiments
- Estimated cost: $30-50 depending on model usage
- Rate limited to 50 calls/minute

## Next Steps
1. Run pilot study (n=10) to validate methodology
2. Execute full experiments if pilot successful
3. Statistical analysis with effect sizes
4. Generate publication-ready figures
5. Write up findings with proper citations

## Critical Files to Preserve
- `data/ground_truth/scientific_facts.csv` - Core dataset
- `utils/api_handlers.py` - API interface
- `experiments/exp1_degradation/scripts/run_degradation.py` - Main experiment
- `.env` - API keys (NEVER COMMIT)

## Dependencies (Python 3.12)
- numpy<2.0 (for PyMC compatibility)
- pandas, scipy<1.14, scikit-learn
- openai==1.35.7, anthropic, google-generativeai
- pingouin (power analysis), pymc (Bayesian)
- loguru, tenacity, python-dotenv

## Scientific Rigor Notes
- All facts from peer-reviewed sources with citations
- Temperature=0.1 for consistency across runs
- Stratified sampling by complexity
- Pre-registered hypotheses before data collection
- Transparent about LLM limitations as cognitive models

## Contact & Repository
- GitHub: https://github.com/HillaryDanan/recursive-reality
- Implementation uses bash/terminal on Mac
- VS Code for editing, python3 for execution
- All paths relative to repo root