# Recursive Reality

**Information Transformation Through Digital Interpretation Layers**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Status: Active](https://img.shields.io/badge/status-active-success.svg)]()

---

## Overview

This repository explores how large language models transform information through recursive interpretation. Using the serial reproduction paradigm (Bartlett, 1932), we investigate whether LLMs preserve, degrade, or enhance information across multiple processing layers.

**Key Finding**: LLMs exhibit a dissociation between semantic preservation and source attribution - models can preserve meaning while destroying citations, or vice versa.

---

## Quick Start

```bash
# Clone repository
git clone https://github.com/HillaryDanan/recursive-reality.git
cd recursive-reality

# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env with your API keys

# View results
open experiments/exp2_attribution/results/exp2_comprehensive_analysis.png
```

---

## Current Status

**Phase**: Experiments 1 & 2 Complete, Paper Draft Ready

### Completed Experiments

**Experiment 1**: Information Transformation (n=1,326 interpretations)
- Three distinct processing strategies identified
- Claude: 2.67× text expansion, -0.3% semantic loss (NS)
- GPT-4: 1.17× expansion, -7.7% semantic loss (NS)
- Gemini: 1.01× expansion, -22.1% semantic loss (p=0.006)

**Experiment 2**: Citation Preservation (n=660 interpretations)
- 85.6% citation loss across all models
- Claude: 0% preservation (catastrophic failure)
- GPT-4: 15% preservation
- Gemini: 24.2% preservation (best)

### Key Results

| Model | Semantic Preservation | Citation Preservation | Finding |
|-------|----------------------|----------------------|---------|
| Claude | ✅ Perfect (-0.3% loss) | ❌ Total failure (0%) | **Dissociation** |
| GPT-4 | ✅ Strong (-7.7% loss) | ⚠️ Moderate (15%) | Balanced |
| Gemini | ❌ Poor (-22% loss) | ✅ Best (24.2%) | **Dissociation** |

**Statistical Validation**:
- Model differences: χ²=51.36, p<0.0001
- Layer effects: χ²=91.22, p<0.0001
- Text expansion negatively correlates with citation preservation: ρ=-0.20 to -0.23, p<0.003

---

## Repository Structure

```
recursive-reality/
├── experiments/
│   ├── exp1_degradation/     # Information transformation (COMPLETE)
│   ├── exp2_attribution/     # Citation preservation (COMPLETE)
│   ├── exp3_elaboration/     # Causality test (DESIGNED)
│   └── exp4_verbatim/        # Mechanism test (DESIGNED)
├── docs/
│   └── paper_draft.md        # 5,200 words, publication-ready
├── utils/
│   └── api_handlers.py       # Async LLM interface
├── STATUS.md                 # Current status (this file duplicates info)
└── README.md                 # Project overview (this file)
```

---

## Methodology

### Design
- **Paradigm**: Serial reproduction (Bartlett, 1932)
- **Procedure**: Pass facts through 5 recursive interpretation layers
- **Temperature**: 0.1 (reproducibility)
- **Models**: GPT-4, Claude-3.5-Sonnet, Gemini-2.5-Flash

### Metrics
- **Semantic**: Cosine similarity (sentence embeddings)
- **Attribution**: Citation extraction (regex + manual validation)
- **Statistics**: ANOVA, Chi-square, Spearman correlation

### Ground Truth
- 100 scientific facts (Experiment 1)
- 50 facts with verified citations (Experiment 2)
- All sources peer-reviewed and independently verified

---

## Key Findings

### 1. Three Processing Strategies

LLMs do not uniformly degrade information. Instead, they exhibit distinct transformation patterns:

- **Elaborator** (Claude): Expands text 2.67× while preserving semantics
- **Maintainer** (GPT-4): Minimal change with slight paraphrasing
- **Degrader** (Gemini): Classical information loss

This contradicts Shannon's information theory (1948), which predicts uniform exponential degradation.

### 2. Catastrophic Citation Loss

Overall citation preservation was only 14.2%, with 85.6% of attributions lost by layer 5. This occurred despite semantic content preservation in Claude and GPT-4.

### 3. Semantic-Attribution Dissociation

The double dissociation between Claude (semantic ✓ / attribution ✗) and Gemini (semantic ✗ / attribution ✓) provides evidence for independent processing mechanisms, analogous to human source monitoring dissociations (Johnson & Raye, 1981).

---

## Theoretical Contributions

**Challenges to Information Theory**
- Shannon (1948) models predict uniform degradation
- We observe heterogeneous transformation (elaboration, maintenance, degradation)
- Generative systems require new theoretical frameworks

**Support for Source Monitoring Framework**
- Johnson & Raye (1981) predict content/context dissociation
- Our results demonstrate this in computational systems
- Source information is more fragile than semantic content

**Implications for AI Systems**
- Multi-agent architectures need explicit citation tracking
- Text expansion ≠ information quality
- Model selection matters for task requirements

---

## Practical Applications

### Multi-Agent System Design

```
Source → [Extract Citations] → Content Pipeline → [Re-inject Citations] → Output
```

**Model Selection**:
- **GPT-4**: Balanced preservation, good for general use
- **Claude**: Explanation generation, **requires citation re-injection**
- **Gemini**: Content compression with better attribution

### Citation Integrity

85.6% citation loss demonstrates that source attribution cannot be assumed in AI-mediated information. Systems must implement explicit tracking mechanisms.

---

## Running Experiments

### Experiment 1 (Complete)
```bash
python3 experiments/exp1_degradation/scripts/run_degradation.py
python3 experiments/exp1_degradation/scripts/analyze_full_results.py
```

### Experiment 2 (Complete)
```bash
python3 experiments/exp2_attribution/scripts/run_attribution.py
python3 experiments/exp2_attribution/scripts/comprehensive_analysis.py
```

### Future Experiments (Designed)
```bash
# Experiment 3: Elaboration causality test
python3 experiments/exp3_elaboration/scripts/run_elaboration.py

# Experiment 4: Verbatim copying mechanism
python3 experiments/exp4_verbatim/scripts/run_verbatim.py
```

---

## Data Availability

All data, code, and analysis scripts are publicly available:
- **Raw data**: `experiments/*/results/*.json`
- **Processed data**: `experiments/*/results/*.csv`
- **Visualizations**: `experiments/*/results/*.png`
- **Analysis scripts**: `experiments/*/scripts/*.py`

---

## Publication

**Status**: Manuscript ready for submission

**Paper**: `docs/paper_draft.md` (5,200 words)  
**Target**: Nature Communications  
**Preprint**: Coming soon

---

## Requirements

```
Python 3.12+
numpy<2.0
pandas
scipy<1.14
scikit-learn
sentence-transformers
openai==1.35.7
anthropic
google-generativeai
loguru
python-dotenv
matplotlib
seaborn
```

Install: `pip install -r requirements.txt`

---

## Citation

If you use this dataset or methodology:

```bibtex
@misc{danan2025recursive,
  author = {Danan, Hillary},
  title = {Recursive Reality: Information Transformation Through Digital Interpretation Layers},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/HillaryDanan/recursive-reality}
}
```

---

## License

MIT License - See LICENSE file for details

---

## Contact

For questions or collaboration:
- Open an issue in this repository
- Repository: https://github.com/HillaryDanan/recursive-reality

---

## Acknowledgments

We thank the developers of GPT-4 (OpenAI), Claude-3.5-Sonnet (Anthropic), and Gemini-2.5-Flash (Google) for API access.

---

**Last Updated**: September 30, 2025  
**Status**: Active research, findings preliminary pending peer review