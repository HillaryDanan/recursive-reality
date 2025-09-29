# Recursive Reality

## Research Question

How does increasing reliance on digitally-mediated information affect human epistemic practices and reality perception? This repository explores the hypothesis that consuming multiple layers of human interpretation rather than direct empirical observation may alter how people construct models of reality.

## Current Status

### Pilot Experiment Complete (September 29, 2025)

We have completed initial pilot testing of information transformation through recursive interpretation using Large Language Models (LLMs) as experimental systems. Preliminary findings suggest model-specific processing strategies that diverge from predictions based on Shannon's information theory (1948).

## Theoretical Framework

**Traditional Empirical Process**
```
Physical Reality → Direct Observation → Hypothesis → Testing → Knowledge
```

**Digital Information Flow**
```
Physical Reality → Observer's Interpretation → Media Framing → Social Distribution → Individual Reception
```

We hypothesize that each interpretive layer may introduce transformations, with effects varying by processing system. Initial pilot data suggests these transformations may involve both information loss and information generation.

## Preliminary Findings

### Experiment 1: Information Transformation Through Layers (Pilot Phase)

**Method**: 100 scientific facts passed through 5 interpretation layers using three LLMs (GPT-4, Claude-3.5, Gemini-2.5)

**Initial Observations**:
- Models exhibit distinct processing strategies rather than uniform degradation
- Some systems appear to elaborate information while others compress it
- Effects vary by content category and complexity

**Data Available**: 
- Pilot dataset: 1,326 data points (88.4% completion)
- Analysis scripts and visualizations in `/experiments/exp1_degradation/`

*Note: These are preliminary findings from pilot testing. Full experimental validation and peer review are required before drawing definitive conclusions.*

## Research Program

### Completed
- ✅ Experiment 1 pilot: Serial reproduction through interpretation layers
- ✅ Ground truth dataset: 100 scientific facts with peer-reviewed sources
- ✅ Statistical framework and power analysis

### In Progress
- 🔄 Semantic analysis of transformation patterns
- 🔄 Validation of pilot findings with expanded sample

### Planned Experiments

#### Experiment 2: Source Attribution Preservation
- Research question: How does source attribution degrade through interpretation layers?
- Hypothesis: Attribution accuracy decreases with each layer (Johnson & Raye, 1981)
- Method: Track citation preservation through recursive processing

#### Experiment 3: Confidence Calibration
- Research question: How does confidence calibration change through mediated information?
- Hypothesis: Overconfidence increases with interpretation distance (Lichtenstein et al., 1982)
- Method: Force confidence ratings at each interpretation layer

#### Experiment 4: Semantic Coherence Analysis
- Research question: When information expands through interpretation, is meaning preserved or distorted?
- Hypothesis: Elaboration may introduce semantic drift (Bartlett, 1932)
- Method: Deep semantic analysis of expansions vs. original content

## Methodological Approach

### Design Principles
- Serial reproduction paradigm (Bartlett, 1932)
- Controlled temperature settings (0.1) for reproducibility
- Multiple similarity metrics for robustness
- Stratified sampling across complexity levels
- Pre-registered hypotheses before data collection

### Statistical Framework
- Power analysis targeting 80% at α=0.05
- Effect size calculations (Cohen's d)
- Mixed-effects models for nested data
- Multiple comparison corrections where appropriate

## Repository Structure

```
recursive-reality/
├── experiments/
│   ├── exp1_degradation/     # Information transformation study
│   ├── exp2_attribution/     # Source attribution (planned)
│   ├── exp3_calibration/     # Confidence calibration (planned)
│   └── exp4_coherence/       # Semantic coherence (planned)
├── data/
│   └── ground_truth/         # Validated scientific facts
├── analysis/
│   ├── statistical/          # Statistical analysis scripts
│   └── visualization/        # Data visualization
├── utils/
│   ├── api_handlers.py       # LLM interface utilities
│   └── metrics.py            # Evaluation metrics
└── theory/
    └── THEORETICAL_FOUNDATION.md
```

## Technical Requirements

- Python 3.12+
- See `requirements.txt` for dependencies
- API keys required for LLM access (OpenAI, Anthropic, Google)

## Usage

```bash
# Clone repository
git clone https://github.com/HillaryDanan/recursive-reality.git
cd recursive-reality

# Install dependencies
pip install -r requirements.txt

# Configure API keys in .env file
cp .env.example .env
# Edit .env with your API keys

# Run pilot analysis
python experiments/exp1_degradation/scripts/analyze_degradation.py
```

## Theoretical Contributions

This research aims to contribute to:
- Information theory and communication
- Digital epistemology
- Cognitive models of information processing
- AI system behavior and interpretability

## Limitations and Scope

- Current findings are preliminary and based on pilot data
- LLMs serve as model systems; human cognition may differ substantially
- Results require replication and peer review
- Focus is on understanding patterns, not prescribing behaviors

## Data Availability

All pilot data and analysis code are publicly available in this repository. We encourage replication and extension of these preliminary findings.

## Contributing

We welcome collaboration from researchers interested in:
- Information theory and transmission
- Cognitive psychology and source monitoring
- AI interpretability and behavior
- Digital epistemology and media effects

Please see CONTRIBUTING.md for guidelines.

## Ethics and Transparency

- All experiments use publicly available models
- No human subjects involved in current phase
- Temperature and parameters documented for reproducibility
- Both expected and unexpected findings reported

## Citation

If you use this dataset or methodology, please cite:
```
@misc{recursive_reality_2025,
  author = {Danan, Hillary},
  title = {Recursive Reality: Information Transformation Through Digital Interpretation Layers},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/HillaryDanan/recursive-reality}
}
```

## License

MIT License - See LICENSE file for details

## Contact

For questions about this research, please open an issue in this repository.

---

*This is an active research project in pilot phase. All findings are preliminary and require validation through peer review before drawing firm conclusions.*

## References

- Bartlett, F. C. (1932). Remembering: A study in experimental and social psychology. Cambridge University Press.
- Johnson, M. K., & Raye, C. L. (1981). Reality monitoring. Psychological Review, 88(1), 67-85.
- Lichtenstein, S., Fischhoff, B., & Phillips, L. D. (1982). Calibration of probabilities. In D. Kahneman, P. Slovic, & A. Tversky (Eds.), Judgment under uncertainty (pp. 306-334).
- Shannon, C. E. (1948). A mathematical theory of communication. Bell System Technical Journal, 27(3), 379-423.