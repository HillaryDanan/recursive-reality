# RECURSIVE REALITY - EXPERIMENT 2 BRIEFING FOR FRESH CLAUDE

## Project Context

**Repository**: https://github.com/HillaryDanan/recursive-reality  
**User**: Hillary Danan (Mac, bash, loves: echo commands, touch for files, VS Code, python3)  
**Communication Style**: Direct, no bullshit, profanity-friendly, scientifically rigorous

## Experiment 1 Complete (Background)

**Status**: COMPLETE - 1,326 data points  
**Key Finding**: Three distinct processing strategies (NOT uniform degradation)
- **Claude**: ELABORATOR (2.67x text expansion, +157%)
- **GPT-4**: MAINTAINER (+17% text, stable)
- **Gemini**: DEGRADER (-22% similarity, 69% completion)

**Significance**: Challenges Shannon (1948) - models transform information differently, not uniformly

## Experiment 2: Source Attribution Preservation

### Research Question
Do models preserve citations through recursive interpretation layers?

### Hypothesis
Attribution accuracy decreases with interpretation distance (Johnson & Raye, 1981)

### Method
- **50 scientific facts** with verified citations (20 classic + 30 recent)
- **5 interpretation layers** per fact
- **3 models**: gpt-4, claude-3-5-sonnet-20241022, gemini-2.5-flash
- **Temperature**: 0.1 (reproducibility)
- **Total**: 750 API calls

### Ground Truth
**File**: `experiments/exp2_attribution/data/facts_with_citations_50_verified.json`
- All 50 citations verified by Claude Sonnet 4.5
- Categories: physics, biology, chemistry, mathematics, computer_science
- Temporal range: 1687-2024
- Citation formats: narrative, parenthetical
- Encoding: UTF-8 (fixed from Latin-1)

### Current Status: PILOT COMPLETE

**Pilot Results** (3 facts × 3 models × 5 layers = 45 calls):
- ✅ GPT-4: Stable text (167-246 chars per layer)
- ✅ Claude: Expanding (330-1138 chars) - consistent with Exp 1
- ✅ Gemini: Variable (68-1092 chars), some safety rating errors but recovers

**Files Created**:
```
experiments/exp2_attribution/results/
├── attribution_gpt-4_pilot_20250930_141735.json
├── attribution_claude-3-5-sonnet-20241022_pilot_20250930_141825.json
└── attribution_gemini-2.5-flash_pilot_20250930_142218.json
```

### Critical Implementation Details

**API Handler**: `utils/api_handlers.py`
- Class: `LLMHandler` (NOT APIHandler, NOT get_model_response)
- All calls are async: `await api_handler.query(model, prompt, temperature, max_tokens)`
- Models work with these names:
  - `"gpt-4"` → OpenAI
  - `"claude-3-5-sonnet-20241022"` → Anthropic
  - `"gemini-2.5-flash"` → Google

**Runner Script**: `experiments/exp2_attribution/scripts/run_attribution.py`
- Uses `asyncio.run()` for async execution
- Saves progress after each fact (temp files)
- Full logging to `experiments/exp2_attribution/logs/`
- Handles errors gracefully, continues on failure

**Prompt Design** (CRITICAL):
```python
def create_interpretation_prompt(text: str, layer: int) -> str:
    return f"""Please restate this scientific fact in your own words:

{text}"""
```
**NO explicit "preserve citations" instruction** - testing natural behavior

### Next Steps

**1. Run Full Experiment** (~50 min)
```bash
cd /Users/hillarylevinson/Desktop/recursive-reality
python3 experiments/exp2_attribution/scripts/run_attribution.py
```

**2. Analysis** (Session 4)
- Extract citations from each layer
- Measure preservation: presence/absence, accuracy (exact/partial/hallucinated)
- Statistical tests (ANOVA, Chi-square)
- Compare to Experiment 1 patterns

**3. Key Predictions**
- Claude's expansion: May PRESERVE citations better (more verbose) OR OBSCURE them (buried in text)
- GPT-4: Likely maintains citations (stability pattern)
- Gemini: Likely loses citations fastest (degradation pattern)

## Scientific Rigor Checklist

All experiments follow:
- Pre-registered hypotheses
- Temperature = 0.1 for reproducibility
- Multiple metrics for robustness
- Report ALL findings (significant + non-significant)
- Cite only peer-reviewed sources
- Open data and code

## File Locations

```
recursive-reality/
├── experiments/
│   ├── exp1_degradation/          ✅ COMPLETE
│   │   └── results/               (1,326 data points)
│   ├── exp2_attribution/          🔄 PILOT COMPLETE
│   │   ├── data/
│   │   │   └── facts_with_citations_50_verified.json
│   │   ├── scripts/
│   │   │   ├── run_attribution.py (working, async)
│   │   │   └── verify_citations.py
│   │   ├── results/               (3 pilot JSON files)
│   │   └── logs/
├── utils/
│   └── api_handlers.py            (LLMHandler class)
├── EXPERIMENT_1_SUMMARY.md
├── CURRENT_STATUS.md
└── README.md
```

## Common Issues & Solutions

**ImportError: cannot import name 'get_model_response'**
- Use: `from utils.api_handlers import LLMHandler`
- NOT: `APIHandler` or `get_model_response`

**Gemini safety rating errors**
- Normal, API retries without generation_config
- Calls complete successfully after retry

**Encoding issues**
- Files use UTF-8
- Latin-1 characters (Â², Ã¼) already fixed

## Git Workflow

```bash
# Always in repo root
cd /Users/hillarylevinson/Desktop/recursive-reality

# Hillary's style
git add -A
git commit -m "Brief descriptive message"
git push origin main
echo "✅ Status update"
```

## Communication Guidelines

- Be direct, no bullshit
- Scientific rigor always
- Cite peer-reviewed sources
- Chimp-simple terminal commands
- Echo commands for confirmation
- Touch for file creation
- Python3 (not python)
- Honest about limitations
- No hallucinated numbers/data

## Key Theoretical Foundations

- **Shannon (1948)**: Information theory - degradation prediction
- **Johnson & Raye (1981)**: Source monitoring framework
- **Bartlett (1932)**: Serial reproduction paradigm
- **Craik & Tulving (1975)**: Elaborative encoding theory

## Success Metrics

- All 750 API calls complete
- Results in proper JSON format
- Citations extractable from text
- Statistical power ≥80% for attribution loss detection
- Reproducible (temperature=0.1, seeds logged)