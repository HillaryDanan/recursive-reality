# Recursive Reality - Quick Reference

## Essential Commands
```bash
# Test all APIs
python3 utils/api_handlers.py

# Generate ground truth
python3 utils/ground_truth_generator.py  

# Run main experiment
python3 experiments/exp1_degradation/scripts/run_degradation.py

# Check logs
tail -f logs/api_calls.log

# Test entire setup
python3 test_setup.py
```

## Models We're Testing
- **OpenAI**: gpt-4
- **Anthropic**: claude-3-sonnet (claude-3-5-sonnet-20241022)
- **Google**: gemini-pro

## Key Parameters
- Temperature: 0.1 (low for consistency)
- Max tokens: 2000
- Rate limit: 50 calls/minute
- Retry attempts: 3 with exponential backoff

## File Locations
- API keys: `.env`
- Ground truth: `data/ground_truth/scientific_facts.csv`
- API logs: `logs/api_calls.log`
- Results: `experiments/*/results/`
- Call history: `logs/api_history.json`

## Troubleshooting
- **"No module named X"** → `python3 -m pip install -r requirements.txt`
- **"API not working"** → Check `.env` has valid keys
- **"Rate limit"** → Decrease calls_per_minute in APIConfig
- **"Out of credits"** → Check API provider dashboard
- **"Proxy error"** → Using openai==1.35.7 to avoid httpx issues
- **"Google model not found"** → Use 'gemini-pro' not 'gemini-1.5-pro'

## Git Commands
```bash
# Add all changes
git add -A

# Commit with message
git commit -m "message"

# Push to GitHub
git push origin main

# Check status
git status

# Pull latest changes
git pull origin main
```

## Environment Setup
```bash
# Create .env file
touch .env

# Add API keys (edit with VS Code)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AIza...

# NEVER commit .env!
```

## Running Experiments Workflow
1. **Test APIs first**: `python3 test_setup.py`
2. **Check ground truth**: `ls data/ground_truth/`
3. **Run pilot**: Modify n=10 in experiment script
4. **Check logs**: `tail logs/api_calls.log`
5. **Run full experiment**: Reset n=100
6. **Analyze results**: `python3 analysis/statistical/power_analysis.py`

## API Response Structure
```python
{
    'timestamp': '2025-09-29T...',
    'model': 'gpt-4',
    'prompt_length': 150,
    'response_length': 200,
    'duration': 2.3,
    'prompt_preview': '...',
    'response_preview': '...'
}
```

## Key Functions
```python
# Initialize handler
handler = LLMHandler()

# Query any model
response = await handler.query(
    model='gpt-4',  # or 'claude-3-sonnet', 'gemini-pro'
    prompt="Your prompt",
    temperature=0.1
)

# Test all APIs
results = await handler.test_all_apis()

# Save call history
handler.save_history()
```

## Cost Management
- Monitor with: `wc -l logs/api_calls.log`
- Rough cost: ~$0.01-0.02 per GPT-4 call
- Total budget: $30-50 for full experiment
- Emergency stop: Ctrl+C in terminal

Remember: 
- NEVER commit .env or API keys!
- Always check logs before full runs
- Keep temperature=0.1 for reproducibility
- Save results frequently