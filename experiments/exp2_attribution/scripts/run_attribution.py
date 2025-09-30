#!/usr/bin/env python3
"""
Experiment 2: Source Attribution Preservation Through Recursive Interpretation

Research Question: Do models preserve citations through recursive interpretation layers?
Hypothesis (Johnson & Raye, 1981): Attribution accuracy decreases with interpretation distance.

Author: Hillary Danan
Date: 2025-09-30
"""

import json
import os
import sys
import asyncio
from datetime import datetime
from typing import Dict, List, Optional
from loguru import logger

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from utils.api_handlers import LLMHandler

# Configure logging
log_dir = os.path.join(os.path.dirname(__file__), '../logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f'experiment_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")}.log')
logger.add(log_file, rotation="100 MB")


def load_facts(filepath: str) -> List[Dict]:
    """Load verified facts with citations."""
    logger.info(f"Loading facts from {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    facts = data['facts']
    logger.info(f"Loaded {len(facts)} verified facts")
    logger.info(f"Verification method: {data['metadata'].get('verification_method', 'unknown')}")
    
    return facts


def create_interpretation_prompt(text: str, layer: int) -> str:
    """
    Create prompt for interpretation layer.
    
    Critical: Prompt must NOT explicitly instruct to preserve citations,
    as we're testing natural preservation/loss patterns.
    """
    return f"""Please restate this scientific fact in your own words:

{text}"""


async def run_single_fact(
    fact: Dict, 
    model: str,
    api_handler: LLMHandler,
    temperature: float = 0.1
) -> Dict:
    """
    Pass one fact through 5 interpretation layers.
    
    Each layer interprets the previous layer's output.
    No explicit instruction to preserve citations - testing natural behavior.
    """
    
    logger.info(f"Processing fact {fact['id']}: {fact['citation_text']} with {model}")
    
    results = {
        'fact_id': fact['id'],
        'original_fact': fact['fact'],
        'original_citation': fact['citation_text'],
        'full_citation': fact['full_citation'],
        'category': fact['category'],
        'complexity': fact['complexity'],
        'citation_format': fact['citation_format'],
        'model': model,
        'temperature': temperature,
        'layers': []
    }
    
    # Layer 1: Interpret original fact
    current_text = fact['fact']
    
    for layer in range(1, 6):
        prompt = create_interpretation_prompt(current_text, layer)
        
        try:
            response = await api_handler.query(
                model=model,
                prompt=prompt,
                temperature=temperature,
                max_tokens=1000
            )
            
            results['layers'].append({
                'layer': layer,
                'text': response,
                'char_length': len(response),
                'prompt_used': prompt
            })
            
            # Next layer interprets this output
            current_text = response
            
            logger.debug(f"  Layer {layer}: {len(response)} chars")
            
        except Exception as e:
            logger.error(f"Error on layer {layer}: {e}")
            raise
    
    return results


async def run_experiment(
    facts_file: str,
    models: List[str],
    output_dir: str,
    temperature: float = 0.1,
    pilot: bool = False
):
    """
    Run Experiment 2: Source Attribution Preservation.
    
    Args:
        facts_file: Path to verified facts JSON
        models: List of model names to test
        output_dir: Directory to save results
        temperature: Sampling temperature (0.1 for reproducibility)
        pilot: If True, only run first 3 facts
    """
    
    # Initialize API handler
    api_handler = LLMHandler()
    
    # Load facts
    facts = load_facts(facts_file)
    
    if pilot:
        facts = facts[:3]
        logger.info(f"PILOT MODE: Using {len(facts)} facts")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Track overall stats
    experiment_metadata = {
        'experiment': 'exp2_attribution',
        'start_time': datetime.now().isoformat(),
        'facts_file': facts_file,
        'num_facts': len(facts),
        'models': models,
        'temperature': temperature,
        'pilot_mode': pilot,
        'total_planned_calls': len(facts) * len(models) * 5
    }
    
    # Run for each model
    for model in models:
        logger.info(f"\n{'='*70}")
        logger.info(f"Starting {model}")
        logger.info(f"{'='*70}")
        
        all_results = []
        errors = []
        
        for i, fact in enumerate(facts, 1):
            logger.info(f"Fact {i}/{len(facts)}: {fact['citation_text']}")
            
            try:
                result = await run_single_fact(fact, model, api_handler, temperature)
                all_results.append(result)
                
                # Save progress after each fact
                temp_file = os.path.join(output_dir, f'temp_{model}.json')
                with open(temp_file, 'w', encoding='utf-8') as f:
                    json.dump(all_results, f, indent=2, ensure_ascii=False)
                
            except Exception as e:
                logger.error(f"ERROR on fact {fact['id']}: {e}")
                errors.append({
                    'fact_id': fact['id'],
                    'error': str(e)
                })
                continue
        
        # Save final results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        mode = "pilot" if pilot else "full"
        output_file = os.path.join(
            output_dir, 
            f'attribution_{model}_{mode}_{timestamp}.json'
        )
        
        final_output = {
            'metadata': {
                **experiment_metadata,
                'model': model,
                'end_time': datetime.now().isoformat(),
                'facts_completed': len(all_results),
                'facts_failed': len(errors),
                'completion_rate': f"{len(all_results)/len(facts)*100:.1f}%"
            },
            'results': all_results,
            'errors': errors if errors else None
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n✅ Completed {model}")
        logger.info(f"   Results: {output_file}")
        logger.info(f"   Completed: {len(all_results)}/{len(facts)} facts")
        
        # Clean up temp file
        if os.path.exists(temp_file):
            os.remove(temp_file)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Experiment 2: Source Attribution Preservation'
    )
    parser.add_argument(
        '--pilot', 
        action='store_true', 
        help='Run pilot with 3 facts only'
    )
    parser.add_argument(
        '--models',
        nargs='+',
        default=["gpt-4", "claude-3-5-sonnet-20241022", "gemini-2.5-flash"],
        help='Models to test'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.1,
        help='Sampling temperature (default: 0.1 for reproducibility)'
    )
    
    args = parser.parse_args()
    
    # Configuration
    FACTS_FILE = "experiments/exp2_attribution/data/facts_with_citations_50_verified.json"
    OUTPUT_DIR = "experiments/exp2_attribution/results"
    
    num_facts = 3 if args.pilot else 50
    total_calls = num_facts * len(args.models) * 5
    
    logger.info("="*70)
    logger.info("EXPERIMENT 2: SOURCE ATTRIBUTION PRESERVATION")
    logger.info("="*70)
    logger.info(f"Research Question: Do models preserve citations through layers?")
    logger.info(f"Hypothesis: Attribution accuracy decreases with distance (Johnson & Raye, 1981)")
    logger.info("")
    logger.info(f"Mode: {'PILOT' if args.pilot else 'FULL'}")
    logger.info(f"Facts: {num_facts}")
    logger.info(f"Models: {', '.join(args.models)}")
    logger.info(f"Layers per fact: 5")
    logger.info(f"Temperature: {args.temperature}")
    logger.info(f"Total API calls: {total_calls}")
    logger.info(f"Estimated time: ~{total_calls * 4 / 60:.0f} minutes")
    logger.info("="*70)
    logger.info("")
    
    # Verify facts file exists
    if not os.path.exists(FACTS_FILE):
        logger.error(f"Facts file not found: {FACTS_FILE}")
        sys.exit(1)
    
    # Run experiment
    try:
        asyncio.run(run_experiment(
            facts_file=FACTS_FILE,
            models=args.models,
            output_dir=OUTPUT_DIR,
            temperature=args.temperature,
            pilot=args.pilot
        ))
        
        logger.info("")
        logger.info("="*70)
        logger.info("✅ EXPERIMENT COMPLETE")
        logger.info("="*70)
        logger.info(f"Results saved to: {OUTPUT_DIR}")
        logger.info("")
        logger.info("Next steps:")
        logger.info("1. Check results files in experiments/exp2_attribution/results/")
        logger.info("2. Run analysis: python3 experiments/exp2_attribution/scripts/analyze_attribution.py")
        logger.info("3. Git commit results")
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        raise


if __name__ == "__main__":
    main()