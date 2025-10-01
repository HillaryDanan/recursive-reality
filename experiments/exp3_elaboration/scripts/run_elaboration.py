#!/usr/bin/env python3
"""
Experiment 3: Elaboration and Citation Loss - Runner Script
Tests whether explicit instructions can preserve citations during elaboration
"""

import asyncio
import json
from pathlib import Path
from datetime import datetime
from loguru import logger
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from utils.api_handlers import LLMHandler


class Experiment3Runner:
    """Run elaboration experiment with multiple prompt conditions"""
    
    def __init__(self):
        self.api_handler = LLMHandler()
        self.results_dir = Path("experiments/exp3_elaboration/results")
        self.logs_dir = Path("experiments/exp3_elaboration/logs")
        self.data_dir = Path("experiments/exp2_attribution/data")
        
        # Create directories
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure logging
        logger.add(
            self.logs_dir / f"exp3_runner_{datetime.now():%Y%m%d_%H%M%S}.log",
            rotation="100 MB"
        )
        
        # Models to test
        self.models = [
            "gpt-4",
            "claude-3-5-sonnet-20241022",
            "gemini-2.5-flash"
        ]
        
        # Prompt conditions
        self.conditions = {
            "baseline": "Please restate this scientific fact in your own words:\n\n{fact}",
            
            "preserve": "Please restate this scientific fact in your own words. IMPORTANT: Preserve all citations exactly as they appear in the original.\n\n{fact}",
            
            "elaborate": "Please explain this scientific fact in detail, providing additional context and elaboration. IMPORTANT: Preserve all citations exactly as they appear in the original.\n\n{fact}"
        }
    
    def load_facts(self) -> list:
        """Load verified facts from Experiment 2"""
        facts_file = self.data_dir / "facts_with_citations_50_verified.json"
        
        with open(facts_file) as f:
            data = json.load(f)
        
        logger.info(f"Loaded {len(data)} facts from Experiment 2")
        return data
    
    async def run_single_interpretation(
        self, 
        model: str, 
        prompt: str, 
        temperature: float = 0.1,
        max_tokens: int = 2000
    ) -> dict:
        """Run single interpretation through model"""
        
        try:
            response = await self.api_handler.query(
                model=model,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return {
                "text": response,
                "char_length": len(response),
                "success": True,
                "error": None
            }
            
        except Exception as e:
            logger.error(f"Error in interpretation: {str(e)}")
            return {
                "text": None,
                "char_length": 0,
                "success": False,
                "error": str(e)
            }
    
    async def run_fact_through_layers(
        self,
        fact: dict,
        model: str,
        condition: str,
        n_layers: int = 5
    ) -> dict:
        """Run fact through recursive interpretation layers"""
        
        prompt_template = self.conditions[condition]
        
        # Initialize result structure
        result = {
            "fact_id": fact["id"],
            "original_fact": fact["fact"],
            "original_citation": fact["citation"],
            "full_citation": fact["full_citation"],
            "category": fact["category"],
            "complexity": fact["complexity"],
            "citation_format": fact["citation_format"],
            "model": model,
            "condition": condition,
            "temperature": 0.1,
            "layers": []
        }
        
        # Layer 1: Original fact
        current_text = fact["fact"]
        
        for layer_num in range(1, n_layers + 1):
            logger.info(f"  Layer {layer_num}/5...")
            
            # Create prompt
            prompt = prompt_template.format(fact=current_text)
            
            # Get interpretation
            interpretation = await self.run_single_interpretation(
                model=model,
                prompt=prompt
            )
            
            if not interpretation["success"]:
                logger.warning(f"  Layer {layer_num} failed, stopping this fact")
                break
            
            # Store layer result
            result["layers"].append({
                "layer": layer_num,
                "text": interpretation["text"],
                "char_length": interpretation["char_length"],
                "prompt_used": prompt
            })
            
            # Update current text for next layer
            current_text = interpretation["text"]
            
            # Small delay to respect rate limits
            await asyncio.sleep(1)
        
        return result
    
    async def run_model_condition(
        self,
        model: str,
        condition: str,
        facts: list
    ) -> list:
        """Run all facts for one model-condition combination"""
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting: {model} / {condition}")
        logger.info(f"{'='*60}\n")
        
        results = []
        
        for i, fact in enumerate(facts, 1):
            logger.info(f"Fact {i}/{len(facts)}: {fact['id']}")
            
            result = await self.run_fact_through_layers(
                fact=fact,
                model=model,
                condition=condition
            )
            
            results.append(result)
            
            # Save progress after each fact
            self.save_progress(model, condition, results)
            
            logger.info(f"  ✓ Complete ({len(result['layers'])} layers)")
        
        return results
    
    def save_progress(self, model: str, condition: str, results: list):
        """Save intermediate results"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"elaboration_{model}_{condition}_{timestamp}.json"
        
        output = {
            "metadata": {
                "experiment": "exp3_elaboration",
                "model": model,
                "condition": condition,
                "timestamp": timestamp,
                "n_facts": len(results)
            },
            "results": results
        }
        
        output_path = self.results_dir / filename
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        logger.debug(f"Progress saved: {filename}")
    
    def save_final_results(self, model: str, condition: str, results: list):
        """Save final results with clean filename"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"elaboration_{model}_{condition}_full_{timestamp}.json"
        
        output = {
            "metadata": {
                "experiment": "exp3_elaboration",
                "model": model,
                "condition": condition,
                "timestamp": timestamp,
                "n_facts": len(results),
                "completion_rate": sum(1 for r in results if len(r['layers']) == 5) / len(results) * 100
            },
            "results": results,
            "errors": [
                {
                    "fact_id": r["fact_id"],
                    "layers_completed": len(r["layers"])
                }
                for r in results if len(r["layers"]) < 5
            ]
        }
        
        output_path = self.results_dir / filename
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"\n✅ Final results saved: {filename}")
        return output_path
    
    async def run_full_experiment(self):
        """Run complete experiment: all models × all conditions × all facts"""
        
        # Load facts
        facts = self.load_facts()
        
        logger.info(f"\n{'='*60}")
        logger.info("EXPERIMENT 3: ELABORATION AND CITATION LOSS")
        logger.info(f"{'='*60}")
        logger.info(f"Models: {len(self.models)}")
        logger.info(f"Conditions: {len(self.conditions)}")
        logger.info(f"Facts: {len(facts)}")
        logger.info(f"Total API calls: {len(self.models) * len(self.conditions) * len(facts) * 5}")
        logger.info(f"{'='*60}\n")
        
        # Run each model-condition combination
        for model in self.models:
            for condition in self.conditions.keys():
                try:
                    results = await self.run_model_condition(
                        model=model,
                        condition=condition,
                        facts=facts
                    )
                    
                    # Save final results
                    self.save_final_results(model, condition, results)
                    
                    logger.info(f"✓ Completed: {model} / {condition}")
                    
                except Exception as e:
                    logger.error(f"✗ Failed: {model} / {condition}")
                    logger.error(f"  Error: {str(e)}")
                    continue
        
        logger.info("\n" + "="*60)
        logger.info("EXPERIMENT 3 COMPLETE")
        logger.info("="*60 + "\n")


async def main():
    """Main entry point"""
    runner = Experiment3Runner()
    await runner.run_full_experiment()


if __name__ == "__main__":
    asyncio.run(main())