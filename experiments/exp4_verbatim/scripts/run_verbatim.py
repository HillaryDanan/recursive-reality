#!/usr/bin/env python3
"""
Experiment 4: Verbatim Copying vs. Semantic Paraphrasing - Runner Script
Tests whether citation preservation correlates with literal copying
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


class Experiment4Runner:
    """Run verbatim copying experiment with length constraints"""
    
    def __init__(self):
        self.api_handler = LLMHandler()
        self.results_dir = Path("experiments/exp4_verbatim/results")
        self.logs_dir = Path("experiments/exp4_verbatim/logs")
        self.data_dir = Path("experiments/exp2_attribution/data")
        
        # Create directories
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure logging
        logger.add(
            self.logs_dir / f"exp4_runner_{datetime.now():%Y%m%d_%H%M%S}.log",
            rotation="100 MB"
        )
        
        # Models to test
        self.models = [
            "gpt-4",
            "claude-3-5-sonnet-20241022",
            "gemini-2.5-flash"
        ]
        
        # Length conditions (max_tokens)
        self.length_conditions = {
            "short": 100,
            "medium": 200,
            "long": 400
        }
        
        # Base prompt
        self.prompt_template = "Please restate this scientific fact in your own words:\n\n{fact}"
    
    def load_facts(self) -> list:
        """Load verified facts from Experiment 2"""
        facts_file = self.data_dir / "facts_with_citations_50_verified.json"
        
        with open(facts_file) as f:
            data = json.load(f)
        
        logger.info(f"Loaded {len(data)} facts from Experiment 2")
        return data
    
    def longest_common_substring(self, text1: str, text2: str) -> str:
        """Find longest common substring using dynamic programming"""
        
        if not text1 or not text2:
            return ""
        
        m, n = len(text1), len(text2)
        
        # Create DP table
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        max_length = 0
        end_pos = 0
        
        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if text1[i-1] == text2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                    if dp[i][j] > max_length:
                        max_length = dp[i][j]
                        end_pos = i
        
        # Extract substring
        return text1[end_pos - max_length:end_pos]
    
    def calculate_verbatim_metrics(self, original: str, interpretation: str) -> dict:
        """Calculate verbatim overlap metrics"""
        
        if not interpretation:
            return {
                "lcs": "",
                "lcs_length": 0,
                "verbatim_ratio": 0.0,
                "char_overlap": 0
            }
        
        # Find longest common substring
        lcs = self.longest_common_substring(original.lower(), interpretation.lower())
        
        return {
            "lcs": lcs,
            "lcs_length": len(lcs),
            "verbatim_ratio": len(lcs) / len(original) if len(original) > 0 else 0,
            "char_overlap": len(lcs)
        }
    
    async def run_single_interpretation(
        self, 
        model: str, 
        prompt: str, 
        max_tokens: int,
        temperature: float = 0.1
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
        length_condition: str,
        n_layers: int = 5
    ) -> dict:
        """Run fact through recursive interpretation layers"""
        
        max_tokens = self.length_conditions[length_condition]
        
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
            "length_condition": length_condition,
            "max_tokens": max_tokens,
            "temperature": 0.1,
            "layers": []
        }
        
        # Layer 1: Original fact
        current_text = fact["fact"]
        
        for layer_num in range(1, n_layers + 1):
            logger.info(f"  Layer {layer_num}/5...")
            
            # Create prompt
            prompt = self.prompt_template.format(fact=current_text)
            
            # Get interpretation
            interpretation = await self.run_single_interpretation(
                model=model,
                prompt=prompt,
                max_tokens=max_tokens
            )
            
            if not interpretation["success"]:
                logger.warning(f"  Layer {layer_num} failed, stopping this fact")
                break
            
            # Calculate verbatim metrics
            verbatim_metrics = self.calculate_verbatim_metrics(
                original=fact["fact"],
                interpretation=interpretation["text"]
            )
            
            # Store layer result
            result["layers"].append({
                "layer": layer_num,
                "text": interpretation["text"],
                "char_length": interpretation["char_length"],
                "verbatim_lcs": verbatim_metrics["lcs"],
                "verbatim_length": verbatim_metrics["lcs_length"],
                "verbatim_ratio": verbatim_metrics["verbatim_ratio"],
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
        length_condition: str,
        facts: list
    ) -> list:
        """Run all facts for one model-condition combination"""
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting: {model} / {length_condition} (max_tokens={self.length_conditions[length_condition]})")
        logger.info(f"{'='*60}\n")
        
        results = []
        
        for i, fact in enumerate(facts, 1):
            logger.info(f"Fact {i}/{len(facts)}: {fact['id']}")
            
            result = await self.run_fact_through_layers(
                fact=fact,
                model=model,
                length_condition=length_condition
            )
            
            results.append(result)
            
            # Save progress after each fact
            self.save_progress(model, length_condition, results)
            
            logger.info(f"  ✓ Complete ({len(result['layers'])} layers)")
        
        return results
    
    def save_progress(self, model: str, length_condition: str, results: list):
        """Save intermediate results"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"verbatim_{model}_{length_condition}_{timestamp}.json"
        
        output = {
            "metadata": {
                "experiment": "exp4_verbatim",
                "model": model,
                "length_condition": length_condition,
                "max_tokens": self.length_conditions[length_condition],
                "timestamp": timestamp,
                "n_facts": len(results)
            },
            "results": results
        }
        
        output_path = self.results_dir / filename
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        logger.debug(f"Progress saved: {filename}")
    
    def save_final_results(self, model: str, length_condition: str, results: list):
        """Save final results with clean filename"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"verbatim_{model}_{length_condition}_full_{timestamp}.json"
        
        # Calculate average verbatim ratio
        all_ratios = []
        for r in results:
            for layer in r["layers"]:
                all_ratios.append(layer["verbatim_ratio"])
        
        avg_verbatim = sum(all_ratios) / len(all_ratios) if all_ratios else 0
        
        output = {
            "metadata": {
                "experiment": "exp4_verbatim",
                "model": model,
                "length_condition": length_condition,
                "max_tokens": self.length_conditions[length_condition],
                "timestamp": timestamp,
                "n_facts": len(results),
                "completion_rate": sum(1 for r in results if len(r['layers']) == 5) / len(results) * 100,
                "avg_verbatim_ratio": avg_verbatim
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
        logger.info(f"   Average verbatim ratio: {avg_verbatim:.3f}")
        return output_path
    
    async def run_full_experiment(self):
        """Run complete experiment: all models × all conditions × all facts"""
        
        # Load facts
        facts = self.load_facts()
        
        logger.info(f"\n{'='*60}")
        logger.info("EXPERIMENT 4: VERBATIM COPYING VS PARAPHRASING")
        logger.info(f"{'='*60}")
        logger.info(f"Models: {len(self.models)}")
        logger.info(f"Length conditions: {len(self.length_conditions)}")
        logger.info(f"Facts: {len(facts)}")
        logger.info(f"Total API calls: {len(self.models) * len(self.length_conditions) * len(facts) * 5}")
        logger.info(f"{'='*60}\n")
        
        # Run each model-condition combination
        for model in self.models:
            for length_condition in self.length_conditions.keys():
                try:
                    results = await self.run_model_condition(
                        model=model,
                        length_condition=length_condition,
                        facts=facts
                    )
                    
                    # Save final results
                    self.save_final_results(model, length_condition, results)
                    
                    logger.info(f"✓ Completed: {model} / {length_condition}")
                    
                except Exception as e:
                    logger.error(f"✗ Failed: {model} / {length_condition}")
                    logger.error(f"  Error: {str(e)}")
                    continue
        
        logger.info("\n" + "="*60)
        logger.info("EXPERIMENT 4 COMPLETE")
        logger.info("="*60 + "\n")


async def main():
    """Main entry point"""
    runner = Experiment4Runner()
    await runner.run_full_experiment()


if __name__ == "__main__":
    asyncio.run(main())