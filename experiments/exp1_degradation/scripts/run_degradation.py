"""
Experiment 1: Information Degradation Through Interpretation Layers
Tests Shannon's (1948) information theory applied to LLM cascades
"""
import os
import json
import asyncio
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.api_handlers import LLMHandler
from utils.metrics import calculate_accuracy, semantic_similarity
from utils.data_loader import load_ground_truth
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DegradationExperiment:
    """
    Test information degradation through serial interpretation.
    Based on: Shannon, C. E. (1948). A mathematical theory of communication.
    """
    
    def __init__(self, models: List[str] = ['gpt-4', 'claude-3', 'gemini-pro']):
        self.models = models
        self.llm = LLMHandler()
        self.layers = 5  # Number of interpretation layers
        self.results = []
        
    def create_degradation_prompt(self, info: str, layer: int) -> str:
        """Generate prompts for each interpretation layer."""
        if layer == 0:
            return f"State this fact precisely: {info}"
        else:
            return f"Summarize this information in your own words: {info}"
    
    async def run_cascade(self, 
                         fact: str, 
                         ground_truth: str,
                         model: str) -> Dict:
        """Run information through interpretation cascade."""
        cascade_results = []
        current_info = fact
        
        for layer in range(self.layers):
            prompt = self.create_degradation_prompt(current_info, layer)
            response = await self.llm.query(model, prompt)
            
            # Measure accuracy against ground truth
            accuracy = calculate_accuracy(response, ground_truth)
            similarity = semantic_similarity(response, ground_truth)
            
            cascade_results.append({
                'layer': layer,
                'response': response,
                'accuracy': accuracy,
                'semantic_similarity': similarity,
                'length': len(response.split())
            })
            
            current_info = response  # Feed forward to next layer
        
        return {
            'model': model,
            'fact': fact,
            'ground_truth': ground_truth,
            'cascade': cascade_results,
            'timestamp': datetime.now().isoformat()
        }
    
    async def run_experiment(self, facts_df: pd.DataFrame):
        """Run full experiment across all models and facts."""
        logger.info(f"Starting degradation experiment with {len(facts_df)} facts")
        
        for idx, row in facts_df.iterrows():
            for model in self.models:
                result = await self.run_cascade(
                    row['fact'], 
                    row['ground_truth'],
                    model
                )
                self.results.append(result)
                
                if idx % 10 == 0:
                    logger.info(f"Processed {idx}/{len(facts_df)} facts")
        
        # Save results
        output_path = Path('../data/exp1_results.json')
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")
        return self.results

if __name__ == "__main__":
    # Load ground truth facts
    facts = pd.read_csv('../../../data/ground_truth/scientific_facts.csv')
    
    # Run experiment
    exp = DegradationExperiment()
    asyncio.run(exp.run_experiment(facts))
