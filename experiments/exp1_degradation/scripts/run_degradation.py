#!/usr/bin/env python3
"""
Information Degradation Through Interpretation Layers
Testing Shannon's information theory (1948) on recursive interpretation
"""

import asyncio
import sys
import os
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
import argparse
from typing import Dict, List, Tuple
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from utils.api_handlers import LLMHandler
from utils.metrics import calculate_similarity, calculate_accuracy
from loguru import logger

# Configure logging
logger.add("experiments/exp1_degradation/logs/experiment_{time}.log", 
           rotation="100 MB", 
           level="INFO")

class DegradationExperiment:
    """
    Tests information degradation through recursive interpretation layers.
    Based on Shannon (1948) information theory and Bartlett (1932) serial reproduction.
    """
    
    def __init__(self, n_facts: int = 100, n_layers: int = 5, pilot: bool = False):
        self.n_facts = n_facts if not pilot else 10
        self.n_layers = n_layers
        self.pilot = pilot
        self.handler = LLMHandler()
        self.results = []
        
        logger.info(f"Initialized experiment: n_facts={self.n_facts}, n_layers={n_layers}, pilot={pilot}")
        
    async def load_ground_truth(self) -> pd.DataFrame:
        """Load scientific facts from ground truth dataset"""
        path = Path("data/ground_truth/scientific_facts.csv")
        df = pd.read_csv(path)
        
        if self.pilot:
            # Stratified sample for pilot: 2 from each category
            df = df.groupby('category').head(2)
            logger.info(f"Pilot mode: Using {len(df)} facts")
        
        return df
    
    async def run_single_cascade(self, fact: str, fact_id: int, model: str) -> List[Dict]:
        """
        Run a single fact through n_layers of interpretation
        Following serial reproduction paradigm (Bartlett, 1932)
        """
        cascade_results = []
        current_text = fact
        
        for layer in range(self.n_layers):
            prompt = f"""Please paraphrase the following scientific fact in your own words, 
            maintaining accuracy while expressing it differently:
            
            {current_text}
            
            Paraphrased version:"""
            
            try:
                # Query the model
                response = await self.handler.query(
                    model=model,
                    prompt=prompt,
                    temperature=0.1  # Low temp for consistency
                )
                
                # Store results for this layer
                cascade_results.append({
                    'fact_id': fact_id,
                    'model': model,
                    'layer': layer + 1,
                    'input_text': current_text,
                    'output_text': response,
                    'timestamp': datetime.now().isoformat()
                })
                
                # Use output as input for next layer
                current_text = response
                
                logger.debug(f"Fact {fact_id}, Model {model}, Layer {layer+1} complete")
                
            except Exception as e:
                logger.error(f"Error in cascade for fact {fact_id}, layer {layer}: {e}")
                break
                
        return cascade_results
    
    async def run_experiment(self):
        """Main experiment loop"""
        logger.info("Starting degradation experiment")
        
        # Load ground truth
        facts_df = await self.load_ground_truth()
        
        # Models to test
        models = ['gpt-4', 'claude-3-5-sonnet-20241022', 'gemini-2.5-flash']
        
        # Run cascades for each fact and model
        for idx, row in facts_df.iterrows():
            fact = row['fact']
            fact_id = row['fact_id']
            
            logger.info(f"Processing fact {fact_id}/{len(facts_df)}")
            
            for model in models:
                cascade_results = await self.run_single_cascade(fact, fact_id, model)
                self.results.extend(cascade_results)
                
                # Small delay to respect rate limits
                await asyncio.sleep(0.5)
        
        # Save results
        await self.save_results()
        
    async def save_results(self):
        """Save experimental results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save raw results as JSON
        results_dir = Path("experiments/exp1_degradation/results")
        results_dir.mkdir(exist_ok=True)
        
        output_file = results_dir / f"degradation_{'pilot' if self.pilot else 'full'}_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Results saved to {output_file}")
        
        # Create DataFrame for analysis
        df = pd.DataFrame(self.results)
        csv_file = results_dir / f"degradation_{'pilot' if self.pilot else 'full'}_{timestamp}.csv"
        df.to_csv(csv_file, index=False)
        
        logger.info(f"CSV saved to {csv_file}")
        
        # Quick summary statistics
        if len(df) > 0:
            logger.info(f"Total data points: {len(df)}")
            logger.info(f"Models tested: {df['model'].unique()}")
            logger.info(f"Layers completed: {df['layer'].max()}")

async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Run information degradation experiment')
    parser.add_argument('--pilot', action='store_true', help='Run pilot with n=10')
    parser.add_argument('--n_facts', type=int, default=100, help='Number of facts to test')
    parser.add_argument('--n_layers', type=int, default=5, help='Number of interpretation layers')
    
    args = parser.parse_args()
    
    # Run experiment
    experiment = DegradationExperiment(
        n_facts=args.n_facts,
        n_layers=args.n_layers,
        pilot=args.pilot
    )
    
    await experiment.run_experiment()
    
    logger.info("Experiment complete!")

if __name__ == "__main__":
    asyncio.run(main())
