#!/usr/bin/env python3
"""
Comprehensive analysis of full degradation experiment
Testing information theory predictions at scale
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import json
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import curve_fit
from utils.metrics import calculate_similarity
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

class FullAnalysis:
    def __init__(self, results_file):
        with open(results_file) as f:
            self.raw_data = json.load(f)
        self.df = pd.DataFrame(self.raw_data)
        self.ground_truth = pd.read_csv('data/ground_truth/scientific_facts.csv')
        logger.info(f"Loaded {len(self.df)} results for analysis")
        
    def comprehensive_analysis(self):
        """Run all analyses"""
        
        print("\n" + "="*70)
        print("FULL EXPERIMENT ANALYSIS - RECURSIVE REALITY")
        print("="*70)
        
        # 1. Overall statistics
        self.overall_statistics()
        
        # 2. Model-specific degradation
        self.model_degradation_analysis()
        
        # 3. Category effects
        self.category_analysis()
        
        # 4. Complexity effects
        self.complexity_analysis()
        
        # 5. Claude's expansion investigation
        self.investigate_claude_expansion()
        
        # 6. Statistical tests
        self.hypothesis_testing()
        
        # 7. Generate visualizations
        self.create_comprehensive_plots()
        
    def overall_statistics(self):
        print("\n1. OVERALL STATISTICS")
        print("-"*40)
        
        total_facts = len(self.df['fact_id'].unique())
        print(f"Facts analyzed: {total_facts}/100")
        
        for model in self.df['model'].unique():
            model_data = self.df[self.df['model'] == model]
            print(f"\n{model}:")
            print(f"  Data points: {len(model_data)}")
            print(f"  Unique facts: {len(model_data['fact_id'].unique())}")
            
            # Text length statistics
            for layer in range(1, 6):
                layer_data = model_data[model_data['layer'] == layer]
                if len(layer_data) > 0:
                    avg_len = layer_data['output_text'].str.len().mean()
                    print(f"  Layer {layer} avg length: {avg_len:.0f} chars")
    
    def model_degradation_analysis(self):
        print("\n2. DEGRADATION PATTERNS BY MODEL")
        print("-"*40)
        
        results = []
        
        for model in self.df['model'].unique():
            model_data = self.df[self.df['model'] == model]
            
            similarities = []
            for fact_id in model_data['fact_id'].unique():
                original = self.ground_truth[self.ground_truth['fact_id'] == fact_id]
                if len(original) == 0:
                    continue
                original_text = original.iloc[0]['fact']
                
                fact_data = model_data[model_data['fact_id'] == fact_id]
                
                for layer in range(1, 6):
                    layer_text = fact_data[fact_data['layer'] == layer]
                    if len(layer_text) > 0:
                        similarity = calculate_similarity(
                            layer_text.iloc[0]['output_text'], 
                            original_text, 
                            'cosine'
                        )
                        similarities.append({
                            'model': model,
                            'layer': layer,
                            'similarity': similarity,
                            'fact_id': fact_id
                        })
            
            sim_df = pd.DataFrame(similarities)
            if len(sim_df) > 0:
                # Calculate degradation
                layer1_mean = sim_df[sim_df['layer'] == 1]['similarity'].mean()
                layer5_mean = sim_df[sim_df['layer'] == 5]['similarity'].mean()
                degradation = (1 - layer5_mean/layer1_mean) * 100 if layer1_mean > 0 else 0
                
                # Effect size
                layer1_vals = sim_df[sim_df['layer'] == 1]['similarity'].values
                layer5_vals = sim_df[sim_df['layer'] == 5]['similarity'].values
                
                if len(layer1_vals) > 0 and len(layer5_vals) > 0:
                    cohen_d = (layer1_vals.mean() - layer5_vals.mean()) / \
                              np.sqrt((layer1_vals.var() + layer5_vals.var()) / 2)
                    
                    # T-test
                    t_stat, p_val = stats.ttest_ind(layer1_vals, layer5_vals)
                    
                    print(f"\n{model}:")
                    print(f"  Degradation: {degradation:.1f}%")
                    print(f"  Cohen's d: {cohen_d:.3f}")
                    print(f"  t-statistic: {t_stat:.3f}")
                    print(f"  p-value: {p_val:.6f}")
                    print(f"  Significant: {'YES' if p_val < 0.001 else 'NO'}")
    
    def category_analysis(self):
        print("\n3. CATEGORY-SPECIFIC EFFECTS")
        print("-"*40)
        
        # Merge with ground truth to get categories
        merged = self.df.merge(
            self.ground_truth[['fact_id', 'category', 'complexity']], 
            on='fact_id', 
            how='left'
        )
        
        for category in merged['category'].dropna().unique():
            cat_data = merged[merged['category'] == category]
            
            print(f"\n{category.upper()}:")
            for model in cat_data['model'].unique():
                model_cat_data = cat_data[cat_data['model'] == model]
                
                # Calculate average text expansion
                layer1_len = model_cat_data[model_cat_data['layer'] == 1]['output_text'].str.len().mean()
                layer5_len = model_cat_data[model_cat_data['layer'] == 5]['output_text'].str.len().mean()
                
                if layer1_len > 0:
                    expansion = (layer5_len - layer1_len) / layer1_len * 100
                    print(f"  {model}: {expansion:+.1f}% length change")
    
    def complexity_analysis(self):
        print("\n4. COMPLEXITY LEVEL EFFECTS")
        print("-"*40)
        
        merged = self.df.merge(
            self.ground_truth[['fact_id', 'complexity']], 
            on='fact_id', 
            how='left'
        )
        
        for complexity in ['simple', 'moderate', 'complex']:
            comp_data = merged[merged['complexity'] == complexity]
            if len(comp_data) == 0:
                continue
                
            print(f"\n{complexity.upper()} facts:")
            
            for model in comp_data['model'].unique():
                model_comp = comp_data[comp_data['model'] == model]
                
                # Text length change
                layer1 = model_comp[model_comp['layer'] == 1]['output_text'].str.len()
                layer5 = model_comp[model_comp['layer'] == 5]['output_text'].str.len()
                
                if len(layer1) > 0 and len(layer5) > 0:
                    change = (layer5.mean() - layer1.mean()) / layer1.mean() * 100
                    print(f"  {model}: {change:+.1f}% expansion")
    
    def investigate_claude_expansion(self):
        print("\n5. CLAUDE EXPANSION INVESTIGATION")
        print("-"*40)
        
        claude_data = self.df[self.df['model'] == 'claude-3-5-sonnet-20241022']
        
        # Sample some expansions for qualitative analysis
        print("\nExample of Claude's elaboration pattern:")
        print("-"*40)
        
        # Find a fact with complete cascade
        for fact_id in claude_data['fact_id'].unique():
            fact_cascade = claude_data[claude_data['fact_id'] == fact_id]
            if len(fact_cascade) == 5:  # Complete cascade
                original = self.ground_truth[self.ground_truth['fact_id'] == fact_id]
                if len(original) > 0:
                    print(f"\nFact ID {fact_id}:")
                    print(f"ORIGINAL ({len(original.iloc[0]['fact'])} chars):")
                    print(f"  '{original.iloc[0]['fact'][:100]}...'")
                    
                    layer5 = fact_cascade[fact_cascade['layer'] == 5].iloc[0]['output_text']
                    print(f"\nLAYER 5 ({len(layer5)} chars):")
                    print(f"  '{layer5[:200]}...'")
                    
                    print(f"\nExpansion factor: {len(layer5)/len(original.iloc[0]['fact']):.1f}x")
                    break
        
        # Statistical analysis of expansion
        expansions = []
        for fact_id in claude_data['fact_id'].unique():
            fact_data = claude_data[claude_data['fact_id'] == fact_id]
            layer1 = fact_data[fact_data['layer'] == 1]
            layer5 = fact_data[fact_data['layer'] == 5]
            
            if len(layer1) > 0 and len(layer5) > 0:
                expansion_factor = len(layer5.iloc[0]['output_text']) / len(layer1.iloc[0]['output_text'])
                expansions.append(expansion_factor)
        
        if expansions:
            print(f"\nClaude expansion statistics:")
            print(f"  Mean expansion factor: {np.mean(expansions):.2f}x")
            print(f"  Median: {np.median(expansions):.2f}x")
            print(f"  Std dev: {np.std(expansions):.2f}")
            print(f"  Max expansion: {np.max(expansions):.2f}x")
    
    def hypothesis_testing(self):
        print("\n6. HYPOTHESIS TESTING")
        print("-"*40)
        
        print("\nH0: Information remains constant through layers")
        print("H1: Information degrades through recursive interpretation")
        
        # ANOVA across layers for each model
        for model in self.df['model'].unique():
            model_data = self.df[self.df['model'] == model]
            
            # Group text lengths by layer
            groups = []
            for layer in range(1, 6):
                layer_lengths = model_data[model_data['layer'] == layer]['output_text'].str.len()
                if len(layer_lengths) > 0:
                    groups.append(layer_lengths.values)
            
            if len(groups) >= 2:
                f_stat, p_val = stats.f_oneway(*groups)
                print(f"\n{model} ANOVA:")
                print(f"  F-statistic: {f_stat:.3f}")
                print(f"  p-value: {p_val:.9f}")
                print(f"  Result: {'REJECT H0' if p_val < 0.001 else 'FAIL TO REJECT H0'}")
    
    def create_comprehensive_plots(self):
        print("\n7. GENERATING VISUALIZATIONS")
        print("-"*40)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Degradation curves by model
        ax = axes[0, 0]
        for model in self.df['model'].unique():
            model_data = self.df[self.df['model'] == model]
            mean_lengths = []
            for layer in range(1, 6):
                layer_mean = model_data[model_data['layer'] == layer]['output_text'].str.len().mean()
                mean_lengths.append(layer_mean)
            ax.plot(range(1, 6), mean_lengths, marker='o', label=model.split('-')[0])
        
        ax.set_xlabel('Layer')
        ax.set_ylabel('Average Text Length')
        ax.set_title('Text Length Evolution Through Layers')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Distribution of text lengths at layer 5
        ax = axes[0, 1]
        layer5_data = self.df[self.df['layer'] == 5]
        
        for model in self.df['model'].unique():
            model_layer5 = layer5_data[layer5_data['model'] == model]['output_text'].str.len()
            ax.hist(model_layer5, alpha=0.5, label=model.split('-')[0], bins=20)
        
        ax.set_xlabel('Text Length at Layer 5')
        ax.set_ylabel('Frequency')
        ax.set_title('Final Layer Text Length Distribution')
        ax.legend()
        
        # Plot 3: Category effects
        ax = axes[1, 0]
        merged = self.df.merge(self.ground_truth[['fact_id', 'category']], on='fact_id', how='left')
        
        category_effects = []
        for cat in merged['category'].dropna().unique():
            cat_data = merged[merged['category'] == cat]
            layer1 = cat_data[cat_data['layer'] == 1]['output_text'].str.len().mean()
            layer5 = cat_data[cat_data['layer'] == 5]['output_text'].str.len().mean()
            if layer1 > 0:
                change = (layer5 - layer1) / layer1 * 100
                category_effects.append({'category': cat, 'change': change})
        
        if category_effects:
            cat_df = pd.DataFrame(category_effects)
            cat_df.plot(x='category', y='change', kind='bar', ax=ax)
            ax.set_ylabel('% Change in Length')
            ax.set_title('Category-Specific Length Changes (Layer 1→5)')
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # Plot 4: Model comparison boxplot
        ax = axes[1, 1]
        expansion_data = []
        
        for model in self.df['model'].unique():
            model_data = self.df[self.df['model'] == model]
            for fact_id in model_data['fact_id'].unique():
                fact_data = model_data[model_data['fact_id'] == fact_id]
                layer1 = fact_data[fact_data['layer'] == 1]
                layer5 = fact_data[fact_data['layer'] == 5]
                
                if len(layer1) > 0 and len(layer5) > 0:
                    ratio = len(layer5.iloc[0]['output_text']) / len(layer1.iloc[0]['output_text'])
                    expansion_data.append({'model': model.split('-')[0], 'expansion': ratio})
        
        if expansion_data:
            exp_df = pd.DataFrame(expansion_data)
            exp_df.boxplot(column='expansion', by='model', ax=ax)
            ax.set_ylabel('Expansion Factor (Layer5/Layer1)')
            ax.set_title('Text Expansion Distribution by Model')
            ax.axhline(y=1, color='red', linestyle='--', alpha=0.5)
        
        plt.suptitle('Information Transformation Through Recursive Interpretation - Full Study', fontsize=16)
        plt.tight_layout()
        
        output_path = Path('experiments/exp1_degradation/results/full_analysis_plots.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plots saved to {output_path}")

if __name__ == "__main__":
    import sys
    results_file = sys.argv[1] if len(sys.argv) > 1 else \
                   'experiments/exp1_degradation/results/degradation_full_20250929_142704.json'
    
    analysis = FullAnalysis(results_file)
    analysis.comprehensive_analysis()
