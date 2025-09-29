#!/usr/bin/env python3
"""
Analyze information degradation through interpretation layers
Testing Shannon's (1948) information theory predictions
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
from utils.metrics import calculate_similarity, calculate_degradation_rate
from loguru import logger

# Set style for publication-quality figures
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300

class DegradationAnalysis:
    """
    Statistical analysis of information degradation
    Following Cohen (1988) for effect sizes and power analysis
    """
    
    def __init__(self, results_file: str):
        self.results_file = results_file
        self.load_data()
        
    def load_data(self):
        """Load experimental results and ground truth"""
        # Load cascade results
        with open(self.results_file) as f:
            self.raw_data = json.load(f)
        
        self.df = pd.DataFrame(self.raw_data)
        
        # Load ground truth for comparison
        self.ground_truth = pd.read_csv('data/ground_truth/scientific_facts.csv')
        
        logger.info(f"Loaded {len(self.df)} cascade results")
        
    def calculate_similarities(self):
        """
        Calculate similarity to original at each layer
        Using multiple metrics for robustness (Zhang et al., 2019)
        """
        results = []
        
        for fact_id in self.df['fact_id'].unique():
            # Get original fact
            original = self.ground_truth[self.ground_truth['fact_id'] == fact_id]['fact'].iloc[0]
            
            for model in self.df['model'].unique():
                model_data = self.df[(self.df['fact_id'] == fact_id) & 
                                     (self.df['model'] == model)]
                
                for layer in range(1, 6):
                    layer_data = model_data[model_data['layer'] == layer]
                    if not layer_data.empty:
                        text = layer_data.iloc[0]['output_text']
                        
                        # Calculate multiple similarity metrics
                        cosine_sim = calculate_similarity(text, original, 'cosine')
                        jaccard_sim = calculate_similarity(text, original, 'jaccard')
                        
                        results.append({
                            'fact_id': fact_id,
                            'model': model,
                            'layer': layer,
                            'cosine_similarity': cosine_sim,
                            'jaccard_similarity': jaccard_sim,
                            'text_length': len(text),
                            'original_length': len(original)
                        })
        
        self.similarity_df = pd.DataFrame(results)
        logger.info("Calculated similarities for all layers")
        
    def test_exponential_decay(self):
        """
        Test if degradation follows exponential decay
        H0: Linear decay, H1: Exponential decay (Shannon, 1948)
        """
        decay_results = []
        
        def exponential_decay(x, a, b):
            """y = a * exp(-b * x)"""
            return a * np.exp(-b * x)
        
        for model in self.similarity_df['model'].unique():
            model_data = self.similarity_df[self.similarity_df['model'] == model]
            
            # Average similarity per layer across all facts
            avg_by_layer = model_data.groupby('layer')['cosine_similarity'].mean()
            layers = avg_by_layer.index.values
            similarities = avg_by_layer.values
            
            try:
                # Fit exponential model
                popt, pcov = curve_fit(exponential_decay, layers, similarities, p0=[1, 0.1])
                
                # Calculate R-squared for exponential fit
                residuals = similarities - exponential_decay(layers, *popt)
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((similarities - np.mean(similarities))**2)
                r_squared_exp = 1 - (ss_res / ss_tot)
                
                # Compare with linear fit
                linear_coeffs = np.polyfit(layers, similarities, 1)
                linear_fit = np.polyval(linear_coeffs, layers)
                linear_residuals = similarities - linear_fit
                linear_ss_res = np.sum(linear_residuals**2)
                r_squared_linear = 1 - (linear_ss_res / ss_tot)
                
                # AIC for model comparison (Akaike, 1974)
                n = len(similarities)
                aic_exp = n * np.log(ss_res/n) + 4  # 2 parameters * 2
                aic_linear = n * np.log(linear_ss_res/n) + 4
                
                decay_results.append({
                    'model': model,
                    'decay_coefficient': popt[1],
                    'r_squared_exponential': r_squared_exp,
                    'r_squared_linear': r_squared_linear,
                    'aic_exponential': aic_exp,
                    'aic_linear': aic_linear,
                    'better_fit': 'exponential' if aic_exp < aic_linear else 'linear'
                })
                
                logger.info(f"{model}: Decay coefficient = {popt[1]:.4f}, "
                          f"R² exp = {r_squared_exp:.4f}, R² lin = {r_squared_linear:.4f}")
                
            except Exception as e:
                logger.error(f"Could not fit exponential for {model}: {e}")
        
        self.decay_df = pd.DataFrame(decay_results)
        
    def calculate_effect_sizes(self):
        """
        Calculate Cohen's d for degradation effect
        Following Cohen (1988) conventions: small=0.2, medium=0.5, large=0.8
        """
        effect_sizes = []
        
        for model in self.similarity_df['model'].unique():
            model_data = self.similarity_df[self.similarity_df['model'] == model]
            
            # Compare layer 1 vs layer 5
            layer1 = model_data[model_data['layer'] == 1]['cosine_similarity'].values
            layer5 = model_data[model_data['layer'] == 5]['cosine_similarity'].values
            
            if len(layer1) > 0 and len(layer5) > 0:
                # Cohen's d = (mean1 - mean2) / pooled_std
                mean_diff = np.mean(layer1) - np.mean(layer5)
                pooled_std = np.sqrt((np.var(layer1) + np.var(layer5)) / 2)
                cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
                
                # T-test for significance
                t_stat, p_value = stats.ttest_ind(layer1, layer5)
                
                effect_sizes.append({
                    'model': model,
                    'cohens_d': cohens_d,
                    'mean_layer1': np.mean(layer1),
                    'mean_layer5': np.mean(layer5),
                    'degradation': np.mean(layer1) - np.mean(layer5),
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'effect_size_interpretation': 
                        'large' if abs(cohens_d) > 0.8 else 
                        'medium' if abs(cohens_d) > 0.5 else 
                        'small' if abs(cohens_d) > 0.2 else 'negligible'
                })
        
        self.effect_sizes_df = pd.DataFrame(effect_sizes)
        
    def plot_degradation_curves(self):
        """Create publication-quality visualization"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        models = self.similarity_df['model'].unique()
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        for idx, (model, color) in enumerate(zip(models, colors)):
            ax = axes[idx]
            model_data = self.similarity_df[self.similarity_df['model'] == model]
            
            # Plot individual trajectories (faint)
            for fact_id in model_data['fact_id'].unique():
                fact_data = model_data[model_data['fact_id'] == fact_id]
                ax.plot(fact_data['layer'], fact_data['cosine_similarity'], 
                       alpha=0.2, color=color, linewidth=0.5)
            
            # Plot mean with confidence interval
            mean_by_layer = model_data.groupby('layer')['cosine_similarity'].agg(['mean', 'sem'])
            
            ax.plot(mean_by_layer.index, mean_by_layer['mean'], 
                   color=color, linewidth=2, marker='o', label='Mean')
            
            # 95% CI (1.96 * SEM)
            ax.fill_between(mean_by_layer.index,
                           mean_by_layer['mean'] - 1.96 * mean_by_layer['sem'],
                           mean_by_layer['mean'] + 1.96 * mean_by_layer['sem'],
                           alpha=0.3, color=color)
            
            # Add decay coefficient if available
            if not self.decay_df.empty:
                decay_info = self.decay_df[self.decay_df['model'] == model]
                if not decay_info.empty:
                    decay_coeff = decay_info.iloc[0]['decay_coefficient']
                    ax.text(0.05, 0.05, f'λ = {decay_coeff:.3f}',
                           transform=ax.transAxes, fontsize=10)
            
            ax.set_xlabel('Interpretation Layer', fontsize=12)
            ax.set_ylabel('Cosine Similarity to Original', fontsize=12)
            ax.set_title(model.split('-')[0].upper(), fontsize=14)
            ax.set_ylim([0, 1])
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Information Degradation Through Recursive Interpretation', 
                    fontsize=16, y=1.02)
        plt.tight_layout()
        
        # Save figure
        output_dir = Path('experiments/exp1_degradation/results')
        plt.savefig(output_dir / 'degradation_curves_pilot.png', 
                   bbox_inches='tight', dpi=300)
        #plt.show()
        
        logger.info("Saved degradation curves plot")
    
    def generate_report(self):
        """Generate scientific report of findings"""
        print("\n" + "="*70)
        print("INFORMATION DEGRADATION ANALYSIS - PILOT RESULTS")
        print("="*70)
        
        print("\n1. DESCRIPTIVE STATISTICS")
        print("-" * 40)
        summary = self.similarity_df.groupby(['model', 'layer'])['cosine_similarity'].agg(['mean', 'std', 'count'])
        print(summary)
        
        print("\n2. DEGRADATION PATTERNS")
        print("-" * 40)
        if not self.decay_df.empty:
            print(self.decay_df.to_string(index=False))
        
        print("\n3. EFFECT SIZES (Cohen's d)")
        print("-" * 40)
        if not self.effect_sizes_df.empty:
            for _, row in self.effect_sizes_df.iterrows():
                print(f"\n{row['model']}:")
                print(f"  Layer 1 mean: {row['mean_layer1']:.4f}")
                print(f"  Layer 5 mean: {row['mean_layer5']:.4f}")
                print(f"  Degradation: {row['degradation']:.4f}")
                print(f"  Cohen's d: {row['cohens_d']:.4f} ({row['effect_size_interpretation']})")
                print(f"  p-value: {row['p_value']:.4f} {'*' if row['significant'] else '(ns)'}")
        
        print("\n4. HYPOTHESIS TESTING")
        print("-" * 40)
        
        # Test if degradation is significant across all models
        layer1_all = self.similarity_df[self.similarity_df['layer'] == 1]['cosine_similarity']
        layer5_all = self.similarity_df[self.similarity_df['layer'] == 5]['cosine_similarity']
        
        if len(layer1_all) > 0 and len(layer5_all) > 0:
            t_stat, p_val = stats.ttest_ind(layer1_all, layer5_all)
            print(f"Overall degradation test (Layer 1 vs 5):")
            print(f"  t({len(layer1_all) + len(layer5_all) - 2}) = {t_stat:.4f}")
            print(f"  p = {p_val:.6f}")
            print(f"  Result: {'SIGNIFICANT' if p_val < 0.05 else 'NOT SIGNIFICANT'} at α=0.05")
        
        # Test exponential vs linear
        if not self.decay_df.empty:
            exp_better = sum(self.decay_df['better_fit'] == 'exponential')
            total = len(self.decay_df)
            print(f"\nModel comparison (AIC):")
            print(f"  Exponential better fit: {exp_better}/{total} models")
            print(f"  Supports Shannon (1948): {'YES' if exp_better > total/2 else 'NO'}")
        
        print("\n5. SCIENTIFIC INTERPRETATION")
        print("-" * 40)
        print("Based on pilot data (n=10 facts, 3 models, 5 layers):")
        
        if not self.effect_sizes_df.empty:
            mean_effect = self.effect_sizes_df['cohens_d'].mean()
            print(f"• Mean effect size: d = {mean_effect:.3f}")
            
            if mean_effect > 0.8:
                print("• LARGE degradation effect detected")
            elif mean_effect > 0.5:
                print("• MEDIUM degradation effect detected")
            elif mean_effect > 0.2:
                print("• SMALL degradation effect detected")
            else:
                print("• NEGLIGIBLE degradation effect")
        
        print("\nNOTE: These are pilot results. Full experiment (n=100) needed for")
        print("      robust conclusions. Current power may be insufficient.")
        
        print("\n" + "="*70)

    def run_analysis(self):
        """Execute complete analysis pipeline"""
        self.calculate_similarities()
        self.test_exponential_decay()
        self.calculate_effect_sizes()
        self.plot_degradation_curves()
        self.generate_report()
        
        # Save analysis results
        output_dir = Path('experiments/exp1_degradation/results')
        
        self.similarity_df.to_csv(output_dir / 'similarity_analysis_pilot.csv', index=False)
        self.decay_df.to_csv(output_dir / 'decay_analysis_pilot.csv', index=False)
        self.effect_sizes_df.to_csv(output_dir / 'effect_sizes_pilot.csv', index=False)
        
        logger.info("Analysis complete - results saved")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        results_file = sys.argv[1]
    else:
        # Find most recent results
        import glob
        json_files = sorted(glob.glob('experiments/exp1_degradation/results/*.json'))
        if json_files:
            results_file = json_files[-1]
        else:
            print("No results files found!")
            sys.exit(1)
    
    print(f"Analyzing: {results_file}")
    
    analysis = DegradationAnalysis(results_file)
    analysis.run_analysis()
