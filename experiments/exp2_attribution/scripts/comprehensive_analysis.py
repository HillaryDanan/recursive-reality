#!/usr/bin/env python3
"""
Comprehensive Statistical Analysis - Experiment 2
Citation Preservation Through Recursive Interpretation Layers
"""

import json
import re
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

class CitationAnalyzer:
    """Comprehensive citation preservation analysis"""
    
    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
        self.models = ['gpt-4', 'claude-3-5-sonnet-20241022', 'gemini-2.5-flash']
        self.data = self.load_all_data()
        
    def load_all_data(self) -> Dict:
        """Load all experimental results"""
        all_data = {}
        
        for model in self.models:
            # Find the full run file for this model
            pattern = f"attribution_{model}_full_*.json"
            files = list(self.results_dir.glob(pattern))
            
            if files:
                with open(files[0]) as f:
                    file_data = json.load(f)
                    # Extract the 'results' list from the dict
                    all_data[model] = file_data['results']
                print(f"✅ Loaded {model}: {len(all_data[model])} fact runs")
            else:
                print(f"❌ No data found for {model}")
                
        return all_data
    
    def extract_citations(self, text: str) -> List[str]:
        """Extract all citations from text using regex patterns"""
        if not text:
            return []
            
        patterns = [
            r'\([A-Z][a-z]+(?:\s+et\s+al\.)?,?\s+\d{4}\)',  # (Author, YYYY) or (Author et al., YYYY)
            r'[A-Z][a-z]+(?:\s+et\s+al\.)?\s+\(\d{4}\)',     # Author (YYYY) or Author et al. (YYYY)
            r'[A-Z][a-z]+\s+and\s+[A-Z][a-z]+\s+\(\d{4}\)', # Author and Author (YYYY)
            r'[A-Z][a-z]+\s+&\s+[A-Z][a-z]+\s+\(\d{4}\)',   # Author & Author (YYYY)
        ]
        
        citations = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            citations.extend(matches)
            
        return list(set(citations))  # Remove duplicates
    
    def categorize_citation(self, 
                          original_citations: List[str], 
                          layer_citations: List[str]) -> str:
        """Categorize citation as exact, partial, absent, or hallucinated"""
        
        if not layer_citations:
            return 'absent'
        
        # Check for exact matches
        for orig in original_citations:
            if orig in layer_citations:
                return 'exact'
        
        # Check for partial matches (same author or year)
        orig_authors = set()
        orig_years = set()
        for orig in original_citations:
            years = re.findall(r'\d{4}', orig)
            authors = re.findall(r'[A-Z][a-z]+', orig)
            orig_years.update(years)
            orig_authors.update(authors)
        
        layer_authors = set()
        layer_years = set()
        for cite in layer_citations:
            years = re.findall(r'\d{4}', cite)
            authors = re.findall(r'[A-Z][a-z]+', cite)
            layer_years.update(years)
            layer_authors.update(authors)
        
        # Partial match if any author or year matches
        if orig_authors & layer_authors or orig_years & layer_years:
            # But also check if there are NEW citations (hallucinations)
            if len(layer_citations) > len(original_citations):
                return 'partial+hallucinated'
            return 'partial'
        
        # If citations exist but don't match original at all
        return 'hallucinated'
    
    def analyze_preservation_rates(self) -> pd.DataFrame:
        """Calculate citation preservation rates by model and layer"""
        
        results = []
        
        for model, facts in self.data.items():
            model_short = model.split('-')[0] if model != 'gpt-4' else 'gpt-4'
            
            for fact_run in facts:
                fact_id = fact_run['fact_id']
                original_text = fact_run['original_fact']
                original_cites = self.extract_citations(original_text)
                
                if not original_cites:
                    continue  # Skip facts with no citations
                
                # Layers is a list, not a dict
                layers = fact_run['layers']
                
                for layer_data in layers:
                    layer_int = layer_data['layer']
                    layer_text = layer_data['text']
                    layer_cites = self.extract_citations(layer_text)
                    
                    category = self.categorize_citation(original_cites, layer_cites)
                    
                    # Calculate text length metrics
                    orig_len = len(original_text)
                    layer_len = len(layer_text)
                    expansion = (layer_len / orig_len) if orig_len > 0 else 0
                    
                    results.append({
                        'model': model_short,
                        'fact_id': fact_id,
                        'layer': layer_int,
                        'category': category,
                        'n_original_cites': len(original_cites),
                        'n_layer_cites': len(layer_cites),
                        'original_length': orig_len,
                        'layer_length': layer_len,
                        'expansion_factor': expansion,
                        'original_citations': '; '.join(original_cites),
                        'layer_citations': '; '.join(layer_cites)
                    })
        
        return pd.DataFrame(results)
    
    def calculate_statistics(self, df: pd.DataFrame) -> Dict:
        """Calculate comprehensive statistics"""
        
        stats_results = {
            'overall': {},
            'by_model': {},
            'by_layer': {},
            'statistical_tests': {}
        }
        
        # Overall preservation rates
        total = len(df)
        for cat in ['exact', 'partial', 'absent', 'hallucinated', 'partial+hallucinated']:
            count = (df['category'] == cat).sum()
            stats_results['overall'][cat] = {
                'count': int(count),
                'percentage': float(count / total * 100) if total > 0 else 0
            }
        
        # By model
        for model in df['model'].unique():
            model_df = df[df['model'] == model]
            model_total = len(model_df)
            
            stats_results['by_model'][model] = {
                'n_observations': int(model_total),
                'mean_expansion': float(model_df['expansion_factor'].mean()),
                'preservation_rates': {}
            }
            
            for cat in ['exact', 'partial', 'absent', 'hallucinated']:
                count = (model_df['category'] == cat).sum()
                stats_results['by_model'][model]['preservation_rates'][cat] = {
                    'count': int(count),
                    'percentage': float(count / model_total * 100) if model_total > 0 else 0
                }
        
        # By layer
        for layer in sorted(df['layer'].unique()):
            layer_df = df[df['layer'] == layer]
            layer_total = len(layer_df)
            
            stats_results['by_layer'][f'layer_{layer}'] = {
                'n_observations': int(layer_total),
                'preservation_rates': {}
            }
            
            for cat in ['exact', 'partial', 'absent', 'hallucinated']:
                count = (layer_df['category'] == cat).sum()
                stats_results['by_layer'][f'layer_{layer}']['preservation_rates'][cat] = {
                    'count': int(count),
                    'percentage': float(count / layer_total * 100) if layer_total > 0 else 0
                }
        
        # Statistical tests
        
        # 1. Chi-square: Citation preservation varies by model?
        contingency_models = pd.crosstab(df['model'], df['category'])
        chi2_models, p_models, dof_models, expected_models = stats.chi2_contingency(contingency_models)
        
        stats_results['statistical_tests']['model_differences'] = {
            'test': 'Chi-square',
            'chi2': float(chi2_models),
            'p_value': float(p_models),
            'df': int(dof_models),
            'significant': p_models < 0.05
        }
        
        # 2. Chi-square: Citation preservation varies by layer?
        contingency_layers = pd.crosstab(df['layer'], df['category'])
        chi2_layers, p_layers, dof_layers, expected_layers = stats.chi2_contingency(contingency_layers)
        
        stats_results['statistical_tests']['layer_effects'] = {
            'test': 'Chi-square',
            'chi2': float(chi2_layers),
            'p_value': float(p_layers),
            'df': int(dof_layers),
            'significant': p_layers < 0.001
        }
        
        # 3. Text expansion correlation with citation loss
        # Binary: citation preserved (exact or partial) vs lost (absent or hallucinated)
        df['citation_preserved'] = df['category'].isin(['exact', 'partial']).astype(int)
        
        for model in df['model'].unique():
            model_df = df[df['model'] == model]
            if len(model_df) > 3:  # Need sufficient data
                corr, p_corr = stats.spearmanr(
                    model_df['expansion_factor'], 
                    model_df['citation_preserved']
                )
                stats_results['statistical_tests'][f'{model}_expansion_preservation_corr'] = {
                    'test': 'Spearman correlation',
                    'correlation': float(corr),
                    'p_value': float(p_corr),
                    'significant': p_corr < 0.05
                }
        
        return stats_results
    
    def create_visualizations(self, df: pd.DataFrame, output_path: Path):
        """Create comprehensive visualization"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Experiment 2: Citation Preservation Analysis', 
                     fontsize=16, fontweight='bold', y=0.995)
        
        # 1. Overall preservation rates by model
        ax1 = axes[0, 0]
        model_cats = df.groupby(['model', 'category']).size().unstack(fill_value=0)
        model_cats_pct = model_cats.div(model_cats.sum(axis=1), axis=0) * 100
        
        model_cats_pct.plot(kind='bar', ax=ax1, stacked=True,
                           color=['#2ecc71', '#f39c12', '#e74c3c', '#9b59b6'])
        ax1.set_title('Citation Preservation by Model', fontweight='bold')
        ax1.set_xlabel('Model')
        ax1.set_ylabel('Percentage')
        ax1.legend(title='Category', bbox_to_anchor=(1.05, 1))
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
        
        # 2. Preservation by layer
        ax2 = axes[0, 1]
        layer_cats = df.groupby(['layer', 'category']).size().unstack(fill_value=0)
        layer_cats_pct = layer_cats.div(layer_cats.sum(axis=1), axis=0) * 100
        
        layer_cats_pct.plot(kind='line', ax=ax2, marker='o',
                           color=['#2ecc71', '#f39c12', '#e74c3c', '#9b59b6'])
        ax2.set_title('Citation Preservation Across Layers', fontweight='bold')
        ax2.set_xlabel('Layer')
        ax2.set_ylabel('Percentage')
        ax2.legend(title='Category')
        ax2.grid(True, alpha=0.3)
        
        # 3. Text expansion by model (compare with Exp 1)
        ax3 = axes[0, 2]
        expansion_by_model = df.groupby('model')['expansion_factor'].mean().sort_values()
        expansion_by_model.plot(kind='barh', ax=ax3, color='#3498db')
        ax3.set_title('Mean Text Expansion by Model', fontweight='bold')
        ax3.set_xlabel('Expansion Factor (Layer/Original Length)')
        ax3.axvline(x=1.0, color='red', linestyle='--', alpha=0.5, label='No change')
        ax3.legend()
        
        # 4. Exact citation preservation rate over layers by model
        ax4 = axes[1, 0]
        for model in df['model'].unique():
            model_df = df[df['model'] == model]
            exact_by_layer = model_df.groupby('layer').apply(
                lambda x: (x['category'] == 'exact').sum() / len(x) * 100
            )
            ax4.plot(exact_by_layer.index, exact_by_layer.values, 
                    marker='o', label=model, linewidth=2)
        
        ax4.set_title('Exact Citation Preservation Rate by Layer', fontweight='bold')
        ax4.set_xlabel('Layer')
        ax4.set_ylabel('Exact Preservation Rate (%)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Distribution of citations per response
        ax5 = axes[1, 1]
        df.boxplot(column='n_layer_cites', by='model', ax=ax5)
        ax5.set_title('Distribution of Citation Counts by Model', fontweight='bold')
        ax5.set_xlabel('Model')
        ax5.set_ylabel('Number of Citations')
        plt.suptitle('')  # Remove auto-title from boxplot
        
        # 6. Text expansion vs citation preservation
        ax6 = axes[1, 2]
        for model in df['model'].unique():
            model_df = df[df['model'] == model]
            preserved = model_df['category'].isin(['exact', 'partial'])
            ax6.scatter(model_df['expansion_factor'], preserved.astype(int),
                       alpha=0.3, label=model, s=20)
        
        ax6.set_title('Text Expansion vs Citation Preservation', fontweight='bold')
        ax6.set_xlabel('Expansion Factor')
        ax6.set_ylabel('Citation Preserved (1=Yes, 0=No)')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Visualization saved to {output_path}")
        
        return fig
    
    def generate_report(self, df: pd.DataFrame, stats: Dict, output_path: Path):
        """Generate comprehensive markdown report"""
        
        report = []
        report.append("# Experiment 2: Comprehensive Analysis Report")
        report.append(f"\n**Generated**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report.append("---\n")
        
        # Sample size
        report.append("## Sample Size\n")
        report.append(f"- **Total observations**: {len(df)}")
        report.append(f"- **Unique facts**: {df['fact_id'].nunique()}")
        report.append(f"- **Models**: {', '.join(df['model'].unique())}")
        report.append(f"- **Layers**: {sorted(df['layer'].unique())}\n")
        
        # Overall preservation rates
        report.append("## Overall Citation Preservation\n")
        report.append("| Category | Count | Percentage |")
        report.append("|----------|-------|------------|")
        for cat, data in stats['overall'].items():
            report.append(f"| {cat.capitalize()} | {data['count']} | {data['percentage']:.1f}% |")
        report.append("")
        
        # By model
        report.append("## Preservation Rates by Model\n")
        for model, data in stats['by_model'].items():
            report.append(f"### {model.upper()}\n")
            report.append(f"- **Observations**: {data['n_observations']}")
            report.append(f"- **Mean expansion**: {data['mean_expansion']:.2f}x\n")
            report.append("| Category | Count | Percentage |")
            report.append("|----------|-------|------------|")
            for cat, cat_data in data['preservation_rates'].items():
                report.append(f"| {cat.capitalize()} | {cat_data['count']} | {cat_data['percentage']:.1f}% |")
            report.append("")
        
        # Statistical tests
        report.append("## Statistical Tests\n")
        
        model_test = stats['statistical_tests']['model_differences']
        report.append(f"### Model Differences (Chi-square)")
        report.append(f"- χ² = {model_test['chi2']:.2f}")
        report.append(f"- p = {model_test['p_value']:.4f}")
        report.append(f"- df = {model_test['df']}")
        report.append(f"- **Significant**: {'Yes ✓' if model_test['significant'] else 'No'}\n")
        
        layer_test = stats['statistical_tests']['layer_effects']
        report.append(f"### Layer Effects (Chi-square)")
        report.append(f"- χ² = {layer_test['chi2']:.2f}")
        report.append(f"- p = {layer_test['p_value']:.4f}")
        report.append(f"- df = {layer_test['df']}")
        report.append(f"- **Significant**: {'Yes ✓' if layer_test['significant'] else 'No'}\n")
        
        # Expansion-preservation correlations
        report.append("### Text Expansion vs Citation Preservation Correlations\n")
        for key, test in stats['statistical_tests'].items():
            if 'expansion_preservation_corr' in key:
                model = key.split('_')[0]
                report.append(f"**{model.upper()}**:")
                report.append(f"- Spearman ρ = {test['correlation']:.3f}")
                report.append(f"- p = {test['p_value']:.4f}")
                report.append(f"- Significant: {'Yes ✓' if test['significant'] else 'No'}\n")
        
        # Key findings
        report.append("## Key Findings\n")
        
        # Find model with best preservation
        best_model = max(stats['by_model'].items(), 
                        key=lambda x: x[1]['preservation_rates']['exact']['percentage'])
        worst_model = min(stats['by_model'].items(),
                         key=lambda x: x[1]['preservation_rates']['exact']['percentage'])
        
        report.append(f"1. **Best citation preservation**: {best_model[0].upper()} "
                     f"({best_model[1]['preservation_rates']['exact']['percentage']:.1f}% exact)")
        report.append(f"2. **Worst citation preservation**: {worst_model[0].upper()} "
                     f"({worst_model[1]['preservation_rates']['exact']['percentage']:.1f}% exact)")
        
        # Text expansion comparison
        expansions = {m: d['mean_expansion'] for m, d in stats['by_model'].items()}
        max_exp = max(expansions.items(), key=lambda x: x[1])
        report.append(f"3. **Greatest text expansion**: {max_exp[0].upper()} ({max_exp[1]:.2f}x)")
        
        report.append(f"4. **Model differences**: "
                     f"{'Statistically significant' if model_test['significant'] else 'Not significant'} "
                     f"(χ² = {model_test['chi2']:.2f}, p = {model_test['p_value']:.4f})")
        
        report.append(f"5. **Layer effects**: "
                     f"{'Statistically significant' if layer_test['significant'] else 'Not significant'} "
                     f"(χ² = {layer_test['chi2']:.2f}, p = {layer_test['p_value']:.4f})")
        
        # Write report
        report_text = '\n'.join(report)
        with open(output_path, 'w') as f:
            f.write(report_text)
        
        print(f"✅ Report saved to {output_path}")
        
        return report_text


def main():
    """Run comprehensive analysis"""
    
    print("\n" + "="*60)
    print("EXPERIMENT 2: COMPREHENSIVE CITATION PRESERVATION ANALYSIS")
    print("="*60 + "\n")
    
    # Setup paths
    results_dir = Path("experiments/exp2_attribution/results")
    output_dir = results_dir
    
    # Initialize analyzer
    print("Loading data...")
    analyzer = CitationAnalyzer(results_dir)
    
    # Analyze preservation rates
    print("\nAnalyzing citation preservation patterns...")
    df = analyzer.analyze_preservation_rates()
    
    print(f"\n✅ Analyzed {len(df)} layer interpretations")
    print(f"   - {df['fact_id'].nunique()} unique facts")
    print(f"   - {df['model'].nunique()} models")
    print(f"   - {df['layer'].nunique()} layers per fact\n")
    
    # Calculate statistics
    print("Running statistical tests...")
    stats_results = analyzer.calculate_statistics(df)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    fig = analyzer.create_visualizations(
        df, 
        output_dir / "exp2_comprehensive_analysis.png"
    )
    
    # Generate report
    print("\nGenerating comprehensive report...")
    report = analyzer.generate_report(
        df,
        stats_results,
        output_dir / "exp2_comprehensive_report.md"
    )
    
    # Save detailed data
    print("\nSaving detailed results...")
    df.to_csv(output_dir / "exp2_detailed_data.csv", index=False)
    
    with open(output_dir / "exp2_statistical_results.json", 'w') as f:
        json.dump(stats_results, f, indent=2)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print(f"\nOutput files:")
    print(f"  - exp2_comprehensive_analysis.png (visualization)")
    print(f"  - exp2_comprehensive_report.md (markdown report)")
    print(f"  - exp2_detailed_data.csv (full dataset)")
    print(f"  - exp2_statistical_results.json (statistics)")
    print("\n")


if __name__ == "__main__":
    main()
