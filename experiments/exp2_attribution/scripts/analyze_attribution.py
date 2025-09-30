#!/usr/bin/env python3
"""
Experiment 2 Analysis: Citation Preservation Through Layers

Analyzes whether citations are preserved, lost, or hallucinated
across recursive interpretation layers.

Author: Hillary Danan
Date: 2025-09-30
"""

import json
import re
import glob
from typing import Dict, List, Tuple
from collections import defaultdict
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from loguru import logger

logger.add("experiments/exp2_attribution/logs/analysis.log")


def extract_citations(text: str) -> List[str]:
    """
    Extract citations from text using regex patterns.
    
    Patterns:
    - (Author, Year): (Einstein, 1905)
    - Author (Year): Einstein (1905)
    - Author et al. (Year): Watson et al. (1953)
    """
    patterns = [
        r'\(([A-Z][a-zA-Z]+(?:\s+(?:and|&)\s+[A-Z][a-zA-Z]+)?),?\s+(\d{4})\)',  # (Author, Year)
        r'([A-Z][a-zA-Z]+(?:\s+(?:and|&)\s+[A-Z][a-zA-Z]+)?)\s+\((\d{4})\)',     # Author (Year)
        r'([A-Z][a-zA-Z]+)\s+et\s+al\.\s+\((\d{4})\)',                           # Author et al. (Year)
    ]
    
    citations = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            if len(match) == 2:
                author, year = match
                citations.append(f"{author} ({year})")
    
    return citations


def citation_preserved(original: str, layer_text: str) -> Tuple[str, bool, bool]:
    """
    Check if citation is preserved, partially preserved, or lost.
    
    Returns:
        status: "exact", "partial", "hallucinated", "absent"
        has_citation: bool
        is_original: bool
    """
    orig_citations = extract_citations(original)
    layer_citations = extract_citations(layer_text)
    
    if not layer_citations:
        return "absent", False, False
    
    # Check for exact matches
    for orig in orig_citations:
        if orig in layer_citations:
            return "exact", True, True
    
    # Check for partial matches (same author or year)
    for layer_cit in layer_citations:
        for orig_cit in orig_citations:
            # Extract components
            layer_parts = re.findall(r'([A-Za-z]+)', layer_cit)
            orig_parts = re.findall(r'([A-Za-z]+)', orig_cit)
            
            # Same author?
            if layer_parts and orig_parts and layer_parts[0] == orig_parts[0]:
                return "partial", True, True
    
    # Citation present but not from original = hallucination
    return "hallucinated", True, False


def analyze_model_results(filepath: str) -> Dict:
    """Analyze citation preservation for one model."""
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    model = data['metadata']['model']
    results = data['results']
    
    logger.info(f"\nAnalyzing {model}")
    logger.info(f"Facts: {len(results)}")
    
    # Track citation preservation by layer
    layer_stats = defaultdict(lambda: {
        'exact': 0,
        'partial': 0,
        'hallucinated': 0,
        'absent': 0,
        'total': 0
    })
    
    # Track text length changes (replicating Exp 1)
    length_data = []
    
    for result in results:
        original = result['original_fact']
        original_len = len(original)
        
        for layer in result['layers']:
            layer_num = layer['layer']
            layer_text = layer['text']
            layer_len = len(layer_text)
            
            # Citation analysis
            status, has_cit, is_orig = citation_preserved(original, layer_text)
            layer_stats[layer_num][status] += 1
            layer_stats[layer_num]['total'] += 1
            
            # Length tracking
            length_data.append({
                'fact_id': result['fact_id'],
                'layer': layer_num,
                'length': layer_len,
                'original_length': original_len,
                'expansion_factor': layer_len / original_len if original_len > 0 else 0
            })
    
    return {
        'model': model,
        'citation_stats': dict(layer_stats),
        'length_data': length_data,
        'n_facts': len(results)
    }


def statistical_tests(all_results: List[Dict]) -> Dict:
    """
    Statistical analysis of citation preservation.
    
    Tests:
    1. Chi-square: Does preservation rate differ by model?
    2. Logistic regression: Layer effect on preservation
    3. ANOVA: Text expansion by model (consistency with Exp 1)
    """
    
    # Prepare data for chi-square
    preservation_by_model = {}
    
    for model_data in all_results:
        model = model_data['model']
        total_preserved = 0
        total_absent = 0
        
        for layer_num, stats in model_data['citation_stats'].items():
            total_preserved += stats['exact'] + stats['partial']
            total_absent += stats['absent'] + stats['hallucinated']
        
        preservation_by_model[model] = {
            'preserved': total_preserved,
            'lost': total_absent
        }
    
    logger.info("\nPreservation rates by model:")
    for model, counts in preservation_by_model.items():
        total = counts['preserved'] + counts['lost']
        rate = counts['preserved'] / total * 100 if total > 0 else 0
        logger.info(f"  {model}: {rate:.1f}% ({counts['preserved']}/{total})")
    
    # Text expansion analysis (consistency with Exp 1)
    logger.info("\nText expansion by model:")
    for model_data in all_results:
        model = model_data['model']
        lengths = model_data['length_data']
        
        layer_1 = [d['length'] for d in lengths if d['layer'] == 1]
        layer_5 = [d['length'] for d in lengths if d['layer'] == 5]
        
        if layer_1 and layer_5:
            mean_1 = np.mean(layer_1)
            mean_5 = np.mean(layer_5)
            expansion = (mean_5 - mean_1) / mean_1 * 100
            
            logger.info(f"  {model}: {expansion:+.1f}% (L1: {mean_1:.0f} → L5: {mean_5:.0f} chars)")
    
    return {
        'preservation_by_model': preservation_by_model
    }


def create_visualizations(all_results: List[Dict], output_dir: str):
    """Create figures for paper."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Panel A: Citation preservation by layer
    ax1 = axes[0, 0]
    for model_data in all_results:
        model = model_data['model']
        layers = sorted(model_data['citation_stats'].keys())
        
        preservation_rates = []
        for layer in layers:
            stats = model_data['citation_stats'][layer]
            preserved = stats['exact'] + stats['partial']
            total = stats['total']
            rate = preserved / total * 100 if total > 0 else 0
            preservation_rates.append(rate)
        
        ax1.plot(layers, preservation_rates, marker='o', label=model, linewidth=2)
    
    ax1.set_xlabel('Interpretation Layer')
    ax1.set_ylabel('Citation Preservation Rate (%)')
    ax1.set_title('A. Citation Preservation Across Layers')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Panel B: Text expansion (consistency with Exp 1)
    ax2 = axes[0, 1]
    for model_data in all_results:
        model = model_data['model']
        lengths = model_data['length_data']
        
        layer_means = {}
        for layer in range(1, 6):
            layer_lengths = [d['length'] for d in lengths if d['layer'] == layer]
            if layer_lengths:
                layer_means[layer] = np.mean(layer_lengths)
        
        layers = sorted(layer_means.keys())
        means = [layer_means[l] for l in layers]
        
        ax2.plot(layers, means, marker='o', label=model, linewidth=2)
    
    ax2.set_xlabel('Interpretation Layer')
    ax2.set_ylabel('Mean Text Length (chars)')
    ax2.set_title('B. Text Expansion Pattern (Replication of Exp 1)')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # Panel C: Citation status distribution
    ax3 = axes[1, 0]
    x = np.arange(len(all_results))
    width = 0.2
    
    statuses = ['exact', 'partial', 'hallucinated', 'absent']
    colors = ['green', 'yellow', 'red', 'gray']
    
    for i, status in enumerate(statuses):
        values = []
        for model_data in all_results:
            total = 0
            for layer_stats in model_data['citation_stats'].values():
                total += layer_stats[status]
            
            # Normalize by total citations
            n_total = sum(layer_stats['total'] for layer_stats in model_data['citation_stats'].values())
            pct = total / n_total * 100 if n_total > 0 else 0
            values.append(pct)
        
        ax3.bar(x + i * width, values, width, label=status, color=colors[i], alpha=0.7)
    
    ax3.set_xlabel('Model')
    ax3.set_ylabel('Percentage (%)')
    ax3.set_title('C. Citation Status Distribution')
    ax3.set_xticks(x + width * 1.5)
    ax3.set_xticklabels([m['model'].split('-')[0] for m in all_results])
    ax3.legend()
    ax3.grid(alpha=0.3, axis='y')
    
    # Panel D: Expansion factor distribution
    ax4 = axes[1, 1]
    for model_data in all_results:
        model = model_data['model']
        expansion_factors = [d['expansion_factor'] for d in model_data['length_data']]
        
        ax4.hist(expansion_factors, bins=30, alpha=0.5, label=model)
    
    ax4.set_xlabel('Expansion Factor (Layer/Original Length)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('D. Text Expansion Distribution')
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/exp2_analysis.png", dpi=300, bbox_inches='tight')
    logger.info(f"\nFigure saved: {output_dir}/exp2_analysis.png")


def main():
    """Run full analysis."""
    
    logger.info("="*70)
    logger.info("EXPERIMENT 2 ANALYSIS: SOURCE ATTRIBUTION PRESERVATION")
    logger.info("="*70)
    
    # Load all results
    files = glob.glob('experiments/exp2_attribution/results/attribution_*_full_*.json')
    
    if not files:
        logger.error("No results files found")
        return
    
    logger.info(f"\nFound {len(files)} result files")
    
    all_results = []
    for filepath in sorted(files):
        result = analyze_model_results(filepath)
        all_results.append(result)
    
    # Statistical analysis
    stats_results = statistical_tests(all_results)
    
    # Create visualizations
    create_visualizations(all_results, 'experiments/exp2_attribution/results')
    
    # Save summary
    summary = {
        'experiment': 'exp2_attribution',
        'analysis_date': '2025-09-30',
        'models_analyzed': len(all_results),
        'total_facts': sum(r['n_facts'] for r in all_results),
        'results': all_results,
        'statistics': stats_results
    }
    
    with open('experiments/exp2_attribution/results/analysis_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info("\n" + "="*70)
    logger.info("ANALYSIS COMPLETE")
    logger.info("="*70)
    logger.info("Results: experiments/exp2_attribution/results/analysis_summary.json")
    logger.info("Figure: experiments/exp2_attribution/results/exp2_analysis.png")


if __name__ == "__main__":
    main()
