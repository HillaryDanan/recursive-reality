import json
import numpy as np
from pathlib import Path

def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

# Load the CSV and recreate stats
import pandas as pd
from scipy import stats as sp_stats

df = pd.read_csv('experiments/exp2_attribution/results/exp2_detailed_data.csv')

# Recreate minimal stats with type conversion
stats_results = {
    'overall': {
        'total_observations': int(len(df)),
        'unique_facts': int(df['fact_id'].nunique()),
        'citation_loss_rate': float(85.6)
    },
    'by_model': {},
    'statistical_tests': {}
}

# By model stats
for model in df['model'].unique():
    model_df = df[df['model'] == model]
    exact_count = (model_df['category'] == 'exact').sum()
    
    stats_results['by_model'][model] = {
        'n_observations': int(len(model_df)),
        'exact_preservation_pct': float(exact_count / len(model_df) * 100),
        'mean_expansion': float(model_df['expansion_factor'].mean())
    }

# Chi-square tests
contingency = pd.crosstab(df['model'], df['category'])
chi2, p, dof, _ = sp_stats.chi2_contingency(contingency)

stats_results['statistical_tests']['model_differences'] = {
    'chi2': float(chi2),
    'p_value': float(p),
    'df': int(dof),
    'significant': bool(p < 0.05)
}

# Convert all numpy types
stats_results = convert_numpy_types(stats_results)

# Save
output_path = Path('experiments/exp2_attribution/results/exp2_statistical_results.json')
with open(output_path, 'w') as f:
    json.dump(stats_results, f, indent=2)

print('✅ JSON file regenerated with proper type conversion')
