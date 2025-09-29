#!/usr/bin/env python3
"""
Post-hoc power analysis for degradation experiment
Based on observed effect sizes from pilot
"""

import numpy as np
from scipy import stats
from statsmodels.stats.power import TTestPower

# Initialize power analysis object
power_analysis = TTestPower()

# From pilot results (you'll update these)
observed_effect_size = 0.5  # Update after running analysis
pilot_n = 10
full_n = 100

# Calculate power for pilot
pilot_power = power_analysis.solve_power(
    effect_size=observed_effect_size, 
    nobs=pilot_n, 
    alpha=0.05, 
    alternative='two-sided'
)
print(f"Pilot study power: {pilot_power:.2%}")

# Calculate power for full study
full_power = power_analysis.solve_power(
    effect_size=observed_effect_size, 
    nobs=full_n, 
    alpha=0.05, 
    alternative='two-sided'
)
print(f"Full study power: {full_power:.2%}")

# Minimum n for 80% power
min_n = power_analysis.solve_power(
    effect_size=observed_effect_size, 
    power=0.8, 
    alpha=0.05, 
    alternative='two-sided'
)
print(f"Minimum n for 80% power: {int(np.ceil(min_n))}")

if full_power > 0.8:
    print("\n✅ Full study adequately powered")
else:
    print(f"\n⚠️ Consider n={int(np.ceil(min_n))} for adequate power")
