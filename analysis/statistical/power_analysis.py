"""
Power Analysis for Recursive Reality Experiments
Based on Cohen (1988) and Faul et al. (2007) G*Power framework
"""
import numpy as np
from scipy import stats
import pingouin as pg
from typing import Tuple

def calculate_sample_size(effect_size: float = 0.5, 
                          alpha: float = 0.05, 
                          power: float = 0.8,
                          test_type: str = 'repeated_measures') -> int:
    """
    Calculate required sample size for specified statistical power.
    
    Based on: Faul, F., et al. (2007). G*Power 3: A flexible statistical 
    power analysis program. Behavior Research Methods, 39(2), 175-191.
    """
    if test_type == 'repeated_measures':
        # For repeated measures ANOVA with 3 models × 5 layers
        n = pg.power_rm_anova(eta=effect_size, m=5, n=None, 
                              power=power, alpha=alpha)
        return int(np.ceil(n))
    elif test_type == 'correlation':
        # For confidence-accuracy correlation analysis
        n = pg.power_corr(r=effect_size, n=None, 
                         power=power, alpha=alpha)
        return int(np.ceil(n))
    else:
        # Default to two-sample t-test
        n = pg.power_ttest(d=effect_size, n=None, 
                          power=power, alpha=alpha)
        return int(np.ceil(n))

def calculate_achieved_power(n: int, 
                            effect_size: float,
                            alpha: float = 0.05) -> float:
    """Calculate achieved statistical power for given sample size."""
    return pg.power_ttest(d=effect_size, n=n, 
                         power=None, alpha=alpha)

if __name__ == "__main__":
    print("RECURSIVE REALITY POWER ANALYSIS")
    print("="*50)
    
    # Experiment 1: Degradation
    n1 = calculate_sample_size(effect_size=0.5, test_type='repeated_measures')
    print(f"Exp 1 - Degradation Cascade: n={n1} topics needed")
    
    # Experiment 2: Attribution
    n2 = calculate_sample_size(effect_size=0.4, test_type='repeated_measures')
    print(f"Exp 2 - Source Attribution: n={n2} claims needed")
    
    # Experiment 3: Calibration
    n3 = calculate_sample_size(effect_size=0.3, test_type='correlation')
    print(f"Exp 3 - Confidence Calibration: n={n3} predictions needed")
    
    print(f"\nTotal API calls estimated: {(n1*5*3) + (n2*4*3*3) + (n3*3)}")
