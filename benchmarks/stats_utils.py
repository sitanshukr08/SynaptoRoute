import numpy as np
from scipy import stats

def calculate_statistics(group_a, group_b):
    """
    Calculates statistical significance between two arrays of latency/accuracy values.
    Returns t-statistic, p-value (Welch's t-test), U-statistic, p-value (Mann-Whitney U), and Cohen's d.
    """
    if not group_a or not group_b:
        return None
        
    group_a = np.array(group_a)
    group_b = np.array(group_b)
    
    # Welch's t-test (assumes unequal variances)
    t_stat, p_val_t = stats.ttest_ind(group_a, group_b, equal_var=False)
    
    # Mann-Whitney U test (non-parametric, robust to outliers)
    u_stat, p_val_u = stats.mannwhitneyu(group_a, group_b, alternative='two-sided')
    
    # Cohen's d effect size
    n1, n2 = len(group_a), len(group_b)
    var1, var2 = np.var(group_a, ddof=1), np.var(group_b, ddof=1)
    pooled_se = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_se == 0:
        cohens_d = 0.0
    else:
        cohens_d = (np.mean(group_a) - np.mean(group_b)) / pooled_se
        
    return {
        "welch_t_stat": t_stat,
        "welch_p_val": p_val_t,
        "mann_whitney_u": u_stat,
        "mann_whitney_p": p_val_u,
        "cohens_d": cohens_d,
        "is_significant": p_val_u < 0.05
    }

def print_statistics_report(stats_result, name_a="Group A", name_b="Group B", metric_name="Latency"):
    if not stats_result:
        return
        
    print(f"\n--- Statistical Significance ({name_a} vs {name_b} for {metric_name}) ---")
    print(f"Welch's t-test p-value:    {stats_result['welch_p_val']:.4f}")
    print(f"Mann-Whitney U p-value:    {stats_result['mann_whitney_p']:.4f}")
    print(f"Cohen's d Effect Size:     {stats_result['cohens_d']:.4f}")
    
    if stats_result['is_significant']:
        print("Conclusion: The difference IS statistically significant (p < 0.05).")
    else:
        print("Conclusion: The difference is NOT statistically significant (p >= 0.05).")
