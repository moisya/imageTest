# src/stats.py
import pandas as pd
from scipy import stats
import pingouin as pg
from typing import Dict

def run_statistical_analysis(df: pd.DataFrame, feature: str, analysis_type: str) -> Dict:
    """統計解析を実行し、結果を返す"""
    if df.empty or feature not in df.columns:
        return {}

    if analysis_type == "グループ比較 (好き vs そうでもない)":
        group_like = df[df['preference'] == '好き'][feature].dropna()
        group_neutral = df[df['preference'] == 'そうでもない'][feature].dropna()
        
        if len(group_like) < 2 or len(group_neutral) < 2: return {}

        ttest_res = stats.ttest_ind(group_like, group_neutral, equal_var=False) # Welch's t-test
        effect_size = pg.compute_effsize(group_like, group_neutral, eftype='cohen')
        power = pg.power_ttest2n(len(group_like), len(group_neutral), d=effect_size)

        return {
            'p_value': ttest_res.pvalue,
            'statistic': ttest_res.statistic,
            'effect_size': effect_size,
            'power': power
        }
    
    elif analysis_type == "相関分析 (ダミー連続値)":
        if 'dummy_valence' not in df.columns: return {}
        
        corr_res = stats.pearsonr(df[feature].dropna(), df['dummy_valence'].dropna())
        
        return {
            'corr_coef': corr_res[0],
            'p_value': corr_res[1]
        }
    return {}
