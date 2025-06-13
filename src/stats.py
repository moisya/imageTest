# src/stats.py
import pandas as pd
from scipy import stats
import pingouin as pg
from typing import Dict

def run_statistical_analysis(df: pd.DataFrame, feature: str, analysis_type: str) -> Dict:
    if df.empty or feature not in df.columns: return {}
    try:
        if analysis_type == "グループ比較 (好き vs そうでもない)":
            group_like = df.loc[df['preference'] == '好き', feature].dropna()
            group_neutral = df.loc[df['preference'] == 'そうでもない', feature].dropna()
            if len(group_like) < 2 or len(group_neutral) < 2: return {}
            
            ttest_res = stats.ttest_ind(group_like, group_neutral, equal_var=False)
            effect_size = pg.compute_effsize(group_like, group_neutral, eftype='cohen')
            power = pg.power_ttest2n(len(group_like), len(group_neutral), d=effect_size)
            return {'p_value': ttest_res.pvalue, 'statistic': ttest_res.statistic, 'effect_size': effect_size, 'power': power}
        
        elif analysis_type == "相関分析 (ダミー連続値)":
            if 'dummy_valence' not in df.columns: return {}
            valid_df = df[[feature, 'dummy_valence']].dropna()
            if len(valid_df) < 2: return {}
            res = stats.linregress(valid_df[feature], valid_df['dummy_valence'])
            return {'corr_coef': res.rvalue, 'p_value': res.pvalue, 'slope': res.slope, 'intercept': res.intercept}
    except Exception: return {}
    return {}
