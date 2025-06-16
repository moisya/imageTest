# src/stats.py
import pandas as pd
from scipy import stats
import pingouin as pg
from typing import Dict, Optional

def run_statistical_analysis(df: pd.DataFrame, feature: str, analysis_type: str, target_col: Optional[str] = None) -> Dict:
    """統計解析を実行し、結果を返す。失敗した場合は空の辞書を返す。"""
    if df.empty or feature not in df.columns:
        return {}

    try:
        if analysis_type == "group":
            groups = [df.loc[df['preference'] == pref, feature].dropna() for pref in df['preference'].unique()]
            valid_groups = [g for g in groups if len(g) >= 2]
            
            if len(valid_groups) < 2: return {}
            
            if len(valid_groups) == 2:
                ttest_res = stats.ttest_ind(valid_groups[0], valid_groups[1], equal_var=False)
                return {'p_value': ttest_res.pvalue, 'statistic': ttest_res.statistic}
            else:
                f_val, p_val = stats.f_oneway(*valid_groups)
                return {'p_value': p_val, 'f_value': f_val}
        
        elif analysis_type == "correlation":
            if not target_col or target_col not in df.columns: return {}
            valid_df = df[[feature, target_col]].dropna()
            if len(valid_df) < 2: return {}
            
            res = stats.linregress(valid_df[feature], valid_df[target_col])
            
            return {
                'corr_coef': res.rvalue,
                'p_value': res.pvalue,
                'slope': res.slope,
                'intercept': res.intercept
            }
    except Exception:
        return {}
        
    return {}
