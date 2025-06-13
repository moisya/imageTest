# src/stats.py
import pandas as pd
from scipy import stats
import pingouin as pg
from typing import Dict

def run_statistical_analysis(df: pd.DataFrame, feature: str, analysis_type: str) -> Dict:
    """統計解析を実行し、結果を返す。失敗した場合は空の辞書を返す。"""
    if df.empty or feature not in df.columns:
        return {}

    try:
        if analysis_type == "グループ比較 (好き vs そうでもない)":
            group_like = df.loc[df['preference'] == '好き', feature].dropna()
            group_neutral = df.loc[df['preference'] == 'そうでもない', feature].dropna()
            
            if len(group_like) < 2 or len(group_neutral) < 2:
                return {}

            ttest_res = stats.ttest_ind(group_like, group_neutral, equal_var=False)
            # 効果量の計算でデータが同一の場合などに備える
            if group_like.var() == 0 and group_neutral.var() == 0:
                effect_size = 0.0
            else:
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
            
            valid_df = df[[feature, 'dummy_valence']].dropna()
            if len(valid_df) < 2: return {}

            res = stats.linregress(valid_df[feature], valid_df['dummy_valence'])
            
            return {
                'corr_coef': res.rvalue,
                'p_value': res.pvalue,
                'slope': res.slope,
                'intercept': res.intercept
            }
    except Exception:
        # 予期せぬエラーが発生した場合も空辞書を返す
        return {}
        
    return {}
