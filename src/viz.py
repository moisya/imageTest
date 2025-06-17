# src/viz.py

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Dict

from utils import AppConfig, TrialData
from preprocess import check_window_quality

# (plot_raw_signal_inspector, plot_signal_qc は変更なし)
# ...

def plot_feature_distribution(df: pd.DataFrame, feature: str):
    """特徴量のグループ間分布をボックスプロットで可視化（ホバー情報付き）"""
    fig = go.Figure()
    colors = {'好き': 'mediumseagreen', '嫌い': 'tomato', 'そうでもない': 'lightslategray'}
    
    for pref in df['preference'].unique():
        plot_df = df[df['preference'] == pref]
        
        # ★★★ ホバーテキストを作成 ★★★
        hover_texts = [
            f"Subject: {row['subject_id']}<br>Trial: {row['trial_id']}"
            for index, row in plot_df.iterrows()
        ]
        
        fig.add_trace(go.Box(
            y=plot_df[feature],
            name=pref,
            boxpoints='all',
            jitter=0.3,
            pointpos=-1.8,
            marker_color=colors.get(pref),
            hovertext=hover_texts, # ホバーテキストを設定
            hoverinfo='y+text' # Y値とカスタムテキストを表示
        ))
    
    fig.update_layout(
        title=f'"{feature}" の分布比較',
        yaxis_title="特徴量値",
        boxmode='group'
    )
    return fig

def plot_feature_correlation(df: pd.DataFrame, feature: str, target: str, stats_results: Dict):
    """特徴量と連続値の相関を散布図で可視化（ホバー情報付き）"""
    if target not in df.columns:
        fig = go.Figure().update_layout(title=f"ターゲット列 '{target}' が見つかりません")
        return fig
        
    plot_df = df[[feature, target, 'subject_id', 'trial_id']].dropna() # ID列も取得
    if plot_df.empty:
        fig = go.Figure().update_layout(title="表示できるデータがありません")
        return fig
    
    # ★★★ ホバーテンプレートを定義 ★★★
    hovertemplate = (
        "<b>Subject: %{customdata[0]}</b><br>" +
        "Trial: %{customdata[1]}<br>" +
        f"{target}: %{{x}}<br>" +
        f"{feature}: %{{y:.3f}}" +
        "<extra></extra>" # デフォルトのトレース情報を非表示にする
    )
    
    fig = go.Figure(data=go.Scatter(
        x=plot_df[target],
        y=plot_df[feature],
        mode='markers',
        marker=dict(size=10, opacity=0.7),
        customdata=plot_df[['subject_id', 'trial_id']], # カスタムデータを設定
        hovertemplate=hovertemplate # ホバーテンプレートを適用
    ))
    
    if 'slope' in stats_results and 'intercept' in stats_results:
        slope, intercept = stats_results.get('slope', 0), stats_results.get('intercept', 0)
        x_range = np.array([plot_df[target].min(), plot_df[target].max()])
        y_range = slope * x_range + intercept
        
        fig.add_trace(go.Scatter(
            x=x_range, y=y_range, mode='lines', name='回帰直線', 
            line=dict(color='black', dash='dash')
        ))

    fig.update_layout(
        title=f'"{feature}" と "{target}" の相関',
        xaxis_title=target,
        yaxis_title="特徴量値"
    )
    return fig
