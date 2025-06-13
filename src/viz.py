# src/viz.py
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Dict

from utils import AppConfig, TrialData
from preprocess import check_window_quality # preprocessからインポート

def plot_signal_qc(trial: TrialData, config: AppConfig):
    """フィルタリングとQCの結果を可視化する"""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("FP1", "FP2"))
    sfreq = config.filter.sfreq
    
    # データが存在するかチェック
    if trial.raw_stim_data is None or trial.filtered_stim_data is None:
        fig.update_layout(title="表示するデータがありません")
        return fig
    
    for i, ch_name in enumerate(['FP1', 'FP2']):
        # 生データ
        raw_time = np.arange(trial.raw_stim_data.shape[1]) / sfreq
        fig.add_trace(go.Scatter(x=raw_time, y=trial.raw_stim_data[i] * 1e6, mode='lines', 
                                 name=f'{ch_name} Raw', line=dict(color='lightgrey')), row=i+1, col=1)
        
        # フィルター後データ
        filt_time = np.arange(trial.filtered_stim_data.shape[1]) / sfreq
        fig.add_trace(go.Scatter(x=filt_time, y=trial.filtered_stim_data[i] * 1e6, mode='lines',
                                 name=f'{ch_name} Filtered', line=dict(color='royalblue')), row=i+1, col=1)

        # クリーンな窓をハイライト
        win_len_samples = int(config.win.win_len * sfreq)
        stim_offset_samples = int(config.win.stim_start * sfreq)
        
        data_for_qc = trial.filtered_stim_data[:, stim_offset_samples:]
        total_windows = data_for_qc.shape[1] // win_len_samples
        
        for j in range(total_windows):
            start = j * win_len_samples
            end = start + win_len_samples
            window = data_for_qc[:, start:end]
            if check_window_quality(window, config):
                 fig.add_vrect(
                     x0=(start + stim_offset_samples) / sfreq, 
                     x1=(end + stim_offset_samples) / sfreq, 
                     fillcolor="rgba(0, 255, 0, 0.2)", 
                     layer="below", line_width=0, row=i+1, col=1
                 )

    fig.update_layout(title=f"品質管理結果: {trial.subject_id} - Trial {trial.trial_id}", showlegend=False)
    fig.update_yaxes(title_text="振幅 (µV)")
    fig.update_xaxes(title_text="時間 (s)", row=2, col=1)
    return fig

def plot_feature_distribution(df: pd.DataFrame, feature: str):
    """特徴量のグループ間分布をボックスプロットで可視化"""
    fig = go.Figure()
    colors = {'好き': 'mediumseagreen', 'そうでもない': 'tomato'}
    
    for pref in ['好き', 'そうでもない']:
        fig.add_trace(go.Box(
            y=df[df['preference'] == pref][feature],
            name=pref,
            boxpoints='all', jitter=0.3, pointpos=-1.8,
            marker_color=colors[pref]
        ))
    
    fig.update_layout(
        title=f'"{feature}" の分布比較',
        yaxis_title="特徴量値",
        boxmode='group'
    )
    return fig

def plot_feature_correlation(df: pd.DataFrame, feature: str, target: str, stats_results: Dict):
    """特徴量と連続値の相関を散布図で可視化"""
    # 欠損値を除外
    plot_df = df[[feature, target]].dropna()

    fig = go.Figure(data=go.Scatter(
        x=plot_df[target],
        y=plot_df[feature],
        mode='markers',
        marker=dict(size=10, opacity=0.7)
    ))
    
    # 回帰直線を追加 (scipy.stats.linregressを使用)
    if 'slope' in stats_results and 'intercept' in stats_results:
        slope = stats_results['slope']
        intercept = stats_results['intercept']
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
