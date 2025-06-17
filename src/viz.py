# src/viz.py

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Dict

from utils import AppConfig, TrialData
# check_window_qualityは循環インポートを避けるため、viz.py内に再定義
def check_window_quality(window: np.ndarray, config: AppConfig) -> bool:
    if window is None: return False
    if np.any(np.abs(window) > config.qc.amp_uV): return False
    if np.any(np.abs(np.diff(window, axis=1)) > config.qc.diff_uV): return False
    return True

def plot_raw_signal_inspector(trial: TrialData, config: AppConfig):
    """フィルタリング前の生波形と振幅閾値を検査するためのプロット"""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("FP1", "FP2"))
    sfreq = config.filter.sfreq
    
    if trial.raw_stim_data is None:
        fig.update_layout(title="生データが見つかりません")
        return fig
    
    stim_offset_samples = int(config.win.stim_start * sfreq)
    data_to_plot = trial.raw_stim_data[:, stim_offset_samples:]
    time_axis = (np.arange(data_to_plot.shape[1]) + stim_offset_samples) / sfreq
    
    for i, ch_name in enumerate(['FP1', 'FP2']):
        fig.add_trace(go.Scatter(x=time_axis, y=data_to_plot[i], mode='lines', 
                                 name=f'{ch_name} Raw'), row=i+1, col=1)
        
        threshold = config.qc.amp_uV
        fig.add_hline(y=threshold, line_dash="dash", line_color="red", row=i+1, col=1,
                      annotation_text="振幅閾値", annotation_position="bottom right")
        fig.add_hline(y=-threshold, line_dash="dash", line_color="red", row=i+1, col=1)

    fig.update_layout(
        title=f"Rawデータインスペクター: {trial.subject_id} - Trial {trial.trial_id}",
        showlegend=False,
        yaxis_title="振幅 (データの元の単位)",
        xaxis_title="時間 (s)"
    )
    return fig

def plot_signal_qc(trial: TrialData, config: AppConfig):
    """フィルタリング効果とQCの結果を可視化する"""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("FP1", "FP2"))
    sfreq = config.filter.sfreq
    
    if trial.raw_stim_data is None or trial.filtered_stim_data is None:
        fig.update_layout(title="表示するデータがありません")
        return fig

    stim_offset_samples = int(config.win.stim_start * sfreq)
    raw_data_to_plot = trial.raw_stim_data[:, stim_offset_samples:]
    filt_data_to_plot = trial.filtered_stim_data
    
    raw_time_axis = (np.arange(raw_data_to_plot.shape[1]) + stim_offset_samples) / sfreq
    filt_time_axis = (np.arange(filt_data_to_plot.shape[1]) + stim_offset_samples) / sfreq
    
    for i, ch_name in enumerate(['FP1', 'FP2']):
        fig.add_trace(go.Scatter(x=raw_time_axis, y=raw_data_to_plot[i], mode='lines', 
                                 name=f'{ch_name} Raw', line=dict(color='lightgrey')), row=i+1, col=1)
        
        fig.add_trace(go.Scatter(x=filt_time_axis, y=filt_data_to_plot[i], mode='lines',
                                 name=f'{ch_name} Filtered', line=dict(color='royalblue')), row=i+1, col=1)

        win_len_samples = int(config.win.win_len * sfreq)
        total_windows = filt_data_to_plot.shape[1] // win_len_samples
        
        for j in range(total_windows):
            window = filt_data_to_plot[:, j*win_len_samples:(j+1)*win_len_samples]
            if check_window_quality(window, config):
                 fig.add_vrect(
                     x0=(j*win_len_samples + stim_offset_samples) / sfreq, 
                     x1=((j+1)*win_len_samples + stim_offset_samples) / sfreq, 
                     fillcolor="rgba(0, 255, 0, 0.2)", 
                     layer="below", line_width=0, row=i+1, col=1
                 )

    fig.update_layout(
        title=f"品質管理(QC)後: {trial.subject_id} - Trial {trial.trial_id}",
        showlegend=False,
        yaxis_title="振幅 (データの元の単位)",
        xaxis_title="時間 (s)"
    )
    return fig

def plot_feature_distribution(df: pd.DataFrame, feature: str):
    """特徴量のグループ間分布をボックスプロットで可視化（ホバー情報付き）"""
    fig = go.Figure()
    colors = {'好き': 'mediumseagreen', '嫌い': 'tomato', 'そうでもない': 'lightslategray'}
    
    for pref in df['preference'].unique():
        plot_df = df[df['preference'] == pref]
        if plot_df.empty: continue
        
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
            hovertext=hover_texts,
            hoverinfo='y+text'
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
        
    plot_df = df[[feature, target, 'subject_id', 'trial_id']].dropna()
    if plot_df.empty:
        fig = go.Figure().update_layout(title="表示できるデータがありません")
        return fig
    
    hovertemplate = (
        "<b>Subject: %{customdata[0]}</b><br>" +
        "Trial: %{customdata[1]}<br>" +
        f"{target}: %{{x}}<br>" +
        f"{feature}: %{{y:.3f}}" +
        "<extra></extra>"
    )
    
    fig = go.Figure(data=go.Scatter(
        x=plot_df[target],
        y=plot_df[feature],
        mode='markers',
        marker=dict(size=10, opacity=0.7),
        customdata=plot_df[['subject_id', 'trial_id']],
        hovertemplate=hovertemplate
    ))
    
    if stats_results and 'slope' in stats_results and 'intercept' in stats_results:
        slope = stats_results.get('slope', 0)
        intercept = stats_results.get('intercept', 0)
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
