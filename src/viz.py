# src/viz.py
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Dict

from utils import AppConfig, TrialData
from preprocess import check_window_quality

def plot_raw_signal_inspector(trial: TrialData, config: AppConfig):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("FP1", "FP2"))
    sfreq = config.filter.sfreq
    if trial.raw_stim_data is None:
        fig.update_layout(title="生データが見つかりません")
        return fig
    
    stim_offset = int(config.win.stim_start * sfreq)
    data_to_plot = trial.raw_stim_data[:, stim_offset:]
    time_axis = (np.arange(data_to_plot.shape[1]) + stim_offset) / sfreq
    
    for i, ch_name in enumerate(['FP1', 'FP2']):
        fig.add_trace(go.Scatter(x=time_axis, y=data_to_plot[i], mode='lines', name=f'{ch_name} Raw'), row=i+1, col=1)
        threshold = config.qc.amp_uV
        fig.add_hline(y=threshold, line_dash="dash", line_color="red", row=i+1, col=1, annotation_text="振幅閾値", annotation_position="bottom right")
        fig.add_hline(y=-threshold, line_dash="dash", line_color="red", row=i+1, col=1)
    
    fig.update_layout(title=f"Rawデータインスペクター: {trial.subject_id} - Trial {trial.trial_id}", showlegend=False, yaxis_title="振幅 (元の単位)", xaxis_title="時間 (s)")
    return fig

def plot_signal_qc(trial: TrialData, config: AppConfig):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("FP1", "FP2"))
    sfreq = config.filter.sfreq
    if trial.raw_stim_data is None or trial.filtered_stim_data is None:
        fig.update_layout(title="表示するデータがありません")
        return fig
    
    stim_offset = int(config.win.stim_start * sfreq)
    raw_data = trial.raw_stim_data[:, stim_offset:]
    filt_data = trial.filtered_stim_data
    raw_time = (np.arange(raw_data.shape[1]) + stim_offset) / sfreq
    filt_time = (np.arange(filt_data.shape[1]) + stim_offset) / sfreq
    
    for i, ch_name in enumerate(['FP1', 'FP2']):
        fig.add_trace(go.Scatter(x=raw_time, y=raw_data[i], mode='lines', name=f'{ch_name} Raw', line=dict(color='lightgrey')), row=i+1, col=1)
        fig.add_trace(go.Scatter(x=filt_time, y=filt_data[i], mode='lines', name=f'{ch_name} Filtered', line=dict(color='royalblue')), row=i+1, col=1)
        
        win_len_samples = int(config.win.win_len * sfreq)
        for j in range(filt_data.shape[1] // win_len_samples):
            window = filt_data[:, j*win_len_samples:(j+1)*win_len_samples]
            if check_window_quality(window, config):
                fig.add_vrect(x0=(j*win_len_samples + stim_offset) / sfreq, x1=((j+1)*win_len_samples + stim_offset) / sfreq, fillcolor="rgba(0, 255, 0, 0.2)", layer="below", line_width=0, row=i+1, col=1)
    
    fig.update_layout(title=f"品質管理(QC)後: {trial.subject_id} - Trial {trial.trial_id}", showlegend=False, yaxis_title="振幅 (元の単位)", xaxis_title="時間 (s)")
    return fig

def plot_feature_distribution(df: pd.DataFrame, feature: str):
    fig = go.Figure()
    colors = {'好き': 'mediumseagreen', 'そうでもない': 'tomato'}
    for pref in df['preference'].unique():
        fig.add_trace(go.Box(y=df[df['preference'] == pref][feature], name=pref, boxpoints='all', jitter=0.3, pointpos=-1.8, marker_color=colors.get(pref)))
    fig.update_layout(title=f'"{feature}" の分布比較', yaxis_title="特徴量値", boxmode='group')
    return fig

def plot_feature_correlation(df: pd.DataFrame, feature: str, target: str, stats_results: Dict):
    if target not in df.columns:
        fig = go.Figure().update_layout(title=f"ターゲット列 '{target}' が見つかりません")
        return fig
    plot_df = df[[feature, target]].dropna()
    if plot_df.empty:
        fig = go.Figure().update_layout(title="表示できるデータがありません")
        return fig
    
    fig = go.Figure(data=go.Scatter(x=plot_df[target], y=plot_df[feature], mode='markers', marker=dict(size=10, opacity=0.7)))
    if 'slope' in stats_results and 'intercept' in stats_results:
        slope, intercept = stats_results.get('slope', 0), stats_results.get('intercept', 0)
        x_range = np.array([plot_df[target].min(), plot_df[target].max()])
        y_range = slope * x_range + intercept
        fig.add_trace(go.Scatter(x=x_range, y=y_range, mode='lines', name='回帰直線', line=dict(color='black', dash='dash')))
    fig.update_layout(title=f'"{feature}" と "{target}" の相関', xaxis_title=target, yaxis_title="特徴量値")
    return fig
