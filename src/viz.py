# src/viz.py
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

from utils import AppConfig, TrialData

def plot_signal_qc(trial: TrialData, config: AppConfig):
    """フィルタリングとQCの結果を可視化する"""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("FP1", "FP2"))
    sfreq = config.filter.sfreq
    
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
        
        total_windows = (trial.filtered_stim_data.shape[1] - stim_offset_samples) // win_len_samples
        for j in range(total_windows):
            start = stim_offset_samples + j * win_len_samples
            end = start + win_len_samples
            window = trial.filtered_stim_data[:, start:end]
            if check_window_quality(window, config): # 簡易チェック
                 fig.add_vrect(x0=start/sfreq, x1=end/sfreq, fillcolor="rgba(0, 255, 0, 0.2)", 
                               layer="below", line_width=0, row=i+1, col=1)

    fig.update_layout(title=f"品質管理結果: {trial.trial_id}", showlegend=False)
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

def plot_feature_correlation(df: pd.DataFrame, feature: str, target: str):
    """特徴量と連続値の相関を散布図で可視化"""
    fig = go.Figure(data=go.Scatter(
        x=df[target],
        y=df[feature],
        mode='markers',
        marker=dict(
            color=df['preference'].map({'好き': 'mediumseagreen', 'そうでもない': 'tomato'}),
            size=10,
            opacity=0.7
        )
    ))
    
    # 回帰直線を追加
    import statsmodels.api as sm
    X = sm.add_constant(df[target].dropna())
    y = df[feature].dropna()
    model = sm.OLS(y, X).fit()
    x_range = np.linspace(X[target].min(), X[target].max(), 100)
    y_range = model.predict(sm.add_constant(x_range))
    
    fig.add_trace(go.Scatter(x=x_range, y=y_range, mode='lines', name='回帰直線', line=dict(color='black', dash='dash')))

    fig.update_layout(
        title=f'"{feature}" と "{target}" の相関',
        xaxis_title=target,
        yaxis_title="特徴量値"
    )
    return fig

# `check_window_quality`をviz.py内でも使えるようにヘルパーとして定義
def check_window_quality(window: np.ndarray, config: AppConfig) -> bool:
    if np.any(np.abs(window) * 1e6 > config.qc.amp_uV): return False
    if np.any(np.abs(np.diff(window, axis=1)) * 1e6 > config.qc.diff_uV): return False
    return True
