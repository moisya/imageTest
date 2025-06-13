# app.py
import streamlit as st # ★★★ インポート文を追加 ★★★
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# --- パス設定とモジュールインポート ---
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
if str(project_root / "src") not in sys.path:
    sys.path.append(str(project_root / "src"))

from auth import check_password
from utils import AppConfig, FilterConfig, QCThresholds, WindowConfig, FrequencyBands
from io_module import load_all_trial_data
from preprocess import run_preprocessing_pipeline
from features import extract_all_features
from stats import run_statistical_analysis
from viz import plot_signal_qc, plot_feature_distribution, plot_feature_correlation

# --- ページ設定 ---
st.set_page_config(
    layout="wide", 
    page_title="EEG画像嗜好解析システム",
    page_icon="🧠"
)

# --- パスワード認証 ---
if not check_password():
    st.stop()

# --- メインのアプリケーション ---
st.title("🧠 EEG画像嗜好解析システム")
st.markdown("FP1・FP2チャネルを用いた脳波データから、画像に対する嗜好を多角的に分析します。")

# --- サイドバー ---
with st.sidebar:
    st.header("⚙️ 解析設定")
    
    st.subheader("📁 データファイル")
    uploaded_eeg_file = st.file_uploader(
        "1. EEGファイルをアップロード (.xdf)",
        type=['xdf'],
        accept_multiple_files=False,
        help="被験者1名分のXDFファイルをアップロードします。"
    )
    uploaded_survey_file = st.file_uploader(
        "2. 評価データをアップロード (.csv, .xlsx)",
        type=['csv', 'xlsx'],
        help="'trial_id'と評価スコア列を含むファイル"
    )
    
    with st.expander("詳細パラメータ設定", expanded=False):
        st.subheader("🔧 フィルタ設定")
        l_freq = st.slider("下限周波数 (Hz)", 0.1, 5.0, 1.0, 0.1)
        h_freq = st.slider("上限周波数 (Hz)", 30.0, 100.0, 50.0, 1.0)
        notch_freq = st.selectbox("ノッチフィルタ周波数 (Hz)", [50, 60], index=0)
        
        st.subheader("🎯 品質管理")
        amp_threshold = st.slider("振幅閾値 (µV)", 50.0, 150.0, 80.0, 5.0)
        diff_threshold = st.slider("隣接差閾値 (µV)", 20.0, 50.0, 35.0, 2.5)
        
        st.subheader("⏱️ 時間窓設定")
        baseline_samples = st.slider("ベースラインサンプル数", 1, 3, 2, 1)
        stim_samples = st.slider("刺激区間サンプル数", 3, 10, 5, 1)
    
    st.markdown("---")
    run_analysis = st.button("🚀 解析実行", type="primary", use_container_width=True)

# --- 解析パイプライン関数 ---
@st.cache_data(show_spinner="解析パイプラインを実行中...")
def run_full_pipeline(_uploaded_eeg_file, _uploaded_survey_file, _config):
    eeg_files_list = [_uploaded_eeg_file] if _uploaded_eeg_file else []
    if not eeg_files_list:
        return None, None, None, "EEGファイルがアップロードされていません。"
    
    all_trials, _ = load_all_trial_data(eeg_files_list, _uploaded_survey_file, _config)
    if not all_trials:
        return None, None, None, "有効な試行データが読み込めませんでした。ファイル形式や内容を確認してください。"
    
    processed_trials, qc_stats = run_preprocessing_pipeline(all_trials, _config)
    features_df = extract_all_features(processed_trials, _config)
    
    if features_df.empty:
        return qc_stats, None, None, "有効な試行が全て除去されたため、特徴量を抽出できませんでした。品質管理の閾値を調整してください。"

    return qc_stats, features_df, processed_trials, None

# --- メインエリアの表示ロジック ---
if not uploaded_eeg_file:
    st.info("👈 左側のサイドバーからEEGファイルと評価データをアップロードして解析を開始してください。")
elif run_analysis:
    config = AppConfig(
        filter=FilterConfig(l_freq=l_freq, h_freq=h_freq, notch_freq=float(notch_freq)),
        qc=QCThresholds(amp_uV=amp_threshold, diff_uV=diff_threshold),
        win=WindowConfig(baseline_samples=baseline_samples, stim_samples=stim_samples),
        freq_bands=FrequencyBands()
    )
    
    qc_stats, features_df, processed_trials, error_message = run_full_pipeline(
        uploaded_eeg_file, uploaded_survey_file, config
    )
    
    if error_message:
        st.error(error_message)
    elif qc_stats is not None:
        tabs = st.tabs(["📋 解析サマリー", "🔧 前処理結果", "📈 統計解析"])
        
        with tabs[0]:
            st.header("品質管理サマリー")
            st.dataframe(qc_stats, use_container_width=True)
            if features_df is not None:
                st.header("特徴量データプレビュー")
                st.dataframe(features_df.head(), use_container_width=True)
        
        with tabs[1]:
            st.header("前処理と品質管理の視覚化")
            if processed_trials:
                subjects = sorted(list(set(t.subject_id for t in processed_trials)))
                selected_subject = st.selectbox("被験者を選択", subjects)
                
                valid_trials_for_subject = [t for t in processed_trials if t.subject_id == selected_subject and t.is_valid]
                if valid_trials_for_subject:
                    selected_trial_id = st.selectbox(
                        "試行を選択", 
                        [t.trial_id for t in valid_trials_for_subject],
                        key=f"trial_selector_{selected_subject}"
                    )
                    selected_trial = next((t for t in valid_trials_for_subject if t.trial_id == selected_trial_id), None)
                    if selected_trial:
                        fig = plot_signal_qc(selected_trial, config)
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning(f"{selected_subject}には有効な試行がありません。")
        
        with tabs[2]:
            st.header("特徴量の統計的比較")
            if features_df is not None and not features_df.empty:
                # ... (統計タブのUIとロジックは前回と同じ) ...
                pass # ここに前回の統計タブのコードを貼り付け
    
# ... (フッターは同じ) ...
