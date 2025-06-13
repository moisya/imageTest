# app.py

import streamlit as st
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
from viz import (
    plot_signal_qc, 
    plot_feature_distribution, 
    plot_feature_correlation,
    plot_raw_signal_inspector
)

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
        "1. EEGファイルをアップロード (.xdf, etc.)",
        type=['xdf', 'edf', 'bdf', 'fif'],
        accept_multiple_files=False,
        help="被験者1名分のEEGファイルをアップロードします。"
    )
    uploaded_survey_file = st.file_uploader(
        "2. 評価データをアップロード (.csv, .xlsx)",
        type=['csv', 'xlsx'],
        help="'trial_id'と評価スコア列を含むファイル"
    )
    
    with st.expander("詳細パラメータ設定", expanded=True):
        st.subheader("🔧 フィルタ設定")
        l_freq = st.slider("下限周波数 (Hz)", 0.1, 5.0, 1.0, 0.1, key="l_freq")
        h_freq = st.slider("上限周波数 (Hz)", 30.0, 100.0, 50.0, 1.0, key="h_freq")
        
        st.subheader("🎯 品質管理 (µV単位)")
        st.info("データの単位がボルト(V)の場合、100µVは 0.0001 Vです。")
        amp_threshold = st.number_input("振幅閾値 (µV)", min_value=1.0, max_value=500.0, value=150.0, step=5.0, key="amp_thresh")
        diff_threshold = st.number_input("隣接差閾値 (µV)", min_value=1.0, max_value=100.0, value=50.0, step=2.5, key="diff_thresh")
        
        st.subheader("⏱️ 時間窓設定")
        baseline_samples = st.slider("ベースラインサンプル数", 1, 5, 2, 1, key="base_samples")
        stim_samples = st.slider("刺激区間サンプル数", 1, 10, 5, 1, key="stim_samples")
    
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
        return None, None, None, "有効な試行データの読み込みに失敗しました。ファイル形式や内容を確認してください。"
    
    processed_trials, qc_stats = run_preprocessing_pipeline(all_trials, _config)
    features_df = extract_all_features(processed_trials, _config)
    
    if features_df.empty:
        return qc_stats, None, processed_trials, "有効な試行が全て除去されたため、特徴量を抽出できませんでした。品質管理の閾値を調整してください。"

    return qc_stats, features_df, processed_trials, None

# --- メインエリアの表示ロジック ---
if not uploaded_eeg_file:
    st.info("👈 左側のサイドバーからEEGファイルと評価データをアップロードして解析を開始してください。")
elif run_analysis:
    config = AppConfig(
        filter=FilterConfig(l_freq=l_freq, h_freq=h_freq, notch_freq=50.0), # notchは固定
        qc=QCThresholds(amp_uV=amp_threshold, diff_uV=diff_threshold),
        win=WindowConfig(baseline_samples=baseline_samples, stim_samples=stim_samples),
        freq_bands=FrequencyBands()
    )
    
    qc_stats, features_df, processed_trials, error_message = run_full_pipeline(
        uploaded_eeg_file, uploaded_survey_file, config
    )
    
    # 解析サマリータブは常に表示
    st.header("📋 解析サマリー")
    if qc_stats is not None:
        st.dataframe(qc_stats, use_container_width=True)
    else:
        st.warning("品質管理サマリーを生成できませんでした。")

    if error_message:
        st.error(error_message)

    # 結果表示用のタブを作成
    if processed_trials:
        tab_list = [" raw データ検査", "🔧 前処理結果"]
        if features_df is not None and not features_df.empty:
            tab_list.append("📈 統計解析")
        
        tabs = st.tabs(tab_list)

        with tabs[0]: # Rawデータ検査
            st.header("Rawデータインスペクター")
            st.info("品質管理前の生の波形と、設定された振幅閾値（赤線）を確認します。波形が常に赤線を超えている場合、閾値が厳しすぎるか、データのスケールが大きい可能性があります。")
            subjects_raw = sorted(list(set(t.subject_id for t in processed_trials)))
            selected_subject_raw = st.selectbox("被験者を選択", subjects_raw, key="raw_subject_selector")
            
            trials_for_subject_raw = [t for t in processed_trials if t.subject_id == selected_subject_raw]
            if trials_for_subject_raw:
                selected_trial_id_raw = st.selectbox(
                    "試行を選択", [t.trial_id for t in trials_for_subject_raw], key=f"raw_trial_selector_{selected_subject_raw}"
                )
                selected_trial_raw = next((t for t in trials_for_subject_raw if t.trial_id == selected_trial_id_raw), None)
                if selected_trial_raw:
                    fig_raw = plot_raw_signal_inspector(selected_trial_raw, config)
                    st.plotly_chart(fig_raw, use_container_width=True)

        with tabs[1]: # 前処理結果
            st.header("前処理と品質管理の視覚化")
            subjects_qc = sorted(list(set(t.subject_id for t in processed_trials if t.is_valid)))
            if subjects_qc:
                selected_subject_qc = st.selectbox("被験者を選択", subjects_qc, key="qc_subject_selector")
                valid_trials_for_subject = [t for t in processed_trials if t.subject_id == selected_subject_qc and t.is_valid]
                if valid_trials_for_subject:
                    selected_trial_id_qc = st.selectbox(
                        "試行を選択", [t.trial_id for t in valid_trials_for_subject], key=f"qc_trial_selector_{selected_subject_qc}"
                    )
                    selected_trial_qc = next((t for t in valid_trials_for_subject if t.trial_id == selected_trial_id_qc), None)
                    if selected_trial_qc:
                        fig_qc = plot_signal_qc(selected_trial_qc, config)
                        st.plotly_chart(fig_qc, use_container_width=True)
            else:
                st.warning("表示できる有効な試行がありません。")

        if len(tabs) > 2:
            with tabs[2]: # 統計解析
                st.header("特徴量の統計的比較")
                col1, col2 = st.columns(2)
                with col1:
                    feature_to_analyze = st.selectbox(
                        "分析する特徴量を選択",
                        sorted(features_df.columns.drop(['subject_id', 'trial_id', 'preference', 'dummy_valence'], errors='ignore'))
                    )
                with col2:
                    analysis_type = st.selectbox("分析方法を選択", ["グループ比較 (好き vs そうでもない)", "相関分析 (ダミー連続値)"])
                
                stats_results = run_statistical_analysis(features_df, feature_to_analyze, analysis_type)
                
                if analysis_type == "グループ比較 (好き vs そうでもない)":
                    fig_dist = plot_feature_distribution(features_df, feature_to_analyze)
                    st.plotly_chart(fig_dist, use_container_width=True)
                    st.subheader("統計検定結果 (t検定)")
                    res_col1, res_col2, res_col3 = st.columns(3)
                    res_col1.metric("p値", f"{stats_results.get('p_value', 'N/A'):.4f}")
                    res_col2.metric("効果量 (Cohen's d)", f"{stats_results.get('effect_size', 'N/A'):.3f}")
                    res_col3.metric("検定力 (Power)", f"{stats_results.get('power', 'N/A'):.3f}")
                else:
                    fig_corr = plot_feature_correlation(features_df, feature_to_analyze, "dummy_valence", stats_results)
                    st.plotly_chart(fig_corr, use_container_width=True)
                    st.subheader("統計検定結果 (ピアソン相関)")
                    res_col1, res_col2 = st.columns(2)
                    res_col1.metric("相関係数 (r)", f"{stats_results.get('corr_coef', 'N/A'):.3f}")
                    res_col2.metric("p値", f"{stats_results.get('p_value', 'N/A'):.4f}")
