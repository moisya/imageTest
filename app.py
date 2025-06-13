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
# ★★★ ここを修正！実際に使っている関数のみをインポート ★★★
from viz import (
    plot_signal_qc, 
    plot_feature_distribution, 
    plot_feature_correlation
)

# ... (以降のコードは変更なし) ...
# (前回の回答の app.py の残りの部分をここに貼り付け)
# --- ページ設定 ---
st.set_page_config(
    layout="wide", 
    page_title="EEG画像嗜好解析システム",
    page_icon="🧠"
)

# --- ★★★ パスワード認証 ★★★ ---
if not check_password():
    st.stop()  # 認証されなければここで処理を停止

# --- メインのアプリケーション ---
st.title("🧠 EEG画像嗜好解析システム")
st.markdown("FP1・FP2チャネルを用いた脳波データから、画像に対する嗜好（好き／そうでもない）を多角的に分析します。")

# --- サイドバー ---
with st.sidebar:
    st.header("⚙️ 解析設定")
    
    # 1. ファイルアップロード
    st.subheader("📁 データファイル")
    uploaded_files = st.file_uploader(
        "EEGファイルをアップロード (.xdf, .edf, .bdf, .fif)", 
        type=['xdf', 'edf', 'bdf', 'fif'], # XDFを追加
        accept_multiple_files=True,
        help="複数の被験者のEEGファイルを同時にアップロードできます。"
    )
    
    # 2. パラメータ設定 (アコーディオンメニューで整理)
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
    
    # 3. 解析実行ボタン
    st.markdown("---")
    run_analysis = st.button("🚀 解析実行", type="primary", use_container_width=True)

# --- 解析パイプライン関数（キャッシュあり） ---
@st.cache_data(show_spinner="解析パイプラインを実行中...")
def run_full_pipeline(_uploaded_files, _config):
    # 1. データ読み込み
    all_trials, meta_info = load_all_trial_data(_uploaded_files, _config)
    if not all_trials:
        return None, None, None, "有効な試行データが読み込めませんでした。ファイル形式や内容を確認してください。"
    
    # 2. 前処理と品質管理
    processed_trials, qc_stats = run_preprocessing_pipeline(all_trials, _config)
    
    # 3. 特徴量抽出
    features_df = extract_all_features(processed_trials, _config)
    
    if features_df.empty:
        return qc_stats, None, None, "有効な試行が全て除去されたため、特徴量を抽出できませんでした。品質管理の閾値を調整してください。"

    return qc_stats, features_df, processed_trials, None

# --- メインエリアの表示ロジック ---
if not uploaded_files:
    st.info("👈 左側のサイドバーからEEGファイルをアップロードして解析を開始してください。")
elif run_analysis:
    # 設定オブジェクトを作成
    config = AppConfig(
        filter=FilterConfig(l_freq=l_freq, h_freq=h_freq, notch_freq=float(notch_freq)),
        qc=QCThresholds(amp_uV=amp_threshold, diff_uV=diff_threshold),
        win=WindowConfig(baseline_samples=baseline_samples, stim_samples=stim_samples),
        freq_bands=FrequencyBands()
    )
    
    # 解析パイプラインを実行
    qc_stats, features_df, processed_trials, error_message = run_full_pipeline(uploaded_files, config)
    
    if error_message:
        st.error(error_message)
    elif qc_stats is not None:
        # 結果表示用のタブを作成
        tabs = st.tabs(["📋 解析サマリー", "🔧 前処理結果", "📈 統計解析"])
        
        with tabs[0]: # 解析サマリー
            st.header("品質管理サマリー")
            st.dataframe(qc_stats, use_container_width=True)
            if features_df is not None:
                st.header("特徴量データプレビュー")
                st.dataframe(features_df.head(), use_container_width=True)
        
        with tabs[1]: # 前処理結果
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
                    selected_trial = next(t for t in valid_trials_for_subject if t.trial_id == selected_trial_id)
                    
                    fig = plot_signal_qc(selected_trial, config)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning(f"{selected_subject}には有効な試行がありません。")
            else:
                st.warning("表示できる有効な試行がありませんでした。")

        with tabs[2]: # 統計解析
            st.header("特徴量の統計的比較")
            if features_df is not None and not features_df.empty:
                col1, col2 = st.columns(2)
                with col1:
                    feature_to_analyze = st.selectbox(
                        "分析する特徴量を選択",
                        sorted(features_df.columns.drop(['subject_id', 'trial_id', 'preference', 'dummy_valence'], errors='ignore'))
                    )
                with col2:
                    analysis_type = st.selectbox(
                        "分析方法を選択",
                        ["グループ比較 (好き vs そうでもない)", "相関分析 (ダミー連続値)"]
                    )
                
                stats_results = run_statistical_analysis(features_df, feature_to_analyze, analysis_type)
                
                if analysis_type == "グループ比較 (好き vs そうでもない)":
                    fig = plot_feature_distribution(features_df, feature_to_analyze)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.subheader("統計検定結果 (t検定)")
                    res_col1, res_col2, res_col3 = st.columns(3)
                    res_col1.metric("p値", f"{stats_results.get('p_value', 'N/A'):.4f}")
                    res_col2.metric("効果量 (Cohen's d)", f"{stats_results.get('effect_size', 'N/A'):.3f}")
                    res_col3.metric("検定力 (Power)", f"{stats_results.get('power', 'N/A'):.3f}")

                else: # 相関分析
                    if 'dummy_valence' in features_df.columns:
                        fig = plot_feature_correlation(features_df, feature_to_analyze, "dummy_valence", stats_results)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.subheader("統計検定結果 (ピアソン相関)")
                        res_col1, res_col2 = st.columns(2)
                        res_col1.metric("相関係数 (r)", f"{stats_results.get('corr_coef', 'N/A'):.3f}")
                        res_col2.metric("p値", f"{stats_results.get('p_value', 'N/A'):.4f}")
                    else:
                        st.warning("相関分析に必要な 'dummy_valence' 列が見つかりません。")
            else:
                st.warning("分析できる特徴量データがありません。")

else:
    st.info("👈 左側のサイドバーからEEGファイルをアップロードし、「解析実行」ボタンを押してください。")

st.markdown("---")
st.markdown("<div style='text-align: center; color: #888;'>🧠 EEG画像嗜好解析システム v1.2</div>", unsafe_allow_html=True)
