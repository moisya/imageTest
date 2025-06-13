# app.py

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# --- パス設定とモジュールインポート ---
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

try:
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
except ImportError as e:
    st.error(f"必要なモジュールのインポートに失敗しました: {e}")
    st.stop()

# --- ページ設定 ---
st.set_page_config(layout="wide", page_title="EEG画像嗜好解析システム", page_icon="🧠")

# --- パスワード認証 ---
if not check_password():
    st.stop()

# --- ★★★ Session State の初期化 ★★★ ---
# 再実行されても値が消えないように、ここで変数の置き場所を初期化
if 'analysis_run' not in st.session_state:
    st.session_state['analysis_run'] = False
    st.session_state['results'] = {}

# --- メインのアプリケーション ---
st.title("🧠 EEG画像嗜好解析システム")
st.markdown("FP1・FP2チャネルを用いた脳波データから、画像に対する嗜好を多角的に分析します。")

# --- サイドバー ---
with st.sidebar:
    st.header("⚙️ 解析設定")
    
    st.subheader("📁 データファイル")
    uploaded_eeg_file = st.file_uploader(
        "1. EEGファイルをアップロード (.xdf, etc.)", type=['xdf', 'edf', 'bdf', 'fif']
    )
    uploaded_survey_file = st.file_uploader(
        "2. 評価データをアップロード (.csv, .xlsx)", type=['csv', 'xlsx']
    )
    
    with st.expander("詳細パラメータ設定", expanded=True):
        # (パラメータ設定のスライダー等は変更なし)
        l_freq = st.slider(...) 
        # ...

    run_analysis = st.button("🚀 解析実行", type="primary", use_container_width=True)

# --- 解析パイプライン関数 (変更なし) ---
@st.cache_data(show_spinner="解析パイプラインを実行中...")
def run_full_pipeline(_uploaded_eeg_file, _uploaded_survey_file, _config):
    # (この関数の中身は前回と同じ)
    # ...
    return qc_stats, features_df, processed_trials, error_message

# --- ★★★ ボタンが押されたときの処理 ★★★ ---
if run_analysis:
    if not uploaded_eeg_file:
        st.error("EEGファイルがアップロードされていません。")
    else:
        config = AppConfig(...) # configオブジェクト作成
        
        # 解析を実行し、結果を session_state に保存
        qc_stats, features_df, processed_trials, error_message = run_full_pipeline(
            uploaded_eeg_file, uploaded_survey_file, config
        )
        st.session_state['results'] = {
            "qc_stats": qc_stats,
            "features_df": features_df,
            "processed_trials": processed_trials,
            "error_message": error_message,
            "config": config # configも保存しておくと便利
        }
        st.session_state['analysis_run'] = True # 解析実行フラグを立てる

# --- ★★★ メインエリアの表示ロジック ★★★ ---
# 条件を「ボタンが押された時」から「解析が一度でも実行された後」に変更
if st.session_state['analysis_run']:
    # session_stateから結果を取り出す
    results = st.session_state['results']
    qc_stats = results.get("qc_stats")
    features_df = results.get("features_df")
    processed_trials = results.get("processed_trials")
    error_message = results.get("error_message")
    config = results.get("config")

    try:
        if error_message:
            st.error(error_message)

        if qc_stats is not None:
            st.header("📋 解析サマリー")
            st.dataframe(qc_stats, use_container_width=True)
        
        if processed_trials:
            tab_list = [" raw データ検査", "🔧 前処理結果"]
            if features_df is not None and not features_df.empty:
                tab_list.append("📈 統計解析")
            
            tabs = st.tabs(tab_list)

            with tabs[0]: # Rawデータ検査
                # (前回と同じコード)
                # ...
                
            with tabs[1]: # 前処理結果
                # (前回と同じコード)
                # ...

            if len(tabs) > 2:
                with tabs[2]: # 統計解析
                    st.header("特徴量の統計的比較")
                    col1, col2 = st.columns(2)
                    with col1:
                        # 特徴量リストを作成
                        feature_options = sorted(features_df.columns.drop(['subject_id', 'trial_id', 'preference', 'dummy_valence'], errors='ignore'))
                        # プルダウンを作成
                        feature_to_analyze = st.selectbox(
                            "分析する特徴量を選択", feature_options
                        )
                    with col2:
                        analysis_type = st.selectbox("分析方法を選択", ["グループ比較 (好き vs そうでもない)", "相関分析 (ダミー連続値)"])
                    
                    # プルダウンで何が選ばれても、ここから下のコードが実行される
                    stats_results = run_statistical_analysis(features_df, feature_to_analyze, analysis_type)
                    
                    if not stats_results:
                        st.warning(f"特徴量 '{feature_to_analyze}' の統計値を計算できませんでした。")
                    else:
                        # (グラフ描画とメトリクス表示のコードは前回と同じ)
                        # ...
    
    except Exception as e:
        st.error("アプリケーションの表示中に予期せぬエラーが発生しました。")
        st.exception(e)
else:
    # アプリ起動時の初期画面
    st.info("👈 左側のサイドバーからEEGファイル等をアップロードし、「解析実行」ボタンを押してください。")

# --- フッター ---
# (変更なし)

# --- フッター ---
st.markdown("---")
st.markdown("<div style='text-align: center; color: #888;'>🧠 EEG画像嗜好解析システム v1.3</div>", unsafe_allow_html=True)
