# app.py

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import time

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
    st.info("プロジェクトのフォルダ構成が正しいか、`src`フォルダ内に必要な.pyファイルがすべて存在するか確認してください。")
    st.stop()

# --- ページ設定 ---
st.set_page_config(layout="wide", page_title="EEG画像嗜好解析システム", page_icon="🧠")

# --- パスワード認証 ---
if not check_password():
    st.stop()

# --- Session State の初期化 ---
if 'analysis_run' not in st.session_state:
    st.session_state['analysis_run'] = False
    st.session_state['results'] = {}

# --- メインのアプリケーション ---
st.title("🧠 EEG画像嗜好解析システム")
st.markdown("複数の被験者の脳波データとアンケート結果を統合し、画像に対する嗜好を多角的に分析します。")

# --- サイドバー ---
with st.sidebar:
    st.header("⚙️ 解析設定")
    
    st.subheader("📁 データファイル")
    uploaded_eeg_files = st.file_uploader(
        "1. EEGファイルをアップロード (.xdf, etc.)",
        type=['xdf', 'edf', 'bdf', 'fif'],
        accept_multiple_files=True
    )
    uploaded_survey_files = st.file_uploader(
        "2. 評価データをアップロード (.csv, .xlsx)",
        type=['csv', 'xlsx'],
        accept_multiple_files=True
    )
    
    with st.expander("詳細パラメータ設定", expanded=True):
        # --- ★★★ ここから修正 ★★★ ---
        st.subheader("🔧 フィルタ設定")
        # 2つのスライダーを1つのレンジスライダーに統合
        freq_range = st.slider(
            "周波数帯域 (Hz)",
            min_value=0.1,
            max_value=100.0,
            value=(1.0, 50.0), # デフォルトの範囲をタプルで指定
            step=0.5,
            key="freq_range_slider",
            help="解析対象とする周波数の下限と上限を設定します。"
        )
        # レンジスライダーの結果（タプル）をそれぞれの変数に分解
        l_freq, h_freq = freq_range
        # --- ★★★ 修正ここまで ★★★ ---
        
        st.subheader("🎯 品質管理 (µV単位)")
        st.info("データの単位がボルト(V)の場合、100µVは 0.0001 Vです。")
        amp_threshold = st.number_input("振幅閾値", 0.0, 500.0, 150.0, 5.0, key="amp_thresh", format="%.1f")
        diff_threshold = st.number_input("隣接差閾値", 0.0, 100.0, 50.0, 2.5, key="diff_thresh", format="%.1f")
        
        st.subheader("⏱️ 時間窓設定")
        baseline_samples = st.slider("ベースラインサンプル数", 1, 5, 2, 1, key="base_samples")
        stim_samples = st.slider("刺激区間サンプル数", 1, 10, 5, 1, key="stim_samples")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        run_analysis = st.button("🚀 解析実行", type="primary", use_container_width=True)
    with col2:
        reset_app = st.button("リセット", use_container_width=True)

# --- 解析パイプライン関数 ---
@st.cache_data(show_spinner="解析パイプラインを実行中...")
def run_full_pipeline(_uploaded_eeg_files_list, _uploaded_survey_files_list, _config):
    if not _uploaded_eeg_files_list:
        return None, None, None, "EEGファイルがアップロードされていません。"
    all_trials, _ = load_all_trial_data(_uploaded_eeg_files_list, _uploaded_survey_files_list, _config)
    if not all_trials:
        return None, None, None, "有効な試行データの読み込みに失敗しました。"
    processed_trials, qc_stats = run_preprocessing_pipeline(all_trials, _config)
    features_df = extract_all_features(processed_trials, _config)
    if features_df.empty:
        return qc_stats, None, processed_trials, "有効な試行が全て除去されました。"
    return qc_stats, features_df, processed_trials, None

# --- リセットボタンの処理ロジック ---
if reset_app:
    st.cache_data.clear()
    st.session_state.clear()
    st.success("状態がリセットされました。ページをリロードして、再度解析を開始してください。")
    time.sleep(2)
    st.rerun()

# --- ボタンが押されたときの処理 ---
if run_analysis:
    if not uploaded_eeg_files:
        st.error("EEGファイルがアップロードされていません。")
    else:
        config = AppConfig(
            filter=FilterConfig(l_freq=l_freq, h_freq=h_freq, notch_freq=50.0),
            qc=QCThresholds(amp_uV=amp_threshold, diff_uV=diff_threshold),
            win=WindowConfig(baseline_samples=baseline_samples, stim_samples=stim_samples),
            freq_bands=FrequencyBands()
        )
        qc_stats, features_df, processed_trials, error_message = run_full_pipeline(
            uploaded_eeg_files, uploaded_survey_files, config
        )
        st.session_state['results'] = {
            "qc_stats": qc_stats, "features_df": features_df,
            "processed_trials": processed_trials, "error_message": error_message,
            "config": config
        }
        st.session_state['analysis_run'] = True
        st.rerun()

# --- メインエリアの表示ロジック ---
if st.session_state.get('analysis_run', False):
    results = st.session_state.get('results', {})
    qc_stats = results.get("qc_stats")
    features_df = results.get("features_df")
    processed_trials = results.get("processed_trials")
    error_message = results.get("error_message")
    config = results.get("config")

    try:
        if error_message:
            st.error(error_message)

        if qc_stats is not None and not qc_stats.empty:
            st.header("📋 解析サマリー")
            valid_df = qc_stats[qc_stats['is_valid']]
            total_valid_trials = len(valid_df)
            counts = valid_df['preference'].value_counts()
            
            st.subheader("全体合計")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("✅ 有効試行 (合計)", f"{total_valid_trials} 件")
            col2.metric("👍 好き", f"{counts.get('好き', 0)} 件")
            col3.metric("👎 嫌い", f"{counts.get('嫌い', 0)} 件")
            col4.metric("😐 そうでもない", f"{counts.get('そうでもない', 0)} 件")
            
            st.markdown("---")
            st.subheader("被験者ごとの内訳")
            subject_list = sorted(list(qc_stats['subject_id'].unique()))
            num_subjects = len(subject_list)
            cols = st.columns(num_subjects if num_subjects > 0 else 1)
            
            for i, subject_id in enumerate(subject_list):
                with cols[i]:
                    st.markdown(f"**{subject_id}**")
                    subject_valid_df = valid_df[valid_df['subject_id'] == subject_id]
                    subject_counts = subject_valid_df['preference'].value_counts()
                    st.markdown(f"✅ **有効: {len(subject_valid_df)}件**")
                    st.markdown(f"👍 好き: {subject_counts.get('好き', 0)}件")
                    st.markdown(f"👎 嫌い: {subject_counts.get('嫌い', 0)}件")
                    st.markdown(f"😐 そうでもない: {subject_counts.get('そうでもない', 0)}件")
            
            if features_df is not None and not features_df.empty:
                st.markdown("---")
                st.subheader("⬇️ 解析結果のダウンロード")
                csv_data = features_df.to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    label="特徴量データをCSVでダウンロード",
                    data=csv_data,
                    file_name="eeg_features_analysis.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            with st.expander("詳細な品質管理レポートを表示"):
                st.dataframe(qc_stats, use_container_width=True)
        
        if processed_trials:
            tab_list = [" raw データ検査", "🔧 前処理結果"]
            if features_df is not None and not features_df.empty:
                tab_list.append("📈 統計解析")
            
            tabs = st.tabs(tab_list)
            subject_list = sorted(list(qc_stats['subject_id'].unique())) if qc_stats is not None and not qc_stats.empty else []

            with tabs[0]:
                st.header("Rawデータインスペクター")
                if subject_list:
                    selected_subject_raw = st.selectbox("被験者を選択", subject_list, key="raw_subject_selector")
                    trials_for_subject_raw = [t for t in processed_trials if t.subject_id == selected_subject_raw]
                    if trials_for_subject_raw:
                        trial_ids_raw = [t.trial_id for t in trials_for_subject_raw]
                        selected_trial_id_raw = st.selectbox("試行を選択", trial_ids_raw, key="raw_trial_selector")
                        selected_trial_raw = next((t for t in trials_for_subject_raw if t.trial_id == selected_trial_id_raw), None)
                        if selected_trial_raw:
                            fig_raw = plot_raw_signal_inspector(selected_trial_raw, config)
                            st.plotly_chart(fig_raw, use_container_width=True)
                else: st.warning("表示できる被験者データがありません。")

            with tabs[1]:
                st.header("前処理と品質管理の視覚化")
                valid_subjects = sorted(list(qc_stats[qc_stats['is_valid']]['subject_id'].unique())) if qc_stats is not None and not qc_stats.empty else []
                if valid_subjects:
                    selected_subject_qc = st.selectbox("有効な試行がある被験者を選択", valid_subjects, key="qc_subject_selector")
                    valid_trials = [t for t in processed_trials if t.subject_id == selected_subject_qc and t.is_valid]
                    if valid_trials:
                        trial_ids_qc = [t.trial_id for t in valid_trials]
                        selected_trial_id_qc = st.selectbox("有効な試行を選択", trial_ids_qc, key="qc_trial_selector")
                        selected_trial_qc = next((t for t in valid_trials if t.trial_id == selected_trial_id_qc), None)
                        if selected_trial_qc:
                            fig_qc = plot_signal_qc(selected_trial_qc, config)
                            st.plotly_chart(fig_qc, use_container_width=True)
                else: st.warning("表示できる有効な試行がありません。")

            if len(tabs) > 2:
                with tabs[2]:
                    st.header("特徴量の統計的比較")
                    st.info("この統計解析は、アップロードされた全被験者の有効な試行データを統合して行われます。")
                    col1, col2 = st.columns(2)
                    with col1:
                        feature_options = sorted(features_df.columns.drop(['subject_id', 'trial_id', 'preference', 'valence', 'arousal'], errors='ignore'))
                        feature_to_analyze = st.selectbox("1. 分析したい脳波特徴量を選択", feature_options)
                    with col2:
                        analysis_options = ["好き/嫌い/そうでもない (グループ比較)"]
                        if 'valence' in features_df.columns and features_df['valence'].notna().any(): analysis_options.append("Valenceスコア (相関分析)")
                        if 'arousal' in features_df.columns and features_df['arousal'].notna().any(): analysis_options.append("Arousalスコア (相関分析)")
                        analysis_choice = st.selectbox("2. 比較したい評価軸を選択", analysis_options)
                    
                    if "グループ比較" in analysis_choice:
                        stats_results = run_statistical_analysis(features_df, feature_to_analyze, "group")
                        fig_dist = plot_feature_distribution(features_df, feature_to_analyze)
                        st.plotly_chart(fig_dist, use_container_width=True)
                        st.subheader("統計検定結果 (ANOVA / t-test)")
                        p_val = stats_results.get('p_value')
                        st.metric("p値", f"{p_val:.4f}" if p_val is not None else "N/A", help="グループ間に統計的に意味のある差があるかを示します (p < 0.05が一般的基準)")
                    else:
                        target_col = 'valence' if 'Valence' in analysis_choice else 'arousal'
                        stats_results = run_statistical_analysis(features_df, feature_to_analyze, "correlation", target_col)
                        fig_corr = plot_feature_correlation(features_df, feature_to_analyze, target_col, stats_results)
                        st.plotly_chart(fig_corr, use_container_width=True)
                        st.subheader(f"統計検定結果 ({target_col}とのピアソン相関)")
                        res_col1, res_col2 = st.columns(2)
                        r_val, p_val = stats_results.get('corr_coef'), stats_results.get('p_value')
                        res_col1.metric("相関係数 (r)", f"{r_val:.3f}" if r_val is not None else "N/A", help="関係の強さと方向を示します (-1から+1)")
                        res_col2.metric("p値", f"{p_val:.4f}" if p_val is not None else "N/A", help="この相関が偶然でないかを示します (p < 0.05が一般的基準)")
    
    except Exception as e:
        st.error("アプリケーションの表示中に予期せぬエラーが発生しました。")
        st.exception(e)
else:
    st.info("👈 左側のサイドバーからEEGファイル等をアップロードし、「解析実行」ボタンを押してください。")

# --- フッター ---
st.markdown("---")
st.markdown("<div style='text-align: center; color: #888;'>🧠 EEG画像嗜好解析システム v2.3 (UI-Refined)</div>", unsafe_allow_html=True)
