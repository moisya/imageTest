# app.py
# ... (インポート部分は同じ) ...
# from io_module import load_all_trial_data

with st.sidebar:
    st.header("⚙️ 解析設定")
    
    st.subheader("📁 データファイル")
    uploaded_eeg_files = st.file_uploader(
        "EEGファイルをアップロード (.xdf, .edf, .bdf, .fif)",
        type=['xdf', 'edf', 'bdf', 'fif'],
        accept_multiple_files=True,
        help="複数の被験者のEEGファイルを同時にアップロードできます。"
    )
    # ★★★ アンケートファイル用アップローダーを追加 ★★★
    uploaded_survey_file = st.file_uploader(
        "アンケート結果CSVをアップロード (任意)",
        type=['csv'],
        help="'subject_id'と'trial_id'列を含むCSVファイル"
    )

    # ... (パラメータ設定は同じ) ...
    run_analysis = st.button("🚀 解析実行", type="primary", use_container_width=True)

# 解析パイプライン関数を修正
@st.cache_data(show_spinner="解析パイプラインを実行中...")
def run_full_pipeline(_uploaded_eeg_files, _uploaded_survey_file, _config):
    # ★★★ 引数を追加 ★★★
    all_trials, meta_info = load_all_trial_data(_uploaded_eeg_files, _uploaded_survey_file, _config)
    # ... (以降のロジックは同じ) ...
    if not all_trials:
        return None, None, None, "有効な試行データが読み込めませんでした。ファイル形式や内容を確認してください。"
    
    processed_trials, qc_stats = run_preprocessing_pipeline(all_trials, _config)
    features_df = extract_all_features(processed_trials, _config)
    
    if features_df.empty:
        return qc_stats, None, None, "有効な試行が全て除去されたため、特徴量を抽出できませんでした。品質管理の閾値を調整してください。"

    return qc_stats, features_df, processed_trials, None


# メインエリアのロジックを修正
if not uploaded_eeg_files:
    st.info("👈 左側のサイドバーからEEGファイルをアップロードして解析を開始してください。")
elif run_analysis:
    # ... (configオブジェクト作成は同じ) ...
    config = AppConfig(...)
    
    # ★★★ 引数を渡してパイプラインを実行 ★★★
    qc_stats, features_df, processed_trials, error_message = run_full_pipeline(
        uploaded_eeg_files, uploaded_survey_file, config
    )
    
    # ... (以降の表示ロジックは同じ) ...
    if error_message:
        st.error(error_message)
    elif qc_stats is not None:
        # ...
        pass
