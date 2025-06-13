# app.py
# (インポートとパスワード認証は同じ)
# ...

with st.sidebar:
    st.header("⚙️ 解析設定")
    
    st.subheader("📁 データファイル")
    # ★★★ UIを更新 ★★★
    uploaded_eeg_file = st.file_uploader(
        "1. EEGファイルをアップロード (.xdf)",
        type=['xdf'],
        accept_multiple_files=False, # シンプルにするため一旦単一ファイルに
        help="被験者1名分のXDFファイルをアップロードします。"
    )
    uploaded_survey_file = st.file_uploader(
        "2. 評価データをアップロード (.csv, .xlsx)",
        type=['csv', 'xlsx'],
        help="'trial_id'と評価スコア列を含むファイル"
    )

    # ... (パラメータ設定は同じ) ...
    run_analysis = st.button("🚀 解析実行", type="primary", use_container_width=True)

# --- 解析パイプライン関数を修正 ---
@st.cache_data(show_spinner="解析パイプラインを実行中...")
def run_full_pipeline(_uploaded_eeg_file, _uploaded_survey_file, _config):
    # アップロードされたファイルがNoneでないことを確認し、リストに入れる
    eeg_files_list = [_uploaded_eeg_file] if _uploaded_eeg_file else []
    if not eeg_files_list:
        return None, None, None, "EEGファイルがアップロードされていません。"
    
    all_trials, _ = load_all_trial_data(eeg_files_list, _uploaded_survey_file, _config)
    # ... (以降のロジックは同じ) ...
    if not all_trials:
        # ...
        pass
    # ...

# --- メインエリアのロジックを修正 ---
if not uploaded_eeg_file:
    st.info("👈 左側のサイドバーからEEGファイルと評価データをアップロードして解析を開始してください。")
elif run_analysis:
    # ... (configオブジェクト作成は同じ) ...
    qc_stats, features_df, processed_trials, error_message = run_full_pipeline(
        uploaded_eeg_file, uploaded_survey_file, config
    )
    # ... (以降の表示ロジックは同じ) ...
