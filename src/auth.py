# src/auth.py
import streamlit as st
import os

def check_password() -> bool:
    """Streamlitアプリにシンプルなパスワード認証を実装する"""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return True

    try:
        # Streamlit CloudのSecretsからパスワードを取得
        expected_password = st.secrets["APP_PASSWORD"]
    except (FileNotFoundError, KeyError):
        # Secretsがない場合、環境変数またはデフォルト値を使用
        st.warning("Streamlit CloudのSecretsに'APP_PASSWORD'が設定されていません。開発用のデフォルトパスワードを使用します。")
        expected_password = os.getenv("APP_PASSWORD", "eeg2024")

    with st.form("password_form"):
        st.title("🔒 アプリケーション認証")
        password = st.text_input("パスワード", type="password")
        submitted = st.form_submit_button("ログイン")

        if submitted:
            if password == expected_password:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("パスワードが正しくありません。")
    
    return False
