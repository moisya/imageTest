# src/auth.py
import streamlit as st

def check_password():
    """Returns `True` if the user had the correct password."""

    # st.session_state を使って、一度認証されたらパスワード入力をスキップ
    if st.session_state.get("password_correct", False):
        return True

    # secrets.toml からパスワードを取得
    try:
        correct_password = st.secrets["PASSWORD"]
    except KeyError:
        st.error("パスワードが設定されていません。管理者にお問い合わせください。")
        return False

    # パスワード入力フォーム
    with st.form("password_form"):
        st.title("🔒 認証")
        st.markdown("このアプリケーションにアクセスするにはパスワードが必要です。")
        password = st.text_input("パスワード", type="password")
        submitted = st.form_submit_button("認証")

        if submitted:
            if password == correct_password:
                st.session_state["password_correct"] = True
                st.rerun()  # 認証後にページを再読み込みしてフォームを消す
            else:
                st.error("パスワードが間違っています。")
    
    return False
