"""
Streamlit UI - RAG Application
"""

import streamlit as st

from db import init_db
from rest_user import (
    UserCreate,
    register_user,
    authenticate_user,
)
from rag_system import (
    process_pdf_upload,
    process_s3_connection,
    ask_question,
)

st.set_page_config(
    page_title="RAG Document Assistant",
    layout="wide",
    initial_sidebar_state="collapsed",
)

init_db()

DEFAULTS = {
    "authenticated": False,
    "user_id": None,
    "username": None,
    "token": None,
    "upload_complete": False,
    "upload_id": None,
    "namespace": None,
    "active_tab": "upload",
    "show_source_popup": False,
    "chat_messages": [],
}
for key, val in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val

st.markdown("""
<style>
    .main-header { font-size: 2rem; font-weight: 700; text-align: center; padding: 1rem 0; }
    .status-ready { color: #28a745; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


def render_login_page():
    st.markdown('<div class="main-header">RAG Document Assistant</div>', unsafe_allow_html=True)
    st.markdown("---")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        auth_tab = st.radio("", ["Login", "Register"], horizontal=True, label_visibility="collapsed")

        if auth_tab == "Login":
            with st.form("login_form"):
                st.subheader("Welcome Back")
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Sign In", use_container_width=True)

                if submitted:
                    if not username or not password:
                        st.error("Please fill in all fields.")
                    else:
                        result = authenticate_user(username, password)
                        if result:
                            st.session_state.authenticated = True
                            st.session_state.user_id = result["user_id"]
                            st.session_state.username = result["username"]
                            st.session_state.token = result["access_token"]
                            st.session_state.show_source_popup = True
                            st.rerun()
                        else:
                            st.error("Invalid username or password.")

        else:
            with st.form("register_form"):
                st.subheader("Create Account")
                username = st.text_input("Username")
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                confirm = st.text_input("Confirm Password", type="password")
                submitted = st.form_submit_button("Register", use_container_width=True)

                if submitted:
                    if not all([username, email, password, confirm]):
                        st.error("Please fill in all fields.")
                    elif password != confirm:
                        st.error("Passwords do not match.")
                    elif len(password) < 4:
                        st.error("Password must be at least 4 characters.")
                    else:
                        result = register_user(UserCreate(
                            username=username, email=email, password=password,
                        ))
                        if result["success"]:
                            st.success("Account created! Please sign in.")
                        else:
                            st.error(result["message"])


def render_source_popup():
    @st.dialog("Choose Your Data Source")
    def source_dialog():
        st.write("How would you like to add your data?")
        col_a, col_b = st.columns(2)

        with col_a:
            if st.button("Upload a PDF", use_container_width=True):
                st.session_state.show_source_popup = False
                st.session_state.active_tab = "upload"
                st.rerun()

        with col_b:
            if st.button("Connect to AWS", use_container_width=True):
                st.session_state.show_source_popup = False
                st.session_state.active_tab = "aws"
                st.rerun()

    source_dialog()


def render_upload_tab():
    st.subheader("Upload a PDF Document")
    st.write("Upload your PDF and the system will extract, chunk, embed, and index its contents.")

    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"], accept_multiple_files=False)

    if uploaded_file is not None:
        file_size_kb = uploaded_file.size / 1024
        st.info(f"File: **{uploaded_file.name}** ({file_size_kb:.1f} KB)")

        if st.button("Process & Index", type="primary", use_container_width=True):
            with st.spinner("Extracting text, chunking, embedding, and storing..."):
                result = process_pdf_upload(
                    user_id=st.session_state.user_id,
                    filename=uploaded_file.name,
                    pdf_bytes=uploaded_file.read(),
                )

            if result["status"] == "ready":
                st.session_state.upload_complete = True
                st.session_state.upload_id = result["upload_id"]
                st.session_state.namespace = result["namespace"]
                st.success(result["message"])
                st.balloons()
                st.session_state.active_tab = "ask"
                st.rerun()
            else:
                st.error(result["message"])


def render_aws_tab():
    st.subheader("Connect to AWS S3")
    st.write("Provide an S3 prefix to index all supported files from your bucket.")

    with st.form("aws_form"):
        bucket = st.text_input("S3 Bucket (leave blank for default from .env)")
        prefix = st.text_input("S3 Prefix / Folder", value="")
        submitted = st.form_submit_button("Connect & Index", use_container_width=True)

    if submitted:
        with st.spinner("Connecting to S3 and processing files..."):
            result = process_s3_connection(
                user_id=st.session_state.user_id,
                s3_prefix=prefix,
                bucket=bucket if bucket else None,
            )

        if result["status"] == "ready":
            st.session_state.upload_complete = True
            st.session_state.upload_id = result["upload_id"]
            st.session_state.namespace = result["namespace"]
            st.success(result["message"])
            st.balloons()
            st.session_state.active_tab = "ask"
            st.rerun()
        else:
            st.error(result["message"])


def render_ask_tab():
    st.subheader("Ask Questions About Your Document")

    if not st.session_state.upload_complete:
        st.warning("Please upload a document first before asking questions.")
        return

    llm_choice = st.selectbox(
        "Choose LLM",
        ["qwen", "gemini"],
        index=0,
        help="Qwen runs locally via Ollama (free). Gemini requires an API key.",
    )

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if prompt := st.chat_input("Ask a question about your document..."):
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching and generating answer..."):
                result = ask_question(
                    user_id=st.session_state.user_id,
                    upload_id=st.session_state.upload_id,
                    question=prompt,
                    llm_name=llm_choice,
                )

            st.write(result["answer"])

            sources = result.get("sources", [])
            if sources:
                num = len(sources)
                with st.expander(f"View {num} source chunks"):
                    for i, src in enumerate(sources):
                        score_val = src["score"]
                        st.markdown(f"**Chunk {i+1}** (score: {score_val:.3f})")
                        text_preview = src["text"][:300]
                        if len(src["text"]) > 300:
                            text_preview = text_preview + "..."
                        st.text(text_preview)
                        st.markdown("---")

            llm_used = result["llm_used"]
            st.caption(f"Answered by: {llm_used}")

        st.session_state.chat_messages.append({
            "role": "assistant",
            "content": result["answer"],
        })


def render_sidebar():
    with st.sidebar:
        uname = st.session_state.username
        st.markdown(f"### Hello, {uname}\!")
        st.markdown("---")

        if st.session_state.upload_complete:
            st.markdown('<span class="status-ready">Document indexed</span>', unsafe_allow_html=True)
            uid = st.session_state.upload_id
            st.caption(f"Upload ID: {uid}")

        st.markdown("---")

        if st.button("Upload New Document"):
            st.session_state.upload_complete = False
            st.session_state.upload_id = None
            st.session_state.namespace = None
            st.session_state.chat_messages = []
            st.session_state.show_source_popup = True
            st.rerun()

        if st.button("Logout"):
            for key in DEFAULTS:
                st.session_state[key] = DEFAULTS[key]
            st.rerun()


def main():
    if not st.session_state.authenticated:
        render_login_page()
        return

    if st.session_state.show_source_popup:
        render_source_popup()
        return

    render_sidebar()

    tab_labels = ["Upload PDF", "Ask Questions"]
    if st.session_state.active_tab == "aws":
        tab_labels = ["Connect to AWS", "Ask Questions"]

    tab1, tab2 = st.tabs(tab_labels)

    with tab1:
        if st.session_state.active_tab == "aws":
            render_aws_tab()
        else:
            render_upload_tab()

    with tab2:
        render_ask_tab()


if __name__ == "__main__":
    main()
