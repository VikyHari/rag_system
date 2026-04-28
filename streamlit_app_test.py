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

st.set_page_config(page_title="RAG Document Assistant", layout="wide", initial_sidebar_state="collapsed")
init_db()

DEFAULTS = {"authenticated": False, "user_id": None, "username": None, "token": None, "upload_complete": False, "upload_id": None, "namespace": None, "active_tab": "upload", "show_source_popup": False, "chat_messages": []}
for key, val in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val