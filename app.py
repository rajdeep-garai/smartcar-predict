from pathlib import Path
import sys
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
import io
import time
import hmac
from pymongo import MongoClient
from pymongo.errors import PyMongoError
import bcrypt

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import load_dataset, find_target_column, DATASET_PATH
from src.preprocessing import clean_and_engineer
from src.predict import load_model, predict_price
from src.eda import (
    price_distribution,
    brand_average_price,
    categorical_distribution,
    year_price_trend,
    mileage_price,
    correlation_heatmap,
)

st.set_page_config(
    page_title="SMARTCAR",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if "ui_theme" not in st.session_state:
    st.session_state["ui_theme"] = "light"


def mongo_credentials():
    try:
        uri = str(st.secrets["MONGODB_URI"]).strip()
        database = str(st.secrets.get("MONGODB_DATABASE", "smartcar")).strip()
        collection = str(st.secrets.get("MONGODB_COLLECTION", "users")).strip()
    except Exception:
        return None
    if not uri:
        return None
    return uri, database, collection


@st.cache_resource(ttl=300)
def mongo_client(uri: str):
    return MongoClient(uri, serverSelectionTimeoutMS=8000)


def authenticate_mongo(username: str, password: str) -> tuple[bool, str]:
    credentials = mongo_credentials()
    if credentials is None:
        return False, "MongoDB login is not configured."

    uri, database, collection_name = credentials
    try:
        client = mongo_client(uri)
        client.admin.command("ping")
        user = client[database][collection_name].find_one({
            "$or": [
                {"email": username},
                {"username": username},
            ]
        })
        if not user:
            return False, "Invalid username or password."

        stored_password = user.get("password")
        if stored_password is None:
            return False, "Invalid username or password."

        stored_password = str(stored_password)
        try:
            password_valid = bcrypt.checkpw(password.encode("utf-8"), stored_password.encode("utf-8"))
        except (ValueError, TypeError):
            password_valid = hmac.compare_digest(password, stored_password)

        if password_valid:
            return True, ""
        return False, "Invalid username or password."
    except PyMongoError as exc:
        return False, f"Database connection failed: {exc}"


def create_mongo_user(full_name: str, email: str, password: str) -> tuple[bool, str]:
    credentials = mongo_credentials()
    if credentials is None:
        return False, "MongoDB login is not configured."

    full_name = full_name.strip()
    email = email.strip().lower()
    if len(full_name) < 2:
        return False, "Name must contain at least 2 characters."
    if "@" not in email or "." not in email.rsplit("@", 1)[-1]:
        return False, "Enter a valid email address."
    if len(password) < 8:
        return False, "Password must contain at least 8 characters."

    uri, database, collection_name = credentials
    try:
        client = mongo_client(uri)
        users = client[database][collection_name]
        if users.find_one({"email": email}):
            return False, "An account already exists with that email."
        hashed_password = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
        users.insert_one({
            "name": full_name,
            "email": email,
            "username": email,
            "password": hashed_password,
        })
        return True, ""
    except PyMongoError as exc:
        return False, f"Database connection failed: {exc}"


def reset_mongo_password(username: str, new_password: str) -> tuple[bool, str]:
    credentials = mongo_credentials()
    if credentials is None:
        return False, "MongoDB login is not configured."

    username = username.strip()
    if len(username) < 3:
        return False, "Enter a valid username."
    if len(new_password) < 8:
        return False, "New password must contain at least 8 characters."

    uri, database, collection_name = credentials
    try:
        client = mongo_client(uri)
        users = client[database][collection_name]
        if not users.find_one({"username": username}):
            return False, "No account was found with that username."
        hashed_password = bcrypt.hashpw(new_password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
        users.update_one(
            {"username": username},
            {"$set": {"password": hashed_password}},
        )
        return True, ""
    except PyMongoError as exc:
        return False, f"Database connection failed: {exc}"


if not st.session_state["authenticated"]:
    credentials = mongo_credentials()

    st.markdown(
        """
        <style>
        [data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stSidebar"] {
            display: none;
        }
        .stApp {
            background: #FAFAF9;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        [data-testid="stMainBlockContainer"] {
            max-width: 430px;
            margin: auto;
            padding: 24px;
        }
        [data-testid="stForm"] {
            background: #08131f;
            border: 1px solid #17354d;
            border-radius: 2px;
            padding: 18px 28px 16px;
            box-shadow: 0 18px 40px rgba(4, 12, 20, 0.30);
        }
        [data-testid="stForm"] label {
            color: #A8D0E2;
            font-size: 11px;
            text-transform: uppercase;
        }
        [data-testid="stForm"] input {
            height: 42px;
            border-radius: 2px;
            border: 1px solid #D6DCE2;
            background: #FFFFFF;
            color: #111827;
        }
        [data-testid="stFormSubmitButton"] button {
            height: 42px;
            margin-top: 4px;
            border: 0;
            border-radius: 1px;
            background: #15324a;
            color: #F8FAFC;
            font-size: 12px;
            letter-spacing: 0.12em;
        }
        [data-testid="stFormSubmitButton"] button:hover {
            background: #1d4768;
            color: #FFFFFF;
        }
        .login-card-title {
            color: #A8D0E2;
            text-align: center;
            letter-spacing: 0.22em;
            font-size: 13px;
            margin: 0 0 16px;
            font-weight: 500;
        }
        .login-register-button button {
            width: 100%;
            height: 46px;
            margin-top: 10px;
            border-radius: 1px;
            background: #08131f;
            color: #A8D0E2;
            border: 1px solid #17354d;
            letter-spacing: 0.12em;
        }
        .login-options {
            color: #A8D0E2;
            font-size: 11px;
        }
        [data-testid="stButton"] button {
            font-size: 11px;
            min-height: 34px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state["ui_theme"] == "dark":
        st.markdown(
            """
            <style>
            .stApp, [data-testid="stAppViewContainer"] { background: #171717 !important; color: #F5F5F4 !important; }
            [data-testid="stSidebar"], [data-testid="stSidebarContent"] { background: #292524 !important; border-color: #44403C !important; }
            [data-testid="stMetric"], [data-testid="stTable"], [data-testid="stDataFrame"],
            [data-testid="stContainer"] { background: #292524 !important; border-color: #44403C !important; }
            [data-testid="stMetricLabel"] { color: #A8A29E !important; }
            [data-testid="stMetricValue"], h1, h2, h3, h4, p, label { color: #F5F5F4 !important; }
            [data-testid="stSidebar"] .stButton > button { color: #D6D3D1 !important; }
            [data-testid="stSidebar"] .stButton > button[kind="primary"] {
                background: #44403C !important; color: #F0FDFA !important; border-color: #57534E !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

    login_col1, login_col2, login_col3 = st.columns([1, 10, 1])
    with login_col2:
        logo_col1, logo_col2, logo_col3 = st.columns([1, 2, 1])
        with logo_col2:
            st.image("assets/smartcar-logo.png", width=220)
        st.markdown('<div class="login-card-title">USER LOGIN</div>', unsafe_allow_html=True)
        if credentials is None:
            st.error("Login is not configured.")
            st.info("Add MONGODB_URI to `.streamlit/secrets.toml`.")
            st.stop()

        if "login_mode" not in st.session_state:
            st.session_state["login_mode"] = "Sign In"

        if st.session_state["login_mode"] == "Sign In":
            with st.form("smartcar_login", clear_on_submit=False):
                username = st.text_input("Email ID", placeholder="Enter your email address")
                password = st.text_input("Password", type="password", placeholder="Enter your password")
                submitted = st.form_submit_button("LOGIN", use_container_width=True)

            if submitted:
                authenticated, message = authenticate_mongo(username.strip(), password)
                if authenticated:
                    st.session_state["authenticated"] = True
                    st.success("Login successful.")
                    st.rerun()
                else:
                    st.error(message)

            left, right = st.columns(2)
            if left.button("Remember me", use_container_width=True):
                st.session_state["remember_user"] = username
            if right.button("Forgot Password?", use_container_width=True):
                st.session_state["login_mode"] = "Forgot Password"
                st.rerun()

            st.markdown('<div class="login-register-button">', unsafe_allow_html=True)
            if st.button("REGISTER", use_container_width=True):
                st.session_state["login_mode"] = "Create Account"
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

        elif st.session_state["login_mode"] == "Create Account":
            with st.form("smartcar_create_account", clear_on_submit=False):
                full_name = st.text_input("Name", placeholder="Enter your full name")
                email = st.text_input("Email ID", placeholder="Enter your email address")
                password = st.text_input("Password", type="password", placeholder="Create a password")
                confirm_password = st.text_input("Confirm Password", type="password", placeholder="Confirm password")
                submitted = st.form_submit_button("CREATE ACCOUNT", use_container_width=True)

            if submitted:
                if password != confirm_password:
                    st.error("Passwords do not match.")
                else:
                    created, message = create_mongo_user(full_name, email, password)
                    if created:
                        st.success("Account created. You can now sign in.")
                        st.session_state["login_mode"] = "Sign In"
                        st.rerun()
                    else:
                        st.error(message)
            if st.button("BACK TO LOGIN", use_container_width=True):
                st.session_state["login_mode"] = "Sign In"
                st.rerun()

        else:
            with st.form("smartcar_reset_password", clear_on_submit=False):
                username = st.text_input("Username", placeholder="Enter your username")
                new_password = st.text_input("New Password", type="password", placeholder="Create a new password")
                confirm_password = st.text_input("Confirm New Password", type="password", placeholder="Confirm new password")
                submitted = st.form_submit_button("RESET PASSWORD", use_container_width=True)

            if submitted:
                if new_password != confirm_password:
                    st.error("Passwords do not match.")
                else:
                    reset, message = reset_mongo_password(username, new_password)
                    if reset:
                        st.success("Password reset successfully. You can now sign in.")
                        st.session_state["login_mode"] = "Sign In"
                        st.rerun()
                    else:
                        st.error(message)
            if st.button("BACK TO LOGIN", use_container_width=True):
                st.session_state["login_mode"] = "Sign In"
                st.rerun()
    st.stop()

st.markdown(
    """
    <style>
    html, body, [class*="css"] { font-family: "Inter", "Segoe UI", sans-serif; }
    html, body, #root { margin: 0 !important; padding: 0 !important; }
    [data-testid="stHeader"] { height: 0 !important; min-height: 0 !important; display: none !important; }
    [data-testid="stToolbar"] { display: none !important; }
    [data-testid="stDecoration"] { display: none !important; }
    .stApp { background: #FAFAF9; color: #292524; }
    [data-testid="stSidebar"],
    section[data-testid="stSidebar"] {
        position: relative !important;
        left: 0 !important;
        top: 0 !important;
        right: auto !important;
        margin: 0 !important;
        padding: 0 !important;
        background: #FFFFFF;
        width: 150px !important;
        min-width: 150px !important;
        max-width: 150px !important;
        border-right: 1px solid #E7E5E4;
        box-shadow: 4px 0 24px rgba(41, 37, 36, 0.04);
        z-index: 100;
        transform: none !important;
    }
    [data-testid="stSidebarContent"] {
        width: 150px;
        max-width: 150px;
        margin: 0;
        padding: 6px 8px 10px;
    }
    [data-testid="stSidebarUserContent"] {
        height: calc(100vh - 1px);
        overflow-y: auto;
        overflow-x: hidden;
    }
    [data-testid="stAppViewContainer"] {
        margin-left: 0;
        width: 100%;
    }
    [data-testid="stMainBlockContainer"] {
        max-width: none;
        padding-left: 2.5rem;
        padding-right: 2.5rem;
        overflow-x: hidden;
    }
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        align-items: center;
    }
    [data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] {
        width: 134px;
        max-width: 134px;
    }
    [data-testid="stSidebar"] .stButton {
        width: 134px;
        margin: 1px 0;
    }
    [data-testid="stSidebar"] .stButton > button {
        width: 134px;
        min-height: 42px;
        height: 42px;
        border: 1px solid transparent;
        border-radius: 12px;
        background: transparent;
        color: #57534E;
        display: flex;
        flex-direction: row;
        align-items: center;
        justify-content: center;
        gap: 7px;
        padding: 4px 8px;
        box-shadow: none;
        white-space: normal;
        line-height: 1.1;
        font-size: 14px;
        text-align: center;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: #F0FDFA;
        border-color: #CCFBF1;
        color: #0F766E;
    }
    [data-testid="stSidebar"] .stButton > button[kind="secondary"] {
        background: transparent;
    }
    [data-testid="stSidebar"] .stButton > button[kind="primary"] {
        background: #F0FDFA;
        color: #0F766E;
        border-color: #CCFBF1;
        box-shadow: none;
    }
    @media (max-width: 768px) {
        [data-testid="stSidebar"],
        section[data-testid="stSidebar"] {
            width: 150px !important;
            min-width: 150px !important;
            max-width: 150px !important;
        }
        [data-testid="stSidebarContent"] {
            width: 150px;
            max-width: 150px;
            padding: 6px 8px 10px;
        }
        [data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"],
        [data-testid="stSidebar"] .stButton,
        [data-testid="stSidebar"] .stButton > button {
            width: 134px !important;
            max-width: 134px !important;
        }
    }
        [data-testid="stSidebar"] .theme-actions {
            display: flex;
            justify-content: center;
            gap: 8px;
            width: 134px;
            margin: 8px auto 0;
        }
    [data-testid="stSidebar"] [data-testid="stColumn"] {
        width: 36px !important;
        min-width: 36px !important;
    }
    [data-testid="stSidebar"] [data-testid="stColumn"] .stButton,
    [data-testid="stSidebar"] [data-testid="stColumn"] .stButton > button {
        width: 36px !important;
        min-width: 36px !important;
        max-width: 36px !important;
        height: 36px !important;
        min-height: 36px !important;
        margin: 0 !important;
        padding: 0 !important;
        line-height: 36px;
        font-size: 14px;
        border-radius: 8px;
    }
    [data-testid="stSidebar"] [data-testid="stColumn"] .stButton > button:active {
        transform: scale(0.94);
        background: #CCFBF1 !important;
        color: #0F766E !important;
    }
    [data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #E7E5E4;
        border-radius: 14px;
        padding: 16px;
        box-shadow: 0 10px 32px rgba(41, 37, 36, 0.06);
    }
    [data-testid="stMetricLabel"] { color: #A8A29E; }
    [data-testid="stMetricValue"] { color: #292524; }
    [data-testid="stContainer"] { border-radius: 14px; }
    [data-testid="stImage"] img { border-radius: 16px; border: 1px solid #E7E5E4;
        box-shadow: 0 10px 24px rgba(41,37,36,.08); }
    [data-testid="stTextInput"] input,
    [data-testid="stNumberInput"] input,
    [data-testid="stSelectbox"] div[data-baseweb="select"] > div {
        border-radius: 10px !important;
        border-color: #D6D3D1 !important;
    }
    .st-key-prediction_form [data-testid="stForm"] {
        background: transparent;
        border: 1px solid #E7E5E4;
        border-radius: 14px;
        padding: 18px;
        overflow: visible;
    }
    .st-key-prediction_form [data-testid="stSelectbox"] label,
    .st-key-prediction_form [data-testid="stNumberInput"] label {
        font-size: 13px;
        font-weight: 700;
        line-height: 1.25;
        min-height: 18px;
        margin-bottom: 4px;
        white-space: normal;
        overflow-wrap: anywhere;
    }
    .st-key-prediction_form [data-testid="stSelectbox"] div[data-baseweb="select"] > div,
    .st-key-prediction_form [data-testid="stNumberInput"] {
        min-height: 46px;
        height: 46px;
        border-radius: 10px !important;
    }
    .st-key-prediction_form [data-testid="stNumberInput"] input {
        min-height: 44px;
        height: 44px;
        border-radius: 10px 0 0 10px !important;
    }
    .st-key-prediction_form [data-testid="stNumberInput"] button {
        min-width: 36px;
        width: 36px;
        height: 44px;
        border-radius: 8px;
    }
    .st-key-predict_prediction_price button {
        background: #0F766E !important;
        color: #FFFFFF !important;
        border: none !important;
        min-height: 46px;
        height: 46px;
        border-radius: 10px;
        transition: background-color 160ms ease, box-shadow 160ms ease;
    }
    .st-key-predict_prediction_price button:hover {
        background: #115E59 !important;
        box-shadow: 0 4px 12px rgba(15, 118, 110, 0.22);
    }
    [data-testid="stTable"], [data-testid="stDataFrame"] {
        border: 1px solid #E7E5E4;
        border-radius: 12px;
        overflow: hidden;
    }
    .stButton > button, .stFormSubmitButton > button {
        border-radius: 10px;
        border: 1px solid #B45309;
        min-height: 44px;
        font-weight: 700;
    }
    .stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] {
        background: linear-gradient(135deg, #B45309, #92400E);
        color: #FFFFFF;
        border: none;
    }
    .stSelectbox div[data-baseweb="select"] > div,
    .stNumberInput div[data-baseweb="input"] {
        border-radius: 10px;
    }
    .smartcar-hero {
        background: linear-gradient(120deg, #FFFFFF, #F5F5F4);
        border: 1px solid #E7E5E4;
        border-radius: 18px;
        padding: 24px 28px;
        margin-bottom: 18px;
    }
    .smartcar-hero h2, .smartcar-hero h3 { margin: 0; }
    .smartcar-hero p { color: #A8A29E; margin: 8px 0 0 0; }
    .luxury-price-card {
        background: linear-gradient(135deg, #292524 0%, #B45309 100%);
        border-radius: 18px;
        color: #FFFFFF;
        padding: 26px;
        text-align: center;
        box-shadow: 0 16px 40px rgba(180, 83, 9, 0.20);
    }
    .luxury-price-card .label { color: #F5F5F4; font-size: 0.95rem; }
    .luxury-price-card .value { font-size: 2.25rem; font-weight: 800; letter-spacing: -0.04em;
        animation: priceReveal 0.8s ease-out both; }
    .prediction-complete {
        background: #ECFDF5;
        border: 1px solid #A7F3D0;
        border-radius: 10px;
        color: #065F46;
        padding: 10px 14px;
        margin: 8px 0;
        font-weight: 600;
    }
    .st-key-reset_prediction_form button {
        background: #C2183A !important;
        color: #FFFFFF !important;
        border: none !important;
        min-height: 46px;
        height: 46px;
        border-radius: 10px;
        transition: background-color 160ms ease, box-shadow 160ms ease;
    }
    .st-key-reset_prediction_form button:hover {
        background: #9F1239 !important;
        color: #FFFFFF !important;
        box-shadow: 0 4px 12px rgba(159, 18, 57, 0.24);
    }
    .st-key-reset_prediction_form button p,
    .st-key-reset_prediction_form button span,
    .st-key-reset_prediction_form button div {
        color: #FFFFFF !important;
    }
    @keyframes priceReveal {
        from { opacity: 0; transform: translateY(12px) scale(0.96); }
        to { opacity: 1; transform: translateY(0) scale(1); }
    }
    @media (max-width: 700px) {
        .luxury-price-card .value { font-size: 1.65rem; }
        [data-testid="stHorizontalBlock"] { flex-wrap: wrap; }
        [data-testid="stMainBlockContainer"] { padding-left: 1rem; padding-right: 1rem; }
        .st-key-prediction_form [data-testid="stForm"] { padding: 12px; }
        .st-key-prediction_form [data-testid="stHorizontalBlock"] {
            flex-direction: column;
            gap: 12px !important;
        }
        .st-key-prediction_form [data-testid="stColumn"] {
            width: 100% !important;
            min-width: 100% !important;
        }
        .st-key-prediction_form [data-testid="stSelectbox"] div[data-baseweb="select"] > div,
        .st-key-prediction_form [data-testid="stNumberInput"] {
            width: 100%;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if st.session_state["ui_theme"] == "dark":
    st.markdown(
        """
        <style>
        .stApp, [data-testid="stAppViewContainer"] {
            background: #171717 !important;
            color: #F5F5F4 !important;
        }
        [data-testid="stSidebar"], [data-testid="stSidebarContent"] {
            background: #292524 !important;
            border-color: #44403C !important;
        }
        [data-testid="stMetric"], [data-testid="stTable"], [data-testid="stDataFrame"],
        [data-testid="stContainer"], [data-testid="stVerticalBlockBorderWrapper"] {
            background: #292524 !important;
            border-color: #44403C !important;
        }
        [data-testid="stMetricLabel"] { color: #A8A29E !important; }
        [data-testid="stMetricValue"], h1, h2, h3, h4, h5, h6, p, label, span {
            color: #F5F5F4 !important;
        }
        [data-testid="stMarkdownContainer"],
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li,
        [data-testid="stMarkdownContainer"] strong,
        [data-testid="stCaptionContainer"] {
            color: #F5F5F4 !important;
        }
        [data-testid="stSidebar"] .stButton > button {
            color: #D6D3D1 !important;
            background: transparent;
        }
        [data-testid="stSidebar"] .stButton > button[kind="primary"] {
            background: #44403C !important;
            color: #F0FDFA !important;
            border-color: #57534E !important;
        }
        [data-testid="stSidebar"] .stButton > button:hover {
            background: #3F3A36 !important;
            color: #CCFBF1 !important;
        }
        [data-testid="stSidebar"] .stButton > button p,
        [data-testid="stSidebar"] .stButton > button span,
        [data-testid="stSidebar"] .stButton > button div {
            color: inherit !important;
        }
        .prediction-complete {
            background: #064E3B !important;
            border-color: #047857 !important;
            color: #ECFDF5 !important;
        }
        .st-key-prediction_form [data-testid="stForm"] {
            background: #292524 !important;
            border-color: #44403C !important;
        }
        .stButton > button,
        .stFormSubmitButton > button {
            color: #F5F5F4 !important;
        }
        .stButton > button[kind="secondary"],
        .stFormSubmitButton > button[kind="secondary"] {
            background: #292524 !important;
            color: #F5F5F4 !important;
            border-color: #57534E !important;
        }
        .stButton > button[kind="primary"],
        .stFormSubmitButton > button[kind="primary"] {
            color: #FFFFFF !important;
        }
        [data-testid="stDownloadButton"] button {
            background: #B45309 !important;
            color: #FFFFFF !important;
            border: 1px solid #D97706 !important;
        }
        [data-testid="stDownloadButton"] button p,
        [data-testid="stDownloadButton"] button span,
        [data-testid="stDownloadButton"] button div {
            color: #FFFFFF !important;
        }
        [data-testid="stFormSubmitButton"] button {
            color: #F5F5F4 !important;
        }
        [data-testid="stFormSubmitButton"] button[kind="secondary"] {
            background: #292524 !important;
            color: #F5F5F4 !important;
            border-color: #57534E !important;
        }
        .st-key-predict_prediction_price button {
            background: #0F766E !important;
            color: #FFFFFF !important;
            border: none !important;
            min-height: 46px;
            height: 46px;
            border-radius: 10px;
        }
        .st-key-predict_prediction_price button:hover {
            background: #115E59 !important;
            color: #FFFFFF !important;
        }
        .st-key-reset_prediction_form button {
            background: #C2183A !important;
            color: #FFFFFF !important;
            border: none !important;
            min-height: 46px;
            height: 46px;
            border-radius: 10px;
            transition: background-color 160ms ease, box-shadow 160ms ease;
        }
        .st-key-reset_prediction_form button:hover {
            background: #9F1239 !important;
            color: #FFFFFF !important;
            box-shadow: 0 4px 12px rgba(159, 18, 57, 0.24);
        }
        .st-key-reset_prediction_form button p,
        .st-key-reset_prediction_form button span,
        .st-key-reset_prediction_form button div {
            color: #FFFFFF !important;
        }
        .stButton > button p,
        .stButton > button span,
        .stButton > button div,
        .stFormSubmitButton > button p,
        .stFormSubmitButton > button span,
        .stFormSubmitButton > button div {
            color: inherit !important;
        }
        [data-testid="stTextInput"] input,
        [data-testid="stNumberInput"] input,
        [data-testid="stSelectbox"] div[data-baseweb="select"] > div {
            background: #292524 !important;
            color: #F5F5F4 !important;
            border-color: #57534E !important;
        }
        [data-testid="stTable"], [data-testid="stDataFrame"] {
            background: #292524 !important;
            border-color: #44403C !important;
        }
        [data-testid="stTable"] *, [data-testid="stDataFrame"] *,
        [role="columnheader"], [role="gridcell"], [role="rowheader"] {
            color: #F5F5F4 !important;
        }
        [data-testid="stAlert"], [data-testid="stAlert"] p,
        [data-testid="stAlert"] span, [data-testid="stNotification"],
        [data-testid="stNotification"] p, [data-testid="stNotification"] span {
            color: #F5F5F4 !important;
        }
        .smartcar-hero {
            background: linear-gradient(120deg, #292524, #1C1917) !important;
            border-color: #44403C !important;
        }
        .smartcar-hero p { color: #D6D3D1 !important; }
        .model-selection-card {
            background: #292524 !important;
            border-color: #57534E !important;
            color: #F5F5F4 !important;
        }
        .stButton > button, .stFormSubmitButton > button {
            border-color: #57534E;
        }
        [data-testid="stExpander"] {
            background: #292524;
            border-color: #44403C;
        }
        [data-testid="stExpander"] details,
        [data-testid="stExpander"] summary {
            background: #292524 !important;
            color: #F5F5F4 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(ttl=300)
def load_data():
    return load_dataset(DATASET_PATH)


def format_rupees(value):
    value = float(value)
    if value >= 10_000_000:
        return f"₹{value / 10_000_000:.2f} Cr"
    if value >= 100_000:
        return f"₹{value / 100_000:.2f} Lakh"
    return f"₹{value:,.0f}"


def find_display_column(df, logical_names):
    for col in df.columns:
        name = "".join(ch for ch in str(col).lower() if ch.isalnum())
        if any(key in name for key in logical_names):
            return col
    return None


def chart_card(title, fig):
    with st.container(border=True):
        st.subheader(title)
        if fig is None:
            st.info("This chart is unavailable because the required column was not found.")
        else:
            st.plotly_chart(fig, use_container_width=True)


@st.cache_data(ttl=86400)
def find_car_thumbnail(brand, model):
    api_key = None
    try:
        api_key = st.secrets["CARSXE_API_KEY"]
    except Exception:
        api_key = None

    if not api_key or str(api_key).startswith("CARSXE_"):
        return None

    make = str(brand).strip()
    model_name = str(model).strip()

    try:
        response = requests.get(
            "https://api.carsxe.com/images",
            params={
                "key": str(api_key),
                "make": make,
                "model": model_name,
                "validate": "true",
            },
            timeout=12,
            headers={"Accept": "application/json"},
        )
        if not response.ok:
            return {"error": f"CarsXE returned HTTP {response.status_code}"}

        data = response.json()
        if not isinstance(data, dict) or not data.get("success", True):
            message = data.get("message") or data.get("error") or "CarsXE rejected the image request"
            return {"error": str(message)}

        images = data.get("images", [])
        if not isinstance(images, list):
            return {"error": "CarsXE returned an unexpected images response"}
        for image in images:
            if not isinstance(image, dict):
                continue
            url = (
                image.get("link")
                or image.get("thumbnailLink")
                or image.get("image")
                or image.get("imageUrl")
                or image.get("url")
                or image.get("thumbnail")
            )
            if url:
                return {"url": url}
    except (requests.RequestException, ValueError) as exc:
        return {"error": f"CarsXE request failed: {exc}"}
    return {"error": "No exact image match was found"}


def show_dataset_status(df):
    target = find_target_column(df)
    if target is None:
        st.error("Could not identify a selling price target column. Supported names include selling_price, price, and selling price.")


with st.sidebar:
    st.markdown('<div style="display:flex;justify-content:center;margin:0 0 8px;">', unsafe_allow_html=True)
    st.image("assets/smartcar-logo.png", width=150)
    st.markdown('</div>', unsafe_allow_html=True)
    current_page = st.session_state.get("page", "Home Dashboard")
    nav_items = [
        ("🏠  Home", "Home Dashboard"),
        ("🚙  Search", "Car Price Prediction"),
        ("📊  Analytics", "Data Analytics"),
        ("⚙️  Models", "Model Performance"),
    ]
    for label, page_name in nav_items:
        if st.button(label, key=f"nav_{page_name}", use_container_width=True,
                     type="primary" if page_name == current_page else "secondary"):
            st.session_state["page"] = page_name
            st.rerun()
    page = current_page
    st.session_state["page"] = page
    st.markdown('<div class="theme-actions">', unsafe_allow_html=True)
    theme_col1, theme_col2 = st.columns(2)
    if theme_col1.button("☀", key="theme_light",
                         type="primary" if st.session_state["ui_theme"] == "light" else "secondary"):
        st.session_state["ui_theme"] = "light"
        st.rerun()
    if theme_col2.button("◐", key="theme_dark",
                         type="primary" if st.session_state["ui_theme"] == "dark" else "secondary"):
        st.session_state["ui_theme"] = "dark"
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    if st.button("Logout", key="logout", use_container_width=True):
        st.session_state["authenticated"] = False
        st.rerun()


try:
    df = load_data()
    target = find_target_column(df)
    if not target:
        st.error("Could not identify a selling price target column.")
        st.stop()
except FileNotFoundError as exc:
    st.title("SMARTCAR")
    st.error(str(exc))
    st.info("Place the downloaded CSV at: dataset/car_data.csv")
    st.stop()
except Exception as exc:
    st.title("SMARTCAR")
    st.error(f"Unable to load dataset: {exc}")
    st.stop()

show_dataset_status(df)


try:
    clean_df, info = clean_and_engineer(df, target)
except Exception as exc:
    clean_df = df
    info = {"features": [], "target": target}
    st.warning(f"Training data preparation warning: {exc}")

artifact = None
try:
    artifact = load_model()
except (FileNotFoundError, ValueError) as exc:
    artifact_error = str(exc)
else:
    artifact_error = None


if page == "Home Dashboard":
    st.markdown(
        '<div class="smartcar-hero">'
        '<h2>Intelligent Used Car Valuation Powered by Machine Learning</h2>'
        '<p>A data-driven estimate based on the uploaded Indian used-car dataset.</p></div>',
        unsafe_allow_html=True,
    )

    brand_col = find_display_column(df, ("brand", "make"))
    brand_count = df[brand_col].nunique(dropna=True) if brand_col else "N/A"
    model_name = artifact["metadata"]["selected_model"] if artifact else "Not trained"
    metrics = artifact["metadata"]["test_metrics"] if artifact else None

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", f"{len(df):,}", border=True)
    col2.metric("Average Price", format_rupees(pd.to_numeric(df[target], errors="coerce").mean()), border=True)
    col3.metric("Car Brands", f"{brand_count}", border=True)
    col4.metric("Selected Model", model_name, border=True)

    st.divider()
    left, right = st.columns([1.3, 1])
    with left:
        chart_card("Price Distribution", price_distribution(clean_df, target))
    with right:
        st.markdown("#### Project Overview")
        st.write(
            "SMARTCAR uses CatBoost as the main model and compares it with Random Forest "
            "and XGBoost using 3-fold cross-validation and measured test performance."
        )
        if metrics:
            m1, m2, m3 = st.columns(3)
            m1.metric("MAE", format_rupees(metrics["MAE"]))
            m2.metric("RMSE", format_rupees(metrics["RMSE"]))
            m3.metric("R²", f"{metrics['R2']:.3f}")
        else:
            st.info("Train the model to view measured evaluation metrics.")
            if st.button("Train Model"):
                st.switch_page("app.py")
        if st.button("Go to Price Prediction", type="primary"):
            st.session_state["page"] = "Car Price Prediction"
            st.rerun()


elif page == "Car Price Prediction":
    st.markdown("## Car Price Prediction")
    if artifact is None:
        st.error(artifact_error or "No trained model is available.")
        st.code("python -m src.train", language="powershell")
        st.stop()

    metadata = artifact["metadata"]
    features = [c for c in metadata["features"] if not str(c).startswith("_derived_")]
    categories = metadata["categories"]
    ranges = metadata["numeric_ranges"]

    brand_feature = find_display_column(clean_df, ("_derived_brand", "brand", "make"))
    car_name_feature = find_display_column(clean_df, ("_derived_model", "car_name", "model"))
    if "_derived_brand" not in clean_df.columns and "name" in clean_df.columns:
        name_text = clean_df["name"].fillna("").astype(str).str.strip()
        clean_df["_derived_brand"] = name_text.str.extract(r"^([A-Za-z0-9]+)", expand=False)
        clean_df["_derived_model"] = name_text.str.replace(r"^[A-Za-z0-9]+\s+", "", regex=True).str.strip()
        brand_feature = "_derived_brand"
        car_name_feature = "_derived_model"
    selected_brand = None
    selected_car_name = None

    st.caption("Select the brand and the exact car model.")
    brand_car, model_car = st.columns(2, gap="medium")

    if brand_feature:
        brand_choices = sorted(str(v) for v in clean_df[brand_feature].dropna().unique())
        default_brand = st.session_state.get("selected_brand", brand_choices[0])
        if default_brand not in brand_choices:
            default_brand = brand_choices[0]
        selected_brand = brand_car.selectbox(
            "Brand",
            brand_choices,
            index=brand_choices.index(default_brand),
            key="selected_brand",
        )

    if car_name_feature and selected_brand:
        matching_models = sorted(
            str(v)
            for v in clean_df.loc[
                clean_df[brand_feature].astype(str) == str(selected_brand),
                car_name_feature,
            ].dropna().unique()
        )
        if not matching_models:
            matching_models = ["Unknown"]
        default_model = st.session_state.get("selected_car_name")
        if default_model not in matching_models:
            default_model = matching_models[0]
        selected_car_name = model_car.selectbox(
            "Car Model",
            matching_models,
            index=matching_models.index(default_model),
            key="selected_car_name",
        )

    st.caption("Complete the specifications below. Values outside the dataset range are rejected.")
    with st.form("prediction_form", clear_on_submit=False):
        values = {}

        def find_feature(keywords):
            for feature in features:
                normalized_feature = "".join(ch for ch in str(feature).lower() if ch.isalnum())
                if any("".join(ch for ch in str(keyword).lower() if ch.isalnum()) in normalized_feature for keyword in keywords):
                    return feature
            return None

        ordered_features = []
        field_pairs = (
            (find_feature(("year", "vehicleage", "make_year")), "Year"),
            (find_feature(("km_driven", "kilometer", "odometer", "km")), "KM Driven"),
            (find_feature(("fuel", "fuel_type")), "Fuel"),
            (find_feature(("transmission", "gear")), "Transmission"),
            (find_feature(("seller_type", "owner")), "Seller Type"),
            (find_feature(("mileage",)), "Mileage"),
            (find_feature(("engine", "engine_capacity", "displacement")), "Engine"),
            (find_feature(("max_power", "power", "bhp")), "Max Power"),
            (find_feature(("seats", "seat")), "Seats"),
        )

        for feature, _ in field_pairs:
            if feature and feature not in ordered_features:
                ordered_features.append(feature)
        for feature in features:
            if feature not in ordered_features:
                ordered_features.append(feature)

        def render_field(feature, label, column):
            label = str(label)
            if feature in categories:
                choices = categories[feature]
                if not choices:
                    choices = ["Unknown"]
                values[feature] = column.selectbox(
                    label,
                    choices,
                    key=f"prediction_{feature}",
                )
            else:
                bounds = ranges.get(feature)
                numeric = pd.to_numeric(clean_df[feature], errors="coerce") if feature in clean_df.columns else pd.Series(dtype=float)
                if bounds:
                    lo, hi, default = bounds["min"], bounds["max"], bounds["median"]
                elif numeric.notna().any():
                    lo, hi, default = float(numeric.min()), float(numeric.max()), float(numeric.median())
                else:
                    lo, hi, default = 0.0, 1_000_000.0, 0.0
                if float(lo) == float(hi):
                    hi = lo + 1
                step = max((hi - lo) / 100, 0.1)
                values[feature] = column.number_input(
                    label,
                    min_value=float(lo),
                    max_value=float(hi),
                    value=float(default),
                    step=float(step),
                    key=f"prediction_{feature}",
                )

        if brand_feature and selected_brand:
            values[brand_feature] = selected_brand
        if car_name_feature and selected_car_name:
            values[car_name_feature] = selected_car_name

        for index in range(0, len(field_pairs), 2):
            left_feature, left_label = field_pairs[index]
            right_feature = field_pairs[index + 1][0] if index + 1 < len(field_pairs) else None
            right_label = field_pairs[index + 1][1] if index + 1 < len(field_pairs) else None
            left_col, right_col = st.columns(2, gap="medium")
            if left_feature:
                render_field(left_feature, left_label, left_col)
            if right_feature:
                render_field(right_feature, right_label, right_col)

        for feature in ordered_features:
            if feature not in {pair_feature for pair_feature, _ in field_pairs if pair_feature}:
                extra_col1, extra_col2 = st.columns(2, gap="medium")
                render_field(feature, str(feature).replace("_", " ").title(), extra_col1)

        button_col1, button_col2 = st.columns(2, gap="medium")
        submitted = button_col1.form_submit_button(
            "Predict Price",
            type="primary",
            key="predict_prediction_price",
        )
        reset = button_col2.form_submit_button(
            "Reset Form",
            key="reset_prediction_form",
            type="secondary",
        )

    if reset:
        st.session_state.pop("selected_brand", None)
        st.session_state.pop("selected_car_name", None)
        st.rerun()

    if submitted:
        try:
            for feature, value in values.items():
                bounds = ranges.get(feature)
                if bounds:
                    if value is not None and (float(value) < bounds["min"] or float(value) > bounds["max"]):
                        raise ValueError(
                            f"{feature.replace('_', ' ').title()} must be between "
                            f"{bounds['min']:g} and {bounds['max']:g}."
                        )
            with st.spinner("CatBoost is analyzing your vehicle..."):
                time.sleep(0.35)
                prediction, returned_metadata = predict_price(values)
            st.markdown(
                '<div class="prediction-complete">'
                "Prediction completed from the trained machine learning model."
                "</div>",
                unsafe_allow_html=True,
            )
            st.caption(
                "The estimate includes dataset-derived depreciation: older age and higher kilometers reduce the price."
            )
            st.markdown(
                f'<div class="luxury-price-card"><div class="label">Estimated Resale Price</div>'
                f'<div class="value">{format_rupees(prediction)}</div></div>',
                unsafe_allow_html=True,
            )
            st.markdown("### Selected Car")
            thumbnail = (
                find_car_thumbnail(str(selected_brand), str(selected_car_name))
                if selected_brand and selected_car_name
                else None
            )
            if isinstance(thumbnail, dict) and thumbnail.get("url"):
                st.image(
                    thumbnail["url"],
                    caption=f"{selected_brand} {selected_car_name}",
                    use_container_width=True,
                )
            else:
                error = thumbnail.get("error") if isinstance(thumbnail, dict) else "Image lookup failed"
                st.warning(error)
            st.markdown("### Vehicle Summary")
            summary = pd.DataFrame(
                {"Specification": [str(k).replace("_", " ").title() for k in values],
                 "Value": [str(v) for v in values.values()]}
            )
            st.dataframe(summary, hide_index=True, use_container_width=True)
            st.caption(f"Selected model: {returned_metadata['selected_model']} | Status: Completed")

            report = summary.copy()
            report["Estimated Resale Price"] = format_rupees(prediction)
            csv_buffer = io.StringIO()
            report.to_csv(csv_buffer, index=False)
            st.download_button(
                "Download Valuation Report",
                data=csv_buffer.getvalue(),
                file_name="smartcar_valuation_report.csv",
                mime="text/csv",
                use_container_width=True,
                type="primary",
            )

            if st.button("Add Vehicle to Comparison", use_container_width=True):
                comparison = st.session_state.setdefault("vehicle_comparison", [])
                comparison_entry = {
                    "Vehicle": f"{selected_brand} {selected_car_name}",
                    "Predicted Price": prediction,
                }
                if not any(item["Vehicle"] == comparison_entry["Vehicle"] for item in comparison):
                    comparison.append(comparison_entry)
                st.success("Vehicle added to comparison.")

            comparison = st.session_state.get("vehicle_comparison", [])
            if comparison:
                st.markdown("### Vehicle Comparison")
                comparison_df = pd.DataFrame(comparison)
                comparison_df["Predicted Price"] = comparison_df["Predicted Price"].apply(format_rupees)
                st.dataframe(comparison_df, hide_index=True, use_container_width=True)
                if st.button("Clear Comparison"):
                    st.session_state["vehicle_comparison"] = []
                    st.rerun()
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")


elif page == "Data Analytics":
    st.markdown("## Data Analytics")
    target_numeric = pd.to_numeric(clean_df[target], errors="coerce")
    fuel_col = find_display_column(clean_df, ("fuel",))
    transmission_col = find_display_column(clean_df, ("transmission", "gear"))

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Car Records", f"{len(clean_df):,}", border=True)
    c2.metric("Average Price", format_rupees(target_numeric.mean()), border=True)
    c3.metric(
        "Most Common Fuel",
        str(clean_df[fuel_col].mode().iloc[0]) if fuel_col and not clean_df[fuel_col].mode().empty else "N/A",
        border=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        chart_card("Price Distribution", price_distribution(clean_df, target))
    with c2:
        chart_card("Brand-wise Average Price", brand_average_price(clean_df, target))

    c1, c2 = st.columns(2)
    with c1:
        chart_card("Fuel Type Distribution", categorical_distribution(clean_df, "fuel"))
    with c2:
        chart_card("Transmission Distribution", categorical_distribution(clean_df, "transmission"))

    c1, c2 = st.columns(2)
    with c1:
        chart_card("Year-wise Price Trend", year_price_trend(clean_df, target))
    with c2:
        chart_card("Kilometers vs. Price", mileage_price(clean_df, target))

    chart_card("Numerical Correlation Heatmap", correlation_heatmap(clean_df))


elif page == "Model Performance":
    st.markdown("## Model Performance")
    if artifact is None:
        st.error(artifact_error or "No trained model is available.")
        st.code("python -m src.train", language="powershell")
        st.stop()

    metadata = artifact["metadata"]
    model_metrics = metadata["model_metrics"]
    validation_mae = metadata["validation_mae"]
    selected_name = metadata["selected_model"]

    st.markdown(
        '<div class="model-selection-card" style="background:#F0F7F6;border:1px solid #CCFBF1;border-radius:10px;'
        'padding:12px 16px;color:#292524;font-size:14px;">'
        f"Selected model: <strong>{selected_name}</strong> — chosen using the lowest mean "
        "validation MAE from 3-fold cross-validation."
        "</div>",
        unsafe_allow_html=True,
    )

    rows = []
    for name, metrics in model_metrics.items():
        rows.append({
            "Model": name,
            "Validation MAE": validation_mae.get(name),
            "Test MAE": metrics["MAE"],
            "Test RMSE": metrics["RMSE"],
            "Test R²": metrics["R2"],
        })
    comparison = pd.DataFrame(rows)
    st.dataframe(comparison, hide_index=True, use_container_width=True)

    selected_metrics = metadata["test_metrics"]
    c1, c2, c3 = st.columns(3)
    c1.metric("MAE", format_rupees(selected_metrics["MAE"]), border=True)
    c2.metric("RMSE", format_rupees(selected_metrics["RMSE"]), border=True)
    c3.metric("R²", f"{selected_metrics['R2']:.3f}", border=True)

    shap_summary = metadata.get("shap_summary", {})
    if shap_summary:
        st.markdown("### SHAP Feature Impact")
        shap_df = pd.DataFrame(
            {"Feature": list(shap_summary.keys()), "Mean |SHAP value|": list(shap_summary.values())}
        )
        shap_df["Feature"] = shap_df["Feature"].str.replace(r"^(numeric|categorical)__", "", regex=True)
        fig = px.bar(
            shap_df.sort_values("Mean |SHAP value|"),
            x="Mean |SHAP value|",
            y="Feature",
            orientation="h",
        )
        if st.session_state["ui_theme"] == "dark":
            fig.update_layout(template="plotly_dark", paper_bgcolor="#292524", plot_bgcolor="#292524",
                              font_color="#F5F5F4")
            fig.update_xaxes(showgrid=True, gridcolor="#57534E", zeroline=False)
            fig.update_yaxes(showgrid=True, gridcolor="#57534E", zeroline=False)
        else:
            fig.update_layout(template="plotly_white", paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
                              font_color="#292524")
            fig.update_xaxes(showgrid=True, gridcolor="#E7E5E4", zeroline=False)
            fig.update_yaxes(showgrid=True, gridcolor="#E7E5E4", zeroline=False)
        st.plotly_chart(fig, use_container_width=True)

    actual = np.asarray(metadata["test_actual"])
    predicted = np.asarray(metadata["test_predictions"])
    residuals = actual - predicted

    c1, c2 = st.columns(2)
    with c1:
        fig = px.scatter(x=actual, y=predicted, labels={"x": "Actual Price", "y": "Predicted Price"})
        line = np.linspace(actual.min(), actual.max(), 100)
        fig.add_scatter(x=line, y=line, mode="lines", name="Ideal prediction",
                        line=dict(color="#22d3ee", dash="dash"))
        if st.session_state["ui_theme"] == "dark":
            fig.update_layout(template="plotly_dark", paper_bgcolor="#292524", plot_bgcolor="#292524",
                              font_color="#F5F5F4")
            fig.update_xaxes(showgrid=True, gridcolor="#57534E", zeroline=False)
            fig.update_yaxes(showgrid=True, gridcolor="#57534E", zeroline=False)
        else:
            fig.update_layout(template="plotly_white", paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
                              font_color="#292524")
            fig.update_xaxes(showgrid=True, gridcolor="#E7E5E4", zeroline=False)
            fig.update_yaxes(showgrid=True, gridcolor="#E7E5E4", zeroline=False)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.histogram(residuals, nbins=40, labels={"value": "Residual"})
        if st.session_state["ui_theme"] == "dark":
            fig.update_layout(template="plotly_dark", paper_bgcolor="#292524", plot_bgcolor="#292524",
                              font_color="#F5F5F4")
            fig.update_xaxes(showgrid=True, gridcolor="#57534E", zeroline=False)
            fig.update_yaxes(showgrid=True, gridcolor="#57534E", zeroline=False)
        else:
            fig.update_layout(template="plotly_white", paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
                              font_color="#292524")
            fig.update_xaxes(showgrid=True, gridcolor="#E7E5E4", zeroline=False)
            fig.update_yaxes(showgrid=True, gridcolor="#E7E5E4", zeroline=False)
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Metric explanations"):
        st.write("- **MAE**: the average absolute difference between actual and predicted prices.")
        st.write("- **RMSE**: similar to MAE, but gives larger errors more weight.")
        st.write("- **R²**: how much price variation is explained by the model. Values closer to 1 are better.")
