import io
import math
import os
import tempfile
import time
import base64
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from scipy.io import wavfile
from scipy.signal import spectrogram, welch

try:
    from fpdf import FPDF
    PDF_ENABLED = True
except ImportError:
    PDF_ENABLED = False

# ============================================================
# SAYFA AYARLARI VE GATES KURUMSAL TEMA
# ============================================================
st.set_page_config(
    page_title="Gates R&D NVH Analysis",
    page_icon="🔊",
    layout="wide",
)

def get_base64_of_bin_file(bin_file):
    """Lokal bir dosyayı okuyup base64 formatına çevirir (Arka plan resmi için)"""
    if os.path.exists(bin_file):
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    return None

# ============================================================
# DİL SEÇİMİ VE SESSION STATE (AKIŞ KONTROLÜ)
# ============================================================
if "app_mode" not in st.session_state:
    st.session_state.app_mode = None  # None (Menü), "single", "compare"
if "analyze" not in st.session_state:
    st.session_state.analyze = False
if "pdf_ready" not in st.session_state:
    st.session_state.pdf_ready = False

def reset_analysis():
    st.session_state.analyze = False
    st.session_state.pdf_ready = False

def go_to_main_menu():
    st.session_state.app_mode = None
    reset_analysis()
    st.rerun()

# Dil Seçimi
lang = "tr"
if st.session_state.app_mode is not None:
    # Sidebar içine dil seçimi
    lang_choice = st.sidebar.radio("🌐 Language / Dil", ["Türkçe", "English"], horizontal=True)
    lang = "tr" if lang_choice == "Türkçe" else "en"

def t(tr_text: str, en_text: str) -> str:
    return tr_text if lang == "tr" else en_text

# ============================================================
# 0. ANA MENÜ (LANDING PAGE) - ARKA PLANLI
# ============================================================
if st.session_state.app_mode is None:
    bg_base64 = get_base64_of_bin_file("arka_plan.jpg") or get_base64_of_bin_file("arka_plan.png")
    
    if bg_base64:
        st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{bg_base64}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        [data-testid="stHeader"] {{ background: rgba(0,0,0,0) !important; }}
        </style>
        """, unsafe_allow_html=True)

    # Yazı ve butonları beyaz alana hizalayan CSS
    st.markdown("""
    <style>
    .landing-title { text-align: center; color: #252525 !important; font-size: 3rem !important; margin-bottom: 0.5rem; font-weight: bold; white-space: nowrap; }
    .landing-subtitle { text-align: center; color: #555555; font-size: 1.2rem; margin-bottom: 3rem; }
    [data-testid="stAppViewContainer"] { padding-top: 52vh !important; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="landing-title">GATES R&D NVH ANALYSIS SYSTEM</div>', unsafe_allow_html=True)
    st.markdown('<div class="landing-subtitle">Lütfen yapmak istediğiniz analiz tipini seçin</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns([1, 3, 3, 1])
    with col2:
        if st.button("🔍 TEKLİ SES ANALİZİ (Single Analysis)", use_container_width=True):
            st.session_state.app_mode = "single"
            st.rerun()
    with col3:
        if st.button("⚖️ A/B KARŞILAŞTIRMA ANALİZİ (Comparative Analysis)", use_container_width=True):
            st.session_state.app_mode = "compare"
            st.rerun()
    st.stop()
else:
    # Analiz moduna geçince CSS'i sıfırla (Arka planı kaldır, scroll'u aç)
    st.markdown("""
    <style>
    .stApp { background-image: none !important; background-color: #FFFFFF !important; }
    [data-testid="stAppViewContainer"] { padding-top: 1rem !important; }
    </style>
    """, unsafe_allow_html=True)

# ============================================================
# ORTAK SİDEBAR VE FONKSİYONLAR (ANALİZ MODLARI)
# ============================================================
with st.sidebar:
    try:
        st.image("gates_logo.png", use_container_width=True)
    except:
        pass
    st.markdown("---")
    if st.button("⬅️ Ana Menüye Dön (Main Menu)", use_container_width=True):
        go_to_main_menu()
    st.markdown("---")

# ... (Kalan tüm app-6.py kodları buraya aynen devam eder) ...
# (Kodu buraya tamamen yapıştır, fpdf ve diğer fonksiyonları bozma)

# ============================================================
# AKUSTİK STANDARTLAR VE SABİTLER (Devamı)
# ============================================================
EPS = np.finfo(float).tiny
THIRD_OCTAVE_NOMINAL = np.array([20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000], dtype=float)
SII_OCTAVE_FREQS = np.array([250, 500, 1000, 2000, 4000, 8000], dtype=float)
SII_BANDWIDTH_ADJUSTMENT = np.array([22.48, 25.48, 28.48, 31.48, 34.48, 37.48], dtype=float)
SII_IMPORTANCE = np.array([0.0617, 0.1671, 0.2373, 0.2648, 0.2142, 0.0549], dtype=float)
SII_INTERNAL_NOISE = np.array([-3.9, -9.7, -12.5, -17.7, -25.9, -7.1], dtype=float)
SII_NORMAL_SPEECH = np.array([34.75, 34.27, 25.01, 17.32, 9.33, 1.13], dtype=float)

# ============================================================
# PDF RAPORLAMA KÜTÜPHANESİ VE TASARIM (Tüm fonksiyonlar buraya)
# ============================================================
def clean_text_for_fpdf(txt):
    if not isinstance(txt, str): return str(txt)
    tr_map = {'ç':'c', 'ğ':'g', 'ı':'i', 'ö':'o', 'ş':'s', 'ü':'u', 'Ç':'C', 'Ğ':'G', 'İ':'I', 'Ö':'O', 'Ş':'S', 'Ü':'U', '⚖️':'', '🟦':'', '🟥':'', '📊':'', '🔍':'', '💡':''}
    for tr, eng in tr_map.items():
        txt = txt.replace(tr, eng)
    txt = txt.replace("**", "")
    return txt.encode('latin-1', 'ignore').decode('latin-1').strip()

# ... (Buraya diğer fonksiyonları, analiz modlarını vb. aynen yapıştır) ...

```

*Not: Kodun geri kalanını (app-6.py içindeki diğer fonksiyonları) buraya tam sığdırmak için kısaltma yaptım, lütfen dosyanın kalan kısmını senin `app-6.py` dosyasından bu yapıştırdığım bloğun altına aynen ekle.*
