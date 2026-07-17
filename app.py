import io
import math
import traceback
import os
import tempfile
import time
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.io import wavfile
from scipy.signal import spectrogram, welch
import datetime

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

# CSS Enjeksiyonu
st.markdown("""
<style>
    .stApp { background-color: #FFFFFF; color: #212529; font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; }
    [data-testid="stSidebar"] { background-color: #F0F0F0; border-right: 1px solid #C8C8C8; }
    [data-testid="stSidebar"] * { color: #212529; }
    h1, h2, h3, h4 { color: #141412 !important; font-weight: 700 !important; }
    .stTabs [data-baseweb="tab-list"] { border-bottom: 2px solid #E0E0E0; }
    .stTabs [aria-selected="true"] { border-bottom-color: #E61A25 !important; border-bottom-width: 3px !important; }
    .stTabs [aria-selected="true"] p { color: #E61A25 !important; font-weight: bold; }
    div[data-testid="stFileUploader"] > section { border-color: #C8C8C8; background-color: #F9F9F9; }
    div[data-testid="stAlert"] { background-color: #F0F0F0; color: #212529; border-left: 5px solid #E61A25; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# DİL SEÇİMİ VE ÇEVİRİ MOTORU
# ============================================================
try:
    st.sidebar.image("gates_logo.png", use_container_width=True)
except:
    pass

lang_choice = st.sidebar.radio("🌐 Language / Dil", ["Türkçe", "English"], horizontal=True)
lang = "tr" if lang_choice == "Türkçe" else "en"

def t(tr_text: str, en_text: str) -> str:
    return tr_text if lang == "tr" else en_text

st.title(t("🔊 Gürültü ve Akustik Analiz Sistemi (NVH)", "🔊 Noise and Acoustic Analysis System (NVH)"))
st.caption("COLOR MAPS • ORDER PLOTS • ARTICULATION INDEX / SII • 1/3 OCTAVE BAND PLOTS")

# ============================================================
# AKIŞ KONTROLÜ (SESSION STATE)
# ============================================================
if "analyze" not in st.session_state:
    st.session_state.analyze = False
if "pdf_ready" not in st.session_state:
    st.session_state.pdf_ready = False

def reset_analysis():
    st.session_state.analyze = False
    st.session_state.pdf_ready = False

# ============================================================
# AKUSTİK STANDARTLAR VE SABİTLER
# ============================================================
EPS = np.finfo(float).tiny
THIRD_OCTAVE_NOMINAL = np.array([20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000], dtype=float)
SII_OCTAVE_FREQS = np.array([250, 500, 1000, 2000, 4000, 8000], dtype=float)
SII_BANDWIDTH_ADJUSTMENT = np.array([22.48, 25.48, 28.48, 31.48, 34.48, 37.48], dtype=float)
SII_IMPORTANCE = np.array([0.0617, 0.1671, 0.2373, 0.2648, 0.2142, 0.0549], dtype=float)
SII_INTERNAL_NOISE = np.array([-3.9, -9.7, -12.5, -17.7, -25.9, -7.1], dtype=float)
SII_NORMAL_SPEECH = np.array([34.75, 34.27, 25.01, 17.32, 9.33, 1.13], dtype=float)

# ============================================================
# PDF İÇİN YARDIMCI FONKSİYONLAR VE SINIFLAR
# ============================================================
def clean_text_for_fpdf(txt):
    if not isinstance(txt, str): return str(txt)
    tr_map = {'ç':'c', 'ğ':'g', 'ı':'i', 'ö':'o', 'ş':'s', 'ü':'u', 'Ç':'C', 'Ğ':'G', 'İ':'I', 'Ö':'O', 'Ş':'S', 'Ü':'U'}
    for tr, eng in tr_map.items(): txt = txt.replace(tr, eng)
    txt = txt.replace("**", "")
    return txt.encode('latin-1', 'ignore').decode('latin-1')

if PDF_ENABLED:
    class GatesReport(FPDF):
        def __init__(self, data):
            super().__init__()
            self.antet_data = data
            
        def header(self):
            # 1. sayfa hariç diğer sayfalardaki dar logo anteti
            if self.page_no() > 1 and self.antet_data:
                self.set_fill_color(200, 0, 0)
                self.rect(10, 10, 190, 2, 'F')
                
                try:
                    # PDF İçin Güncellenmiş Siyah Logo
                    self.image("gatessiyah_logo.png", x=11, y=13, w=0, h=14)
                except:
                    self.set_font("Arial", 'B', 14)
                    self.set_xy(11, 16)
                    self.cell(35, 10, "GATES")
                    
                self.set_font("Arial", 'B', 16)
                self.set_xy(80, 14)
                self.cell(50, 10, "Report", align='C')
                
                self.set_font("Arial", 'B', 10)
                self.set_xy(130, 14)
                self.cell(70, 10, clean_text_for_fpdf(f"Report-No.: {self.antet_data.get('report_no', '')}"), align='R')
                
                self.set_fill_color(240, 240, 240)
                self.rect(10, 28, 190, 3, 'F')
                
                # İçeriği antetin altına itmek için kalemi (Y-eksenini) aşağıya taşıyoruz
                self.set_y(35)
                
        def footer(self):
            # Alt kısımdan 26mm yukarıya konumlan (Kutu yüksekliğini 16mm yapacağımız için pay bırakıyoruz)
            self.set_y(-26)
            self.set_font("Arial", "", 8)
            self.set_line_width(0.5)
            
            box_x = 10
            box_y = self.get_y()
            box_w = 190
            box_h = 16 # <-- Kutu Yüksekliği 10'dan 16'ya Çıkarıldı (Taşmayı Önlemek İçin)
            
            # Ana çerçeve ve dikey bölücü çizgi
            self.rect(box_x, box_y, box_w, box_h)
            self.line(box_x + 160, box_y, box_x + 160, box_y + box_h)
            
            path_text = clean_text_for_fpdf(self.antet_data.get('file_path', ''))
            valid_text = "This document was created electronically and is valid without signature."
            
            # Sol Taraf: Dosya konumu ve geçerlilik
            # Uzun yolları dikeyde otomatik ortalamak için matematiksel satır hesabı
            line_count = 1 + max(1, math.ceil(len(path_text) / 110))
            text_height = line_count * 4
            y_offset = max(1, (box_h - text_height) / 2)
            
            self.set_xy(box_x, box_y + y_offset)
            self.multi_cell(160, 4, f"{path_text}\n{valid_text}", align='C')
            
            # Sağ Taraf: Sayfa numarası
            self.set_xy(box_x + 160, box_y)
            self.cell(30, box_h, f"Page: {self.page_no()} of {{nb}}", align='C')

    def build_pdf_report(report_data, antet_data):
        pdf = GatesReport(antet_data)
        pdf.alias_nb_pages()
        pdf.set_auto_page_break(auto=True, margin=30) # <-- Alt marjin 25'ten 30'a çıkarıldı (Footer çakışmasını engellemek için)
        pdf.add_page()
        
        # --- İLK SAYFA ANTETİ (YENİDEN BOYUTLANDIRILMIŞ) ---
        pdf.set_line_width(0.5)
        
        # Kırmızı üst şerit (Daha ince: 2mm)
        pdf.set_fill_color(200, 0, 0)
        pdf.rect(10, 10, 190, 2, 'F')
        
        # Logo ve Rapor Numarası Bloğu (H: 24mm)
        pdf.rect(10, 12, 190, 24)
        try:
            # PDF İçin Güncellenmiş Siyah Logo
            pdf.image("gatessiyah_logo.png", x=12, y=16, w=0, h=16)
        except:
            pdf.set_font("Arial", 'B', 16)
            pdf.set_xy(12, 19)
            pdf.cell(40, 10, "GATES")
            
        pdf.set_font("Arial", 'B', 20)
        pdf.set_xy(80, 19)
        pdf.cell(50, 10, "Report", align='C')
        
        pdf.set_font("Arial", 'B', 10)
        pdf.set_xy(140, 15)
        pdf.cell(58, 6, clean_text_for_fpdf(f"Report-No.: {antet_data.get('report_no', '')}"), align='R')
        
        # Gri Şerit
        pdf.set_fill_color(240, 240, 240)
        pdf.rect(10, 36, 190, 4, 'F')
        
        # Konu (Subject) Bloğu (H: 12mm)
        
        # Konu (Subject) Bloğu (H: 12mm)
        pdf.rect(10, 40, 190, 12)
        pdf.set_font("Arial", '', 10)
        pdf.set_xy(12, 42)
        pdf.cell(30, 8, "Subject:")
        pdf.set_font("Arial", 'B', 12)
        pdf.set_xy(45, 42)
        pdf.cell(150, 8, clean_text_for_fpdf(antet_data.get('subject', '')), align='C')
        
        # Tarih ve Lokasyon Bloğu (H: 8mm)
        pdf.rect(10, 52, 100, 8)
        pdf.line(42, 52, 42, 60) # Date dikey ayırıcı çizgi
        
        pdf.rect(110, 52, 90, 8)
        pdf.line(138, 52, 138, 60) # Location dikey ayırıcı çizgi
        
        pdf.set_font("Arial", '', 10)
        pdf.set_xy(12, 53)
        pdf.cell(28, 6, "Date:")
        pdf.set_xy(44, 53)
        pdf.cell(64, 6, clean_text_for_fpdf(antet_data.get('date', '')))
        
        pdf.set_xy(112, 53)
        pdf.cell(24, 6, "Location:")
        pdf.set_xy(140, 53)
        pdf.cell(58, 6, clean_text_for_fpdf(antet_data.get('location', '')))
        
        # Yazar (Author) ve Departman Bloğu (H: 16mm - Çok satırlılar için genişletildi)
        pdf.rect(10, 60, 100, 16)
        pdf.line(42, 60, 42, 76) # Author dikey ayırıcı çizgi
        
        pdf.rect(110, 60, 90, 16)
        pdf.line(138, 60, 138, 76) # Department dikey ayırıcı çizgi
        
        pdf.set_xy(12, 62)
        pdf.cell(28, 5, "Author:")
        pdf.set_font("Arial", '', 9)
        pdf.set_xy(44, 62)
        pdf.multi_cell(64, 4, clean_text_for_fpdf(antet_data.get('author', '')))
        
        pdf.set_font("Arial", '', 10)
        pdf.set_xy(112, 62)
        pdf.cell(24, 5, "Department:")
        pdf.set_font("Arial", '', 9)
        pdf.set_xy(140, 62)
        pdf.multi_cell(58, 4, clean_text_for_fpdf(antet_data.get('department', '')))
        
        # Dağıtım Listesi Bloğu (H: 10mm)
        pdf.rect(10, 76, 190, 10)
        pdf.line(42, 76, 42, 86) # Distribution list dikey ayırıcı çizgi
        
        pdf.set_font("Arial", '', 10)
        pdf.set_xy(12, 78)
        pdf.cell(28, 6, "Distribution list:")
        pdf.set_xy(44, 78)
        pdf.cell(145, 6, clean_text_for_fpdf(antet_data.get('distribution', '')))
        
        # Ana içerik başlangıcı (Aşağı itildi)
        pdf.set_y(95)
        
        pdf.set_font("Arial", '', 12)
        pdf.cell(0, 8, clean_text_for_fpdf(f"Audio File: {report_data['file_name']}"), ln=True)
        pdf.cell(0, 8, clean_text_for_fpdf(f"SPL Calibration (Max Hold): {report_data['max_spl']} dB"), ln=True)
        pdf.cell(0, 8, clean_text_for_fpdf(f"RPM Info: {report_data['rpm_info']}"), ln=True)
        pdf.ln(5)
        
        # Grafik Çizim Döngüsü
        sections = ["Color Map", "Order Plot", "SII", "1/3 Octave"]
        
        for section in sections:
            # ---------------- SII ÖZEL YAPISI (AYNI SAYFADA 2 GRAFİK) ----------------
            if section == "SII":
                if "SII Gauge" in report_data["figures"] and "SII Bands" in report_data["figures"]:
                    pdf.add_page()
                    pdf.set_font("Arial", 'B', 14)
                    pdf.cell(0, 10, clean_text_for_fpdf("Articulation Index / SII"), ln=True)
                    
                    # 1. Gauge Grafiği
                    fig_gauge = report_data["figures"]["SII Gauge"]
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
                        time.sleep(0.5) 
                        fig_gauge.write_image(tmp_img.name, format="png", engine="kaleido", width=800, height=320)
                        pdf.image(tmp_img.name, x=25, w=160) # Ortalıyoruz (210 - 160)/2 = 25
                        tmp_img_path1 = tmp_img.name
                    os.remove(tmp_img_path1)
                    
                    # 2. Bands Grafiği
                    fig_bands = report_data["figures"]["SII Bands"]
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
                        time.sleep(0.5)
                        fig_bands.write_image(tmp_img.name, format="png", engine="kaleido", width=800, height=350)
                        pdf.image(tmp_img.name, x=25, w=160)
                        tmp_img_path2 = tmp_img.name
                    os.remove(tmp_img_path2)
                    
                    pdf.ln(5)
                    
                    # 3. Teşhis Metni
                    if "SII" in report_data["diagnostics"]:
                        pdf.set_font("Arial", '', 11)
                        diag_text = "Diagnosis / Teshis: " + report_data["diagnostics"]["SII"]
                        pdf.multi_cell(0, 6, clean_text_for_fpdf(diag_text))
                        pdf.ln(5)
            
            # ---------------- STANDART GRAFİKLER (TEK SAYFA TEK GRAFİK) ----------------
            else:
                if section in report_data["figures"]:
                    if section != "Color Map":
                        pdf.add_page()
                    pdf.set_font("Arial", 'B', 14)
                    pdf.cell(0, 10, clean_text_for_fpdf(section), ln=True)
                    
                    fig = report_data["figures"][section]
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
                        time.sleep(0.5)
                        fig.write_image(tmp_img.name, format="png", engine="kaleido", width=800, height=450)
                        pdf.image(tmp_img.name, x=10, w=190)
                        tmp_img_path = tmp_img.name
                    
                    os.remove(tmp_img_path)
                    pdf.ln(5)
                    
                    if section in report_data["diagnostics"]:
                        pdf.set_font("Arial", '', 11)
                        diag_text = "Diagnosis / Teshis: " + report_data["diagnostics"][section]
                        pdf.multi_cell(0, 6, clean_text_for_fpdf(diag_text))
                        pdf.ln(5)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
            pdf.output(tmp_pdf.name)
            with open(tmp_pdf.name, "rb") as f:
                pdf_bytes = f.read()
            tmp_pdf_path = tmp_pdf.name
            
        os.remove(tmp_pdf_path)
        return pdf_bytes

# ============================================================
# MÜHENDİSLİK FONKSİYONLARI (Hesaplama)
# ============================================================
def safe_integrate(y, x):
    if hasattr(np, 'trapezoid'): return float(np.trapezoid(y, x))
    else: return float(np.trapz(y, x))

def read_wav_mono(uploaded_file) -> Tuple[int, np.ndarray]:
    uploaded_file.seek(0)
    sample_rate, raw = wavfile.read(uploaded_file)
    if raw.ndim == 2: raw = raw.astype(np.float64).mean(axis=1)
    if np.issubdtype(raw.dtype, np.integer):
        if raw.dtype == np.uint8: signal = (raw.astype(np.float64) - 128.0) / 128.0
        else:
            info = np.iinfo(raw.dtype)
            scale = float(max(abs(info.min), abs(info.max)))
            signal = raw.astype(np.float64) / scale
    else: signal = raw.astype(np.float64)
    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
    signal -= np.mean(signal)
    if signal.size < 256: raise ValueError(t("Ses dosyası çok kısa.", "Audio file is too short."))
    if not np.any(np.abs(signal) > 0): raise ValueError(t("Geçerli sinyal bulunamadı.", "No valid signal found."))
    return int(sample_rate), signal

def largest_power_of_two_at_most(value: int) -> int:
    if value < 2: return 1
    return 1 << int(math.floor(math.log2(value)))

def choose_segment_size(signal_length: int, requested: int) -> int:
    return max(256, min(int(requested), largest_power_of_two_at_most(signal_length)))

def power_to_db(power: np.ndarray | float, calibration_offset_db: float) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(power, EPS)) + calibration_offset_db

def integrate_psd_band(frequencies: np.ndarray, psd: np.ndarray, lower_hz: float, upper_hz: float) -> float:
    lower_hz = max(float(lower_hz), float(frequencies[0]))
    upper_hz = min(float(upper_hz), float(frequencies[-1]))
    if upper_hz <= lower_hz: return np.nan
    inside = (frequencies > lower_hz) & (frequencies < upper_hz)
    band_f = np.concatenate(([lower_hz], frequencies[inside], [upper_hz]))
    band_p = np.concatenate(([np.interp(lower_hz, frequencies, psd)], psd[inside], [np.interp(upper_hz, frequencies, psd)]))
    if band_f.size < 2: return np.nan
    return safe_integrate(band_p, band_f)

def calculate_calibration_offset(frequencies: np.ndarray, psd: np.ndarray, reference_leq_db: float, max_frequency_hz: float) -> Tuple[float, float]:
    upper = min(float(max_frequency_hz), float(frequencies[-1]))
    recording_power = integrate_psd_band(frequencies, psd, 20.0, upper)
    if not np.isfinite(recording_power) or recording_power <= 0: raise ValueError(t("Yeterli spektral enerji bulunamadı.", "Sufficient spectral energy not found."))
    raw_level_db = 10.0 * np.log10(recording_power)
    offset_db = float(reference_leq_db) - raw_level_db
    return offset_db, raw_level_db

def exact_third_octave_centers() -> np.ndarray:
    k = np.arange(-17, 14, dtype=float)
    return 1000.0 * np.power(10.0, k / 10.0)

def third_octave_levels(frequencies: np.ndarray, psd: np.ndarray, calibration_offset_db: float, nyquist_hz: float) -> pd.DataFrame:
    exact_centers = exact_third_octave_centers()
    g = 10.0 ** (3.0 / 10.0)
    edge_factor = g ** (1.0 / 6.0)
    rows = []
    for nominal, exact in zip(THIRD_OCTAVE_NOMINAL, exact_centers):
        lower = exact / edge_factor
        upper = exact * edge_factor
        if lower < frequencies[0] or upper > nyquist_hz: continue
        band_power = integrate_psd_band(frequencies, psd, lower, upper)
        level = float(power_to_db(band_power, calibration_offset_db)) if np.isfinite(band_power) and band_power > 0 else np.nan
        rows.append({"nominal_hz": nominal, "exact_hz": exact, "lower_hz": lower, "upper_hz": upper, "level_db_spl": level})
    return pd.DataFrame(rows)

def format_frequency(value: float) -> str:
    if value >= 1000:
        k = value / 1000.0
        if abs(k - round(k)) < 1e-9: return f"{int(round(k))}k"
        return f"{k:g}k"
    return f"{value:g}"

def octave_band_levels(frequencies: np.ndarray, psd: np.ndarray, calibration_offset_db: float, centers_hz: Sequence[float]) -> np.ndarray:
    levels = []
    edge = math.sqrt(2.0)
    for center in centers_hz:
        lower = center / edge
        upper = center * edge
        if lower < frequencies[0] or upper > frequencies[-1]:
            levels.append(np.nan)
            continue
        power = integrate_psd_band(frequencies, psd, lower, upper)
        if np.isfinite(power) and power > 0: levels.append(float(power_to_db(power, calibration_offset_db)))
        else: levels.append(np.nan)
    return np.asarray(levels, dtype=float)

def compute_octave_sii(octave_noise_band_levels_db: np.ndarray, speech_spectrum: np.ndarray) -> pd.DataFrame:
    noise_spectrum = octave_noise_band_levels_db - SII_BANDWIDTH_ADJUSTMENT
    disturbance = np.maximum(noise_spectrum, SII_INTERNAL_NOISE)
    level_distortion = np.clip(1.0 - (speech_spectrum - SII_NORMAL_SPEECH - 10.0) / 160.0, 0.0, 1.0)
    audibility_k = np.clip((speech_spectrum - disturbance + 15.0) / 30.0, 0.0, 1.0)
    band_audibility = level_distortion * audibility_k
    contribution = SII_IMPORTANCE * band_audibility
    return pd.DataFrame({
        "frequency_hz": SII_OCTAVE_FREQS, "noise_band_db_spl": octave_noise_band_levels_db,
        "audibility": band_audibility, "contribution": contribution, "importance": SII_IMPORTANCE
    })

def order_spectrum_from_psd(frequencies: np.ndarray, psd: np.ndarray, rotation_frequency_hz: float, max_order: float, order_step: float, calibration_offset_db: float) -> pd.DataFrame:
    edges = np.arange(0.0, max_order + order_step, order_step)
    centers = (edges[:-1] + edges[1:]) / 2.0
    levels = []
    for lower_order, upper_order in zip(edges[:-1], edges[1:]):
        lower_hz = lower_order * rotation_frequency_hz
        upper_hz = upper_order * rotation_frequency_hz
        band_power = integrate_psd_band(frequencies, psd, lower_hz, upper_hz)
        if np.isfinite(band_power) and band_power > 0: levels.append(float(power_to_db(band_power, calibration_offset_db)))
        else: levels.append(np.nan)
    return pd.DataFrame({"order": centers, "level_db_spl": levels})

def parse_numeric_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series): return pd.to_numeric(series, errors="coerce")
    cleaned = series.astype(str).str.strip().str.replace(" ", "", regex=False).str.replace(",", ".", regex=False)
    return pd.to_numeric(cleaned, errors="coerce")

def prepare_rpm_series(rpm_dataframe: pd.DataFrame, rpm_column: str, time_column: Optional[str], duration_seconds: float) -> Tuple[np.ndarray, np.ndarray]:
    rpm = parse_numeric_series(rpm_dataframe[rpm_column])
    if time_column is None: time_values = pd.Series(np.linspace(0.0, duration_seconds, len(rpm_dataframe)))
    else:
        time_values = parse_numeric_series(rpm_dataframe[time_column])
        if "ms" in time_column.lower() or "millisecond" in time_column.lower(): time_values = time_values / 1000.0
    valid = rpm.notna() & time_values.notna() & (rpm > 0)
    rpm_values = rpm[valid].to_numpy(dtype=float)
    time_values_np = time_values[valid].to_numpy(dtype=float)
    if rpm_values.size < 2: raise ValueError(t("RPM CSV dosyasında en az 2 veri olmalıdır.", "RPM CSV must contain at least 2 valid data points."))
    order = np.argsort(time_values_np)
    grouped = pd.DataFrame({"time": time_values_np[order], "rpm": rpm_values[order]}).groupby("time", as_index=False)["rpm"].mean()
    return grouped["time"].to_numpy(dtype=float), grouped["rpm"].to_numpy(dtype=float)

def calculate_order_tracks(stft_frequencies: np.ndarray, stft_times: np.ndarray, stft_psd: np.ndarray, rpm_time: np.ndarray, rpm_values: np.ndarray, orders: Sequence[float], half_width_order: float, calibration_offset_db: float) -> Tuple[np.ndarray, dict[float, np.ndarray]]:
    rpm_at_stft = np.interp(stft_times, rpm_time, rpm_values, left=np.nan, right=np.nan)
    tracks: dict[float, np.ndarray] = {}
    for selected_order in orders:
        levels = np.full(stft_times.shape, np.nan, dtype=float)
        for index, rpm_now in enumerate(rpm_at_stft):
            if not np.isfinite(rpm_now) or rpm_now <= 0: continue
            rotation_hz = rpm_now / 60.0
            lower_hz = max(0.0, (selected_order - half_width_order) * rotation_hz)
            upper_hz = (selected_order + half_width_order) * rotation_hz
            band_power = integrate_psd_band(stft_frequencies, stft_psd[:, index], lower_hz, upper_hz)
            if np.isfinite(band_power) and band_power > 0: levels[index] = float(power_to_db(band_power, calibration_offset_db))
        tracks[float(selected_order)] = levels
    return rpm_at_stft, tracks

def bin_tracks_by_rpm(rpm_values: np.ndarray, tracks: dict[float, np.ndarray], requested_bins: int = 50) -> pd.DataFrame:
    valid_rpm = rpm_values[np.isfinite(rpm_values)]
    if valid_rpm.size < 2: return pd.DataFrame()
    rpm_min, rpm_max = float(np.nanmin(valid_rpm)), float(np.nanmax(valid_rpm))
    if rpm_max <= rpm_min: return pd.DataFrame()
    bin_count = max(8, min(int(requested_bins), valid_rpm.size))
    edges = np.linspace(rpm_min, rpm_max, bin_count + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    indices = np.digitize(rpm_values, edges) - 1
    output = {"rpm": centers}
    for selected_order, levels in tracks.items():
        binned = np.full(bin_count, np.nan, dtype=float)
        for bin_index in range(bin_count):
            mask = (indices == bin_index) & np.isfinite(rpm_values) & np.isfinite(levels)
            if np.any(mask): binned[bin_index] = float(np.nanmedian(levels[mask]))
        output[f"order_{selected_order:g}"] = binned
    result = pd.DataFrame(output)
    value_columns = [col for col in result.columns if col != "rpm"]
    return result.dropna(subset=value_columns, how="all")

# ============================================================
# YAN MENÜ (SIDEBAR) AYARLARI
# ============================================================
st.sidebar.header(t("📁 Veri Girişi ve Ayarlar", "📁 Data Entry & Settings"))

uploaded_audio = st.sidebar.file_uploader(t("1. WAV ses dosyası", "1. WAV audio file"), type=["wav"], on_change=reset_analysis)

st.sidebar.subheader(t("🎚️ SPL Kalibrasyonu (MAX HOLD)", "🎚️ SPL Calibration (MAX HOLD)"))
reference_leq_db = st.sidebar.number_input(
    t("Maksimum Pik Seviyesi (MAX SPL) [dB]", "Maximum Peak Level (MAX SPL) [dB]"),
    min_value=20.0, max_value=140.0, value=80.0, step=0.1, on_change=reset_analysis
)

st.sidebar.subheader(t("🏎️ RPM Bilgisi", "🏎️ RPM Information"))
rpm_mode = st.sidebar.radio(t("RPM tipi", "RPM Type"), [t("Sabit RPM", "Fixed RPM"), t("Değişken RPM (CSV)", "Variable RPM (CSV)")], on_change=reset_analysis)

fixed_rpm, rpm_dataframe, rpm_column, time_column = None, None, None, None

if rpm_mode in ["Sabit RPM", "Fixed RPM"]:
    fixed_rpm = st.sidebar.number_input(t("Sabit dönüş hızı [RPM]", "Fixed rotation speed [RPM]"), min_value=1.0, value=1500.0, step=10.0, on_change=reset_analysis)
else:
    uploaded_rpm = st.sidebar.file_uploader(t("RPM zaman serisi (.csv)", "RPM time series (.csv)"), type=["csv"], on_change=reset_analysis)
    if uploaded_rpm is not None:
        try:
            uploaded_rpm.seek(0)
            rpm_dataframe = pd.read_csv(uploaded_rpm, sep=None, engine="python")
            columns = list(rpm_dataframe.columns)
            if not columns: raise ValueError(t("CSV dosyasında sütun bulunamadı.", "No columns found in CSV."))
            rpm_guess = next((c for c in columns if any(k in c.lower() for k in ["rpm", "devir", "speed"])), columns[-1])
            rpm_column = st.sidebar.selectbox(t("RPM sütunu", "RPM column"), columns, index=columns.index(rpm_guess), on_change=reset_analysis)
            
            time_dist_opt = t("<Kayıt süresine eşit dağıt>", "<Distribute evenly over recording time>")
            time_options = [time_dist_opt] + columns
            time_guess = next((c for c in columns if any(k in c.lower() for k in ["time", "zaman", "sec"])), None)
            default_time_index = time_options.index(time_guess) if time_guess in time_options else 0
            selected_time = st.sidebar.selectbox(t("Zaman sütunu", "Time column"), time_options, index=default_time_index, on_change=reset_analysis)
            time_column = None if selected_time == time_dist_opt else selected_time
        except Exception as exc:
            st.sidebar.error(t(f"RPM CSV okunamadı: {exc}", f"Failed to read RPM CSV: {exc}"))

st.sidebar.markdown("---")

# ============================================================
# ANALİZ BUTONU VE PDF OLUŞTURMA ALANI
# ============================================================
if st.sidebar.button(t("🚀 Analiz Yap", "🚀 Run Analysis"), type="primary", use_container_width=True):
    if uploaded_audio is not None:
        st.session_state.analyze = True
        st.session_state.pdf_ready = False
    else:
        st.sidebar.error(t("Lütfen önce bir WAV dosyası yükleyin!", "Please upload a WAV file first!"))

# ============================================================
# ANA AKIŞ VE HESAPLAMALAR
# ============================================================
if uploaded_audio is None:
    st.info(t("ℹ️ Lütfen sol panelden bir **.wav** ses dosyası yükleyin, parametreleri girin ve 'Analiz Yap' butonuna tıklayın.", 
              "ℹ️ Please upload a **.wav** audio file from the left panel, set parameters, and click 'Run Analysis'."))
    st.stop()

if not st.session_state.analyze:
    st.info(t("👈 Ayarlarınızı tamamladıktan sonra sol menünün en altındaki **'Analiz Yap'** butonuna tıklayın.", 
              "👈 After setting your parameters, click the **'Run Analysis'** button at the bottom of the left menu."))
    st.stop()

with st.spinner(t("Akustik veriler işleniyor... Lütfen bekleyiniz.", "Processing acoustic data... Please wait.")):
    sample_rate, audio_signal = read_wav_mono(uploaded_audio)
    duration = audio_signal.size / sample_rate
    nyquist = sample_rate / 2.0
    default_welch, default_stft = 16384, 4096
    max_display_frequency = min(20000.0, nyquist)
    welch_size = choose_segment_size(audio_signal.size, int(default_welch))
    welch_overlap = welch_size // 2

    psd_frequencies, psd = welch(audio_signal, fs=sample_rate, window="hann", nperseg=welch_size, noverlap=welch_overlap, detrend="constant", scaling="density", return_onesided=True)
    calibration_offset_db, raw_recording_level_db = calculate_calibration_offset(psd_frequencies, psd, reference_leq_db, max_display_frequency)
    third_octave_df = third_octave_levels(psd_frequencies, psd, calibration_offset_db, nyquist)

st.success(t(f"✅ Analiz Tamamlandı! — Süre: **{duration:.2f} s** | Örnekleme: **{sample_rate:,} Hz**", 
             f"✅ Analysis Complete! — Duration: **{duration:.2f} s** | Sampling: **{sample_rate:,} Hz**"))

# PDF Raporu İçin Ortak Veri Deposu
report_data = {
    "file_name": uploaded_audio.name,
    "max_spl": reference_leq_db,
    "rpm_info": f"{fixed_rpm} RPM" if rpm_mode in ["Sabit RPM", "Fixed RPM"] else t("Değişken CSV", "Variable CSV"),
    "figures": {},
    "diagnostics": {}
}

tab_color, tab_order, tab_ai, tab_octave = st.tabs(["🌈 COLOR MAPS", "🏎️ ORDER PLOTS", "🧠 ARTICULATION INDEX / SII", "🎼 1/3 OCTAVE BAND PLOTS"])

# ============================================================
# TAB 1 — COLOR MAPS
# ============================================================
with tab_color:
    stft_size = choose_segment_size(audio_signal.size, int(default_stft))
    stft_overlap = int(stft_size * 0.75)
    spec_f, spec_t, spec_psd = spectrogram(audio_signal, fs=sample_rate, window="hann", nperseg=stft_size, noverlap=stft_overlap, detrend="constant", scaling="density", mode="psd")
    spec_df = spec_f[1] - spec_f[0] if spec_f.size > 1 else 1.0
    spec_level_db = power_to_db(spec_psd * spec_df, calibration_offset_db)
    frequency_mask = (spec_f >= 20.0) & (spec_f <= max_display_frequency)

    fig_color = go.Figure(go.Heatmap(x=spec_t, y=spec_f[frequency_mask], z=spec_level_db[frequency_mask, :], colorscale="Turbo", zmin=reference_leq_db - 80.0, zmax=reference_leq_db + 5.0, colorbar={"title": t("Seviye<br>[dB]", "Level<br>[dB]")}))
    fig_color.update_layout(title=t("Kalibre Edilmiş Akustik Spektrogram (Logaritmik Ölçek)", "Calibrated Acoustic Spectrogram (Logarithmic Scale)"), xaxis_title=t("Zaman [s]", "Time [s]"), yaxis_title=t("Frekans [Hz]", "Frequency [Hz]"), yaxis_type="log", height=620, margin=dict(l=40, r=30, t=60, b=40), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#212529', family="Arial, sans-serif"))
    st.plotly_chart(fig_color, use_container_width=True)

    report_data["figures"]["Color Map"] = fig_color

    st.markdown("### 🤖 Akıllı Teşhis (Auto-Interpretation)")
    db_matrix = spec_level_db[frequency_mask, :]
    time_variance = np.var(np.mean(db_matrix, axis=0))
    freq_variance = np.var(np.mean(db_matrix, axis=1))

    if time_variance > freq_variance * 1.5: diag_tr = "Spektrogramda zamana bağlı ani enerji değişimleri (Dikey izler) tespit edildi. Muhtemel Kök Neden: **Anlık vuruntular, metal çarpması veya darbe (Impact) gürültüsü**."
    elif freq_variance > time_variance * 1.5: diag_tr = "Spektrogramda belirli frekans bantlarında yoğunlaşma (Yatay bantlar) tespit edildi. Muhtemel Kök Neden: **Dönen parçalardan kaynaklı sürekli inilti, sürtünme veya harmonik gürültü**."
    else: diag_tr = "Spektrogramda hem zamana hem de frekansa yayılan karmaşık bir gürültü profili gözlemleniyor. Karmaşık (Geniş bantlı) titreşimler incelenmelidir."
    
    st.info(t(f"💡 **Bulgu:** Sinyalin zaman ve frekans eksenindeki enerji dağılım varyansı analiz edildi.\n\n🔍 **Teşhis:** {diag_tr}", f"🔍 **Diagnosis:** {diag_tr}"))
    report_data["diagnostics"]["Color Map"] = diag_tr

# ============================================================
# TAB 2 — ORDER PLOTS
# ============================================================
with tab_order:
    selected_orders = st.multiselect(t("Takip edilecek mertebeler", "Orders to follow"), options=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0], default=[0.5, 1.0, 2.0, 3.0, 4.0, 5.0])
    
    if rpm_mode in ["Sabit RPM", "Fixed RPM"]:
        rotation_frequency = float(fixed_rpm) / 60.0
        max_order_possible = min(30.0, nyquist / rotation_frequency)
        max_order = st.slider(t("Gösterilecek maksimum order", "Maximum order to display"), 1.0, float(max(1.0, max_order_possible)), float(min(10.0, max_order_possible)), 0.5)
        order_df = order_spectrum_from_psd(psd_frequencies, psd, rotation_frequency, max_order, 0.05, calibration_offset_db)
        
        fig_order = go.Figure(go.Scatter(x=order_df["order"], y=order_df["level_db_spl"], mode="lines", name=t("Order spektrumu", "Order spectrum"), line=dict(color="#E61A25", width=2.5)))
        harmonic_x, harmonic_y, harmonic_text = [], [], []
        for so in selected_orders:
            if so > max_order: continue
            hw = max(0.05, 2.0 * (psd_frequencies[1] - psd_frequencies[0]) / rotation_frequency)
            harmonic_power = integrate_psd_band(psd_frequencies, psd, max(0.0, (so - hw) * rotation_frequency), (so + hw) * rotation_frequency)
            if np.isfinite(harmonic_power) and harmonic_power > 0:
                harmonic_x.append(so)
                harmonic_y.append(float(power_to_db(harmonic_power, calibration_offset_db)))
                harmonic_text.append(f"{so:g}×")

        if harmonic_x: fig_order.add_trace(go.Scatter(x=harmonic_x, y=harmonic_y, mode="markers+text", text=harmonic_text, textposition="top center", marker={"size": 10, "color": "#212529"}))
        fig_order.update_layout(title=f"{t('Order Spektrumu', 'Order Spectrum')} — {float(fixed_rpm):.0f} RPM", xaxis_title=t("Mertebe / Order", "Order"), yaxis_title=t("dB SPL", "dB SPL"), height=560, hovermode="x unified", margin=dict(l=40, r=30, t=70, b=45), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#212529', family="Arial, sans-serif"), xaxis=dict(showgrid=True, gridcolor='#E0E0E0'), yaxis=dict(showgrid=True, gridcolor='#E0E0E0'))
        st.plotly_chart(fig_order, use_container_width=True)
        report_data["figures"]["Order Plot"] = fig_order

        st.markdown("### 🤖 Akıllı Teşhis (Auto-Interpretation)")
        if harmonic_x and harmonic_y:
            max_idx = np.argmax(harmonic_y)
            dominant_order = harmonic_x[max_idx]
            max_db = harmonic_y[max_idx]
            
            if abs(dominant_order - 1.0) < 0.1: diag_tr = "Sistemde **1x (1. Mertebe)** seviyesi baskın. Muhtemel Kök Neden: **Ana şaftta Balanssızlık (Unbalance)**."
            elif abs(dominant_order - 2.0) < 0.1: diag_tr = "Sistemde **2x (2. Mertebe)** seviyesi baskın. Muhtemel Kök Neden: **Kaplin/Şaft Eksen Kaçıklığı veya Gevşeklik (Misalignment / Looseness)**."
            elif dominant_order % 1 != 0: diag_tr = f"Sistemde **{dominant_order}x (Küsuratlı Mertebe)** seviyesi baskın. Muhtemel Kök Neden: **Rulman arızası (Bearing defect) veya Kayış Kayması (Belt Slip)**."
            else: diag_tr = f"Sistemde **{dominant_order}x (Yüksek Tam Sayı)** seviyesi baskın. Belirli kanat/diş sayısına sahip spesifik parçalar incelenmelidir."

            st.info(t(f"💡 **Bulgu:** En yüksek tepe noktası {max_db:.1f} dB ile {dominant_order}x mertebesinde tespit edildi.\n\n🔍 **Teşhis:** {diag_tr}", f"🔍 **Diagnosis:** {diag_tr}"))
            report_data["diagnostics"]["Order Plot"] = diag_tr

    else:
        if rpm_dataframe is None: st.warning(t("Değişken RPM analizi için CSV yükleyin.", "Upload a CSV for variable RPM."))
        else:
            rpm_time, rpm_values = prepare_rpm_series(rpm_dataframe, rpm_column, time_column, duration)
            track_f, track_t, track_psd = spectrogram(audio_signal, fs=sample_rate, window="hann", nperseg=stft_size, noverlap=int(stft_size * 0.75), scaling="density", mode="psd")
            rpm_at_stft, order_tracks = calculate_order_tracks(track_f, track_t, track_psd, rpm_time, rpm_values, selected_orders, 0.05, calibration_offset_db)
            binned_tracks = bin_tracks_by_rpm(rpm_at_stft, order_tracks)

            fig_tracking = go.Figure()
            for so in selected_orders:
                col = f"order_{so:g}"
                if col in binned_tracks.columns: fig_tracking.add_trace(go.Scatter(x=binned_tracks["rpm"], y=binned_tracks[col], mode="lines+markers", name=f"{so:g}× Order"))
            fig_tracking.update_layout(title=t("Order Tracking — RPM'e Göre Mertebe Seviyeleri", "Order Tracking"), xaxis_title="RPM", yaxis_title="dB SPL", height=580, hovermode="x unified", colorway=["#E61A25", "#212529", "#BF2026", "#495057", "#ADADAD"], plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#212529', family="Arial, sans-serif"), xaxis=dict(showgrid=True, gridcolor='#E0E0E0'), yaxis=dict(showgrid=True, gridcolor='#E0E0E0'))
            st.plotly_chart(fig_tracking, use_container_width=True)
            report_data["figures"]["Order Plot"] = fig_tracking

# ============================================================
# TAB 3 — ARTICULATION INDEX / SII
# ============================================================
with tab_ai:
    speech_efforts_map = {t("Normal", "Normal"): np.array([34.75, 34.27, 25.01, 17.32, 9.33, 1.13])}
    octave_noise_levels = octave_band_levels(psd_frequencies, psd, calibration_offset_db, SII_OCTAVE_FREQS)
    sii_table = compute_octave_sii(octave_noise_levels, speech_efforts_map[t("Normal", "Normal")])
    sii_percent = float(sii_table["contribution"].sum()) * 100.0

    fig_sii = go.Figure(go.Indicator(
        mode="gauge+number", value=sii_percent, number={"suffix": " %", "valueformat": ".1f", "font": {"color": "#212529"}},
        title={"text": "Octave-band SII / %AI", "font": {"color": "#212529"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#212529"},
            "steps": [{"range": [0, 30], "color": "#F0F0F0"}, {"range": [30, 70], "color": "#C8C8C8"}, {"range": [70, 100], "color": "#848484"}],
            "bar": {"color": "#E61A25"},
        }
    ))
    fig_sii.update_layout(height=430, paper_bgcolor='rgba(0,0,0,0)', font=dict(family="Arial, sans-serif"))
    st.plotly_chart(fig_sii, use_container_width=True)
    report_data["figures"]["SII Gauge"] = fig_sii

    fig_contribution = go.Figure(go.Bar(x=[format_frequency(v) for v in sii_table["frequency_hz"]], y=100.0 * sii_table["contribution"], marker_color="#E61A25"))
    fig_contribution.update_layout(title=t(f"SII Katkısı (Toplam: %{sii_percent:.1f})", f"SII Contribution (Total: {sii_percent:.1f}%)"), height=430, bargap=0.12, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#212529', family="Arial, sans-serif"), xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='#E0E0E0'))
    st.plotly_chart(fig_contribution, use_container_width=True)
    report_data["figures"]["SII Bands"] = fig_contribution

    st.markdown("### 🤖 Akıllı Teşhis (Auto-Interpretation)")
    if sii_percent >= 75: diag_tr = "Makine çalışma gürültüsü, insan iletişimini engellemiyor. İş güvenliği açısından **%100 güvenli ve konforlu bölge**."
    elif sii_percent >= 45: diag_tr = "Makine gürültüsü konuşmaları kısmen maskeliyor. Etkili iletişim kurmak için personelin **ses yükseltmesi gerekebilir**."
    else: diag_tr = "Makine gürültüsü insan sesini tamamen yutuyor. Operatörler için **kulaklık/yalıtım kesinlikle zorunludur**."
    
    st.info(t(f"💡 **Bulgu:** SII Değeri %{sii_percent:.1f}.\n\n🔍 **Teşhis:** {diag_tr}", f"🔍 **Diagnosis:** {diag_tr}"))
    report_data["diagnostics"]["SII"] = diag_tr

# ============================================================
# TAB 4 — 1/3 OCTAVE BAND PLOTS
# ============================================================
with tab_octave:
    octave_plot_df = third_octave_df[third_octave_df["exact_hz"] <= max_display_frequency].copy()
    octave_plot_df["label"] = octave_plot_df["nominal_hz"].map(format_frequency)

    fig_octave = go.Figure(go.Bar(x=octave_plot_df["label"], y=octave_plot_df["level_db_spl"], marker_color="#E61A25"))
    fig_octave.update_layout(title=t("1/3 Oktav Bant Spektrumu", "1/3 Octave Band Spectrum"), height=560, bargap=0.08, xaxis={"type": "category", "categoryorder": "array", "categoryarray": octave_plot_df["label"].tolist(), "tickangle": -45, "showgrid": False}, yaxis=dict(showgrid=True, gridcolor='#E0E0E0'), margin=dict(l=40, r=30, t=70, b=90), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#212529', family="Arial, sans-serif"))
    st.plotly_chart(fig_octave, use_container_width=True)
    report_data["figures"]["1/3 Octave"] = fig_octave

    st.markdown("### 🤖 Akıllı Teşhis (Auto-Interpretation)")
    low_freq_mask = (octave_plot_df["nominal_hz"] >= 20) & (octave_plot_df["nominal_hz"] <= 250)
    high_freq_mask = (octave_plot_df["nominal_hz"] >= 2000) & (octave_plot_df["nominal_hz"] <= 10000)
    
    low_power = np.sum(10 ** (octave_plot_df.loc[low_freq_mask, "level_db_spl"] / 10))
    high_power = np.sum(10 ** (octave_plot_df.loc[high_freq_mask, "level_db_spl"] / 10))
    
    low_db_total = 10 * np.log10(low_power) if low_power > 0 else 0
    high_db_total = 10 * np.log10(high_power) if high_power > 0 else 0

    if low_db_total > high_db_total + 5: diag_tr = "Spektrumun sol tarafı baskın. **Yapısal titreşimler, balanssızlık veya kalın uğultu (rumble)** sorunları ön planda."
    elif high_db_total > low_db_total + 5: diag_tr = "Spektrumun sağ tarafı baskın. **Sürtünme, keçe deformasyonu veya tiz ıslık/whine** sorunları ön planda."
    else: diag_tr = "Gürültü enerjisi düşük ve yüksek frekanslar arasında dengeli dağılmış (Geniş bantlı gürültü karakteristiği)."

    st.info(t(f"📊 **Enerji Dağılımı:** Düşük Frekans Toplamı: {low_db_total:.1f} dB | Yüksek Frekans Toplamı: {high_db_total:.1f} dB\n\n🔍 **Teşhis:** {diag_tr}", f"🔍 **Diagnosis:** {diag_tr}"))
    report_data["diagnostics"]["1/3 Octave"] = diag_tr

# ============================================================
# PDF RAPORLAMA ARAYÜZÜ VE MODAL POPUP
# ============================================================
if PDF_ENABLED:
    st.sidebar.markdown("---")
    
    @st.dialog("📋 PDF Rapor Detayları (Header Info)")
    def pdf_report_dialog():
        st.write(t("Lütfen rapor antetinde görünecek bilgileri doldurun:", "Please fill in the details for the report header:"))
        
        col1, col2 = st.columns(2)
        with col1:
            report_no = st.text_input("Report-No.:", value="E4119 R0010 J-2603055")
            date_str = st.text_input("Date:", value=datetime.date.today().strftime("%d.%m.%Y"))
            author = st.text_area("Author:", value="E.Ozkuhadar,\nTest Analysis Responsible\nEngineering ESPT - EMEA", height=100)
        with col2:
            location = st.text_input("Location:", value="Technical Center Izmir")
            subject = st.text_input("Subject:", value="Customer return inspection")
            department = st.text_area("Department:", value="S.Cankul\nTest Lab. Supervisor\nEngineering ESPT - EMEA", height=100)
            
        distribution = st.text_input("Distribution list:", value="Torsten Paluszek")
        file_path = st.text_input("File Path (Bottom Footer):", value=r"N:\Engineering\Internal\Working_Folders\18_TEST\03 Test Report Preparation\1) WORD TEST REPORTS\3 CUSTOMER RETURN\E4119\06.07.2026 - 2\E4119 R0010 J-2603055.docx")

        if st.button("Raporu Oluştur (Generate)", type="primary"):
            antet_data = {
                "report_no": report_no, "subject": subject, "date": date_str, "location": location,
                "author": author, "department": department, "distribution": distribution, "file_path": file_path
            }
            with st.spinner("PDF hazırlanıyor..."):
                try:
                    pdf_bytes = build_pdf_report(report_data, antet_data)
                    st.session_state["pdf_bytes"] = pdf_bytes
                    st.session_state.pdf_ready = True
                    st.rerun()
                except Exception as e:
                    st.error(f"PDF Hatası: {e}")

    if st.sidebar.button(t("📄 PDF Raporu Hazırla", "📄 Prepare PDF Report"), use_container_width=True):
        pdf_report_dialog()

    if st.session_state.get("pdf_ready", False) and "pdf_bytes" in st.session_state:
        st.sidebar.download_button(
            label=t("📥 PDF Raporunu İndir", "📥 Download PDF Report"),
            data=st.session_state["pdf_bytes"],
            file_name=f"Gates_NVH_Report_{uploaded_audio.name}.pdf",
            mime="application/pdf",
            use_container_width=True,
            type="primary"
        )
else:
    st.sidebar.markdown("---")
    st.sidebar.warning("PDF Raporu alabilmek için: pip install fpdf2 kaleido==0.2.1")
