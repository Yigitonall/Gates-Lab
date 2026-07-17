import io
import math
import os
import tempfile
import time
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
    
    /* Ana menüdeki büyük butonların tasarımı için ufak dokunuşlar */
    .landing-title { text-align: center; color: #E61A25 !important; font-size: 3rem !important; margin-bottom: 2rem; }
    .landing-subtitle { text-align: center; color: #555555; font-size: 1.2rem; margin-bottom: 3rem; }
</style>
""", unsafe_allow_html=True)

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

# Dil Seçimi Sidebar'ın en üstünde (Sadece mod seçiliyse sidebar gösterilir)
lang = "tr"
if st.session_state.app_mode is not None:
    try:
        st.sidebar.image("gates_logo.png", use_container_width=True)
    except:
        pass
    lang_choice = st.sidebar.radio("🌐 Language / Dil", ["Türkçe", "English"], horizontal=True)
    lang = "tr" if lang_choice == "Türkçe" else "en"

def t(tr_text: str, en_text: str) -> str:
    return tr_text if lang == "tr" else en_text

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
# PDF RAPORLAMA KÜTÜPHANESİ VE TASARIM
# ============================================================
def clean_text_for_fpdf(txt):
    if not isinstance(txt, str): return str(txt)
    tr_map = {'ç':'c', 'ğ':'g', 'ı':'i', 'ö':'o', 'ş':'s', 'ü':'u', 'Ç':'C', 'Ğ':'G', 'İ':'I', 'Ö':'O', 'Ş':'S', 'Ü':'U'}
    for tr, eng in tr_map.items():
        txt = txt.replace(tr, eng)
    txt = txt.replace("**", "")
    return txt.encode('latin-1', 'ignore').decode('latin-1')

if PDF_ENABLED:
    class GatesReport(FPDF):
        def __init__(self, data, antet_data):
            super().__init__()
            self.report_data = data
            self.antet_data = antet_data
            
        def header(self):
            # İlk sayfa hariç diğer sayfaların üst anteti
            if self.page_no() > 1:
                self.set_line_width(0.5)
                # Kırmızı üst şerit
                self.set_fill_color(200, 0, 0)
                self.rect(10.25, 10.25, 189.5, 2, 'F')
                # Dış Çerçeve
                self.rect(10, 10, 190, 24)
                
                # Logo alanı (Siyah-Beyaz Logo)
                try:
                    self.image("gatessiyah_logo.png", x=12, y=14, w=0, h=14)
                except:
                    self.set_font("Arial", 'B', 20)
                    self.set_xy(12, 18)
                    self.cell(40, 10, "GATES")
                    
                # Ortada Report Yazısı
                self.set_font("Arial", 'B', 18)
                self.set_xy(80, 17)
                self.cell(50, 10, "Report", align='C')
                
                # Sağda Report-No
                self.set_font("Arial", 'B', 10)
                self.set_xy(140, 14)
                self.cell(58, 6, clean_text_for_fpdf(f"Report-No.: {self.antet_data.get('report_no', '')}"), align='R')
                
                # Alt gri şerit
                self.set_fill_color(240, 240, 240)
                self.rect(10.25, 30, 189.5, 4, 'F')
                
                # Çizim imlecini (pen) antetin altına alıyoruz ki grafik yazıları taşmasın
                self.set_y(40)

        def footer(self):
            # Alt kısımdan 25mm yukarıya konumlan
            self.set_y(-25)
            self.set_font("Arial", "", 8)
            self.set_line_width(0.5)
            
            box_x = 10
            box_y = self.get_y()
            box_w = 190
            box_h = 16 
            
            # Dış çerçeve ve iç dikey çizgi
            self.rect(box_x, box_y, box_w, box_h)
            self.line(box_x + 160, box_y, box_x + 160, box_y + box_h)
            
            # Sol bölüm - Metin (Yol + Geçerlilik)
            path_text = clean_text_for_fpdf(self.antet_data.get('file_path', ''))
            valid_text = "This document was created electronically and is valid without signature."
            combined_text = f"{path_text}\n{valid_text}"
            
            num_lines = len(combined_text.split('\n'))
            line_height = 4
            text_total_height = num_lines * line_height
            start_y = box_y + (box_h - text_total_height) / 2
            
            self.set_xy(box_x, start_y)
            self.multi_cell(160, line_height, combined_text, align='C')
            
            # Sağ bölüm - Sayfa Numarası
            self.set_xy(box_x + 160, box_y)
            self.cell(30, box_h, f"Page: {self.page_no()} of {{nb}}", align='C')

    def build_pdf_report(report_data, antet_data):
        pdf = GatesReport(report_data, antet_data)
        pdf.alias_nb_pages()
        pdf.set_auto_page_break(auto=True, margin=30)
        pdf.add_page()
        
        # --- İLK SAYFA ANTETİ ---
        pdf.set_line_width(0.5)
        
        # Dış Çerçeveler ve Kutular (Çizgiler)
        pdf.rect(10, 10, 190, 30) # Ana Logo Bloğu
        pdf.rect(10, 40, 190, 12) # Subject Bloğu
        pdf.rect(10, 52, 190, 8)  # Date/Location Bloğu
        pdf.rect(10, 60, 190, 16) # Author/Dept Bloğu
        pdf.rect(10, 76, 190, 10) # Distribution Bloğu
        
        # Dikey Ayırıcı Çizgiler (Seperatörler)
        pdf.line(42, 52, 42, 86)   # Sol etiketlerin (Date, Author, Dist) ayırıcısı
        pdf.line(125, 52, 125, 76) # Orta ayırıcı (Location, Dept)
        pdf.line(152, 52, 152, 76) # Sağ etiketlerin ayırıcısı
        
        # Kırmızı üst şerit ve Gri alt şerit (Dolgular)
        pdf.set_fill_color(200, 0, 0)
        pdf.rect(10.25, 10.25, 189.5, 2, 'F')
        pdf.set_fill_color(240, 240, 240)
        pdf.rect(10.25, 36, 189.5, 4, 'F')
        
        # Logo
        try:
            pdf.image("gatessiyah_logo.png", x=12, y=14, w=0, h=14)
        except:
            pdf.set_font("Arial", 'B', 20)
            pdf.set_xy(12, 19)
            pdf.cell(40, 10, "GATES")
            
        # Report ve Report-No yazıları
        pdf.set_font("Arial", 'B', 20)
        pdf.set_xy(80, 20)
        pdf.cell(50, 10, "Report", align='C')
        pdf.set_font("Arial", 'B', 10)
        pdf.set_xy(140, 16)
        pdf.cell(58, 6, clean_text_for_fpdf(f"Report-No.: {antet_data.get('report_no', '')}"), align='R')
        
        # Antet Metinleri (Hücre İçerikleri)
        pdf.set_font("Arial", '', 10)
        
        pdf.set_xy(11, 43)
        pdf.cell(30, 6, "Subject:")
        pdf.set_font("Arial", 'B', 14)
        pdf.set_xy(43, 43)
        pdf.cell(146, 6, clean_text_for_fpdf(antet_data.get('subject', '')), align='C')
        
        pdf.set_font("Arial", '', 10)
        pdf.set_xy(11, 53)
        pdf.cell(30, 6, "Date:")
        pdf.set_xy(43, 53)
        pdf.cell(81, 6, clean_text_for_fpdf(antet_data.get('date', '')))
        
        pdf.set_xy(126, 53)
        pdf.cell(25, 6, "Location:")
        pdf.set_xy(153, 53)
        pdf.cell(46, 6, clean_text_for_fpdf(antet_data.get('location', '')))
        
        pdf.set_xy(11, 62)
        pdf.cell(30, 6, "Author:")
        pdf.set_font("Arial", '', 9)
        pdf.set_xy(43, 62)
        pdf.multi_cell(81, 4, clean_text_for_fpdf(antet_data.get('author', '')))
        
        pdf.set_font("Arial", '', 10)
        pdf.set_xy(126, 62)
        pdf.cell(25, 6, "Department:")
        pdf.set_font("Arial", '', 9)
        pdf.set_xy(153, 62)
        pdf.multi_cell(46, 4, clean_text_for_fpdf(antet_data.get('department', '')))
        
        pdf.set_font("Arial", '', 10)
        pdf.set_xy(11, 78)
        pdf.cell(30, 6, "Distribution list:")
        pdf.set_xy(43, 78)
        pdf.cell(146, 6, clean_text_for_fpdf(antet_data.get('distribution', '')))
        
        # İçeriğin antetin altına taşmaması için kalem Y pozisyonunu güncelliyoruz
        pdf.set_y(95)
        
        # --- GRAFİKLER VE TEŞHİSLER ---
        # Compare ve Single modlarında basılacak grafik isimleri
        sections = [
            ("Color Map", "Color Map"), 
            ("Color Map (A - Referans)", "Color Map A"), 
            ("Color Map (B - Test)", "Color Map B"),
            ("Order Plot", "Order Plot"), 
            ("SII Bands", "1/3 Octave"), 
            ("SII Gauge", "SII"), 
            ("1/3 Octave", "1/3 Octave")
        ]

        for fig_key, diag_key in sections:
            if fig_key in report_data["figures"]:
                if fig_key != "Color Map" and fig_key != "Color Map (A - Referans)": 
                    pdf.add_page()
                
                pdf.set_font("Arial", 'B', 14)
                title_text = "Articulation Index / SII Analysis" if fig_key == "SII Gauge" else fig_key
                pdf.cell(0, 10, clean_text_for_fpdf(title_text), ln=True)
                
                fig = report_data["figures"][fig_key]
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
                    time.sleep(0.5) # Kaleido rendering delay safety
                    fig.write_image(tmp_img.name, format="png", engine="kaleido", width=800, height=400)
                    pdf.image(tmp_img.name, x=10, w=190)
                    tmp_img_path = tmp_img.name
                
                os.remove(tmp_img_path)
                
                # Compare modunda bands grafiğini SII Gauge ile aynı sayfaya koyalım
                if fig_key == "SII Gauge" and "SII Bands" in report_data["figures"]:
                    fig2 = report_data["figures"]["SII Bands"]
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img2:
                        time.sleep(0.5)
                        fig2.write_image(tmp_img2.name, format="png", engine="kaleido", width=800, height=350)
                        pdf.image(tmp_img2.name, x=10, w=190)
                        tmp_img_path2 = tmp_img2.name
                    os.remove(tmp_img_path2)
                
                if fig_key == "SII Bands":
                    continue # SII Gauge'in altında basıldığı için atlıyoruz
                
                pdf.ln(5)
                
                if diag_key in report_data["diagnostics"]:
                    pdf.set_font("Arial", '', 11)
                    diag_text = ("Teşhis: " if lang_choice == "Türkçe" else "Diagnosis: ") + report_data["diagnostics"][diag_key]
                    pdf.multi_cell(0, 6, clean_text_for_fpdf(diag_text))
                    pdf.ln(10)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
            pdf.output(tmp_pdf.name)
            with open(tmp_pdf.name, "rb") as f:
                pdf_bytes = f.read()
            tmp_pdf_path = tmp_pdf.name
            
        os.remove(tmp_pdf_path)
        return pdf_bytes

# ============================================================
# MÜHENDİSLİK FONKSİYONLARI
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
# 1. KARŞILAMA EKRANI (LANDING PAGE)
# ============================================================
if st.session_state.app_mode is None:
    st.markdown('<div class="landing-title">GATES R&D NVH ANALYSIS SYSTEM</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="landing-subtitle">{t("Lütfen yapmak istediğiniz analiz tipini seçin", "Please select the type of analysis you want to perform")}</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns([1, 3, 3, 1])
    
    with col2:
        if st.button(t("🔍 TEKLİ SES ANALİZİ\n(Single Analysis)", "🔍 SINGLE NOISE ANALYSIS\n(Standard)"), use_container_width=True):
            st.session_state.app_mode = "single"
            st.rerun()
            
    with col3:
        if st.button(t("⚖️ A/B KARŞILAŞTIRMA ANALİZİ\n(Comparative Analysis)", "⚖️ A/B COMPARATIVE ANALYSIS\n(Reference vs Test)"), use_container_width=True):
            st.session_state.app_mode = "compare"
            st.rerun()
            
    st.stop()

# ============================================================
# ORTAK YAN MENÜ (SIDEBAR) BİLEŞENLERİ
# ============================================================
st.sidebar.button(t("⬅️ Ana Menüye Dön", "⬅️ Back to Main Menu"), on_click=go_to_main_menu, use_container_width=True)
st.sidebar.markdown("---")

st.title(t("🔊 Gürültü ve Akustik Analiz Sistemi (NVH)", "🔊 Noise and Acoustic Analysis System (NVH)"))
st.caption("COLOR MAPS • ORDER PLOTS • ARTICULATION INDEX / SII • 1/3 OCTAVE BAND PLOTS")

# PDF DİALOG MODÜLÜ
@st.dialog(t("📄 PDF Rapor Bilgilerini Girin", "📄 Enter PDF Report Information"))
def pdf_info_dialog(report_data):
    st.write(t("Lütfen raporun ilk sayfasındaki antette görünecek bilgileri doldurun.", 
               "Please fill in the information that will appear in the header on the first page of the report."))
    col1, col2 = st.columns(2)
    with col1:
        subject = st.text_input("Subject:", value="Customer return inspection")
        date_val = st.text_input("Date:", value="10.07.2026")
        author = st.text_area("Author:", value="E.Ozcuhadar,\nTest Analysis Responsible\nEngineering ESPT – EMEA", height=100)
    with col2:
        report_no = st.text_input("Report-No.:", value="E4119 R0010 J-2603055")
        location = st.text_input("Location:", value="Technical Center Izmir")
        department = st.text_area("Department:", value="S.Cankul\nTest Lab. Supervisor\nEngineering ESPT – EMEA", height=100)
        
    distribution = st.text_input("Distribution list:", value="Torsten Paluszek")
    file_path = st.text_input(t("Dosya Yolu (Alt Footer):", "File Path (Bottom Footer):"), value=r"N:\Engineering\Internal\Working_Folders\18_TEST\03 Test Report Preparation\1) WORD TEST REPORTS\3 CUSTOMER RETURN\E4119\06.07.2026 - 2\E4119 R0010 J-2603055.docx")
    
    if st.button(t("✅ Raporu Oluştur", "✅ Generate Report"), use_container_width=True):
        antet_data = {
            "subject": subject, "date": date_val, "author": author, "report_no": report_no,
            "location": location, "department": department, "distribution": distribution, "file_path": file_path
        }
        with st.spinner(t("PDF oluşturuluyor, grafikler işleniyor (Lütfen bekleyin)...", "Generating PDF, processing charts (Please wait)...")):
            try:
                pdf_bytes = build_pdf_report(report_data, antet_data)
                st.session_state["pdf_bytes"] = pdf_bytes
                st.session_state.pdf_ready = True
                st.rerun()
            except Exception as e:
                st.error(t(f"PDF Oluşturma Hatası: {e}\nLütfen 'pip install fpdf2 kaleido' kurulu olduğundan emin olun.", f"PDF Error: {e}"))

# ============================================================
# 2. TEKLİ ANALİZ MODU (SINGLE ANALYSIS)
# ============================================================
if st.session_state.app_mode == "single":
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

    if st.sidebar.button(t("🚀 Analiz Yap", "🚀 Run Analysis"), type="primary", use_container_width=True):
        if uploaded_audio is not None:
            st.session_state.analyze = True
            st.session_state.pdf_ready = False
        else:
            st.sidebar.error(t("Lütfen önce bir WAV dosyası yükleyin!", "Please upload a WAV file first!"))

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

    report_data = {"file_name": uploaded_audio.name, "max_spl": reference_leq_db, "rpm_info": f"{fixed_rpm} RPM" if rpm_mode in ["Sabit RPM", "Fixed RPM"] else t("Değişken CSV", "Variable CSV"), "figures": {}, "diagnostics": {}}

    tab_color, tab_order, tab_ai, tab_octave = st.tabs(["🌈 COLOR MAPS", "🏎️ ORDER PLOTS", "🧠 ARTICULATION INDEX / SII", "🎼 1/3 OCTAVE BAND PLOTS"])

    with tab_color:
        stft_size = choose_segment_size(audio_signal.size, int(default_stft))
        stft_overlap = int(stft_size * 0.75)
        spec_f, spec_t, spec_psd = spectrogram(audio_signal, fs=sample_rate, window="hann", nperseg=stft_size, noverlap=stft_overlap, detrend="constant", scaling="density", mode="psd")
        spec_df = spec_f[1] - spec_f[0] if spec_f.size > 1 else 1.0
        spec_level_db = power_to_db(spec_psd * spec_df, calibration_offset_db)
        frequency_mask = (spec_f >= 20.0) & (spec_f <= max_display_frequency)

        fig_color = go.Figure(go.Heatmap(x=spec_t, y=spec_f[frequency_mask], z=spec_level_db[frequency_mask, :], colorscale="Turbo", zmin=reference_leq_db - 80.0, zmax=reference_leq_db + 5.0, colorbar={"title": t("Seviye<br>[dB]", "Level<br>[dB]")}))
        fig_color.update_layout(title=t("Kalibre Edilmiş Akustik Spektrogram (Logaritmik Ölçek)", "Calibrated Acoustic Spectrogram (Logarithmic Scale)"), xaxis_title=t("Zaman [s]", "Time [s]"), yaxis_title=t("Frekans [Hz]", "Frequency [Hz]"), yaxis_type="log", height=620)
        st.plotly_chart(fig_color, use_container_width=True)
        report_data["figures"]["Color Map"] = fig_color

        st.markdown(t("### 🤖 Akıllı Teşhis", "### 🤖 Auto-Interpretation"))
        db_matrix = spec_level_db[frequency_mask, :]
        time_variance = np.var(np.mean(db_matrix, axis=0))
        freq_variance = np.var(np.mean(db_matrix, axis=1))

        if time_variance > freq_variance * 1.5: 
            diag_tr = "Spektrogramda zamana bağlı ani enerji değişimleri (Dikey izler) tespit edildi. Muhtemel Kök Neden: **Anlık vuruntular, metal çarpması veya darbe (Impact) gürültüsü**."
            diag_en = "Sudden time-dependent energy changes (Vertical traces) detected in the spectrogram. Probable Root Cause: **Instantaneous knocks, metal clashing, or impact noise**."
        elif freq_variance > time_variance * 1.5: 
            diag_tr = "Spektrogramda belirli frekans bantlarında yoğunlaşma (Yatay bantlar) tespit edildi. Muhtemel Kök Neden: **Dönen parçalardan kaynaklı sürekli inilti, sürtünme veya harmonik gürültü**."
            diag_en = "Concentration in specific frequency bands (Horizontal bands) detected in the spectrogram. Probable Root Cause: **Continuous whine, friction, or harmonic noise caused by rotating parts**."
        else: 
            diag_tr = "Spektrogramda hem zamana hem de frekansa yayılan karmaşık bir gürültü profili gözlemleniyor. Karmaşık (Geniş bantlı) titreşimler incelenmelidir."
            diag_en = "A complex noise profile spreading across both time and frequency is observed. Complex (Broadband) vibrations should be investigated."
        
        diag_final = t(diag_tr, diag_en)
        st.info(t(f"💡 **Bulgu:** Sinyalin zaman ve frekans eksenindeki enerji dağılım varyansı analiz edildi.\n\n🔍 **Teşhis:** {diag_final}", f"💡 **Finding:** The energy distribution variance of the signal in time and frequency axes was analyzed.\n\n🔍 **Diagnosis:** {diag_final}"))
        report_data["diagnostics"]["Color Map"] = diag_final

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
            fig_order.update_layout(title=f"{t('Order Spektrumu', 'Order Spectrum')} — {float(fixed_rpm):.0f} RPM", xaxis_title=t("Mertebe / Order", "Order"), yaxis_title=t("dB SPL", "dB SPL"), height=560)
            st.plotly_chart(fig_order, use_container_width=True)
            report_data["figures"]["Order Plot"] = fig_order

            st.markdown(t("### 🤖 Akıllı Teşhis", "### 🤖 Auto-Interpretation"))
            if harmonic_x and harmonic_y:
                max_idx = np.argmax(harmonic_y)
                dominant_order = harmonic_x[max_idx]
                max_db = harmonic_y[max_idx]
                
                if abs(dominant_order - 1.0) < 0.1: 
                    diag_tr = "Sistemde **1x (1. Mertebe)** seviyesi baskın. Muhtemel Kök Neden: **Ana şaftta Balanssızlık (Unbalance)**."
                    diag_en = "**1x (1st Order)** level is dominant in the system. Probable Root Cause: **Main shaft Unbalance**."
                elif abs(dominant_order - 2.0) < 0.1: 
                    diag_tr = "Sistemde **2x (2. Mertebe)** seviyesi baskın. Muhtemel Kök Neden: **Kaplin/Şaft Eksen Kaçıklığı veya Gevşeklik (Misalignment / Looseness)**."
                    diag_en = "**2x (2nd Order)** level is dominant in the system. Probable Root Cause: **Coupling/Shaft Misalignment or Looseness**."
                elif dominant_order % 1 != 0: 
                    diag_tr = f"Sistemde **{dominant_order}x (Küsuratlı Mertebe)** seviyesi baskın. Muhtemel Kök Neden: **Rulman arızası (Bearing defect) veya Kayış Kayması (Belt Slip)**."
                    diag_en = f"**{dominant_order}x (Fractional Order)** level is dominant. Probable Root Cause: **Bearing defect or Belt Slip**."
                else: 
                    diag_tr = f"Sistemde **{dominant_order}x (Yüksek Tam Sayı)** seviyesi baskın. Belirli kanat/diş sayısına sahip spesifik parçalar incelenmelidir."
                    diag_en = f"**{dominant_order}x (High Integer Order)** level is dominant. Specific parts with a matching number of blades/teeth should be investigated."

                diag_final = t(diag_tr, diag_en)
                st.info(t(f"💡 **Bulgu:** En yüksek tepe noktası {max_db:.1f} dB ile {dominant_order}x mertebesinde tespit edildi.\n\n🔍 **Teşhis:** {diag_final}", f"💡 **Finding:** The highest peak was detected at order {dominant_order}x with {max_db:.1f} dB.\n\n🔍 **Diagnosis:** {diag_final}"))
                report_data["diagnostics"]["Order Plot"] = diag_final

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
                fig_tracking.update_layout(title=t("Order Tracking — RPM'e Göre Mertebe Seviyeleri", "Order Tracking"), xaxis_title="RPM", yaxis_title="dB SPL", height=580)
                st.plotly_chart(fig_tracking, use_container_width=True)
                report_data["figures"]["Order Plot"] = fig_tracking

    with tab_ai:
        speech_efforts_map = {
            t("Normal", "Normal"): np.array([34.75, 34.27, 25.01, 17.32, 9.33, 1.13]),
            t("Yüksek Sesle", "Raised"): np.array([41.5, 41.6, 35.5, 29.5, 22.8, 14.2]),
            t("Bağırarak", "Loud"): np.array([45.4, 48.6, 46.1, 41.1, 35.6, 27.2]),
            t("Çığlık Atarak", "Shout"): np.array([46.4, 53.3, 56.4, 52.1, 46.5, 38.0])
        }
        
        col_speech, col_space = st.columns([1, 2])
        with col_speech:
            selected_effort = st.selectbox(t("İletişim Eforu", "Speech Effort"), list(speech_efforts_map.keys()))

        octave_noise_levels = octave_band_levels(psd_frequencies, psd, calibration_offset_db, SII_OCTAVE_FREQS)
        sii_table = compute_octave_sii(octave_noise_levels, speech_efforts_map[selected_effort])
        sii_percent = float(sii_table["contribution"].sum()) * 100.0

        color_sii = "#28a745" if sii_percent >= 75 else "#ffc107" if sii_percent >= 45 else "#dc3545"

        col_gauge, col_bar = st.columns(2)
        with col_gauge:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=sii_percent, title={'text': "SII Skoru"}, domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': color_sii},
                    'bgcolor': "white", 'borderwidth': 2, 'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 45], 'color': 'rgba(220, 53, 69, 0.3)'},
                        {'range': [45, 75], 'color': 'rgba(255, 193, 7, 0.3)'},
                        {'range': [75, 100], 'color': 'rgba(40, 167, 69, 0.3)'}
                    ]
                }
            ))
            fig_gauge.update_layout(height=400, margin=dict(l=30, r=30, t=50, b=30))
            st.plotly_chart(fig_gauge, use_container_width=True)
            report_data["figures"]["SII Gauge"] = fig_gauge

        with col_bar:
            fig_contribution = go.Figure(go.Bar(
                x=[format_frequency(v) for v in sii_table["frequency_hz"]], y=100.0 * sii_table["contribution"], 
                marker_color="#E61A25", marker_line_color="#B0101A", marker_line_width=1
            ))
            fig_contribution.update_layout(title=t(f"Frekanslara Göre SII Katkısı", f"SII Contribution by Frequency"), height=400, bargap=0.1, xaxis_type='category')
            st.plotly_chart(fig_contribution, use_container_width=True)
            report_data["figures"]["SII Bands"] = fig_contribution

        st.markdown(t("### 🤖 Akıllı Teşhis", "### 🤖 Auto-Interpretation"))
        if sii_percent >= 75: 
            diag_tr = "Makine çalışma gürültüsü, insan iletişimini engellemiyor. İş güvenliği açısından **%100 güvenli ve konforlu bölge**."
            diag_en = "Machine operating noise does not hinder human communication. **100% safe and comfortable zone** in terms of occupational safety."
        elif sii_percent >= 45: 
            diag_tr = "Makine gürültüsü konuşmaları kısmen maskeliyor. Etkili iletişim kurmak için personelin **ses yükseltmesi gerekebilir**."
            diag_en = "Machine noise partially masks conversations. Personnel **may need to raise their voice** to communicate effectively."
        else: 
            diag_tr = "Makine gürültüsü insan sesini tamamen yutuyor. Operatörler için **kulaklık/yalıtım kesinlikle zorunludur**."
            diag_en = "Machine noise completely swallows the human voice. **Headsets/insulation are absolutely mandatory** for operators."
        
        diag_final = t(diag_tr, diag_en)
        st.info(t(f"💡 **Bulgu:** SII Değeri %{sii_percent:.1f}.\n\n🔍 **Teşhis:** {diag_final}", f"💡 **Finding:** SII Score is {sii_percent:.1f}%.\n\n🔍 **Diagnosis:** {diag_final}"))
        report_data["diagnostics"]["SII"] = diag_final

    with tab_octave:
        octave_plot_df = third_octave_df[third_octave_df["exact_hz"] <= max_display_frequency].copy()
        octave_plot_df["label"] = octave_plot_df["nominal_hz"].map(format_frequency)

        fig_octave = go.Figure(go.Bar(
            x=octave_plot_df["label"], y=octave_plot_df["level_db_spl"], 
            marker_color="#E61A25", marker_line_color="#B0101A", marker_line_width=1
        ))
        fig_octave.update_layout(title=t("1/3 Oktav Bant Spektrumu", "1/3 Octave Band Spectrum"), height=560, bargap=0.05, xaxis_type='category', xaxis_tickangle=-45)
        st.plotly_chart(fig_octave, use_container_width=True)
        report_data["figures"]["1/3 Octave"] = fig_octave

        st.markdown(t("### 🤖 Akıllı Teşhis", "### 🤖 Auto-Interpretation"))
        low_freq_mask = (octave_plot_df["nominal_hz"] >= 20) & (octave_plot_df["nominal_hz"] <= 250)
        high_freq_mask = (octave_plot_df["nominal_hz"] >= 2000) & (octave_plot_df["nominal_hz"] <= 10000)
        
        low_power = np.sum(10 ** (octave_plot_df.loc[low_freq_mask, "level_db_spl"] / 10))
        high_power = np.sum(10 ** (octave_plot_df.loc[high_freq_mask, "level_db_spl"] / 10))
        
        low_db_total = 10 * np.log10(low_power) if low_power > 0 else 0
        high_db_total = 10 * np.log10(high_power) if high_power > 0 else 0

        if low_db_total > high_db_total + 5: 
            diag_tr = "Spektrumun sol tarafı baskın. **Yapısal titreşimler, balanssızlık veya kalın uğultu (rumble)** sorunları ön planda."
            diag_en = "Left side of the spectrum is dominant. **Structural vibrations, unbalance, or deep rumble** issues are prominent."
        elif high_db_total > low_db_total + 5: 
            diag_tr = "Spektrumun sağ tarafı baskın. **Sürtünme, keçe deformasyonu veya tiz ıslık/whine** sorunları ön planda."
            diag_en = "Right side of the spectrum is dominant. **Friction, seal deformation, or high-pitched whistle/whine** issues are prominent."
        else: 
            diag_tr = "Gürültü enerjisi düşük ve yüksek frekanslar arasında dengeli dağılmış (Geniş bantlı gürültü karakteristiği)."
            diag_en = "Noise energy is evenly distributed between low and high frequencies (Broadband noise characteristic)."

        diag_final = t(diag_tr, diag_en)
        st.info(t(f"📊 **Enerji Dağılımı:** Düşük Frekans Toplamı: {low_db_total:.1f} dB | Yüksek Frekans Toplamı: {high_db_total:.1f} dB\n\n🔍 **Teşhis:** {diag_final}", f"📊 **Energy Distribution:** Low Freq Total: {low_db_total:.1f} dB | High Freq Total: {high_db_total:.1f} dB\n\n🔍 **Diagnosis:** {diag_final}"))
        report_data["diagnostics"]["1/3 Octave"] = diag_final

    if PDF_ENABLED:
        st.sidebar.markdown("---")
        if st.sidebar.button(t("📄 PDF Raporu Hazırla", "📄 Prepare PDF Report"), use_container_width=True):
            pdf_info_dialog(report_data)

        if st.session_state.get("pdf_ready", False) and "pdf_bytes" in st.session_state:
            st.sidebar.download_button(label=t("📥 PDF Raporunu İndir", "📥 Download PDF Report"), data=st.session_state["pdf_bytes"], file_name=f"Gates_NVH_Report_{uploaded_audio.name}.pdf", mime="application/pdf", use_container_width=True, type="primary")

# ============================================================
# 3. KARŞILAŞTIRMA MODU (A/B COMPARATIVE ANALYSIS)
# ============================================================
elif st.session_state.app_mode == "compare":
    st.sidebar.header(t("📁 Karşılaştırmalı Veri Girişi", "📁 Comparative Data Entry"))

    uploaded_files = st.sidebar.file_uploader(t("1. WAV Dosyaları (A ve B)", "1. WAV Files (A and B)"), type=["wav"], accept_multiple_files=True, on_change=reset_analysis)

    ref_spl, test_spl = 80.0, 80.0
    if len(uploaded_files) == 2:
        st.sidebar.subheader(t("🎚️ SPL Kalibrasyonları (MAX HOLD)", "🎚️ SPL Calibrations (MAX HOLD)"))
        ref_spl = st.sidebar.number_input(t(f"MAX SPL: {uploaded_files[0].name} (A)", f"MAX SPL: {uploaded_files[0].name} (A)"), min_value=20.0, max_value=140.0, value=80.0, step=0.1, on_change=reset_analysis)
        test_spl = st.sidebar.number_input(t(f"MAX SPL: {uploaded_files[1].name} (B)", f"MAX SPL: {uploaded_files[1].name} (B)"), min_value=20.0, max_value=140.0, value=80.0, step=0.1, on_change=reset_analysis)
    elif len(uploaded_files) > 2:
        st.sidebar.error(t("Lütfen A/B analizi için tam olarak 2 adet dosya bırakın.", "Please leave exactly 2 files for A/B analysis."))
    
    st.sidebar.subheader(t("🏎️ RPM Bilgisi (Ortak)", "🏎️ Shared RPM Information"))
    fixed_rpm = st.sidebar.number_input(t("Sabit dönüş hızı [RPM]", "Fixed rotation speed [RPM]"), min_value=1.0, value=1500.0, step=10.0, on_change=reset_analysis)

    st.sidebar.markdown("---")

    if st.sidebar.button(t("🚀 Karşılaştırmalı Analiz Yap", "🚀 Run Comparative Analysis"), type="primary", use_container_width=True):
        if len(uploaded_files) == 2:
            st.session_state.analyze = True
            st.session_state.pdf_ready = False
        else:
            st.sidebar.error(t("Karşılaştırma için 2 adet dosya yüklemelisiniz!", "You must upload 2 files for comparison!"))

    if not uploaded_files or len(uploaded_files) != 2:
        st.info(t("ℹ️ A/B Testi için lütfen sol panelden **tam olarak 2 adet .wav** dosyası yükleyin.", 
                  "ℹ️ For A/B Testing, please upload **exactly 2 .wav** files from the left panel."))
        st.stop()

    if not st.session_state.analyze:
        st.info(t("👈 Ayarları tamamladıktan sonra 'Karşılaştırmalı Analiz Yap' butonuna tıklayın.", 
                  "👈 After completing the settings, click 'Run Comparative Analysis'."))
        st.stop()

    with st.spinner(t("Karşılaştırmalı akustik veriler işleniyor...", "Processing comparative acoustic data...")):
        
        # Dosya A işleme
        sr_A, sig_A = read_wav_mono(uploaded_files[0])
        nyq_A = sr_A / 2.0
        w_size_A = choose_segment_size(sig_A.size, 16384)
        psd_f_A, psd_A = welch(sig_A, fs=sr_A, window="hann", nperseg=w_size_A, noverlap=w_size_A // 2, scaling="density")
        calib_A, _ = calculate_calibration_offset(psd_f_A, psd_A, ref_spl, min(20000.0, nyq_A))
        third_oct_A = third_octave_levels(psd_f_A, psd_A, calib_A, nyq_A)

        # Dosya B işleme
        sr_B, sig_B = read_wav_mono(uploaded_files[1])
        nyq_B = sr_B / 2.0
        w_size_B = choose_segment_size(sig_B.size, 16384)
        psd_f_B, psd_B = welch(sig_B, fs=sr_B, window="hann", nperseg=w_size_B, noverlap=w_size_B // 2, scaling="density")
        calib_B, _ = calculate_calibration_offset(psd_f_B, psd_B, test_spl, min(20000.0, nyq_B))
        third_oct_B = third_octave_levels(psd_f_B, psd_B, calib_B, nyq_B)

        max_display_frequency = min(20000.0, nyq_A, nyq_B)

    st.success(t("✅ Karşılaştırmalı Analiz Tamamlandı!", "✅ Comparative Analysis Complete!"))

    report_data = {
        "file_name": f"{uploaded_files[0].name} vs {uploaded_files[1].name}",
        "max_spl": f"A:{ref_spl} / B:{test_spl}",
        "report_no": "COMP-REPORT",
        "rpm_info": f"{fixed_rpm} RPM",
        "figures": {}, "diagnostics": {}
    }

    tab_color, tab_order, tab_ai, tab_octave = st.tabs(["🌈 COLOR MAPS", "🏎️ ORDER PLOTS (A vs B)", "🧠 SII (A vs B)", "🎼 1/3 OCTAVE (A vs B)"])

    # --- COLOR MAPS ---
    with tab_color:
        stft_size_A = choose_segment_size(sig_A.size, 4096)
        sf_A, st_A, spsd_A = spectrogram(sig_A, fs=sr_A, window="hann", nperseg=stft_size_A, noverlap=int(stft_size_A * 0.75), mode="psd")
        slvl_A = power_to_db(spsd_A * (sf_A[1]-sf_A[0]), calib_A)
        mask_A = (sf_A >= 20.0) & (sf_A <= min(20000.0, nyq_A))

        stft_size_B = choose_segment_size(sig_B.size, 4096)
        sf_B, st_B, spsd_B = spectrogram(sig_B, fs=sr_B, window="hann", nperseg=stft_size_B, noverlap=int(stft_size_B * 0.75), mode="psd")
        slvl_B = power_to_db(spsd_B * (sf_B[1]-sf_B[0]), calib_B)
        mask_B = (sf_B >= 20.0) & (sf_B <= min(20000.0, nyq_B))

        col_A, col_B = st.columns(2)
        with col_A:
            fig_cmap_A = go.Figure(go.Heatmap(x=st_A, y=sf_A[mask_A], z=slvl_A[mask_A, :], colorscale="Turbo", zmin=ref_spl-80, zmax=ref_spl+5))
            fig_cmap_A.update_layout(title=t(f"Dosya A: {uploaded_files[0].name}", f"File A: {uploaded_files[0].name}"), yaxis_type="log", height=450)
            st.plotly_chart(fig_cmap_A, use_container_width=True)
            report_data["figures"]["Color Map (A - Referans)"] = fig_cmap_A

        with col_B:
            fig_cmap_B = go.Figure(go.Heatmap(x=st_B, y=sf_B[mask_B], z=slvl_B[mask_B, :], colorscale="Turbo", zmin=test_spl-80, zmax=test_spl+5))
            fig_cmap_B.update_layout(title=t(f"Dosya B: {uploaded_files[1].name}", f"File B: {uploaded_files[1].name}"), yaxis_type="log", height=450)
            st.plotly_chart(fig_cmap_B, use_container_width=True)
            report_data["figures"]["Color Map (B - Test)"] = fig_cmap_B

    # --- ORDER PLOTS ---
    with tab_order:
        rot_hz = float(fixed_rpm) / 60.0
        max_ord = st.slider(t("Gösterilecek maksimum order", "Maximum order to display"), 1.0, 10.0, 5.0, 0.5)
        
        ord_df_A = order_spectrum_from_psd(psd_f_A, psd_A, rot_hz, max_ord, 0.05, calib_A)
        ord_df_B = order_spectrum_from_psd(psd_f_B, psd_B, rot_hz, max_ord, 0.05, calib_B)

        fig_ord_comp = go.Figure()
        fig_ord_comp.add_trace(go.Scatter(x=ord_df_A["order"], y=ord_df_A["level_db_spl"], mode="lines", name=t(f"A: {uploaded_files[0].name}", f"A: {uploaded_files[0].name}"), line=dict(color="#1f77b4", width=2)))
        fig_ord_comp.add_trace(go.Scatter(x=ord_df_B["order"], y=ord_df_B["level_db_spl"], mode="lines", name=t(f"B: {uploaded_files[1].name}", f"B: {uploaded_files[1].name}"), line=dict(color="#E61A25", width=2)))
        fig_ord_comp.update_layout(title=t("Karşılaştırmalı Order Spektrumu (A vs B)", "Comparative Order Spectrum (A vs B)"), xaxis_title="Order", yaxis_title="dB SPL", height=500)
        st.plotly_chart(fig_ord_comp, use_container_width=True)
        report_data["figures"]["Order Plot"] = fig_ord_comp

        # Teşhis
        if not ord_df_A.empty and not ord_df_B.empty:
            max_idx_A = ord_df_A["level_db_spl"].idxmax()
            max_idx_B = ord_df_B["level_db_spl"].idxmax()
            ord_A_dom = ord_df_A.loc[max_idx_A, "order"]
            ord_B_dom = ord_df_B.loc[max_idx_B, "order"]
            val_A = ord_df_A.loc[max_idx_B, "level_db_spl"] # B'nin zirve yaptığı noktadaki A'nın değeri
            val_B = ord_df_B.loc[max_idx_B, "level_db_spl"]
            diff = val_B - val_A

            if diff > 3:
                diag_tr = f"B dosyasında {ord_B_dom}x mertebesinde, A dosyasına göre **+{diff:.1f} dB artış** tespit edildi. Bu durum B parçasında mekanik bir aşınma/bozulma olduğunu gösterir."
                diag_en = f"A **+{diff:.1f} dB increase** was detected in File B at order {ord_B_dom}x compared to File A. This indicates mechanical wear/degradation in part B."
            elif diff < -3:
                diag_tr = f"B dosyasında gürültü {ord_B_dom}x mertebesinde A'ya göre **{abs(diff):.1f} dB iyileşmiş** (düşmüş)."
                diag_en = f"The noise in File B has **improved (decreased) by {abs(diff):.1f} dB** at order {ord_B_dom}x compared to A."
            else:
                diag_tr = "Her iki dosyanın mertebe gürültü seviyeleri genel olarak benzer karakteristikte."
                diag_en = "The order noise levels of both files have generally similar characteristics."

            st.info(t(f"💡 **Bulgu:** B dosyasındaki en baskın tepe noktası {ord_B_dom}x.\n\n🔍 **Kıyaslama Teşhisi:** {diag_tr}", 
                      f"💡 **Finding:** The most dominant peak in File B is {ord_B_dom}x.\n\n🔍 **Comparative Diagnosis:** {diag_en}"))
            report_data["diagnostics"]["Order Plot"] = t(diag_tr, diag_en)

    # --- SII COMP ---
    with tab_ai:
        speech_efforts_map = {
            t("Normal", "Normal"): np.array([34.75, 34.27, 25.01, 17.32, 9.33, 1.13]),
            t("Yüksek Sesle", "Raised"): np.array([41.5, 41.6, 35.5, 29.5, 22.8, 14.2])
        }
        selected_eff_comp = st.selectbox(t("İletişim Eforu", "Speech Effort"), list(speech_efforts_map.keys()), key="comp_eff")
        
        oct_A = octave_band_levels(psd_f_A, psd_A, calib_A, SII_OCTAVE_FREQS)
        sii_df_A = compute_octave_sii(oct_A, speech_efforts_map[selected_eff_comp])
        sii_pct_A = float(sii_df_A["contribution"].sum()) * 100.0
        
        oct_B = octave_band_levels(psd_f_B, psd_B, calib_B, SII_OCTAVE_FREQS)
        sii_df_B = compute_octave_sii(oct_B, speech_efforts_map[selected_eff_comp])
        sii_pct_B = float(sii_df_B["contribution"].sum()) * 100.0

        fig_gauge_comp = make_subplots(rows=1, cols=2, specs=[[{'type': 'indicator'}, {'type': 'indicator'}]], subplot_titles=(uploaded_files[0].name, uploaded_files[1].name))
        
        color_A = "#28a745" if sii_pct_A >= 75 else "#ffc107" if sii_pct_A >= 45 else "#dc3545"
        color_B = "#28a745" if sii_pct_B >= 75 else "#ffc107" if sii_pct_B >= 45 else "#dc3545"

        fig_gauge_comp.add_trace(go.Indicator(mode="gauge+number", value=sii_pct_A, title={'text': "SII A"},
            gauge={'axis': {'range': [0, 100]}, 'bar': {'color': color_A}, 'steps': [{'range': [0, 45], 'color': '#ffcdd2'}, {'range': [45, 75], 'color': '#fff3cd'}, {'range': [75, 100], 'color': '#d4edda'}]}
        ), row=1, col=1)
        fig_gauge_comp.add_trace(go.Indicator(mode="gauge+number", value=sii_pct_B, title={'text': "SII B"},
            gauge={'axis': {'range': [0, 100]}, 'bar': {'color': color_B}, 'steps': [{'range': [0, 45], 'color': '#ffcdd2'}, {'range': [45, 75], 'color': '#fff3cd'}, {'range': [75, 100], 'color': '#d4edda'}]}
        ), row=1, col=2)
        fig_gauge_comp.update_layout(height=350, margin=dict(t=50, b=30))
        st.plotly_chart(fig_gauge_comp, use_container_width=True)
        report_data["figures"]["SII Gauge"] = fig_gauge_comp

        fig_sii_bands = go.Figure()
        fig_sii_bands.add_trace(go.Bar(x=[format_frequency(v) for v in sii_df_A["frequency_hz"]], y=100.0 * sii_df_A["contribution"], name="File A", marker_color="#1f77b4"))
        fig_sii_bands.add_trace(go.Bar(x=[format_frequency(v) for v in sii_df_B["frequency_hz"]], y=100.0 * sii_df_B["contribution"], name="File B", marker_color="#E61A25"))
        fig_sii_bands.update_layout(title="SII Katkısı (A vs B)", barmode='group', height=400, xaxis_type='category')
        st.plotly_chart(fig_sii_bands, use_container_width=True)
        report_data["figures"]["SII Bands"] = fig_sii_bands

        diff_sii = sii_pct_B - sii_pct_A
        if diff_sii < -10:
            diag_tr = f"B dosyasında makine gürültüsü, A dosyasına göre **%{abs(diff_sii):.1f} daha fazla maskeleme** yapıyor. İletişim ergonomisi kötüleşmiş."
            diag_en = f"Machine noise in File B masks human communication **{abs(diff_sii):.1f}% more** than File A. Communication ergonomics have worsened."
        elif diff_sii > 10:
            diag_tr = f"B dosyasında makine gürültüsü azalmış ve iletişim ortamı **%{diff_sii:.1f} oranında iyileşmiş**."
            diag_en = f"Machine noise has decreased in File B, and the communication environment has **improved by {diff_sii:.1f}%**."
        else:
            diag_tr = "İki dosya arasında insan iletişimini engelleme açısından belirgin bir fark yoktur."
            diag_en = "There is no significant difference between the two files in terms of hindering human communication."

        st.info(t(f"🔍 **Kıyaslama Teşhisi:** {diag_tr}", f"🔍 **Comparative Diagnosis:** {diag_en}"))
        report_data["diagnostics"]["SII"] = t(diag_tr, diag_en)

    # --- OCTAVE COMP ---
    with tab_octave:
        oct_df_A = third_oct_A[third_oct_A["exact_hz"] <= max_display_frequency].copy()
        oct_df_B = third_oct_B[third_oct_B["exact_hz"] <= max_display_frequency].copy()
        
        fig_oct_comp = go.Figure()
        fig_oct_comp.add_trace(go.Bar(x=oct_df_A["nominal_hz"].map(format_frequency), y=oct_df_A["level_db_spl"], name="File A", marker_color="#1f77b4"))
        fig_oct_comp.add_trace(go.Bar(x=oct_df_B["nominal_hz"].map(format_frequency), y=oct_df_B["level_db_spl"], name="File B", marker_color="#E61A25"))
        fig_oct_comp.update_layout(title="1/3 Oktav Spektrumu (A vs B)", barmode='group', height=500, xaxis_type='category', xaxis_tickangle=-45)
        st.plotly_chart(fig_oct_comp, use_container_width=True)
        report_data["figures"]["1/3 Octave"] = fig_oct_comp

        low_mask = (oct_df_A["nominal_hz"] >= 20) & (oct_df_A["nominal_hz"] <= 250)
        hi_mask = (oct_df_A["nominal_hz"] >= 2000) & (oct_df_A["nominal_hz"] <= 10000)
        
        low_A = 10 * np.log10(np.sum(10 ** (oct_df_A.loc[low_mask, "level_db_spl"] / 10)))
        hi_A = 10 * np.log10(np.sum(10 ** (oct_df_A.loc[hi_mask, "level_db_spl"] / 10)))
        low_B = 10 * np.log10(np.sum(10 ** (oct_df_B.loc[low_mask, "level_db_spl"] / 10)))
        hi_B = 10 * np.log10(np.sum(10 ** (oct_df_B.loc[hi_mask, "level_db_spl"] / 10)))

        diff_low = low_B - low_A
        diff_hi = hi_B - hi_A

        if diff_hi > 3 and diff_hi > diff_low:
            diag_tr = "Test dosyasında (B), yüksek frekanslarda (sürtünme/ıslık) referansa (A) göre ciddi artış var."
            diag_en = "In the test file (B), there is a significant increase in high frequencies (friction/whistle) compared to the reference (A)."
        elif diff_low > 3 and diff_low > diff_hi:
            diag_tr = "Test dosyasında (B), düşük frekanslarda (uğultu/titreşim) referansa (A) göre ciddi artış var."
            diag_en = "In the test file (B), there is a significant increase in low frequencies (rumble/vibration) compared to the reference (A)."
        else:
            diag_tr = "Frekans bantlarındaki genel enerji değişimleri orantılı veya birbirine yakındır."
            diag_en = "The overall energy changes in the frequency bands are proportional or close to each other."

        st.info(t(f"🔍 **Kıyaslama Teşhisi:** {diag_tr}", f"🔍 **Comparative Diagnosis:** {diag_en}"))
        report_data["diagnostics"]["1/3 Octave"] = t(diag_tr, diag_en)

    if PDF_ENABLED:
        st.sidebar.markdown("---")
        if st.sidebar.button(t("📄 PDF Raporu Hazırla", "📄 Prepare PDF Report"), use_container_width=True):
            pdf_info_dialog(report_data)

        if st.session_state.get("pdf_ready", False) and "pdf_bytes" in st.session_state:
            st.sidebar.download_button(label=t("📥 PDF Raporunu İndir", "📥 Download PDF Report"), data=st.session_state["pdf_bytes"], file_name="Gates_NVH_Comparative_Report.pdf", mime="application/pdf", use_container_width=True, type="primary")
