import io
import math
import traceback
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.io import wavfile
from scipy.signal import spectrogram, welch

# ============================================================
# SAYFA AYARLARI VE GATES KURUMSAL TEMA
# ============================================================
st.set_page_config(
    page_title="Gates R&D NVH Analysis",
    page_icon="🔊",
    layout="wide",
)

# CSS Enjeksiyonu: navigates.gates.com Tasarım Dili
st.markdown("""
<style>
    /* Genel Arka Plan ve Ana Metin Rengi */
    .stApp {
        background-color: #FFFFFF;
        color: #212529;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    
    /* Yan Menü (Sidebar) Teması - Açık Gri */
    [data-testid="stSidebar"] {
        background-color: #F0F0F0;
        border-right: 1px solid #C8C8C8;
    }
    [data-testid="stSidebar"] * {
        color: #212529;
    }
    
    /* Ana Başlıklar - Koyu Antrasit */
    h1, h2, h3, h4 {
        color: #141412 !important;
        font-weight: 700 !important;
    }
    
    /* Sekmeler (Tabs) Teması - Seçili Sekme Gates Kırmızısı */
    .stTabs [data-baseweb="tab-list"] {
        border-bottom: 2px solid #E0E0E0;
    }
    .stTabs [aria-selected="true"] {
        border-bottom-color: #E61A25 !important;
        border-bottom-width: 3px !important;
    }
    .stTabs [aria-selected="true"] p {
        color: #E61A25 !important;
        font-weight: bold;
    }
    
    /* Yükleme Butonları ve Etkileşimli Alanlar */
    div[data-testid="stFileUploader"] > section {
        border-color: #C8C8C8;
        background-color: #F9F9F9;
    }
    
    /* Uyarı ve Bilgi Kutuları */
    div[data-testid="stAlert"] {
        background-color: #F0F0F0;
        color: #212529;
        border-left: 5px solid #E61A25;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# DİL SEÇİMİ VE ÇEVİRİ MOTORU (BILINGUAL SUPPORT)
# ============================================================
# Firma Logosu
try:
    st.sidebar.image("gates_logo.png", use_container_width=True)
except:
    pass

lang_choice = st.sidebar.radio("🌐 Language / Dil", ["Türkçe", "English"], horizontal=True)
lang = "tr" if lang_choice == "Türkçe" else "en"

def t(tr_text: str, en_text: str) -> str:
    """Seçili dile göre metni döndüren çeviri fonksiyonu."""
    return tr_text if lang == "tr" else en_text

st.title(t("🔊 Gürültü ve Akustik Analiz Sistemi (NVH)", "🔊 Noise and Acoustic Analysis System (NVH)"))
st.caption(
    "COLOR MAPS • ORDER PLOTS • ARTICULATION INDEX / SII • 1/3 OCTAVE BAND PLOTS"
)

# ============================================================
# AKUSTİK STANDARTLAR VE SABİTLER
# ============================================================
EPS = np.finfo(float).tiny

THIRD_OCTAVE_NOMINAL = np.array(
    [20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160,
     200, 250, 315, 400, 500, 630, 800, 1000, 1250,
     1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000,
     10000, 12500, 16000, 20000], dtype=float
)

SII_OCTAVE_FREQS = np.array([250, 500, 1000, 2000, 4000, 8000], dtype=float)
SII_BANDWIDTH_ADJUSTMENT = np.array([22.48, 25.48, 28.48, 31.48, 34.48, 37.48], dtype=float)
SII_IMPORTANCE = np.array([0.0617, 0.1671, 0.2373, 0.2648, 0.2142, 0.0549], dtype=float)
SII_INTERNAL_NOISE = np.array([-3.9, -9.7, -12.5, -17.7, -25.9, -7.1], dtype=float)
SII_NORMAL_SPEECH = np.array([34.75, 34.27, 25.01, 17.32, 9.33, 1.13], dtype=float)

# ============================================================
# MÜHENDİSLİK FONKSİYONLARI
# ============================================================
def safe_integrate(y, x):
    if hasattr(np, 'trapezoid'):
        return float(np.trapezoid(y, x))
    else:
        return float(np.trapz(y, x))

def read_wav_mono(uploaded_file) -> Tuple[int, np.ndarray]:
    uploaded_file.seek(0)
    sample_rate, raw = wavfile.read(uploaded_file)
    if raw.ndim == 2: raw = raw.astype(np.float64).mean(axis=1)

    if np.issubdtype(raw.dtype, np.integer):
        if raw.dtype == np.uint8:
            signal = (raw.astype(np.float64) - 128.0) / 128.0
        else:
            info = np.iinfo(raw.dtype)
            scale = float(max(abs(info.min), abs(info.max)))
            signal = raw.astype(np.float64) / scale
    else:
        signal = raw.astype(np.float64)

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
# YAN MENÜ (SIDEBAR)
# ============================================================
st.sidebar.header(t("📁 Veri Girişi ve Ayarlar", "📁 Data Entry & Settings"))

uploaded_audio = st.sidebar.file_uploader(t("1. WAV ses dosyası", "1. WAV audio file"), type=["wav"])

st.sidebar.subheader(t("🎚️ SPL Kalibrasyonu (MAX HOLD)", "🎚️ SPL Calibration (MAX HOLD)"))
reference_leq_db = st.sidebar.number_input(
    t("Maksimum Pik Seviyesi (MAX SPL) [dB]", "Maximum Peak Level (MAX SPL) [dB]"),
    min_value=20.0, max_value=140.0, value=80.0, step=0.1,
    help=t("Voltcraft cihazından okuduğunuz tepe (peak) desibel değerini girin.", "Enter the peak decibel value read from the Voltcraft device.")
)

st.sidebar.subheader(t("🏎️ RPM Bilgisi", "🏎️ RPM Information"))
rpm_mode = st.sidebar.radio(t("RPM tipi", "RPM Type"), [t("Sabit RPM", "Fixed RPM"), t("Değişken RPM (CSV)", "Variable RPM (CSV)")])

fixed_rpm, rpm_dataframe, rpm_column, time_column = None, None, None, None

if rpm_mode in ["Sabit RPM", "Fixed RPM"]:
    fixed_rpm = st.sidebar.number_input(t("Sabit dönüş hızı [RPM]", "Fixed rotation speed [RPM]"), min_value=1.0, value=1500.0, step=10.0)
else:
    uploaded_rpm = st.sidebar.file_uploader(t("RPM zaman serisi (.csv)", "RPM time series (.csv)"), type=["csv"])
    if uploaded_rpm is not None:
        try:
            uploaded_rpm.seek(0)
            rpm_dataframe = pd.read_csv(uploaded_rpm, sep=None, engine="python")
            columns = list(rpm_dataframe.columns)
            if not columns: raise ValueError(t("CSV dosyasında sütun bulunamadı.", "No columns found in CSV."))
            rpm_guess = next((c for c in columns if any(k in c.lower() for k in ["rpm", "devir", "speed"])), columns[-1])
            rpm_column = st.sidebar.selectbox(t("RPM sütunu", "RPM column"), columns, index=columns.index(rpm_guess))
            
            time_dist_opt = t("<Kayıt süresine eşit dağıt>", "<Distribute evenly over recording time>")
            time_options = [time_dist_opt] + columns
            time_guess = next((c for c in columns if any(k in c.lower() for k in ["time", "zaman", "sec"])), None)
            default_time_index = time_options.index(time_guess) if time_guess in time_options else 0
            selected_time = st.sidebar.selectbox(t("Zaman sütunu", "Time column"), time_options, index=default_time_index)
            time_column = None if selected_time == time_dist_opt else selected_time
            st.sidebar.success(t("RPM CSV dosyası okundu.", "RPM CSV successfully read."))
        except Exception as exc:
            st.sidebar.error(t(f"RPM CSV okunamadı: {exc}", f"Failed to read RPM CSV: {exc}"))

# ============================================================
# ANA AKIŞ VE HESAPLAMALAR
# ============================================================
if uploaded_audio is None:
    st.info(t("ℹ️ Lütfen sol panelden bir **.wav** ses dosyası yükleyin.", "ℹ️ Please upload a **.wav** audio file from the left panel."))
    st.stop()

with st.spinner(t("Akustik veriler işleniyor... Lütfen bekleyiniz.", "Processing acoustic data... Please wait.")):
    try:
        sample_rate, audio_signal = read_wav_mono(uploaded_audio)
    except Exception as exc:
        st.error(t(f"WAV dosyası okunamadı: {exc}", f"Failed to read WAV file: {exc}"))
        st.stop()

    duration = audio_signal.size / sample_rate
    nyquist = sample_rate / 2.0

    default_welch = 16384
    default_stft = 4096
    max_display_frequency = min(20000.0, nyquist)

    welch_size = choose_segment_size(audio_signal.size, int(default_welch))
    welch_overlap = welch_size // 2

    psd_frequencies, psd = welch(
        audio_signal, fs=sample_rate, window="hann", nperseg=welch_size,
        noverlap=welch_overlap, detrend="constant", scaling="density", return_onesided=True,
    )

    try:
        calibration_offset_db, raw_recording_level_db = calculate_calibration_offset(
            psd_frequencies, psd, reference_leq_db, max_display_frequency
        )
    except Exception as exc:
        st.error(t(f"SPL kalibrasyonu yapılamadı: {exc}", f"SPL calibration failed: {exc}"))
        st.stop()

    third_octave_df = third_octave_levels(psd_frequencies, psd, calibration_offset_db, nyquist)

st.success(t(f"✅ Analiz Tamamlandı! — Süre: **{duration:.2f} s** | Örnekleme: **{sample_rate:,} Hz**", 
             f"✅ Analysis Complete! — Duration: **{duration:.2f} s** | Sampling: **{sample_rate:,} Hz**"))

tab_color, tab_order, tab_ai, tab_octave = st.tabs([
    "🌈 COLOR MAPS", "🏎️ ORDER PLOTS", "🧠 ARTICULATION INDEX / SII", "🎼 1/3 OCTAVE BAND PLOTS"
])

# ============================================================
# TAB 1 — COLOR MAPS
# ============================================================
with tab_color:
    try:
        st.subheader(t("Color Map — Zaman / Frekans / Seviye", "Color Map — Time / Frequency / Level"))
        stft_size = choose_segment_size(audio_signal.size, int(default_stft))
        stft_overlap = int(stft_size * 0.75)

        spec_f, spec_t, spec_psd = spectrogram(
            audio_signal, fs=sample_rate, window="hann", nperseg=stft_size,
            noverlap=stft_overlap, detrend="constant", scaling="density", mode="psd",
        )
        spec_df = spec_f[1] - spec_f[0] if spec_f.size > 1 else 1.0
        spec_level_db = power_to_db(spec_psd * spec_df, calibration_offset_db)
        frequency_mask = (spec_f >= 20.0) & (spec_f <= max_display_frequency)

        fig_color = go.Figure(go.Heatmap(
            x=spec_t, y=spec_f[frequency_mask], z=spec_level_db[frequency_mask, :],
            colorscale="Turbo", zmin=reference_leq_db - 80.0, zmax=reference_leq_db + 5.0,
            colorbar={"title": t("Seviye<br>[dB]", "Level<br>[dB]")}
        ))

        fig_color.update_layout(
            title=t("Kalibre Edilmiş Akustik Spektrogram (Logaritmik Ölçek)", "Calibrated Acoustic Spectrogram (Logarithmic Scale)"),
            xaxis_title=t("Zaman [s]", "Time [s]"), yaxis_title=t("Frekans [Hz]", "Frequency [Hz]"), yaxis_type="log",
            height=620, margin=dict(l=40, r=30, t=60, b=40),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#212529', family="Arial, sans-serif")
        )
        st.plotly_chart(fig_color, use_container_width=True)
    except Exception as e:
        st.error(t(f"Color Map oluşturulurken bir hata oluştu: {e}", f"Error generating Color Map: {e}"))

# ============================================================
# TAB 2 — ORDER PLOTS
# ============================================================
with tab_order:
    try:
        st.subheader(t("Order Plots — Dönüş Hızıyla İlişkili Ses Mertebeleri", "Order Plots — Rotational Speed-Linked Harmonics"))
        selected_orders = st.multiselect(
            t("Takip edilecek mertebeler", "Orders to follow"), 
            options=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0], default=[0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
        )

        if rpm_mode in ["Sabit RPM", "Fixed RPM"]:
            rotation_frequency = float(fixed_rpm) / 60.0
            max_order_possible = min(30.0, nyquist / rotation_frequency)
            max_order = st.slider(t("Gösterilecek maksimum order", "Maximum order to display"), 1.0, float(max(1.0, max_order_possible)), float(min(10.0, max_order_possible)), 0.5)
            order_step = 0.05
            
            order_df = order_spectrum_from_psd(psd_frequencies, psd, rotation_frequency, max_order, order_step, calibration_offset_db)
            fig_order = go.Figure(go.Scatter(x=order_df["order"], y=order_df["level_db_spl"], mode="lines", name=t("Order spektrumu", "Order spectrum"), line=dict(color="#E61A25", width=2.5)))

            harmonic_x, harmonic_y, harmonic_text = [], [], []
            for so in selected_orders:
                if so > max_order: continue
                hw = max(order_step, 2.0 * (psd_frequencies[1] - psd_frequencies[0]) / rotation_frequency)
                harmonic_power = integrate_psd_band(psd_frequencies, psd, max(0.0, (so - hw) * rotation_frequency), (so + hw) * rotation_frequency)
                if np.isfinite(harmonic_power) and harmonic_power > 0:
                    harmonic_x.append(so)
                    harmonic_y.append(float(power_to_db(harmonic_power, calibration_offset_db)))
                    harmonic_text.append(f"{so:g}×")

            if harmonic_x:
                fig_order.add_trace(go.Scatter(x=harmonic_x, y=harmonic_y, mode="markers+text", text=harmonic_text, textposition="top center", marker={"size": 10, "color": "#212529"}, name=t("Seçili harmonikler", "Selected harmonics")))

            fig_order.update_layout(
                title=f"{t('Order Spektrumu', 'Order Spectrum')} — {float(fixed_rpm):.0f} RPM (1× = {rotation_frequency:.2f} Hz)",
                xaxis_title=t("Mertebe / Order [× dönme frekansı]", "Order [× rotation frequency]"), yaxis_title=t("Bant seviyesi [dB SPL]", "Band level [dB SPL]"),
                height=560, hovermode="x unified", margin=dict(l=40, r=30, t=70, b=45),
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#212529', family="Arial, sans-serif"),
                xaxis=dict(showgrid=True, gridcolor='#E0E0E0'), yaxis=dict(showgrid=True, gridcolor='#E0E0E0')
            )
            st.plotly_chart(fig_order, use_container_width=True)

        else:
            if rpm_dataframe is None or rpm_column is None:
                st.warning(t("Değişken RPM analizi için sol panelden CSV dosyası yükleyin.", "Upload a CSV file from the left panel for variable RPM analysis."))
            elif not selected_orders:
                st.warning(t("En az bir order seçin.", "Please select at least one order."))
            else:
                rpm_time, rpm_values = prepare_rpm_series(rpm_dataframe, rpm_column, time_column, duration)
                track_f, track_t, track_psd = spectrogram(audio_signal, fs=sample_rate, window="hann", nperseg=stft_size, noverlap=int(stft_size * 0.75), detrend="constant", scaling="density", mode="psd")
                rpm_at_stft, order_tracks = calculate_order_tracks(track_f, track_t, track_psd, rpm_time, rpm_values, selected_orders, 0.05, calibration_offset_db)
                binned_tracks = bin_tracks_by_rpm(rpm_at_stft, order_tracks)

                if not binned_tracks.empty:
                    fig_tracking = go.Figure()
                    for so in selected_orders:
                        col = f"order_{so:g}"
                        if col in binned_tracks.columns:
                            fig_tracking.add_trace(go.Scatter(x=binned_tracks["rpm"], y=binned_tracks[col], mode="lines+markers", name=f"{so:g}× Order"))
                    fig_tracking.update_layout(
                        title=t("Order Tracking — RPM'e Göre Mertebe Seviyeleri", "Order Tracking — Harmonic Levels across RPM"),
                        xaxis_title=t("Dönüş hızı [RPM]", "Rotational Speed [RPM]"), yaxis_title=t("Order bant seviyesi [dB SPL]", "Order band level [dB SPL]"),
                        height=580, hovermode="x unified",
                        colorway=["#E61A25", "#212529", "#BF2026", "#495057", "#ADADAD"],
                        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#212529', family="Arial, sans-serif"),
                        xaxis=dict(showgrid=True, gridcolor='#E0E0E0'), yaxis=dict(showgrid=True, gridcolor='#E0E0E0')
                    )
                    st.plotly_chart(fig_tracking, use_container_width=True)
                else:
                    st.error(t("Grafik oluşturmak için devir aralığı yetersiz.", "Insufficient RPM range to generate tracking plot."))
    except Exception as e:
        st.error(t(f"Order analizi yapılırken hata oluştu: {e}", f"Error during Order analysis: {e}"))

# ============================================================
# TAB 3 — ARTICULATION INDEX / SII
# ============================================================
with tab_ai:
    try:
        st.subheader(t("Articulation Index / Speech Intelligibility Index (%)", "Articulation Index / Speech Intelligibility Index (%)"))
        
        speech_efforts_map = {
            t("Normal", "Normal"): np.array([34.75, 34.27, 25.01, 17.32, 9.33, 1.13]),
            t("Yükseltilmiş", "Raised"): np.array([38.98, 40.15, 33.86, 25.32, 16.78, 5.07]),
            t("Yüksek", "Loud"): np.array([41.55, 44.85, 42.16, 34.39, 25.41, 11.39]),
            t("Bağırma", "Shout"): np.array([42.50, 49.24, 51.31, 44.32, 34.41, 20.72]),
        }
        
        vocal_effort = st.selectbox(t("Standart konuşma eforu", "Standard vocal effort"), list(speech_efforts_map.keys()), index=0)
        octave_noise_levels = octave_band_levels(psd_frequencies, psd, calibration_offset_db, SII_OCTAVE_FREQS)

        if np.any(~np.isfinite(octave_noise_levels)):
            st.warning(t("Örnekleme frekansı 8 kHz oktav bandını kapsamıyor; SII sonucu hesaplanamadı.", "Sample rate does not cover the 8 kHz octave band; SII cannot be calculated."))
        else:
            sii_table = compute_octave_sii(octave_noise_levels, speech_efforts_map[vocal_effort])
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

            fig_contribution = go.Figure(go.Bar(
                x=[format_frequency(v) for v in sii_table["frequency_hz"]], y=100.0 * sii_table["contribution"], marker_color="#E61A25"
            ))
            fig_contribution.update_layout(
                title=t("Frekans Bantlarının SII Katkısı", "SII Contribution by Frequency Band"), 
                xaxis_title=t("Oktav merkez frekansı [Hz]", "Octave center frequency [Hz]"), 
                yaxis_title=t("SII katkısı [yüzde puan]", "SII contribution [percentage points]"),
                height=430, bargap=0.12, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#212529', family="Arial, sans-serif"), xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='#E0E0E0')
            )
            st.plotly_chart(fig_contribution, use_container_width=True)

            if sii_percent >= 75: st.success(t(f"İletişim ortamı mükemmel: **%{sii_percent:.1f} SII**.", f"Communication environment is excellent: **{sii_percent:.1f}% SII**."))
            elif sii_percent >= 45: st.warning(t(f"İletişim koşula bağlı: **%{sii_percent:.1f} SII**.", f"Communication is conditional: **{sii_percent:.1f}% SII**."))
            else: st.error(t(f"Gürültü iletişimi güçlü biçimde maskeliyor: **%{sii_percent:.1f} SII**.", f"Noise heavily masks communication: **{sii_percent:.1f}% SII**."))
    except Exception as e:
        st.error(t(f"SII Endeksi hesaplanırken hata oluştu: {e}", f"Error calculating SII Index: {e}"))

# ============================================================
# TAB 4 — 1/3 OCTAVE BAND PLOTS
# ============================================================
with tab_octave:
    try:
        st.subheader(t("1/3 Octave Band Plot", "1/3 Octave Band Plot"))
        octave_plot_df = third_octave_df[third_octave_df["exact_hz"] <= max_display_frequency].copy()
        octave_plot_df["label"] = octave_plot_df["nominal_hz"].map(format_frequency)

        fig_octave = go.Figure(go.Bar(x=octave_plot_df["label"], y=octave_plot_df["level_db_spl"], marker_color="#E61A25"))
        fig_octave.update_layout(
            title=t("IEC 61260-1 Mantığıyla 1/3 Oktav Bant Spektrumu", "1/3 Octave Band Spectrum (IEC 61260-1)"),
            xaxis_title=t("Nominal merkez frekansı [Hz]", "Nominal center frequency [Hz]"), 
            yaxis_title=t("Bant ses basınç seviyesi [dB SPL]", "Band sound pressure level [dB SPL]"),
            height=560, bargap=0.08,
            xaxis={"type": "category", "categoryorder": "array", "categoryarray": octave_plot_df["label"].tolist(), "tickangle": -45, "showgrid": False},
            yaxis=dict(showgrid=True, gridcolor='#E0E0E0'),
            margin=dict(l=40, r=30, t=70, b=90),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#212529', family="Arial, sans-serif")
        )
        st.plotly_chart(fig_octave, use_container_width=True)

        with st.expander(t("1/3 oktav sonuç tablosu", "1/3 octave result table")):
            df_display = octave_plot_df[["nominal_hz", "exact_hz", "lower_hz", "upper_hz", "level_db_spl"]].round(3).copy()
            df_display.columns = [
                t("Nominal merkez [Hz]", "Nominal center [Hz]"),
                t("Exact merkez [Hz]", "Exact center [Hz]"),
                t("Alt sınır [Hz]", "Lower limit [Hz]"),
                t("Üst sınır [Hz]", "Upper limit [Hz]"),
                t("Bant seviyesi [dB SPL]", "Band level [dB SPL]")
            ]
            st.dataframe(df_display, use_container_width=True)
            
    except Exception as e:
        st.error(t(f"1/3 Oktav grafiği çizilirken hata oluştu: {e}", f"Error generating 1/3 Octave plot: {e}"))
