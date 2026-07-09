import io
import math
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.io import wavfile
from scipy.signal import spectrogram, welch


# ============================================================
# SAYFA AYARLARI
# ============================================================
st.set_page_config(
    page_title="Gates Ar-Ge Akustik Analiz",
    page_icon="🔊",
    layout="wide",
)

st.title("🔊 Gürültü ve Akustik Analiz Sistemi (NVH)")
st.caption(
    "COLOR MAPS • ORDER PLOTS • ARTICULATION INDEX / SII • 1/3 OCTAVE BAND PLOTS"
)


# ============================================================
# SABİTLER
# ============================================================
EPS = np.finfo(float).tiny

# IEC 61260-1 taban-10 seri mantığına uygun nominal 1/3 oktav merkezleri.
THIRD_OCTAVE_NOMINAL = np.array(
    [
        20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160,
        200, 250, 315, 400, 500, 630, 800, 1000, 1250,
        1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000,
        10000, 12500, 16000, 20000,
    ],
    dtype=float,
)

# ANSI S3.5-1997 octave-band SII procedure constants.
SII_OCTAVE_FREQS = np.array([250, 500, 1000, 2000, 4000, 8000], dtype=float)
SII_BANDWIDTH_ADJUSTMENT = np.array(
    [22.48, 25.48, 28.48, 31.48, 34.48, 37.48], dtype=float
)
SII_IMPORTANCE = np.array(
    [0.0617, 0.1671, 0.2373, 0.2648, 0.2142, 0.0549], dtype=float
)
SII_INTERNAL_NOISE = np.array(
    [-3.9, -9.7, -12.5, -17.7, -25.9, -7.1], dtype=float
)
SII_NORMAL_SPEECH = np.array(
    [34.75, 34.27, 25.01, 17.32, 9.33, 1.13], dtype=float
)
SII_SPEECH_SPECTRA = {
    "Normal": np.array([34.75, 34.27, 25.01, 17.32, 9.33, 1.13]),
    "Raised / Yükseltilmiş": np.array([38.98, 40.15, 33.86, 25.32, 16.78, 5.07]),
    "Loud / Yüksek": np.array([41.55, 44.85, 42.16, 34.39, 25.41, 11.39]),
    "Shout / Bağırma": np.array([42.50, 49.24, 51.31, 44.32, 34.41, 20.72]),
}


# ============================================================
# YARDIMCI FONKSİYONLAR
# ============================================================
def read_wav_mono(uploaded_file) -> Tuple[int, np.ndarray]:
    """WAV dosyasını oku, mono ve [-1, 1] float sinyale dönüştür."""
    uploaded_file.seek(0)
    sample_rate, raw = wavfile.read(uploaded_file)

    if raw.ndim == 2:
        raw = raw.astype(np.float64).mean(axis=1)

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

    if signal.size < 256:
        raise ValueError("Ses dosyası analiz için çok kısa.")

    if not np.any(np.abs(signal) > 0):
        raise ValueError("Ses dosyasında ölçülebilir bir sinyal bulunamadı.")

    return int(sample_rate), signal


def largest_power_of_two_at_most(value: int) -> int:
    if value < 2:
        return 1
    return 1 << int(math.floor(math.log2(value)))


def choose_segment_size(signal_length: int, requested: int) -> int:
    return max(256, min(int(requested), largest_power_of_two_at_most(signal_length)))


def power_to_db(power: np.ndarray | float, calibration_offset_db: float) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(power, EPS)) + calibration_offset_db


def integrate_psd_band(
    frequencies: np.ndarray,
    psd: np.ndarray,
    lower_hz: float,
    upper_hz: float,
) -> float:
    """
    PSD'yi frekans bandında integre ederek toplam ortalama-kare gücü verir.
    Kenar noktaları doğrusal enterpolasyonla eklenir.
    """
    lower_hz = max(float(lower_hz), float(frequencies[0]))
    upper_hz = min(float(upper_hz), float(frequencies[-1]))

    if upper_hz <= lower_hz:
        return np.nan

    inside = (frequencies > lower_hz) & (frequencies < upper_hz)
    band_f = np.concatenate(
        ([lower_hz], frequencies[inside], [upper_hz])
    )
    band_p = np.concatenate(
        (
            [np.interp(lower_hz, frequencies, psd)],
            psd[inside],
            [np.interp(upper_hz, frequencies, psd)],
        )
    )

    if band_f.size < 2:
        return np.nan

    return float(np.trapezoid(band_p, band_f))


def calculate_calibration_offset(
    frequencies: np.ndarray,
    psd: np.ndarray,
    reference_leq_db: float,
    max_frequency_hz: float,
) -> Tuple[float, float]:
    """
    WAV kaydının 20 Hz–üst sınır arasındaki toplam gücünü, kullanıcı tarafından
    girilen aynı-kayıt Leq referansına eşler.
    """
    upper = min(float(max_frequency_hz), float(frequencies[-1]))
    recording_power = integrate_psd_band(frequencies, psd, 20.0, upper)

    if not np.isfinite(recording_power) or recording_power <= 0:
        raise ValueError("Kalibrasyon için yeterli spektral enerji bulunamadı.")

    raw_level_db = 10.0 * np.log10(recording_power)
    offset_db = float(reference_leq_db) - raw_level_db
    return offset_db, raw_level_db


def exact_third_octave_centers() -> np.ndarray:
    """
    1000 Hz referanslı, taban-10 exact 1/3 oktav merkez frekansları.
    k=-17...13, yaklaşık 20 Hz...20 kHz.
    """
    k = np.arange(-17, 14, dtype=float)
    return 1000.0 * np.power(10.0, k / 10.0)


def third_octave_levels(
    frequencies: np.ndarray,
    psd: np.ndarray,
    calibration_offset_db: float,
    nyquist_hz: float,
) -> pd.DataFrame:
    exact_centers = exact_third_octave_centers()
    g = 10.0 ** (3.0 / 10.0)
    edge_factor = g ** (1.0 / 6.0)

    rows = []
    for nominal, exact in zip(THIRD_OCTAVE_NOMINAL, exact_centers):
        lower = exact / edge_factor
        upper = exact * edge_factor

        if lower < frequencies[0] or upper > nyquist_hz:
            continue

        band_power = integrate_psd_band(frequencies, psd, lower, upper)
        level = (
            float(power_to_db(band_power, calibration_offset_db))
            if np.isfinite(band_power) and band_power > 0
            else np.nan
        )

        rows.append(
            {
                "nominal_hz": nominal,
                "exact_hz": exact,
                "lower_hz": lower,
                "upper_hz": upper,
                "level_db_spl": level,
            }
        )

    return pd.DataFrame(rows)


def format_frequency(value: float) -> str:
    if value >= 1000:
        k = value / 1000.0
        if abs(k - round(k)) < 1e-9:
            return f"{int(round(k))}k"
        return f"{k:g}k"
    return f"{value:g}"


def octave_band_levels(
    frequencies: np.ndarray,
    psd: np.ndarray,
    calibration_offset_db: float,
    centers_hz: Sequence[float],
) -> np.ndarray:
    levels = []
    edge = math.sqrt(2.0)

    for center in centers_hz:
        lower = center / edge
        upper = center * edge

        # Eksik oktav bandını kısmi bant gibi kullanma.
        if lower < frequencies[0] or upper > frequencies[-1]:
            levels.append(np.nan)
            continue

        power = integrate_psd_band(frequencies, psd, lower, upper)
        if np.isfinite(power) and power > 0:
            levels.append(float(power_to_db(power, calibration_offset_db)))
        else:
            levels.append(np.nan)

    return np.asarray(levels, dtype=float)


def compute_octave_sii(
    octave_noise_band_levels_db: np.ndarray,
    vocal_effort: str,
) -> pd.DataFrame:
    """
    ANSI S3.5-1997 octave-band SII procedure.
    Varsayımlar: normal işitme eşiği, iletim kaybı yok, seçilen standart konuşma spektrumu.
    """
    speech = SII_SPEECH_SPECTRA[vocal_effort]

    # Ölçülen oktav bant toplam seviyesini spektrum seviyesine çevir.
    noise_spectrum = octave_noise_band_levels_db - SII_BANDWIDTH_ADJUSTMENT

    equivalent_masking = noise_spectrum
    equivalent_internal_noise = SII_INTERNAL_NOISE
    disturbance = np.maximum(equivalent_masking, equivalent_internal_noise)

    level_distortion = 1.0 - (
        speech - SII_NORMAL_SPEECH - 10.0
    ) / 160.0
    level_distortion = np.clip(level_distortion, 0.0, 1.0)

    audibility_k = (speech - disturbance + 15.0) / 30.0
    audibility_k = np.clip(audibility_k, 0.0, 1.0)

    band_audibility = level_distortion * audibility_k
    contribution = SII_IMPORTANCE * band_audibility

    return pd.DataFrame(
        {
            "frequency_hz": SII_OCTAVE_FREQS,
            "noise_band_db_spl": octave_noise_band_levels_db,
            "noise_spectrum_db": noise_spectrum,
            "speech_spectrum_db": speech,
            "disturbance_db": disturbance,
            "importance": SII_IMPORTANCE,
            "audibility": band_audibility,
            "contribution": contribution,
        }
    )


def order_spectrum_from_psd(
    frequencies: np.ndarray,
    psd: np.ndarray,
    rotation_frequency_hz: float,
    max_order: float,
    order_step: float,
    calibration_offset_db: float,
) -> pd.DataFrame:
    """
    PSD'yi eşit genişlikli order kutularına integre eder.
    Böylece x ekseni sürekli Order, y ekseni her order kutusunun toplam SPL seviyesidir.
    """
    edges = np.arange(0.0, max_order + order_step, order_step)
    centers = (edges[:-1] + edges[1:]) / 2.0

    levels = []
    for lower_order, upper_order in zip(edges[:-1], edges[1:]):
        lower_hz = lower_order * rotation_frequency_hz
        upper_hz = upper_order * rotation_frequency_hz
        band_power = integrate_psd_band(
            frequencies, psd, lower_hz, upper_hz
        )
        if np.isfinite(band_power) and band_power > 0:
            levels.append(float(power_to_db(band_power, calibration_offset_db)))
        else:
            levels.append(np.nan)

    return pd.DataFrame({"order": centers, "level_db_spl": levels})


def parse_numeric_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    cleaned = (
        series.astype(str)
        .str.strip()
        .str.replace(" ", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def prepare_rpm_series(
    rpm_dataframe: pd.DataFrame,
    rpm_column: str,
    time_column: Optional[str],
    duration_seconds: float,
) -> Tuple[np.ndarray, np.ndarray]:
    rpm = parse_numeric_series(rpm_dataframe[rpm_column])

    if time_column is None:
        time_values = pd.Series(
            np.linspace(0.0, duration_seconds, len(rpm_dataframe))
        )
    else:
        time_values = parse_numeric_series(rpm_dataframe[time_column])

        # Sütun adı milisaniyeyi işaret ediyorsa saniyeye çevir.
        lower_name = time_column.lower()
        if "ms" in lower_name or "millisecond" in lower_name:
            time_values = time_values / 1000.0

    valid = rpm.notna() & time_values.notna() & (rpm > 0)
    rpm_values = rpm[valid].to_numpy(dtype=float)
    time_values_np = time_values[valid].to_numpy(dtype=float)

    if rpm_values.size < 2:
        raise ValueError("RPM CSV dosyasında en az iki geçerli RPM noktası olmalıdır.")

    order = np.argsort(time_values_np)
    time_values_np = time_values_np[order]
    rpm_values = rpm_values[order]

    # Aynı zaman damgalarını ortalama RPM ile birleştir.
    grouped = (
        pd.DataFrame({"time": time_values_np, "rpm": rpm_values})
        .groupby("time", as_index=False)["rpm"]
        .mean()
    )

    return (
        grouped["time"].to_numpy(dtype=float),
        grouped["rpm"].to_numpy(dtype=float),
    )


def integrate_column_band(
    frequencies: np.ndarray,
    psd_column: np.ndarray,
    lower_hz: float,
    upper_hz: float,
) -> float:
    return integrate_psd_band(frequencies, psd_column, lower_hz, upper_hz)


def calculate_order_tracks(
    stft_frequencies: np.ndarray,
    stft_times: np.ndarray,
    stft_psd: np.ndarray,
    rpm_time: np.ndarray,
    rpm_values: np.ndarray,
    orders: Sequence[float],
    half_width_order: float,
    calibration_offset_db: float,
) -> Tuple[np.ndarray, dict[float, np.ndarray]]:
    rpm_at_stft = np.interp(
        stft_times,
        rpm_time,
        rpm_values,
        left=np.nan,
        right=np.nan,
    )

    tracks: dict[float, np.ndarray] = {}

    for selected_order in orders:
        levels = np.full(stft_times.shape, np.nan, dtype=float)

        for index, rpm_now in enumerate(rpm_at_stft):
            if not np.isfinite(rpm_now) or rpm_now <= 0:
                continue

            rotation_hz = rpm_now / 60.0
            lower_hz = max(
                0.0, (selected_order - half_width_order) * rotation_hz
            )
            upper_hz = (
                selected_order + half_width_order
            ) * rotation_hz

            band_power = integrate_column_band(
                stft_frequencies,
                stft_psd[:, index],
                lower_hz,
                upper_hz,
            )

            if np.isfinite(band_power) and band_power > 0:
                levels[index] = float(
                    power_to_db(band_power, calibration_offset_db)
                )

        tracks[float(selected_order)] = levels

    return rpm_at_stft, tracks


def bin_tracks_by_rpm(
    rpm_values: np.ndarray,
    tracks: dict[float, np.ndarray],
    requested_bins: int = 50,
) -> pd.DataFrame:
    valid_rpm = rpm_values[np.isfinite(rpm_values)]
    if valid_rpm.size < 2:
        return pd.DataFrame()

    rpm_min = float(np.nanmin(valid_rpm))
    rpm_max = float(np.nanmax(valid_rpm))

    if rpm_max <= rpm_min:
        return pd.DataFrame()

    bin_count = max(8, min(int(requested_bins), valid_rpm.size))
    edges = np.linspace(rpm_min, rpm_max, bin_count + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    indices = np.digitize(rpm_values, edges) - 1

    output = {"rpm": centers}

    for selected_order, levels in tracks.items():
        binned = np.full(bin_count, np.nan, dtype=float)
        for bin_index in range(bin_count):
            mask = (
                (indices == bin_index)
                & np.isfinite(rpm_values)
                & np.isfinite(levels)
            )
            if np.any(mask):
                binned[bin_index] = float(np.nanmedian(levels[mask]))
        output[f"order_{selected_order:g}"] = binned

    result = pd.DataFrame(output)
    value_columns = [column for column in result.columns if column != "rpm"]
    return result.dropna(subset=value_columns, how="all")


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.header("📁 Veri Girişi ve Analiz Ayarları")

uploaded_audio = st.sidebar.file_uploader(
    "1. WAV ses dosyası",
    type=["wav"],
)

st.sidebar.subheader("🎚️ SPL Kalibrasyonu (MAX HOLD)")
reference_leq_db = st.sidebar.number_input(
    "Maksimum Pik Seviyesi (MAX SPL) [dB]",
    min_value=20.0,
    max_value=140.0,
    value=80.0,
    step=0.1,
    help=(
        "Test sırasında Voltcraft cihazındaki 'MAX/MIN' tuşunu kullanarak "
        "ekranda sabitlediğiniz en yüksek desibel değerini buraya girin."
    ),
)
st.sidebar.caption(
    "Yazılım, dosya içindeki en şiddetli tepe noktasını bu değere eşitleyerek "
    "tüm analiz grafiklerini gerçek laboratuvar ölçeğine kalibre edecektir."
)

st.sidebar.subheader("🏎️ RPM Bilgisi")
rpm_mode = st.sidebar.radio(
    "RPM tipi",
    ["Sabit RPM", "Değişken RPM (CSV)"],
)

fixed_rpm = None
rpm_dataframe = None
rpm_column = None
time_column = None

if rpm_mode == "Sabit RPM":
    fixed_rpm = st.sidebar.number_input(
        "Sabit dönüş hızı [RPM]",
        min_value=1.0,
        value=1500.0,
        step=10.0,
    )
else:
    uploaded_rpm = st.sidebar.file_uploader(
        "RPM zaman serisi (.csv)",
        type=["csv"],
    )

    if uploaded_rpm is not None:
        try:
            uploaded_rpm.seek(0)
            rpm_dataframe = pd.read_csv(
                uploaded_rpm,
                sep=None,
                engine="python",
            )
            columns = list(rpm_dataframe.columns)

            if not columns:
                raise ValueError("CSV dosyasında sütun bulunamadı.")

            rpm_guess = next(
                (
                    column
                    for column in columns
                    if "rpm" in column.lower()
                    or "devir" in column.lower()
                    or "speed" in column.lower()
                ),
                columns[-1],
            )

            rpm_column = st.sidebar.selectbox(
                "RPM sütunu",
                columns,
                index=columns.index(rpm_guess),
            )

            time_options = ["<Kayıt süresine eşit dağıt>"] + columns
            time_guess = next(
                (
                    column
                    for column in columns
                    if "time" in column.lower()
                    or "zaman" in column.lower()
                    or "sec" in column.lower()
                ),
                None,
            )
            default_time_index = (
                time_options.index(time_guess)
                if time_guess in time_options
                else 0
            )

            selected_time = st.sidebar.selectbox(
                "Zaman sütunu",
                time_options,
                index=default_time_index,
            )
            time_column = (
                None
                if selected_time == "<Kayıt süresine eşit dağıt>"
                else selected_time
            )

            st.sidebar.success("RPM CSV dosyası okundu.")
        except Exception as exc:
            st.sidebar.error(f"RPM CSV okunamadı: {exc}")
            rpm_dataframe = None


# ============================================================
# ANA UYGULAMA
# ============================================================
if uploaded_audio is None:
    st.info(
        "Sol panelden bir **WAV ses dosyası** yükleyin. Değişken devirli order "
        "tracking için ayrıca zaman–RPM CSV dosyası gerekir."
    )
    st.stop()


try:
    sample_rate, audio_signal = read_wav_mono(uploaded_audio)
except Exception as exc:
    st.error(f"WAV dosyası okunamadı: {exc}")
    st.stop()

duration = audio_signal.size / sample_rate
nyquist = sample_rate / 2.0

available_fft_sizes = [
    value
    for value in [1024, 2048, 4096, 8192, 16384, 32768, 65536]
    if value <= largest_power_of_two_at_most(audio_signal.size)
]

if not available_fft_sizes:
    available_fft_sizes = [choose_segment_size(audio_signal.size, 1024)]

default_welch = min(16384, max(available_fft_sizes))
default_stft = min(4096, max(available_fft_sizes))

# Sinyal işleme parametrelerini arayüzden kaldırdık, arka planda ideal standartlara sabitledik.
welch_size = default_welch
stft_size = default_stft

max_display_frequency = st.sidebar.number_input(
    "Grafik üst frekans sınırı [Hz]",
    min_value=100.0,
    max_value=float(max(100.0, nyquist)),
    value=float(min(20000.0, nyquist)),
    step=100.0,
)

# Welch PSD: 1/3 oktav, sabit RPM order spektrumu ve SII için ortak temel.
welch_size = choose_segment_size(audio_signal.size, int(welch_size))
welch_overlap = welch_size // 2

psd_frequencies, psd = welch(
    audio_signal,
    fs=sample_rate,
    window="hann",
    nperseg=welch_size,
    noverlap=welch_overlap,
    detrend="constant",
    scaling="density",
    return_onesided=True,
)

try:
    calibration_offset_db, raw_recording_level_db = calculate_calibration_offset(
        psd_frequencies,
        psd,
        reference_leq_db,
        min(20000.0, nyquist),
    )
except Exception as exc:
    st.error(f"SPL kalibrasyonu yapılamadı: {exc}")
    st.stop()

third_octave_df = third_octave_levels(
    psd_frequencies,
    psd,
    calibration_offset_db,
    nyquist,
)

st.success(
    f"Ses dosyası hazır — Süre: **{duration:.2f} s** | "
    f"Örnekleme: **{sample_rate:,} Hz** | "
    f"Frekans çözünürlüğü: **{psd_frequencies[1] - psd_frequencies[0]:.3f} Hz**"
)

with st.expander("Kalibrasyon notu", expanded=False):
    st.write(
        "Bu yazılım WAV kaydının 20 Hz–seçilen üst frekans arasındaki toplam "
        "spektral gücünü, girdiğiniz referans Leq seviyesine eşler. Bu yaklaşım "
        "ölçüm zincirinin frekans cevabını düzeltmez; mikrofon ve kayıt cihazı "
        "kalibrasyonu ayrıca doğrulanmalıdır."
    )
    st.write(
        f"Hesaplanan dijital kayıt seviyesi: **{raw_recording_level_db:.2f} dB** | "
        f"Uygulanan SPL ofseti: **{calibration_offset_db:+.2f} dB**"
    )


tab_color, tab_order, tab_ai, tab_octave = st.tabs(
    [
        "🌈 COLOR MAPS",
        "🏎️ ORDER PLOTS",
        "🧠 ARTICULATION INDEX / SII",
        "🎼 1/3 OCTAVE BAND PLOTS",
    ]
)


# ============================================================
# TAB 1 — COLOR MAPS
# ============================================================
with tab_color:
    st.subheader("Color Map — Zaman / Frekans / Seviye")

    stft_size = choose_segment_size(audio_signal.size, int(stft_size))
    stft_overlap = int(stft_size * 0.75)

    spec_f, spec_t, spec_psd = spectrogram(
        audio_signal,
        fs=sample_rate,
        window="hann",
        nperseg=stft_size,
        noverlap=stft_overlap,
        detrend="constant",
        scaling="density",
        mode="psd",
    )

    spec_df = spec_f[1] - spec_f[0] if spec_f.size > 1 else 1.0

    # Her renk hücresi, ilgili FFT hücresinin bant gücü olarak gösterilir.
    spec_level_db = power_to_db(
        spec_psd * spec_df,
        calibration_offset_db,
    )

    frequency_mask = (
        (spec_f >= 20.0)
        & (spec_f <= min(max_display_frequency, nyquist))
    )

    axis_scale = st.radio(
        "Frekans ekseni",
        ["Logaritmik", "Doğrusal"],
        horizontal=True,
        key="color_axis_scale",
    )

    fig_color = go.Figure(
        go.Heatmap(
            x=spec_t,
            y=spec_f[frequency_mask],
            z=spec_level_db[frequency_mask, :],
            colorscale="Turbo",
            zmin=reference_leq_db - 80.0,
            zmax=reference_leq_db + 5.0,
            colorbar={"title": "Seviye<br>[dB SPL/bin]"},
            hovertemplate=(
                "Zaman: %{x:.3f} s<br>"
                "Frekans: %{y:.1f} Hz<br>"
                "Seviye: %{z:.1f} dB SPL/bin<extra></extra>"
            ),
        )
    )

    fig_color.update_layout(
        title="Kalibre Edilmiş Akustik Spektrogram",
        xaxis_title="Zaman [s]",
        yaxis_title="Frekans [Hz]",
        yaxis_type="log" if axis_scale == "Logaritmik" else "linear",
        height=620,
        margin=dict(l=40, r=30, t=60, b=40),
    )

    st.plotly_chart(fig_color, use_container_width=True)
    st.caption(
        "Color map gerçek bir spektrogramdır: x=zaman, y=frekans, renk=dar bant SPL seviyesi."
    )


# ============================================================
# TAB 2 — ORDER PLOTS
# ============================================================
with tab_order:
    st.subheader("Order Plots — Dönüş Hızıyla İlişkili Ses Mertebeleri")

    selected_orders = st.multiselect(
        "Takip edilecek mertebeler",
        options=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0],
        default=[0.5, 1.0, 2.0, 3.0, 4.0, 5.0],
    )

    if rpm_mode == "Sabit RPM":
        rotation_frequency = float(fixed_rpm) / 60.0

        max_order_possible = min(
            30.0,
            nyquist / rotation_frequency,
        )
        max_order = st.slider(
            "Gösterilecek maksimum order",
            min_value=1.0,
            max_value=float(max(1.0, max_order_possible)),
            value=float(min(10.0, max_order_possible)),
            step=0.5,
        )

        order_step = st.select_slider(
            "Order çözünürlüğü",
            options=[0.01, 0.02, 0.025, 0.05, 0.1],
            value=0.05,
        )

        order_df = order_spectrum_from_psd(
            psd_frequencies,
            psd,
            rotation_frequency,
            max_order,
            order_step,
            calibration_offset_db,
        )

        fig_order = go.Figure()

        fig_order.add_trace(
            go.Scatter(
                x=order_df["order"],
                y=order_df["level_db_spl"],
                mode="lines",
                name="Order spektrumu",
                hovertemplate=(
                    "Order: %{x:.3f}×<br>"
                    "Seviye: %{y:.1f} dB SPL<extra></extra>"
                ),
            )
        )

        harmonic_x = []
        harmonic_y = []
        harmonic_text = []

        for selected_order in selected_orders:
            if selected_order > max_order:
                continue

            half_width = max(
                order_step,
                2.0
                * (psd_frequencies[1] - psd_frequencies[0])
                / rotation_frequency,
            )
            lower_hz = max(
                0.0, (selected_order - half_width) * rotation_frequency
            )
            upper_hz = (
                selected_order + half_width
            ) * rotation_frequency

            harmonic_power = integrate_psd_band(
                psd_frequencies,
                psd,
                lower_hz,
                upper_hz,
            )

            if np.isfinite(harmonic_power) and harmonic_power > 0:
                harmonic_level = float(
                    power_to_db(harmonic_power, calibration_offset_db)
                )
                harmonic_x.append(selected_order)
                harmonic_y.append(harmonic_level)
                harmonic_text.append(f"{selected_order:g}×")

        if harmonic_x:
            fig_order.add_trace(
                go.Scatter(
                    x=harmonic_x,
                    y=harmonic_y,
                    mode="markers+text",
                    text=harmonic_text,
                    textposition="top center",
                    marker={"size": 10},
                    name="Seçili harmonikler",
                    hovertemplate=(
                        "Order: %{x:g}×<br>"
                        "Entegre seviye: %{y:.1f} dB SPL<extra></extra>"
                    ),
                )
            )

        fig_order.update_layout(
            title=(
                f"Order Spectrum — {float(fixed_rpm):.0f} RPM "
                f"(1× = {rotation_frequency:.2f} Hz)"
            ),
            xaxis_title="Mertebe / Order [× dönme frekansı]",
            yaxis_title="Bant seviyesi [dB SPL]",
            height=560,
            hovermode="x unified",
            margin=dict(l=40, r=30, t=70, b=45),
        )

        st.plotly_chart(fig_order, use_container_width=True)
        st.caption(
            "Sabit RPM kaydında doğru çıktı bir **order spectrum**dur. "
            "X ekseni order, Y ekseni ilgili order kutusuna integre edilmiş SPL seviyesidir."
        )

    else:
        if rpm_dataframe is None or rpm_column is None:
            st.warning(
                "Değişken RPM order tracking için sol panelden RPM CSV dosyası yükleyin."
            )
        elif not selected_orders:
            st.warning("En az bir order seçin.")
        else:
            try:
                rpm_time, rpm_values = prepare_rpm_series(
                    rpm_dataframe,
                    rpm_column,
                    time_column,
                    duration,
                )

                order_half_width = st.select_slider(
                    "Order takip yarı bant genişliği [±order]",
                    options=[0.02, 0.025, 0.05, 0.075, 0.1, 0.15],
                    value=0.05,
                )

                track_f, track_t, track_psd = spectrogram(
                    audio_signal,
                    fs=sample_rate,
                    window="hann",
                    nperseg=stft_size,
                    noverlap=stft_overlap,
                    detrend="constant",
                    scaling="density",
                    mode="psd",
                )

                rpm_at_stft, order_tracks = calculate_order_tracks(
                    track_f,
                    track_t,
                    track_psd,
                    rpm_time,
                    rpm_values,
                    selected_orders,
                    order_half_width,
                    calibration_offset_db,
                )

                binned_tracks = bin_tracks_by_rpm(
                    rpm_at_stft,
                    order_tracks,
                    requested_bins=50,
                )

                if binned_tracks.empty:
                    st.error(
                        "RPM aralığı order tracking grafiği oluşturmak için yetersiz."
                    )
                else:
                    fig_tracking = go.Figure()

                    for selected_order in selected_orders:
                        column = f"order_{selected_order:g}"
                        if column not in binned_tracks.columns:
                            continue

                        fig_tracking.add_trace(
                            go.Scatter(
                                x=binned_tracks["rpm"],
                                y=binned_tracks[column],
                                mode="lines+markers",
                                name=f"{selected_order:g}× Order",
                                hovertemplate=(
                                    "RPM: %{x:.0f}<br>"
                                    "Seviye: %{y:.1f} dB SPL<extra></extra>"
                                ),
                            )
                        )

                    fig_tracking.update_layout(
                        title="Order Tracking — RPM'e Göre Mertebe Seviyeleri",
                        xaxis_title="Dönüş hızı [RPM]",
                        yaxis_title="Order bant seviyesi [dB SPL]",
                        height=580,
                        hovermode="x unified",
                        margin=dict(l=40, r=30, t=70, b=45),
                    )

                    st.plotly_chart(fig_tracking, use_container_width=True)
                    st.caption(
                        "Bu grafik gerçek anlamda **dönüş hızına göre order plot**tur. "
                        "Her eğri seçilen bir mertebenin RPM boyunca değişimini gösterir."
                    )

            except Exception as exc:
                st.error(f"Order tracking hesaplanamadı: {exc}")


# ============================================================
# TAB 3 — ARTICULATION INDEX / SII
# ============================================================
with tab_ai:
    st.subheader("Articulation Index / Speech Intelligibility Index (%)")

    st.info(
        "ANSI S3.5-1997 ile eski Articulation Index yaklaşımı Speech "
        "Intelligibility Index (SII) olarak yeniden tanımlanmıştır. "
        "Aşağıdaki hesap octave-band SII prosedürünü kullanır."
    )

    vocal_effort = st.selectbox(
        "Standart konuşma eforu",
        list(SII_SPEECH_SPECTRA.keys()),
        index=0,
    )

    octave_noise_levels = octave_band_levels(
        psd_frequencies,
        psd,
        calibration_offset_db,
        SII_OCTAVE_FREQS,
    )

    if np.any(~np.isfinite(octave_noise_levels)):
        st.warning(
            "Örnekleme frekansı 8 kHz oktav bandının tamamını kapsamıyor; "
            "SII sonucu hesaplanamadı."
        )
    else:
        sii_table = compute_octave_sii(
            octave_noise_levels,
            vocal_effort,
        )
        sii_value = float(sii_table["contribution"].sum())
        sii_percent = 100.0 * sii_value

        fig_sii = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=sii_percent,
                number={"suffix": " %", "valueformat": ".1f"},
                title={"text": "Octave-band SII / %AI"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "steps": [
                        {"range": [0, 30], "color": "#f8d7da"},
                        {"range": [30, 70], "color": "#fff3cd"},
                        {"range": [70, 100], "color": "#d1e7dd"},
                    ],
                    "bar": {"color": "#1f4e79"},
                },
            )
        )
        fig_sii.update_layout(height=430)
        st.plotly_chart(fig_sii, use_container_width=True)

        contribution_percent = (
            100.0 * sii_table["contribution"]
        )

        fig_contribution = go.Figure(
            go.Bar(
                x=[
                    format_frequency(value)
                    for value in sii_table["frequency_hz"]
                ],
                y=contribution_percent,
                customdata=np.column_stack(
                    [
                        sii_table["noise_band_db_spl"],
                        sii_table["audibility"],
                        sii_table["importance"],
                    ]
                ),
                hovertemplate=(
                    "Oktav merkezi: %{x} Hz<br>"
                    "SII katkısı: %{y:.2f} puan<br>"
                    "Gürültü: %{customdata[0]:.1f} dB SPL<br>"
                    "İşitilebilirlik: %{customdata[1]:.3f}<br>"
                    "Önem katsayısı: %{customdata[2]:.4f}<extra></extra>"
                ),
            )
        )
        fig_contribution.update_layout(
            title="Frekans Bantlarının SII Katkısı",
            xaxis_title="Oktav merkez frekansı [Hz]",
            yaxis_title="SII katkısı [yüzde puan]",
            height=430,
            bargap=0.12,
        )
        st.plotly_chart(fig_contribution, use_container_width=True)

        with st.expander("SII hesap tablosu"):
            display_table = sii_table.copy()
            numeric_columns = display_table.select_dtypes(
                include=[np.number]
            ).columns
            display_table[numeric_columns] = display_table[
                numeric_columns
            ].round(4)
            st.dataframe(display_table, use_container_width=True)

        if sii_percent >= 75:
            st.success(
                f"Konuşma işitilebilirliği yüksek görünüyor: **%{sii_percent:.1f} SII**."
            )
        elif sii_percent >= 45:
            st.warning(
                f"Konuşma işitilebilirliği orta / koşula bağlı: **%{sii_percent:.1f} SII**."
            )
        else:
            st.error(
                f"Gürültü konuşma bantlarını güçlü biçimde maskeliyor: "
                f"**%{sii_percent:.1f} SII**."
            )

        st.caption(
            "Bu sonuç; normal işitme, iletim kaybı olmaması ve seçilen standart "
            "konuşma spektrumu varsayımlarıyla hesaplanır. Reverberasyon ve gerçek "
            "konuşma yolu ayrıca ölçülmediyse laboratuvar doğrulamasının yerini tutmaz."
        )


# ============================================================
# TAB 4 — 1/3 OCTAVE BAND PLOTS
# ============================================================
with tab_octave:
    st.subheader("1/3 Octave Band Plot")

    octave_plot_df = third_octave_df.copy()
    octave_plot_df = octave_plot_df[
        octave_plot_df["exact_hz"]
        <= min(max_display_frequency, nyquist)
    ]

    octave_plot_df["label"] = octave_plot_df["nominal_hz"].map(
        format_frequency
    )

    fig_octave = go.Figure(
        go.Bar(
            x=octave_plot_df["label"],
            y=octave_plot_df["level_db_spl"],
            customdata=np.column_stack(
                [
                    octave_plot_df["exact_hz"],
                    octave_plot_df["lower_hz"],
                    octave_plot_df["upper_hz"],
                ]
            ),
            hovertemplate=(
                "Nominal merkez: %{x} Hz<br>"
                "Exact merkez: %{customdata[0]:.2f} Hz<br>"
                "Bant: %{customdata[1]:.2f}–%{customdata[2]:.2f} Hz<br>"
                "Seviye: %{y:.1f} dB SPL<extra></extra>"
            ),
        )
    )

    fig_octave.update_layout(
        title="IEC 61260-1 Mantığıyla 1/3 Oktav Bant Spektrumu",
        xaxis_title="Nominal merkez frekansı [Hz]",
        yaxis_title="Bant ses basınç seviyesi [dB SPL]",
        height=560,
        bargap=0.08,
        xaxis={
            "type": "category",
            "categoryorder": "array",
            "categoryarray": octave_plot_df["label"].tolist(),
            "tickangle": -45,
        },
        margin=dict(l=40, r=30, t=70, b=90),
    )

    st.plotly_chart(fig_octave, use_container_width=True)

    st.caption(
        "Merkez frekansları kategorik olarak eşit aralıklı gösterilir; "
        "her sütun ilgili 1/3 oktav bandına integre edilmiş toplam SPL seviyesidir."
    )

    with st.expander("1/3 oktav sonuç tablosu"):
        result_table = octave_plot_df[
            [
                "nominal_hz",
                "exact_hz",
                "lower_hz",
                "upper_hz",
                "level_db_spl",
            ]
        ].copy()
        result_table.columns = [
            "Nominal merkez [Hz]",
            "Exact merkez [Hz]",
            "Alt sınır [Hz]",
            "Üst sınır [Hz]",
            "Bant seviyesi [dB SPL]",
        ]
        st.dataframe(result_table.round(3), use_container_width=True)

        csv_bytes = result_table.to_csv(
            index=False
        ).encode("utf-8-sig")
        st.download_button(
            "1/3 oktav sonuçlarını CSV indir",
            data=csv_bytes,
            file_name="one_third_octave_results.csv",
            mime="text/csv",
        )
