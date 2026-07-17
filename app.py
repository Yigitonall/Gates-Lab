import streamlit as st
import numpy as np
import pandas as pd
import scipy.signal as signal
from scipy.io import wavfile
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from fpdf import FPDF
import tempfile
import time
import base64

# Sayfa yapılandırması
st.set_page_config(layout="wide", page_title="Gates R&D NVH Analysis", page_icon="⚙️")

def get_base64_of_bin_file(bin_file):
    """Lokal bir dosyayı okuyup base64 formatına çevirir (Arka plan resmi için)"""
    if os.path.exists(bin_file):
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    return None

if "app_mode" not in st.session_state:
    st.session_state.app_mode = "landing"

# =====================================================================
# 0. ANA MENÜ (LANDING PAGE)
# =====================================================================
if st.session_state.app_mode == "landing":
    # Arka plan resmi kontrolü (Klasörde arka_plan.jpg veya arka_plan.png varsa onu kullanır)
    bg_base64 = get_base64_of_bin_file("arka_plan.jpg") or get_base64_of_bin_file("arka_plan.png")
    
    if bg_base64:
        bg_css = f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{bg_base64}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """
    else:
        bg_css = ""

    # Sadece Ana Menüye özel CSS (Sidebar'ı gizle, üst barı gizle, ortala, butonları şıklaştır, scroll'u kapat)
    landing_css = f"""
    {bg_css}
    <style>
    /* Ana menüde scroll (kaydırma) özelliğini tamamen kapat */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {{
        overflow: hidden !important;
    }}
    
    [data-testid="stSidebar"] {{ display: none !important; }}
    [data-testid="collapsedControl"] {{ display: none !important; }}
    header[data-testid="stHeader"] {{ display: none !important; }}
    
    .block-container {{
        padding-top: 52vh !important; /* Yazıları ve butonları beyaz kısma (aşağı) indir */
        max-width: 1200px !important; /* Başlığın tek satıra sığması için genişliği artırdık */
    }}
    
    div.stButton > button {{
        height: 70px;
        border: 1px solid #ccc;
        background-color: rgba(255, 255, 255, 0.95);
        color: #333;
        border-radius: 8px;
        transition: all 0.3s ease;
    }}
    div.stButton > button p {{
        font-size: 1.1rem;
        font-weight: 500;
    }}
    div.stButton > button:hover {{
        border-color: #E61A25;
        color: #E61A25;
        box-shadow: 0 4px 15px rgba(230, 26, 37, 0.15);
        background-color: white;
    }}
    </style>
    """
    st.markdown(landing_css, unsafe_allow_html=True)

    # Başlık ve Alt Başlık (Tek satır olması için white-space: nowrap ve font-size ayarlandı)
    st.markdown("<h1 style='text-align: center; color: #252525; font-size: 3.2rem; white-space: nowrap; font-family: sans-serif; margin-top: 0px;'>GATES R&D NVH ANALYSIS SYSTEM</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #444; font-size: 1.2rem; margin-bottom: 4rem; font-weight: 400;'>Lütfen yapmak istediğiniz analiz tipini seçin</h3>", unsafe_allow_html=True)

    # Butonlar
    col1, col2, col3, col4 = st.columns([1, 4, 4, 1])
    with col2:
        if st.button("🔍 TEKLİ SES ANALİZİ (Single Analysis)", use_container_width=True):
            st.session_state.app_mode = "single"
            st.rerun()
    with col3:
        if st.button("⚖️ A/B KARŞILAŞTIRMA ANALİZİ (Comparative Analysis)", use_container_width=True):
            st.session_state.app_mode = "compare"
            st.rerun()

else:
    # =====================================================================
    # ORTAK SİDEBAR VE FONKSİYONLAR (ANALİZ MODLARI)
    # =====================================================================
    with st.sidebar:
        if os.path.exists("gates_logo.png"):
            st.image("gates_logo.png", use_container_width=True)
        st.markdown("---")
        if st.button("⬅️ Ana Menüye Dön (Main Menu)", use_container_width=True):
            st.session_state.app_mode = "landing"
            st.rerun()
        st.markdown("---")

    # Sabit parametreler ve ayarlar
    max_display_frequency = 20000 
    sii_bands = [250, 500, 1000, 2000, 4000, 8000]
    sii_weights = {250: 0.06, 500: 0.14, 1000: 0.17, 2000: 0.21, 4000: 0.28, 8000: 0.14}
    
    # Octave bant merkez frekansları (IEC 61260 standardı)
    nominal_bands = [20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000]
    
    def get_exact_band_frequency(nominal):
        if nominal < 1000:
            return 1000 / (10 ** (3/10)) ** round(10 * np.log10(1000/nominal) / 3)
        else:
            return 1000 * (10 ** (3/10)) ** round(10 * np.log10(nominal/1000) / 3)
    
    exact_bands = [get_exact_band_frequency(f) for f in nominal_bands]

    def calculate_sii(spectrum_freqs, spectrum_dbs):
        sii_value = 0.0
        contributions = {}
        for band in sii_bands:
            band_mask = (spectrum_freqs >= band * 0.707) & (spectrum_freqs <= band * 1.414)
            if np.any(band_mask):
                band_db = 10 * np.log10(np.sum(10**(spectrum_dbs[band_mask]/10)))
            else:
                band_db = 0
                
            speech_level = 75 
            snr = speech_level - band_db
            snr = max(-15, min(15, snr)) 
            band_contribution = ((snr + 15) / 30) * sii_weights[band]
            sii_value += band_contribution
            contributions[band] = band_contribution * 100
        return sii_value * 100, contributions

    def generate_sii_diagnosis(sii_value, lang="TR"):
        if sii_value > 75:
            return "İletişim mükemmel (Normal konuşma duyulabilir)." if lang == "TR" else "Communication is excellent (Normal speech is audible)."
        elif sii_value > 45:
            return "İletişim kısmen maskeleniyor." if lang == "TR" else "Communication is partially masked."
        else:
            return "İletişim tamamen yutuluyor (İzolasyon zorunlu)." if lang == "TR" else "Communication is completely masked (Isolation required)."

    def apply_a_weighting(freqs):
        f_sq = freqs**2
        c1 = 12194**2
        c2 = 20.6**2
        c3 = 107.7**2
        c4 = 737.9**2
        R_A = (c1 * f_sq**2) / ((f_sq + c2) * np.sqrt((f_sq + c3) * (f_sq + c4)) * (f_sq + c1))
        A_weight = 20 * np.log10(R_A) + 2.0
        A_weight[freqs < 10] = -70.0
        return A_weight

    class CustomPDF(FPDF):
        def header(self):
            # Antet (Logo, Departman, Rapor Başlığı)
            self.set_line_width(0.5)
            self.rect(10, 10, 190, 25)
            if os.path.exists("gates_logo.png"):
                self.image("gates_logo.png", 12, 12, 35)
            
            self.set_font("Arial", "B", 16)
            self.set_xy(50, 15)
            self.cell(100, 15, "Report", border=0, align="C")
            
            self.set_font("Arial", "B", 9)
            self.set_xy(150, 12)
            self.cell(48, 5, f"Report-No.: {self.report_no}", border=0, align="R")
            
            # Kırmızı Şerit - (X=10.25 tam iç kenar hizası)
            self.set_fill_color(230, 26, 37)
            self.rect(10.25, 30, 189.5, 4.5, "F")
            
            # Antet Bölücü Çizgileri - Department yazısı için x=152'ye kaydırıldı
            self.line(50, 10, 50, 30)
            self.line(152, 10, 152, 30)
            
            # Sağ üst detaylar (Department vb.)
            self.set_font("Arial", "", 8)
            self.set_xy(153, 20)
            self.cell(45, 4, "Department: S.Cankul", border=0, align="L")
            self.set_xy(153, 24)
            self.cell(45, 4, "Test Lab. Engineer", border=0, align="L")

            # Ana Rapor Bilgileri
            self.set_xy(10, 40)
            self.set_font("Arial", "B", 10)
            self.cell(30, 6, "Customer:", border=1)
            self.set_font("Arial", "", 10)
            self.cell(65, 6, getattr(self, 'customer', ''), border=1)
            
            self.set_font("Arial", "B", 10)
            self.cell(30, 6, "Sample No.:", border=1)
            self.set_font("Arial", "", 10)
            self.cell(65, 6, getattr(self, 'sample_no', ''), border=1)
            self.ln()
            
            self.set_font("Arial", "B", 10)
            self.cell(30, 6, "Project:", border=1)
            self.set_font("Arial", "", 10)
            self.cell(65, 6, getattr(self, 'project', ''), border=1)
            
            self.set_font("Arial", "B", 10)
            self.cell(30, 6, "Material:", border=1)
            self.set_font("Arial", "", 10)
            self.cell(65, 6, getattr(self, 'material', ''), border=1)
            self.ln()
            
            self.set_font("Arial", "B", 10)
            self.cell(30, 6, "Technician:", border=1)
            self.set_font("Arial", "", 10)
            self.cell(65, 6, getattr(self, 'technician', ''), border=1)
            
            self.set_font("Arial", "B", 10)
            self.cell(30, 6, "Date:", border=1)
            self.set_font("Arial", "", 10)
            self.cell(65, 6, getattr(self, 'test_date', ''), border=1)
            self.ln(10)

        def footer(self):
            self.set_y(-25)
            self.set_font("Arial", "", 8)
            self.set_line_width(0.5)
            self.rect(10, 272, 190, 15)
            self.line(165, 272, 165, 287)
            
            footer_text = "N:\Engineering\Internal\Working_Folders\18_TEST\03 Test Report Preparation\1) WORD TEST REPORTS\3 CUSTOMER RETURN\E4119\06.07.2026 - 2\E4119 R0010 J-2603055.docx\nThis document was created electronically and is valid without signature."
            self.set_xy(12, 275)
            self.multi_cell(150, 4, footer_text, border=0, align="C")
            
            self.set_xy(165, 277)
            self.cell(35, 5, f"Page: {self.page_no()} of {{nb}}", border=0, align="C")

    def build_pdf_report(report_data, antet_data):
        pdf = CustomPDF()
        pdf.report_no = antet_data['report_no']
        pdf.customer = antet_data['customer']
        pdf.project = antet_data['project']
        pdf.technician = antet_data['technician']
        pdf.sample_no = antet_data['sample_no']
        pdf.material = antet_data['material']
        pdf.test_date = antet_data['test_date']
        
        pdf.alias_nb_pages()
        
        # Karşılaştırma Modu Mantığı
        if st.session_state.app_mode == "compare":
            # Çizilecek figürlerin sırası ve başlıkları (Color Map A ve B ayrıldı, SII Bands hariç tutuldu)
            figures_to_draw = [
                ("Color Map (A - Ref)", "Acoustic Spectrogram - Reference File"),
                ("Color Map (B - Test)", "Acoustic Spectrogram - Test File"),
                ("Order Plot", "Order / Harmonic Analysis"),
                ("SII Gauge", "Articulation Index / SII Analysis"),
                ("1/3 Octave", "1/3 Octave Band Spectrum")
            ]
            
            for fig_key, title in figures_to_draw:
                if fig_key in report_data["figures"]:
                    # Color Map A ve B dışındaki grafikler için veya Color Map A için yeni sayfa aç
                    if fig_key != "Color Map (B - Test)":
                        pdf.add_page()
                    elif fig_key == "Color Map (B - Test)":
                        pdf.add_page() # Color Map B için zorunlu yeni sayfa
                        
                    pdf.set_font("Arial", "B", 14)
                    pdf.cell(0, 10, title, ln=True)
                    
                    fig = report_data["figures"][fig_key]
                    
                    img_height = 400
                    height_pdf = 100
                    # Genişletilmiş / Panoramik Yükseklik Ayarları
                    if fig_key == "SII Gauge":
                        img_height = 300 
                        height_pdf = 75
                    elif fig_key == "1/3 Octave" or fig_key == "Order Plot":
                        img_height = 380 
                        height_pdf = 95
                        
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
                        time.sleep(0.5)
                        fig.write_image(tmp_img.name, format="png", engine="kaleido", width=800, height=img_height, scale=4)
                        pdf.image(tmp_img.name, x=10, w=190, h=height_pdf)
                        tmp_img_path = tmp_img.name
                        
                    # Eğer çizilen grafik SII Gauge ise, hemen altına SII Bands (Çubuk) grafiğini de bas
                    if fig_key == "SII Gauge" and "SII Bands" in report_data["figures"]:
                        fig2 = report_data["figures"]["SII Bands"]
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img2:
                            time.sleep(0.5)
                            fig2.write_image(tmp_img2.name, format="png", engine="kaleido", width=800, height=260, scale=4)
                            pdf.image(tmp_img2.name, x=10, w=190, h=65)
                            tmp_img_path2 = tmp_img2.name
                    
                    pdf.ln(5)
                    
                    # Teşhis ve Karşılaştırma Tablosu (Sadece B dosyası çizildikten sonra veya diğer grafiklerde)
                    diag_key = fig_key.replace(" (A - Ref)", "").replace(" (B - Test)", "")
                    if diag_key in report_data["diagnostics"] and fig_key != "Color Map (A - Ref)":
                        diag = report_data["diagnostics"][diag_key]
                        
                        # Tablo Başlığı (Kırmızı)
                        pdf.set_fill_color(200, 0, 0)
                        pdf.set_text_color(255, 255, 255)
                        pdf.set_font("Arial", "B", 12)
                        pdf.cell(190, 8, "TESHIS / DIAGNOSIS", border=1, align="C", fill=True, ln=True)
                        
                        # A ve B Sütun Başlıkları (Gri)
                        pdf.set_fill_color(230, 230, 230)
                        pdf.set_text_color(0, 0, 0)
                        pdf.set_font("Arial", "B", 10)
                        pdf.cell(95, 6, "A (Reference)", border=1, align="C", fill=True)
                        pdf.cell(95, 6, "B (Test)", border=1, align="C", fill=True, ln=True)
                        
                        # İçerik Hücreleri (Dinamik Yükseklik)
                        pdf.set_font("Arial", "", 10)
                        text_A = diag.get("A", "No diagnosis.")
                        text_B = diag.get("B", "No diagnosis.")
                        
                        # En uzun metne göre satır yüksekliğini belirle
                        lines_A = pdf.get_string_width(text_A) / 90
                        lines_B = pdf.get_string_width(text_B) / 90
                        max_lines = max(int(np.ceil(lines_A)), int(np.ceil(lines_B)), 1)
                        line_height = 6
                        total_height = max_lines * line_height
                        
                        x_start = pdf.get_x()
                        y_start = pdf.get_y()
                        
                        # Yeni sayfaya taşma kontrolü
                        if y_start + total_height + 25 > 270: 
                            pdf.add_page()
                            x_start = pdf.get_x()
                            y_start = pdf.get_y()
                            
                        pdf.multi_cell(95, line_height, text_A, border=1, align="L")
                        pdf.set_xy(x_start + 95, y_start)
                        pdf.multi_cell(95, line_height, text_B, border=1, align="L")
                        
                        # Karşılaştırma Başlığı
                        pdf.set_fill_color(200, 0, 0)
                        pdf.set_text_color(255, 255, 255)
                        pdf.set_font("Arial", "B", 12)
                        pdf.cell(190, 8, "KARSILASTIRMA", border=1, align="C", fill=True, ln=True)
                        
                        # Karşılaştırma İçeriği
                        pdf.set_text_color(0, 0, 0)
                        pdf.set_font("Arial", "", 10)
                        text_diff = diag.get("Diff", "No comparison.")
                        pdf.multi_cell(190, 6, text_diff, border=1, align="L")
        
        else:
            # Tekli Analiz Modu
            for fig_key, fig in report_data["figures"].items():
                if fig_key == "SII Bands": continue # Kopya sayfayı önlemek için atla
                
                pdf.add_page()
                pdf.set_font("Arial", "B", 14)
                pdf.cell(0, 10, f"{fig_key} Analysis", ln=True)
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
                    time.sleep(0.5)
                    fig.write_image(tmp_img.name, format="png", engine="kaleido", width=800, height=400, scale=4)
                    pdf.image(tmp_img.name, x=10, w=190)
                    tmp_img_path = tmp_img.name
                    
                if fig_key == "SII Gauge" and "SII Bands" in report_data["figures"]:
                    fig2 = report_data["figures"]["SII Bands"]
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img2:
                        time.sleep(0.5)
                        fig2.write_image(tmp_img2.name, format="png", engine="kaleido", width=800, height=260, scale=4)
                        pdf.image(tmp_img2.name, x=10, w=190)
                        tmp_img_path2 = tmp_img2.name
                
                pdf.ln(5)
                if fig_key in report_data["diagnostics"]:
                    pdf.set_font("Arial", "B", 12)
                    pdf.cell(0, 10, "Diagnosis:", ln=True)
                    pdf.set_font("Arial", "", 10)
                    pdf.multi_cell(0, 5, report_data["diagnostics"][fig_key].encode('latin-1', 'replace').decode('latin-1'))
                
        # Çıktı dosyasını oluştur        
        temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        pdf.output(temp_pdf.name)
        return temp_pdf.name

    # =====================================================================
    # 1. TEKLİ ANALİZ MODU (SINGLE ANALYSIS)
    # =====================================================================
    if st.session_state.app_mode == "single":
        with st.sidebar:
            st.header("1. Tekli Analiz Verileri")
            uploaded_file = st.file_uploader("1. WAV ses dosyası yükleyin:", type=["wav"])
            max_db_input = st.number_input("Maksimum Pik Seviyesi (MAX SPL) [dB]:", min_value=0.0, max_value=200.0, value=85.0, step=0.1)
            
            rpm_type = st.radio("RPM Tipi:", ["Sabit RPM", "Değişken RPM (CSV)"])
            rpm_fixed = 1500
            rpm_df = None
            if rpm_type == "Sabit RPM":
                rpm_fixed = st.number_input("Sabit Motor Devri (RPM):", min_value=1, max_value=20000, value=1500)
            else:
                rpm_file = st.file_uploader("RPM CSV dosyası yükleyin:", type=["csv"])
                if rpm_file is not None:
                    rpm_df = pd.read_csv(rpm_file)
                    time_col = st.selectbox("Zaman Kolonu:", rpm_df.columns)
                    rpm_col = st.selectbox("RPM Kolonu:", rpm_df.columns)

            st.markdown("---")
            lang = st.radio("Rapor Dili (Language):", ["TR", "EN"])

        if uploaded_file is not None:
            st.success("Ses dosyası başarıyla yüklendi!" if lang=="TR" else "Audio file loaded successfully!")
            
            with st.spinner("Analiz ediliyor... (Processing...)" if lang=="TR" else "Processing..."):
                sample_rate, data = wavfile.read(uploaded_file)
                if len(data.shape) > 1:
                    data = data[:, 0]
                    
                data = data.astype(np.float32)
                data_max = np.max(np.abs(data))
                if data_max == 0: data_max = 1
                
                calib_factor = (10 ** (max_db_input / 20)) / data_max
                data_calibrated = data * calib_factor

                f_welch, Pxx = signal.welch(data_calibrated, sample_rate, nperseg=16384, window='hann')
                f_welch = f_welch[f_welch > 0]
                Pxx = Pxx[f_welch > 0]
                spectrum_db = 10 * np.log10(Pxx / (2e-5)**2)
                
                f_mask = f_welch <= max_display_frequency
                f_welch = f_welch[f_mask]
                spectrum_db = spectrum_db[f_mask]

                f_stft, t_stft, Zxx = signal.stft(data_calibrated, sample_rate, nperseg=4096, window='hann')
                Zxx_db = 10 * np.log10(np.abs(Zxx)**2 / (2e-5)**2)
                Zxx_db = np.clip(Zxx_db, 0, None)
                f_stft_mask = (f_stft > 0) & (f_stft <= max_display_frequency)
                f_stft = f_stft[f_stft_mask]
                Zxx_db = Zxx_db[f_stft_mask, :]

                # 1/3 Octave 
                third_oct_dbs = []
                third_oct_freqs = []
                for band in exact_bands:
                    lower = band / (2 ** (1/6))
                    upper = band * (2 ** (1/6))
                    mask = (f_welch >= lower) & (f_welch <= upper)
                    if np.any(mask):
                        band_db = 10 * np.log10(np.sum(10**(spectrum_db[mask]/10)))
                        third_oct_dbs.append(band_db)
                        third_oct_freqs.append(band)
                
                third_oct_df = pd.DataFrame({
                    "exact_hz": third_oct_freqs,
                    "Band (Hz)": [f"{f/1000}k" if f>=1000 else str(int(f)) for f in nominal_bands[:len(third_oct_freqs)]],
                    "SPL (dB)": third_oct_dbs
                })
                oct_df = third_oct_df[third_oct_df["exact_hz"] <= max_display_frequency].copy()
                
                A_weight = apply_a_weighting(oct_df["exact_hz"].values)
                oct_df["dBA"] = oct_df["SPL (dB)"] + A_weight

            report_data = {"figures": {}, "diagnostics": {}}
            tab1, tab2, tab3, tab4 = st.tabs(["Color Maps", "Order Plots", "Articulation Index (SII)", "1/3 Octave"])
            
            # --- TAB 1: COLOR MAPS ---
            with tab1:
                st.subheader("Color Maps (Acoustic Spectrogram)")
                fig_cm = go.Figure(data=go.Heatmap(
                    z=Zxx_db, x=t_stft, y=f_stft,
                    colorscale='Jet',
                    colorbar=dict(title="Seviye [dB]" if lang=="TR" else "Level [dB]")
                ))
                fig_cm.update_layout(
                    xaxis_title="Zaman [s]" if lang=="TR" else "Time [s]",
                    yaxis_title="Frekans [Hz]" if lang=="TR" else "Frequency [Hz]",
                    yaxis_type="log",
                    height=500,
                    margin=dict(l=50, r=50, t=30, b=50)
                )
                st.plotly_chart(fig_cm, use_container_width=True)
                report_data["figures"]["Color Map"] = fig_cm

            # --- TAB 2: ORDER PLOTS ---
            with tab2:
                st.subheader("Order Plots (Harmonic Analysis)")
                base_freq = rpm_fixed / 60.0
                orders = f_welch / base_freq
                
                fig_op = go.Figure()
                fig_op.add_trace(go.Scatter(x=orders, y=spectrum_db, mode='lines', name='Order', line=dict(color='#E61A25')))
                fig_op.update_layout(
                    xaxis_title="Mertebe (Order)" if lang=="TR" else "Order",
                    yaxis_title="Genlik (dB)" if lang=="TR" else "Amplitude (dB)",
                    xaxis=dict(range=[0, 10]),
                    height=450,
                    margin=dict(l=50, r=50, t=30, b=50)
                )
                st.plotly_chart(fig_op, use_container_width=True)
                
                diag_text = ""
                peak_order = orders[np.argmax(spectrum_db)]
                if 0.8 < peak_order < 1.2:
                    diag_text = "Ana saftta balanssizlik (Unbalance) tespiti." if lang=="TR" else "Main shaft unbalance detected."
                elif 1.8 < peak_order < 2.2:
                    diag_text = "Kaplin/saft eksen kacikligi (Misalignment) tespiti." if lang=="TR" else "Coupling/shaft misalignment detected."
                else:
                    diag_text = f"Baskin mertebe: {peak_order:.1f}x. Spesifik bilesen (rulman/fan) kaynagi." if lang=="TR" else f"Dominant order: {peak_order:.1f}x. Specific component source."
                
                st.info(f"**{'Teşhis' if lang=='TR' else 'Diagnosis'}:** {diag_text}")
                report_data["figures"]["Order Plot"] = fig_op
                report_data["diagnostics"]["Order Plot"] = diag_text

            # --- TAB 3: SII ---
            with tab3:
                st.subheader("Articulation Index / SII Analysis")
                sii_val, sii_contrib = calculate_sii(f_welch, spectrum_db)
                
                fig_sii = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = sii_val,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    gauge = {
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkgray"},
                        'steps': [
                            {'range': [0, 45], 'color': "lightpink"},
                            {'range': [45, 75], 'color': "palegoldenrod"},
                            {'range': [75, 100], 'color': "lightgreen"}
                        ]
                    }
                ))
                fig_sii.update_layout(height=350, margin=dict(t=20, b=20))
                st.plotly_chart(fig_sii, use_container_width=True)
                
                bands_str = [f"{b/1000}k" if b>=1000 else str(b) for b in sii_bands]
                contrib_vals = [sii_contrib[b] for b in sii_bands]
                
                fig_contrib = go.Figure(data=[go.Bar(
                    x=bands_str, y=contrib_vals, 
                    marker_color='#E61A25',
                    marker_line_color='#8B0000',
                    marker_line_width=1.5
                )])
                fig_contrib.update_layout(
                    title="SII Bant Katkıları" if lang=="TR" else "SII Band Contributions",
                    xaxis_type='category',
                    height=250, 
                    margin=dict(t=30, b=20)
                )
                st.plotly_chart(fig_contrib, use_container_width=True)
                
                diag_sii = generate_sii_diagnosis(sii_val, lang)
                st.info(f"**{'Teşhis' if lang=='TR' else 'Diagnosis'}:** {diag_sii}")
                
                report_data["figures"]["SII Gauge"] = fig_sii
                report_data["figures"]["SII Bands"] = fig_contrib
                report_data["diagnostics"]["SII Gauge"] = diag_sii

            # --- TAB 4: 1/3 OCTAVE ---
            with tab4:
                st.subheader("1/3 Octave Band Spectrum")
                fig_oct = go.Figure(data=[go.Bar(
                    x=oct_df["Band (Hz)"], 
                    y=oct_df["SPL (dB)"], 
                    marker_color='#E61A25',
                    marker_line_color='#8B0000',
                    marker_line_width=1
                )])
                fig_oct.update_layout(
                    xaxis_title="Frekans Bandı (Hz)" if lang=="TR" else "Frequency Band (Hz)",
                    yaxis_title="SPL (dB)",
                    xaxis_type='category',
                    height=400,
                    margin=dict(l=50, r=50, t=30, b=50)
                )
                st.plotly_chart(fig_oct, use_container_width=True)
                
                max_band = oct_df.loc[oct_df["SPL (dB)"].idxmax(), "Band (Hz)"]
                if oct_df["SPL (dB)"].idxmax() < len(oct_df)/3:
                    diag_oct = f"Dusuk frekanslarda ({max_band} Hz) yuksek enerji. Ugultu/Titresim sorunu." if lang=="TR" else f"High energy at low frequencies ({max_band} Hz). Hum/Vibration issue."
                else:
                    diag_oct = f"Yuksek frekanslarda ({max_band} Hz) sivriler. Tiz islik/surtunme problemi." if lang=="TR" else f"Spikes at high frequencies ({max_band} Hz). Whistle/Friction issue."
                
                st.info(f"**{'Teşhis' if lang=='TR' else 'Diagnosis'}:** {diag_oct}")
                report_data["figures"]["1/3 Octave"] = fig_oct
                report_data["diagnostics"]["1/3 Octave"] = diag_oct

            # Rapor Oluşturma
            st.markdown("---")
            st.header("📄 PDF Raporu Oluştur (Generate Report)")
            with st.form("antet_form"):
                col1, col2 = st.columns(2)
                with col1:
                    report_no = st.text_input("Report-No.:", "E4119 R0010")
                    customer = st.text_input("Customer:", "Gates Internal")
                    project = st.text_input("Project:", "NVH Assessment")
                with col2:
                    sample_no = st.text_input("Sample No.:", "Sample-01")
                    material = st.text_input("Material:", "EPDM Belt")
                    technician = st.text_input("Technician:", "Test Lab. Eng.")
                    test_date = st.date_input("Date:")
                
                submit_btn = st.form_submit_button("Raporu Hazırla" if lang=="TR" else "Generate Report")
                
                if submit_btn:
                    antet_data = {
                        "report_no": report_no, "customer": customer, "project": project,
                        "technician": technician, "sample_no": sample_no, "material": material,
                        "test_date": str(test_date)
                    }
                    pdf_path = build_pdf_report(report_data, antet_data)
                    with open(pdf_path, "rb") as f:
                        st.download_button("📥 PDF İndir (Download PDF)", data=f, file_name=f"NVH_Report_{report_no}.pdf", mime="application/pdf")

    # =====================================================================
    # 2. KARŞILAŞTIRMA MODU (A/B COMPARATIVE ANALYSIS)
    # =====================================================================
    elif st.session_state.app_mode == "compare":
        with st.sidebar:
            st.header("2. Karşılaştırma Verileri (A/B)")
            file_A = st.file_uploader("Dosya A (Referans):", type=["wav"], key="file_a")
            file_B = st.file_uploader("Dosya B (Test/Arızalı):", type=["wav"], key="file_b")
            
            st.markdown("---")
            max_db_A = st.number_input("MAX SPL - Dosya A [dB]:", value=85.0, step=0.1)
            max_db_B = st.number_input("MAX SPL - Dosya B [dB]:", value=85.0, step=0.1)
            
            rpm_fixed_A = st.number_input("RPM - Dosya A:", value=1500, step=10)
            rpm_fixed_B = st.number_input("RPM - Dosya B:", value=1500, step=10)
            
            st.markdown("---")
            lang = st.radio("Rapor Dili (Language):", ["TR", "EN"], key="lang_cmp")

        if file_A is not None and file_B is not None:
            st.success("Her iki dosya da yüklendi!" if lang=="TR" else "Both files loaded successfully!")
            
            with st.spinner("Karşılaştırma hesaplanıyor... (Processing...)" if lang=="TR" else "Processing Comparison..."):
                # --- SİNYAL İŞLEME FONKSİYONU ---
                def process_signal(file_obj, max_db):
                    sr, data = wavfile.read(file_obj)
                    if len(data.shape) > 1: data = data[:, 0]
                    data = data.astype(np.float32)
                    d_max = np.max(np.abs(data))
                    if d_max == 0: d_max = 1
                    data_cal = data * ((10 ** (max_db / 20)) / d_max)
                    
                    fw, Pxx = signal.welch(data_cal, sr, nperseg=16384, window='hann')
                    fw = fw[fw > 0]
                    Pxx = Pxx[fw > 0]
                    spec_db = 10 * np.log10(Pxx / (2e-5)**2)
                    
                    fstft, tstft, Zxx = signal.stft(data_cal, sr, nperseg=4096, window='hann')
                    Zdb = 10 * np.log10(np.abs(Zxx)**2 / (2e-5)**2)
                    Zdb = np.clip(Zdb, 0, None)
                    return sr, fw, spec_db, fstft, tstft, Zdb

                sr_A, fw_A, spec_A, fstft_A, tstft_A, Zdb_A = process_signal(file_A, max_db_A)
                sr_B, fw_B, spec_B, fstft_B, tstft_B, Zdb_B = process_signal(file_B, max_db_B)

                # Frekans kırpma (20 kHz sınırı)
                mask_A = fw_A <= max_display_frequency
                fw_A, spec_A = fw_A[mask_A], spec_A[mask_A]
                mask_B = fw_B <= max_display_frequency
                fw_B, spec_B = fw_B[mask_B], spec_B[mask_B]

                stft_mask_A = fstft_A <= max_display_frequency
                fstft_A, Zdb_A = fstft_A[stft_mask_A], Zdb_A[stft_mask_A, :]
                stft_mask_B = fstft_B <= max_display_frequency
                fstft_B, Zdb_B = fstft_B[stft_mask_B], Zdb_B[stft_mask_B, :]

                # 1/3 Octave İşlemleri
                def get_octave(fw, spec):
                    t_dbs, t_freqs = [], []
                    for band in exact_bands:
                        l = band / (2**(1/6)); u = band * (2**(1/6))
                        m = (fw >= l) & (fw <= u)
                        if np.any(m):
                            t_dbs.append(10 * np.log10(np.sum(10**(spec[m]/10))))
                            t_freqs.append(band)
                    df = pd.DataFrame({
                        "exact_hz": t_freqs,
                        "Band (Hz)": [f"{f/1000}k" if f>=1000 else str(int(f)) for f in nominal_bands[:len(t_freqs)]],
                        "SPL (dB)": t_dbs
                    })
                    return df

                third_oct_A = get_octave(fw_A, spec_A)
                third_oct_B = get_octave(fw_B, spec_B)
                
                oct_df_A = third_oct_A[third_oct_A["exact_hz"] <= max_display_frequency].copy()
                oct_df_B = third_oct_B[third_oct_B["exact_hz"] <= max_display_frequency].copy()

            report_data = {"figures": {}, "diagnostics": {}}
            tab_cm, tab_op, tab_ai, tab_oc = st.tabs(["Color Maps", "Order Plots", "Articulation Index (SII)", "1/3 Octave"])
            
            # --- TAB 1: COLOR MAPS (A/B) ---
            with tab_cm:
                st.subheader("Acoustic Spectrogram Comparison")
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"**A:** {file_A.name}")
                    fig_cm_A = go.Figure(data=go.Heatmap(z=Zdb_A, x=tstft_A, y=fstft_A, colorscale='Jet', colorbar=dict(title="Seviye [dB]" if lang=="TR" else "Level [dB]")))
                    fig_cm_A.update_layout(xaxis_title="Zaman [s]" if lang=="TR" else "Time [s]", yaxis_title="Frekans [Hz]" if lang=="TR" else "Frequency [Hz]", yaxis_type="log", height=400, margin=dict(l=40, r=40, t=20, b=40))
                    st.plotly_chart(fig_cm_A, use_container_width=True)
                with c2:
                    st.markdown(f"**B:** {file_B.name}")
                    fig_cm_B = go.Figure(data=go.Heatmap(z=Zdb_B, x=tstft_B, y=fstft_B, colorscale='Jet', colorbar=dict(title="Seviye [dB]" if lang=="TR" else "Level [dB]")))
                    fig_cm_B.update_layout(xaxis_title="Zaman [s]" if lang=="TR" else "Time [s]", yaxis_title="Frekans [Hz]" if lang=="TR" else "Frequency [Hz]", yaxis_type="log", height=400, margin=dict(l=40, r=40, t=20, b=40))
                    st.plotly_chart(fig_cm_B, use_container_width=True)
                
                # Spectrogram Akıllı Teşhis Kıyaslaması
                avg_A = np.mean(Zdb_A)
                avg_B = np.mean(Zdb_B)
                diff = avg_B - avg_A
                
                diag_A = f"Ortalama enerji: {avg_A:.1f} dB." if lang=="TR" else f"Average energy: {avg_A:.1f} dB."
                diag_B = f"Ortalama enerji: {avg_B:.1f} dB." if lang=="TR" else f"Average energy: {avg_B:.1f} dB."
                
                if diff > 3:
                    diag_diff = f"B dosyasinda genel spektral gurultu ortalama +{diff:.1f} dB artmistir." if lang=="TR" else f"Overall spectral noise in file B increased by +{diff:.1f} dB."
                elif diff < -3:
                    diag_diff = f"B dosyasinda genel spektral gurultu {abs(diff):.1f} dB azalmistir." if lang=="TR" else f"Overall spectral noise in file B decreased by {abs(diff):.1f} dB."
                else:
                    diag_diff = "Iki dosyanin genel spektral enerji dagilimlari benzerdir." if lang=="TR" else "The overall spectral energy distributions are similar."

                st.info(f"🟦 **A ({file_A.name}):** {diag_A}\n\n🟥 **B ({file_B.name}):** {diag_B}\n\n⚖️ **{'Kıyaslama' if lang=='TR' else 'Comparison'}:** {diag_diff}")

                report_data["figures"]["Color Map (A - Ref)"] = fig_cm_A
                report_data["figures"]["Color Map (B - Test)"] = fig_cm_B
                report_data["diagnostics"]["Color Map"] = {"A": diag_A, "B": diag_B, "Diff": diag_diff}

            # --- TAB 2: ORDER PLOTS (A/B) ---
            with tab_op:
                st.subheader("Order Plots (A/B Overlay)")
                ord_A = fw_A / (rpm_fixed_A / 60.0)
                ord_B = fw_B / (rpm_fixed_B / 60.0)
                
                fig_op = go.Figure()
                fig_op.add_trace(go.Scatter(x=ord_A, y=spec_A, mode='lines', name=f'A: {file_A.name}', line=dict(color='#1f77b4', width=2)))
                fig_op.add_trace(go.Scatter(x=ord_B, y=spec_B, mode='lines', name=f'B: {file_B.name}', line=dict(color='#d62728', width=2)))
                fig_op.update_layout(xaxis_title="Order", yaxis_title="Amplitude (dB)", xaxis=dict(range=[0, 10]), height=450, margin=dict(l=40, r=40, t=20, b=40), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                st.plotly_chart(fig_op, use_container_width=True)
                
                peak_A = ord_A[np.argmax(spec_A)]
                peak_B = ord_B[np.argmax(spec_B)]
                
                diag_A = "Ana saft balanssizligi" if 0.8 < peak_A < 1.2 else "Kaplin/saft eksen kacikligi" if 1.8 < peak_A < 2.2 else f"Baskin mertebe: {peak_A:.1f}x"
                diag_B = "Ana saft balanssizligi" if 0.8 < peak_B < 1.2 else "Kaplin/saft eksen kacikligi" if 1.8 < peak_B < 2.2 else f"Baskin mertebe: {peak_B:.1f}x"
                if lang == "EN":
                    diag_A = "Main shaft unbalance" if 0.8 < peak_A < 1.2 else "Coupling/shaft misalignment" if 1.8 < peak_A < 2.2 else f"Dominant order: {peak_A:.1f}x"
                    diag_B = "Main shaft unbalance" if 0.8 < peak_B < 1.2 else "Coupling/shaft misalignment" if 1.8 < peak_B < 2.2 else f"Dominant order: {peak_B:.1f}x"
                
                diff_val = np.max(spec_B) - np.max(spec_A)
                if diff_val > 0:
                    diag_diff = f"B kaydindaki mekanik titresim (pik noktasi) {diff_val:.1f} dB daha sidetlidir." if lang=="TR" else f"Mechanical vibration (peak) in file B is {diff_val:.1f} dB higher."
                else:
                    diag_diff = f"B kaydindaki mekanik titresim {abs(diff_val):.1f} dB daha zayiftir." if lang=="TR" else f"Mechanical vibration in file B is {abs(diff_val):.1f} dB lower."

                st.info(f"🟦 **A ({file_A.name}):** {diag_A}\n\n🟥 **B ({file_B.name}):** {diag_B}\n\n⚖️ **{'Kıyaslama' if lang=='TR' else 'Comparison'}:** {diag_diff}")

                report_data["figures"]["Order Plot"] = fig_op
                report_data["diagnostics"]["Order Plot"] = {"A": diag_A, "B": diag_B, "Diff": diag_diff}

            # --- TAB 3: SII (A/B) ---
            with tab_ai:
                st.subheader("SII Analysis (A vs B)")
                sii_val_A, sii_contrib_A = calculate_sii(fw_A, spec_A)
                sii_val_B, sii_contrib_B = calculate_sii(fw_B, spec_B)
                
                fig_sii = make_subplots(rows=1, cols=2, specs=[[{'type': 'indicator'}, {'type': 'indicator'}]], subplot_titles=(file_A.name, file_B.name))
                
                fig_sii.add_trace(go.Indicator(mode="gauge+number", value=sii_val_A, gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "darkgray"}, 'steps': [{'range': [0, 45], 'color': "lightpink"}, {'range': [45, 75], 'color': "palegoldenrod"}, {'range': [75, 100], 'color': "lightgreen"}]}), row=1, col=1)
                fig_sii.add_trace(go.Indicator(mode="gauge+number", value=sii_val_B, gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "darkgray"}, 'steps': [{'range': [0, 45], 'color': "lightpink"}, {'range': [45, 75], 'color': "palegoldenrod"}, {'range': [75, 100], 'color': "lightgreen"}]}), row=1, col=2)
                
                fig_sii.update_layout(height=320, margin=dict(t=40, b=20))
                st.plotly_chart(fig_sii, use_container_width=True)
                
                # SII Çubuk Karşılaştırması
                b_str = [f"{b/1000}k" if b>=1000 else str(b) for b in sii_bands]
                c_A = [sii_contrib_A[b] for b in sii_bands]
                c_B = [sii_contrib_B[b] for b in sii_bands]
                
                fig_cb = go.Figure()
                fig_cb.add_trace(go.Bar(x=b_str, y=c_A, name='File A', marker_color='#1f77b4', marker_line_color='#003366', marker_line_width=1))
                fig_cb.add_trace(go.Bar(x=b_str, y=c_B, name='File B', marker_color='#d62728', marker_line_color='#8B0000', marker_line_width=1))
                fig_cb.update_layout(title="SII Katkısı (A vs B)" if lang=="TR" else "SII Contributions (A vs B)", xaxis_type='category', barmode='group', height=280, margin=dict(t=40, b=20))
                st.plotly_chart(fig_cb, use_container_width=True)

                diag_A = generate_sii_diagnosis(sii_val_A, lang)
                diag_B = generate_sii_diagnosis(sii_val_B, lang)
                
                diff_sii = sii_val_B - sii_val_A
                if diff_sii > 5:
                    diag_diff = f"B dosyasinda makine gurultusu azalmis ve iletisim ortami %{diff_sii:.1f} iyilesmis." if lang=="TR" else f"Machine noise decreased in file B, communication environment improved by {diff_sii:.1f}%."
                elif diff_sii < -5:
                    diag_diff = f"B dosyasinda gurultu artisi iletisimi %{abs(diff_sii):.1f} daha kotulestirmis." if lang=="TR" else f"Noise increase in file B worsened communication by {abs(diff_sii):.1f}%."
                else:
                    diag_diff = "Iki durum arasinda belirgin bir iletisim (ergonomi) farki yoktur." if lang=="TR" else "No significant communication (ergonomic) difference between the two states."

                st.info(f"🟦 **A ({file_A.name}) [SII: %{sii_val_A:.1f}]:** {diag_A}\n\n🟥 **B ({file_B.name}) [SII: %{sii_val_B:.1f}]:** {diag_B}\n\n⚖️ **{'Kıyaslama' if lang=='TR' else 'Comparison'}:** {diag_diff}")

                report_data["figures"]["SII Gauge"] = fig_sii
                report_data["figures"]["SII Bands"] = fig_cb
                report_data["diagnostics"]["SII Gauge"] = {"A": diag_A, "B": diag_B, "Diff": diag_diff}

            # --- TAB 4: 1/3 OCTAVE (A/B) ---
            with tab_oc:
                st.subheader("1/3 Octave Band Spectrum (A vs B)")
                fig_oc = go.Figure()
                fig_oc.add_trace(go.Bar(x=oct_df_A["Band (Hz)"], y=oct_df_A["SPL (dB)"], name='File A', marker_color='#1f77b4', marker_line_color='#003366', marker_line_width=1))
                fig_oc.add_trace(go.Bar(x=oct_df_B["Band (Hz)"], y=oct_df_B["SPL (dB)"], name='File B', marker_color='#d62728', marker_line_color='#8B0000', marker_line_width=1))
                fig_oc.update_layout(xaxis_title="Frequency Band (Hz)", yaxis_title="SPL (dB)", xaxis_type='category', barmode='group', height=450, margin=dict(l=40, r=40, t=20, b=40))
                st.plotly_chart(fig_oc, use_container_width=True)
                
                max_A = oct_df_A.loc[oct_df_A["SPL (dB)"].idxmax(), "Band (Hz)"]
                max_B = oct_df_B.loc[oct_df_B["SPL (dB)"].idxmax(), "Band (Hz)"]
                
                diag_A = f"{max_A} Hz bandinda enerji yogunlasmasi." if lang=="TR" else f"Energy concentration at {max_A} Hz band."
                diag_B = f"{max_B} Hz bandinda enerji yogunlasmasi." if lang=="TR" else f"Energy concentration at {max_B} Hz band."
                diag_diff = f"Referans {max_A} Hz'den {max_B} Hz frekansina kayma/degisim gozlemlendi." if max_A != max_B else "Baskin oktav bandi her iki durumda da aynidir."
                if lang == "EN":
                    diag_diff = f"Shift observed from reference {max_A} Hz to {max_B} Hz." if max_A != max_B else "Dominant octave band is identical in both states."

                st.info(f"🟦 **A ({file_A.name}):** {diag_A}\n\n🟥 **B ({file_B.name}):** {diag_B}\n\n⚖️ **{'Kıyaslama' if lang=='TR' else 'Comparison'}:** {diag_diff}")

                report_data["figures"]["1/3 Octave"] = fig_oc
                report_data["diagnostics"]["1/3 Octave"] = {"A": diag_A, "B": diag_B, "Diff": diag_diff}

            # Rapor Oluşturma Formu (Karşılaştırma Modu)
            st.markdown("---")
            st.header("📄 PDF Raporu Oluştur (Generate Report)")
            with st.form("antet_form_cmp"):
                col1, col2 = st.columns(2)
                with col1:
                    report_no = st.text_input("Report-No.:", "E4119 CMP")
                    customer = st.text_input("Customer:", "Gates Internal")
                    project = st.text_input("Project:", "A/B Comparison")
                with col2:
                    sample_no = st.text_input("Sample No.:", "Sample-A vs B")
                    material = st.text_input("Material:", "EPDM Belt")
                    technician = st.text_input("Technician:", "Test Lab. Eng.")
                    test_date = st.date_input("Date:")
                
                submit_btn = st.form_submit_button("Raporu Hazırla" if lang=="TR" else "Generate Report")
                
                if submit_btn:
                    antet_data = {
                        "report_no": report_no, "customer": customer, "project": project,
                        "technician": technician, "sample_no": sample_no, "material": material,
                        "test_date": str(test_date)
                    }
                    pdf_path = build_pdf_report(report_data, antet_data)
                    with open(pdf_path, "rb") as f:
                        st.download_button("📥 PDF İndir (Download PDF)", data=f, file_name=f"NVH_Compare_{report_no}.pdf", mime="application/pdf")
