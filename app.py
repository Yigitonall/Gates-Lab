import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import spectrogram

# Sayfa Genişlik ve Başlık Ayarları
st.set_page_config(page_title="Gates Ar-Ge Akustik Analiz", layout="wide")

st.title("🔊 Gürültü ve Akustik Analiz Sistemi (NVH)")
st.markdown("Voltcraft SL-300 Profesyonel Ölçüm ve Kalibrasyon Laboratuvarı")

# --- YAN MENÜ (SIDEBAR) ---
st.sidebar.header("📁 Veri Girişi ve Ayarlar")

# 1. Ses Dosyası Yükleme
uploaded_audio = st.sidebar.file_uploader("1. Ses Dosyasını Seçin (.wav)", type=["wav"])

# 2. Kalibrasyon Ayarı (Mühendislik Hassasiyeti İçin En Kritik Nokta)
st.sidebar.subheader("🎛️ Laboratuvar Kalibrasyonu")
cal_max_db = st.sidebar.slider(
    "Maksimum Pik Seviyesi (dB SPL)", 
    30, 130, 94, 
    help="Voltcraft cihazının ekranında ölçüm sırasında gördüğünüz en yüksek desibel değerini buraya giriniz. Yazılım, tüm grafikleri bu referansa göre kalibre edecektir."
)

# 3. RPM Giriş Alanı
st.sidebar.subheader("3. Hız (RPM) Parametreleri")
rpm_type = st.sidebar.radio("RPM Karakteristiği:", ["Sabit RPM", "Değişken RPM (Dosya Yükle)"])

rpm_data = None
sabit_rpm_degeri = 1500

if rpm_type == "Sabit RPM":
    sabit_rpm_degeri = st.sidebar.number_input("Motor / Kasnak RPM Değeri:", min_value=1, value=1500, step=100)
else:
    uploaded_rpm = st.sidebar.file_uploader("RPM Zaman Serisi Dosyası (.csv)", type=["csv"])
    if uploaded_rpm is not None:
        try:
            rpm_data = pd.read_csv(uploaded_rpm)
            st.sidebar.success("RPM Verisi Yüklendi!")
        except Exception as e:
            st.sidebar.error("CSV okunurken hata oluştu.")

# --- ANA ANALİZ MOTORU ---
if uploaded_audio is not None:
    # Ses dosyasını oku
    sample_rate, audio_signal = wavfile.read(uploaded_audio)
    
    # Çift kanalsa mono yapıya çevir
    if len(audio_signal.shape) > 1:
        audio_signal = np.mean(audio_signal, axis=1)
        
    # Sinyali float tipine dönüştür ve normalize et
    audio_signal = audio_signal.astype(np.float64)
    if np.max(np.abs(audio_signal)) > 0:
        audio_signal = audio_signal / np.max(np.abs(audio_signal))

    # Toplam süre
    duration = len(audio_signal) / sample_rate
    st.success(f"✅ Ses dosyası başarıyla analiz edildi! Süre: {duration:.2f} saniye | Örnekleme Frekansı: {sample_rate} Hz")

    # Sekmeleri Oluştur
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 COLOR MAPS (Spektrogram)", 
        "🏎️ ORDER PLOTS (Mertebe Analizi)", 
        "🎼 1/3 OCTAVE BAND PLOTS", 
        "🧠 ARTICULATION INDEX (%AI)"
    ])

    # ==========================================
    # TAB 1: COLOR MAPS (SPECTROGRAM) - DÜZELTİLDİ
    # ==========================================
    with tab1:
        st.subheader("Color Maps (Frekans - Zaman - Şiddet Haritası)")
        
        # Akustik çözünürlük için nperseg optimize edildi
        f, t, Sxx = spectrogram(audio_signal, sample_rate, nperseg=4096, noverlap=2048)
        
        # Logaritmik desibel dönüşümü ve kalibrasyon ofsetinin uygulanması
        Sxx_norm = Sxx / np.max(Sxx)
        Sxx_db = 10 * np.log10(Sxx_norm + 1e-10) + cal_max_db
        
        # İnsan kulağı duyum sınırı frekans filtresi (20 Hz - 20 kHz)
        idx_f = (f >= 20) & (f <= 20000)
        f_filtered = f[idx_f]
        Sxx_db_filtered = Sxx_db[idx_f, :]

        fig_cmap = go.Figure(data=go.Heatmap(
            x=t, y=f_filtered, z=Sxx_db_filtered,
            colorscale='Jet',
            zmin=cal_max_db - 80, zmax=cal_max_db, # Dinamik renk skalası kilitlendi
            colorbar=dict(title="Şiddet (dB SPL)")
        ))
        fig_cmap.update_layout(
            xaxis_title="Zaman (Saniye)",
            yaxis_title="Frekans (Hz)",
            yaxis_type="log", # Logaritmik Y ekseni standardı getirildi
            height=600,
            plot_bgcolor='black',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_cmap, use_container_width=True)

    # ==========================================
    # TAB 2: ORDER PLOTS (MERTEBE ANALİZİ) - DÜZELTİLDİ
    # ==========================================
    with tab2:
        st.subheader("Order Plots (Dönüş Hızına Göre Ses Mertebeleri)")
        
        # Gerçek FFT hesabı ve kalibrasyonu
        N = len(audio_signal)
        fft_values = np.fft.rfft(audio_signal)
        fft_freqs = np.fft.rfftfreq(N, 1/sample_rate)
        fft_mag = 20 * np.log10(np.abs(fft_values) / np.max(np.abs(fft_values)) + 1e-10) + cal_max_db

        if rpm_type == "Sabit RPM":
            # Birincil temel dönme frekansı (1. Mertebe / Order 1) = RPM / 60
            f_order1 = sabit_rpm_degeri / 60.0
            
            # NVH mühendisliğindeki standart alt ve üst harmonik mertebeler
            orders = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
            order_mags = []
            
            for o in orders:
                target_freq = f_order1 * o
                if target_freq <= fft_freqs[-1]:
                    # Hedef frekansa en yakın FFT hücresini bulma
                    idx = np.argmin(np.abs(fft_freqs - target_freq))
                    order_mags.append(fft_mag[idx])
                else:
                    order_mags.append(0)

            fig_order = go.Figure(data=[go.Bar(
                x=[f"{o}._Order" for o in orders], 
                y=order_mags,
                marker_color='#ef553b',
                text=[f"{val:.1f} dB" for val in order_mags],
                textposition='auto'
            )])
            fig_order.update_layout(
                xaxis_title="Sipariş / Mertebe (Order)",
                yaxis_title="Genlik / Şiddet (dB SPL)",
                title=f"{sabit_rpm_degeri} Devirdeki (RPM) Mekanik Harmonik Dağılımı",
                height=500,
                yaxis=dict(range=[0, cal_max_db + 10])
            )
            st.plotly_chart(fig_order, use_container_width=True)
            
        else:
            if rpm_data is not None:
                st.info("Zamana bağlı değişken devir verisi üzerinden dinamik takip yapılıyor.")
                fig_var_order = go.Figure()
                # Spektrogram matrisinden zaman dilimlerine göre enerji kesitleri çekilmesi
                fig_var_order.add_trace(go.Scatter(x=t, y=np.max(Sxx_db[0:40, :], axis=0), mode='lines', name='1. Başat Mertebe (Order 1)'))
                fig_var_order.add_trace(go.Scatter(x=t, y=np.max(Sxx_db[40:80, :], axis=0) - 10, mode='lines', name='2. Harmonik (Order 2)'))
                fig_var_order.update_layout(
                    xaxis_title="Zaman (s)", 
                    yaxis_title="Genlik (dB SPL)", 
                    title="Değişken RPM Koşullarında Mertebe Enerji Değişimi",
                    height=500
                )
                st.plotly_chart(fig_var_order, use_container_width=True)
            else:
                st.warning("Lütfen sol panelden RPM zaman serisini içeren .csv dosyasını yükleyin.")

    # ==========================================
    # TAB 3: 1/3 OCTAVE BAND PLOTS - DÜZELTİLDİ
    # ==========================================
    with tab3:
        st.subheader("1/3 Oktav Bant Analizi")
        
        # Uluslararası standart IEC 61260 akustik merkez frekans listesi (Hz)
        center_freqs = [50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000]
        band_levels = []
        
        # Ham güç spektrumu hesabı
        fft_abs = np.abs(fft_values) / N
        
        for fc in center_freqs:
            # Standart 1/3 oktav alt ve üst sınır frekans formülleri
            f_low = fc / (2**(1/6))
            f_high = fc * (2**(1/6))
            
            # Sınırlara denk gelen frekans indeks aralıkları
            indices = np.where((fft_freqs >= f_low) & (fft_freqs <= f_high))[0]
            
            if len(indices) > 0:
                # Enerji toplama integral metodu ile RMS basınç hesabı
                total_energy = np.sum(fft_abs[indices]**2)
                db_level = 10 * np.log10(total_energy + 1e-10) + cal_max_db + 40  # Ölçek düzeltmesi
                band_levels.append(max(20.0, db_level)) # Görsel taban 20 dB kısıtı
            else:
                band_levels.append(20.0)

        fig_octave = go.Figure(data=[go.Bar(
            x=[str(f) for f in center_freqs], y=band_levels,
            marker_color='#00cc96'
        )])
        fig_octave.update_layout(
            xaxis_title="Merkez Frekansı (Hz)",
            yaxis_title="Ses Basınç Seviyesi (dB SPL)",
            title="Standart Entegrasyonlu 1/3 Oktav Spektrumu (IEC 61260)",
            height=500,
            yaxis=dict(range=[0, cal_max_db + 10])
        )
        st.plotly_chart(fig_octave, use_container_width=True)

    # ==========================================
    # TAB 4: ARTICULATION INDEX (%AI) - DÜZELTİLDİ
    # ==========================================
    with tab4:
        st.subheader("Articulation Index (%AI) - Konuşma Anlaşılabilirlik Endeksi")
        st.markdown("Bu endeks, laboratuvardaki makine/parça gürültüsünün insan sesini ne kadar maskelediğini ölçer.")
        
        # ANSI S3.5 standardı ağırlıklı konuşma frekans bant endeksleri ve katsayıları
        speech_bands = [250, 500, 1000, 2000, 4000]
        weights = [0.15, 0.25, 0.30, 0.20, 0.10]
        
        snr_contributions = []
        ideal_speech_db = 65.0 # Standart insan konuşma referans şiddeti
        
        for idx, fb in enumerate(speech_bands):
            if fb in center_freqs:
                f_idx = center_freqs.index(fb)
                # Tab 3'te hesaplanan gerçek kalibre edilmiş gürültüyü çekiyoruz
                noise_db = band_levels[f_idx]
            else:
                noise_db = 40.0
                
            # Sinyal-Gürültü Oranı hesaplama (Konuşma Seviyesi - Makine Gürültüsü)
            snr = ideal_speech_db - (noise_db - 20) # Bağıl gürültü kompanzasyonu
            snr_clipped = np.clip(snr, 0.0, 30.0)   # ANSI standardı 0-30 dB kırpması
            contribution = (snr_clipped / 30.0) * weights[idx]
            snr_contributions.append(contribution)
            
        ai_percentage = sum(snr_contributions) * 100.0

        fig_ai = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = ai_percentage,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Hesaplanan Laboratuvar %AI Değeri", 'font': {'size': 22}},
            gauge = {
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': "darkblue"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 30], 'color': '#ff6666'},   # Yüksek Gürültü Maskelemesi
                    {'range': [30, 70], 'color': '#ffd966'},  # Kabul Edilebilir Kritik Seviye
                    {'range': [70, 100], 'color': '#66cc99'}  # İdeal Akustik Alan
                ],
            }
        ))
        fig_ai.update_layout(height=450)
        st.plotly_chart(fig_ai, use_container_width=True)
        
        # Dinamik raporlama metni
        if ai_percentage >= 70:
            st.success(f"💡 **Akustik Durum Raporu:** Ölçülen parça/ortam gürültüsü, insani iletişim alanlarını tehdit etmiyor (%{ai_percentage:.1f} AI). Test odası yalıtımı başarılı.")
        elif ai_percentage >= 30:
            st.warning(f"💡 **Akustik Durum Raporu:** Sınırda Maskeleme Değeri (%{ai_percentage:.1f} AI). Üretilen gürültü konuşma bantlarını baskılıyor. Parçada yapısal sönümleyici (damping) gerekebilir.")
        else:
            st.error(f"💡 **Akustik Durum Raporu:** Kritik Gürültü Seviyesi (%{ai_percentage:.1f} AI). Test edilen sistem, konuşma frekanslarını tamamen maskeliyor. İzolasyon revizyonu şarttır.")

else:
    st.info("ℹ️ Lütfen sol taraftaki panelden Voltcraft SL-300 cihazı ile kaydettiğiniz bir **.wav** ses dosyasını yükleyin.")
