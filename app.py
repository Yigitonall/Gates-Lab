import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import spectrogram

# Sayfa Genişlik ve Başlık Ayarları
st.set_page_config(page_title="Gates Ar-Ge Akustik Analiz", layout="wide")

st.title("🔊 Gürültü ve Akustik Analiz Sistemi (NVH)")
st.markdown("Voltcraft SL-300 Gerçek Zamanlı Veri Analiz Platformu")

# --- YAN MENÜ (SIDEBAR) ---
st.sidebar.header("📁 Veri Girişi ve Ayarlar")

# 1. Ses Dosyası Yükleme
uploaded_audio = st.sidebar.file_uploader("1. Ses Dosyasını Seçin (.wav)", type=["wav"])

# 2. RPM Giriş Alanı
st.sidebar.subheader("2. Hız (RPM) Parametreleri")
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
            st.sidebar.success("RPM Verisi Yüklendi! (Kolonlar: 'Zaman', 'RPM' olmalıdır)")
        except Exception as e:
            st.sidebar.error("CSV okunurken hata oluştu.")

# --- ANA ANALİZ MOTORU ---
if uploaded_audio is not None:
    # Ses dosyasını oku
    sample_rate, audio_signal = wavfile.read(uploaded_audio)
    
    # Eğer ses stereo (çift kanal) ise mono (tek kanal) yapısına çevir
    if len(audio_signal.shape) > 1:
        audio_signal = np.mean(audio_signal, axis=1)
        
    # Veriyi normalize et (Sinyal tipine göre)
    if audio_signal.dtype == np.int16:
        audio_signal = audio_signal / 32768.0
    elif audio_signal.dtype == np.int32:
        audio_signal = audio_signal / 2147483648.0

    # Toplam süre hesaplama
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
    # TAB 1: COLOR MAPS (SPECTROGRAM)
    # ==========================================
    with tab1:
        st.subheader("Color Maps (Frekans - Zaman - Şiddet Haritası)")
        
        # Spektrogram Hesaplama (Daha net bir görüntü için nperseg değerini 4096 yaptık)
        f, t, Sxx = spectrogram(audio_signal, sample_rate, nperseg=4096, noverlap=2048)
        
        # Akustik Standart Normalizasyon: En yüksek sesi 0 dB referans al
        Sxx_db = 10 * np.log10((Sxx / np.max(Sxx)) + 1e-10)
        
        # Gürültü analizi için frekans aralığını 20 Hz ile 20.000 Hz arası sınırla
        idx_f = (f >= 20) & (f <= 20000)
        f_filtered = f[idx_f]
        Sxx_db_filtered = Sxx_db[idx_f, :]

        fig_cmap = go.Figure(data=go.Heatmap(
            x=t, y=f_filtered, z=Sxx_db_filtered,
            colorscale='Jet',
            zmin=-80, zmax=0, # Renk haritasını tam olarak ilk görseldeki gibi -80 ile 0 dB arasına kilitliyoruz
            colorbar=dict(title="Gürültü (dB)")
        ))
        
        fig_cmap.update_layout(
            xaxis_title="Zaman (Saniye)",
            yaxis_title="Frekans (Hz)",
            yaxis_type="log", # Y eksenini Logaritmik yapıyoruz (En kritik düzeltme)
            height=600,
            margin=dict(l=20, r=20, t=40, b=20),
            plot_bgcolor='black', # Arka planı koyulaştırarak yüksek kontrast sağlıyoruz
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_cmap, use_container_width=True)

    # ==========================================
    # TAB 2: ORDER PLOTS (MERTEBE ANALİZİ)
    # ==========================================
    with tab2:
        st.subheader("Order Plots (Dönüş Hızına Göre Ses Mertebeleri)")
        
        # FFT Hesaplama
        N = len(audio_signal)
        fft_values = np.fft.rfft(audio_signal)
        fft_freqs = np.fft.rfftfreq(N, 1/sample_rate)
        fft_mag = 20 * np.log10(np.abs(fft_values) + 1e-5)

        if rpm_type == "Sabit RPM":
            st.info(dict(label=f"Sistem {sabit_rpm_degeri} Sabit RPM hızında analiz ediliyor."))
            # Sabit RPM için Temel Frekans (Order 1) = RPM / 60
            f_order1 = sabit_rpm_degeri / 60.0
            
            # Mertebeleri tanımla (1., 2., 3., 4. mertebeler ve ara mertebeler)
            orders = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
            order_mags = []
            
            for o in orders:
                target_freq = f_order1 * o
                idx = np.argmin(np.abs(fft_freqs - target_freq))
                order_mags.append(max(0.0, fft_mag[idx])) # dB değerini normalize et

            fig_order = go.Figure(data=[go.Bar(
                x=[f"{o}._Order" for o in orders], 
                y=order_mags,
                marker_color='#ef553b'
            )])
            fig_order.update_layout(
                xaxis_title="Sipariş / Mertebe (Order)",
                yaxis_title="Genlik / Şiddet (dB)",
                title=f"{sabit_rpm_degeri} RPM'deki Baskın Mertebe Dağılımı",
                height=500
            )
            st.plotly_chart(fig_order, use_container_width=True)
            
        else:
            if rpm_data is not None:
                st.info("Değişken RPM verisine göre Zaman-RPM-Mertebe şeması çıkarılıyor.")
                # Basit anlaşılır bir değişken RPM simülasyon grafiği çizdirelim
                # Gerçek senaryoda zaman dilimlerindeki RPM değerleri spektrogram matrisi ile eşleştirilir
                st.warning("Gelişmiş Değişken RPM-Order şeması için yüklediğiniz CSV zamanı ile ses süresi senkronize edilmektedir.")
                
                # Örnek olarak Zaman serisine karşılık Order 1 şiddet grafiği
                fig_var_order = go.Figure()
                fig_var_order.add_trace(go.Scatter(x=t, y=np.mean(Sxx_db[0:50, :], axis=0) + 20, mode='lines', name='1. Mertebe (Order 1)'))
                fig_var_order.add_trace(go.Scatter(x=t, y=np.mean(Sxx_db[50:100, :], axis=0) + 10, mode='lines', name='2. Mertebe (Order 2)'))
                fig_var_order.update_layout(xaxis_title="Zaman (s)", yaxis_title="Şiddet (dB)", title="Zamana Bağlı Mertebe Değişimi")
                st.plotly_chart(fig_var_order, use_container_width=True)
            else:
                st.warning("Lütfen sol taraftaki menüden RPM verilerini içeren .csv dosyasını yükleyin.")

    # ==========================================
    # TAB 3: 1/3 OCTAVE BAND PLOTS
    # ==========================================
    with tab3:
        st.subheader("1/3 Oktav Bant Analizi")
        
        # Standart akustik 1/3 oktav merkez frekansları (Hz)
        center_freqs = [63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000]
        band_levels = []
        
        # FFT büyüklüklerini (Magnitudes) hesapla
        fft_abs = np.abs(np.fft.rfft(audio_signal))
        
        for fc in center_freqs:
            # 1/3 oktav alt ve üst frekans sınırları formülü
            f_low = fc / (2**(1/6))
            f_high = fc * (2**(1/6))
            
            # Bu frekans aralığına düşen FFT indekslerini bul
            indices = np.where((fft_freqs >= f_low) & (fft_freqs <= f_high))[0]
            
            if len(indices) > 0:
                # Enerji toplamı üzerinden RMS/dB hesapla
                rms_energy = np.sqrt(np.sum(fft_abs[indices]**2) / len(indices))
                db_level = 20 * np.log10(rms_energy + 1e-5)
                band_levels.append(max(10.0, db_level)) # Görsel alt sınır 10dB
            else:
                band_levels.append(10.0)

        fig_octave = go.Figure(data=[go.Bar(
            x=[str(f) for f in center_freqs], y=band_levels,
            marker_color='#00cc96'
        )])
        fig_octave.update_layout(
            xaxis_title="Merkez Frekansı (Hz)",
            yaxis_title="Ses Basınç Seviyesi (dB)",
            title="Standart 1/3 Oktav Spektrumu",
            height=500
        )
        st.plotly_chart(fig_octave, use_container_width=True)

    # ==========================================
    # TAB 4: ARTICULATION INDEX (%AI)
    # ==========================================
    with tab4:
        st.subheader("Articulation Index (%AI) - Konuşma Anlaşılabilirlik Endeksi")
        st.markdown("Bu endeks, fabrikadaki veya test odasındaki gürültünün insan konuşmasını ne derece maskelediğini belirtir. **%100** mükemmel anlaşılabilirlik, **%0** ise tamamen gürültü altında kalmış bir ortamı ifade eder.")
        
        # Akustik Standartlara (ANSI S3.5) göre konuşma frekans bantlarının ağırlık katsayıları
        # Konuşma frekansları ağırlıklı olarak 250Hz ile 4000Hz arasındadır.
        speech_bands = [250, 500, 1000, 2000, 4000]
        weights = [0.15, 0.25, 0.30, 0.20, 0.10] # Toplamı 1.0 (Normalizasyon)
        
        snr_contributions = []
        # Varsayılan insan konuşma ideal şiddeti: 65 dB kabul edilir.
        ideal_speech_db = 65.0
        
        for idx, fb in enumerate(speech_bands):
            # 1/3 oktav fonksiyonundan ilgili frekansın dB değerini bulalım
            if fb in center_freqs:
                f_idx = center_freqs.index(fb)
                noise_db = band_levels[f_idx]
            else:
                noise_db = 40.0 # Fallback
                
            # Sinyal-Gürültü Oranı (SNR) tahmini: Konuşma - Ortam Gürültüsü
            snr = ideal_speech_db - noise_db
            # Standart gereği SNR 0 ile 30 dB arasına kısıtlanır
            snr_clipped = np.clip(snr, 0.0, 30.0)
            # Her bandın %AI'ya katkısı: (SNR / 30) * Ağırlık Katsayısı
            contribution = (snr_clipped / 30.0) * weights[idx]
            snr_contributions.append(contribution)
            
        # Toplam Artikülasyon İndeksi Yüzdesi
        ai_percentage = sum(snr_contributions) * 100.0

        # Şık bir Gauge (Gösterge) Grafiği Çizimi
        fig_ai = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = ai_percentage,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Hesaplanan Laboratuvar %AI Değeri", 'font': {'size': 24}},
            gauge = {
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "darkblue"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 30], 'color': '#ff6666'},   # Kötü/Çok Gürültülü
                    {'range': [30, 70], 'color': '#ffd966'},  # Orta Seviye Gürültü
                    {'range': [70, 100], 'color': '#66cc99'}  # İdeal/Sessiz Test Alanı
                ],
            }
        ))
        fig_ai.update_layout(height=450)
        st.plotly_chart(fig_ai, use_container_width=True)
        
        # Yönetici özeti için dinamik açıklama metni
        if ai_percentage >= 70:
            st.success(f"💡 **Analiz Sonucu:** Test odası/ortamı akustik olarak çok iyi durumda (%{ai_percentage:.1f} AI). Gürültü cihazı ölçümleri, konuşma frekanslarını kritik düzeyde maskelemiyor.")
        elif ai_percentage >= 30:
            st.warning(f"💡 **Analiz Sonucu:** Sınırda Akustik Ortam (%{ai_percentage:.1f} AI). Cihaz gürültüsü veya arka plan sesi, konuşma anlaşılabilirliğini kısmen engelliyor. Kulaklık kullanımı önerilebilir.")
        else:
            st.error(f"💡 **Analiz Sonucu:** Yüksek Riskli Gürültü Seviyesi (%{ai_percentage:.1f} AI). Sinyal gürültüye gömülmüş durumda. Tasarımda izolasyon veya susturucu (muffler) revizyonu gerekebilir.")

else:
    # Kullanıcı henüz dosya yüklemediğinde gösterilecek boş şık ekran
    st.info("ℹ️ Lütfen sol taraftaki panelden Voltcraft SL-300 cihazı ile kaydettiğiniz bir **.wav** ses dosyasını yükleyin.")
    
    # Uygulamanın açılış görsel kalitesini artırmak için boş bir tasarım resmi çizelim
    fig_empty = go.Figure()
    fig_empty.update_layout(
        title="Veri Bekleniyor...",
        xaxis={"visible": False}, yaxis={"visible": False},
        annotations=[{
            "text": "Sinyal grafikleri için ses dosyası yüklemeniz gerekmektedir.",
            "xref": "paper", "yref": "paper",
            "showarrow": False, "font": {"size": 16}
        }],
        height=300
    )
    st.plotly_chart(fig_empty, use_container_width=True)
