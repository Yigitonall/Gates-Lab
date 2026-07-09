import streamlit as st
import plotly.graph_objects as go
import numpy as np

# Sayfa ayarları - Geniş ekran ve başlık
st.set_page_config(page_title="Ar-Ge Akustik Analiz", layout="wide")

st.title("Gürültü ve Akustik Analiz Sistemi")
st.markdown("Voltcraft SL-300 verileri için geliştirilmiş interaktif analiz aracıdır.")

# --- YAN MENÜ (SIDEBAR) VERİ YÜKLEME ALANI ---
st.sidebar.header("Veri Yükleme ve Ayarlar")

# 1. Ses Dosyası Yükleme
uploaded_audio = st.sidebar.file_uploader("1. Ses Dosyasını Yükle (.wav)", type=["wav"])

# 2. RPM Ayarları
st.sidebar.subheader("2. RPM Ayarları")
rpm_type = st.sidebar.radio("RPM Tipi Seçiniz:", ["Sabit RPM", "Değişken RPM (Dosya Yükle)"])

if rpm_type == "Sabit RPM":
    rpm_value = st.sidebar.number_input("Sabit RPM Değerini Giriniz:", min_value=0, value=1500, step=100)
else:
    uploaded_rpm = st.sidebar.file_uploader("RPM Verisini Yükle (.csv, .txt)", type=["csv", "txt"])

# --- ANA EKRAN (GRAFİKLER) ---
st.write("### Analiz Grafikleri")

# Sekmeler (Tabs) ile modern ve kurumsal bir görünüm
tab1, tab2, tab3, tab4 = st.tabs(["Color Maps", "Order Plots", "1/3 Octave Band", "Articulation Index (%AI)"])

# Dummy (Örnek) Veri Üretimi (Cihaz gelene kadar boş durmaması için)
x_dummy = np.linspace(0, 10, 100)
y_dummy = np.sin(x_dummy)

with tab1:
    st.subheader("Color Maps (Spectrogram)")
    st.info("Cihazdan .wav verisi geldiğinde burası gerçek frekans/zaman haritası olacak.")
    # Örnek Plotly Heatmap
    fig1 = go.Figure(data=go.Heatmap(
        z=[[1, 20, 30], [20, 1, 60], [30, 60, 1]],
        colorscale='Viridis'))
    fig1.update_layout(title="Örnek Color Map (Test Verisi)")
    st.plotly_chart(fig1, use_container_width=True)

with tab2:
    st.subheader("Order Plots")
    st.info("Sistemin dönüş hızına (RPM) göre titreşim/ses mertebeleri burada gösterilecek.")
    fig2 = go.Figure(data=go.Scatter(x=x_dummy, y=y_dummy, mode='lines', name='Order 1'))
    fig2.update_layout(title="Örnek Order Plot", xaxis_title="Zaman / RPM", yaxis_title="Genlik")
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.subheader("1/3 Octave Band Plots")
    st.info("Standart frekans bantlarındaki ses şiddeti çubuk grafik olarak gösterilecek.")
    bands = ['63', '125', '250', '500', '1k', '2k', '4k', '8k']
    levels = [40, 45, 55, 60, 50, 45, 35, 30]
    fig3 = go.Figure(data=[go.Bar(x=bands, y=levels, marker_color='#1f77b4')])
    fig3.update_layout(title="Örnek 1/3 Octave Band", xaxis_title="Frekans (Hz)", yaxis_title="Ses Şiddeti (dB)")
    st.plotly_chart(fig3, use_container_width=True)

with tab4:
    st.subheader("Articulation Index (%AI)")
    st.info("Ortamdaki konuşma anlaşılabilirliği indeksi burada hesaplanacak.")
    fig4 = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = 75,
        title = {'text': "Hesaplanan %AI"},
        gauge = {'axis': {'range': [None, 100]},
                 'bar': {'color': "darkblue"},
                 'steps': [
                     {'range': [0, 40], 'color': "red"},
                     {'range': [40, 70], 'color': "orange"},
                     {'range': [70, 100], 'color': "green"}]}
    ))
    st.plotly_chart(fig4, use_container_width=True)
