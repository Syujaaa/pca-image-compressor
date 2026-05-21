import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.metrics import structural_similarity as calculate_ssim

st.set_page_config(layout="wide", initial_sidebar_state="expanded")

# Custom CSS untuk styling
st.markdown("""
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #f0f3f7 0%, #e8ecf1 100%);
            color: #2c3e50;
        }
        
        .main {
            background-color: #ffffff;
            border-radius: 12px;
            padding: 2rem 1.5rem;
        }
        
        h1 {
            background: linear-gradient(135deg, #1e3a5f 0%, #2C5AA0 100%);
            color: white;
            padding: 2.5rem 2rem;
            border-radius: 12px;
            margin-bottom: 2rem;
            font-size: 2.4rem;
            font-weight: 700;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.15);
            box-shadow: 0 4px 15px rgba(30, 58, 95, 0.25);
            letter-spacing: -0.5px;
        }
        
        h2 {
            color: #1e3a5f;
            border-bottom: 4px solid #2C5AA0;
            padding-bottom: 1rem;
            margin-top: 2.5rem;
            margin-bottom: 1.5rem;
            font-size: 1.7rem;
            font-weight: 600;
            letter-spacing: -0.3px;
        }
        
        h3 {
            color: #2C5AA0;
            margin-top: 2rem;
            margin-bottom: 1.2rem;
            font-size: 1.35rem;
            font-weight: 600;
        }
        
        .metric-card {
            background: linear-gradient(135deg, #2C5AA0 0%, #1e3a5f 100%);
            color: white;
            padding: 1.8rem 1.5rem;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 2px 8px rgba(30, 58, 95, 0.15);
            margin-bottom: 1rem;
            transition: all 0.3s ease;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .metric-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 6px 16px rgba(30, 58, 95, 0.25);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        .metric-label {
            font-size: 0.9rem;
            opacity: 0.95;
            margin-bottom: 0.6rem;
            font-weight: 500;
            letter-spacing: 0.3px;
            text-transform: uppercase;
        }
        
        .metric-value {
            font-size: 2rem;
            font-weight: 700;
            line-height: 1;
        }
        
        .info-box {
            background: linear-gradient(135deg, #fef5e7 0%, #fef9e7 100%);
            color: #7d5608;
            padding: 1.5rem;
            border-radius: 10px;
            margin: 1.5rem 0;
            box-shadow: 0 2px 8px rgba(243, 156, 18, 0.15);
            border-left: 5px solid #f39c12;
            border: 1px solid rgba(243, 156, 18, 0.25);
            line-height: 1.8;
        }
        
        .info-box-blue {
            background: linear-gradient(135deg, #eef6fb 0%, #e8f4f8 100%);
            color: #1e3a5f;
            padding: 1.5rem;
            border-radius: 10px;
            margin: 1.5rem 0;
            box-shadow: 0 2px 8px rgba(44, 90, 160, 0.12);
            border-left: 5px solid #2C5AA0;
            border: 1px solid rgba(44, 90, 160, 0.25);
        }
        
        .info-box-green {
            background: linear-gradient(135deg, #eafaf1 0%, #d5f4e6 100%);
            color: #0b5345;
            padding: 1.5rem;
            border-radius: 10px;
            margin: 1.5rem 0;
            box-shadow: 0 2px 8px rgba(22, 160, 133, 0.12);
            border-left: 5px solid #16a085;
            border: 1px solid rgba(22, 160, 133, 0.25);
        }
        
        .interpretation-text {
            background: linear-gradient(135deg, #f8f9fa 0%, #f5f7fa 100%);
            padding: 1.2rem;
            border-radius: 8px;
            border-left: 4px solid #2C5AA0;
            margin: 1.2rem 0;
            color: #2c3e50;
            font-size: 0.97rem;
            line-height: 1.7;
            box-shadow: 0 1px 4px rgba(30, 58, 95, 0.08);
        }
        
        .table-container {
            overflow-x: auto;
            margin: 1.5rem 0;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            background-color: white;
            font-size: 0.95rem;
        }
        
        th {
            background: linear-gradient(135deg, #1e3a5f 0%, #2C5AA0 100%);
            color: white;
            padding: 1.1rem;
            font-weight: 600;
            text-align: left;
            letter-spacing: 0.3px;
        }
        
        td {
            padding: 0.9rem 1.1rem;
            border-bottom: 1px solid #ecf0f1;
            color: #34495e;
        }
        
        tr:hover {
            background-color: #f8f9fa;
            transition: background-color 0.2s ease;
        }
        
        .file-uploader-section {
            background: linear-gradient(135deg, #eef6fb 0%, #e8f4f8 100%);
            padding: 2.5rem;
            border-radius: 12px;
            margin-bottom: 2rem;
            text-align: center;
            box-shadow: 0 2px 10px rgba(44, 90, 160, 0.12);
            border: 2px dashed #2C5AA0;
        }
        
        .step-indicator {
            display: inline-block;
            background: #2C5AA0;
            color: white;
            padding: 0.4rem 0.8rem;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 600;
            margin-right: 0.8rem;
            letter-spacing: 0.3px;
        }
        
        @media (max-width: 768px) {
            h1 {
                font-size: 1.7rem;
                padding: 1.5rem 1.2rem;
                margin-bottom: 1.5rem;
            }
            
            h2 {
                font-size: 1.4rem;
                margin-top: 2rem;
            }
            
            .main {
                padding: 1.2rem;
            }
            
            .metric-value {
                font-size: 1.5rem;
            }
            
            .metric-card {
                padding: 1.3rem;
            }
            
            table {
                font-size: 0.85rem;
            }
            
            th, td {
                padding: 0.7rem 0.9rem;
            }
        }
        
        @media (max-width: 480px) {
            h1 {
                font-size: 1.4rem;
                padding: 1.2rem 1rem;
            }
            
            h2 {
                font-size: 1.2rem;
            }
            
            .metric-card {
                padding: 1rem;
            }
            
            .metric-label {
                font-size: 0.75rem;
            }
            
            .metric-value {
                font-size: 1.3rem;
            }
            
            .file-uploader-section {
                padding: 1.5rem;
            }
            
            .interpretation-text {
                padding: 0.9rem;
                font-size: 0.9rem;
            }
        }
    </style>
""", unsafe_allow_html=True)

st.markdown(
    """
    <div style='background: linear-gradient(135deg, #1e3a5f 0%, #2C5AA0 100%); 
                color: white; padding: 2.5rem 2rem; border-radius: 12px; 
                margin-bottom: 2rem; text-align: center;
                box-shadow: 0 4px 15px rgba(30, 58, 95, 0.25); 
                border: 1px solid rgba(255, 255, 255, 0.1);'>
        <h1 style='font-size: 2.4rem; font-weight: 700; text-shadow: 1px 1px 2px rgba(0,0,0,0.15); 
                   margin: 0; letter-spacing: -0.5px; margin-bottom: 0.5rem;'>
            🎯 PROGRAM KOMPRESI CITRA PCA & EDA
        </h1>
        <p style='margin: 0; opacity: 0.95; font-size: 1rem; letter-spacing: 0.3px;'>
            Teknik Pengurangan Dimensi dengan Eigenvalue dan Eigenvector
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    "<p style='font-size: 1.05rem; color: #555; line-height: 1.6; margin-bottom: 2rem;'>"
    "Aplikasi ini mengimplementasikan kompresi citra berbasis <b>Principal Component Analysis (PCA)</b> "
    "menggunakan Eigenvalue dan Eigenvector, dilengkapi dengan <b>Exploratory Data Analysis (EDA)</b> "
    "sebelum, saat, dan setelah kompresi."
    "</p>", 
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader("📤 Upload gambar (JPG / PNG / JPEG)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    # ---------------------------------------------------------------
    # TAHAP 1 & 3: Membaca Citra & Mengubah Menjadi Matriks
    # ---------------------------------------------------------------
    gambar = Image.open(uploaded_file).convert("L")
    matriks_gambar = np.array(gambar, dtype=float)
    tinggi, lebar = matriks_gambar.shape

    # ---------------------------------------------------------------
    # TAHAP 2: EDA Awal pada Citra
    # ---------------------------------------------------------------
    st.header("📊 EDA Awal pada Citra")

    nilai_min = np.min(matriks_gambar)
    nilai_max = np.max(matriks_gambar)
    nilai_mean = np.mean(matriks_gambar)
    nilai_std = np.std(matriks_gambar)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Ukuran Citra</div>
                <div class='metric-value'>{tinggi}×{lebar}</div>
                <div style='font-size: 0.85rem; opacity: 0.9;'>px</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Rentang Piksel</div>
                <div class='metric-value'>{int(nilai_min)}–{int(nilai_max)}</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Mean (Kecerahan)</div>
                <div class='metric-value'>{nilai_mean:.2f}</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Std Dev (Kontras)</div>
                <div class='metric-value'>{nilai_std:.2f}</div>
            </div>
        """, unsafe_allow_html=True)

    # Interpretasi otomatis sesuai tabel di PDF
    ket_mean = "Citra cenderung terang" if nilai_mean > 127 else "Citra cenderung gelap"
    ket_std  = "Kontras tinggi" if nilai_std > 60 else "Kontras rendah"
    st.markdown(
        f"<div class='info-box-green'><b>📊 Interpretasi EDA Awal:</b> {ket_mean} | {ket_std}</div>", 
        unsafe_allow_html=True
    )

    fig1, axes1 = plt.subplots(1, 2, figsize=(12, 4))

    axes1[0].imshow(matriks_gambar, cmap="gray")
    axes1[0].set_title("Citra Asli (Grayscale)", fontweight="bold")
    axes1[0].axis("off")

    axes1[1].hist(matriks_gambar.ravel(), bins=256, range=[0, 256], color="gray")
    axes1[1].set_title("Histogram Citra Asli (EDA Awal)", fontweight="bold")
    axes1[1].set_xlabel("Intensitas Piksel")
    axes1[1].set_ylabel("Frekuensi")

    plt.tight_layout()
    st.pyplot(fig1)

    # ---------------------------------------------------------------
    # TAHAP 4: Normalisasi / Centering Data
    # ---------------------------------------------------------------
    st.header("🔧 Normalisasi / Centering Data")
    st.markdown(
    "<div class='interpretation-text'>"
    "Centering dilakukan agar PCA fokus pada <b>variasi data</b> "
    "bukan pada nilai rata-rata intensitas."
    "</div>",
    unsafe_allow_html=True
    )

    st.latex(r"X_c = X - \mu")

    st.markdown("""
    <div class='interpretation-text'>
    <b>Keterangan:</b><br><br>

    • <b>X<sub>c</sub></b> = Matriks data hasil centering<br>
    • <b>X</b> = Matriks citra asli<br>
    • <b>μ</b> = Nilai rata-rata (mean) data citra<br><br>
    </div>
    """, unsafe_allow_html=True)
    rata_rata_kolom = np.mean(matriks_gambar, axis=0)
    data_terpusat   = matriks_gambar - rata_rata_kolom

    # ---------------------------------------------------------------
    # TAHAP 5: Matriks Kovarians
    # ---------------------------------------------------------------
    st.header("📐 Matriks Kovarians")
    st.markdown(
        "<div class='interpretation-text'>"
        "Matriks kovarians dihitung dengan rumus berikut:"
        "</div>",
        unsafe_allow_html=True
    )

    st.latex(r"C = \frac{1}{n-1} X_c^T X_c")
    st.markdown("""
    <div class='interpretation-text'>
    <b>Keterangan:</b><br><br>

    • <b>C</b> = Matriks kovarians<br>
    • <b>X<sub>c</sub></b> = Data citra yang sudah dicentering<br>
    • <b>n</b> = Jumlah data / jumlah sampel<br>
    • <b>X<sub>c</sub><sup>T</sup></b> = Transpose dari matriks <b>X<sub>c</sub></b><br><br>

    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        "<div class='interpretation-text'>"
        "Matriks ini menjadi dasar untuk mencari eigenvalue dan eigenvector."
        "</div>",
        unsafe_allow_html=True
    )
    matriks_kovarians = np.cov(data_terpusat, rowvar=False)

    # ---------------------------------------------------------------
    # TAHAP 6 & 7: Eigenvalue & Eigenvector — urutkan terbesar ke terkecil
    # ---------------------------------------------------------------
    st.header("🔢 Eigenvalue, Eigenvector, dan Pengurutan")
    eigen_value, eigen_vector = np.linalg.eigh(matriks_kovarians)

    indeks_urut  = np.argsort(eigen_value)[::-1]
    eigen_value  = eigen_value[indeks_urut]
    eigen_vector = eigen_vector[:, indeks_urut]
    
    # Hitung total eigenvalue untuk persentase
    total_eigen_value = np.sum(eigen_value)

    # Tampilkan tabel 10 eigenvalue teratas
    top_n = min(10, lebar)
    tabel_data = {
        "Komponen": [f"PC{i+1}" for i in range(top_n)],
        "Eigenvalue": [f"{eigen_value[i]:.4f}" for i in range(top_n)],
        "Persentase Informasi": [f"{(eigen_value[i]/total_eigen_value)*100:.2f}%" for i in range(top_n)],
        "Makna": ["Informasi sangat besar" if i == 0
                  else "Informasi besar" if i == 1
                  else "Informasi sedang" if i < 5
                  else "Informasi kecil"
                  for i in range(top_n)],
    }
    
    st.markdown("<div class='table-container'>", unsafe_allow_html=True)
    st.table(tabel_data)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Visualisasi Top 10 Eigenvalue
    fig_eigen, ax_eigen = plt.subplots(figsize=(12, 5))
    colors = ['#1e3a5f', '#2C5AA0', '#3d7ab8', '#5a8fc7', '#7aa3d1', '#9ab8db', '#b5c9e3', '#cdd8eb', '#e0e5f0', '#f0f3f7']
    bars = ax_eigen.bar(range(1, top_n + 1), eigen_value[:top_n], color=colors[:top_n], edgecolor='#1e3a5f', linewidth=1.5)
    
    # Tambahkan nilai di atas setiap bar
    for i, (bar, val) in enumerate(zip(bars, eigen_value[:top_n])):
        height = bar.get_height()
        ax_eigen.text(bar.get_x() + bar.get_width()/2., height,
                     f'{val:.2f}\n({(val/total_eigen_value)*100:.1f}%)',
                     ha='center', va='bottom', fontsize=9, fontweight='600', color='#2c3e50')
    
    ax_eigen.set_xlabel('Komponen Utama (PC)', fontsize=11, fontweight='600', color='#2c3e50')
    ax_eigen.set_ylabel('Eigenvalue', fontsize=11, fontweight='600', color='#2c3e50')
    ax_eigen.set_title('Top 10 Eigenvalue dengan Persentase Informasi', fontsize=12, fontweight='bold', color='#1e3a5f', pad=20)
    ax_eigen.grid(axis='y', linestyle='--', alpha=0.3, color='#2C5AA0')
    ax_eigen.set_axisbelow(True)
    ax_eigen.set_xticks(range(1, top_n + 1))
    
    plt.tight_layout()
    st.pyplot(fig_eigen)

    # ---------------------------------------------------------------
    # TAHAP 8: Memilih Jumlah Komponen Utama (EDA — Scree Plot)
    # ---------------------------------------------------------------
    st.header("📈 Memilih Jumlah Komponen Utama (k)")
    st.markdown(
        "<div class='interpretation-text'>EDA membantu menentukan nilai <b>k</b> melalui <b>Scree Plot</b> dan "
        "<b>Cumulative Explained Variance</b>. Pilih titik <i>elbow</i> pada scree plot "
        "atau gunakan batas 95% sebagai titik awal.</div>",
        unsafe_allow_html=True
    )

    explained_variance_ratio = eigen_value / total_eigen_value
    cumulative_variance    = np.cumsum(explained_variance_ratio)

    max_tampil = min(100, lebar)

    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

    axes2[0].plot(range(1, max_tampil + 1), eigen_value[:max_tampil],
                  marker="o", linestyle="-", color="steelblue", markersize=4)
    axes2[0].set_title(f"Scree Plot (Top {max_tampil} Eigenvalue)", fontweight="bold")
    axes2[0].set_xlabel("Komponen Utama (PC)")
    axes2[0].set_ylabel("Eigenvalue")
    axes2[0].grid(True, linestyle="--", alpha=0.6)

    axes2[1].plot(cumulative_variance, color="green", linewidth=2)
    axes2[1].axhline(y=0.95, color="red", linestyle="--", label="Batas 95% Informasi")
    axes2[1].set_title("Cumulative Explained Variance", fontweight="bold")
    axes2[1].set_xlabel("Jumlah PC (k)")
    axes2[1].set_ylabel("Kumulatif Informasi")
    axes2[1].legend()
    axes2[1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    st.pyplot(fig2)

    # Cari otomatis k untuk 95%
    k_95 = int(np.searchsorted(cumulative_variance, 0.95)) + 1
    # st.markdown(
    #     f"<div class='info-box-green'><b>💡 Saran Otomatis:</b> Diperlukan <b>k = {k_95}</b> komponen untuk mempertahankan ≥ 95% informasi.</div>",
    #     unsafe_allow_html=True
    # )

    st.markdown("<br>", unsafe_allow_html=True)
    k = st.slider(
        "🎚️ Pilih jumlah komponen utama (k) untuk Kompresi:",
        min_value=1,
        max_value=lebar,
        value=min(k_95, lebar),
        step=1,
    )
    info_dipertahankan = cumulative_variance[k - 1] * 100
    st.markdown(
        f"<div class='info-box-green'><b>✓ Informasi yang Dipertahankan:</b> Dengan k = {k}, informasi yang dipertahankan: <b>{info_dipertahankan:.2f}%</b></div>",
        unsafe_allow_html=True
    )

    # ---------------------------------------------------------------
    # TAHAP 9 & 10: Proyeksi Data & Rekonstruksi Citra
    # ---------------------------------------------------------------
    st.header("🎯 Proyeksi Data dan Rekonstruksi Citra")
    st.markdown(
    "<div class='interpretation-text'>"
    "<b>Proyeksi Data PCA</b>"
    "</div>",
    unsafe_allow_html=True
    )

    st.latex(r"Z = X_c W_k")
    st.markdown("""
    <div class='interpretation-text'>
    <b>Keterangan:</b><br><br>

    • <b>Z</b> = Data hasil proyeksi PCA / data terkompresi<br>
    • <b>X<sub>c</sub></b> = Data citra yang sudah dicentering<br>
    • <b>W<sub>k</sub></b> = Matriks eigenvector terpilih sebanyak <b>k</b> komponen utama<br><br>

    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        "<div class='interpretation-text'>"
        "<b>Rekonstruksi Citra</b>"
        "</div>",
        unsafe_allow_html=True
    )

    st.latex(r"\hat{X} = Z W_k^T + \mu")

    st.markdown("""
    <div class='interpretation-text'>
    <b>Keterangan:</b><br><br>

    • <b>X̂</b> = Citra hasil rekonstruksi<br>
    • <b>Z</b> = Data hasil proyeksi PCA / data terkompresi<br>
    • <b>W<sub>k</sub><sup>T</sup></b> = Transpose dari matriks eigenvector terpilih<br>
    • <b>μ</b> = Nilai rata-rata (mean) data citra asli<br><br>

    </div>
    """, unsafe_allow_html=True)

    eigen_vector_terpilih = eigen_vector[:, :k]
    data_pca              = np.dot(data_terpusat, eigen_vector_terpilih)          # Proyeksi
    data_rekonstruksi     = np.dot(data_pca, eigen_vector_terpilih.T)             # Rekonstruksi
    gambar_kompresi       = data_rekonstruksi + rata_rata_kolom
    gambar_kompresi_clip  = np.clip(gambar_kompresi, 0, 255).astype(np.uint8)

    # ---------------------------------------------------------------
    # TAHAP 11: Evaluasi Kualitas Hasil Kompresi
    # ---------------------------------------------------------------
    st.header(f"⭐ Evaluasi Kualitas Hasil Kompresi (k = {k})")

    mse = np.mean((matriks_gambar - gambar_kompresi_clip) ** 2)
    psnr = 100.0 if mse == 0 else 10 * np.log10((255 ** 2) / mse)
    ssim_value = calculate_ssim(
        matriks_gambar.astype(np.uint8), gambar_kompresi_clip, data_range=255
    )

    ukuran_asli      = tinggi * lebar
    ukuran_kompresi  = (tinggi * k) + (k * lebar) + lebar   # proyeksi + eigenvector + mean
    rasio_kompresi   = ukuran_asli / ukuran_kompresi
    persentase_hemat = (1 - (ukuran_kompresi / ukuran_asli)) * 100

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Informasi Dipertahankan</div>
                <div class='metric-value'>{info_dipertahankan:.1f}%</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col_m2:
        st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>MSE</div>
                <div class='metric-value'>{mse:.2f}</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col_m3:
        st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>PSNR</div>
                <div class='metric-value'>{psnr:.2f}</div>
                <div style='font-size: 0.85rem; opacity: 0.9;'>dB</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col_m4:
        st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>SSIM</div>
                <div class='metric-value'>{ssim_value:.4f}</div>
            </div>
        """, unsafe_allow_html=True)

    # Interpretasi PSNR sesuai tabel di PDF
    if psnr < 20:
        kualitas_psnr = "Buruk (< 20 dB)"
    elif psnr < 30:
        kualitas_psnr = "Sedang (20–30 dB)"
    elif psnr < 40:
        kualitas_psnr = "Baik (30–40 dB)"
    else:
        kualitas_psnr = "Sangat Baik (> 40 dB)"

    # Interpretasi SSIM sesuai PDF
    if ssim_value >= 0.95:
        kualitas_ssim = "Sangat mirip dengan citra asli"
    elif ssim_value >= 0.80:
        kualitas_ssim = "Cukup mirip"
    elif ssim_value >= 0.50:
        kualitas_ssim = "Kemiripan sedang"
    else:
        kualitas_ssim = "Kemiripan rendah"

    st.markdown(
        f"<div class='info-box'>"
        f"<b>💾 Efisiensi Memori:</b> Ukuran terkompresi <b>{rasio_kompresi:.2f}x</b> lebih kecil (Hemat <b>{persentase_hemat:.2f}%</b>)<br><br>"
        f"<b>📊 Kualitas PSNR:</b> {kualitas_psnr}<br><br>"
        f"<b>👁️ Kualitas SSIM:</b> {kualitas_ssim}"
        f"</div>",
        unsafe_allow_html=True
    )

    # Tabel perbandingan beberapa nilai k (sesuai tabel di PDF halaman 14-15)
    st.subheader("📋 Tabel Evaluasi untuk Beberapa Nilai k")
    nilai_k_uji = sorted(set([
        max(1, lebar // 20),
        max(1, lebar // 10),
        max(1, lebar // 5),
        k_95,
        min(lebar, k_95 * 2),
    ]))

    tabel_eval = []
    for k_uji in nilai_k_uji:
        ev_terpilih = eigen_vector[:, :k_uji]
        d_pca       = np.dot(data_terpusat, ev_terpilih)
        d_rek       = np.dot(d_pca, ev_terpilih.T) + rata_rata_kolom
        d_clip      = np.clip(d_rek, 0, 255).astype(np.uint8)
        mse_k       = np.mean((matriks_gambar - d_clip) ** 2)
        psnr_k      = 100.0 if mse_k == 0 else 10 * np.log10((255 ** 2) / mse_k)
        ssim_k      = calculate_ssim(matriks_gambar.astype(np.uint8), d_clip, data_range=255)
        ev_k        = cumulative_variance[k_uji - 1] * 100
        uk_k        = (tinggi * k_uji) + (k_uji * lebar) + lebar
        rasio_k     = ukuran_asli / uk_k
        tabel_eval.append({
            "k": k_uji,
            "Explained Variance": f"{ev_k:.1f}%",
            "MSE": f"{mse_k:.2f}",
            "PSNR (dB)": f"{psnr_k:.2f}",
            "SSIM": f"{ssim_k:.4f}",
            "Rasio Kompresi": f"{rasio_k:.2f}x",
        })

    st.markdown("<div class='table-container'>", unsafe_allow_html=True)
    st.table(tabel_eval)
    st.markdown("</div>", unsafe_allow_html=True)

    # ---------------------------------------------------------------
    # TAHAP 12: EDA Setelah Kompresi
    # ---------------------------------------------------------------
    st.header("🎨 EDA Setelah Kompresi (Analisis Visual & Perbandingan)")

    error_image = np.abs(matriks_gambar - gambar_kompresi_clip)

    fig3, axes3 = plt.subplots(2, 2, figsize=(12, 10))

    # Panel 1: Citra Asli
    axes3[0, 0].imshow(matriks_gambar, cmap="gray")
    axes3[0, 0].set_title("Citra Asli", fontweight="bold")
    axes3[0, 0].axis("off")

    # Panel 2: Citra Rekonstruksi
    axes3[0, 1].imshow(gambar_kompresi_clip, cmap="gray")
    axes3[0, 1].set_title(f"Hasil Rekonstruksi (k = {k})", fontweight="bold")
    axes3[0, 1].axis("off")

    # Panel 3: Error Image (E = |X - X_hat|) sesuai rumus PDF hal. 14
    im_err = axes3[1, 0].imshow(error_image, cmap="hot")
    axes3[1, 0].set_title("Error Image |X − X̂| (Area terang = error besar)", fontweight="bold")
    axes3[1, 0].axis("off")
    fig3.colorbar(im_err, ax=axes3[1, 0], fraction=0.046, pad=0.04)

    # Panel 4: Perbandingan Histogram
    axes3[1, 1].hist(matriks_gambar.ravel(),       bins=256, range=[0, 256],
                     color="blue", alpha=0.5, label="Asli")
    axes3[1, 1].hist(gambar_kompresi_clip.ravel(), bins=256, range=[0, 256],
                     color="red",  alpha=0.5, label="Rekonstruksi")
    axes3[1, 1].set_title("Perbandingan Histogram (EDA Setelah Kompresi)", fontweight="bold")
    axes3[1, 1].set_xlabel("Intensitas Piksel")
    axes3[1, 1].set_ylabel("Frekuensi")
    axes3[1, 1].legend()

    plt.tight_layout()
    st.pyplot(fig3)

    # Interpretasi error image sesuai PDF
    st.markdown(
        "<div class='interpretation-text'>"
        "<b>Interpretasi Error Image:</b><br>"
        "• Area <b>gelap</b> → perbedaan kecil (rekonstruksi baik di area tersebut)<br>"
        "• Area <b>terang</b> → perbedaan besar (detail hilang, kemungkinan di tepi objek atau tekstur)"
        "</div>",
        unsafe_allow_html=True
    )

    # Kesimpulan akhir
    st.header("📚 Ringkasan Hubungan PCA, Eigenvalue, Eigenvector, dan EDA")
    st.markdown("""
    <div style='background: linear-gradient(135deg, #f0f3f7 0%, #e8ecf1 100%); padding: 2rem; border-radius: 12px; border: 1px solid #d0dae8;'>
    <table style='width: 100%; border-collapse: collapse;'>
        <tr style='background: linear-gradient(135deg, #1e3a5f 0%, #2C5AA0 100%); color: white;'>
            <th style='padding: 1.2rem; text-align: left; font-weight: 600;'>Konsep</th>
            <th style='padding: 1.2rem; text-align: left; font-weight: 600;'>Peran dalam Kompresi Citra</th>
        </tr>
        <tr style='border-bottom: 1px solid #d0dae8;'>
            <td style='padding: 1rem; color: #2C5AA0; font-weight: 600;'><b>PCA</b></td>
            <td style='padding: 1rem; color: #34495e;'>Metode untuk mengurangi dimensi data citra</td>
        </tr>
        <tr style='background-color: #f8f9fa; border-bottom: 1px solid #d0dae8;'>
            <td style='padding: 1rem; color: #2C5AA0; font-weight: 600;'><b>Eigenvalue</b></td>
            <td style='padding: 1rem; color: #34495e;'>Menentukan besar informasi pada tiap komponen</td>
        </tr>
        <tr style='border-bottom: 1px solid #d0dae8;'>
            <td style='padding: 1rem; color: #2C5AA0; font-weight: 600;'><b>Eigenvector</b></td>
            <td style='padding: 1rem; color: #34495e;'>Menentukan arah komponen utama</td>
        </tr>
        <tr style='background-color: #f8f9fa; border-bottom: 1px solid #d0dae8;'>
            <td style='padding: 1rem; color: #2C5AA0; font-weight: 600;'><b>Komponen Utama</b></td>
            <td style='padding: 1rem; color: #34495e;'>Representasi baru citra dengan informasi dominan</td>
        </tr>
        <tr style='border-bottom: 1px solid #d0dae8;'>
            <td style='padding: 1rem; color: #2C5AA0; font-weight: 600;'><b>Explained Variance</b></td>
            <td style='padding: 1rem; color: #34495e;'>Mengukur jumlah informasi yang dipertahankan</td>
        </tr>
        <tr style='background-color: #f8f9fa; border-bottom: 1px solid #d0dae8;'>
            <td style='padding: 1rem; color: #2C5AA0; font-weight: 600;'><b>EDA</b></td>
            <td style='padding: 1rem; color: #34495e;'>Membantu memahami data, memilih komponen, dan mengevaluasi hasil</td>
        </tr>
        <tr style='border-bottom: 1px solid #d0dae8;'>
            <td style='padding: 1rem; color: #2C5AA0; font-weight: 600;'><b>MSE</b></td>
            <td style='padding: 1rem; color: #34495e;'>Mengukur error piksel</td>
        </tr>
        <tr style='background-color: #f8f9fa; border-bottom: 1px solid #d0dae8;'>
            <td style='padding: 1rem; color: #2C5AA0; font-weight: 600;'><b>PSNR</b></td>
            <td style='padding: 1rem; color: #34495e;'>Mengukur kualitas citra berdasarkan error</td>
        </tr>
        <tr style='border-bottom: 1px solid #d0dae8;'>
            <td style='padding: 1rem; color: #2C5AA0; font-weight: 600;'><b>SSIM</b></td>
            <td style='padding: 1rem; color: #34495e;'>Mengukur kemiripan struktur citra</td>
        </tr>
        <tr style='background-color: #f8f9fa;'>
            <td style='padding: 1rem; color: #2C5AA0; font-weight: 600;'><b>Rasio Kompresi</b></td>
            <td style='padding: 1rem; color: #34495e;'>Mengukur efisiensi pengurangan ukuran data</td>
        </tr>
    </table>
    </div>
    """, unsafe_allow_html=True)

else:
    st.markdown(
        "<div class='info-box-blue' style='text-align: center; padding: 4rem 2rem;'>"
        "<h2 style='color: #1e3a5f; margin-bottom: 1.5rem;'>⬆️ Silakan Upload Gambar</h2>"
        "<p style='color: #1e3a5f; font-size: 1.1rem; line-height: 1.8;'>Unggah gambar (JPG / PNG / JPEG) untuk memulai analisis kompresi citra berbasis <b>PCA & EDA</b></p>"
        "</div>",
        unsafe_allow_html=True
    )