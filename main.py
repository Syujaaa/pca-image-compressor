import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.metrics import structural_similarity as calculate_ssim

st.set_page_config(layout="wide") # Membuat tampilan web lebih luas
st.title("PROGRAM KOMPRESI CITRA PCA & EDA")

uploaded_file = st.file_uploader("Upload gambar", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # TAHAP 1 & 3: Membaca Citra & Mengubah Menjadi Matriks
    gambar = Image.open(uploaded_file).convert("L")
    matriks_gambar = np.array(gambar, dtype=float)
    tinggi, lebar = matriks_gambar.shape

    st.subheader("1. Karakteristik Citra Asli (EDA Awal)")
    
    # Menghitung statistik EDA Awal
    nilai_min = np.min(matriks_gambar)
    nilai_max = np.max(matriks_gambar)
    nilai_mean = np.mean(matriks_gambar)
    nilai_std = np.std(matriks_gambar)

    # Menampilkan statistik di Streamlit menggunakan kolom
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    col_stat1.metric("Ukuran Citra", f"{tinggi} x {lebar} px")
    col_stat2.metric("Rentang Piksel", f"{nilai_min} - {nilai_max}")
    col_stat3.metric("Mean (Kecerahan)", f"{nilai_mean:.2f}")
    col_stat4.metric("Std Dev (Kontras)", f"{nilai_std:.2f}")

    # Menampilkan Histogram EDA Awal
    fig1, axes1 = plt.subplots(1, 2, figsize=(12, 4))
    axes1[0].imshow(matriks_gambar, cmap="gray")
    axes1[0].set_title("Citra Asli")
    axes1[0].axis("off")

    axes1[1].hist(matriks_gambar.ravel(), bins=256, range=[0, 256], color='gray')
    axes1[1].set_title("Histogram Citra Asli (EDA Awal)")
    axes1[1].xlabel("Intensitas Piksel")
    axes1[1].ylabel("Frekuensi")
    st.pyplot(fig1)

    # TAHAP 4: Normalisasi / Centering Data
    rata_rata_kolom = np.mean(matriks_gambar, axis=0)
    data_terpusat = matriks_gambar - rata_rata_kolom

    # TAHAP 5, 6, & 7: Matriks Kovarians, Eigenvalue, & Eigenvector
    matriks_kovarians = np.cov(data_terpusat, rowvar=False)
    eigen_value, eigen_vector = np.linalg.eigh(matriks_kovarians)

    # Mengurutkan dari yang terbesar ke terkecil
    indeks_urut = np.argsort(eigen_value)[::-1]
    eigen_value = eigen_value[indeks_urut]
    eigen_vector = eigen_vector[:, indeks_urut]

    # TAHAP 8: Memilih Jumlah Komponen Utama (Kaitan dengan EDA)
    st.subheader("2. Analisis Komponen Utama (Scree Plot & Variance)")
    total_eigen_value = np.sum(eigen_value)
    explained_variance_ratio = eigen_value / total_eigen_value
    cumulative_variance = np.cumsum(explained_variance_ratio)

    # Plot Scree Plot & Cumulative Variance
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    axes2[0].plot(eigen_value[:100], marker='o', linestyle='-', color='b', markersize=4)
    axes2[0].set_title("Scree Plot (Top 100 Eigenvalue)")
    axes2[0].xlabel("Komponen Utama (PC)")
    axes2[0].ylabel("Eigenvalue")
    axes2[0].grid(True, linestyle='--', alpha=0.6)

    axes2[1].plot(cumulative_variance, color='g', linewidth=2)
    axes2[1].axhline(y=0.95, color='r', linestyle='--', label='Batas 95% Informasi')
    axes2[1].set_title("Cumulative Explained Variance")
    axes2[1].xlabel("Jumlah PC")
    axes2[1].ylabel("Kumulatif Informasi")
    axes2[1].legend()
    axes2[1].grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig2)

    # Interaksi Pengguna untuk memilih k
    k = st.slider(
        "Pilih jumlah komponen utama (k) untuk Kompresi:",
        1,
        int(min(lebar, 100)),
        50
    )
    info_dipertahankan = cumulative_variance[k-1] * 100

    # TAHAP 9 & 10: Proyeksi Data & Rekonstruksi Citra
    eigen_vector_terpilih = eigen_vector[:, :k]
    data_pca = np.dot(data_terpusat, eigen_vector_terpilih) # Proyeksi
    data_rekonstruksi = np.dot(data_pca, eigen_vector_terpilih.T) # Rekonstruksi
    gambar_kompresi = data_rekonstruksi + rata_rata_kolom
    gambar_kompresi_clip = np.clip(gambar_kompresi, 0, 255).astype(np.uint8)

    # TAHAP 11: Evaluasi Kualitas Hasil Kompresi
    st.subheader(f"3. Hasil Evaluasi Metrik (k = {k})")
    
    mse = np.mean((matriks_gambar - gambar_kompresi_clip) ** 2)
    if mse == 0:
        psnr = 100.0
    else:
        psnr = 10 * np.log10((255 ** 2) / mse)
        
    ssim_value = calculate_ssim(matriks_gambar.astype(np.uint8), gambar_kompresi_clip, data_range=255)
    
    ukuran_asli = tinggi * lebar
    ukuran_kompresi = (tinggi * k) + (k * lebar) + lebar
    rasio_kompresi = ukuran_asli / ukuran_kompresi  # Dibalik agar menunjukkan rasio lipat (misal: 2x lebih kecil)
    persentase_hemat = (1 - (ukuran_kompresi / ukuran_asli)) * 100

    col_met1, col_met2, col_met3, col_met4 = st.columns(4)
    col_met1.metric("Information Retained", f"{info_dipertahankan:.2f}%")
    col_met2.metric("MSE", f"{mse:.4f}")
    col_met3.metric("PSNR", f"{psnr:.2f} dB")
    col_met4.metric("SSIM", f"{ssim_value:.4f}")
    st.info(f"**Efisiensi Memori:** Ukuran terkompresi {rasio_kompresi:.2f}x lebih kecil (Hemat {persentase_hemat:.2f}%)")

    # TAHAP 12: EDA Setelah Kompresi (Visualisasi Utama)
    st.subheader("4. Analisis Visual & Perbandingan (EDA Setelah Kompresi)")
    error_image = np.abs(matriks_gambar - gambar_kompresi_clip)

    fig3, axes3 = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Citra Asli
    axes3[0, 0].imshow(matriks_gambar, cmap="gray")
    axes3[0, 0].set_title("Citra Asli", fontweight='bold')
    axes3[0, 0].axis("off")

    # 2. Citra Rekonstruksi
    axes3[0, 1].imshow(gambar_kompresi_clip, cmap="gray")
    axes3[0, 1].set_title(f"Hasil Rekonstruksi (k={k})", fontweight='bold')
    axes3[0, 1].axis("off")

    # 3. Error Image
    im3 = axes3[1, 0].imshow(error_image, cmap="hot")
    axes3[1, 0].set_title("Error Image (Selisih Piksel)", fontweight='bold')
    axes3[1, 0].axis("off")
    fig3.colorbar(im3, ax=axes3[1, 0], fraction=0.046, pad=0.04)

    # 4. Perbandingan Histogram
    axes3[1, 1].hist(matriks_gambar.ravel(), bins=256, range=[0, 256], color='blue', alpha=0.5, label='Asli')
    axes3[1, 1].hist(gambar_kompresi_clip.ravel(), bins=256, range=[0, 256], color='red', alpha=0.5, label='Rekonstruksi')
    axes3[1, 1].set_title("Perbandingan Histogram", fontweight='bold')
    axes3[1, 1].set_xlabel("Intensitas Piksel")
    axes3[1, 1].set_ylabel("Frekuensi")
    axes3[1, 1].legend()

    plt.tight_layout()
    st.pyplot(fig3) # Menampilkan 4 panel visualisasi ke Streamlit