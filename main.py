import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.metrics import structural_similarity as calculate_ssim

st.set_page_config(layout="wide")
st.title("PROGRAM KOMPRESI CITRA PCA & EDA")
st.markdown(
    "Aplikasi ini mengimplementasikan kompresi citra berbasis **Principal Component Analysis (PCA)** "
    "menggunakan Eigenvalue dan Eigenvector, dilengkapi dengan **Exploratory Data Analysis (EDA)** "
    "sebelum, saat, dan setelah kompresi."
)

uploaded_file = st.file_uploader("Upload gambar (JPG / PNG / JPEG)", type=["jpg", "png", "jpeg"])

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
    st.header("Tahap 2 — EDA Awal pada Citra")

    nilai_min = np.min(matriks_gambar)
    nilai_max = np.max(matriks_gambar)
    nilai_mean = np.mean(matriks_gambar)
    nilai_std = np.std(matriks_gambar)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Ukuran Citra", f"{tinggi} x {lebar} px")
    col2.metric("Rentang Piksel", f"{int(nilai_min)} – {int(nilai_max)}")
    col3.metric("Mean (Kecerahan)", f"{nilai_mean:.2f}")
    col4.metric("Std Dev (Kontras)", f"{nilai_std:.2f}")

    # Interpretasi otomatis sesuai tabel di PDF
    ket_mean = "Citra cenderung terang" if nilai_mean > 127 else "Citra cenderung gelap"
    ket_std  = "Kontras tinggi" if nilai_std > 60 else "Kontras rendah"
    st.info(f"📊 **Interpretasi EDA Awal:** {ket_mean} | {ket_std}")

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
    st.header("Tahap 4 — Normalisasi / Centering Data")
    st.markdown(
        r"Centering dilakukan dengan rumus $X_c = X - \mu$, agar PCA fokus pada **variasi data** "
        "bukan pada nilai rata-rata intensitas."
    )
    rata_rata_kolom = np.mean(matriks_gambar, axis=0)
    data_terpusat   = matriks_gambar - rata_rata_kolom

    # ---------------------------------------------------------------
    # TAHAP 5: Matriks Kovarians
    # ---------------------------------------------------------------
    st.header("Tahap 5 — Matriks Kovarians")
    st.markdown(
        r"Matriks kovarians dihitung dengan $C = \frac{1}{n-1} X_c^T X_c$. "
        "Matriks ini menjadi dasar untuk mencari eigenvalue dan eigenvector."
    )
    matriks_kovarians = np.cov(data_terpusat, rowvar=False)

    # ---------------------------------------------------------------
    # TAHAP 6 & 7: Eigenvalue & Eigenvector — urutkan terbesar ke terkecil
    # ---------------------------------------------------------------
    st.header("Tahap 6 & 7 — Eigenvalue, Eigenvector, dan Pengurutan")
    eigen_value, eigen_vector = np.linalg.eigh(matriks_kovarians)

    indeks_urut  = np.argsort(eigen_value)[::-1]
    eigen_value  = eigen_value[indeks_urut]
    eigen_vector = eigen_vector[:, indeks_urut]

    # Tampilkan tabel 5 eigenvalue teratas
    top_n = min(5, lebar)
    tabel_data = {
        "Komponen": [f"PC{i+1}" for i in range(top_n)],
        "Eigenvalue": [f"{eigen_value[i]:.4f}" for i in range(top_n)],
        "Makna": ["Informasi sangat besar" if i == 0
                  else "Informasi besar" if i == 1
                  else "Informasi sedang" if i == 2
                  else "Informasi kecil"
                  for i in range(top_n)],
    }
    st.table(tabel_data)

    # ---------------------------------------------------------------
    # TAHAP 8: Memilih Jumlah Komponen Utama (EDA — Scree Plot)
    # ---------------------------------------------------------------
    st.header("Tahap 8 — Memilih Jumlah Komponen Utama (k)")
    st.markdown(
        "EDA membantu menentukan nilai **k** melalui **Scree Plot** dan "
        "**Cumulative Explained Variance**. Pilih titik *elbow* pada scree plot "
        "atau gunakan batas 95% sebagai titik awal."
    )

    total_eigen_value      = np.sum(eigen_value)
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
    st.info(f"ℹ️ Diperlukan **k = {k_95}** komponen untuk mempertahankan ≥ 95% informasi.")

    k = st.slider(
        "Pilih jumlah komponen utama (k) untuk Kompresi:",
        min_value=1,
        max_value=lebar,
        value=min(k_95, lebar),
        step=1,
    )
    info_dipertahankan = cumulative_variance[k - 1] * 100
    st.write(f"Dengan **k = {k}**, informasi yang dipertahankan: **{info_dipertahankan:.2f}%**")

    # ---------------------------------------------------------------
    # TAHAP 9 & 10: Proyeksi Data & Rekonstruksi Citra
    # ---------------------------------------------------------------
    st.header("Tahap 9 & 10 — Proyeksi Data dan Rekonstruksi Citra")
    st.markdown(
        r"**Proyeksi:** $Z = X_c W_k$ &nbsp;&nbsp;|&nbsp;&nbsp; "
        r"**Rekonstruksi:** $\hat{X} = Z W_k^T + \mu$"
    )

    eigen_vector_terpilih = eigen_vector[:, :k]
    data_pca              = np.dot(data_terpusat, eigen_vector_terpilih)          # Proyeksi
    data_rekonstruksi     = np.dot(data_pca, eigen_vector_terpilih.T)             # Rekonstruksi
    gambar_kompresi       = data_rekonstruksi + rata_rata_kolom
    gambar_kompresi_clip  = np.clip(gambar_kompresi, 0, 255).astype(np.uint8)

    # ---------------------------------------------------------------
    # TAHAP 11: Evaluasi Kualitas Hasil Kompresi
    # ---------------------------------------------------------------
    st.header(f"Tahap 11 — Evaluasi Kualitas Hasil Kompresi (k = {k})")

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
    col_m1.metric("Information Retained", f"{info_dipertahankan:.2f}%")
    col_m2.metric("MSE", f"{mse:.4f}")
    col_m3.metric("PSNR", f"{psnr:.2f} dB")
    col_m4.metric("SSIM", f"{ssim_value:.4f}")

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

    st.info(
        f"**Efisiensi Memori:** Ukuran terkompresi **{rasio_kompresi:.2f}x** lebih kecil "
        f"(Hemat **{persentase_hemat:.2f}%**)  \n"
        f"**Kualitas PSNR:** {kualitas_psnr}  \n"
        f"**Kualitas SSIM:** {kualitas_ssim}"
    )

    # Tabel perbandingan beberapa nilai k (sesuai tabel di PDF halaman 14-15)
    st.subheader("Tabel Evaluasi untuk Beberapa Nilai k")
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

    st.table(tabel_eval)

    # ---------------------------------------------------------------
    # TAHAP 12: EDA Setelah Kompresi
    # ---------------------------------------------------------------
    st.header("Tahap 12 — EDA Setelah Kompresi (Analisis Visual & Perbandingan)")

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
        "**Interpretasi Error Image:**  \n"
        "- Area **gelap** → perbedaan kecil (rekonstruksi baik di area tersebut)  \n"
        "- Area **terang** → perbedaan besar (detail hilang, kemungkinan di tepi objek atau tekstur)"
    )

    # Kesimpulan akhir
    st.header("Ringkasan Hubungan PCA, Eigenvalue, Eigenvector, dan EDA")
    st.markdown("""
| Konsep | Peran dalam Kompresi Citra |
|---|---|
| **PCA** | Metode untuk mengurangi dimensi data citra |
| **Eigenvalue** | Menentukan besar informasi pada tiap komponen |
| **Eigenvector** | Menentukan arah komponen utama |
| **Komponen Utama** | Representasi baru citra dengan informasi dominan |
| **Explained Variance** | Mengukur jumlah informasi yang dipertahankan |
| **EDA** | Membantu memahami data, memilih komponen, dan mengevaluasi hasil |
| **MSE** | Mengukur error piksel |
| **PSNR** | Mengukur kualitas citra berdasarkan error |
| **SSIM** | Mengukur kemiripan struktur citra |
| **Rasio Kompresi** | Mengukur efisiensi pengurangan ukuran data |
    """)

else:
    st.info("⬆️ Silakan upload gambar terlebih dahulu untuk memulai analisis.")