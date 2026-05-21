import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.metrics import structural_similarity as calculate_ssim

print("=== PROGRAM KOMPRESI CITRA PCA & EDA ===")

# TAHAP 1 & 3: Membaca Citra & Mengubah Menjadi Matriks
print("\n[Tahap 1] Silakan unggah file gambar Anda:")
nama_file = input("Masukkan path gambar: ")

# Membaca gambar dalam mode grayscale (1 kanal warna)
gambar = Image.open(nama_file).convert("L")
matriks_gambar = np.array(gambar, dtype=float)
tinggi, lebar = matriks_gambar.shape

# TAHAP 2: EDA Awal pada Citra
print("\n[Tahap 2] Melakukan EDA Awal...")
nilai_min = np.min(matriks_gambar)
nilai_max = np.max(matriks_gambar)
nilai_mean = np.mean(matriks_gambar)
nilai_std = np.std(matriks_gambar)

print(f"Ukuran Citra       : {tinggi} x {lebar} piksel")
print(f"Nilai Piksel (Min) : {nilai_min}")
print(f"Nilai Piksel (Max) : {nilai_max}")
print(f"Mean (Kecerahan)   : {nilai_mean:.2f}")
print(f"Std Dev (Kontras)  : {nilai_std:.2f}")

# Menampilkan Histogram EDA Awal
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.imshow(matriks_gambar, cmap="gray")
plt.title("Citra Asli")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.hist(matriks_gambar.ravel(), bins=256, range=[0, 256], color='gray')
plt.title("Histogram Citra Asli (EDA Awal)")
plt.xlabel("Intensitas Piksel")
plt.ylabel("Frekuensi")
plt.tight_layout()
plt.show()

# TAHAP 4: Normalisasi / Centering Data
print("\n[Tahap 4] Melakukan Centering Data...")
rata_rata_kolom = np.mean(matriks_gambar, axis=0)
data_terpusat = matriks_gambar - rata_rata_kolom

# TAHAP 5, 6, & 7: Matriks Kovarians, Eigenvalue, & Eigenvector
print("[Tahap 5-7] Menghitung Kovarians dan Eigen Dekomposisi...")
matriks_kovarians = np.cov(data_terpusat, rowvar=False)
eigen_value, eigen_vector = np.linalg.eigh(matriks_kovarians)

# Mengurutkan dari yang terbesar ke terkecil
indeks_urut = np.argsort(eigen_value)[::-1]
eigen_value = eigen_value[indeks_urut]
eigen_vector = eigen_vector[:, indeks_urut]

# TAHAP 8: Memilih Jumlah Komponen Utama (Kaitan dengan EDA)
print("\n[Tahap 8] EDA Pemilihan Komponen (Scree Plot & Explained Variance)")
total_eigen_value = np.sum(eigen_value)
explained_variance_ratio = eigen_value / total_eigen_value
cumulative_variance = np.cumsum(explained_variance_ratio)

# Plot Scree Plot & Cumulative Variance
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(eigen_value[:100], marker='o', linestyle='-', color='b', markersize=4)
plt.title("Scree Plot (Top 100 Eigenvalue)")
plt.xlabel("Komponen Utama (PC)")
plt.ylabel("Eigenvalue")
plt.grid(True, linestyle='--', alpha=0.6)

plt.subplot(1, 2, 2)
plt.plot(cumulative_variance, color='g', linewidth=2)
plt.axhline(y=0.95, color='r', linestyle='--', label='Batas 95% Informasi')
plt.title("Cumulative Explained Variance")
plt.xlabel("Jumlah PC")
plt.ylabel("Kumulatif Informasi")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# Meminta input dari user untuk nilai k
k = int(input("\nBerdasarkan grafik di atas, masukkan jumlah komponen utama (k) yang ingin digunakan: "))
info_dipertahankan = cumulative_variance[k-1] * 100

# TAHAP 9 & 10: Proyeksi Data & Rekonstruksi Citra
print(f"\n[Tahap 9-10] Melakukan Proyeksi dan Rekonstruksi menggunakan k={k}...")
eigen_vector_terpilih = eigen_vector[:, :k]

# Proyeksi (Kompresi)
data_pca = np.dot(data_terpusat, eigen_vector_terpilih)

# Rekonstruksi
data_rekonstruksi = np.dot(data_pca, eigen_vector_terpilih.T)
gambar_kompresi = data_rekonstruksi + rata_rata_kolom
gambar_kompresi_clip = np.clip(gambar_kompresi, 0, 255).astype(np.uint8)

# TAHAP 11: Evaluasi Kualitas Hasil Kompresi
print("\n[Tahap 11] Menghitung Metrik Evaluasi...")
# 1. MSE
mse = np.mean((matriks_gambar - gambar_kompresi_clip) ** 2)
# 2. PSNR
if mse == 0:
    psnr = 100 # Nilai maksimal jika identik
else:
    psnr = 10 * np.log10((255 ** 2) / mse)
# 3. SSIM
ssim_value = calculate_ssim(matriks_gambar.astype(np.uint8), gambar_kompresi_clip, data_range=255)
# 4. Rasio Kompresi
ukuran_asli = tinggi * lebar
ukuran_kompresi = (tinggi * k) + (k * lebar) + lebar
rasio_kompresi = ukuran_kompresi / ukuran_asli

print(f"--- HASIL EVALUASI (k={k}) ---")
print(f"Explained Variance : {info_dipertahankan:.2f}%")
print(f"MSE                : {mse:.4f}")
print(f"PSNR               : {psnr:.2f} dB")
print(f"SSIM               : {ssim_value:.4f}")
print(f"Rasio Kompresi     : {rasio_kompresi:.4f} (Hemat {(1-rasio_kompresi)*100:.2f}%)")

# TAHAP 12: EDA Setelah Kompresi
print("\n[Tahap 12] EDA Setelah Kompresi (Visualisasi & Error Image)...")

# Membuat Error Image (Absolute Difference)
error_image = np.abs(matriks_gambar - gambar_kompresi_clip)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Citra Asli
axes[0, 0].imshow(matriks_gambar, cmap="gray")
axes[0, 0].set_title("Citra Asli", fontweight='bold')
axes[0, 0].axis("off")

# 2. Citra Rekonstruksi
axes[0, 1].imshow(gambar_kompresi_clip, cmap="gray")
axes[0, 1].set_title(f"Hasil Rekonstruksi (k={k})", fontweight='bold')
axes[0, 1].axis("off")

# 3. Error Image
im3 = axes[1, 0].imshow(error_image, cmap="hot")
axes[1, 0].set_title("Error Image (Selisih Piksel)", fontweight='bold')
axes[1, 0].axis("off")
fig.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)

# 4. Perbandingan Histogram
axes[1, 1].hist(matriks_gambar.ravel(), bins=256, range=[0, 256], color='blue', alpha=0.5, label='Asli')
axes[1, 1].hist(gambar_kompresi_clip.ravel(), bins=256, range=[0, 256], color='red', alpha=0.5, label='Rekonstruksi')
axes[1, 1].set_title("Perbandingan Histogram", fontweight='bold')
axes[1, 1].set_xlabel("Intensitas Piksel")
axes[1, 1].set_ylabel("Frekuensi")
axes[1, 1].legend()

plt.tight_layout()
plt.show()