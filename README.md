# Tugas Deep Learning Sesi 10

Implementasi mekanisme **attention** dan **blok transformer sederhana** untuk
Praktikum Deep Learning Sesi 10.

Repository ini mengerjakan bagian **Instruksi** pada
[`DL_Ass_10.pdf`](./DL_Ass_10.pdf). Bagian **Pertanyaan Analisis 1--9** tidak
dikerjakan di source code ini karena menjadi tanggung jawab anggota kelompok
yang lain.

## Tujuan

Implementasi dan verifikasi yang tersedia mencakup:

1. scaled dot-product attention;
2. verifikasi hasil attention dengan perhitungan manual;
3. multi-head attention;
4. padding mask dan causal mask;
5. sinusoidal positional encoding;
6. transformer block dengan residual connection, feed-forward network, dan
   layer normalization;
7. klasifikasi sekuens pendek pada data sintetis;
8. pemeriksaan shape, jumlah parameter, jumlah baris softmax, dan overfit satu
   batch.

## Struktur Repository

```text
.
├── DL_Ass_10.pdf
├── README.md
├── attention_transformer.py
├── Laporan_Tugas_Deep_Learning_Sesi_10.tex
├── Laporan_Tugas_Deep_Learning_Sesi_10.pdf
├── image.png
└── artifacts/
    ├── training_metrics.csv
    ├── verification_report.json
    └── verification_report.md
```

### File penting

- [`attention_transformer.py`](./attention_transformer.py): implementasi
  utama dan seluruh verifikasi yang dapat dijalankan.
- [`artifacts/verification_report.md`](./artifacts/verification_report.md):
  ringkasan hasil verifikasi yang mudah dibaca.
- [`artifacts/verification_report.json`](./artifacts/verification_report.json):
  output numerik lengkap dari verifikasi.
- [`artifacts/training_metrics.csv`](./artifacts/training_metrics.csv): loss
  dan akurasi setiap epoch.
- [`Laporan_Tugas_Deep_Learning_Sesi_10.tex`](./Laporan_Tugas_Deep_Learning_Sesi_10.tex):
  template laporan LaTeX dengan tabel identitas anggota Kelompok 5, tabel
  hasil, dan placeholder bagian analisis.
- [`DL_Ass_10.pdf`](./DL_Ass_10.pdf): instruksi resmi praktikum.

## Konfigurasi Eksperimen

Konfigurasi utama yang digunakan:

| Parameter | Nilai |
|---|---:|
| Framework | PyTorch |
| Seed | `10` |
| Format tensor | `(B, T, d_model)` |
| Batch size `B` | `2` |
| Panjang sekuens `T` | `4` |
| Dimensi model `d_model` | `8` |
| Jumlah head `h` | `2` |
| Dimensi tiap head `d_head` | `4` |
| Dimensi feed-forward `d_ff` | `32` |
| Aktivasi FFN | GELU |
| Optimizer | Adam |
| Learning rate | `0.02` |
| Epoch | `100` |

## Persiapan Environment

Pastikan Python dan PyTorch tersedia. Gunakan interpreter Python yang aktif pada
environment lokal:

```bash
python --version
```

Periksa instalasi:

```bash
python -c "import torch; print(torch.__version__)"
```

Jika sistem menggunakan perintah `python3` sebagai pengganti `python`, gunakan `python3`
pada seluruh perintah. Jika memakai virtual environment atau Conda, aktifkan
environment tersebut terlebih dahulu.

## Menjalankan Implementasi

Jalankan dari root repository:

```bash
python attention_transformer.py
```

Script akan:

1. menjalankan verifikasi scaled dot-product attention;
2. memeriksa shape dan jumlah parameter;
3. menguji padding mask dan causal mask;
4. memeriksa positional encoding;
5. melatih classifier sekuens sintetis;
6. menguji kemampuan overfit satu batch;
7. menulis hasil ke direktori `artifacts/`.

Output yang dihasilkan atau diperbarui:

- `artifacts/verification_report.md`
- `artifacts/verification_report.json`
- `artifacts/training_metrics.csv`

## Hasil yang Diharapkan

Dengan seed dan konfigurasi yang sama, hasil verifikasi utama adalah:

| Pemeriksaan | Hasil |
|---|---:|
| Error maksimum perhitungan manual attention | `0.0` |
| Parameter multi-head attention manual/program | `288 / 288` |
| Parameter transformer block manual/program | `872 / 872` |
| Shape input | `(2, 4, 8)` |
| Shape Q/K/V setelah split head | `(2, 2, 4, 4)` |
| Shape skor dan bobot attention | `(2, 2, 4, 4)` |
| Shape output transformer block | `(2, 4, 8)` |
| Bobot maksimum padding terlarang | `0.0` |
| Bobot maksimum causal terlarang | `0.0` |
| Loss awal | `0.734517` |
| Loss akhir | `0.000168` |
| Akurasi train akhir | `1.0000` |
| Akurasi test | `0.8750` |
| Overfit satu batch | **PASS** |

Nilai detail, termasuk jumlah baris softmax dan matriks mask, tersedia pada
[`verification_report.json`](./artifacts/verification_report.json).

## Ringkasan Implementasi

### Scaled dot-product attention

Fungsi `scaled_dot_product_attention` menghitung:

```text
scores = QK^T / sqrt(d_k)
weights = softmax(scores)
output = weights V
```

Mask diterapkan sebelum softmax dengan memberi nilai sangat negatif pada posisi
terlarang. Fungsi juga menolak baris mask yang seluruhnya melarang key valid.

### Multi-head attention

Konfigurasi `d_model=8` dan `h=2` menghasilkan `d_head=4`. Empat proyeksi
linear yang digunakan adalah:

- proyeksi Q;
- proyeksi K;
- proyeksi V;
- output projection.

Dengan bias pada setiap layer linear, total parameter multi-head attention
adalah `288`.

### Masking

- **Padding mask** melarang attention membaca token padding pada dimensi key.
- **Causal mask** melarang query membaca token masa depan.

Kedua pengujian memverifikasi bahwa bobot pada posisi terlarang bernilai `0.0`
dan jumlah setiap baris bobot softmax tetap mendekati `1.0`.

### Transformer block

Block menggunakan urutan:

```text
input
  → multi-head attention
  → residual connection + LayerNorm
  → feed-forward network
  → residual connection + LayerNorm
  → output
```

Dengan `d_ff=32`, jumlah parameter block adalah `872`.

### Uji fungsional

Model classifier menggunakan mean pooling pada output transformer block,
kemudian layer linear dengan dua kelas. Label data sintetis ditentukan dari
tanda rata-rata fitur pertama sepanjang sekuens. Uji ini ditujukan sebagai
bukti bahwa model dapat belajar pada tugas kecil dan meng-overfit satu batch,
bukan sebagai klaim performa generalisasi pada dataset besar.

## Template Laporan

Template laporan tersedia pada
[`Laporan_Tugas_Deep_Learning_Sesi_10.tex`](./Laporan_Tugas_Deep_Learning_Sesi_10.tex).
Template tersebut
sudah memuat:

- halaman judul;
- tabel identitas Kelompok 5;
- bagian tujuan dan konfigurasi;
- bagian verifikasi manual;
- tabel parameter dan shape;
- bagian masking, positional encoding, dan transformer block;
- bagian uji fungsional;
- ringkasan hasil verifikasi;
- placeholder Pertanyaan Analisis 1--9 untuk diisi anggota lain;
- lampiran perintah menjalankan program.

### Instalasi `latexmk` pada Windows

Pada Windows, cara yang paling praktis adalah memasang distribusi LaTeX
MiKTeX. `latexmk` dapat dipasang melalui MiKTeX Console:

1. Unduh dan pasang MiKTeX dari
   [miktex.org/download](https://miktex.org/download).
2. Buka **MiKTeX Console**, lalu lakukan pemeriksaan update melalui menu
   **Updates**.
3. Buka menu **Packages**, cari `latexmk`, lalu klik **Install**.
4. Pastikan direktori bin MiKTeX sudah masuk ke `PATH`. Tutup dan buka kembali
   terminal setelah mengubah `PATH`.

Alternatifnya, pada instalasi MiKTeX yang mendukung `mpm`, jalankan PowerShell:

```powershell
mpm --install=latexmk
```

Verifikasi dari PowerShell atau Command Prompt:

```powershell
latexmk --version
```

Jika menggunakan distribusi TeX Live, pasang TeX Live dari
[tug.org/texlive](https://www.tug.org/texlive/). Paket `latexmk` biasanya
tersedia di instalasi TeX Live; jika belum tersedia, pasang melalui TeX Live
Manager:

```powershell
tlmgr install latexmk
```

### Instalasi `latexmk` pada macOS

Pada macOS, instalasi yang direkomendasikan adalah MacTeX:

1. Unduh dan pasang MacTeX dari
   [tug.org/mactex](https://www.tug.org/mactex/).
2. Buka terminal baru agar `PATH` TeX Live diperbarui.
3. Verifikasi instalasi:

```bash
latexmk --version
```

Jika MacTeX sudah terpasang tetapi `latexmk` belum tersedia, pasang paketnya
melalui TeX Live Manager:

```bash
sudo tlmgr install latexmk
```

Alternatif yang lebih ringan adalah memasang BasicTeX dari halaman MacTeX,
kemudian menjalankan:

```bash
sudo tlmgr update --self
sudo tlmgr install latexmk
```

### Kompilasi laporan

Jika instalasi LaTeX sudah lengkap, jalankan:

```bash
latexmk -pdf -interaction=nonstopmode Laporan_Tugas_Deep_Learning_Sesi_10.tex
```

Alternatif:

```bash
pdflatex Laporan_Tugas_Deep_Learning_Sesi_10.tex
```

Perintah perlu dijalankan dua kali agar daftar isi dan referensi silang
terbarui. Laporan saat ini sudah berhasil dikompilasi menjadi
`Laporan_Tugas_Deep_Learning_Sesi_10.pdf` dengan 11 halaman. File intermediate
LaTeX seperti `.aux`, `.log`, `.fls`, `.fdb_latexmk`, `.out`, dan `.toc`
diabaikan oleh `.gitignore` dan tidak perlu diunggah ke GitHub.

## Checklist Pengumpulan

- [x] Implementasi scaled dot-product attention.
- [x] Verifikasi dengan perhitungan manual.
- [x] Implementasi multi-head attention.
- [x] Perbandingan jumlah parameter manual dan program.
- [x] Padding mask.
- [x] Causal mask.
- [x] Bukti bobot posisi terlarang bernilai nol.
- [x] Sinusoidal positional encoding.
- [x] Transformer block.
- [x] Pemeriksaan shape setiap tahap.
- [x] Pemeriksaan jumlah baris matriks attention setelah softmax.
- [x] Uji fungsional pada data sintetis.
- [x] Bukti overfit satu batch.
- [x] Template laporan LaTeX.
- [ ] Mengisi Pertanyaan Analisis 1--9 oleh anggota yang bertanggung jawab.
- [ ] Menambahkan grafik training ke laporan.
- [x] Mengompilasi PDF laporan akhir.
- [ ] Memeriksa PDF akhir secara visual untuk memastikan tabel tidak terpotong,
  persamaan benar, dan referensi gambar tidak hilang.

## Catatan Scope

Repository ini sengaja membatasi implementasi pada bagian **Instruksi**.
Pembahasan konseptual untuk Pertanyaan Analisis, termasuk topik evolusi
attention/transformer dan pengayaan modern, harus ditambahkan secara terpisah
oleh anggota kelompok yang ditugaskan.
