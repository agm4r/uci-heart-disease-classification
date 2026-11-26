# Desain: Pengerjaan Tugas Heart Disease Fase per Fase

Tanggal: 2026-11-26
Acuan tugas: `01-panduan-tugas/Panduan Tugas Kelompok ML - Studi Kasus 4 (Penyakit Jantung).pdf`

## Tujuan dan alasan

Tugas kelompok Machine Learning Studi Kasus 4 (Diagnosis Penyakit Jantung) dikerjakan bertahap mengikuti 5 fase di panduan. Claude yang menulis kode, user yang mereview dan memahami hasilnya. Sebagian fase mungkin dikerjakan anggota kelompok lain. Kalau itu terjadi, kode mereka dimasukkan ke section fase yang sesuai di notebook yang sama.

Pengerjaan per fase dipilih supaya setiap fase bisa direview dan dipahami sebelum lanjut, dan supaya riwayat git menunjukkan dengan jelas pekerjaan mana yang masuk fase berapa.

## Deliverable

1. Notebook `03-notebook/Kelompok6_Heart_Disease_Prediction.ipynb`. Nama file sudah memakai nomor kelompok (Kelompok 6). Notebook harus bisa Run All dari awal sampai akhir tanpa error.
2. Laporan Teknis & Bisnis (PDF) dengan bab: Cover, Pendahuluan, Metodologi, Hasil & Pembahasan, Kesimpulan & Rekomendasi Klinis, Daftar Pustaka.
3. Slide presentasi (PPTX, diekspor juga ke PDF) dengan alur: Masalah Klinis, Analisis Data, Solusi Model, Dampak & Rekomendasi Medis.

## Goals

- Semua poin wajib di panduan untuk Fase 1 sampai Fase 5 terpenuhi, termasuk nilai parameter yang disebut eksplisit (`stratify=y`, `random_state=42`, K-Fold K=5 atau 10, grid parameter SVM dan KNN).
- Setiap fase menghasilkan tiga hal: section notebook (kode dan Markdown), grafik yang disimpan sebagai file PNG, dan draf tulisan untuk bab laporan serta isi slide yang terkait fase itu.
- Riwayat git rapi: setiap commit jelas fasenya.

## Non-goals

- Deployment sungguhan (web app atau API). Fase 5 hanya meminta "Deployment Concept" dan dashboard sederhana, jadi cukup grafik ringkasan dan simulasi skrining di notebook.
- Streamlit. Panduan mengizinkan Matplotlib, Seaborn, atau Streamlit. Kita pakai Matplotlib/Seaborn supaya semuanya tetap di satu notebook.
- Algoritma di luar tiga yang diminta (SVM, KNN, Logistic Regression).

## Anti-goals (gagal walaupun kodenya jalan)

- Data leakage: scaler atau imputer di-fit memakai data uji atau seluruh data.
- Run All error di notebook final.
- Interpretasi faktor risiko yang memakai label atau kamus data versi Kaggle tanpa koreksi (lihat bagian Constraints), sehingga kesimpulan medisnya terbalik tanpa disadari.
- Model hasil tuning yang Recall-nya tinggi karena menebak hampir semua pasien sebagai kelas 1 (precision ambruk), lalu tetap dilaporkan sebagai model terbaik.

## Constraints

- Dataset: `02-dataset/heart_disease.csv`, 303 baris, 14 kolom, versi Kaggle dari Heart Disease UCI. File diawali BOM UTF-8.
- Temuan data yang wajib ditangani dan dibahas:
  - 1 baris duplikat.
  - `thal=0` (2 baris) dan `ca=4` (5 baris) adalah kode yang tidak ada di dataset UCI asli, jadi dianggap missing value.
  - Label `target` versi Kaggle terbalik total. Hasil pencocokan 303 baris dengan `processed.cleveland.data` dari UCI (kunci `age, trestbps, chol, thalach, oldpeak`, cocok satu-satu) menunjukkan semua 165 baris `target=1` adalah pasien sehat (`num=0`) di data asli, dan semua 138 baris `target=0` adalah pasien sakit.
  - Kamus data `cp` dan `thal` versi Kaggle juga keliru. Arti yang benar: `cp` 0 = asymptomatic, 1 = atypical angina, 2 = non-anginal pain, 3 = typical angina; `thal` 1 = fixed defect, 2 = normal, 3 = reversible defect.
- Environment: Python 3.12 di `.venv` (dibuat dengan `uv`, sudah di-ignore git). Dependency dicatat di `requirements.txt` supaya anggota lain bisa install.
- Notebook membaca dataset dengan path relatif `../02-dataset/heart_disease.csv`, jadi notebook harus dijalankan dari folder `03-notebook`.

## Pendekatan yang dipertimbangkan

| Pendekatan | Kelebihan | Kekurangan |
|---|---|---|
| A. Satu notebook final, section per fase (dipilih) | Deliverable langsung terbentuk, Run All bisa dicek tiap fase, variabel antar fase tersambung | Rawan conflict git kalau anggota lain mengedit `.ipynb` yang sama bersamaan |
| B. Satu notebook per fase, digabung di akhir | Anggota lain bisa kerja paralel tanpa conflict | Kode berulang, hasil antar fase harus disimpan ke file, langkah penggabungan di akhir rawan error |
| C. Kode di file `.py`, notebook tipis | Kode lebih modular | Dosen ingin kode dan penjelasan Markdown di notebook, jadi ini hanya menambah kerumitan |

User memilih A. Untuk menghindari conflict, anggota lain mengirim kode mereka, lalu Claude yang memasukkannya ke notebook.

## Keputusan yang sudah disepakati

- Claude menulis semua kode. Laporan dan slide ikut dikerjakan.
- Git: setiap fase dikerjakan di branch sendiri (`fase-1` sampai `fase-5`), lalu di-merge ke `main` saat fasenya selesai. Persiapan awal (dokumen ini, `requirements.txt`) ada di branch `setup`.
- Commit message ditulis dalam bahasa Indonesia dengan prefix fase, contohnya `feat(fase-1): tambah grafik distribusi target`. Tipe commit (`feat`, `fix`, `docs`, `chore`) tetap memakai keyword standar Conventional Commits. Persiapan memakai scope `setup`, finalisasi memakai scope `final`.
- Semua laporan, slide, Markdown notebook, dan dokumentasi di repo ditulis dalam bahasa Indonesia.
- (Direvisi 2026-11-26 setelah review Fase 1) Label `target` dibalik di notebook (`1 - target`) supaya `1` benar-benar berarti ada penyakit jantung, sesuai definisi panduan. File `02-dataset/heart_disease.csv` tidak diubah. Bukti pencocokan ditampilkan di notebook memakai salinan data UCI asli di `02-dataset/uci-cleveland-asli.data`. Setelah koreksi: 138 pasien sakit, 164 sehat (setelah duplikat dihapus).
- Baris duplikat dihapus. `thal=0` dan `ca=4` diubah jadi NaN, lalu diisi modus.
- Markdown notebook, laporan, dan slide ditulis dalam bahasa Indonesia.

## Keputusan dari Claude (bisa dikoreksi user)

- Setiap akhir fase, Claude berhenti untuk direview user. Merge ke `main` dan mulai fase berikutnya baru dilakukan setelah user setuju.
- Merge memakai `git merge --no-ff` supaya setiap fase punya merge commit sendiri sebagai penanda fase selesai.
- Laporan DOCX dan slide PPTX di-generate satu kali di Fase 5 dari draf Markdown. Setelah itu, revisi dilakukan langsung di file DOCX/PPTX, supaya editan manual anggota kelompok tidak tertimpa saat generate ulang.

## Isi per fase

### Fase 1: Clinical & Data Understanding

- Markdown Clinical Understanding: masalah klinis, tujuan deteksi dini, metrik sukses (Recall kelas 1 tinggi, False Negative seminimal mungkin), dan alasannya.
- Load data, `df.info()`, `df.describe()`, cek missing value dan duplikat, hapus duplikat, ubah `thal=0` dan `ca=4` jadi NaN.
- Visualisasi EDA: distribusi `target`, `cp` vs `target`, distribusi `age`, `trestbps`, `chol`, `thalach` per `target`, heatmap korelasi.
- Markdown temuan EDA, termasuk anomali label.
- Draf laporan: Pendahuluan dan Metodologi bagian Data & EDA. Isi slide: Masalah Klinis dan Insight EDA.

### Fase 2: Data Preparation

- One-Hot Encoding untuk fitur nominal: `cp`, `restecg`, `slope`, `thal`. Panduan menyebut `cp`, `slope`, `thal` sebagai contoh. `restecg` ikut di-encode karena juga nominal. `ca` dibiarkan numerik karena urutannya bermakna (jumlah pembuluh darah).
- Train-test split 80:20, `stratify=y`, `random_state=42`.
- Imputasi modus untuk `thal` dan `ca` serta StandardScaler untuk `age`, `trestbps`, `chol`, `thalach`, `oldpeak`. Keduanya disusun dalam `ColumnTransformer` supaya fit hanya terjadi di data latih, termasuk di setiap fold saat cross-validation.
- Tampilkan bukti tidak ada leakage: fit di `X_train`, transform di `X_test`, lalu perbandingan statistik sebelum dan sesudah scaling.
- Draf laporan: Metodologi bagian Preprocessing & Feature Scaling.

### Fase 3: Modeling

- Tiga `Pipeline` (preprocessor + model): SVM (`probability=True`), KNN, dan Logistic Regression. Logistic Regression dipilih daripada Naive Bayes karena koefisiennya dipakai untuk interpretasi faktor risiko di Fase 5.
- Model baseline dengan parameter default, dievaluasi dengan Stratified 5-Fold CV di data latih.
- Draf laporan: penjelasan ketiga algoritma.

### Fase 4: Evaluation & Improvement

- GridSearchCV dengan `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`. SVM: `C`, `kernel`, `gamma`. KNN: `n_neighbors`, `weights`. Logistic Regression: `C`.
- Scoring utama `recall` sesuai panduan. Precision dan F1 ikut dicatat. Kalau model terbaik versi recall ternyata degenerate (anti-goal di atas), pindah ke `f1` dan jelaskan alasannya di Markdown.
- Evaluasi di data uji: confusion matrix beserta implikasi klinis, classification report, ROC curve gabungan ketiga model dengan AUC, dan tabel perbandingan.
- Draf laporan: tabel parameter terbaik serta Hasil & Pembahasan. Isi slide: Solusi Model.

### Fase 5: Clinical Insight & Finalisasi

- Faktor risiko dari koefisien Logistic Regression dan Permutation Importance untuk SVM dan KNN, dengan interpretasi medis yang memperhitungkan anomali label.
- Minimal 3 rekomendasi klinis.
- Grafik ringkasan bergaya dashboard dan simulasi skrining beberapa pasien contoh.
- Finalisasi: Run All bersih, susun laporan (DOCX lalu PDF) dan slide (PPTX lalu PDF) dari draf tiap fase.

## Struktur folder baru

```
03-notebook/Kelompok6_Heart_Disease_Prediction.ipynb
05-output/figures/fase-N-<nama-grafik>.png
06-laporan/drafts/fase-N.md        # draf bab laporan + isi slide per fase
06-laporan/Laporan_Heart_Disease.docx / .pdf
06-laporan/Slide_Heart_Disease.pptx / .pdf
requirements.txt
```

Notebook lama `03-notebook/heart-disease-uci-dataset.ipynb` (isinya latihan regression) tidak diubah dan tidak dipakai.

## Diserahkan ke builder

- Gaya visual grafik, misalnya palet warna dan ukuran.
- Rentang nilai grid di luar yang disebut panduan, misalnya nilai `n_neighbors`.
- Cara generate DOCX dan PPTX (misalnya `python-docx` dan `python-pptx`) serta cara konversi ke PDF (Microsoft Office terpasang di laptop user).
- Isi Cover memakai Kelompok 6. Nama anggota dan NIM diisi placeholder sampai user memberikan datanya.
