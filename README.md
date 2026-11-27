# Prediksi Penyakit Jantung (Tugas Kelompok Machine Learning)

Repo ini berisi pengerjaan Kelompok 6 untuk Studi Kasus 4: Diagnosis Penyakit Jantung, mata kuliah Machine Learning. Kita membangun model klasifikasi (SVM, KNN, dan Logistic Regression) untuk menandai pasien berisiko penyakit jantung dari data pemeriksaan medis.

Tugas dikerjakan **fase per fase** sesuai panduan. Tidak ada pembagian tetap siapa mengerjakan fase apa. Siapa pun yang sedang senggang boleh mengambil fase berikutnya yang belum dikerjakan, asalkan mengikuti alur di dokumen ini.

Hasil akhir yang dikumpulkan:

1. Notebook `03-notebook/Kelompok6_Heart_Disease_Prediction.ipynb` yang bisa Run All tanpa error.
2. Laporan Teknis & Bisnis (PDF).
3. Slide presentasi (PPTX/PDF).

## Status Fase

Tabel ini diperbarui setiap kali sebuah fase selesai di-merge ke `main`.

| Fase | Isi | Status |
|---|---|---|
| 1 | Clinical & Data Understanding | Selesai |
| 2 | Data Preparation | Selesai |
| 3 | Modeling (SVM, KNN, Logistic Regression) | Belum dikerjakan |
| 4 | Model Evaluation & Improvement | Belum dikerjakan |
| 5 | Clinical Insight, Laporan, dan Slide | Belum dikerjakan |

## Struktur Folder

```
01-panduan-tugas/      panduan tugas dari dosen (acuan utama: PDF Studi Kasus 4)
02-dataset/            heart_disease.csv (dataset utama, JANGAN diubah)
                       uci-cleveland-asli.data (data UCI asli, dipakai sebagai bukti koreksi label)
03-notebook/           Kelompok6_Heart_Disease_Prediction.ipynb (notebook utama)
04-materi-referensi/   materi kuliah pendukung
05-output/figures/     grafik hasil notebook, nama file: fase-N-<nama>.png
06-laporan/drafts/     draf tulisan laporan dan isi slide per fase: fase-N.md
docs/superpowers/      dokumen desain dan rencana kerja
requirements.txt       daftar library Python beserta versinya
```

File `03-notebook/heart-disease-uci-dataset.ipynb` adalah notebook latihan lama. File ini tidak dipakai dan tidak perlu diubah.

## Menjalankan Project

### 1. Siapkan Python dan library

Project ini memakai Python 3.12. Semua library dipasang di virtual environment `.venv` (folder Python khusus untuk project ini, supaya library-nya tidak bercampur dengan project lain). Pilih salah satu cara berikut, lalu jalankan perintahnya dari folder root repo.

Cara A, kalau kamu punya [uv](https://docs.astral.sh/uv/):

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
```

Cara B, memakai Python bawaan:

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Pengguna macOS atau Linux cukup mengganti `.venv\Scripts\...` dengan `.venv/bin/...`.

### 2. Buka notebook di VS Code

1. Install extension **Jupyter** dan **Python** di VS Code.
2. Buka `03-notebook/Kelompok6_Heart_Disease_Prediction.ipynb`.
3. Klik pilihan kernel di pojok kanan atas, pilih **Select Another Kernel**, lalu **Python Environments**, lalu pilih yang mengarah ke `.venv`.
4. Klik **Run All**.

Notebook harus dijalankan dari folder `03-notebook`, karena dataset dibaca dengan path relatif `../02-dataset/heart_disease.csv`. VS Code sudah otomatis melakukan ini.

### 3. Cek Run All dari terminal (wajib sebelum commit)

Perintah berikut menjalankan seluruh notebook dari awal dengan kernel baru, sama seperti saat dosen menilai:

```bash
cd 03-notebook
../.venv/Scripts/jupyter nbconvert --to notebook --execute --inplace Kelompok6_Heart_Disease_Prediction.ipynb
```

Kalau perintah ini selesai tanpa error, notebook aman di-commit. Kalau ada cell `assert` yang gagal, perintahnya berhenti dan menunjukkan cell yang bermasalah.

### Masalah yang sering muncul

| Pesan error | Penyebab | Solusi |
|---|---|---|
| `Running cells with 'Python 3.x' requires the ipykernel package` | Kernel yang dipilih bukan `.venv` | Pilih kernel `.venv` (langkah 2 di atas) |
| `ModuleNotFoundError: No module named 'sklearn'` | Library belum terpasang di kernel yang dipilih | Pastikan kernel `.venv`, lalu ulangi langkah 1 |
| `FileNotFoundError: ../02-dataset/heart_disease.csv` | Notebook dijalankan dari folder lain | Jalankan dari folder `03-notebook` |

## Hal Penting tentang Dataset

Wajib dibaca sebelum mengerjakan fase mana pun. Detailnya ada di notebook bagian 1.7 sampai 1.10.

1. **Label `target` di file CSV versi Kaggle terbalik.** Setelah 303 baris dicocokkan dengan data UCI asli, semua baris `target = 1` ternyata pasien sehat. Notebook sudah membalik label ini, jadi di variabel `df` nilai `1` benar-benar berarti sakit jantung (138 pasien) dan `0` berarti sehat (164 pasien). Jangan membaca ulang CSV di fase berikutnya. Selalu pakai `df` dari Fase 1.
2. **Kamus data `cp`, `restecg`, `slope`, dan `thal` di halaman Kaggle keliru.** Arti yang benar: `cp` 0 = asymptomatic, 1 = atypical angina, 2 = non-anginal pain, 3 = typical angina; `restecg` 0 = kemungkinan hipertrofi ventrikel kiri, 1 = normal, 2 = kelainan gelombang ST-T; `slope` 0 = downsloping, 1 = flat, 2 = upsloping; `thal` 1 = fixed defect, 2 = normal, 3 = reversible defect.
3. **`thal` dan `ca` punya missing value (NaN).** Ada 2 baris di `thal` dan 4 baris di `ca`. Nilainya sengaja belum diisi. Pengisian dengan modus dilakukan di Fase 2, di dalam pipeline setelah train-test split, supaya tidak terjadi data leakage (informasi data uji bocor ke proses training).
4. File `02-dataset/heart_disease.csv` tidak boleh diubah. Semua koreksi dilakukan lewat kode di notebook.

## Alur Kerja Mengambil dan Mengerjakan Fase

### Aturan dasar

- Fase dikerjakan **berurutan**. Fase N baru boleh dimulai setelah fase N-1 sudah di-merge ke `main`, karena setiap fase memakai variabel dari fase sebelumnya. Contohnya, Fase 3 membutuhkan `X_train` dari Fase 2.
- Dalam satu waktu, hanya **satu orang** yang mengedit notebook. Isi file `.ipynb` berupa JSON, jadi kalau dua orang mengeditnya bersamaan, conflict saat merge akan sangat sulit dibereskan.
- Satu fase dikerjakan di satu branch bernama `fase-N`.

### Langkah mengambil fase

1. Lihat tabel **Status Fase** di `README.md` pada branch `main`. Cari fase pertama yang statusnya "Belum dikerjakan".
2. Cek di GitHub apakah branch `fase-N` sudah ada. Kalau sudah ada, berarti fase itu sedang dikerjakan orang lain.
3. Umumkan di grup kelompok, misalnya: "Aku ambil Fase 2 ya."
4. Buat branch dari `main` terbaru, lalu langsung push. Branch yang sudah ada di GitHub menjadi tanda bahwa fase ini sedang dikerjakan.

```bash
git checkout main
git pull
git checkout -b fase-2
git push -u origin fase-2
```

### Langkah mengerjakan fase

1. Baca bagian fase tersebut di PDF panduan, lalu baca **Checklist per Fase** di bawah.
2. Tambahkan section baru di notebook, **tepat di bawah section fase sebelumnya**, dengan heading `## Fase N: <judul>`.
3. Commit sedikit demi sedikit, setiap kali satu bagian selesai dan sudah dicek Run All.
4. Tulis draf laporan dan isi slide fase tersebut di `06-laporan/drafts/fase-N.md`.
5. Ubah status fase di tabel README menjadi "Selesai".
6. Buka Pull Request dari `fase-N` ke `main`, lalu minta satu anggota lain untuk review.
7. Setelah disetujui, merge memakai opsi **Create a merge commit**. Jangan pakai "Squash and merge", supaya riwayat commit per fase tetap terlihat.
8. Umumkan di grup bahwa fase N sudah selesai. Fase berikutnya sekarang boleh diambil.

### Kalau menemukan bug di fase sebelumnya

Jangan diam-diam mengubah section fase lain. Perbaiki di commit terpisah dengan scope fase yang bermasalah, misalnya `fix(fase-1): ...`, lalu jelaskan perubahannya di deskripsi Pull Request dan kabari grup.

## Aturan Commit

Commit message ditulis dalam **bahasa Indonesia** dengan format:

```
<tipe>(<scope>): <deskripsi singkat>
```

| Tipe | Dipakai untuk |
|---|---|
| `feat` | Menambah isi baru: cell analisis, model, grafik |
| `fix` | Memperbaiki kesalahan |
| `docs` | Draf laporan, slide, README, dokumentasi |
| `chore` | Hal teknis lain, misalnya update `requirements.txt` |

Scope diisi `fase-1` sampai `fase-5`. Untuk persiapan project dipakai `setup`, dan untuk finalisasi dipakai `final`.

Contoh:

```
feat(fase-2): tambah one-hot encoding untuk cp, restecg, slope, dan thal
feat(fase-2): tambah train-test split 80:20 dengan stratify
fix(fase-2): pindahkan imputasi modus ke dalam pipeline
docs(fase-2): tambah draf bab metodologi preprocessing
```

## Aturan Menulis di Notebook

- Setiap langkah diawali cell Markdown yang menjelaskan apa yang dilakukan dan **kenapa**. Dosen menilai kerapian dan dokumentasi, bukan hanya kode yang jalan.
- Semua Markdown ditulis dalam bahasa Indonesia. Istilah teknis yang umum dipakai dalam bahasa Inggris (misalnya *train-test split*, *recall*, *pipeline*) tidak perlu diterjemahkan.
- Penomoran subbagian melanjutkan pola yang ada: Fase 2 memakai 2.1, 2.2, dan seterusnya.
- Section setiap fase **ditutup dengan cell "Pengecekan otomatis"** berisi `assert` untuk hasil terpenting fase itu. Contoh dari Fase 1:

  ```python
  assert df.shape == (302, 14)
  assert df["target"].value_counts().to_dict() == {0: 164, 1: 138}
  print("Pengecekan pembersihan data: OK")
  ```

  Tujuannya, kalau ada perubahan yang tidak sengaja merusak hasil, Run All langsung gagal di cell itu dan tidak lolos sampai ke laporan.
- Semua proses yang memakai angka acak wajib memakai `random_state=RANDOM_STATE` (nilainya 42).
- Grafik disimpan dengan `simpan_grafik(fig, "fase-N-<nama>")`, supaya file PNG-nya masuk ke `05-output/figures` dan bisa dipakai di laporan dan slide.
- Angka yang tampil di grafik memakai koma desimal, sama seperti laporan. Pakai fungsi bantuan dari Fase 1 (lihat tabel di bawah).
- Sebelum commit, jalankan cek Run All dari terminal. Jangan commit notebook yang masih menyimpan output error.

### Variabel dan fungsi dari Fase 1 yang bisa langsung dipakai

| Nama | Isi |
|---|---|
| `df` | Data bersih: 302 baris, 14 kolom, label `target` sudah dikoreksi, `thal`/`ca` berisi NaN |
| `df_raw` | Data mentah dari CSV (jangan diubah) |
| `RANDOM_STATE` | `42` |
| `FIG_DIR` | Folder `05-output/figures` |
| `simpan_grafik(fig, nama)` | Simpan grafik ke `FIG_DIR/<nama>.png` |
| `angka_id(nilai, desimal)` | `angka_id(54.321, 1)` menghasilkan `"54,3"` |
| `persen_id(proporsi, desimal)` | `persen_id(0.543, 1)` menghasilkan `"54,3%"` |
| `format_sumbu_id(ax.yaxis, desimal, persen)` | Mengubah label angka di sumbu grafik agar memakai koma |
| `LABEL_TARGET`, `WARNA_TARGET` | Label "Sehat (0)" dan "Sakit Jantung (1)" beserta warnanya |

### Variabel dari Fase 2 yang bisa langsung dipakai

| Nama | Isi |
|---|---|
| `X`, `y` | 13 fitur dan label dari `df` (302 pasien) |
| `X_train`, `y_train` | 241 pasien data latih (110 sakit) |
| `X_test`, `y_test` | 61 pasien data uji (28 sakit). Hanya dipakai untuk evaluasi akhir di Fase 4 |
| `preprocessor` | `ColumnTransformer` yang **belum di-fit**: scaling, imputasi, dan One-Hot Encoding. Hasilnya 22 fitur. Pasang di awal `Pipeline` setiap model, jangan di-fit manual |
| `KOLOM_NUMERIK`, `KOLOM_NOMINAL`, `KOLOM_ORDINAL`, `KOLOM_BINER` | Daftar kolom per kelompok |

## Checklist per Fase

Checklist ini merangkum panduan dosen dan keputusan kelompok. Kalau ada yang berbeda, PDF panduan tetap menjadi acuan utama.

### Fase 1: Clinical & Data Understanding (selesai)

- [x] Clinical understanding: masalah klinis, tujuan, dan alasan Recall kelas 1 jadi metrik utama
- [x] `df.info()`, `df.describe()`, penanganan duplikat dan missing value
- [x] Koreksi label `target` dan kamus data dengan bukti dari data UCI asli
- [x] EDA: distribusi target, `cp` vs target, `age`/`trestbps`/`chol`/`thalach` vs target, heatmap korelasi
- [x] Draf laporan `06-laporan/drafts/fase-1.md`

### Fase 2: Data Preparation (selesai)

- [x] Pisahkan fitur dan target: `X = df.drop(columns="target")`, `y = df["target"]`
- [x] Train-test split 80:20 dengan `stratify=y` dan `random_state=RANDOM_STATE`
- [x] One-Hot Encoding untuk fitur nominal `cp`, `restecg`, `slope`, `thal`. `ca` tetap numerik karena nilainya berurutan.
- [x] Imputasi modus untuk `thal` dan `ca`
- [x] StandardScaler untuk `age`, `trestbps`, `chol`, `thalach`, `oldpeak`
- [x] Susun encoding, imputasi, dan scaling dalam `ColumnTransformer`, supaya bisa dimasukkan ke `Pipeline` di Fase 3 dan fit hanya terjadi di data latih
- [x] Tunjukkan bukti tidak ada data leakage: fit di `X_train`, transform di `X_test`, lalu bandingkan statistik sebelum dan sesudah scaling
- [x] Draf laporan: Metodologi bagian Preprocessing & Feature Scaling

### Fase 3: Modeling

- [ ] Tiga `Pipeline` (preprocessor dari Fase 2 + model): SVM (`probability=True`), KNN, Logistic Regression
- [ ] Latih model baseline dengan parameter default, lalu evaluasi dengan `StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)` di data latih
- [ ] Draf laporan: penjelasan ketiga algoritma

### Fase 4: Model Evaluation & Improvement

- [ ] GridSearchCV: SVM (`C`, `kernel`, `gamma`), KNN (`n_neighbors`, `weights`), Logistic Regression (`C`)
- [ ] Scoring utama `recall`. Cek juga precision. Kalau model terbaik ternyata menebak hampir semua pasien sebagai sakit (precision anjlok), ganti ke `f1` dan jelaskan alasannya di Markdown.
- [ ] Evaluasi di data uji: confusion matrix beserta implikasi klinisnya, classification report (fokus Recall kelas 1), ROC curve gabungan ketiga model dengan AUC
- [ ] Tabel perbandingan ketiga model dan tabel parameter terbaik
- [ ] Draf laporan: Hasil & Pembahasan

### Fase 5: Clinical Insight dan Finalisasi

- [ ] Faktor risiko utama dari koefisien Logistic Regression dan Permutation Importance untuk SVM/KNN, beserta interpretasi medisnya
- [ ] Minimal 3 rekomendasi klinis untuk rumah sakit
- [ ] Grafik ringkasan bergaya dashboard dan simulasi skrining beberapa pasien contoh
- [ ] Pastikan Run All bersih di notebook final `Kelompok6_Heart_Disease_Prediction.ipynb`
- [ ] Susun laporan (DOCX lalu PDF) dan slide (PPTX lalu PDF) dari semua file `06-laporan/drafts/fase-N.md`

## Aturan Draf Laporan

Setiap fase menulis satu file `06-laporan/drafts/fase-N.md`. File ini nanti digabung menjadi laporan dan slide di Fase 5. Ikuti pola `fase-1.md`:

- Heading level 2 untuk bab laporan, misalnya `## Bab 2 Metodologi: Preprocessing`.
- Bagian `## Isi Slide` berisi poin slide (maksimal 4 poin per slide) dan nama file gambar yang dipakai.
- Bagian `## Referensi Sementara` untuk sumber yang dikutip.
- Setiap angka di draf harus sama dengan output notebook terakhir. Kalau notebook berubah, cek ulang angkanya.

## Kalau Ada Pertanyaan

Tanyakan di grup sebelum mengambil keputusan yang mengubah hasil fase lain, misalnya mengganti cara encoding atau mengganti metrik utama. Keputusan desain yang sudah disepakati tercatat di `docs/superpowers/specs/2026-11-26-heart-disease-phases-design.md`.
