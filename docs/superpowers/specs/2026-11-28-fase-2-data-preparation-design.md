# Desain Singkat: Fase 2 (Data Preparation)

Tanggal: 2026-11-28
Induk: `docs/superpowers/specs/2026-11-26-heart-disease-phases-design.md` (bagian "Fase 2: Data Preparation")

## Tujuan

Menyiapkan `df` hasil Fase 1 menjadi data yang siap dipakai SVM, KNN, dan Logistic Regression di Fase 3, tanpa data leakage. Fase ini juga menghasilkan draf laporan bagian Metodologi Preprocessing beserta isi slide.

## Goals

- Variabel untuk Fase 3: `X`, `y`, `X_train`, `X_test`, `y_train`, `y_test`, daftar kolom per jenis, dan `preprocessor` (ColumnTransformer yang **belum** di-fit).
- Bukti tertulis bahwa fit hanya dilakukan di data latih, sesuai permintaan panduan.
- Dua grafik (`fase-2-split.png`, `fase-2-scaling.png`) dan draf `06-laporan/drafts/fase-2.md`.

## Non-goals

- Melatih model. Itu bagian Fase 3.
- SMOTE atau teknik penyeimbangan kelas, karena kelasnya cukup seimbang (54:46).
- Feature selection. Semua 13 fitur dipertahankan sesuai temuan EDA.

## Anti-goals

- `fit` preprocessing memakai data uji atau seluruh data.
- `preprocessor` yang diserahkan ke Fase 3 sudah dalam kondisi fit, karena preprocessor harus di-fit ulang di setiap fold cross-validation.
- Kode yang membaca ulang CSV, sehingga koreksi label dari Fase 1 hilang.

## Pendekatan yang dipertimbangkan

| Pendekatan | Kelebihan | Kekurangan |
|---|---|---|
| A. `ColumnTransformer` yang belum di-fit, dipakai di dalam `Pipeline` di Fase 3 (dipilih) | Aturan fit hanya di data latih terjaga otomatis, termasuk di setiap fold CV | Hasil transform tidak disimpan sebagai variabel tetap, jadi harus ada cell demo terpisah untuk bukti |
| B. Transform manual sekali (`X_train_prep`, `X_test_prep`), lalu dipakai langsung di Fase 3 | Sederhana untuk dibaca | Saat GridSearchCV, scaler sudah di-fit di seluruh data latih, sehingga fold validasi bocor |
| C. `pd.get_dummies` dan scaling manual dengan pandas | Tidak perlu belajar API scikit-learn | Rawan leakage dan kategori yang tidak muncul di data uji membuat kolom tidak sama |

Pendekatan A sudah ditetapkan di dokumen desain induk dan disetujui user.

## Keputusan

- Split 80:20, `stratify=y`, `random_state=RANDOM_STATE`. Hasil yang diharapkan: 241 data latih (110 sakit) dan 61 data uji (28 sakit).
- Kelompok kolom:
  - Numerik: `age`, `trestbps`, `chol`, `thalach`, `oldpeak`, diproses dengan StandardScaler.
  - Nominal: `cp`, `restecg`, `slope`, `thal`, diproses dengan `SimpleImputer(strategy="most_frequent")` lalu `OneHotEncoder(handle_unknown="ignore")`.
  - Ordinal: `ca`, diproses dengan `SimpleImputer(strategy="most_frequent")` dan tetap berupa angka.
  - Biner: `sex`, `fbs`, `exang`, dilewatkan apa adanya.
- Total fitur setelah transform: 5 + 13 + 1 + 3 = 22 kolom.
- Bukti tanpa leakage: `clone(preprocessor)` di-fit di `X_train`, lalu transform `X_train` dan `X_test`. Tampilkan mean dan std kolom numerik sebelum dan sesudah. Data latih harus bermean 0 (toleransi 1e-9), data uji tidak harus.
- Sebelum Fase 2 dimulai, ada dua commit `fix(fase-1)`: arti kode `restecg` dan `slope` di kamus data (hasil pencocokan dengan UCI asli), dan label sumbu persen di grafik `cp` tanpa desimal.

## Diserahkan ke builder

- Isi detail Markdown penjelasan dan gaya grafik (mengikuti gaya Fase 1).
- Bentuk tabel perbandingan sebelum dan sesudah scaling.
