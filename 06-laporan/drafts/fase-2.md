# Draf Laporan dan Slide: Fase 2 (Data Preparation)

Draf ini adalah bahan untuk laporan PDF dan slide presentasi. Semua angka diambil dari output notebook `03-notebook/Kelompok6_Heart_Disease_Prediction.ipynb` (section Fase 2). Kalau notebook berubah, angka di sini perlu dicek ulang.

## Bab 2 Metodologi: Data Preparation

### 2.6 Pemisahan Fitur dan Target

Data hasil pembersihan (302 pasien) dipisahkan menjadi 13 fitur (`X`) dan 1 label (`y`). Ke-13 fitur dikelompokkan berdasarkan jenisnya, karena setiap jenis membutuhkan perlakuan yang berbeda:

| Kelompok | Fitur | Perlakuan |
|---|---|---|
| Numerik | `age`, `trestbps`, `chol`, `thalach`, `oldpeak` | StandardScaler |
| Nominal | `cp`, `restecg`, `slope`, `thal` | Imputasi modus, lalu One-Hot Encoding |
| Ordinal | `ca` | Imputasi modus, tetap berupa angka |
| Biner | `sex`, `fbs`, `exang` | Tidak diubah |

### 2.7 Pembagian Data Latih dan Data Uji

Data dibagi dengan rasio 80:20 memakai `train_test_split` dengan parameter `stratify=y` dan `random_state=42`. Rasio 80:20 dipilih agar model mendapat data latih sebanyak mungkin dari dataset yang berukuran kecil. Parameter `stratify` menjaga proporsi pasien sakit tetap sama di kedua bagian, dan `random_state` membuat pembagian dapat direplikasi (Gambar `05-output/figures/fase-2-split.png`).

| Bagian | Jumlah pasien | Pasien sakit | Proporsi sakit |
|---|---|---|---|
| Keseluruhan | 302 | 138 | 45,7% |
| Data latih | 241 | 110 | 45,6% |
| Data uji | 61 | 28 | 45,9% |

### 2.8 Encoding, Imputasi, dan Feature Scaling

Fitur nominal diubah dengan **One-Hot Encoding**, yaitu mengubah satu kolom kategori menjadi beberapa kolom bernilai 0 atau 1, satu kolom untuk setiap kategori. Langkah ini diperlukan karena kode kategori seperti `cp = 0` sampai `cp = 3` tidak memiliki urutan. Tanpa encoding, algoritma akan menganggap kode yang lebih besar memiliki nilai yang lebih besar. Encoder diatur dengan `handle_unknown="ignore"`, sehingga kategori yang tidak dikenal tidak menyebabkan error. Kondisi ini bisa terjadi pada kategori langka seperti `restecg = 2` (hanya 4 pasien), yang mungkin tidak muncul di bagian latih salah satu fold saat cross-validation, atau pada pasien baru dengan kode di luar rentang dataset.

Missing value pada `thal` (2 baris) dan `ca` (4 baris) diisi dengan **modus** menggunakan `SimpleImputer`. Modus yang dipelajari dari data latih adalah `thal = 2` (normal) dan `ca = 0`.

Fitur numerik distandarkan dengan **StandardScaler**:

$$z = \frac{x - \mu}{\sigma}$$

dengan $\mu$ adalah rata-rata dan $\sigma$ adalah standard deviation fitur di data latih. Feature scaling wajib dilakukan karena SVM dan KNN menghitung jarak atau margin antar data, sehingga fitur dengan satuan besar (misalnya `chol` dalam ratusan mg/dl) akan mendominasi fitur dengan satuan kecil (misalnya `oldpeak`).

Setelah preprocessing, jumlah fitur menjadi 22: 5 fitur numerik, 13 kolom hasil One-Hot Encoding (`cp` 4 kolom, `restecg`, `slope`, dan `thal` masing-masing 3 kolom), 1 fitur ordinal, dan 3 fitur biner.

### 2.9 Pencegahan Data Leakage

Seluruh langkah preprocessing disusun dalam satu `ColumnTransformer`. Statistik yang dibutuhkan (mean, standard deviation, dan modus) dihitung hanya dari data latih melalui `fit()`, kemudian diterapkan ke data uji melalui `transform()`. Hasilnya ditunjukkan pada tabel berikut:

| Fitur | Mean latih sebelum | Std latih sebelum | Mean latih sesudah | Std latih sesudah | Mean uji sesudah | Std uji sesudah |
|---|---|---|---|---|---|---|
| `age` | 54,52 | 8,99 | 0,00 | 1,00 | -0,06 | 1,02 |
| `trestbps` | 131,26 | 17,55 | 0,00 | 1,00 | 0,10 | 0,99 |
| `chol` | 248,88 | 53,33 | 0,00 | 1,00 | -0,22 | 0,81 |
| `thalach` | 149,93 | 21,93 | 0,00 | 1,00 | -0,08 | 1,19 |
| `oldpeak` | 1,06 | 1,16 | 0,00 | 1,00 | -0,08 | 0,98 |

Data latih memiliki mean tepat 0 dan standard deviation tepat 1 setelah scaling, sedangkan data uji hanya mendekati nilai tersebut karena di-transform memakai statistik data latih. Nilai mean data uji yang tidak tepat 0 merupakan akibat dari cara transform tersebut, tetapi belum menjadi bukti, karena scaler yang di-fit dengan seluruh data juga menghasilkan mean data uji yang tidak tepat 0 (Gambar `05-output/figures/fase-2-scaling.png`).

Bukti tidak adanya kebocoran diperoleh dari parameter scaler itu sendiri:

| Fitur | $\mu$ scaler | Mean data latih | Mean seluruh data | $\sigma$ scaler | Std data latih | Std seluruh data |
|---|---|---|---|---|---|---|
| `age` | 54,52 | 54,52 | 54,42 | 8,99 | 8,99 | 9,03 |
| `trestbps` | 131,26 | 131,26 | 131,60 | 17,55 | 17,55 | 17,53 |
| `chol` | 248,88 | 248,88 | 246,50 | 53,33 | 53,33 | 51,67 |
| `thalach` | 149,93 | 149,93 | 149,57 | 21,93 | 21,93 | 22,87 |
| `oldpeak` | 1,06 | 1,06 | 1,04 | 1,16 | 1,16 | 1,16 |

Nilai $\mu$ dan $\sigma$ yang disimpan scaler identik dengan statistik data latih dan berbeda dengan statistik seluruh data. Hal ini membuktikan bahwa scaler hanya belajar dari 241 pasien data latih, sehingga informasi dari 61 pasien data uji tidak ikut digunakan dalam proses preprocessing.

Pada tahap pemodelan, `ColumnTransformer` ini dipasang di awal `Pipeline` setiap model. Dengan cara ini, saat K-Fold Cross-Validation statistik preprocessing dihitung ulang di setiap fold hanya dari bagian data latih fold tersebut, sehingga kebocoran data juga tercegah selama proses tuning.

## Isi Slide

### Slide: Persiapan Data

- Data dibagi 80:20 secara stratified: 241 pasien data latih dan 61 pasien data uji, dengan proporsi pasien sakit sekitar 46% di keduanya.
- Fitur kategori diubah dengan One-Hot Encoding, dan missing value `thal` serta `ca` diisi modus.
- Fitur numerik distandarkan dengan StandardScaler, sehingga total ada 22 fitur.
- Semua statistik preprocessing dihitung dari data latih saja untuk mencegah data leakage.

Gambar: `fase-2-split.png` dan `fase-2-scaling.png`.

## Referensi Sementara

- Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, E. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research, 12*, 2825-2830.
- scikit-learn developers. (2026). User Guide: Preprocessing data dan Column Transformer with Mixed Types. https://scikit-learn.org/stable/modules/preprocessing.html
