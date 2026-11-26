# Draf Laporan dan Slide: Fase 1 (Clinical & Data Understanding)

Draf ini adalah bahan untuk laporan PDF dan slide presentasi. Semua angka diambil dari output notebook `03-notebook/KelompokX_Heart_Disease_Prediction.ipynb` (section Fase 1). Kalau notebook berubah, angka di sini perlu dicek ulang.

## Bab 1 Pendahuluan

### 1.1 Latar Belakang

Penyakit kardiovaskular, termasuk penyakit jantung koroner, merupakan penyebab kematian nomor satu di dunia. Menurut WHO, penyakit ini menyebabkan sekitar 17,9 juta kematian setiap tahun. Salah satu tantangannya, penyakit jantung sering tidak menimbulkan gejala yang jelas pada tahap awal. Banyak pasien baru terdiagnosis ketika kondisinya sudah serius, sehingga penanganannya lebih sulit dan lebih mahal.

Di sisi lain, rumah sakit sudah mengumpulkan banyak data pemeriksaan rutin, seperti usia, tekanan darah, kadar kolesterol, hasil elektrokardiogram (EKG), dan hasil tes latihan jantung. Data ini dapat dimanfaatkan untuk membangun sistem pendukung keputusan klinis (*Clinical Decision Support System*) berbasis *machine learning*. Sistem seperti ini dapat membantu dokter menandai pasien yang berisiko tinggi sejak awal, sehingga pasien tersebut bisa diprioritaskan untuk pemeriksaan lanjutan.

Dalam sistem skrining medis, kesalahan prediksi tidak memiliki dampak yang sama. Pasien sakit yang diprediksi sehat (*False Negative*) dapat pulang tanpa penanganan dan berisiko fatal. Sebaliknya, pasien sehat yang diprediksi sakit (*False Positive*) hanya perlu menjalani pemeriksaan tambahan. Karena itu, penelitian ini menekankan kemampuan model dalam mendeteksi pasien sakit, yang diukur dengan metrik *Recall* kelas positif.

### 1.2 Rumusan Masalah

1. Bagaimana karakteristik data pemeriksaan medis pasien pada Heart Disease UCI Dataset, dan fitur apa saja yang membedakan pasien sakit jantung dari pasien sehat?
2. Bagaimana performa algoritma *Support Vector Machine* (SVM), *K-Nearest Neighbors* (KNN), dan *Logistic Regression* dalam mengklasifikasikan pasien penyakit jantung, terutama dari sisi *Recall* kelas sakit jantung?
3. Faktor risiko apa yang paling memengaruhi prediksi model, dan rekomendasi klinis apa yang dapat diberikan kepada rumah sakit?

### 1.3 Tujuan

1. Melakukan eksplorasi data untuk memahami pola dan kualitas data pemeriksaan medis pasien.
2. Membangun dan membandingkan tiga model klasifikasi (SVM, KNN, dan *Logistic Regression*) dengan *feature scaling* dan *hyperparameter tuning*.
3. Mengevaluasi model menggunakan *Confusion Matrix*, *Classification Report*, *K-Fold Cross-Validation*, dan ROC-AUC, dengan fokus pada *Recall* kelas sakit jantung.
4. Mengidentifikasi faktor risiko utama dan menyusun rekomendasi sistem skrining awal bagi rumah sakit.

### 1.4 Manfaat

- Bagi rumah sakit: memberikan gambaran sistem skrining awal yang dapat membantu dokter memprioritaskan pasien berisiko tinggi.
- Bagi tenaga medis: memberikan informasi tentang fitur pemeriksaan yang paling berkaitan dengan diagnosis penyakit jantung.
- Bagi mahasiswa: menerapkan tahapan *supervised learning* secara lengkap pada studi kasus medis, mulai dari pemahaman data sampai interpretasi hasil.

## Bab 2 Metodologi: Data dan EDA

### 2.1 Deskripsi Dataset

Penelitian ini menggunakan Heart Disease UCI Dataset versi Kaggle. Dataset ini berisi 303 baris data pasien dan 14 kolom, terdiri atas 13 fitur hasil pemeriksaan dan 1 kolom target.

| Kolom | Arti medis | Tipe |
|---|---|---|
| `age` | Usia pasien (tahun) | Numerik |
| `sex` | Jenis kelamin (1 = laki-laki, 0 = perempuan) | Biner |
| `cp` | Tipe nyeri dada (0 = typical angina, 1 = atypical angina, 2 = non-anginal pain, 3 = asymptomatic) | Nominal |
| `trestbps` | Tekanan darah saat istirahat (mm Hg) | Numerik |
| `chol` | Kadar kolesterol serum (mg/dl) | Numerik |
| `fbs` | Gula darah puasa > 120 mg/dl (1 = ya, 0 = tidak) | Biner |
| `restecg` | Hasil EKG saat istirahat (0, 1, 2) | Nominal |
| `thalach` | Detak jantung maksimum saat tes latihan | Numerik |
| `exang` | Angina yang dipicu olahraga (1 = ya, 0 = tidak) | Biner |
| `oldpeak` | Depresi segmen ST saat latihan dibanding saat istirahat | Numerik |
| `slope` | Kemiringan segmen ST pada puncak latihan (0, 1, 2) | Nominal |
| `ca` | Jumlah pembuluh darah utama yang terlihat lewat fluoroskopi (0 sampai 3) | Ordinal |
| `thal` | Hasil tes thalium (1 = normal, 2 = fixed defect, 3 = reversible defect) | Nominal |
| `target` | Diagnosis (1 = ada penyakit jantung, 0 = tidak ada) | Target |

### 2.2 Pemeriksaan Kualitas Data

Inspeksi awal dengan `df.info()` dan `df.describe()` menunjukkan bahwa semua kolom bertipe numerik dan tidak ada sel yang kosong. Meski begitu, ada dua masalah kualitas data yang ditemukan:

1. **Data duplikat.** Terdapat 1 baris duplikat (baris 163 dan 164 identik di seluruh 14 kolom). Baris duplikat dihapus agar pasien yang sama tidak terhitung dua kali, sehingga data menjadi 302 baris.
2. **Missing value tersembunyi.** Kolom `thal` memiliki kode 0 sebanyak 2 baris, dan kolom `ca` memiliki kode 4 sebanyak 4 baris (setelah duplikat dihapus). Kedua kode ini tidak ada dalam definisi dataset UCI asli, dan pada dataset asli merupakan data yang hilang. Kode tersebut diubah menjadi nilai kosong (NaN). Pengisiannya dilakukan pada tahap *preprocessing* dengan modus yang dihitung hanya dari data latih, untuk mencegah *data leakage*.

### 2.3 Exploratory Data Analysis

**Distribusi kelas target** (Gambar `05-output/figures/fase-1-distribusi-target.png`). Dari 302 pasien, 164 pasien (54,3%) tergolong sakit jantung dan 138 pasien (45,7%) tergolong sehat. Kedua kelas cukup seimbang, sehingga teknik penyeimbangan data seperti SMOTE tidak diperlukan. Pembagian data latih dan data uji tetap dilakukan secara *stratified* agar proporsi kelas terjaga.

**Tipe nyeri dada terhadap diagnosis** (Gambar `05-output/figures/fase-1-cp-vs-target.png`). Hanya 27% pasien dengan `cp = 0` yang tergolong sakit jantung. Pada tipe nyeri dada lainnya, proporsi pasien sakit jantung jauh lebih tinggi, yaitu 82% (`cp = 1`), 79% (`cp = 2`), dan 70% (`cp = 3`). Hal ini menunjukkan bahwa tipe nyeri dada adalah fitur yang kuat untuk membedakan kedua kelas.

**Distribusi fitur numerik** (Gambar `05-output/figures/fase-1-numerik-vs-target.png`).

| Fitur | Median kelas sehat | Median kelas sakit jantung |
|---|---|---|
| `age` | 58 tahun | 52 tahun |
| `trestbps` | 130 mm Hg | 130 mm Hg |
| `chol` | 249 mg/dl | 234,5 mg/dl |
| `thalach` | 142 | 161 |

Detak jantung maksimum (`thalach`) adalah fitur numerik yang paling jelas memisahkan kedua kelas. Sebaliknya, distribusi tekanan darah (`trestbps`) dan kolesterol (`chol`) hampir tumpang tindih antara kedua kelas.

**Korelasi antar fitur** (Gambar `05-output/figures/fase-1-korelasi.png`). Lima fitur dengan korelasi terkuat terhadap `target` adalah `ca` (-0,46), `exang` (-0,44), `cp` (0,43), `oldpeak` (-0,43), dan `thalach` (0,42). Kolesterol (-0,08) dan tekanan darah (-0,15) memiliki korelasi lemah. Temuan ini berbeda dengan dugaan awal pada panduan tugas bahwa kolesterol dan tekanan darah adalah prediktor terkuat, sehingga akan diuji kembali melalui analisis faktor risiko dari model. Korelasi antar fitur tertinggi adalah `oldpeak` dengan `slope` (-0,58). Tidak ada pasangan fitur dengan korelasi di atas 0,8, sehingga seluruh fitur dipertahankan.

### 2.4 Catatan Anomali Label Target

| Indikator | Rata-rata kelas 0 | Rata-rata kelas 1 |
|---|---|---|
| `thalach` | 139,10 | 158,38 |
| `oldpeak` | 1,59 | 0,59 |
| `exang` | 0,55 | 0,14 |

Secara medis, pasien penyakit jantung umumnya memiliki detak jantung maksimum yang lebih rendah, depresi ST yang lebih tinggi, dan lebih sering mengalami angina saat olahraga. Data ini menunjukkan pola yang berkebalikan pada kelas `target = 1`. Anomali ini dikenal pada Heart Disease UCI versi Kaggle, dan diduga berasal dari perbedaan pengodean label dengan dataset UCI asli. Penelitian ini tetap memakai definisi label dari panduan tugas (`1` = ada penyakit jantung) agar evaluasi model sesuai dengan kriteria penilaian. Dampak anomali ini terhadap interpretasi faktor risiko dibahas pada bab Hasil dan Pembahasan.

## Isi Slide

### Slide: Masalah Klinis

- Penyakit kardiovaskular adalah penyebab kematian nomor satu di dunia (sekitar 17,9 juta kematian per tahun menurut WHO).
- Gejala awal sering tidak jelas, sehingga banyak pasien terlambat terdiagnosis.
- Tujuan: model skrining awal untuk menandai pasien berisiko tinggi dari data pemeriksaan rutin.
- Metrik utama: *Recall* kelas sakit jantung, karena *False Negative* (pasien sakit dianggap sehat) paling berbahaya.

Gambar: tidak ada (slide teks).

### Slide: Insight EDA

- Data: 302 pasien setelah 1 duplikat dihapus. Kelas cukup seimbang (54% sakit, 46% sehat).
- Tipe nyeri dada (`cp`) dan detak jantung maksimum (`thalach`) paling jelas membedakan pasien sakit dan sehat.
- Kolesterol dan tekanan darah ternyata hanya berkorelasi lemah dengan diagnosis (-0,08 dan -0,15).
- Ada anomali label pada dataset versi Kaggle. Label tetap mengikuti panduan, dan dampaknya dibahas di bagian interpretasi.

Gambar: `fase-1-cp-vs-target.png` dan `fase-1-korelasi.png` (atau `fase-1-numerik-vs-target.png` jika ruang slide terbatas).

## Referensi Sementara

- Detrano, R., Janosi, A., Steinbrunn, W., Pfisterer, M., Schmid, J., Sandhu, S., Guppy, K., Lee, S., & Froelicher, V. (1989). International application of a new probability algorithm for the diagnosis of coronary artery disease. *American Journal of Cardiology, 64*(5), 304-310.
- Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1988). Heart Disease [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C52P4X
- World Health Organization. (2021). Cardiovascular diseases (CVDs) [Fact sheet]. https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds)
