# Fase 2 (Data Preparation) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Membereskan dua perbaikan kecil Fase 1, lalu menambah section Fase 2 di notebook: split data, `preprocessor` yang belum di-fit untuk Fase 3, bukti tanpa leakage, dua grafik, dan draf laporan.

**Architecture:** Lanjutan notebook yang sama (`03-notebook/Kelompok6_Heart_Disease_Prediction.ipynb`). Section baru `## Fase 2: Data Preparation` ditambahkan setelah cell pengecekan terakhir Fase 1. Test berupa cell `assert` di akhir setiap bagian, dijalankan lewat `jupyter nbconvert --execute`. Edit notebook dilakukan dengan menyisipkan atau mengganti cell lewat `nbformat` (bukan membangun ulang notebook), supaya id cell Fase 1 tidak berubah.

**Tech Stack:** Python 3.12, pandas, NumPy, scikit-learn 1.9.1 (`train_test_split`, `ColumnTransformer`, `Pipeline`, `SimpleImputer`, `OneHotEncoder`, `StandardScaler`, `clone`), Matplotlib, Seaborn.

**Spec:** `docs/superpowers/specs/2026-11-28-fase-2-data-preparation-design.md` (induk: `docs/superpowers/specs/2026-11-26-heart-disease-phases-design.md`)

## Global Constraints

- Commit message bahasa Indonesia, format `<tipe>(<scope>): <deskripsi>`, scope `fase-1` untuk perbaikan Fase 1 dan `fase-2` untuk pekerjaan Fase 2. **Tanpa** trailer `Co-Authored-By`.
- Markdown, draf, dan README berbahasa Indonesia, tanpa karakter em dash atau en dash.
- Angka di grafik memakai koma desimal lewat `angka_id`, `persen_id`, dan `format_sumbu_id` dari Fase 1.
- Grafik disimpan lewat `simpan_grafik(fig, "fase-2-<nama>")`.
- `random_state=RANDOM_STATE` (42) untuk semua proses acak.
- Fase 2 memakai `df` dari Fase 1. Dilarang membaca ulang CSV.
- Perintah test (dari root repo): `cd 03-notebook && ../.venv/Scripts/jupyter-nbconvert --to notebook --execute --inplace --ClearMetadataPreprocessor.enabled=True --ClearMetadataPreprocessor.clear_notebook_metadata=False Kelompok6_Heart_Disease_Prediction.ipynb`, lalu pastikan `nb.metadata.kernelspec.name == "python3"`.

## Review Focus

- Pasien baru atau data uji yang punya kategori yang tidak pernah muncul di data latih (misalnya `restecg = 2`, hanya 4 pasien) tidak boleh membuat transform error (Task 4, test transform baris dengan kategori tidak dikenal).
- Cell bukti leakage tidak boleh membuat `preprocessor` milik Fase 3 ikut ter-fit (Task 4, assert `preprocessor` belum punya atribut `transformers_` setelah demo).
- Anggota yang menjalankan ulang cell split berkali-kali harus mendapat pembagian yang sama (Task 3, assert jumlah sakit 110 dan 28 dan index data uji stabil lewat `random_state`).
- Label Fase 1 harus tetap terkoreksi di Fase 2 (Task 3, assert `y.sum() == 138`).
- Hasil transform tidak boleh masih berisi NaN di `thal` atau `ca` (Task 4, assert tidak ada NaN di hasil transform data latih dan uji).

---

### Task 1: Lengkapi arti kode restecg dan slope (fix fase-1)

**Files:**
- Modify: notebook, cell Markdown kamus data (bagian 1.4), cell kode pencocokan kategori (bagian 1.8) dan cell Markdown temuannya, cell pengecekan pembersihan data

**Interfaces:**
- Consumes: `cocok` (DataFrame hasil merge Fase 1, kolom `restecg_kaggle`, `restecg_uci`, `slope_kaggle`, `slope_uci`)

- [ ] **Step 1: Tambah assert pemetaan ke cell pengecekan pembersihan data**

```python
peta = lambda k: cocok.groupby(f"{k}_kaggle")[f"{k}_uci"].unique().map(list).to_dict()
assert peta("restecg") == {0: [2.0], 1: [0.0], 2: [1.0]}
assert peta("slope") == {0: [3.0], 1: [2.0], 2: [1.0]}
```

- [ ] **Step 2: Jalankan perintah test.** Expected: lulus (ini mengunci fakta data yang sudah ada, perubahan utamanya dokumentasi).

- [ ] **Step 3: Perbarui dokumentasi**

Kamus 1.4: `restecg` = "0 = kemungkinan hipertrofi ventrikel kiri, 1 = normal, 2 = kelainan gelombang ST-T", `slope` = "0 = downsloping, 1 = flat, 2 = upsloping". Kalimat catatan di bawah kamus menyebut `restecg` dan `slope` juga. Cell kode 1.8 ditambah print crosstab `restecg` dan `slope` (label UCI: restecg 0 normal, 1 kelainan ST-T, 2 hipertrofi; slope 1 upsloping, 2 flat, 3 downsloping). Markdown temuan 1.8 ditambah dua poin untuk kedua kolom itu.

- [ ] **Step 4: Jalankan perintah test.** Expected: lulus, output 1.8 menampilkan dua crosstab baru yang diagonal.

- [ ] **Step 5: Commit** `fix(fase-1): lengkapi arti kode restecg dan slope di kamus data`

### Task 2: Label sumbu persen grafik cp tanpa desimal (fix fase-1)

**Files:**
- Modify: notebook, cell grafik `cp` (bagian 1.9 b), cell pengecekan grafik Fase 1
- Modify: `05-output/figures/fase-1-cp-vs-target.png` (hasil generate)

- [ ] **Step 1: Tambah assert ke cell pengecekan grafik Fase 1**

```python
label_x = [t.get_text() for t in GRAFIK_TERSIMPAN["fase-1-cp-vs-target"].axes[1].get_xticklabels()]
assert label_x and all(re.fullmatch(r"\d+%", t) for t in label_x if t), label_x
```

- [ ] **Step 2: Jalankan perintah test.** Expected: FAIL dengan daftar berisi `'20,00%'`.

- [ ] **Step 3: Ganti `format_sumbu_id(axes[1].xaxis, persen=True)` menjadi `format_sumbu_id(axes[1].xaxis, 0, persen=True)`.**

- [ ] **Step 4: Jalankan perintah test.** Expected: lulus. Buka PNG dan cek label 0%, 20%, ..., 100%.

- [ ] **Step 5: Commit** `fix(fase-1): hilangkan desimal di label sumbu persen grafik cp` (notebook dan PNG)

### Task 3: Pemisahan fitur dan train-test split

**Files:**
- Modify: notebook, cell baru di akhir (heading `## Fase 2: Data Preparation`, bagian 2.1 sampai 2.2)
- Create: `05-output/figures/fase-2-split.png`

**Interfaces:**
- Consumes: `df`, `RANDOM_STATE`, `simpan_grafik`, `persen_id`, `LABEL_TARGET`, `WARNA_TARGET`
- Produces: `KOLOM_NUMERIK = ["age", "trestbps", "chol", "thalach", "oldpeak"]`, `KOLOM_NOMINAL = ["cp", "restecg", "slope", "thal"]`, `KOLOM_ORDINAL = ["ca"]`, `KOLOM_BINER = ["sex", "fbs", "exang"]`, `X`, `y`, `X_train`, `X_test`, `y_train`, `y_test`

- [ ] **Step 1: Tulis cell pengecekan di akhir section 2.2**

```python
assert sorted(KOLOM_NUMERIK + KOLOM_NOMINAL + KOLOM_ORDINAL + KOLOM_BINER) == sorted(X.columns)
assert X.shape == (302, 13) and "target" not in X.columns
assert y.sum() == 138                      # label Fase 1 tetap terkoreksi
assert X_train.shape == (241, 13) and X_test.shape == (61, 13)
assert y_train.sum() == 110 and y_test.sum() == 28
assert set(X_train.index).isdisjoint(X_test.index)
assert (FIG_DIR / "fase-2-split.png").exists()
```

- [ ] **Step 2: Jalankan perintah test.** Expected: FAIL `NameError: name 'KOLOM_NUMERIK' is not defined`.

- [ ] **Step 3: Implementasi**

Markdown pembuka Fase 2 (tujuan fase, ringkasan keputusan dari EDA). Bagian 2.1: daftar kelompok kolom beserta alasan perlakuannya, lalu `X = df.drop(columns="target")`, `y = df["target"]`. Bagian 2.2: `train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)`, tabel ringkas jumlah dan proporsi sakit untuk keseluruhan, latih, dan uji, lalu grafik `fase-2-split` berupa batang bertumpuk horizontal proporsi Sehat/Sakit untuk ketiga kelompok, dengan label `persen_id` dan warna `WARNA_TARGET`. Markdown alasan 80:20 dan `stratify`.

- [ ] **Step 4: Jalankan perintah test dua kali.** Expected: lulus, dan output kedua sama.

- [ ] **Step 5: Commit** `feat(fase-2): tambah pemisahan fitur dan train-test split 80:20 dengan stratify` (notebook dan PNG)

### Task 4: Preprocessor dan bukti tanpa leakage

**Files:**
- Modify: notebook, cell baru (bagian 2.3 sampai 2.5)
- Create: `05-output/figures/fase-2-scaling.png`

**Interfaces:**
- Consumes: semua Produces dari Task 3
- Produces: `preprocessor`: `ColumnTransformer` **belum di-fit**, dengan transformer bernama `"numerik"` (StandardScaler, `KOLOM_NUMERIK`), `"nominal"` (`Pipeline([("isi", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])`, `KOLOM_NOMINAL`), `"ordinal"` (`SimpleImputer(strategy="most_frequent")`, `KOLOM_ORDINAL`), `remainder="passthrough"`, `verbose_feature_names_out=False`, dan `.set_output(transform="pandas")`. Fase 3 memakai `preprocessor` di dalam `Pipeline`.

- [ ] **Step 1: Tulis cell pengecekan di akhir section 2.5**

```python
assert not hasattr(preprocessor, "transformers_")          # tetap belum di-fit untuk Fase 3
assert X_train_demo.shape == (241, 22) and X_test_demo.shape == (61, 22)
assert not X_train_demo.isna().any().any() and not X_test_demo.isna().any().any()
assert np.allclose(X_train_demo[KOLOM_NUMERIK].mean(), 0, atol=1e-9)
assert np.allclose(X_train_demo[KOLOM_NUMERIK].std(ddof=0), 1, atol=1e-9)
assert not np.allclose(X_test_demo[KOLOM_NUMERIK].mean(), 0, atol=1e-9)
assert list(prep_demo.named_transformers_["ordinal"].statistics_) == [0.0]
assert prep_demo.named_transformers_["nominal"].named_steps["isi"].statistics_[3] == 2.0  # modus thal
baris_aneh = X_test.head(1).assign(restecg=9, cp=7)
assert prep_demo.transform(baris_aneh).shape == (1, 22)    # kategori tak dikenal tidak error
assert (FIG_DIR / "fase-2-scaling.png").exists()
```

- [ ] **Step 2: Jalankan perintah test.** Expected: FAIL `NameError: name 'preprocessor' is not defined`.

- [ ] **Step 3: Implementasi**

Bagian 2.3: Markdown penjelasan One-Hot Encoding, imputasi modus, StandardScaler (rumus $z = (x - \mu) / \sigma$), lalu definisi `preprocessor` sesuai Interfaces. Bagian 2.4: `prep_demo = clone(preprocessor).fit(X_train)`, `X_train_demo = prep_demo.transform(X_train)`, `X_test_demo = prep_demo.transform(X_test)`, tampilkan daftar 22 nama fitur, modus hasil imputer, dan tabel perbandingan mean dan std kolom numerik (sebelum, latih sesudah, uji sesudah) dibulatkan 2 desimal, lalu Markdown yang menjelaskan kenapa mean data uji tidak tepat 0. Bagian 2.5: grafik `fase-2-scaling` (2 baris x 5 kolom, histogram setiap fitur numerik data latih sebelum dan sesudah scaling, sumbu memakai `format_sumbu_id`), Markdown temuan, lalu Markdown "yang diserahkan ke Fase 3".

- [ ] **Step 4: Jalankan perintah test.** Expected: lulus. Buka PNG dan cek label terbaca.

- [ ] **Step 5: Commit** `feat(fase-2): tambah preprocessor ColumnTransformer dan bukti tanpa data leakage` (notebook dan PNG)

### Task 5: Draf laporan Fase 2 dan status README

**Files:**
- Create: `06-laporan/drafts/fase-2.md`
- Modify: `README.md` (tabel Status Fase dan checklist Fase 2), `06-laporan/drafts/fase-1.md` (arti kode `restecg` dan `slope` di kamus data 2.1)

- [ ] **Step 1: Tulis draf** dengan heading `## Bab 2 Metodologi: Data Preparation` (pemisahan fitur, split, encoding, imputasi, scaling, pencegahan leakage, tabel bukti), `## Isi Slide` (satu slide "Persiapan Data", maksimal 4 poin, gambar `fase-2-split.png` dan `fase-2-scaling.png`), dan `## Referensi Sementara` (dokumentasi scikit-learn). Tambahkan arti kode `restecg` dan `slope` ke draf Fase 1.

- [ ] **Step 2: Cek silang.** Setiap angka di draf sama dengan output notebook terakhir. `grep -nP "[\x{2013}\x{2014}]"` pada file yang diubah tidak menghasilkan apa-apa.

- [ ] **Step 3: Perbarui README.** Status Fase 2 menjadi "Selesai", checklist Fase 2 dicentang.

- [ ] **Step 4: Commit** `docs(fase-2): tambah draf metodologi data preparation dan perbarui status fase`

### Task 6: Verifikasi akhir dan serah terima

- [ ] **Step 1:** Hapus folder `05-output/figures`, jalankan perintah test. Expected: lulus dan `git status` bersih.
- [ ] **Step 2:** Review seluruh branch, perbaiki temuan penting.
- [ ] **Step 3:** Berhenti, ringkas untuk user, tunggu persetujuan. Setelah disetujui: `git checkout main && git merge --no-ff fase-2 -m "feat(fase-2): selesaikan data preparation"` lalu push `main`.
