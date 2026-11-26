# Setup dan Fase 1 (Clinical & Data Understanding) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Menyiapkan environment yang bisa di-install ulang oleh anggota kelompok, lalu membangun section Fase 1 di notebook final beserta grafik EDA dan draf bab laporan.

**Architecture:** Satu notebook `03-notebook/KelompokX_Heart_Disease_Prediction.ipynb` yang tumbuh per fase. "Test" di project ini berupa cell `assert` di dalam notebook, dijalankan lewat `jupyter nbconvert --execute` dari folder `03-notebook`. Kalau satu assert gagal, nbconvert gagal, dan itu sama dengan test merah. Grafik disimpan ke `05-output/figures/`, draf tulisan ke `06-laporan/drafts/`.

**Tech Stack:** Python 3.12 (`.venv` dari `uv`), pandas, numpy, matplotlib, seaborn, scikit-learn, ipykernel, nbconvert.

**Spec:** `docs/superpowers/specs/2026-11-26-heart-disease-phases-design.md`

Plan ini hanya mencakup setup dan Fase 1. Fase 2 sampai 5 masing-masing dibuatkan plan sendiri setelah fase sebelumnya direview user, karena isi fase berikutnya bergantung pada hasil fase sebelumnya (misalnya keputusan scoring di Fase 4).

## Global Constraints

- Commit message bahasa Indonesia, format `<tipe>(<scope>): <deskripsi>`, tipe `feat|fix|docs|chore`, scope `setup` atau `fase-1`. Setiap commit diakhiri baris `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Markdown notebook, draf laporan, dan dokumentasi ditulis dalam bahasa Indonesia. Tidak memakai karakter em dash atau en dash sebagai tanda pisah.
- Dataset dibaca dari `../02-dataset/heart_disease.csv` (relatif ke `03-notebook`).
- Label `target` tidak diubah: 1 = ada penyakit jantung.
- `random_state=42` untuk semua hal yang memakai random.
- Notebook lama `03-notebook/heart-disease-uci-dataset.ipynb` tidak diubah.
- Setiap akhir fase: berhenti untuk review user, lalu `git merge --no-ff fase-1` ke `main` setelah user setuju.

## Review Focus

- Run All dari kernel baru setelah clone bersih: folder `05-output/figures/` belum tentu ada. Diharapkan notebook membuatnya sendiri (Task 2, assert folder ada setelah cell setup).
- Cell dijalankan ulang tidak berurutan (user sering klik run di tengah): cleaning harus idempotent, hasilnya tetap 302 baris (Task 3, cleaning selalu mulai dari `df_raw.copy()`).
- Kolom `thal`/`ca` berisi NaN setelah cleaning: grafik dan heatmap korelasi tidak boleh error (Task 4, eksekusi penuh notebook).
- Anggota lain install di laptopnya: `pip install -r requirements.txt` di Python 3.12 harus cukup untuk Run All (Task 1, install ulang dari requirements ke venv baru di scratchpad).
- Angka di draf laporan berbeda dengan output notebook: angka harus disalin dari output eksekusi terakhir (Task 5, cek silang angka).

---

### Task 1: Environment dan requirements (branch `setup`)

**Files:**
- Create: `requirements.txt`
- Commit juga: `docs/superpowers/plans/2026-11-26-setup-dan-fase-1.md`

**Interfaces:**
- Produces: `.venv` yang berisi semua library di Tech Stack; `requirements.txt` dengan versi di-pin (`==`) sesuai versi yang ter-install.

- [ ] **Step 1: Verifikasi library belum lengkap**

Run: `.venv/Scripts/python.exe -c "import sklearn, seaborn, matplotlib, nbconvert"`
Expected: `ModuleNotFoundError`

- [ ] **Step 2: Install dan tulis `requirements.txt`**

`uv pip install --python .venv/Scripts/python.exe pandas numpy matplotlib seaborn scikit-learn ipykernel nbconvert`, lalu tulis `requirements.txt` berisi ketujuh package itu (ditambah `openpyxl` yang sudah ter-install) dengan versi dari `uv pip freeze`. Hanya package langsung, bukan seluruh hasil freeze.

- [ ] **Step 3: Verifikasi import dan install ulang dari nol**

Run: `.venv/Scripts/python.exe -c "import sklearn, seaborn, matplotlib, nbconvert, pandas; print('ok')"` lalu buat venv sementara di scratchpad dan `uv pip install -r requirements.txt` ke sana.
Expected: `ok`, dan install ulang selesai tanpa error.

- [ ] **Step 4: Commit dan merge setup**

```bash
git add requirements.txt docs/superpowers/plans/2026-11-26-setup-dan-fase-1.md
git commit -m "chore(setup): tambah requirements.txt dan rencana kerja fase 1"
git checkout main && git merge --no-ff setup -m "chore(setup): gabungkan persiapan project"
git checkout -b fase-1
```

### Task 2: Kerangka notebook, Clinical Understanding, load dan inspeksi data

**Files:**
- Create: `03-notebook/KelompokX_Heart_Disease_Prediction.ipynb`

**Interfaces:**
- Produces (variabel global notebook yang dipakai fase berikutnya):
  - `RANDOM_STATE = 42`, `FIG_DIR` (Path ke `../05-output/figures`, dibuat dengan `mkdir(parents=True, exist_ok=True)`)
  - `simpan_grafik(fig, nama: str) -> Path`: simpan ke `FIG_DIR / f"{nama}.png"` dengan `dpi=150, bbox_inches="tight"`, return path-nya.
  - `df_raw`: DataFrame hasil `pd.read_csv`, tidak pernah dimutasi.

- [ ] **Step 1: Tulis cell assert untuk load data**

```python
assert df_raw.shape == (303, 14)
assert list(df_raw.columns)[-1] == "target"
assert FIG_DIR.exists()
```

- [ ] **Step 2: Jalankan dan pastikan gagal**

Run (dari `03-notebook`): `../.venv/Scripts/jupyter nbconvert --to notebook --execute --inplace KelompokX_Heart_Disease_Prediction.ipynb`
Expected: FAIL, `NameError: name 'df_raw' is not defined`

- [ ] **Step 3: Isi notebook di atas cell assert**

Urutan cell: judul notebook dan daftar isi fase (Markdown); heading `## Fase 1: Clinical & Data Understanding`; Markdown Clinical Understanding (masalah klinis, tujuan deteksi dini, metrik sukses: Recall kelas 1 dan alasan False Negative lebih berbahaya daripada False Positive); cell setup (import, `RANDOM_STATE`, `FIG_DIR`, `simpan_grafik`, style seaborn); load `df_raw`; `df_raw.head()`; `df_raw.info()`; `df_raw.describe()`; Markdown kamus data 14 kolom (nama, arti medis, tipe: numerik/nominal/biner/ordinal).

- [ ] **Step 4: Jalankan dan pastikan lulus**

Run: perintah nbconvert yang sama. Expected: selesai tanpa error.

- [ ] **Step 5: Commit**

```bash
git add 03-notebook/KelompokX_Heart_Disease_Prediction.ipynb
git commit -m "feat(fase-1): tambah kerangka notebook, clinical understanding, dan inspeksi awal data"
```

### Task 3: Penanganan duplikat dan missing value tersembunyi

**Files:**
- Modify: `03-notebook/KelompokX_Heart_Disease_Prediction.ipynb` (section Fase 1, setelah inspeksi awal)

**Interfaces:**
- Consumes: `df_raw`
- Produces: `df`: hasil cleaning, 302 baris, `thal`/`ca` bertipe float dengan NaN. Fase 2 memakai `df` sebagai input.

- [ ] **Step 1: Tulis cell assert**

```python
assert df.shape == (302, 14)
assert df.duplicated().sum() == 0
assert df["thal"].isna().sum() == 2
assert df["ca"].isna().sum() == 4   # 1 dari 5 baris ca=4 ikut terhapus sebagai duplikat
assert (df["thal"] == 0).sum() == 0 and (df["ca"] == 4).sum() == 0
assert df["target"].value_counts().to_dict() == {1: 164, 0: 138}
assert len(df_raw) == 303           # df_raw tidak ikut berubah
```

- [ ] **Step 2: Jalankan, pastikan gagal** (`NameError: name 'df'`)

- [ ] **Step 3: Implementasi cleaning**

Cell yang selalu mulai dari `df = df_raw.copy()`, cek `isna().sum()` dan `duplicated().sum()` (tampilkan baris duplikatnya), `drop_duplicates().reset_index(drop=True)`, lalu `replace` `thal` 0 dan `ca` 4 jadi `np.nan`. Markdown sebelum cell: alasan kode `thal=0`/`ca=4` dianggap missing (di dataset UCI asli kode ini tidak ada). Markdown setelahnya: imputasi modus dilakukan di Fase 2 di dalam pipeline supaya tidak terjadi leakage.

- [ ] **Step 4: Jalankan, pastikan lulus. Jalankan sekali lagi untuk memastikan hasilnya tetap sama.**

- [ ] **Step 5: Commit**

```bash
git commit -am "feat(fase-1): hapus duplikat dan tandai thal=0 serta ca=4 sebagai missing value"
```

### Task 4: Visualisasi EDA dan temuan

**Files:**
- Modify: notebook (section Fase 1, setelah cleaning)
- Create: `05-output/figures/fase-1-distribusi-target.png`, `fase-1-cp-vs-target.png`, `fase-1-numerik-vs-target.png`, `fase-1-korelasi.png`

**Interfaces:**
- Consumes: `df`, `simpan_grafik`
- Produces: empat file PNG di atas (dipakai di slide dan laporan).

- [ ] **Step 1: Tulis cell assert di akhir section EDA**

```python
for nama in ["distribusi-target", "cp-vs-target", "numerik-vs-target", "korelasi"]:
    assert (FIG_DIR / f"fase-1-{nama}.png").exists(), nama
```

- [ ] **Step 2: Hapus PNG lama kalau ada, jalankan, pastikan gagal**

- [ ] **Step 3: Implementasi grafik**

Label sumbu dan legenda dalam bahasa Indonesia, kelas ditulis "Sehat (0)" dan "Sakit Jantung (1)".
1. Distribusi `target`: countplot dengan jumlah dan persentase di atas bar.
2. `cp` vs `target`: countplot berkelompok, tick diberi nama tipe nyeri dada (0 typical angina, 1 atypical angina, 2 non-anginal pain, 3 asymptomatic).
3. `age`, `trestbps`, `chol`, `thalach` per `target`: grid 2x2 KDE/histogram dengan hue `target`.
4. Heatmap korelasi semua kolom `df` (anotasi 2 desimal, colormap diverging, center 0).

Setiap grafik disimpan lewat `simpan_grafik(fig, "fase-1-<nama>")`. Setelah setiap grafik ada Markdown 2 sampai 4 kalimat berisi temuannya.

Tambahkan juga cell tabel `df.groupby("target")[["thalach", "oldpeak", "exang", "cp"]].mean()` dan Markdown "Catatan anomali label": pasien `target=1` punya `thalach` lebih tinggi, `oldpeak` lebih rendah, `exang` lebih jarang, berkebalikan dengan pengetahuan medis. Ini masalah yang dikenal di dataset versi Kaggle. Keputusan kita adalah tetap mengikuti definisi panduan, dan dampaknya dibahas di Fase 5.

- [ ] **Step 4: Jalankan, pastikan lulus, lalu buka keempat PNG dan cek visualnya (label terbaca, tidak terpotong).**

- [ ] **Step 5: Commit**

```bash
git add 03-notebook/KelompokX_Heart_Disease_Prediction.ipynb 05-output/figures/fase-1-*.png
git commit -m "feat(fase-1): tambah visualisasi EDA dan catatan anomali label target"
```

### Task 5: Draf laporan dan isi slide Fase 1

**Files:**
- Create: `06-laporan/drafts/fase-1.md`

**Interfaces:**
- Consumes: output notebook hasil eksekusi terakhir dan PNG dari Task 4.
- Produces: draf yang nanti dirakit jadi DOCX/PPTX di Fase 5, dengan heading level 2 tetap: `## Bab 1 Pendahuluan`, `## Bab 2 Metodologi: Data dan EDA`, `## Isi Slide`.

- [ ] **Step 1: Tulis draf**

Bab 1: latar belakang klinis, rumusan masalah, tujuan, manfaat. Bab 2: deskripsi dataset (sumber, 303 x 14, kamus data), pemeriksaan kualitas data (1 duplikat, `thal`/`ca` missing), temuan EDA beserta rujukan gambar `05-output/figures/fase-1-*.png`, dan catatan anomali label. Isi Slide: slide "Masalah Klinis" dan "Insight EDA", maksimal 4 poin per slide, beserta gambar yang dipakai.

- [ ] **Step 2: Cek silang angka**

Setiap angka di draf (303, 302, 164/138, 2, 4, rata-rata pada tabel anomali) harus sama dengan output notebook terakhir. Cek juga tidak ada karakter em dash atau en dash: `grep -nP "[\x{2013}\x{2014}]" 06-laporan/drafts/fase-1.md` harus tidak menghasilkan apa-apa.

- [ ] **Step 3: Commit**

```bash
git add 06-laporan/drafts/fase-1.md
git commit -m "docs(fase-1): tambah draf bab pendahuluan, metodologi data, dan isi slide EDA"
```

### Task 6: Verifikasi akhir Fase 1 dan serah terima review

- [ ] **Step 1: Run All bersih**

Run (dari `03-notebook`): `../.venv/Scripts/jupyter nbconvert --to notebook --execute --inplace KelompokX_Heart_Disease_Prediction.ipynb`
Expected: tanpa error. `git status` hanya menunjukkan perubahan output (kalau ada), lalu commit `chore(fase-1): perbarui output notebook` bila perlu.

- [ ] **Step 2: Berhenti dan minta review user**

Ringkas untuk user: daftar commit Fase 1 (`git log --oneline main..fase-1`), temuan EDA utama, dan file yang bisa dibuka. Jangan merge sebelum user setuju.

- [ ] **Step 3: Setelah user setuju, merge**

```bash
git checkout main && git merge --no-ff fase-1 -m "feat(fase-1): selesaikan clinical dan data understanding"
```
