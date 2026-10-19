# Model Generatif Terawasi dengan Label dari Model Difusi

Dokumen ini menjelaskan metode yang kami gunakan, bagaimana metode itu kami implementasikan dari paper, dataset dan eksperimennya, hasil yang sudah diperoleh, serta aplikasi yang dibangun di atasnya.

**Paper acuan:** Yanfang Liu, Minglei Yang, Zezhong Zhang, Feng Bao, Yanzhao Cao, Guannan Zhang. *Diffusion-Model-Assisted Supervised Learning of Generative Models for Density Estimation.* Journal of Machine Learning for Modeling and Computing, 5(1):25–38, 2024.

---

## 1. Ringkasan

Kami mengimplementasikan ulang metode dari paper di atas dari nol dengan PyTorch, melatihnya sendiri pada dataset yang sama dengan paper (4 dataset 2D sintetis dan 4 dataset UCI: POWER, GAS, HEPMASS, MINIBOONE), lalu membangun aplikasi web (Streamlit) yang memakai metode ini untuk membuat **data tabular sintetis** dari file CSV apa pun.

Inti metodenya: model difusi dipakai **bukan** sebagai generator, melainkan sebagai **mesin pembuat label**. Label itu kemudian dipakai untuk melatih jaringan saraf biasa (MLP) dengan loss MSE, persis seperti regresi biasa.

---

## 2. Masalah yang Diselesaikan Paper

### 2.1 Estimasi densitas dan model generatif

Diberikan sekumpulan data $\mathcal{X} = \{x_1, \dots, x_J\} \subset \mathbb{R}^d$ yang berasal dari distribusi $p(x)$ yang tidak diketahui. Tujuannya adalah membangun model yang bisa menghasilkan sampel baru dari $p(x)$ sebanyak apa pun. Model generatif yang dicari berbentuk:

$$X = F(Y; \theta), \qquad Y \sim \mathcal{N}(0, I_d)$$

yaitu fungsi $F$ yang memetakan noise Gaussian $Y$ ke data $X$. Setelah $F$ dilatih, sampel baru didapat cukup dengan mengambil $y$ acak lalu menghitung $F(y)$.

### 2.2 Kenapa ini sulit

Masalah utamanya adalah **tidak ada label**: untuk setiap data $x_j$ kita tidak tahu $y$ mana pasangannya. Karena itu $F$ tidak bisa dilatih dengan regresi MSE biasa, dan metode yang ada memakai loss tak-terawasi (unsupervised) yang masing-masing punya kelemahan:

| Metode | Cara melatih | Kelemahan |
|---|---|---|
| GAN | Adversarial loss | Mode collapse, gradien hilang, pelatihan tidak stabil |
| Normalizing flow | Maximum likelihood lewat rumus perubahan variabel | Arsitektur harus bisa dibalik (invertible) dan determinan Jacobian harus mudah dihitung |
| Model difusi | Score matching | Sampling lambat karena harus menyelesaikan SDE/ODE ratusan langkah, dan butuh jaringan saraf untuk mempelajari skor |

### 2.3 Ide paper

Jika kita bisa **membuat pasangan berlabel** $(y_m, x_m)$, maka $F$ cukup dilatih dengan MSE: $\min_\theta \sum_m \lVert F(y_m;\theta) - x_m \rVert^2$. Paper ini membuat pasangan tersebut dengan model difusi berbasis skor, dan skornya pun dihitung **tanpa melatih jaringan saraf** (*training-free score estimation*).

---

## 3. Metode

### 3.1 SDE maju (forward SDE)

Model difusi didefinisikan pada waktu $t \in [0, 1]$. SDE maju mengubah data ($t = 0$) menjadi noise Gaussian ($t = 1$):

$$dZ_t = b(t) Z_t\, dt + \sigma(t)\, dW_t, \qquad Z_0 = X,\; Z_1 = Y \tag{3}$$

dengan koefisien

$$b(t) = \frac{d \log \alpha_t}{dt}, \qquad \sigma^2(t) = \frac{d\beta_t^2}{dt} - 2\frac{d \log \alpha_t}{dt}\beta_t^2 \tag{4}$$

$$\alpha_t = 1 - t, \qquad \beta_t^2 = t \tag{5}$$

Dengan pilihan ini, distribusi bersyarat $Z_t$ diberikan $Z_0$ adalah Gaussian:

$$Q(Z_t \mid Z_0) = \mathcal{N}(\alpha_t Z_0,\; \beta_t^2 I_d) \tag{6}$$

Artinya pada waktu $t$, data "dikecilkan" dengan faktor $\alpha_t$ lalu ditambah noise dengan variansi $\beta_t^2$. Pada $t = 1$ hasilnya $\mathcal{N}(0, I)$ apa pun distribusi datanya.

### 3.2 SDE mundur dan ODE

Arah sebaliknya (noise → data) diberikan oleh SDE mundur:

$$dZ_t = \left[b(t) Z_t - \sigma^2(t) S(Z_t, t)\right] dt + \sigma(t)\, d\overleftarrow{W}_t \tag{7}$$

dengan $S(z, t) = \nabla_z \log Q(Z_t = z)$ adalah **fungsi skor** (8). Seluruh informasi tentang distribusi data tersimpan di fungsi skor ini.

Masalahnya, SDE bersifat acak: titik awal $y$ yang sama bisa berakhir di $x$ yang berbeda, sehingga hubungan $y \to x$ tidak bisa dipelajari sebagai fungsi. Karena itu paper memakai **ODE** yang distribusi marjinalnya sama dengan SDE tetapi deterministik:

$$dZ_t = \left[b(t) Z_t - \tfrac{1}{2}\sigma^2(t) S(Z_t, t)\right] dt \tag{13}$$

ODE ini memberikan pemetaan $y \mapsto x$ yang mulus dan deterministik (Gambar 1 di paper), sehingga cocok dijadikan label untuk regresi.

### 3.3 Estimasi skor tanpa pelatihan (training-free)

Biasanya skor dipelajari oleh jaringan saraf. Paper ini menurunkan rumus skor secara langsung. Karena $Q(Z_t) = \int Q(Z_t \mid Z_0)\, Q(Z_0)\, dZ_0$ dan $Q(Z_t \mid Z_0)$ Gaussian, diperoleh:

$$S(z, t) = \int -\frac{z - \alpha_t Z_0}{\beta_t^2}\; w_t(z, Z_0)\; Q(Z_0)\, dZ_0 \tag{9}$$

$$w_t(z, Z_0) = \frac{Q(z \mid Z_0)}{\int Q(z \mid Z_0')\, Q(Z_0')\, dZ_0'} \tag{10}$$

Integral terhadap $Q(Z_0)$ (distribusi data) diaproksimasi dengan Monte Carlo memakai mini-batch data $\{x_{j_n}\}_{n=1}^N$:

$$S(z, t) \approx \bar S(z, t) = \sum_{n=1}^{N} -\frac{z - \alpha_t x_{j_n}}{\beta_t^2}\; \bar w_t(z, x_{j_n}) \tag{11}$$

$$\bar w_t(z, x_{j_n}) = \frac{Q(z \mid x_{j_n})}{\sum_{m=1}^{N} Q(z \mid x_{j_m})} \tag{12}$$

**Intuisinya:** skor di titik $z$ adalah rata-rata berbobot dari arah "menuju setiap titik data (yang sudah diskalakan $\alpha_t$)". Bobotnya sebanding dengan likelihood Gaussian, jadi titik data yang dekat dengan $z$ punya pengaruh paling besar. Tidak ada jaringan saraf yang perlu dilatih.

### 3.4 Algoritma lengkap (Algorithm 1 di paper)

1. Ambil $M$ sampel $y_1, \dots, y_M \sim \mathcal{N}(0, I_d)$.
2. Untuk setiap $y_m$, selesaikan ODE (13) dari $t = 1$ ke $t = 0$ dengan skor dari (11)–(12). Hasil akhirnya adalah $x_m$.
3. Kumpulkan data berlabel $\mathcal{D}_{\text{train}} = \{(y_m, x_m)\}_{m=1}^{M}$ (14).
4. Latih $F(y; \theta)$ dengan loss MSE pada $\mathcal{D}_{\text{train}}$.

Catatan penting: $x_m$ **bukan** salinan data asli, melainkan sampel baru yang mengikuti distribusi data. Jumlahnya ($M$) juga boleh jauh lebih besar dari jumlah data asli, yang membantu mengurangi overfitting.

### 3.5 Keunggulan dibanding metode lain

- Dibanding **normalizing flow**: $F$ tidak perlu bisa dibalik dan tidak ada determinan Jacobian, jadi cukup MLP biasa.
- Dibanding **model difusi**: setelah $F$ terlatih, sampling hanya butuh **satu forward pass** (tanpa ratusan langkah ODE), sehingga jauh lebih cepat.
- Dibanding **GAN**: pelatihannya regresi MSE biasa yang stabil.

---

## 4. Implementasi

### 4.1 Struktur kode

| File | Isi |
|---|---|
| `sgm.py` | Inti metode: jadwal $\alpha_t, \beta_t^2$, estimasi skor (11)–(12), ODE (13), pembuatan label, MLP, dan pelatihan MSE. `python sgm.py` menjalankan uji otomatis. |
| `data.py` | Dataset 2D sintetis dan pemuat (loader) POWER / GAS / HEPMASS / MINIBOONE dengan preprocessing yang sama seperti penulis paper. |
| `train.py` | Eksperimen Bagian 4 paper: membuat label, melatih generator, menghitung metrik, dan menyimpan gambar ke `runs/<nama>/`. |
| `app.py` | Aplikasi Streamlit (Playground, Synthetic data, Class balancing, Paper reproduction). |
| `ablation_refs.py` | Eksperimen perbaikan pada Bagian 5.5 (pengaruh batch referensi skor terhadap generator). |
| `balance_experiment.py` | Eksperimen penyeimbangan kelas pada Bagian 7.3 (generator difusi vs SMOTE, class weights, dll.). |

Seluruh kode berjalan di **CPU** (laptop AMD Ryzen 5 6600H, 6 core). Paper memakai GPU NVIDIA RTX A5000.

Selain paper, kami juga membaca **kode resmi penulis** (github.com/mlmathphy/supervised_generative_model) karena beberapa detail (jadwal, ukuran batch skor, jumlah label) tidak tertulis lengkap di paper.

### 4.2 Dari persamaan ke kode

**Jadwal $\alpha_t$ dan $\beta_t^2$ (Persamaan 4–5).**

```python
S = 0.99
def alpha(t): return 1 - S * t
def beta2(t): return 1 - S + S * t
def b(t): return -S / alpha(t)                    # d log(alpha_t)/dt
def sigma2(t): return S - 2 * b(t) * beta2(t)     # d beta_t^2/dt - 2 b(t) beta_t^2
```

Paper menulis $\alpha_t = 1 - t$ dan $\beta_t^2 = t$. Dengan rumus itu $b(t) = -1/(1-t)$ menjadi tak hingga di $t = 1$, dan $1/\beta_t^2$ tak hingga di $t = 0$. Kode penulis memakai versi yang sedikit digeser, $\alpha_t = 1 - 0{,}99t$ dan $\beta_t^2 = 0{,}01 + 0{,}99t$, dan kami mengikutinya.

**Estimasi skor (Persamaan 11–12).**

```python
def score(z, t, x0):
    a, v = alpha(t), beta2(t)
    ax0 = a * x0
    w = torch.softmax((z @ ax0.T - 0.5 * ax0.square().sum(1)) / v, dim=1)
    return -(z - w @ ax0) / v
```

Tiga hal yang kami lakukan di sini:

1. **Bobot (12) dihitung dengan softmax.** $Q(z \mid x_n) \propto \exp\left(-\lVert z - \alpha_t x_n \rVert^2 / 2\beta_t^2\right)$, sehingga normalisasi pada (12) persis sama dengan softmax dari $-\lVert z - \alpha_t x_n \rVert^2 / 2\beta_t^2$. Hasilnya identik secara matematis, tetapi stabil secara numerik. Kode penulis memakai `exp` langsung; di dekat $t = 0$ ($\beta_t^2 = 0{,}01$) semua nilai `exp` bisa menjadi 0 dan menghasilkan pembagian 0/0 (NaN).
2. **Penyederhanaan jumlahan (11).** Karena $\sum_n \bar w_n = 1$:
   $$\sum_n -\frac{z - \alpha_t x_n}{\beta_t^2}\,\bar w_n = -\frac{z - \alpha_t \sum_n \bar w_n x_n}{\beta_t^2}$$
   Jadi skor cukup dihitung dari satu perkalian matriks `w @ x0`, tanpa membentuk tensor besar berukuran (jumlah titik × N × d). Memori turun dari $O(BNd)$ menjadi $O(BN)$.
3. **Jarak tanpa `cdist`.** $-\lVert z - \alpha_t x_n \rVert^2 / 2\beta_t^2 = \left(z \cdot \alpha_t x_n - \tfrac{1}{2}\lVert \alpha_t x_n \rVert^2\right)/\beta_t^2 - \lVert z \rVert^2 / 2\beta_t^2$. Suku terakhir sama untuk semua $n$, sehingga tidak mengubah hasil softmax dan boleh dibuang. Bobot cukup dihitung dari satu perkalian matriks `z @ ax0.T`. Hasilnya identik (selisih relatif ≤ $10^{-5}$, setara pembulatan float32) dan sekitar 20% lebih cepat.

**ODE (Persamaan 13) dengan metode Euler eksplisit.**

```python
def ode(z, x0, steps, path=None):
    dt = 1 / steps
    for i in range(steps):
        t = 1 - i * dt
        z = z - dt * (b(t) * z - 0.5 * sigma2(t) * score(z, t, x0))
        if path is not None: path.append(z)
    return z
```

Integrasi berjalan mundur dari $t = 1$ ke $t = 0$, sehingga langkahnya $z \leftarrow z - \Delta t \cdot \text{drift}$. Parameter `path` dipakai aplikasi untuk menggambar lintasan.

**Pembuatan label (Algorithm 1, langkah 1–3).**

```python
def make_labels(x, m, steps=500, n_ref=1000, chunk=5000, ...):
    y = torch.randn(m, x.shape[1])
    for z in y.split(chunk):
        x0 = x[torch.randperm(len(x))[:n_ref]]   # batch referensi untuk skor
        out.append(ode(z, x0, steps))
    return y, torch.cat(out)
```

$M$ lintasan diproses per potongan (chunk) 5000 lintasan. Setiap potongan memakai 1000 titik data acak sebagai batch referensi Monte Carlo, sama seperti kode penulis.

**Generator dan pelatihan (Algorithm 1, langkah 4).**

```python
def mlp(d, hidden, layers):   # Linear -> Tanh (x layers) -> Linear
def fit(y, x, hidden, layers, epochs, lr, batch=None):
    loss = (net(y[idx]) - x[idx]).square().sum(1).mean()   # MSE
```

Generator adalah MLP dengan aktivasi tanh. Dioptimasi dengan Adam dan jadwal learning rate cosine.

### 4.3 Verifikasi implementasi

`python sgm.py` menjalankan dua uji otomatis:

1. **Uji skor.** Jika data $\sim \mathcal{N}(0, I)$, maka $Z_t \sim \mathcal{N}(0, (\alpha_t^2 + \beta_t^2) I)$ dan skor eksaknya $-z/(\alpha_t^2 + \beta_t^2)$. Estimasi Monte Carlo kami harus mendekati nilai eksak ini dengan galat relatif < 5% pada $t = 0{,}2;\ 0{,}5;\ 0{,}9$.
2. **Uji label.** Untuk data campuran dua Gaussian, label hasil ODE harus punya rata-rata dan simpangan baku yang sama dengan data (toleransi 0,15).

Kedua uji lulus.

### 4.4 Dataset dan preprocessing

**Dataset 2D (Bagian 4.1 paper).** Masing-masing 1000 titik data latih, ditambah 2000 titik baru untuk evaluasi.

| Nama | Keterangan |
|---|---|
| `gaussians` | Campuran 6 Gaussian yang tersusun melingkar (seperti Gambar 2 paper) |
| `moons` | Dua bulan sabit |
| `spirals` | Dua spiral (versi FFJORD) |
| `checkerboard` | Papan catur; densitasnya tidak kontinu, kasus tersulit untuk $F$ yang mulus |

Paper memakai dataset "Diffeqml" (logo dari pustaka torchdyn); kami menggantinya dengan `checkerboard`.

**Dataset UCI (Bagian 4.2 paper).** Preprocessing mengikuti Papamakarios dkk. (2017, MAF), sama seperti kode penulis: setiap dimensi dinormalisasi ke rata-rata 0 dan simpangan baku 1, atribut diskret dibuang, atribut dengan korelasi Pearson > 0,98 dibuang, lalu data dibagi menjadi train/validation/test. Data latih POWER, GAS, dan HEPMASS dibatasi 200.000 baris (mengikuti kode penulis).

| Dataset | Dimensi | Data latih | Data uji |
|---|---|---|---|
| POWER | 6 | 200.000 | 204.928 |
| GAS | 8 | 200.000 | 105.206 |
| HEPMASS | 21 | 200.000 | 174.987 |
| MINIBOONE | 43 | 29.556 | 3.648 |

Dua kendala yang kami temui dan selesaikan:

- **File GAS tidak bisa dibaca pandas versi baru.** File `ethylene_CO.pickle` dari MAF dibuat dengan Python 2 dan pandas 0.x. Kami menulis unpickler kecil (`data._maf_pickle`) yang membaca array mentahnya dan menyusun ulang DataFrame-nya.
- **HEPMASS 21 dimensi, bukan 11.** Paper menyebut HEPMASS 11 dimensi, tetapi loader resmi penulis (preprocessing MAF) menghasilkan 21 dimensi, yaitu ukuran standar HEPMASS di literatur. Kami memakai 21 dimensi dan mencatat perbedaan ini.

### 4.5 Hyperparameter

| | Paper | Implementasi kami |
|---|---|---|
| **Data 2D** | | |
| Jumlah label $M$ | 1000 | 1000 |
| Langkah ODE | 100 | 100 |
| Generator | 4 hidden layer × 100 neuron | sama (tanh) |
| Optimasi | Adam, lr 0,005, 5000 epoch | sama (full batch, LR cosine) |
| **Data UCI** | | |
| Jumlah label $M$ | (tidak disebut; kode: 200.000) | 200.000 |
| Langkah ODE | 500 (Euler eksplisit) | 500 |
| Batch referensi skor $N$ | 5000 (kode: 1000) | 1000 |
| Hidden neuron | POWER 256, GAS 512, HEPMASS 1024, MINIBOONE 1400 | sama |
| Jumlah hidden layer | 1 (kode: 2) | 2 |
| Optimasi | Adam, lr 0,01 (kode: 0,002), 20.000 iterasi full batch di GPU | Adam, lr 0,002, mini-batch 1024, 100 epoch ≈ 20.000 langkah, LR cosine |

Pelatihan full batch 200.000 data × 20.000 iterasi terlalu berat untuk CPU, jadi kami memakai mini-batch dengan jumlah langkah optimasi yang setara.

### 4.6 Metrik evaluasi

- **Cross entropy (data 2D, seperti Gambar 2 paper):** $H = -\frac{1}{2000}\sum_{i} \log q(x_i^{\text{uji}})$, dengan $q$ adalah KDE (kernel density estimation) yang dipasang pada sampel. Dihitung untuk data asli, label, dan sampel generator. Makin kecil makin baik.
- **KL divergence marjinal (data UCI, seperti Gambar 3–6 paper):** untuk setiap dimensi $k$, $\mathrm{KL}(p_k \,\Vert\, q_k)$ antara KDE 1D (bandwidth 0,2) data asli dan sampel, dihitung pada 1000 titik grid. Kami menormalisasi ulang kedua densitas pada grid agar nilainya tidak bisa negatif (rumus asli penulis bisa sedikit negatif karena kesalahan integrasi numerik).
- **Rata-rata dan simpangan baku per dimensi** untuk data asli, label, dan sampel generator.
- **Waktu komputasi:** pembuatan label, pelatihan $F$, dan pembuatan 100.000 sampel (seperti Tabel 1 paper).

---

## 5. Hasil

### 5.1 Data 2D

Gambar hasil ada di `runs/<nama>/results.png` (kolom: data asli, data berlabel, sampel generator).

| Dataset | CE data asli | CE label | CE generator | Waktu label | Waktu latih |
|---|---|---|---|---|---|
| gaussians | 2,83 | 2,87 | 2,87 | 0,6 s | 13,8 s |
| moons | 2,85 | 2,88 | 2,87 | 0,1 s | 10,9 s |
| spirals | 3,55 | 3,56 | 3,57 | 0,1 s | 10,8 s |
| checkerboard | 4,08 | 4,10 | 4,09 | 0,1 s | 13,5 s |

Pola hasilnya sama dengan paper: cross entropy sampel generator hanya sedikit di atas data asli (selisih ≤ 0,04), semua mode tertangkap, dan sampel generator sedikit lebih kabur daripada label di bagian yang tidak kontinu (pusat spiral, tepi kotak checkerboard). Paper menjelaskan ini karena MLP yang mulus tidak bisa sepenuhnya meniru diskontinuitas antar-mode. Nilai absolutnya berbeda dari paper karena skala datasetnya berbeda.

### 5.2 Data UCI

Gambar hasil ada di `runs/<nama>/results.png` (format sama dengan Gambar 3–6 paper). Nilai paper dibaca dari gambar paper, jadi sifatnya perkiraan (≈). Rentang KL adalah nilai minimum–maksimum dari semua dimensi, dengan rata-rata dalam kurung.

| Dataset | KL label: paper | KL label: kami | KL generator: paper | KL generator: kami | Std generator: paper | Std generator: kami |
|---|---|---|---|---|---|---|
| POWER (6-D) | ≈0,001–0,026 | 0,001–0,017 (0,0065) | ≈0,002–0,044 | 0,001–0,036 (0,0157) | ≈0,99–1,05 | 0,95–1,00 |
| GAS (8-D) | ≈0,001–0,018 | 0,002–0,019 (0,0050) | ≈0,001–0,048 | 0,001–0,029 (0,0061) | ≈0,92–1,00 | 0,92–1,00 |
| HEPMASS (21-D)* | ≈0,000–0,007 | 0,001–0,016 (0,0019) | ≈0,001–0,036 | **0,058–0,168 (0,1103)** | ≈0,98–1,02 | **0,76–0,84** |
| MINIBOONE (43-D) | ≈0,000–0,015 | 0,001–0,008 (0,0028) | ≈0,000–0,11 | 0,007–0,058 (0,0241) | ≈0,91–1,05 | **0,76–0,96** |

\* Paper memakai HEPMASS 11 dimensi (lihat Bagian 4.4).

Rata-rata per dimensi tetap dekat 0 pada semua dataset (label dalam ±0,013, generator dalam ±0,016; paper dalam ±0,02). Pengamatan utama:

1. **Label (hasil ODE) sangat akurat di keempat dataset** (KL ≤ 0,02, simpangan baku 0,93–1,05). Artinya bagian inti paper, yaitu estimasi skor tanpa pelatihan ditambah ODE, bekerja dengan baik bahkan sampai 43 dimensi.
2. **POWER dan GAS mereproduksi paper dengan baik**, termasuk detailnya. Pada GAS, simpangan baku label dan generator sama-sama turun di dimensi 4–6, persis seperti Gambar 4(d) paper.
3. **Pada HEPMASS dan MINIBOONE, sebaran (simpangan baku) sampel generator menyusut** menjadi 0,76–0,96. Pada HEPMASS ini juga membuat KL generator jauh di atas paper. Penyebabnya kami analisis di Bagian 5.4.
4. **KL generator terhadap data uji hampir sama dengan terhadap data latih** (POWER 0,0179 vs 0,0157; GAS 0,0058 vs 0,0061; HEPMASS 0,1101 vs 0,1103; MINIBOONE 0,0250 vs 0,0241). Generator tidak menghafal data latih.

### 5.3 Waktu komputasi (dibandingkan Tabel 1 paper)

| Dataset | Pelabelan (paper, GPU) | Pelabelan (kami, CPU) | Latih $F$ (paper) | Latih $F$ (kami) | 100k sampel (paper) | 100k sampel (kami) |
|---|---|---|---|---|---|---|
| POWER | 64,83 s | 338,2 s | 182,51 s | 75,2 s | 0,10 s | 0,11 s |
| GAS | 85,78 s | 357,2 s | 426,61 s | 164,4 s | 0,23 s | 0,26 s |
| HEPMASS | 109,12 s | 385,3 s | 1940,87 s | 492,5 s | 0,51 s | 0,81 s |
| MINIBOONE | 408,36 s | 320,0 s | 2220,79 s | 771,5 s | 0,66 s | 1,45 s |

Total waktu keempat dataset di CPU kami sekitar 50 menit (pelabelan ±23 menit, pelatihan ±25 menit).

- **Pelabelan** di CPU 3,5–5× lebih lambat dari GPU paper untuk POWER, GAS dan HEPMASS. Untuk MINIBOONE justru lebih cepat. Kemungkinan penyebabnya adalah perkalian matriks pada implementasi skor kami (Bagian 4.2), yang jauh lebih hemat daripada tensor besar (jumlah titik × N × d) di kode penulis; keuntungan ini paling terasa di dimensi tinggi.
- **Pelatihan $F$** kami 2,4–3,9× lebih cepat karena memakai mini-batch, sedangkan paper memakai full batch (setiap iterasi memproses seluruh data label).
- **Sampling** tetap sangat cepat walaupun di CPU (0,1–1,5 detik untuk 100.000 sampel), dan inilah keunggulan utama metode ini: tidak perlu menyelesaikan ODE 500 langkah setiap kali membuat sampel.

### 5.4 Temuan: penyusutan sebaran generator pada dimensi tinggi

Pada HEPMASS dan MINIBOONE, label hasil ODE sangat akurat tetapi sampel generator sebarannya menyempit. Kami menduga penyebabnya adalah **label yang tidak konsisten antar-potongan**: setiap potongan 5000 lintasan memakai batch referensi acak yang berbeda (1000 titik data) untuk menghitung skor, sehingga noise $y$ yang sama bisa berakhir di $x$ yang berbeda bergantung pada potongannya.

Dugaan ini kami uji langsung. Untuk 2000 vektor noise $y$ yang sama, ODE dijalankan dua kali dengan dua batch referensi acak yang berbeda, lalu kami ukur berapa bagian variansi label yang hanya ditentukan oleh pilihan batch referensi (0 = label konsisten sempurna). Kami juga mengukur $R^2$ generator terlatih terhadap labelnya.

| Dataset | Variansi label akibat pilihan batch referensi | $R^2$ generator pada label | Std generator (rata-rata) | $\sqrt{R^2}$ |
|---|---|---|---|---|
| POWER (6-D) | 6% | 0,94 | 0,98 | 0,97 |
| GAS (8-D) | 4% | 0,98 | 0,96 | 0,99 |
| HEPMASS (21-D) | **41%** | 0,62 | 0,79 | 0,79 |
| MINIBOONE (43-D) | **33%** | 0,82 | 0,88 | 0,91 |

Dugaan terbukti, dan penjelasannya sebagai berikut:

- **Kenapa label tidak konsisten di dimensi tinggi.** Bobot skor (12) adalah softmax dari jarak ke titik-titik referensi. Di ruang berdimensi tinggi, selisih jarak antar-titik sangat besar, sehingga softmax hampir seluruhnya jatuh ke satu titik terdekat. ODE lalu membawa $y$ ke dekat titik tersebut, dan titik mana yang terdekat bergantung pada batch referensi yang kebetulan terambil. Di POWER dan GAS (6–8 dimensi), bobot masih tersebar ke banyak titik, sehingga hasilnya hampir tidak bergantung pada batch. Ini sejalan dengan kesimpulan paper bahwa estimasi skor Monte Carlo makin sulit seiring bertambahnya dimensi.
- **Kenapa sebaran generator menyusut.** Regresi MSE mempelajari rata-rata bersyarat $\mathbb{E}[x \mid y]$. Jika sebagian variansi target hanyalah "noise" yang tidak bisa diprediksi dari $y$, generator hanya bisa mempelajari bagian yang konsisten, sehingga variansi keluarannya ≈ $R^2$ × variansi data dan simpangan bakunya ≈ $\sqrt{R^2}$. Tabel di atas menunjukkan kecocokan itu (HEPMASS: 0,79 vs 0,79).
- **Kenapa paper tidak mengalaminya separah ini.** HEPMASS di paper hanya 11 dimensi, sehingga efeknya jauh lebih kecil. Pada MINIBOONE, paper juga menunjukkan simpangan baku generator turun sampai ≈0,91.

Perbaikannya kami uji di Bagian 5.5.

### 5.5 Eksperimen perbaikan: batch referensi bersama

Berdasarkan analisis di atas, kami menguji tiga cara memperbaiki label yang tidak konsisten dan membandingkannya dengan cara asli penulis (A):

- **A (asli):** setiap potongan 5000 lintasan memakai batch referensi acak baru berisi 1000 titik.
- **D:** label sama dengan A, tetapi generator juga diberi tahu potongan asal labelnya, $F(y, c)$ dengan $c$ one-hot. Saat sampling, $c$ dipilih acak.
- **B:** **satu batch referensi bersama** berisi 1000 titik untuk semua potongan, sehingga label menjadi satu fungsi $y \to x$ yang konsisten.
- **C:** satu batch referensi bersama berisi 5000 titik (ukuran yang tertulis di teks paper).

Eksperimen dijalankan pada HEPMASS dengan skala ¼ (50.000 label, 60 epoch; skrip `ablation_refs.py`). Kolom "rasio penyalinan" adalah median jarak sampel generator ke baris latih terdekat dibagi median jarak data uji asli ke baris latih terdekat. Nilai ≈ 1 berarti sampel generator tidak lebih dekat ke data latih daripada data baru yang wajar.

| Setup | Std generator (target 1) | KL vs data latih | KL vs data uji | Rasio penyalinan | Waktu pelabelan |
|---|---|---|---|---|---|
| A. Batch acak per potongan (asli) | 0,79 | 0,097 | 0,097 | 0,98 | 91 s* |
| D. Generator dikondisikan pada potongan | 0,80 | 0,090 | 0,090 | 1,00 | (label A) |
| **B. Batch bersama, 1000 titik** | **0,85** | **0,053** | **0,052** | 1,01 | **67 s** |
| C. Batch bersama, 5000 titik | 0,85 | 0,049 | 0,049 | 1,00 | 441 s |

\* Lebih lambat dari B hanya karena saat itu ada proses lain yang berjalan bersamaan.

Kesimpulan:

1. **Batch referensi bersama (B) adalah perbaikan terbaik.** KL turun hampir setengahnya (0,097 → 0,053) dan simpangan baku naik dari 0,79 ke 0,85, tanpa tambahan biaya komputasi. Ini mengonfirmasi bahwa ketidakkonsistenan label memang penyebab utama masalahnya.
2. **Batch bersama yang lebih besar (C) hampir tidak menambah perbaikan** (KL 0,053 → 0,049), padahal waktu pelabelan 6,6× lebih lama.
3. **Mengondisikan generator pada potongan (D) tidak membantu.** Setiap potongan hanya punya 5000 sampel, terlalu sedikit untuk mempelajari pemetaannya sendiri di ruang 21 dimensi.
4. **Tidak ada setup yang menyalin data latih** (rasio penyalinan ≈ 1,0), dan KL terhadap data uji sama dengan terhadap data latih. Perbaikan B bukan hasil menghafal data latih.
5. **Perbaikannya baru sebagian.** Simpangan baku 0,85 masih di bawah 1. Dengan batch bersama, label memang konsisten, tetapi pemetaan $y \to x$ di 21 dimensi sangat rumit (bobot skor terpusat pada titik referensi terdekat), sehingga MLP masih memperhalusnya.

---

## 6. Perbedaan dengan Paper (Ringkasan)

1. Jadwal $\alpha_t = 1 - 0{,}99t$, $\beta_t^2 = 0{,}01 + 0{,}99t$ (dari kode penulis), bukan $1 - t$ dan $t$.
2. Bobot skor dihitung dengan softmax: identik secara matematis, tetapi stabil secara numerik.
3. Batch referensi skor 1000 titik per potongan lintasan (dari kode penulis; teks paper menyebut 5000).
4. Generator UCI 2 hidden layer, lr 0,002 (dari kode penulis), dilatih dengan mini-batch Adam di CPU.
5. Dataset 2D "Diffeqml" diganti `checkerboard`.
6. HEPMASS 21 dimensi (loader resmi), bukan 11 seperti tertulis di paper.
7. KL marjinal dinormalisasi ulang pada grid; kami juga melaporkan KL terhadap data uji.

---

## 7. Aplikasi

Aplikasi dibuat dengan Streamlit (`app.py`) dan terdiri dari empat tab.

### 7.1 Playground

Peragaan interaktif cara kerja metode pada data 2D:

- Pilih dataset, jumlah titik, jumlah langkah ODE, dan epoch generator, lalu tekan **Run**.
- Slider waktu menampilkan posisi titik-titik ODE dari $t = 1$ (noise) ke $t = 0$ (data).
- Gambar 40 lintasan noise → data (seperti Gambar 1 paper).
- Perbandingan data asli dengan 10.000 sampel generator.
- Perbandingan kecepatan sampling ODE dan generator. Contoh hasil: ODE 6,5 s vs generator 21 ms untuk 10.000 sampel (≈300× lebih cepat).

### 7.2 Synthetic data: penerapan nyata

Pengguna mengunggah file CSV (atau memakai contoh dataset California Housing, 20.640 baris × 9 kolom), lalu aplikasi:

1. Membaca CSV dengan pemisah `,` maupun `;` (Excel dengan pengaturan bahasa Indonesia menyimpan CSV dengan `;`), lalu mengambil kolom numerik saja (kolom non-numerik, konstan, dan baris kosong dibuang).
2. Menyisihkan 20% baris sebagai data "tak terlihat" untuk uji privasi dan uji kegunaan.
3. **Mentransformasi log** kolom yang semua nilainya positif dan sangat miring ke kanan (skewness > 1, misalnya nilai laboratorium atau harga), lalu menormalisasi semua kolom.
4. Membuat label dengan ODE difusi dan melatih generator.
5. Membuat baris sintetis, mengembalikannya ke skala asli (termasuk membalik transformasi log), memotongnya (clip) ke rentang data asli, dan membulatkan kolom yang aslinya bilangan bulat.
6. Menampilkan laporan kualitas, uji kegunaan, dan tombol unduh CSV sintetis.

**Kenapa transformasi log diperlukan.** Pada kolom yang sangat miring (contoh: `creatinine_phosphokinase` pada data gagal jantung, skewness 4,5), generator menyebar nilai secara simetris di skala asli, sehingga sebagian nilai jatuh di bawah minimum data. Nilai-nilai itu dipotong ke minimum dan menumpuk di sana (36 baris sintetis bernilai tepat 23, padahal di data asli hanya 1), sementara nilai ekstrem yang besar hilang. Di skala log, kolom seperti ini mendekati simetris dan masalahnya hilang.

Laporan kualitas berisi:

- **KL marjinal per kolom:** apakah distribusi tiap kolom mirip.
- **Selisih korelasi:** rata-rata $|\rho_{\text{asli}} - \rho_{\text{sintetis}}|$ dan heatmap korelasi berdampingan, untuk melihat apakah hubungan antar-kolom terjaga.
- **Uji privasi (Distance to Closest Record):** median jarak setiap baris sintetis ke baris latih terdekat, dibandingkan dengan jarak baris asli yang tidak pernah dilihat model ke baris latih terdekat. Jika baris sintetis jauh lebih dekat ke data latih, model menyalin data dan aplikasi memberi peringatan.
- Tabel rata-rata dan simpangan baku per kolom, histogram per kolom, dan scatter plot pasangan kolom (secara default pasangan dengan korelasi terkuat).
- **Uji kegunaan (*train on synthetic, test on real*):** pengguna memilih satu kolom sebagai target. Random forest dilatih sekali dengan baris asli dan sekali dengan baris sintetis, lalu keduanya memprediksi target pada 20% baris asli yang tidak pernah dilihat generator. Metriknya dipilih otomatis: AUC untuk target biner, akurasi untuk target dengan ≤ 10 kelas, dan $R^2$ untuk target kontinu. Jika skor dari data sintetis mendekati skor dari data asli, data sintetis bisa menggantikan data asli untuk melatih model.

Contoh hasil pada California Housing dengan pengaturan default: selisih korelasi rata-rata **0,023**, dan median jarak baris sintetis ke data latih **0,45** dibandingkan **0,35** untuk baris asli yang belum pernah dilihat. Artinya baris sintetis tidak lebih dekat ke data latih daripada data baru yang wajar, jadi model tidak menyalin data. Scatter plot Longitude–Latitude sintetis juga membentuk peta California.

**Contoh pada data medis: Heart Failure Clinical Records** (UCI, 299 pasien, 13 kolom; `heart_failure.csv`). Hasil dengan pengaturan default:

| Aspek | Hasil |
|---|---|
| Rata-rata kolom dan persentase biner | Mirip data asli: meninggal 32,1% → 30,1%, laki-laki 64,9% → 63,5%, merokok 32,1% → 31,4% |
| Selisih korelasi rata-rata | 0,051 (korelasi terpenting, `time` vs `DEATH_EVENT`: −0,53 → −0,52) |
| Penumpukan nilai di minimum `creatinine_phosphokinase` | 36 baris sebelum transformasi log → 0 baris sesudahnya |
| Uji kegunaan (prediksi kematian, random forest) | AUC 0,911 jika dilatih dengan data asli vs **0,892** jika dilatih dengan data sintetis |
| Uji privasi | Tidak ada salinan persis, tetapi jarak ke baris latih terdekat 23% lebih kecil daripada pasien asli yang tidak terlihat (1,63 vs 2,12), sehingga aplikasi memberi peringatan |

Kesimpulannya, data sintetis hampir sama bergunanya dengan data asli untuk memprediksi kematian pasien. Namun pada dataset sekecil ini (239 pasien untuk pelatihan), label difusi berada dekat dengan pasien asli, jadi data sintetis belum boleh dianggap anonim. **Melindungi privasi pada dataset medis yang kecil lebih sulit**, dan aplikasi mendeteksi serta memperingatkan hal ini.

### 7.3 Class balancing: menyeimbangkan kelas

**Masalah.** Jika satu kelas jarang (misalnya 80% pasien selamat dan 20% meninggal), classifier cenderung memihak kelas mayoritas. Akibatnya bukan *overfitting*, melainkan **bias**: model banyak **melewatkan kasus kelas minoritas** (banyak *false negative* untuk kelas itu). Menyeimbangkan kelas memperbaiki bias ini, tetapi dengan **tukar-menukar (trade-off)**: model menangkap lebih banyak kasus minoritas (*recall* naik), tetapi juga lebih sering memberi alarm palsu (*precision* turun). Penyeimbangan tidak menghilangkan *false positive* dan *false negative* sekaligus.

**Ide.** Latih generator difusi **hanya pada baris kelas minoritas**, lalu buat baris sintetis untuk kelas itu sampai jumlahnya sama dengan kelas mayoritas.

**Eksperimen** (`balance_experiment.py`, hasil lengkap di `runs/balance_results.csv`). Data gagal jantung, target `DEATH_EVENT`. Sebanyak 10 pembagian acak 70/30 (stratified); 30% data uji selalu data asli dan tidak pernah disentuh, sedangkan generator hanya melihat baris minoritas di 70% data latih. Ada dua skenario: rasio alami (32% meninggal) dan 80/20 (pasien meninggal di data latih dikurangi menjadi 20%). Lima metode dibandingkan: tanpa penyeimbangan, *class weights* (kesalahan pada kelas minoritas diberi bobot lebih besar), *random oversampling* (menggandakan baris minoritas), SMOTE (interpolasi antar-tetangga minoritas, metode standar), dan generator difusi kami. Metrik dihitung untuk kelas minoritas (pasien meninggal), rata-rata 10 pembagian.

Skenario 80/20, random forest:

| Metode | Recall | Precision | F1 | Balanced accuracy |
|---|---|---|---|---|
| Tanpa penyeimbangan | 0,507 | 0,875 | 0,637 | 0,735 |
| Class weights | 0,686 | 0,777 | 0,726 | 0,795 |
| Random oversampling | 0,569 | 0,821 | 0,666 | 0,754 |
| SMOTE | 0,631 | 0,812 | 0,705 | 0,779 |
| **Generator difusi** | **0,724** | 0,770 | **0,743** | **0,809** |

Skenario 80/20, logistic regression:

| Metode | Recall | Precision | F1 | Balanced accuracy |
|---|---|---|---|---|
| Tanpa penyeimbangan | 0,472 | 0,809 | 0,590 | 0,708 |
| Class weights | 0,734 | 0,649 | **0,687** | **0,771** |
| Random oversampling | 0,721 | 0,640 | 0,676 | 0,763 |
| SMOTE | 0,693 | 0,644 | 0,664 | 0,753 |
| Generator difusi | 0,697 | 0,639 | 0,664 | 0,753 |

Perbandingan berpasangan (pada pembagian data yang sama) antara generator difusi dan metode lain, untuk F1:

| Skenario | Model | vs tanpa penyeimbangan | vs class weights | vs SMOTE | Kenaikan recall vs tanpa penyeimbangan |
|---|---|---|---|---|---|
| 80/20 | Random forest | +0,106 (menang 9/10) | +0,016 (5/10) | +0,037 (7/10) | +0,217 |
| 80/20 | Logistic regression | +0,074 (7/10) | −0,023 (3/10) | ±0,000 (4/10) | +0,224 |
| Alami (32%) | Random forest | +0,026 (8/10) | −0,008 (4/10) | −0,021 (2/10) | +0,093 |
| Alami (32%) | Logistic regression | −0,012 (4/10) | −0,005 (4/10) | −0,004 (4/10) | +0,079 |

Kesimpulan:

1. **Pada ketidakseimbangan yang kuat (80/20), penyeimbangan dengan generator difusi jelas membantu.** Tanpa penyeimbangan, model melewatkan sekitar separuh pasien yang meninggal (recall 0,47–0,51). Dengan generator difusi, recall naik sekitar 0,22 dan F1 naik 0,07–0,11.
2. **Generator difusi setara atau lebih baik dari SMOTE.** Dengan random forest, generator difusi memberi F1 dan balanced accuracy tertinggi dari kelima metode (menang dari SMOTE di 7 dari 10 pembagian). Dengan logistic regression hasilnya sama dengan SMOTE.
3. **Generator difusi bukan solusi ajaib.** *Class weights*, yang jauh lebih sederhana, memberi hasil yang sebanding (lebih baik dengan logistic regression).
4. **Pada rasio alami (32%), penyeimbangan hampir tidak berpengaruh** untuk metode apa pun, karena ketidakseimbangannya terlalu ringan.
5. Seperti yang diharapkan, menaikkan recall selalu disertai turunnya precision. AUC hampir tidak berubah (0,84–0,91), karena penyeimbangan menggeser ambang keputusan, bukan kemampuan model mengurutkan pasien berdasarkan risiko.

**Di aplikasi**, tab *Class balancing* menjalankan prosedur yang sama untuk CSV apa pun:

1. Pilih kolom kelas (kolom dengan 2–10 nilai berbeda). Aplikasi menampilkan jumlah dan persentase tiap kelas.
2. Tekan **Balance classes**. Untuk setiap kelas yang lebih kecil dari kelas terbesar, generator difusi dilatih hanya pada baris kelas itu, lalu baris sintetis ditambahkan sampai semua kelas sama besar.
3. **Uji manfaat:** random forest dilatih tanpa dan dengan baris sintetis pada 5 pembagian acak 70/30, selalu diuji pada baris asli yang tidak pernah dilihat generator. Aplikasi menampilkan recall, precision dan F1 kelas paling jarang, macro F1, dan balanced accuracy, beserta selisihnya terhadap tanpa penyeimbangan. Kesimpulannya ditulis otomatis: membantu, tidak ada perbedaan jelas, atau memperburuk. Penilaian memakai 5 pembagian karena satu pembagian saja terlalu berisik: pada data yang sama, satu pembagian bisa memberi F1 +0,005 dan pembagian lain −0,022.
4. Aplikasi memberi peringatan jika suatu kelas diperbesar lebih dari 5× (misalnya kualitas wine 3: 10 baris asli → 671 sintetis), karena baris sintetisnya hanya variasi dari sedikit baris asli.
5. **Unduh CSV seimbang.** CSV berisi semua baris asli ditambah baris sintetis, dengan kolom `is_synthetic` (0/1). Baris sintetis sebaiknya hanya dipakai untuk melatih model, dan evaluasi selalu dilakukan pada baris asli.

Contoh hasil di aplikasi (rata-rata 5 pembagian): pada data gagal jantung yang dibuat 80/20, recall naik +0,187 dan balanced accuracy naik +0,058. Pada data asli (32% meninggal) recall naik +0,069 tetapi F1 hampir tidak berubah (+0,01). Pada wine quality (6 kelas), macro F1 naik +0,052 dan lebih baik di 5 dari 5 pembagian.

### 7.4 Paper reproduction

Menampilkan hasil `train.py` untuk semua dataset dalam bentuk tabel seperti Tabel 1 paper (waktu pelabelan, waktu latih, waktu sampling, KL rata-rata, cross entropy) beserta gambar hasilnya.

---

## 8. Cara Menjalankan

```sh
# Instalasi
python3.12 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install numpy scipy pandas matplotlib scikit-learn streamlit imbalanced-learn

# Uji implementasi
.venv/bin/python sgm.py

# Reproduksi eksperimen paper
.venv/bin/python train.py gaussians moons spirals checkerboard   # ±1 menit
.venv/bin/python train.py power gas hepmass miniboone            # ±1 jam di CPU

# Eksperimen tambahan
.venv/bin/python ablation_refs.py        # Bagian 5.5, ±15 menit
.venv/bin/python balance_experiment.py   # Bagian 7.3, ±2 menit (butuh heart_failure.csv)

# Aplikasi
.venv/bin/streamlit run app.py
```

Petunjuk mengunduh dataset UCI ada di `README.md`.

---

## 9. Keterbatasan dan Pengembangan Lanjutan

- **Dimensi tinggi.** Estimasi skor Monte Carlo dengan 1000 titik referensi membuat label tidak konsisten antar-potongan pada data 21–43 dimensi, sehingga sebaran generator menyusut (std 0,76–0,96 pada HEPMASS dan MINIBOONE; lihat Bagian 5.4). Label hasil ODE-nya sendiri tetap akurat. Memakai satu batch referensi bersama mengurangi masalah ini (KL HEPMASS turun hampir setengahnya, Bagian 5.5), tetapi belum menghilangkannya.
- **Ekor distribusi yang berat.** Tanpa transformasi log, nilai ekstrem yang langka hilang dan nilai kecil menumpuk di minimum (contoh: AveOccup pada California Housing, simpangan baku sintetis 1,40 vs 10,39 asli). Transformasi log (Bagian 7.2) menghilangkan penumpukan dan memperbaiki sebagian ekor, tetapi nilai yang sangat langka tetap sulit direproduksi (maksimum `serum_creatinine` pada data gagal jantung: 9,4 asli vs 6,8 sintetis).
- **Kolom diskret/kategorikal** belum didukung secara khusus (hanya dibulatkan).
- **Pengembangan:** perbandingan langsung dengan normalizing flow atau GAN pada dataset yang sama; augmentasi data kelas minoritas untuk klasifikasi tidak seimbang; menjalankan pelabelan di GPU.

---

## 10. Referensi

1. Liu, Y., Yang, M., Zhang, Z., Bao, F., Cao, Y., & Zhang, G. (2024). Diffusion-Model-Assisted Supervised Learning of Generative Models for Density Estimation. *Journal of Machine Learning for Modeling and Computing*, 5(1), 25–38.
2. Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., & Poole, B. (2021). Score-Based Generative Modeling through Stochastic Differential Equations. *ICLR*.
3. Papamakarios, G., Pavlakou, T., & Murray, I. (2017). Masked Autoregressive Flow for Density Estimation. *NeurIPS*.
4. Grathwohl, W., Chen, R. T. Q., Bettencourt, J., Sutskever, I., & Duvenaud, D. (2018). FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models. arXiv:1810.01367.
5. Kode resmi penulis: https://github.com/mlmathphy/supervised_generative_model
