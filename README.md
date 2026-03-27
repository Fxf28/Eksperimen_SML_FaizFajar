# 📊 Eksperimen SML - Fatalities Israeli-Palestinian Conflict

Repository ini berisi eksperimen Sistem Machine Learning untuk menganalisis dataset **Fatalities in the Israeli-Palestinian Conflict** dan membangun pipeline preprocessing otomatis.

---

## 📁 Struktur Repository

```bash

Eksperimen_SML_FaizFajar/
│
├── raw_dataset.csv
├── requirements.txt
│
├── preprocessing/
│   ├── Eksperimen_faiz-fajar.ipynb
│   ├── automate_faiz-fajar.py
│   └── preprocessing_output/
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       ├── y_test.csv
│       ├── onehot_encoder.pkl
│       ├── scaler.pkl
│       └── final_dataset.csv
│
└── .github/
└── workflows/
└── preprocessing.yml

```

---

## 📌 Deskripsi Dataset

- **Nama**: Fatalities in the Israeli-Palestinian Conflict
- **Sumber**: Kaggle
- **Periode**: 2000 – 2023
- **Jumlah Data**: 11.124 entri

Dataset berisi informasi korban tewas dalam konflik, termasuk:

- usia
- kewarganegaraan
- lokasi kejadian
- jenis luka
- pihak yang membunuh

---

## 🎯 Tujuan Proyek

Membangun model klasifikasi untuk memprediksi:

> **`took_part_in_the_hostilities`**

- `0` → Tidak terlibat
- `1` → Terlibat

---

## ⚙️ Preprocessing Pipeline

Pipeline preprocessing meliputi:

- Data cleaning
- Handling missing values
- Feature engineering (tanggal → tahun & bulan)
- Encoding (OneHotEncoder)
- Scaling (StandardScaler)
- Train-test split (80:20)

---

## 🤖 Automasi Preprocessing

Script utama:

```bash

preprocessing/automate_faiz-fajar.py

```

### ▶️ Cara Menjalankan

```bash
python preprocessing/automate_faiz-fajar.py
```

Output akan otomatis tersimpan di:

```bash
preprocessing/preprocessing_output/
```

---

## 🔄 GitHub Actions (Automation)

Workflow:

```bash
.github/workflows/preprocessing.yml
```

Fungsi:

- Menjalankan preprocessing otomatis saat `push`
- Menghasilkan dataset terbaru
- Commit hasil preprocessing secara otomatis

---

## 📦 Dependencies

Install semua dependency dengan:

```bash
pip install -r requirements.txt
```

---

## 👨‍💻 Author

**Faiz Fajar**
Eksperimen Sistem Machine Learning

---
