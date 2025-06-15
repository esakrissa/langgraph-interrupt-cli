# 🏨 Hotel Booking Agent dengan Human-in-the-Loop

Sistem booking hotel yang menggunakan **LangGraph** dengan fitur **Human-in-the-Loop (HIL)** dan **Google Gemini 2.5 Flash**. Agent ini memungkinkan kolaborasi antara AI dan manusia untuk ekstraksi data booking hotel dengan interface CLI yang modern.

## 🌟 Fitur Utama

- **🤖 Google Gemini 2.5 Flash**: Ekstraksi data dari natural language Indonesia
- **👤 True LangGraph HIL**: Menggunakan `interrupt()` function untuk review manusia
- **🇮🇩 Indonesian Localization**: Format "Rp 10.000.000", "25 Juli 2025"
- **🎨 Modern CLI**: Interface cantik dengan Rich library
- **🔄 Iterative Refinement**: User dapat menambah/mengoreksi data di setiap langkah

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Install dependencies
pip install -r requirements.txt

# Setup API key
cp .env.example .env
# Edit .env dan masukkan GOOGLE_API_KEY=your_api_key
```

### 2. Get Google API Key
1. Kunjungi: https://aistudio.google.com/app/apikey
2. Buat API key baru untuk Gemini
3. Masukkan ke file `.env`

### 3. Run Application
```bash
python hotel_booking_cli.py
```

## 🎯 Contoh Penggunaan

### Input Lengkap
```
Input: "hotel di ubud tanggal 20-25 juni 2025 untuk 2 orang budget 5 juta"

AI Extract:
✅ Lokasi: Ubud
✅ Check-in: 20 Juni 2025
✅ Check-out: 25 Juni 2025
✅ Jumlah tamu: 2
✅ Budget: Rp 5.000.000
```

### Input Minimal + Iterative
```
Input: "carikan hotel di nusa dua"

AI Extract:
✅ Lokasi: Nusa Dua
❌ Tanggal: Belum disebutkan

User: "checkin 15 juli checkout 18 juli"
User: "2 orang budget 3 juta"
```

## 📁 Struktur File

```
hotel-booking-hil/
├── hotel_booking_cli.py          # Modern CLI dengan Rich interface
├── langgraph_gemini_hil.py       # LangGraph HIL core logic
├── requirements.txt              # Dependencies
├── .env.example                  # Environment template
├── .env                         # Environment variables (user created)
└── README.md                    # Documentation
```

## 🔧 Status & Issues

### ✅ Sudah Tersedia
- LangGraph HIL implementation dengan `interrupt()`
- Google Gemini 2.5 Flash integration
- Modern CLI interface dengan Rich library
- Indonesian formatting (currency, dates)

### ⚠️ Known Issues
- **Import Error**: `hotel_booking_cli.py` mengimport dari `langgraph.langgraph_gemini_hil` yang tidak ada
- **Missing Functions**: Beberapa fungsi yang diimport belum diimplementasi

### 🚧 Next Steps
- Fix import issues
- Test basic functionality
- Add unit tests

## 🛠️ Komponen Utama

### LangGraph State
```python
class BookingState(TypedDict):
    user_input: str                   # Input user
    extracted_data: Dict[str, Any]    # Data yang diekstrak AI
    iteration_count: int              # Jumlah iterasi HIL
    status: str                       # Status proses
```

### HIL Workflow
1. **extract_data_node**: Gemini ekstrak data dari natural language
2. **human_review_node**: `interrupt()` untuk review manusia
3. **finalize_node**: Finalisasi data booking

## 🔧 Troubleshooting

### Common Issues

**❌ "GOOGLE_API_KEY tidak ditemukan"**
```bash
cp .env.example .env
# Edit .env dan masukkan API key
```

**❌ Import Error: "No module named 'langgraph.langgraph_gemini_hil'"**
- Fix: Ganti import di `hotel_booking_cli.py` line 69 menjadi `from langgraph_gemini_hil import`

**❌ Missing Dependencies**
```bash
pip install langchain-google-genai rich
```

## 📚 Resources

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Google Gemini API](https://makersuite.google.com/app/apikey)
- [Rich Library Documentation](https://rich.readthedocs.io/)

---

**🎉 Happy Building with Human-in-the-Loop! 🏨✨**
