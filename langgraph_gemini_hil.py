"""
True LangGraph Human-in-the-Loop dengan Gemini Flash
Menggunakan interrupt() function yang sebenarnya dari LangGraph
"""

import json
import os
import uuid
from datetime import datetime, timedelta
from typing import TypedDict, Literal, Dict, Any

# LangGraph imports
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.types import interrupt, Command

# Gemini import
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# State definition
class BookingState(TypedDict):
    user_input: str
    extracted_data: Dict[str, Any]
    iteration_count: int
    status: str
    messages: list

def setup_gemini():
    """Setup Gemini model"""
    if not GEMINI_AVAILABLE or not os.getenv("GOOGLE_API_KEY"):
        return None

    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-preview-05-20",
        temperature=0.3,
        max_tokens=1000
    )

def extract_data_node(state: BookingState) -> BookingState:
    """Node untuk ekstraksi data menggunakan Gemini"""
    print("🤖 Node: Ekstraksi data dengan Gemini...")

    model = setup_gemini()
    if not model:
        return {
            "extracted_data": {"error": "Gemini not available"},
            "status": "error",
            "messages": state.get("messages", []) + ["❌ Gemini tidak tersedia"]
        }

    user_input = state["user_input"]
    existing_data = state.get("extracted_data", {})

    prompt = f"""
    Analisis permintaan booking hotel dan ekstrak/update informasi:

    Input user: "{user_input}"
    Data existing: {json.dumps(existing_data, ensure_ascii=False)}

    Ekstrak dalam format JSON:
    {{
        "lokasi": "nama kota/daerah (null jika tidak disebutkan)",
        "tanggal_checkin": "YYYY-MM-DD (null jika tidak disebutkan eksplisit)",
        "tanggal_checkout": "YYYY-MM-DD (null jika tidak disebutkan eksplisit)",
        "jumlah_malam": "jumlah malam integer (null jika tidak disebutkan)",
        "jumlah_tamu": "jumlah tamu integer (null jika tidak disebutkan)",
        "budget": "budget rupiah integer (null jika tidak disebutkan)",
        "preferensi": ["array preferensi"]
    }}

    ATURAN PENTING:
    1. PERTAHANKAN DATA EXISTING: Jangan hapus atau ubah data yang sudah ada kecuali ada update eksplisit
    2. JANGAN AUTO-FILL: Hanya isi field jika EKSPLISIT disebutkan dalam input baru
    3. MERGE DATA: Gabungkan data existing dengan data baru

    Contoh BENAR:
    - Data existing: {{"lokasi": "Nusa Dua", "budget": 2000000}}
    - Input baru: "preferensi spa"
    - Output: {{"lokasi": "Nusa Dua", "budget": 2000000, "preferensi": ["spa"]}}

    Contoh SALAH:
    - Data existing: {{"lokasi": "Nusa Dua", "budget": 2000000}}
    - Input baru: "preferensi spa"
    - Output: {{"lokasi": null, "preferensi": ["spa"]}} ← SALAH! Data existing hilang

    Contoh deteksi yang VALID:
    - "20-25 juni 2025" -> checkin: "2025-06-20", checkout: "2025-06-25", jumlah_malam: 5
    - "checkin 20 agustus untuk 3 malam" -> checkin: "2025-08-20", checkout: "2025-08-23", jumlah_malam: 3
    - "2 orang" -> jumlah_tamu: 2
    - "budget 2 juta" -> budget: 2000000
    - "budget 10 juta rupiah" -> budget: 10000000
    - "maksimal 5.5 juta" -> budget: 5500000
    - "budget Rp 3.000.000" -> budget: 3000000

    WAJIB:
    - Pertahankan SEMUA data existing yang tidak diupdate
    - Budget dalam rupiah integer (2000000 untuk 2 juta)
    - Gabungkan preferensi baru dengan existing (jangan replace)
    - Berikan hanya JSON tanpa penjelasan
    - Sekarang tahun 2025
    """

    try:
        response = model.invoke(prompt)
        response_text = response.content.strip()

        # Clean response
        if response_text.startswith("```json"):
            response_text = response_text.replace("```json", "").replace("```", "").strip()
        elif response_text.startswith("```"):
            response_text = response_text.replace("```", "").strip()

        extracted_data = json.loads(response_text)

        # Validasi tanggal HANYA jika keduanya ada
        if extracted_data.get("tanggal_checkin") and extracted_data.get("tanggal_checkout"):
            try:
                checkin_date = datetime.strptime(extracted_data["tanggal_checkin"], "%Y-%m-%d")
                checkout_date = datetime.strptime(extracted_data["tanggal_checkout"], "%Y-%m-%d")

                # Validasi checkout > checkin
                if checkout_date <= checkin_date:
                    print(f"⚠️ Warning: Checkout date invalid, akan diperbaiki oleh user")
                else:
                    # Hitung jumlah malam berdasarkan selisih tanggal
                    nights = (checkout_date - checkin_date).days
                    extracted_data["jumlah_malam"] = nights
            except ValueError:
                print(f"⚠️ Warning: Format tanggal tidak valid")

        # Jika hanya checkin dan jumlah malam yang ada, hitung checkout
        elif extracted_data.get("tanggal_checkin") and extracted_data.get("jumlah_malam"):
            try:
                checkin_date = datetime.strptime(extracted_data["tanggal_checkin"], "%Y-%m-%d")
                nights = extracted_data["jumlah_malam"]
                checkout_date = checkin_date + timedelta(days=nights)
                extracted_data["tanggal_checkout"] = checkout_date.strftime("%Y-%m-%d")
            except ValueError:
                print(f"⚠️ Warning: Format tanggal checkin tidak valid")

        # TIDAK ada auto-fill untuk field yang kosong
        # Biarkan null jika tidak disebutkan eksplisit

        return {
            "extracted_data": extracted_data,
            "iteration_count": state.get("iteration_count", 0) + 1,
            "status": "extracted",
            "messages": state.get("messages", []) + [f"✅ Data diekstrak (iterasi {state.get('iteration_count', 0) + 1})"]
        }

    except Exception as e:
        print(f"❌ Error ekstraksi: {e}")
        return {
            "extracted_data": {"error": str(e)},
            "status": "error",
            "messages": state.get("messages", []) + [f"❌ Error: {e}"]
        }

def human_review_node(state: BookingState) -> Command[Literal["extract_data", "finalize"]]:
    """Node untuk Human-in-the-Loop review menggunakan interrupt()"""

    extracted_data = state["extracted_data"]
    iteration = state.get("iteration_count", 1)

    # Format data untuk display
    display_data = format_display_data(extracted_data)

    # INI ADALAH INTERRUPT() YANG SEBENARNYA DARI LANGGRAPH!
    human_decision = interrupt({
        "message": f"📋 REVIEW DATA EKSTRAKSI (Iterasi {iteration})",
        "instruction": "Silakan review data yang diekstrak AI. Ketik 'setuju' jika benar, atau berikan koreksi/tambahan.",
        "extracted_data": {
            "lokasi": display_data.get("lokasi") or "Belum disebutkan",
            "tanggal_checkin": display_data.get("tanggal_checkin_display") or "Belum disebutkan",
            "tanggal_checkout": display_data.get("tanggal_checkout_display") or "Belum disebutkan",
            "jumlah_malam": display_data.get("jumlah_malam") or "Belum disebutkan",
            "jumlah_tamu": display_data.get("jumlah_tamu") or "Belum disebutkan",
            "budget": display_data.get("budget_display", "Belum disebutkan"),
            "preferensi": display_data.get("preferensi", []) or "Tidak ada"
        },
        "raw_json": extracted_data,
        "options": [
            "'setuju' - Data sudah benar",
            "'selesai' - Lanjut ke booking",
            "Atau ketik koreksi/tambahan"
        ]
    })

    # Process human decision
    if human_decision.lower() in ["setuju", "ok", "benar"]:
        # Check completeness - field yang wajib diisi
        required_fields = {
            "lokasi": "lokasi tujuan",
            "tanggal_checkin": "tanggal check-in",
            "jumlah_malam": "jumlah malam",
            "jumlah_tamu": "jumlah tamu"
        }

        missing = []
        for field, label in required_fields.items():
            if not extracted_data.get(field):
                missing.append(label)

        if missing:
            # Continue to extract more data
            missing_text = ", ".join(missing)
            return Command(
                goto="extract_data",
                update={
                    "user_input": f"Data masih kurang: {missing_text}. Silakan lengkapi informasi yang belum disebutkan.",
                    "messages": state.get("messages", []) + [f"⚠️ Data belum lengkap: {missing_text}"]
                }
            )
        else:
            return Command(goto="finalize")

    elif human_decision.lower() in ["selesai", "lanjut"]:
        return Command(goto="finalize")

    else:
        # Process additional input
        return Command(
            goto="extract_data",
            update={
                "user_input": human_decision,
                "messages": state.get("messages", []) + [f"🔄 Memproses input tambahan: {human_decision}"]
            }
        )

def finalize_node(state: BookingState) -> BookingState:
    """Node untuk finalisasi data"""
    print("✅ Finalisasi data booking...")

    final_data = state["extracted_data"]
    iteration_count = state.get("iteration_count", 1)

    summary_message = f"""
🎉 DATA BOOKING FINAL:
🏨 Lokasi: {final_data.get('lokasi') or 'Tidak disebutkan'}
📅 Check-in: {final_data.get('tanggal_checkin') or 'Tidak disebutkan'}
📅 Check-out: {final_data.get('tanggal_checkout') or 'Tidak disebutkan'}
🌙 Jumlah malam: {final_data.get('jumlah_malam') or 'Tidak disebutkan'}
👥 Jumlah tamu: {final_data.get('jumlah_tamu') or 'Tidak disebutkan'}
💰 Budget: {format_budget(final_data.get('budget'))}
⭐ Preferensi: {', '.join(final_data.get('preferensi', [])) if final_data.get('preferensi') else 'Tidak ada'}

📊 Total iterasi HIL: {iteration_count}
🤖 Model AI: Google Gemini 2.5 Flash
✅ Status: Siap untuk pencarian hotel
    """

    return {
        "status": "completed",
        "messages": state.get("messages", []) + [summary_message]
    }

def format_display_data(data):
    """Format data untuk display"""
    display_data = data.copy()

    # Format budget dengan format Indonesia
    if data.get("budget"):
        budget = data["budget"]
        # Format dengan titik sebagai pemisah ribuan
        budget_formatted = f"Rp {budget:,}".replace(",", ".")
        display_data["budget_display"] = budget_formatted
    else:
        display_data["budget_display"] = "Tidak disebutkan"

    # Mapping bulan ke bahasa Indonesia
    month_mapping = {
        "January": "Januari", "February": "Februari", "March": "Maret",
        "April": "April", "May": "Mei", "June": "Juni",
        "July": "Juli", "August": "Agustus", "September": "September",
        "October": "Oktober", "November": "November", "December": "Desember"
    }

    # Format tanggal checkin
    if data.get("tanggal_checkin"):
        try:
            date_obj = datetime.strptime(data["tanggal_checkin"], "%Y-%m-%d")
            english_date = date_obj.strftime("%d %B %Y")
            # Replace English month with Indonesian
            indonesian_date = english_date
            for eng_month, ind_month in month_mapping.items():
                indonesian_date = indonesian_date.replace(eng_month, ind_month)
            display_data["tanggal_checkin_display"] = indonesian_date
        except:
            display_data["tanggal_checkin_display"] = data["tanggal_checkin"]

    # Format tanggal checkout
    if data.get("tanggal_checkout"):
        try:
            date_obj = datetime.strptime(data["tanggal_checkout"], "%Y-%m-%d")
            english_date = date_obj.strftime("%d %B %Y")
            # Replace English month with Indonesian
            indonesian_date = english_date
            for eng_month, ind_month in month_mapping.items():
                indonesian_date = indonesian_date.replace(eng_month, ind_month)
            display_data["tanggal_checkout_display"] = indonesian_date
        except:
            display_data["tanggal_checkout_display"] = data["tanggal_checkout"]

    return display_data

def format_budget(budget):
    """Format budget untuk display dengan format Indonesia"""
    if not budget:
        return "Tidak disebutkan"

    # Format dengan titik sebagai pemisah ribuan (format Indonesia)
    return f"Rp {budget:,}".replace(",", ".")

def create_langgraph_hil():
    """Buat LangGraph dengan HIL menggunakan interrupt()"""

    builder = StateGraph(BookingState)

    # Add nodes
    builder.add_node("extract_data", extract_data_node)
    builder.add_node("human_review", human_review_node)
    builder.add_node("finalize", finalize_node)

    # Add edges
    builder.add_edge(START, "extract_data")
    builder.add_edge("extract_data", "human_review")
    # human_review menggunakan Command untuk routing
    builder.add_edge("finalize", END)

    # Compile dengan checkpointer (WAJIB untuk interrupt!)
    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)

def run_langgraph_hil_demo():
    """Jalankan demo LangGraph HIL yang sebenarnya"""
    print("🚀 LANGGRAPH HUMAN-IN-THE-LOOP DENGAN GEMINI FLASH")
    print("=" * 60)
    print("Demo menggunakan interrupt() function yang sebenarnya!")
    print("=" * 60)

    # Check requirements
    if not GEMINI_AVAILABLE:
        print("❌ LangChain Google GenAI tidak tersedia!")
        return

    if not os.getenv("GOOGLE_API_KEY"):
        print("❌ GOOGLE_API_KEY tidak ditemukan!")
        return

    # Get initial input
    print("\n📝 MASUKKAN PENCARIAN HOTEL:")
    user_input = input("👤 Kata pencarian hotel: ").strip()

    if not user_input:
        print("❌ Input tidak boleh kosong!")
        return

    # Create graph
    graph = create_langgraph_hil()

    # Create config dengan thread ID (WAJIB untuk persistence)
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    # Initial state
    initial_state = {
        "user_input": user_input,
        "extracted_data": {},
        "iteration_count": 0,
        "status": "start",
        "messages": []
    }

    try:
        print("\n🔄 Memulai LangGraph execution...")

        # Jalankan graph sampai interrupt pertama
        result = graph.invoke(initial_state, config=config)

        # Handle interrupts dalam loop
        while "__interrupt__" in result:
            interrupt_data = result["__interrupt__"][0]

            print("\n" + "="*60)
            print(f"🛑 LANGGRAPH INTERRUPT DETECTED!")
            print("="*60)
            print(f"📋 {interrupt_data.value['message']}")
            print("="*60)
            print(f"Instruksi: {interrupt_data.value['instruction']}")

            # Display extracted data
            extracted_info = interrupt_data.value['extracted_data']
            print(f"\n📊 DATA YANG DIEKSTRAK:")
            print(f"🏨 Lokasi: {extracted_info['lokasi']}")
            print(f"📅 Check-in: {extracted_info['tanggal_checkin']}")
            print(f"📅 Check-out: {extracted_info['tanggal_checkout']}")
            print(f"🌙 Jumlah malam: {extracted_info['jumlah_malam']}")
            print(f"👥 Jumlah tamu: {extracted_info['jumlah_tamu']}")
            print(f"💰 Budget: {extracted_info['budget']}")

            # Display preferensi
            if extracted_info.get('preferensi') and extracted_info['preferensi'] != "Tidak ada":
                if isinstance(extracted_info['preferensi'], list) and extracted_info['preferensi']:
                    print(f"⭐ Preferensi: {', '.join(extracted_info['preferensi'])}")
                else:
                    print(f"⭐ Preferensi: Tidak ada")
            else:
                print(f"⭐ Preferensi: Tidak ada")

            # Show options
            print(f"\n📋 PILIHAN:")
            for option in interrupt_data.value['options']:
                print(f"• {option}")

            # Get human input
            print("\n" + "-"*60)
            human_input = input("👤 Input Anda: ").strip()

            # Resume graph dengan Command
            print(f"\n🔄 Resuming LangGraph dengan input: '{human_input}'")
            result = graph.invoke(Command(resume=human_input), config=config)

        # Final result
        print("\n" + "="*60)
        print("🎉 LANGGRAPH EXECUTION COMPLETED!")
        print("="*60)

        if result.get("status") == "completed":
            print("✅ Proses HIL berhasil!")
            if "messages" in result:
                for msg in result["messages"]:
                    if "DATA BOOKING FINAL" in msg:
                        print(f"\n{msg}")

        print(f"\n📚 YANG BARU SAJA TERJADI:")
        print("✅ LangGraph interrupt() function digunakan")
        print("✅ State persistence dengan MemorySaver")
        print("✅ Command(resume=value) untuk melanjutkan")
        print("✅ Graph routing berdasarkan human decision")
        print("✅ True Human-in-the-Loop implementation!")

    except KeyboardInterrupt:
        print("\n\n👋 Demo dibatalkan oleh user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Load environment
    try:
        with open('.env', 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value
    except:
        pass

    run_langgraph_hil_demo()
