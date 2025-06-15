"""
Hotel Booking Agent - Modern CLI Frontend
Menggunakan Rich library untuk tampilan CLI yang elegan dan modern
"""

import os
import json
import uuid
from datetime import datetime
from typing import Dict, Any, Optional

# Rich imports untuk UI yang cantik
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.theme import Theme
from rich.text import Text
from rich.columns import Columns
from rich.align import Align
from rich.layout import Layout
from rich.live import Live
import time

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

# Custom theme untuk hotel booking
hotel_theme = Theme({
    "info": "cyan",
    "warning": "yellow bold",
    "success": "green bold", 
    "error": "red bold",
    "heading": "magenta bold",
    "highlight": "blue bold",
    "hotel": "gold1 bold",
    "price": "green bold",
    "date": "cyan bold",
    "location": "blue bold",
    "guest": "purple bold"
})

# Initialize Rich console
console = Console(theme=hotel_theme)

# Import dari file sebelumnya
from langgraph_gemini_hil import (
    BookingState, setup_gemini, extract_data_node, 
    format_display_data, format_budget, create_langgraph_hil
)

class HotelBookingCLI:
    """Modern CLI untuk Hotel Booking dengan Rich UI"""
    
    def __init__(self):
        self.console = console
        self.graph = None
        self.config = None
        
    def show_welcome(self):
        """Tampilkan welcome screen yang cantik"""
        welcome_text = Text()
        welcome_text.append("🏨 ", style="hotel")
        welcome_text.append("HOTEL BOOKING AGENT", style="heading")
        welcome_text.append(" 🏨", style="hotel")
        
        subtitle = Text("Human-in-the-Loop dengan Google Gemini 2.5 Flash", style="info")
        
        welcome_panel = Panel(
            Align.center(welcome_text + "\n" + subtitle),
            border_style="hotel",
            padding=(1, 2)
        )
        
        self.console.print()
        self.console.print(welcome_panel)
        self.console.print()
        
        # Show features
        features_table = Table(show_header=False, box=None, padding=(0, 2))
        features_table.add_column(style="success")
        features_table.add_column(style="info")
        
        features_table.add_row("✅", "AI ekstraksi data otomatis dari bahasa natural")
        features_table.add_row("✅", "Human review dan koreksi di setiap langkah")
        features_table.add_row("✅", "Format Indonesia (Rp 10.000.000, Juli)")
        features_table.add_row("✅", "LangGraph interrupt() untuk true HIL")
        features_table.add_row("✅", "State persistence dan recovery")
        
        features_panel = Panel(
            features_table,
            title="[highlight]Fitur Unggulan[/highlight]",
            border_style="cyan"
        )
        
        self.console.print(features_panel)
        self.console.print()
    
    def check_requirements(self) -> bool:
        """Check system requirements dengan progress bar"""
        requirements = [
            ("LangGraph", GEMINI_AVAILABLE),
            ("Google API Key", bool(os.getenv("GOOGLE_API_KEY"))),
            ("Environment", True)
        ]
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            
            task = progress.add_task("Checking requirements...", total=len(requirements))
            
            for name, status in requirements:
                time.sleep(0.5)  # Simulate checking
                if status:
                    self.console.print(f"✅ {name}: [success]OK[/success]")
                else:
                    self.console.print(f"❌ {name}: [error]FAILED[/error]")
                    if name == "Google API Key":
                        self.console.print("[error]Set GOOGLE_API_KEY di file .env[/error]")
                    return False
                progress.advance(task)
        
        return True
    
    def get_user_input(self) -> str:
        """Get user input dengan styling yang cantik"""
        self.console.print()
        input_panel = Panel(
            "Masukkan permintaan booking hotel Anda\n"
            "[dim]Contoh: 'hotel di ubud tanggal 20-25 juni 2025 untuk 2 orang'[/dim]",
            title="[highlight]📝 Input Pencarian[/highlight]",
            border_style="blue"
        )
        self.console.print(input_panel)
        
        user_input = Prompt.ask(
            "[hotel]🏨 Pencarian hotel",
            console=self.console
        )
        
        return user_input
    
    def display_extracted_data(self, data: Dict[str, Any], iteration: int):
        """Display extracted data dengan format yang cantik"""
        
        # Header
        header = Text()
        header.append("🤖 ", style="info")
        header.append(f"DATA EKSTRAKSI AI (Iterasi {iteration})", style="heading")
        
        # Create table untuk data
        data_table = Table(show_header=False, box=None, padding=(0, 1))
        data_table.add_column("Field", style="highlight", width=15)
        data_table.add_column("Value", style="info")
        
        # Format data untuk display
        display_data = format_display_data(data)
        
        # Add rows
        data_table.add_row("🏨 Lokasi:", display_data.get("lokasi") or "[dim]Belum disebutkan[/dim]")
        data_table.add_row("📅 Check-in:", display_data.get("tanggal_checkin_display") or "[dim]Belum disebutkan[/dim]")
        data_table.add_row("📅 Check-out:", display_data.get("tanggal_checkout_display") or "[dim]Belum disebutkan[/dim]")
        data_table.add_row("🌙 Jumlah malam:", str(display_data.get("jumlah_malam") or "[dim]Belum disebutkan[/dim]"))
        data_table.add_row("👥 Jumlah tamu:", str(display_data.get("jumlah_tamu") or "[dim]Belum disebutkan[/dim]"))
        data_table.add_row("💰 Budget:", display_data.get("budget_display", "[dim]Belum disebutkan[/dim]"))
        
        # Preferensi
        if data.get("preferensi"):
            prefs = ", ".join(data["preferensi"])
            data_table.add_row("⭐ Preferensi:", prefs)
        else:
            data_table.add_row("⭐ Preferensi:", "[dim]Tidak ada[/dim]")
        
        # Create main panel
        main_panel = Panel(
            data_table,
            title=header,
            border_style="cyan",
            padding=(1, 2)
        )
        
        self.console.print()
        self.console.print(main_panel)
        
        # Show raw JSON in collapsible format
        if Confirm.ask("[dim]Tampilkan raw JSON?[/dim]", default=False, console=self.console):
            json_text = json.dumps(data, indent=2, ensure_ascii=False)
            json_panel = Panel(
                json_text,
                title="[dim]Raw JSON Data[/dim]",
                border_style="dim",
                expand=False
            )
            self.console.print(json_panel)
    
    def get_user_feedback(self) -> str:
        """Get user feedback dengan options yang jelas"""
        
        # Options table
        options_table = Table(show_header=False, box=None)
        options_table.add_column("Option", style="success", width=12)
        options_table.add_column("Description", style="info")
        
        options_table.add_row("'setuju'", "Data sudah benar, lanjutkan")
        options_table.add_row("'selesai'", "Lanjut ke booking dengan data ini")
        options_table.add_row("atau", "Ketik koreksi/tambahan data")
        
        options_panel = Panel(
            options_table,
            title="[highlight]📋 Pilihan Anda[/highlight]",
            border_style="yellow"
        )
        
        self.console.print(options_panel)
        
        feedback = Prompt.ask(
            "[highlight]👤 Input Anda",
            console=self.console
        )
        
        return feedback
    
    def show_final_summary(self, final_data: Dict[str, Any], iterations: int):
        """Tampilkan summary final yang cantik"""
        
        # Create summary table
        summary_table = Table(show_header=False, box=None, padding=(0, 1))
        summary_table.add_column("Field", style="highlight", width=15)
        summary_table.add_column("Value", style="success")
        
        summary_table.add_row("🏨 Lokasi:", final_data.get("lokasi") or "Tidak disebutkan")
        summary_table.add_row("📅 Check-in:", final_data.get("tanggal_checkin") or "Tidak disebutkan")
        summary_table.add_row("📅 Check-out:", final_data.get("tanggal_checkout") or "Tidak disebutkan")
        summary_table.add_row("🌙 Jumlah malam:", str(final_data.get("jumlah_malam") or "Tidak disebutkan"))
        summary_table.add_row("👥 Jumlah tamu:", str(final_data.get("jumlah_tamu") or "Tidak disebutkan"))
        summary_table.add_row("💰 Budget:", format_budget(final_data.get("budget")))
        
        if final_data.get("preferensi"):
            prefs = ", ".join(final_data["preferensi"])
            summary_table.add_row("⭐ Preferensi:", prefs)
        else:
            summary_table.add_row("⭐ Preferensi:", "Tidak ada")
        
        # Stats table
        stats_table = Table(show_header=False, box=None)
        stats_table.add_column("Metric", style="info")
        stats_table.add_column("Value", style="highlight")
        
        stats_table.add_row("Total iterasi HIL:", str(iterations))
        stats_table.add_row("Model AI:", "Google Gemini 2.5 Flash")
        stats_table.add_row("Status:", "✅ Siap untuk pencarian hotel")
        
        # Combine in columns
        summary_panel = Panel(
            summary_table,
            title="[success]🎉 DATA BOOKING FINAL[/success]",
            border_style="green"
        )
        
        stats_panel = Panel(
            stats_table,
            title="[info]📊 Statistik Proses[/info]",
            border_style="blue"
        )
        
        self.console.print()
        self.console.print(summary_panel)
        self.console.print(stats_panel)
        
        # Next steps
        next_steps = Table(show_header=False, box=None)
        next_steps.add_column("Step", style="success")
        next_steps.add_column("Description", style="info")
        
        next_steps.add_row("1.", "🔍 Pencarian hotel di database")
        next_steps.add_row("2.", "🎯 Filtering berdasarkan kriteria")
        next_steps.add_row("3.", "⭐ Ranking berdasarkan preferensi")
        next_steps.add_row("4.", "📋 Presentasi pilihan hotel")
        
        next_panel = Panel(
            next_steps,
            title="[highlight]🚀 Langkah Selanjutnya[/highlight]",
            border_style="magenta"
        )
        
        self.console.print(next_panel)
    
    def show_error(self, error_msg: str):
        """Tampilkan error dengan styling yang baik"""
        error_panel = Panel(
            f"❌ {error_msg}",
            title="[error]Error[/error]",
            border_style="red"
        )
        self.console.print(error_panel)
    
    def run(self):
        """Main run function"""
        try:
            # Welcome screen
            self.show_welcome()
            
            # Check requirements
            if not self.check_requirements():
                return
            
            # Get user input
            user_input = self.get_user_input()
            if not user_input.strip():
                self.show_error("Input tidak boleh kosong!")
                return
            
            # Initialize graph
            with Progress(
                SpinnerColumn(),
                TextColumn("Initializing LangGraph..."),
                console=self.console
            ) as progress:
                task = progress.add_task("setup", total=None)
                self.graph = create_langgraph_hil()
                self.config = {"configurable": {"thread_id": str(uuid.uuid4())}}
                progress.update(task, completed=100)
            
            # Initial state
            initial_state = {
                "user_input": user_input,
                "extracted_data": {},
                "iteration_count": 0,
                "status": "start",
                "messages": []
            }
            
            self.console.print("\n[info]🔄 Memulai proses ekstraksi data...[/info]")
            
            # Run graph until first interrupt
            result = self.graph.invoke(initial_state, self.config)
            
            # Handle interrupts
            while "__interrupt__" in result:
                interrupt_data = result["__interrupt__"][0]
                
                # Display extracted data
                extracted_data = interrupt_data.value["raw_json"]
                iteration = interrupt_data.value.get("iteration", 1)
                
                self.display_extracted_data(extracted_data, iteration)
                
                # Get user feedback
                feedback = self.get_user_feedback()
                
                # Show processing
                with Progress(
                    SpinnerColumn(),
                    TextColumn("Memproses input Anda..."),
                    console=self.console
                ) as progress:
                    task = progress.add_task("processing", total=None)
                    result = self.graph.invoke(Command(resume=feedback), self.config)
                    progress.update(task, completed=100)
            
            # Show final result
            if result.get("status") == "completed":
                final_data = result["extracted_data"]
                iterations = result.get("iteration_count", 1)
                self.show_final_summary(final_data, iterations)
                
                self.console.print("\n[success]🎉 Proses HIL berhasil diselesaikan![/success]")
            else:
                self.show_error("Proses tidak dapat diselesaikan")
                
        except KeyboardInterrupt:
            self.console.print("\n[warning]👋 Proses dibatalkan oleh user[/warning]")
        except Exception as e:
            self.show_error(f"Terjadi kesalahan: {str(e)}")

def main():
    """Main function"""
    app = HotelBookingCLI()
    app.run()

if __name__ == "__main__":
    main()
