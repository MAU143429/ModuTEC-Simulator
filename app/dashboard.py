import os
import queue
import numpy as np
from PIL import Image
from utils.styles import *
import customtkinter as ctk
from scipy.io import wavfile
from tkinter import filedialog
import matplotlib.pyplot as plt
from utils import windowCenter as wc
from CTkMessagebox import CTkMessagebox
from core.statistics.metrics import NCCPairer
from matplotlib.ticker import MultipleLocator
from app.ui.SamplingStream import SamplingStream
from core.statistics.metrics import digital_accuracy
from core.audio.AudioController import AudioController
from app.ui.VerticalRightToolbar import VerticalRightToolbar
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from core.algorithms.AM import am_process_block 
from core.algorithms.FM import fm_process_block
from core.algorithms.ASK import ask_prepare_state, ask_modulate_block, ask_demodulate_block
              
from core.algorithms.FSK import bfsk_prepare_state, bfsk_modulate_block, bfsk_demodulate_block

        
class Dashboard(ctk.CTk):
    
    def __init__(self, statusData):
        super().__init__()
        
        # Application state variable
        
        self.statusData = statusData
        
        self.SamplingStream = SamplingStream(statusData)
        
        self.audioController = AudioController(statusData)
        
        self.statusData.ncc_pairer = NCCPairer(maxlen=12)
        
        
        # Initialize main panels
        
        self.config_window()
        self.panels()
        self.navbar_panel()
        self.display_signals()
        self.left_sidebar_panel()
        self.right_overview_panel()
        self.right_results_panel()
        self._ui_timer_running = False


    # Configures the main application window.
    def config_window(self):
        self.title("ModuTEC Simulator")
        self.geometry("1600x900")
        wc.center_window(self, 1600, 900)
        self.iconbitmap("assets/logo.ico")


    # Creates the main application panels.
    def panels(self):
        
        btnsFont = ctk.CTkFont(family='FontAwesome', size=16, weight="bold")
        
        # Navbar Frame
        self.navbar = ctk.CTkFrame(self, fg_color=NAVBAR_COLOR, height=50)
        self.navbar.pack(side="top", fill="both")

        # Left Sidebar Frame
        self.sidebar = ctk.CTkFrame(self, fg_color=NAVBAR_COLOR, width=200)
        self.sidebar.pack(side="left", fill="both", expand=False)

        # Sidebar Sub-frame for static options
        self.optionsHeader = ctk.CTkFrame(self.sidebar, fg_color=SIDEBAR_COLOR, width=200)
        self.optionsHeader.pack(side="top", padx=10, pady=(2, 10), fill="x")

        # Sidebar Sub-frame for dynamic options
        self.optionsMenu = ctk.CTkFrame(self.sidebar, fg_color=SIDEBAR_COLOR, width=200)
        self.optionsMenu.pack(padx=10, pady=(2, 10), fill="x")

        # Sidebar button to apply changes
        self.applyButton = ctk.CTkButton(self.sidebar, text="Apply Changes", font=btnsFont, fg_color=OK_BTN_COLOR, command=self.applyChanges)
        self.applyButton.pack(side="bottom", padx=10, pady=(5, 20), fill="x")

        # MainFrame (graphics and right sidebar) 
        self.mainArea = ctk.CTkFrame(self, fg_color=MAIN_BACKGROUND_COLOR)
        self.mainArea.pack(side="right", fill="both", expand=True)

        # Right Sidebar Frame
        self.rightsidebar = ctk.CTkFrame(self.mainArea, fg_color=MAIN_BACKGROUND_COLOR, width=200)  # ← sin height
        self.rightsidebar.pack(side="right", fill="y", padx=(10, 10), pady=(10, 10))  # ← fill="y"

        # Right Sidebar Sub-frame for current configuration
        self.currentConfig = ctk.CTkFrame(self.rightsidebar, fg_color=SIDEBAR_COLOR, width=250, height=250)
        self.currentConfig.pack(side="top", pady=(2, 10), fill="x")

        # Right Sidebar Sub-frame for results logs
        self.resultsArea = ctk.CTkFrame(self.rightsidebar, fg_color=SIDEBAR_COLOR, width=250)
        self.resultsArea.pack(fill="both", expand=True, pady=(2, 10))
    
    
    # Creates and configures the navigation bar buttons and layout. 
    def navbar_panel(self):
        
        # Fonts styles
        titleFont = ctk.CTkFont(family='FontAwesome', size=30, weight="bold")
        btnsFont = ctk.CTkFont(family='FontAwesome', size=18, weight="bold")
        
        # Application Logo
        logoIcon = Image.open("assets/logo.ico")
        self.iconImage = ctk.CTkImage(dark_image=logoIcon, size=(50, 50))
        self.iconLabel = ctk.CTkButton(self.navbar, image=self.iconImage, text= "", fg_color=NAVBAR_COLOR, command=self.toggle_animation, hover= False, width=0, height=0)
        self.iconLabel.pack(side="left", padx=0, pady=0)

        # Application Title
        self.titleLabel = ctk.CTkLabel(self.navbar, text="ModuTEC", text_color="white", font=titleFont)
        self.titleLabel.pack(side="left", padx=0, pady=0)
        
        # Application Dashboard Title 
        titleIcon = Image.open("assets/title.png")
        self.iconImage = ctk.CTkImage(dark_image=titleIcon, size=(400, 50))
        self.dashLabel = ctk.CTkLabel(self.navbar, image=self.iconImage, text= "", fg_color=NAVBAR_COLOR, width=0, height=0)
        self.dashLabel.pack(side="left", fill="x", expand=True)

        # Start/Stop Simulation Button
        self.startSimulation = ctk.CTkButton(self.navbar, text="Run", fg_color=RUNNING_COLOR, text_color="white",
                                             font=btnsFont, width=60, height=30, command=self.simulation_button_animation, hover=False)
        self.startSimulation.pack(side="right", padx=5, pady=5)


    # Right Sidebar Overview Panel
    def right_overview_panel(self):
        # Reset panel (idle)
        for widget in self.currentConfig.winfo_children():
            widget.destroy()

        # Fonts
        labelFont = ctk.CTkFont(family='FontAwesome', size=12)
        labelTitleFont = ctk.CTkFont(family='FontAwesome', size=12, weight="bold")
        titleFont = ctk.CTkFont(family='FontAwesome', size=18, weight="bold")

        # Título
        ctk.CTkLabel(
            self.currentConfig,
            text="Simulation Overview",
            font=titleFont,
            text_color="white",
            anchor="center",
            justify="center"
        ).pack(pady=(5, 10), padx=10, anchor="center")

        # Contenedor del grid
        overview_grid = ctk.CTkFrame(self.currentConfig, fg_color=SIDEBAR_COLOR)
        overview_grid.pack(padx=10, pady=(0, 10), fill="x")

        # Grid: 2 columnas x 5 filas
        for i in range(5):
            overview_grid.grid_rowconfigure(i, weight=1)
        for j in range(2):
            overview_grid.grid_columnconfigure(j, weight=1)

        # Helpers
        def _fmt(v):
            try:
                if isinstance(v, (int, np.integer)) or (isinstance(v, float) and float(v).is_integer()):
                    return f"{int(v)}"
                return f"{float(v):.4f}"
            except Exception:
                return str(v)

        def _short(text, max_chars=28):
            try:
                t = os.path.basename(text) if text else "—"
                if len(t) <= max_chars:
                    return t
                head = max_chars - 6
                return f"{t[:head]}…{t[-5:]}"
            except Exception:
                return "—"

        # Base info
        filename = _short(getattr(self.statusData, "audio_file_path", None))
        mod_type = self.statusData.modulation_type.get() if hasattr(self.statusData, "modulation_type") else "—"
        if not mod_type or mod_type == "Select type":
            mod_type = "—"

        sr = getattr(self.statusData, "sample_rate", None)
        if sr is None:
            sr = getattr(self.statusData, "Fs", None)
        sr_txt = f"Sample Rate: {int(sr)}" if sr is not None else "Sample Rate: —"

        # Fila 0: encabezados
        ctk.CTkLabel(overview_grid, text="Audio File", font=labelTitleFont, text_color="white",
                    anchor="w", justify="left").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        ctk.CTkLabel(overview_grid, text="Modulation Type", font=labelTitleFont, text_color="white",
                    anchor="w", justify="left").grid(row=0, column=1, sticky="w", padx=5, pady=2)

        # Fila 1: valores
        ctk.CTkLabel(overview_grid, text=filename, font=labelFont, text_color="white",
                    anchor="w", justify="left").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        ctk.CTkLabel(overview_grid, text=mod_type, font=labelFont, text_color="white",
                    anchor="w", justify="left").grid(row=1, column=1, sticky="w", padx=5, pady=2)

        # Fila 2: sample rate
        ctk.CTkLabel(overview_grid, text=sr_txt, font=labelFont, text_color="white",
                    anchor="w", justify="left").grid(row=2, column=0, sticky="w", padx=5, pady=2)

        # Filas 3–4: variables por técnica
        row3_col0 = row3_col1 = row4_col0 = row4_col1 = "—"

        if mod_type == "AM":
            row3_col0 = f"Index μ: {_fmt(getattr(self.statusData, 'am_mu', '—'))}"
            row3_col1 = f"Carrier Frequency: {_fmt(getattr(self.statusData, 'am_fc', '—'))}"
            row4_col0 = f"Carrier Amplitude: {_fmt(getattr(self.statusData, 'am_Ac', '—'))}"
            row4_col1 = ""

        elif mod_type == "FM":
            row3_col0 = f"β: {_fmt(getattr(self.statusData, 'fm_beta', '—'))}"
            row3_col1 = f"Carrier Frequency: {_fmt(getattr(self.statusData, 'fm_fc', '—'))}"
            row4_col0 = f"Carrier Amplitude: {_fmt(getattr(self.statusData, 'fm_Ac', '—'))}"
            row4_col1 = ""

        elif mod_type == "ASK":
            br = getattr(self.statusData, 'ask_bitrate', None)
            row3_col0 = f"Bitrate: {int(br)} bps" if br is not None else "Bitrate: —"
            row3_col1 = f"Carrier Frequency: {_fmt(getattr(self.statusData, 'ask_fc', '—'))}"
            row4_col0 = f"Carrier Amplitude: {_fmt(getattr(self.statusData, 'ask_Ac', '—'))}"
            row4_col1 = ""

        elif mod_type == "FSK":
            br = getattr(self.statusData, 'fsk_bitrate', None)
            row3_col0 = f"Bitrate: {int(br)} bps" if br is not None else "Bitrate: —"
            row3_col1 = f"Carrier Freq 1: {_fmt(getattr(self.statusData, 'fsk_fc1', '—'))}"
            row4_col0 = f"Carrier Amplitude: {_fmt(getattr(self.statusData, 'fsk_Ac', '—'))}"
            row4_col1 = f"Carrier Freq 2: {_fmt(getattr(self.statusData, 'fsk_fc2', '—'))}"

        # Pintar
        ctk.CTkLabel(overview_grid, text=row3_col0, font=labelFont, text_color="white",
                    anchor="w", justify="left").grid(row=3, column=0, sticky="w", padx=5, pady=2)
        ctk.CTkLabel(overview_grid, text=row3_col1, font=labelFont, text_color="white",
                    anchor="w", justify="left").grid(row=3, column=1, sticky="w", padx=5, pady=2)
        ctk.CTkLabel(overview_grid, text=row4_col0, font=labelFont, text_color="white",
                    anchor="w", justify="left").grid(row=4, column=0, sticky="w", padx=5, pady=2)
        ctk.CTkLabel(overview_grid, text=row4_col1, font=labelFont, text_color="white",
                    anchor="w", justify="left").grid(row=4, column=1, sticky="w", padx=5, pady=2)

    
    # Right Sidebar Results Panel
    def right_results_panel(self):
        
        titleFont = ctk.CTkFont(family='FontAwesome', size=18, weight="bold")
        ctk.CTkLabel(self.resultsArea, text="Results Log", font=titleFont,
                    text_color="white").pack(pady=(5, 8), padx=10)
        
        self.results_text = ctk.CTkTextbox(self.resultsArea, wrap="word", bg_color=SIDEBAR_COLOR, fg_color=SIDEBAR_COLOR,
                                           font=ctk.CTkFont(family='FontAwesome', size=12), border_width=0)
        self.results_text.pack(fill="both", expand=True, padx=10, pady=(0,10))
        self.results_text.configure(state="disabled")   # inicia solo lectura

    def log_result(self, msg: str, color: str = "white"):
        self.results_text.configure(state="normal")
        self.results_text.insert("end", msg + "\n")
        self.results_text.tag_add(color, "end-2l", "end-1l")
        self.results_text.tag_config(color, foreground=color)
        self.results_text.see("end")
        self.results_text.configure(state="disabled")
        
    # Creates and configures the left sidebar with options and controls.
    def left_sidebar_panel(self):

        # Font styles
        labelFont = ctk.CTkFont(family='FontAwesome', size=12)
        titleFont = ctk.CTkFont(family='FontAwesome', size=18, weight="bold")

        # Sidebar Options Header Title
        ctk.CTkLabel(self.optionsHeader, text="⚙️  Configuration Panel", font=titleFont, text_color="white",
                     anchor="w").pack(pady=(15, 10), padx=10, anchor="w")

        # Upload Audio File Button and placeholder for file name
        uploadIcon = Image.open("assets/upload.png")
        self.uploadIcon = ctk.CTkImage(dark_image=uploadIcon, size=(30, 30))
        ctk.CTkLabel(self.optionsHeader, text="Audio File", font=labelFont, text_color="white").pack(padx=10, anchor="w")
        self.upload_button = ctk.CTkButton(self.optionsHeader, text="Upload audio file", command=self.load_audio, hover=False, compound="left", image=self.uploadIcon, fg_color=SIDEBAR_COLOR)
        self.upload_button.pack(padx=10, fill="x")

        self.songName = ctk.CTkLabel(self.optionsHeader, text="Load a song...", font=labelFont, text_color="white", wraplength=160,)
        self.songName.pack(padx=10, pady=(0, 10),fill="x")

        # Modulation Type Dropdown
        ctk.CTkLabel(self.optionsHeader, text="Select Modulation Type", font=labelFont, text_color="white").pack(padx=10, anchor="w")
        self.statusData.modulation_type = ctk.StringVar(value="Select type")
        self.mod_dropdown = ctk.CTkOptionMenu(self.optionsHeader, variable=self.statusData.modulation_type, values=["AM", "FM", "ASK", "FSK"], command=self.custom_options)
        self.mod_dropdown.pack(padx=10, pady=(0, 15))
        
        
    # Function to load an audio file and update the UI.   
    def load_audio(self):
        
        # Open file dialog to select audio file
        file_path = filedialog.askopenfilename(
            filetypes=[("Audio Files", "*.wav *.mp3"), ("WAV Files", "*.wav"), ("MP3 Files", "*.mp3")],
            title="Select an audio file"
        )

        # MP3 to WAV conversion
        if file_path and file_path.lower().endswith('.mp3'):
            self.statusData.audio_file_path = self.audioController.mp3ToWav(file_path)
        else:
            self.statusData.audio_file_path = file_path

        # Calculate the recommended params 
        
        self.audioController.recommend_params()
        
        
        # Update song name label
        if hasattr(self, "songName") and self.songName:
            self.songName.configure(
                text=os.path.basename(self.statusData.audio_file_path) if self.statusData.audio_file_path else "Load a song..."
            )
        
    
    # Function to handle the Start/Stop simulation button logic and UI updates.
    def simulation_button_animation(self):
        if not self.statusData.is_running:
            # PLAY / RESUME
            if not self.statusData.audio_file_path:
                return
            self.startSimulation.configure(text="Stop", fg_color=STOP_COLOR)

            # Reiniciar estado AM para nueva simulación
            self.statusData.am_initialized = False
            self.statusData.am_phase = 0.0
            
            # Reiniciar estado FM para nueva simulación
            self.statusData.fm_initialized = False
            self.statusData.fm_phase = 0.0
            self.statusData.fm_xscale = None
            self.statusData.fm_phase_unwrap_prev = 0.0
            self.statusData.fm_lp_ym1 = 0.0
            
            # Reiniciar estado ASK
            self.statusData.ask_initialized = False
            self.statusData.ask_state = None
            
            # Reiniciar estado FSK
            self.statusData.fsk_initialized = False
            self.statusData.fsk_state = None

            
            # En tu handler de "Start/Run" (donde reseteás estados AM)
            if hasattr(self.statusData, "ncc_pairer"):
                self.statusData.ncc_pairer = NCCPairer(maxlen=12)
            self.statusData.chunk_seq = 0

            self.statusData.is_running = True

            # Iniciar el stream y el ploteo en tiempo real
            self.SamplingStream.start_stream()
            
            # Arrancar consumidor (UI) si no está ya
            if not self._ui_timer_running:
                self._ui_timer_running = True
                self.after(16, self._ui_timer)
        else:
            # PAUSE
            self.startSimulation.configure(text="Run", fg_color=RUNNING_COLOR)
            self.statusData.is_running = False

            # Pausar el stream y el ploteo en tiempo real
            self.SamplingStream.pause_stream()


    # Left sidebar show/hide animation.
    def toggle_animation(self):
        if self.sidebar.winfo_ismapped():
            self.sidebar.pack_forget()
        else:
            self.sidebar.pack(side="left", fill="both")

    # Creates a embedded matplotlib plot.
    def create_embedded_plot(self, title):
        
        # Create a dark-themed matplotlib figure
        fig, ax = plt.subplots(figsize=(4, 3))
        fig.patch.set_facecolor(MAIN_BACKGROUND_COLOR)
        ax.set_facecolor(MAIN_BACKGROUND_COLOR)
        ax.set_title(title, color=WHITE_COLOR, pad=10)
        ax.tick_params(colors=WHITE_COLOR)
        ax.spines['bottom'].set_color(WHITE_COLOR)
        ax.spines['left'].set_color(WHITE_COLOR)
        ax.margins(x=0.0)                                   
        fig.subplots_adjust(left=0.08, right=0.995, top=0.90, bottom=0.16)
        return fig

    
    # Custom options in the left sidebar based on modulation type selection
    
    def custom_options(self,value):

        for widget in self.optionsMenu.winfo_children():
            widget.destroy()
            
        if value == "AM":
            ctk.CTkLabel(self.optionsMenu, text="Sample Rate AM", text_color="white").pack(padx=10, anchor="w")
            self.sample_rate_input = ctk.CTkEntry(self.optionsMenu, placeholder_text=self.statusData.recommended_Fs)
            self.sample_rate_input.pack(padx=10, pady=(0, 15), fill="x")

            ctk.CTkLabel(self.optionsMenu, text="Carrier Frequency AM", text_color="white").pack(padx=10, anchor="w")
            self.carrier_freq_input = ctk.CTkEntry(self.optionsMenu, placeholder_text=self.statusData.recommended_am_fc)
            self.carrier_freq_input.pack(padx=10, pady=(0, 15), fill="x")

            ctk.CTkLabel(self.optionsMenu, text="Carrier Amplitude AM", text_color="white").pack(padx=10, anchor="w")
            self.carrier_amp_input = ctk.CTkEntry(self.optionsMenu, placeholder_text=self.statusData.recommended_am_Ac)
            self.carrier_amp_input.pack(padx=10, pady=(0, 15), fill="x")
            
            ctk.CTkLabel(self.optionsMenu, text="Modulation Index AM", text_color="white").pack(padx=10, anchor="w")
            self.modulation_index_input = ctk.CTkEntry(self.optionsMenu, placeholder_text=self.statusData.recommended_am_mu)
            self.modulation_index_input.pack(padx=10, pady=(0, 15), fill="x")
            
            
        elif value == "FM":
            ctk.CTkLabel(self.optionsMenu, text="Sample Rate FM", text_color="white").pack(padx=10, anchor="w")
            self.sample_rate_input = ctk.CTkEntry(self.optionsMenu, placeholder_text=self.statusData.recommended_Fs)
            self.sample_rate_input.pack(padx=10, pady=(0, 15), fill="x")

            ctk.CTkLabel(self.optionsMenu, text="Carrier Frequency FM", text_color="white").pack(padx=10, anchor="w")
            self.carrier_freq_input = ctk.CTkEntry(self.optionsMenu, placeholder_text=self.statusData.recommended_fm_fc)
            self.carrier_freq_input.pack(padx=10, pady=(0, 15), fill="x")

            ctk.CTkLabel(self.optionsMenu, text="Carrier Amplitude FM", text_color="white").pack(padx=10, anchor="w")
            self.carrier_amp_input = ctk.CTkEntry(self.optionsMenu, placeholder_text=self.statusData.recommended_fm_Ac)
            self.carrier_amp_input.pack(padx=10, pady=(0, 15), fill="x")
            
            ctk.CTkLabel(self.optionsMenu, text="Modulation Index FM", text_color="white").pack(padx=10, anchor="w")
            self.modulation_index_input = ctk.CTkEntry(self.optionsMenu, placeholder_text=self.statusData.recommended_fm_beta)
            self.modulation_index_input.pack(padx=10, pady=(0, 15), fill="x")
            
        elif value == "ASK":
            ctk.CTkLabel(self.optionsMenu, text="Sample Rate ASK (OOK)", text_color="white").pack(padx=10, anchor="w")
            self.sample_rate_input = ctk.CTkEntry(self.optionsMenu, placeholder_text=self.statusData.recommended_Fs)
            self.sample_rate_input.pack(padx=10, pady=(0, 15), fill="x")
            
            ctk.CTkLabel(self.optionsMenu, text="Carrier Frequency ASK (OOK)", text_color="white").pack(padx=10, anchor="w")
            self.carrier_freq_input = ctk.CTkEntry(self.optionsMenu, placeholder_text=self.statusData.recommended_ask_fc)
            self.carrier_freq_input.pack(padx=10, pady=(0, 15), fill="x")

            ctk.CTkLabel(self.optionsMenu, text="Carrier Amplitude ASK (OOK)", text_color="white").pack(padx=10, anchor="w")
            self.carrier_amp_input = ctk.CTkEntry(self.optionsMenu, placeholder_text=self.statusData.recommended_ask_Ac)
            self.carrier_amp_input.pack(padx=10, pady=(0, 15), fill="x")
            
            ctk.CTkLabel(self.optionsMenu, text="Bitrate ASK (OOK)", text_color="white").pack(padx=10, anchor="w")
            self.bitrate_input = ctk.CTkEntry(self.optionsMenu, placeholder_text=self.statusData.recommended_ask_bitrate)
            self.bitrate_input.pack(padx=10, pady=(0, 15), fill="x")
            
        elif value == "FSK":
            
            ctk.CTkLabel(self.optionsMenu, text="Sample Rate FSK (BFSK)", text_color="white").pack(padx=10, anchor="w")
            self.sample_rate_input = ctk.CTkEntry(self.optionsMenu, placeholder_text=self.statusData.recommended_Fs)
            self.sample_rate_input.pack(padx=10, pady=(0, 15), fill="x")
        
            ctk.CTkLabel(self.optionsMenu, text="Carrier Frequency BFSK 1 (High)", text_color="white").pack(padx=10, anchor="w")
            self.carrier_freq_input = ctk.CTkEntry(self.optionsMenu, placeholder_text=self.statusData.recommended_fsk_fc1)
            self.carrier_freq_input.pack(padx=10, pady=(0, 15), fill="x")
            
            ctk.CTkLabel(self.optionsMenu, text="Carrier Frequency BFSK 2 (Low)", text_color="white").pack(padx=10, anchor="w")
            self.carrier_freq2_input = ctk.CTkEntry(self.optionsMenu, placeholder_text=self.statusData.recommended_fsk_fc2)
            self.carrier_freq2_input.pack(padx=10, pady=(0, 15), fill="x")

            ctk.CTkLabel(self.optionsMenu, text="Carrier Amplitude FSK (BFSK)", text_color="white").pack(padx=10, anchor="w")
            self.carrier_amp_input = ctk.CTkEntry(self.optionsMenu, placeholder_text=self.statusData.recommended_fsk_Ac)
            self.carrier_amp_input.pack(padx=10, pady=(0, 15), fill="x")

            ctk.CTkLabel(self.optionsMenu, text="Bitrate FSK (BFSK)", text_color="white").pack(padx=10, anchor="w")
            self.bitrate_input = ctk.CTkEntry(self.optionsMenu, placeholder_text=self.statusData.recommended_fsk_bitrate)
            self.bitrate_input.pack(padx=10, pady=(0, 15), fill="x")
        
    def checkOptions(self):
        
        # Setting the default recommended values
        # Sample rate
        self.statusData.sample_rate = (
            self.statusData.recommended_Fs if (self.sample_rate_input.get() == "")
            else int(self.sample_rate_input.get())
        )
        
        mod = self.statusData.modulation_type.get()

        if mod == "AM":
            self.statusData.am_fc = (
                self.statusData.recommended_am_fc if (self.carrier_freq_input.get() == "")
                else float(self.carrier_freq_input.get())
            )
            self.statusData.am_Ac = (
                self.statusData.recommended_am_Ac if (self.carrier_amp_input.get() == "")
                else float(self.carrier_amp_input.get())
            )
            self.statusData.am_mu = (
                self.statusData.recommended_am_mu if (self.modulation_index_input.get() == "")
                else float(self.modulation_index_input.get())
            )

        elif mod == "FM":
            self.statusData.fm_fc = (
                self.statusData.recommended_fm_fc if (self.carrier_freq_input.get() == "")
                else float(self.carrier_freq_input.get())
            )
            self.statusData.fm_Ac = (
                self.statusData.recommended_fm_Ac if (self.carrier_amp_input.get() == "")
                else float(self.carrier_amp_input.get())
            )
            self.statusData.fm_beta = (
                self.statusData.recommended_fm_beta if (self.modulation_index_input.get() == "")
                else float(self.modulation_index_input.get())
            )
        elif mod == "ASK":
            self.statusData.ask_fc = (
                self.statusData.recommended_ask_fc if (self.carrier_freq_input.get() == "")
                else float(self.carrier_freq_input.get())
            )
            self.statusData.ask_Ac = (
                self.statusData.recommended_ask_Ac if (self.carrier_amp_input.get() == "")
                else float(self.carrier_amp_input.get())
            )
            self.statusData.ask_bitrate = (
                self.statusData.recommended_ask_bitrate if (self.bitrate_input.get() == "")
                else float(self.bitrate_input.get())
            )

        elif mod == "FSK":
            self.statusData.fsk_fc1 = (
                self.statusData.recommended_fsk_fc1 if (self.carrier_freq_input.get() == "")
                else float(self.carrier_freq_input.get())
            )
            self.statusData.fsk_fc2 = (
                self.statusData.recommended_fsk_fc2 if (self.carrier_freq2_input.get() == "")
                else float(self.carrier_freq2_input.get())
            )
            self.statusData.fsk_Ac = (
                self.statusData.recommended_fsk_Ac if (self.carrier_amp_input.get() == "")
                else float(self.carrier_amp_input.get())
            )
            self.statusData.fsk_bitrate = (
                self.statusData.recommended_fsk_bitrate if (self.bitrate_input.get() == "")
                else float(self.bitrate_input.get())
            )

    # Function to apply changes to the current simulation settings.
    def applyChanges(self):

        if self.statusData.audio_file_path and (self.statusData.modulation_type.get() != "Select type"):
            print("Modulation type:", self.statusData.modulation_type.get())
            sample_rate, data = wavfile.read(self.statusData.audio_file_path)

            # Si es estéreo, toma canal 0 (solo para leer SR y mostrar en UI)
            if len(data.shape) > 1:
                data = data[:, 0]

           # --- [PATCH applyChanges - actualizar SR en UI de forma segura] ---
            sr_entry = getattr(self, "sample_rate_input", None)
            if sr_entry is not None:
                try:
                    sr_entry.delete(0, "end")
                    sr_entry.insert(0, str(self.statusData.sample_rate))
                    print(f"Sample rate updated to {self.statusData.sample_rate} Hz in the UI.")
                except Exception:
                    print("Failed to update sample rate in the UI.")
                    pass
            # 0) Cortar cualquier lector activo y dejar flags/contadores en frío
            try:
                if hasattr(self, "SamplingStream") and self.SamplingStream:
                    self.SamplingStream.stop_stream()   # detiene hilo y lo pone en None internamente
            except Exception:
                pass

            # estado base
            self.statusData.is_running = False
            self.statusData.paused = True
            self.statusData.pos_frames = 0
                
            self.checkOptions()

            self._reset_ring_ui()
        
            # Actualizar parámetros según entradas del usuario
            self.right_overview_panel()
            
            
            print("Changes applied successfully.")
        else :
            CTkMessagebox(title="Error", message="No audio file loaded or modulation type not selected.", icon="warning")
    
    def _reset_ring_ui(self):

        N = int(self.statusData.window_seconds * float(self.statusData.sample_rate))
        N = max(1024, N)
        
        # A) Purga no bloqueante de todos los items pendientes
        try:
            while True:
                self.statusData.q.get_nowait()
        except queue.Empty:
            pass
        except Exception:
            pass
    
        # Ring original
        self.statusData.ring = np.zeros(N, dtype=np.float32)
        x1 = np.arange(N)
        self.line1.set_data(x1, self.statusData.ring)
        self.ax1.set_xlim(0, N)
        self.ax1.relim()
        self.ax1.autoscale_view(scalex=False, scaley=True)
        self.canvas1.draw_idle()

        # Ring modulado
        self.statusData.mod_ring = np.zeros(N, dtype=np.float32)
        x2 = np.arange(N)
        self.line2.set_data(x2, self.statusData.mod_ring)
        self.ax2.set_xlim(0, N)
        self.ax2.relim()
        self.ax2.autoscale_view(scalex=False, scaley=True)
        self.canvas2.draw_idle()

        # Ring demodulado
        self.statusData.demod_ring = np.zeros(N, dtype=np.float32)
        x3 = np.arange(N)
        self.line3.set_data(x3, self.statusData.demod_ring)
        self.ax3.set_xlim(0, N)
        self.ax3.relim()
        self.ax3.autoscale_view(scalex=False, scaley=True)
        self.canvas3.draw_idle()

    def _ui_timer(self):
        drained = False

        # Drenar cola de entrada (puede haber varios chunks por tick)
        while True:
            try:
                chunk = self.statusData.q.get_nowait()
                self.statusData.chunk_seq += 1
                cid = self.statusData.chunk_seq
                             
            except queue.Empty:
                break

            drained = True

            # --- 2) MODULACIÓN + DEMODULACIÓN AM (coherente, escala global) ---
            s_mod = None
            s_demod = None

            if self.statusData.modulation_enabled and self.statusData.modulation_type.get() == "AM":

                # Asegura init una sola vez
                if not self.statusData.am_initialized:
                    self.statusData.am_phase = 0.0
                    self.statusData.am_lp_ym1 = 0.0
                    self.statusData.am_initialized = True

                Fs = float(self.statusData.sample_rate)

                # State mínimo requerido (usa tus valores actuales de UI/estado)
                am_state = {
                    "fc":    float(self.statusData.am_fc),
                    "mu":    float(self.statusData.am_mu),
                    "Ac":    float(self.statusData.am_Ac),
                    "phase": float(self.statusData.am_phase),
                    "lp_ym1": float(getattr(self.statusData, "am_lp_ym1", 0.0)),
                }

                # Ejecuta ciclo AM sobre el chunk actual (chunk debe ser np.float32 o convertible)
                s_mod, s_demod, am_state, blk_stats = am_process_block(
                    x=chunk.astype(np.float32, copy=False),
                    Fs=Fs,
                    state=am_state
                )

                # Actualiza estado persistente
                self.statusData.am_phase  = am_state["phase"]
                self.statusData.am_lp_ym1 = am_state.get("lp_ym1", 0.0)

                # (Opcional) expón estadísticas por-bloque a tu UI/estado para debug
                self.statusData.blk_mean = blk_stats["blk_mean"]
                self.statusData.blk_rms  = blk_stats["blk_rms"]
                self.statusData.blk_peak = blk_stats["blk_peak"]
                self.statusData.blk_fmax = blk_stats["blk_fmax"]

                # Actualiza RING ORIGINAL (PCM)
                ring = self.statusData.ring
                if len(chunk) >= len(ring):
                    ring[:] = chunk[-len(ring):]
                else:
                    L = len(chunk)
                    ring[:-L] = ring[L:]
                    ring[-L:] = chunk

                if self.statusData.ncc_pairer is not None:
                    self.statusData.ncc_pairer.push_original(cid, np.asarray(chunk, dtype=np.float32))

                # Actualiza RING MODULADO
                mring = self.statusData.mod_ring
                if len(s_mod) >= len(mring):
                    mring[:] = s_mod[-len(mring):]
                else:
                    Lm = len(s_mod)
                    mring[:-Lm] = mring[Lm:]
                    mring[-Lm:] = s_mod

                # Actualiza RING DEMODULADO
                dring = self.statusData.demod_ring
                if len(s_demod) >= len(dring):
                    dring[:] = s_demod[-len(dring):]
                else:
                    Ld = len(s_demod)
                    dring[:-Ld] = dring[Ld:]
                    dring[-Ld:] = s_demod

                # NCC (si lo usas)
                if self.statusData.ncc_pairer is not None:
                    res = self.statusData.ncc_pairer.push_demodulated(cid, np.asarray(s_demod, dtype=np.float32))
                    if res is not None and hasattr(self, "log_result"):
                        pct = res["percent"]
                        thr = getattr(self.statusData, "ncc_threshold", 70.0)
                        color = "#00FF00" if pct >= thr else "#FF3B30"
                        self.log_result(f"[AM] Chunk #{res['chunk_id']} processed NCC: {pct:.1f}%", color=color)
                # --- FIN AM ---

            
           # --- 2bis) MODULACIÓN FM (si está activa) ---
            if self.statusData.modulation_enabled and self.statusData.modulation_type.get() == "FM":

                Fs   = float(self.statusData.sample_rate)
                fc   = float(self.statusData.fm_fc)
                Ac   = float(self.statusData.fm_Ac)
                beta = float(self.statusData.fm_beta)

                # Estado que pasa/recibe FM.py (solo intermediario)
                st_in = {
                    "phase":       float(self.statusData.fm_phase),
                    "prev_df":     float(self.statusData.fm_prev_df),
                    "prev_z":      self.statusData.fm_prev_z,
                    "lp_ym1":      float(self.statusData.fm_lp_ym1),
                    "hpf_xm1":     float(self.statusData.fm_hpf_xm1),
                    "hpf_ym1":     float(self.statusData.fm_hpf_ym1),
                    "prev_tail":   self.statusData.fm_prev_tail,
                    "prev_raw":    self.statusData.fm_prev_raw,
                    "hilbert_pad": int(self.statusData.fm_hilbert_pad),
                    "xfade":       int(self.statusData.fm_xfade),
                    "hpf_fc":      float(self.statusData.fm_hpf_fc),
                    "lpf_cut":     self.statusData.fm_lpf_cut,      # puede ser None
                    "demod_gain":  float(self.statusData.fm_demod_gain),
                }

                s_mod, s_demod, st_out, stats = fm_process_block(
                    x=chunk.astype(np.float32, copy=False),
                    Fs=Fs,
                    state=st_in,
                    fc=fc,
                    Ac=Ac,
                    beta=beta
                )

                # Persistir estado devuelto
                self.statusData.fm_phase     = float(st_out["phase"])
                self.statusData.fm_prev_df   = float(st_out.get("prev_df", self.statusData.fm_prev_df))
                self.statusData.fm_prev_z    = st_out["prev_z"]
                self.statusData.fm_lp_ym1    = float(st_out["lp_ym1"])
                self.statusData.fm_hpf_xm1   = float(st_out["hpf_xm1"])
                self.statusData.fm_hpf_ym1   = float(st_out["hpf_ym1"])
                self.statusData.fm_prev_tail = st_out["prev_tail"]
                self.statusData.fm_prev_raw  = st_out["prev_raw"]

                # Stats a panel
                self.statusData.fm_fmax_blk  = float(stats["fmax_blk"])
                self.statusData.fm_df_blk    = float(stats["df_blk"])
                self.statusData.fm_kappa_blk = float(stats["kappa_blk"])

                # Rings
                ring = self.statusData.ring
                if len(chunk) >= len(ring):
                    ring[:] = chunk[-len(ring):]
                else:
                    L = len(chunk)
                    ring[:-L] = ring[L:]
                    ring[-L:] = chunk

                mring = self.statusData.mod_ring
                if len(s_mod) >= len(mring):
                    mring[:] = s_mod[-len(mring):]
                else:
                    Lm = len(s_mod)
                    mring[:-Lm] = mring[Lm:]
                    mring[-Lm:] = s_mod

                dring = self.statusData.demod_ring
                if len(s_demod) >= len(dring):
                    dring[:] = s_demod[-len(dring):]
                else:
                    Ld = len(s_demod)
                    dring[:-Ld] = dring[Ld:]
                    dring[-Ld:] = s_demod

                # NCC (si aplica)
                if self.statusData.ncc_pairer is not None:
                    self.statusData.ncc_pairer.push_original(cid, np.asarray(chunk, dtype=np.float32))
                    res = self.statusData.ncc_pairer.push_demodulated(cid, np.asarray(s_demod, dtype=np.float32))
                    if res is not None:
                        pct = res["percent"]
                        thr = getattr(self.statusData, "ncc_threshold", 70.0)
                        color = "#00FF00" if pct >= thr else "#FF3B30"
                        self.log_result(f"[FM] Chunk #{res['chunk_id']} processed NCC: {pct:.1f}%", color=color)
                                
            # --- 2ter) MODULACIÓN ASK (bloque por bloque, versión actualizada) ---
            if self.statusData.modulation_enabled and self.statusData.modulation_type.get() == "ASK":

                # --- buffers ---
                if (getattr(self.statusData, "mod_ring", None) is None) or (len(self.statusData.mod_ring) == 0):
                    self._reset_ring_ui()
                if (getattr(self.statusData, "demod_ring", None) is None) or (len(self.statusData.demod_ring) == 0):
                    self._reset_ring_ui()

                # --- preparar estado una sola vez ---
                if not self.statusData.ask_initialized:
                    try:
                        st = ask_prepare_state(
                            first_chunk=chunk.astype(np.float32, copy=False),
                            Fs=float(self.statusData.sample_rate),
                            fc=float(self.statusData.ask_fc),
                            Ac=float(self.statusData.ask_Ac),
                            bitrate=float(self.statusData.ask_bitrate),
                        )
                        self.statusData.ask_state = st
                        self.statusData.ask_initialized = True
                    except Exception as e:
                        self.log_result(f"[ASK] Prepare State Error: {e}", color="#FF3B30")
                        return

                # --- modular bloque ---
                try:
                    s_mod, bits_nrz, self.statusData.ask_state, stats = ask_modulate_block(
                        chunk.astype(np.float32, copy=False),
                        self.statusData.ask_state
                    )
                except Exception as e:
                    self.log_result(f"[ASK] Modulate Error: {e}", color="#FF3B30")
                    return

                # --- demodular bloque ---
                try:
                    y_env, bits_hat = ask_demodulate_block(
                        s_mod.astype(np.float32, copy=False),
                        self.statusData.ask_state
                    )
                except Exception as e:
                    self.log_result(f"[ASK] Demodulate Error: {e}", color="#FF3B30")
                    return

                # --- rings ---
                ring = self.statusData.ring
                orig_display = (bits_nrz.astype(np.float32) * 0.2) - 0.1
                if len(orig_display) >= len(ring):
                    ring[:] = orig_display[-len(ring):]
                else:
                    L = len(orig_display)
                    ring[:-L] = ring[L:]
                    ring[-L:] = orig_display

                mring = self.statusData.mod_ring
                if len(s_mod) >= len(mring):
                    mring[:] = s_mod[-len(mring):]
                else:
                    Lm = len(s_mod)
                    mring[:-Lm] = mring[Lm:]
                    mring[-Lm:] = s_mod

                dring = self.statusData.demod_ring
                if len(y_env) >= len(dring):
                    dring[:] = y_env[-len(dring):]
                else:
                    Ld = len(y_env)
                    dring[:-Ld] = dring[Ld:]
                    dring[-Ld:] = y_env

                # --- métrica digital ---
                try:
                    spb = int(stats.get("spb", max(2, int(round(self.statusData.sample_rate / self.statusData.ask_bitrate)))))
                    def to_symbols(bits):
                        L = len(bits)
                        q = L // spb
                        if q <= 0:
                            return np.array([], dtype=np.uint8)
                        b = bits[:q * spb].reshape(q, spb)
                        return (np.mean(b, axis=1) >= 0.5).astype(np.uint8)
                    sym_src = to_symbols(bits_nrz)
                    sym_hat = to_symbols(bits_hat)
                    acc = float(digital_accuracy(sym_src, sym_hat))
                    acc_pct = acc*100.0 if acc <= 1.5 else acc   
                    self.log_result(f"[ASK] Chunk #{cid} processed NCC: {acc_pct:.1f}%", color="#00FF00" if acc_pct >= 70.0 else "#FF3B30")

                except Exception as e:
                    self.log_result(f"[ASK] NCC Metric Error: {e}", color="#FF3B30")


           # --- 2quad) MODULACIÓN FSK (BFSK por bloque, enfoque adaptativo) ---
            if self.statusData.modulation_enabled and self.statusData.modulation_type.get() == "FSK":
                # Buffers UI
                if (getattr(self.statusData, "mod_ring", None) is None) or (len(self.statusData.mod_ring) == 0):
                    self._reset_ring_ui()
                if (getattr(self.statusData, "demod_ring", None) is None) or (len(self.statusData.demod_ring) == 0):
                    self._reset_ring_ui()

                # Prepare state (una vez)
                if not self.statusData.fsk_initialized:
                    try:
                        self.statusData.fsk_state = bfsk_prepare_state(
                            first_chunk=chunk.astype(np.float32, copy=False),
                            Fs=float(self.statusData.sample_rate),
                            f_high=float(self.statusData.fsk_fc1),  # “1”
                            f_low=float(self.statusData.fsk_fc2),   # “0”
                            Ac=float(self.statusData.fsk_Ac),
                            bitrate=float(self.statusData.fsk_bitrate),
                        )
                        self.statusData.fsk_initialized = True
                    except Exception as e:
                        self.log_result(f"[FSK] prepare error: {e}", color="#FF3B30")
                        return

                # MOD
                try:
                    s_mod, bits_nrz, self.statusData.fsk_state, stats = bfsk_modulate_block(
                        chunk.astype(np.float32, copy=False),
                        self.statusData.fsk_state
                    )
                except Exception as e:
                    self.log_result(f"[FSK] mod error: {e}", color="#FF3B30")
                    return

                # Si no produjo símbolos completos aún, posponer UI/log para este ciclo
                if s_mod.size == 0 or bits_nrz.size == 0:
                    return

                # ORIGINAL (NRZ a [-0.1,+0.1]) → ring “original”
                orig_display = (bits_nrz.astype(np.float32) * 0.2) - 0.1
                ring = getattr(self.statusData, "ring", None)
                if ring is not None and len(ring) > 0:
                    if len(orig_display) >= len(ring):
                        ring[:] = orig_display[-len(ring):]
                    else:
                        L = len(orig_display)
                        if L > 0:
                            ring[:-L] = ring[L:]
                            ring[-L:] = orig_display

                # MOD_RING
                mring = self.statusData.mod_ring
                if len(mring) > 0:
                    if len(s_mod) >= len(mring):
                        mring[:] = s_mod[-len(mring):]
                    else:
                        Lm = len(s_mod)
                        if Lm > 0:
                            mring[:-Lm] = mring[Lm:]
                            mring[-Lm:] = s_mod

                # DEMOD
                try:
                    y_plot, bits_hat, self.statusData.fsk_state = bfsk_demodulate_block(
                        s_mod.astype(np.float32, copy=False),
                        self.statusData.fsk_state
                    )
                except Exception as e:
                    self.log_result(f"[FSK] demod error: {e}", color="#FF3B30")
                    return

                # Puede ocurrir que demod no emita (si quedó justo cortado el símbolo)
                if y_plot.size == 0 or bits_hat.size == 0:
                    return

                # DEMOD_RING (y_plot ya viene en [-0.1,+0.1])
                dring = self.statusData.demod_ring
                if len(dring) > 0:
                    if len(y_plot) >= len(dring):
                        dring[:] = y_plot[-len(dring):]
                    else:
                        Ld = len(y_plot)
                        if Ld > 0:
                            dring[:-Ld] = dring[Ld:]
                            dring[-Ld:] = y_plot

                # LOG estilo NCC (texto), usando chunk_seq como cid
                try:
                    cid = getattr(self.statusData, "chunk_seq", 0)
                    spb = int(stats.get("spb", max(2, int(round(self.statusData.sample_rate / max(1.0, self.statusData.fsk_bitrate))))))

                    def to_symbols(bits: np.ndarray, spb_: int) -> np.ndarray:
                        L = int(bits.size)
                        q = L // spb_
                        if q <= 0:
                            return np.array([], dtype=np.uint8)
                        b = bits[:q * spb_].reshape(q, spb_)
                        return (np.mean(b, axis=1) >= 0.5).astype(np.uint8)

                    sym_src = to_symbols(bits_nrz, spb)
                    sym_hat = to_symbols(bits_hat, spb)

                    if sym_src.size > 0 and sym_hat.size > 0 and (sym_src.size == sym_hat.size):
                        acc = float(digital_accuracy(sym_src, sym_hat))  # tu función
                        acc_pct = acc * 100.0 if acc <= 1.5 else acc
                        self.log_result(f"[FSK] Chunk #{cid} processed NCC: {acc_pct:.1f}%",
                                        color="#00FF00" if acc_pct >= 70.0 else "#FF3B30")
                    else:
                        # Si aún no hay símbolos completos alineados, solo loguea el avance
                        tau_blk = stats.get("tau_blk", None)
                        if tau_blk is not None:
                            self.log_result(f"[FSK] Chunk #{cid} processed | spb={spb} | τ_blk≈{tau_blk:.4f}")
                        else:
                            self.log_result(f"[FSK] Chunk #{cid} processed")

                    # incrementa el contador de chunks
                    setattr(self.statusData, "chunk_seq", cid + 1)

                except Exception as e:
                    self.log_result(f"[FSK] NCC error: {e}", color="#FF3B30")



        # --- 4) Pintar si hubo datos ---
        if drained:
            self.line1.set_ydata(self.statusData.ring)

        if self.statusData.modulation_enabled:
            mt = self.statusData.modulation_type.get()
            if mt in ("AM", "FM", "ASK", "FSK"):
                self.line2.set_ydata(self.statusData.mod_ring)
                self.line3.set_ydata(self.statusData.demod_ring)

        # --- 🔧 Autoescala solo en Y (eje vertical fijo) ---
        for ax in [self.ax1, self.ax2, self.ax3]:
            ax.relim()
            ax.autoscale_view(scalex=False, scaley=True)

        # --- Redibujar ---
        self.canvas1.draw_idle()
        self.canvas2.draw_idle()
        self.canvas3.draw_idle()

        # Reprogramar
        if self.statusData.is_running and not self.statusData.paused:
            self.after(16, self._ui_timer)
        else:
            self._ui_timer_running = False





    def add_toolbar_right(self, parent_grid, canvas, row:int, pady=(0,0)):
        
        holder = ctk.CTkFrame(parent_grid, fg_color=MAIN_BACKGROUND_COLOR, width=44)
        holder.grid(row=row, column=1, sticky="ns", padx=0, pady=67)  
        parent_grid.grid_columnconfigure(1, weight=0, minsize=44)

        holder.grid_rowconfigure(0, weight=1)  # ayuda a estirar en alto
        tb = VerticalRightToolbar(canvas, holder)
        tb.update()
        tb.grid(row=0, column=0, sticky="ns", padx=0, pady=0)
        return tb


    # Displays the original, modulated, and demodulated signal plots in the main area.
    def display_signals(self):
        
        # Create embedded plots
        original = self.create_embedded_plot("Señal Original")
        modulated = self.create_embedded_plot("Señal Modulada")
        demodulated = self.create_embedded_plot("Señal Demodulada")

        # Layout for plots and toolbars
        plots_frame = ctk.CTkFrame(self.mainArea, fg_color=MAIN_BACKGROUND_COLOR)
        plots_frame.pack(fill="both", expand=True)

        # Configure grid layout
        for r in (0, 1, 2):
            plots_frame.grid_rowconfigure(r, weight=1, uniform="plots")
            plots_frame.grid_columnconfigure(0, weight=1)              # canvas 
            plots_frame.grid_columnconfigure(1, weight=0, minsize=36)  # toolbar

        # Original Signal Plot
        self.canvas1 = FigureCanvasTkAgg(original, master=plots_frame)
        self.canvas1.get_tk_widget().grid(row=0, column=0, sticky="nsew", padx=0, pady=(5, 5))
        self.ax1 = original.axes[0] if original.axes else original.add_subplot(111)

        x = np.arange(len(self.statusData.ring))
        self.line1, = self.ax1.plot(x, self.statusData.ring, color='cyan', animated=False)
        self.ax1.set_xlim(0, len(self.statusData.ring))
        self.ax1.relim()
        self.ax1.autoscale_view(scalex=False, scaley=True)
        self.canvas1.draw()
        self.ax1.xaxis.set_major_locator(MultipleLocator(self.statusData.block_size * 4))
        self.ax1.xaxis.set_minor_locator(MultipleLocator(self.statusData.block_size))
        self.ax1.grid(axis='x', which='minor', linestyle='--', linewidth=0.5, alpha=0.35)
        self.ax1.grid(axis='x', which='major', linestyle='-', linewidth=0.6, alpha=0.5)
        self.ax1.set_axisbelow(True)
        self.tb1 = self.add_toolbar_right(plots_frame, self.canvas1, row=0)
        


        # Modulated Signal Plot
        self.canvas2 = FigureCanvasTkAgg(modulated, master=plots_frame)
        self.canvas2.get_tk_widget().grid(row=1, column=0, sticky="nsew", padx=0, pady=(0, 5))
        self.ax2 = modulated.axes[0] if modulated.axes else modulated.add_subplot(111)

        x2 = np.arange(len(self.statusData.mod_ring))
        self.line2, = self.ax2.plot(x2, self.statusData.mod_ring, color='orange', animated=False)
        self.ax2.set_xlim(0, len(self.statusData.mod_ring))
        self.ax2.relim()
        self.ax2.autoscale_view(scalex=False, scaley=True)
        self.canvas2.draw()
        self.ax2.xaxis.set_major_locator(MultipleLocator(self.statusData.block_size * 4))
        self.ax2.xaxis.set_minor_locator(MultipleLocator(self.statusData.block_size))
        self.ax2.grid(axis='x', which='minor', linestyle='--', linewidth=0.5, alpha=0.35)
        self.ax2.grid(axis='x', which='major', linestyle='-', linewidth=0.6, alpha=0.5)
        self.ax2.set_axisbelow(True)
        self.tb2 = self.add_toolbar_right(plots_frame, self.canvas2, row=1)
        

         # Demodulated Signal Plot
        self.canvas3 = FigureCanvasTkAgg(demodulated, master=plots_frame)
        self.canvas3.get_tk_widget().grid(row=2, column=0, sticky="nsew", padx=0, pady=0)
        self.ax3 = demodulated.axes[0] if demodulated.axes else demodulated.add_subplot(111)
        
        x3 = np.arange(len(self.statusData.demod_ring))
        self.line3, = self.ax3.plot(x3, self.statusData.demod_ring, color='magenta', animated=False)
        self.ax3.set_xlim(0, len(self.statusData.demod_ring))
        self.ax3.relim()
        self.ax3.autoscale_view(scalex=False, scaley=True)
        self.canvas3.draw()
        self.ax3.xaxis.set_major_locator(MultipleLocator(self.statusData.block_size * 4))
        self.ax3.xaxis.set_minor_locator(MultipleLocator(self.statusData.block_size))
        self.ax3.grid(axis='x', which='minor', linestyle='--', linewidth=0.5, alpha=0.35)
        self.ax3.grid(axis='x', which='major', linestyle='-', linewidth=0.6, alpha=0.5)
        self.ax3.set_axisbelow(True)
        self.tb3 = self.add_toolbar_right(plots_frame, self.canvas3, row=2)

