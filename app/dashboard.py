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
from app.ui.SamplingStream import SamplingStream
from core.audio.AudioController import AudioController
from matplotlib.ticker import MultipleLocator
from app.ui.VerticalRightToolbar import VerticalRightToolbar
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from core.algorithms.AM import am_prepare_state, am_modulate_block, am_demodulate_block

        
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
        for widget in self.currentConfig.winfo_children():
            widget.destroy()
        
        labelFont = ctk.CTkFont(family='FontAwesome', size=12)
        labelTitleFont = ctk.CTkFont(family='FontAwesome', size=12, weight="bold")
        titleFont = ctk.CTkFont(family='FontAwesome', size=18, weight="bold")
    
        ctk.CTkLabel(self.currentConfig, text="Simulation Overview", font=titleFont, text_color="white",
                 anchor="center", justify="center").pack(pady=(5, 10), padx=10, anchor="center")

        # Crear un frame para el grid de overview
        overview_grid = ctk.CTkFrame(self.currentConfig, fg_color=SIDEBAR_COLOR)
        overview_grid.pack(padx=10, pady=(0, 10), fill="x")

        # Configurar 2 columnas y 4 filas
        for i in range(4):
            overview_grid.grid_rowconfigure(i, weight=1)
        for j in range(2):
            overview_grid.grid_columnconfigure(j, weight=1)
            
        if self.statusData.audio_file_path == None and self.statusData.modulation_type.get() == "Select type":
            filename = "No file loaded"
            mod_type = "Not selected"
        else:
            filename = os.path.basename(self.statusData.audio_file_path)
            mod_type = self.statusData.modulation_type.get()
            
        # Ejemplo de widgets en el grid
        ctk.CTkLabel(overview_grid, text="Audio File", font=labelTitleFont, text_color="white",anchor="center", justify="center").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        ctk.CTkLabel(overview_grid, text="Modulation Type", font=labelTitleFont, text_color="white", anchor="center", justify="center").grid(row=0, column=1, sticky="w", padx=5, pady=2)
        #ctk.CTkLabel(overview_grid, text=filename, font=labelFont, text_color="white").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        ctk.CTkLabel(overview_grid,text=filename,font=labelFont,text_color="white",wraplength=180,justify="left").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        ctk.CTkLabel(overview_grid, text=mod_type, font=labelFont, text_color="white").grid(row=1, column=1, sticky="w", padx=5, pady=2)
        ctk.CTkLabel(overview_grid, text="Sample Rate: " + str(self.statusData.sample_rate), font=labelFont, text_color="white").grid(row=2, column=0, sticky="w", padx=5, pady=2)
        ctk.CTkLabel(overview_grid, text="Bitrate: " + str(self.statusData.am_mu), font=labelFont, text_color="white").grid(row=2, column=1, sticky="w", padx=5, pady=2)
        ctk.CTkLabel(overview_grid, text="Carrier Frequency: " + str(self.statusData.am_fc), font=labelFont, text_color="white").grid(row=3, column=0, sticky="w", padx=5, pady=2)
        ctk.CTkLabel(overview_grid, text="Carrier Amplitude: " + str(self.statusData.am_Ac), font=labelFont, text_color="white").grid(row=3, column=1, sticky="w", padx=5, pady=2)
    
    
    
    # Right Sidebar Results Panel
    def right_results_panel(self):
        #for w in self.resultsArea.winfo_children(): w.destroy()

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
            
        #TODO AGREGAR ACA QUE SE CALCULEN LOS VALORES PARA LA MODULACION
        
        
    
    
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
            self.statusData.am_xscale = None
            
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
        
            ctk.CTkLabel(self.optionsMenu, text="Carrier Frequency FSK (BFSK)", text_color="white").pack(padx=10, anchor="w")
            self.carrier_freq_input = ctk.CTkEntry(self.optionsMenu, placeholder_text=self.statusData.recommended_fsk_fc)
            self.carrier_freq_input.pack(padx=10, pady=(0, 15), fill="x")

            ctk.CTkLabel(self.optionsMenu, text="Carrier Amplitude FSK (BFSK)", text_color="white").pack(padx=10, anchor="w")
            self.carrier_amp_input = ctk.CTkEntry(self.optionsMenu, placeholder_text=self.statusData.recommended_fsk_Ac)
            self.carrier_amp_input.pack(padx=10, pady=(0, 15), fill="x")

            ctk.CTkLabel(self.optionsMenu, text="Bitrate FSK (BFSK)", text_color="white").pack(padx=10, anchor="w")
            self.bitrate_input = ctk.CTkEntry(self.optionsMenu, placeholder_text=self.statusData.recommended_fsk_bitrate)
            self.bitrate_input.pack(padx=10, pady=(0, 15), fill="x")
        
    def checkOptions(self):
        
        # Setting the default recommended values
        if (self.sample_rate_input.get() == ""):
            self.statusData.sample_rate = self.statusData.recommended_Fs
        else:
            self.statusData.sample_rate = self.sample_rate_input.get()
            
        if (self.carrier_freq_input.get() == ""):
            self.statusData.am_fc = self.statusData.recommended_am_fc
        else: 
            self.statusData.am_fc = self.carrier_freq_input.get()
            
        if (self.carrier_amp_input.get() == ""):
            self.statusData.am_Ac = self.statusData.recommended_am_Ac
        else:
            self.statusData.am_Ac = self.carrier_amp_input.get()   
                
        if (self.modulation_index_input.get() == ""):
            self.statusData.am_mu = self.statusData.recommended_am_mu
        else:
            self.statusData.am_mu = self.modulation_index_input.get()
        
    
    
    # Function to apply changes to the current simulation settings.
    def applyChanges(self):

        if self.statusData.audio_file_path and (self.statusData.modulation_type.get() != "Select type"):
            print("Modulation type:", self.statusData.modulation_type.get())
            sample_rate, data = wavfile.read(self.statusData.audio_file_path)

            # Actualizar parámetros según entradas del usuario
            self.right_overview_panel()
            
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
                
            self.checkOptions()
            
            print("Valores para la simulacion elegidos")
            print(self.statusData.sample_rate)
            print(self.statusData.am_fc)
            print(self.statusData.am_Ac)
            print(self.statusData.am_mu)
            
            
            print("Changes applied successfully.")
        else :
            CTkMessagebox(title="Error", message="No audio file loaded or modulation type not selected.", icon="warning")
   
    def _reset_ring_ui(self):
        N = int(self.statusData.window_seconds * float(self.statusData.sample_rate))
        N = max(1024, N)

        # Ring original
        self.statusData.ring = np.zeros(N, dtype=np.float32)
        x1 = np.arange(N)
        self.line1.set_data(x1, self.statusData.ring)
        self.ax1.set_xlim(0, N)
        self.ax1.set_ylim(-1.1, 1.1)
        self.canvas1.draw_idle()

        # Ring modulado
        self.statusData.mod_ring = np.zeros(N, dtype=np.float32)
        x2 = np.arange(N)
        self.line2.set_data(x2, self.statusData.mod_ring)
        self.ax2.set_xlim(0, N)
        self.ax2.set_ylim(-1.1, 1.1)
        self.canvas2.draw_idle()

        # Ring demodulado
        self.statusData.demod_ring = np.zeros(N, dtype=np.float32)
        x3 = np.arange(N)
        self.line3.set_data(x3, self.statusData.demod_ring)
        self.ax3.set_xlim(0, N)
        self.ax3.set_ylim(-1.1, 1.1)
        self.canvas3.draw_idle()

        # Reiniciar estado AM para nueva Fs
        self.statusData.am_initialized = False
        self.statusData.am_phase = 0.0
        self.statusData.am_xscale = None



    def _ui_timer(self):
        drained = False

        # SR cambiado => reajustar y reiniciar estado AM
        if getattr(self.statusData, "needs_reset", False):
            self.statusData.needs_reset = False
            self._reset_ring_ui()

        # Drenar cola de entrada (puede haber varios chunks por tick)
        while True:
            try:
                chunk = self.statusData.q.get_nowait()
                self.statusData.chunk_seq += 1
                cid = self.statusData.chunk_seq
                if self.statusData.ncc_pairer is not None:
                    self.statusData.ncc_pairer.push_original(cid, np.asarray(chunk, dtype=np.float32))
            except queue.Empty:
                break

            drained = True

            # --- 1) Actualizar ring ORIGINAL ---
            ring = self.statusData.ring
            if len(chunk) >= len(ring):
                ring[:] = chunk[-len(ring):]
            else:
                L = len(chunk)
                ring[:-L] = ring[L:]
                ring[-L:] = chunk

            # --- 2) MODULACIÓN AM (si está activa) ---
            s_mod = None

            if self.statusData.modulation_enabled and self.statusData.modulation_type.get() == "AM":
                if not self.statusData.am_initialized:
                    xscale = am_prepare_state(first_chunk=chunk.astype(np.float32))
                    self.statusData.am_xscale = float(xscale)
                    self.statusData.am_phase = 0.0
                    self.statusData.am_initialized = True

                state_blk = {
                    "fc":    float(self.statusData.am_fc),
                    "mu":    float(self.statusData.am_mu),
                    "Ac":    float(self.statusData.am_Ac),
                    "phase": float(self.statusData.am_phase),
                    "xscale":float(self.statusData.am_xscale),
                }

                s_mod, state_blk = am_modulate_block(
                    x=chunk.astype(np.float32),
                    Fs=float(self.statusData.sample_rate),
                    state=state_blk
                )
                self.statusData.am_phase = state_blk["phase"]

                # Actualizar ring MODULADO
                mring = self.statusData.mod_ring
                if len(s_mod) >= len(mring):
                    mring[:] = s_mod[-len(mring):]
                else:
                    Lm = len(s_mod)
                    mring[:-Lm] = mring[Lm:]
                    mring[-Lm:] = s_mod

            # --- 3) DEMODULACIÓN AM (si hay modulada y AM activa) ---
            if s_mod is not None:
                demod_state = {
                    "fc": float(self.statusData.am_fc),
                    "mu": float(self.statusData.am_mu),
                    "Ac": float(self.statusData.am_Ac),
                    "fmax": float(self.statusData.fmax),         
                    "lp_ym1": float(getattr(self.statusData, "am_lp_ym1", 0.0)),  
                }

                s_demod = am_demodulate_block(
                    s=s_mod,
                    Fs=float(self.statusData.sample_rate),
                    state=demod_state,          
                )
                
                # Empujar DEMOD y registrar log con color según umbral
                if self.statusData.ncc_pairer is not None:
                    res = self.statusData.ncc_pairer.push_demodulated(cid, np.asarray(s_demod, dtype=np.float32))
                    if res is not None:
                        pct = res["percent"]
                        thr = getattr(self.statusData, "ncc_threshold", 70.0)
                        color = "#00FF00" if pct >= thr else "#FF3B30"
                        # Tu método de logging en el panel derecho:
                        self.log_result(f"Chunk #{res['chunk_id']} processed NCC: {pct:.1f}%", color=color)

                
                self.statusData.am_lp_ym1 = float(demod_state.get("lp_ym1", 0.0))

                dring = self.statusData.demod_ring
                if len(s_demod) >= len(dring):
                    dring[:] = s_demod[-len(dring):]
                else:
                    Ld = len(s_demod)
                    dring[:-Ld] = dring[Ld:]
                    dring[-Ld:] = s_demod

        # --- 4) Pintar si hubo datos ---
        if drained:
            self.line1.set_ydata(self.statusData.ring)
            if self.statusData.modulation_enabled and self.statusData.modulation_type.get() == "AM":
                self.line2.set_ydata(self.statusData.mod_ring)
                self.line3.set_ydata(self.statusData.demod_ring)
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
        self.ax1.set_ylim(-1.1, 1.1)
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
        self.ax2.set_ylim(-1.1, 1.1)
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
        self.ax3.set_ylim(-1.1, 1.1)
        self.canvas3.draw()
        self.ax3.xaxis.set_major_locator(MultipleLocator(self.statusData.block_size * 4))
        self.ax3.xaxis.set_minor_locator(MultipleLocator(self.statusData.block_size))
        self.ax3.grid(axis='x', which='minor', linestyle='--', linewidth=0.5, alpha=0.35)
        self.ax3.grid(axis='x', which='major', linestyle='-', linewidth=0.6, alpha=0.5)
        self.ax3.set_axisbelow(True)
      
        
        self.tb3 = self.add_toolbar_right(plots_frame, self.canvas3, row=2)

