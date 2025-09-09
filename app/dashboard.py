import os
import numpy as np
from PIL import Image
from utils.styles import *
import customtkinter as ctk
from scipy.io import wavfile
from tkinter import filedialog
import matplotlib.pyplot as plt
from utils import windowCenter as wc
from core.audio.AudioController import AudioController
from app.ui.SamplingStream import SamplingStream
from app.ui.VerticalRightToolbar import VerticalRightToolbar
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        
class Dashboard(ctk.CTk):
    
    def __init__(self, statusData):
        super().__init__()
        
        # Application state variable
        
        self.statusData = statusData
        
        self.SamplingStream = SamplingStream(statusData)
        
        self.audioController = AudioController(statusData)
        
        # Initialize main panels
        
        self.config_window()
        self.panels()
        self.navbar_panel()
        self.display_signals()
        self.left_sidebar_panel()


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
        self.rightsidebar = ctk.CTkFrame(self.mainArea, fg_color=MAIN_BACKGROUND_COLOR, width=200, height=500)
        self.rightsidebar.pack(side="right", pady=(10, 10), padx=(10, 10), fill="x")

        # Right Sidebar Sub-frame for current configuration
        self.currentConfig = ctk.CTkFrame(self.rightsidebar, fg_color=SIDEBAR_COLOR, width=250, height=250)
        self.currentConfig.pack(side="top", pady=(2, 10), fill="x")

        # Right Sidebar Sub-frame for results logs
        self.resultsArea = ctk.CTkFrame(self.rightsidebar, fg_color=SIDEBAR_COLOR, width=250, height=600)
        self.resultsArea.pack(pady=(2, 10), fill="x")
    
    
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
        self.mod_type = ctk.StringVar(value="AM")
        self.mod_dropdown = ctk.CTkOptionMenu(self.optionsHeader, variable=self.mod_type, values=["AM", "FM", "ASK", "FSK"], command=self.custom_options)
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
            self.statusData.is_running = True

            # Iniciar el stream y el ploteo en tiempo real
            self.SamplingStream.start_stream()
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
        
        #TODO : ADD CUSTOM OPTIONS LOGIC HERE
        if value == "AM":
            ctk.CTkLabel(self.optionsMenu, text="Sample Rate AM", text_color="white").pack(padx=10, anchor="w")
            self.sample_rate_input = ctk.CTkEntry(self.optionsMenu, placeholder_text="44100")
            self.sample_rate_input.pack(padx=10, pady=(0, 15), fill="x")
        elif value == "FM":
            ctk.CTkLabel(self.optionsMenu, text="Sample Rate FM", text_color="white").pack(padx=10, anchor="w")
            self.sample_rate_input = ctk.CTkEntry(self.optionsMenu, placeholder_text="44100")
            self.sample_rate_input.pack(padx=10, pady=(0, 15), fill="x")
        elif value == "ASK":
            ctk.CTkLabel(self.optionsMenu, text="Sample Rate ASK", text_color="white").pack(padx=10, anchor="w")
            self.sample_rate_input = ctk.CTkEntry(self.optionsMenu, placeholder_text="44100")
            self.sample_rate_input.pack(padx=10, pady=(0, 15), fill="x")
        elif value == "FSK":
            ctk.CTkLabel(self.optionsMenu, text="Sample Rate FSK", text_color="white").pack(padx=10, anchor="w")
            self.sample_rate_input = ctk.CTkEntry(self.optionsMenu, placeholder_text="44100")
            self.sample_rate_input.pack(padx=10, pady=(0, 15), fill="x")

        '''
        ctk.CTkLabel(self.sidebar, text="Carrier Signal Frequency", font=labelFont, text_color="white").pack(padx=10, anchor="w")
        self.carrier_freq_input = ctk.CTkEntry(self.sidebar, placeholder_text="100000")
        self.carrier_freq_input.pack(padx=10, pady=(0, 15), fill="x")
        '''

    # Function to apply changes to the current simulation settings.
    
    def applyChanges(self): 
        if self.statusData.audio_file_path:
            
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
                    
            print("Changes applied successfully.")
   
   
    def add_toolbar_right(self, parent_grid, canvas, row:int):
        
        #TODO: FIX TOOLBAR SIZE ISSUE
        
        holder = ctk.CTkFrame(parent_grid, fg_color=MAIN_BACKGROUND_COLOR, width=44)
        holder.grid(row=row, column=1, sticky="ns", padx=0, pady=0)
        parent_grid.grid_columnconfigure(1, weight=0, minsize=44)

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
        # TODO: Line does not update when audio is playing
        self.canvas1 = FigureCanvasTkAgg(original, master=plots_frame)
        self.canvas1.get_tk_widget().grid(row=0, column=0, sticky="nsew", padx=0, pady=(5, 5))
        self.ax1 = original.axes[0] if original.axes else original.add_subplot(111)
        self.line1, = self.ax1.plot(np.zeros_like(self.statusData.ring), color='cyan', animated=False)
        self.ax1.set_xlim(0, len(self.statusData.ring))
        self.ax1.set_ylim(-1.1, 1.1)
        self.canvas1.draw()
        self.tb1 = self.add_toolbar_right(plots_frame, self.canvas1, row=0)

        # Modulated Signal Plot
        self.canvas2 = FigureCanvasTkAgg(modulated, master=plots_frame)
        self.canvas2.get_tk_widget().grid(row=1, column=0, sticky="nsew", padx=0, pady=(0, 5))
        self.canvas2.draw()
        self.tb2 = self.add_toolbar_right(plots_frame, self.canvas2, row=1)

        # Demodulated Signal Plot
        self.canvas3 = FigureCanvasTkAgg(demodulated, master=plots_frame)
        self.canvas3.get_tk_widget().grid(row=2, column=0, sticky="nsew", padx=0, pady=0)
        self.canvas3.draw()
        self.tb3 = self.add_toolbar_right(plots_frame, self.canvas3, row=2)


