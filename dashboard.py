import customtkinter as ctk
from tkinter import font, filedialog
from scipy.io import wavfile
import os
from styles import *
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from PIL import Image
from pydub import AudioSegment
import utils.windowCenter as wc
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
matplotlib.use('Agg')

class VerticalRightToolbar(NavigationToolbar2Tk):
    toolitems = [t for t in NavigationToolbar2Tk.toolitems
                 if t and t[0] in ("Home", "Pan", "Zoom", "Save")]

    def __init__(self, canvas, window):
        super().__init__(canvas, window, pack_toolbar=False)

        # fondo oscuro del toolbar y sin bordes
        try:
            self.configure(bg=MAIN_BACKGROUND_COLOR, bd=0, relief="flat", highlightthickness=0)
        except Exception:
            pass

        # elimina la etiqueta de estado (es la que suele dejar rectángulos blancos)
        try:
            if hasattr(self, "_message_label") and self._message_label:
                self._message_label.pack_forget()
                self._message_label.destroy()
        except Exception:
            pass
        self.set_message = lambda *a, **k: None

    # Botones en columna y sin fondos/bordes blancos
    def _Button(self, text, image_file, toggle, command):
        b = super()._Button(text, image_file, toggle, command)
        try:
            b.configure(bg=MAIN_BACKGROUND_COLOR,
                        activebackground=MAIN_BACKGROUND_COLOR,
                        bd=0, relief="flat", highlightthickness=0)
        except Exception:
            pass
        b.pack_configure(side="top", pady=6, padx=0)
        return b

    # Separador muy fino con mismo color
    def _Spacer(self):
        s = ctk.CTkFrame(self, fg_color=MAIN_BACKGROUND_COLOR, height=4, width=32, corner_radius=0)
        s.pack(side="top", pady=4)
        return s
        
class Dashboard(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.audio_file_path = None
        self.songName = None
        self.is_running = False
        self.configWindow()
        self.panels()
        self.navbar_buttons()
        self.displaySignals()
        self.configurationPanel()
        

    def configWindow(self):
        self.title("ModuTEC Simulator")
        self.geometry("1600x900")
        wc.center_window(self, 1600, 900)
        self.iconbitmap("assets/logo.ico")

    def panels(self):
        self.navbar = ctk.CTkFrame(self, fg_color=NAVBAR_COLOR, height=50)
        self.navbar.pack(side="top", fill="both")

        self.sidebar = ctk.CTkFrame(self, fg_color=NAVBAR_COLOR, width=200)
        self.sidebar.pack(side="left", fill="both", expand=False)
        
        self.optionsHeader = ctk.CTkFrame(self.sidebar, fg_color=SIDEBAR_COLOR, width=200)
        self.optionsHeader.pack(side="top", padx=10, pady=(2, 10), fill="x")
        
        self.optionsMenu = ctk.CTkFrame(self.sidebar, fg_color=SIDEBAR_COLOR, width=200)
        self.optionsMenu.pack(padx=10, pady=(2, 10), fill="x")

        self.applyButton = ctk.CTkButton(self.sidebar, text="Apply Changes", fg_color="#28a745", command=self.applyChanges)
        self.applyButton.pack(side="bottom", padx=10, pady=(5, 20), fill="x")
        
        self.mainArea = ctk.CTkFrame(self, fg_color=MAIN_BACKGROUND_COLOR)
        self.mainArea.pack(side="right", fill="both", expand=True)
        
        self.rightsidebar = ctk.CTkFrame(self.mainArea, fg_color=MAIN_BACKGROUND_COLOR, width=200, height=500)
        self.rightsidebar.pack(side="right", pady=(10, 10), padx=(10, 10), fill="x")

        self.currentConfig = ctk.CTkFrame(self.rightsidebar, fg_color=SIDEBAR_COLOR, width=250, height=250)
        self.currentConfig.pack(side="top", pady=(2, 10), fill="x")
        
        self.resultsArea = ctk.CTkFrame(self.rightsidebar, fg_color=SIDEBAR_COLOR, width=250, height=600)
        self.resultsArea.pack(pady=(2, 10), fill="x")
        

    def navbar_buttons(self):
        fontAwesome = ctk.CTkFont(family='FontAwesome', size=26, weight="bold")
        fontAwesomeBtns = ctk.CTkFont(family='FontAwesome', size=18, weight="bold")

        logoIcon = Image.open("assets/logo.ico")
        self.iconImage = ctk.CTkImage(dark_image=logoIcon, size=(50, 50))
        self.iconLabel = ctk.CTkButton(self.navbar, image=self.iconImage, text= "", fg_color=NAVBAR_COLOR, command=self.toogleAnimation, hover= False, width=0, height=0)
        self.iconLabel.pack(side="left", padx=0, pady=0)

        self.titleLabel = ctk.CTkLabel(self.navbar, text="ModuTEC", text_color="white", font=fontAwesome)
        self.titleLabel.pack(side="left", padx=0, pady=0)
        
        titleIcon = Image.open("assets/title.png")
        self.iconImage = ctk.CTkImage(dark_image=titleIcon, size=(400, 50))
        self.dashLabel = ctk.CTkLabel(self.navbar, image=self.iconImage, text= "", fg_color=NAVBAR_COLOR, width=0, height=0)
        self.dashLabel.pack(side="left", fill="x", expand=True)

        self.startSimulation = ctk.CTkButton(self.navbar, text="Run", fg_color=RUNNING_COLOR, text_color="white",
                                             font=fontAwesomeBtns, width=60, height=30, command=self.simulationButtonAnimation, hover=False)
        self.startSimulation.pack(side="right", padx=5, pady=5)

    def simulationButtonAnimation(self):
        if not self.is_running:
            self.startSimulation.configure(text="Stop", fg_color=STOP_COLOR)
            self.is_running = True
        else:
            self.startSimulation.configure(text="Run", fg_color=RUNNING_COLOR)
            self.is_running = False

    def toogleAnimation(self):
        if self.sidebar.winfo_ismapped():
            self.sidebar.pack_forget()
        else:
            self.sidebar.pack(side="left", fill="both")

    def create_dark_plot(self, title):
        fig, ax = plt.subplots(figsize=(4, 3))
        fig.patch.set_facecolor("#2a3138")
        ax.set_facecolor("#2a3138")
        x = np.linspace(0, 10, 100)
        y = np.sin(x) + np.random.normal(0, 0.1, size=x.shape)
        ax.plot(x, y, color='cyan')
        ax.set_title(title, color='white', pad=10)
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.margins(x=0.0)                                    # sin margen extra
        fig.subplots_adjust(left=0.08, right=0.995,          # pega el eje al borde derecho
                            top=0.90, bottom=0.16)
        return fig

    def configurationPanel(self):
        
        labelFont = ctk.CTkFont(family='FontAwesome', size=12)
        titleFont = ctk.CTkFont(family='FontAwesome', size=14, weight="bold")

        ctk.CTkLabel(self.optionsHeader, text="⚙️  Configuration Panel", font=titleFont, text_color="white",
                     anchor="w").pack(pady=(15, 10), padx=10, anchor="w")

        uploadIcon = Image.open("assets/upload.png")
        self.uploadIcon = ctk.CTkImage(dark_image=uploadIcon, size=(30, 30))
        ctk.CTkLabel(self.optionsHeader, text="Audio File", font=labelFont, text_color="white").pack(padx=10, anchor="w")
        self.upload_button = ctk.CTkButton(self.optionsHeader, text="Upload audio file", command=self.loadAudio, hover=False, compound="left", image=self.uploadIcon, fg_color=SIDEBAR_COLOR)
        self.upload_button.pack(padx=10, fill="x")
        self.songName = ctk.CTkLabel(self.optionsHeader, text="Load a song...", font=labelFont, text_color="white", wraplength=160,)
        self.songName.pack(padx=10, pady=(0, 10),fill="x")

        ctk.CTkLabel(self.optionsHeader, text="Select Modulation Type", font=labelFont, text_color="white").pack(padx=10, anchor="w")
        self.mod_type = ctk.StringVar(value="AM")
        self.mod_dropdown = ctk.CTkOptionMenu(self.optionsHeader, variable=self.mod_type, values=["AM", "FM", "ASK", "FSK"], command=self.customOptions)
        self.mod_dropdown.pack(padx=10, pady=(0, 15))
        
        
        '''
        ctk.CTkLabel(self.sidebar, text="Sample Rate", font=labelFont, text_color="white").pack(padx=10, anchor="w")
        self.sample_rate_input = ctk.CTkEntry(self.sidebar, placeholder_text="44100")
        self.sample_rate_input.pack(padx=10, pady=(0, 15), fill="x")

        ctk.CTkLabel(self.sidebar, text="Carrier Signal Frequency", font=labelFont, text_color="white").pack(padx=10, anchor="w")
        self.carrier_freq_input = ctk.CTkEntry(self.sidebar, placeholder_text="100000")
        self.carrier_freq_input.pack(padx=10, pady=(0, 15), fill="x")
        '''
        
        
    def customOptions(self,value):
        for widget in self.optionsMenu.winfo_children():
            widget.destroy()
        
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
            
            
    def applyChanges(self):      
        if self.audio_file_path:
            sample_rate, data = wavfile.read(self.audio_file_path)

            if len(data.shape) > 1:
                data = data[:, 0]

            self.sample_rate_input.delete(0, "end")
            self.sample_rate_input.insert(0, str(sample_rate))

            if data.dtype == np.int16:
                data = data / 32768.0

            max_points = 44100
            if len(data) > max_points:
                data = data[:max_points]

            self.canvas1.figure.clf()
            ax = self.canvas1.figure.add_subplot(111)
            ax.plot(data, color='cyan')
            ax.set_title("Señal Original", color='white')
            ax.tick_params(colors='white')
            ax.set_facecolor("#2a3138")  
            ax.spines['bottom'].set_color('white')
            ax.spines['left'].set_color('white')
            self.canvas1.draw()

    def loadAudio(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Audio Files", "*.wav *.mp3"), ("WAV Files", "*.wav"), ("MP3 Files", "*.mp3")],
            title="Select an audio file"
        )

        # If MP3, convert to WAV using pydub
        if file_path and file_path.lower().endswith('.mp3'):
            mp3_audio = AudioSegment.from_mp3(file_path)
            wav_path = file_path + ".temp.wav"
            mp3_audio.export(wav_path, format="wav")
            self.audio_file_path = wav_path
        else:
            self.audio_file_path = file_path    

        self.songName.configure(text=os.path.basename(self.audio_file_path) if self.audio_file_path else "Load a song...")

    def add_toolbar_right(self, parent_grid, canvas, row:int):
        holder = ctk.CTkFrame(parent_grid, fg_color=MAIN_BACKGROUND_COLOR, width=44)
        holder.grid(row=row, column=1, sticky="ns", padx=0, pady=0)
        parent_grid.grid_columnconfigure(1, weight=0, minsize=44)

        tb = VerticalRightToolbar(canvas, holder)
        tb.update()
        # como es un Frame, lo pones con grid dentro del holder
        tb.grid(row=0, column=0, sticky="ns", padx=0, pady=0)
        return tb


    def displaySignals(self):
        fig1 = self.create_dark_plot("Señal Original")
        fig2 = self.create_dark_plot("Señal Modulada")
        fig3 = self.create_dark_plot("Señal Demodulada")

        plots_frame = ctk.CTkFrame(self.mainArea, fg_color=MAIN_BACKGROUND_COLOR)
        plots_frame.pack(fill="both", expand=True)



        for r in (0, 1, 2):
            plots_frame.grid_rowconfigure(r, weight=1, uniform="plots")
            plots_frame.grid_columnconfigure(0, weight=1)              # canvas 
            plots_frame.grid_columnconfigure(1, weight=0, minsize=36)  # toolbar
    
        self.canvas1 = FigureCanvasTkAgg(fig1, master=plots_frame)
        self.canvas1.get_tk_widget().grid(row=0, column=0, sticky="nsew", padx=0, pady=(5, 5))
        self.canvas1.draw()
        self.tb1 = self.add_toolbar_right(plots_frame, self.canvas1, row=0)

        self.canvas2 = FigureCanvasTkAgg(fig2, master=plots_frame)
        self.canvas2.get_tk_widget().grid(row=1, column=0, sticky="nsew", padx=0, pady=(0, 5))
        self.canvas2.draw()
        self.tb2 = self.add_toolbar_right(plots_frame, self.canvas2, row=1)

        self.canvas3 = FigureCanvasTkAgg(fig3, master=plots_frame)
        self.canvas3.get_tk_widget().grid(row=2, column=0, sticky="nsew", padx=0, pady=0)
        self.canvas3.draw()
        self.tb3 = self.add_toolbar_right(plots_frame, self.canvas3, row=2)

if __name__ == "__main__":
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("green")
    app = Dashboard()
    app.mainloop()
