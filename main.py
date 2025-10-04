#############################################################################################################                                 
#                                  Intituto Tecnologico de Costa Rica                                       #
#                         Proyecto de Aplicación de Ingeniería en Computadores                              #
#                                                                                                          #
#                                        ModuTEC Simulator                                                  #
#                                                                                                           #
#                        Developed by: Mauricio Calderon Chavarria - 2019182667                             #
#                                                                                                           #
#                                         Version: 1.0.0                                                    #
#############################################################################################################

from app.dashboard import Dashboard
from app.appstate import AppState
import customtkinter as ctk

def main():
     statusData = AppState()
     ctk.set_appearance_mode("dark")
     ctk.set_default_color_theme("green")
     app = Dashboard(statusData)
     app.mainloop()

if __name__ == "__main__":
    main()
