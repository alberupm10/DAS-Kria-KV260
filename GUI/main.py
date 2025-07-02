import customtkinter as ctk
import tkinter as tk
import os
import threading
from tkinter import messagebox, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from ssh_manager import *
from config import *
import numpy as np


class FPGAApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("TFG KRIA KV260")
        self.geometry("1400x1200")

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")

        # Frame Principal
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(fill="both", expand=True)

        self.config_params = {
            "grid_x": tk.IntVar(value=300),
            "grid_y": tk.IntVar(value=300),
            "x_start": tk.DoubleVar(value=-0.2),
            "x_end": tk.DoubleVar(value=0.3),
            "y_start": tk.DoubleVar(value=-0.05),
            "y_end": tk.DoubleVar(value=0.7),
            "iterations": tk.IntVar(value=1),
            "signal_damage": tk.StringVar(value="data_damage_scaled_processed.csv"),
            "output_filename": tk.StringVar(value="result_output.csv")
        }

        self.project_id = None
        self.fpga_ip = tk.StringVar(value="192.168.0.10")
        self.screens = {}
        self.show_screen("ip")

    def show_screen(self, screen_name):
        print(f"Mostrando pantalla: {screen_name}")

        for screen in self.main_frame.winfo_children():
            screen.pack_forget()

        if screen_name in self.screens:
            self.screens[screen_name].pack(fill="both", expand=True)

        else:
            new_screen = None
            if screen_name == "ip":
                new_screen = self.create_ip_screen()
            elif screen_name == "intermediate":
                new_screen = self.create_intermediate_screen()
            elif screen_name == "selection":
                new_screen = self.create_project_selection_screen()
            elif screen_name == "comparison":
                new_screen = self.create_comparison_screen()
            elif screen_name == "results":
                new_screen = self.create_results_screen()
            elif screen_name == "settings":
                new_screen = self.create_settings_screen()
            elif screen_name in ["config_1", "config_2", "config_3"]:
                project_id = int(screen_name.split("_")[1])
                new_screen = self.create_project_config_screen(project_id)

            if new_screen:
                self.screens[screen_name] = new_screen
                new_screen.pack(fill="both", expand=True)

    def create_ip_screen(self):
        frame = ctk.CTkFrame(self.main_frame, fg_color="gray25", corner_radius=10)

        content_frame = ctk.CTkFrame(frame, fg_color="gray30", corner_radius=10)
        content_frame.place(relx=0.5, rely=0.5, anchor="center")

        title_label = ctk.CTkLabel(content_frame, text="Ingrese la IP de la FPGA",
                                   font=("Arial", 20, "bold"), text_color="white")
        title_label.pack(pady=10, padx=40)

        self.ip_entry = ctk.CTkEntry(content_frame, textvariable=self.fpga_ip, width=200, font=("Arial", 14))
        self.ip_entry.pack(pady=10)

        connect_button = ctk.CTkButton(content_frame, text="Conectar", command=self.try_connect, width=150)
        connect_button.pack(pady=10)

        exit_button = ctk.CTkButton(content_frame, text="Salir", command=self.quit, fg_color="#dc3545", width=150)
        exit_button.pack(pady=10)

        return frame

    def try_connect(self):

        print(f"conectando ssh")
        self.ssh = connect_ssh(self.fpga_ip.get())
        if self.ssh:
            print(f"mostrando pantalla intermedia")
            self.show_screen("intermediate")
        else:
            self.show_screen("intermediate")
            messagebox.showerror("Error", "No se pudo conectar a la FPGA. Verifique la IP.")

    def log_message(self, message):
        for console in [getattr(self, "console_output_settings", None), getattr(self, "console_output_results", None)]:
            if console:
                console.configure(state="normal")
                console.insert("end", message + "\n")
                console.configure(state="disabled")
                console.see("end")

    def create_intermediate_screen(self):

        frame = ctk.CTkFrame(self.main_frame, fg_color="gray25", corner_radius=10)
        frame.pack(fill="both", expand=True)

        container = ctk.CTkFrame(frame, fg_color="gray30", corner_radius=10, width=400, height=300)
        container.place(relx=0.5, rely=0.5, anchor="center")

        title_label = ctk.CTkLabel(container, text="Seleccione el Modo de Operación",
                                   font=("Arial", 22, "bold"), text_color="white")
        title_label.pack(pady=20, padx=40)

        execute_button = ctk.CTkButton(container, text="Ejecutar Proyecto FPGA",
                                       command=lambda: self.show_screen("selection"),
                                       fg_color="#007BFF", font=("Arial", 18, "bold"), width=200)
        execute_button.pack(pady=10)

        compare_button = ctk.CTkButton(container, text="Comparar imágenes",
                                       command=lambda: self.show_screen("comparison"),
                                       fg_color="#28a745", font=("Arial", 18, "bold"), width=200)
        compare_button.pack(pady=10)

        compare_button = ctk.CTkButton(container, text="Configuración App",
                                       command=lambda: self.show_screen("settings"),
                                       fg_color="#FFA500", font=("Arial", 18, "bold"), width=200)
        compare_button.pack(pady=10)

        exit_button_s = ctk.CTkButton(container, text="Cerrar",
                                      command=self.quit,
                                      fg_color="#dc3545", font=("Arial", 16, "bold"), width=200)
        exit_button_s.pack(pady=10)

        return frame

    def create_resource_table(self, parent_frame, project_id):
        project_keys = list(RESOURCE_TABLE.keys())

        if project_id >= len(project_keys):
            return

        resource_key = project_keys[project_id]

        table_frame = ctk.CTkFrame(parent_frame, fg_color="gray30", corner_radius=10)
        table_frame.pack(fill="both", expand=True, padx=10, pady=10)

        title_label = ctk.CTkLabel(table_frame, text="Recursos", font=("Arial", 18, "bold"), text_color="white")
        title_label.pack(pady=5)

        headers = ["LUT", "FF", "BRAM", "URAM", "DSP"]
        header_frame = ctk.CTkFrame(table_frame, fg_color="gray40", corner_radius=5)
        header_frame.pack(fill="x", padx=5, pady=5)

        for header in headers:
            ctk.CTkLabel(header_frame, text=header, font=("Arial", 14, "bold"), text_color="white").pack(side="left",
                                                                                                         expand=True)

        values_frame = ctk.CTkFrame(table_frame, fg_color="gray50", corner_radius=5)
        values_frame.pack(fill="x", padx=5, pady=5)

        for key in headers:
            ctk.CTkLabel(values_frame, text=str(RESOURCE_TABLE[resource_key][key]), font=("Arial", 20),
                         text_color="white").pack(side="left", expand=True)

        percentage_headers = ["LUT%", "FF%", "BRAM%", "URAM%", "DSP%"]
        percentage_frame = ctk.CTkFrame(table_frame, fg_color="gray50", corner_radius=5)
        percentage_frame.pack(fill="x", padx=5, pady=5)

        for key in percentage_headers:
            ctk.CTkLabel(percentage_frame, text=str(RESOURCE_TABLE[resource_key][key]), font=("Arial", 20),
                         text_color="white").pack(side="left", expand=True)

        description_frame = ctk.CTkFrame(parent_frame, fg_color="gray25", corner_radius=10)
        description_frame.pack(fill="both", expand=True, padx=10, pady=10)

        description = PROJECT_CONFIG[project_id + 1]["description"]

        desc_label = ctk.CTkLabel(description_frame, text=description, font=("Arial", 20),
                                  text_color="white", wraplength=300, justify="left")
        desc_label.pack(padx=10, pady=10)

    def create_project_selection_screen(self):
        print(f"Creando pantalla de seleccion de proyecto")
        frame = ctk.CTkFrame(self.main_frame, fg_color="gray25", corner_radius=10)
        frame.pack(fill="both", expand=True)

        title_label = ctk.CTkLabel(frame, text="Seleccione un Proyecto",
                                   font=("Arial", 22, "bold"), text_color="white")
        title_label.pack(pady=20)

        project_container = ctk.CTkFrame(frame, fg_color="gray30", corner_radius=10)
        project_container.pack(fill="both", expand=True, padx=20, pady=20)

        projects = [
            (1, "VDFD", "#007BFF", "config_1"),
            (2, "VDM", "#28a745", "config_2"),
            (3, "VAD", "#ffc107", "config_3")
        ]

        for project_id, name, color, config_screen in projects:
            project_frame = ctk.CTkFrame(project_container, fg_color="gray40", corner_radius=10)
            project_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)

            project_label = ctk.CTkLabel(project_frame, text=name, font=("Arial", 16, "bold"), text_color="white")
            project_label.pack(pady=10)

            self.create_resource_table(project_frame, project_id - 1)

            select_button = ctk.CTkButton(
                project_frame, text="Seleccionar",
                command=lambda screen=config_screen: self.show_screen(screen),
                fg_color=color, font=("Arial", 16, "bold"), width=150, height=50
            )
            select_button.pack(pady=10)

        button_frame = ctk.CTkFrame(frame, fg_color="gray30")
        button_frame.pack(side="bottom", fill="x", padx=20, pady=10)  # 🔹 Expandir en toda la fila

        back_button = ctk.CTkButton(
            button_frame, text="⬅ Volver", command=lambda: self.show_screen("intermediate"),
            fg_color="#6c757d", font=("Arial", 16, "bold"), width=150
        )
        back_button.pack(side="left", padx=10)

        close_button = ctk.CTkButton(
            button_frame, text="Cerrar", command=self.quit,
            fg_color="#dc3545", font=("Arial", 16, "bold"), width=150
        )
        close_button.pack(side="right", padx=10)

        return frame

    def create_comparison_screen(self):
        print(f"Creando pantalla de comparación")

        self.comparison_screen = ctk.CTkFrame(self.main_frame, fg_color="gray25", corner_radius=10)
        self.comparison_screen.pack(fill="both", expand=True)

        title_label = ctk.CTkLabel(self.comparison_screen, text="Comparación de Resultados",
                                   font=("Arial", 22, "bold"), text_color="white")
        title_label.pack(pady=10)

        file_frame = ctk.CTkFrame(self.comparison_screen, fg_color="gray30", corner_radius=10)
        file_frame.pack(pady=10, fill="x", padx=20)

        ctk.CTkLabel(file_frame, text="Seleccione archivos de resultados:", font=("Arial", 16, "bold")).pack(pady=5)

        self.fpga_file_path = tk.StringVar()
        button_fpga = ctk.CTkButton(file_frame, text="Cargar Archivo 1",
                                    command=lambda: self.load_file(self.fpga_file_path),
                                    fg_color="#007BFF", width=150)
        button_fpga.pack(side="left", padx=10, pady=5)

        self.fpga_file_label = ctk.CTkLabel(file_frame, text="Archivo: No seleccionado", text_color="white")
        self.fpga_file_label.pack(side="left", padx=10)

        self.matlab_file_path = tk.StringVar()
        button_matlab = ctk.CTkButton(file_frame, text="Cargar Archivo 2",
                                      command=lambda: self.load_file(self.matlab_file_path),
                                      fg_color="#28a745", width=150)
        button_matlab.pack(side="left", padx=10, pady=5)

        self.matlab_file_label = ctk.CTkLabel(file_frame, text="Archivo: No seleccionado", text_color="white")
        self.matlab_file_label.pack(side="left", padx=10)

        button_compare = ctk.CTkButton(file_frame, text="Comparar",
                                       command=self.compare_images, fg_color="#ffc107",
                                       width=150)
        button_compare.pack(side="left", padx=10, pady=5)

        result_frame = ctk.CTkFrame(self.comparison_screen, fg_color="gray30", corner_radius=10)
        result_frame.pack(fill="both", expand=True, padx=20, pady=10)

        self.graph_frame = ctk.CTkFrame(result_frame, fg_color="gray40", corner_radius=10)
        self.graph_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)

        self.metrics_frame = ctk.CTkFrame(result_frame, fg_color="gray40", corner_radius=10)
        self.metrics_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        button_frame = ctk.CTkFrame(self.comparison_screen, fg_color="gray30")
        button_frame.pack(fill="x", padx=20, pady=10)

        button_back = ctk.CTkButton(button_frame, text="Volver", command=lambda: self.show_screen("intermediate"),
                                    fg_color="#6c757d", font=("Arial", 16, "bold"), width=200)
        button_back.pack(side="left", padx=10)

        button_popup = ctk.CTkButton(button_frame, text="Mostrar Pop-up",
                                     command=self.show_comparison_popup_from_button,
                                     fg_color="#007BFF", font=("Arial", 16, "bold"), width=200)
        button_popup.pack(side="left", expand=True)

        button_exit = ctk.CTkButton(button_frame, text="Cerrar", command=self.quit,
                                    fg_color="#dc3545", font=("Arial", 16, "bold"), width=200)
        button_exit.pack(side="right", padx=10)

        return self.comparison_screen

    def create_project_config_screen(self, project_id):
        print(f"Creando pantalla de configuración para el Proyecto {project_id}")

        self.project_id = project_id

        include_grid = project_id != 2
        #include_entrada = project_id != 3
        include_entrada = True

        self.config_screen = ctk.CTkFrame(self.main_frame, fg_color="gray25", corner_radius=10)
        self.config_screen.pack(fill="both", expand=True)

        form_container = ctk.CTkFrame(self.config_screen, width=600, fg_color="gray30", corner_radius=10)
        form_container.place(relx=0.5, rely=0.4, anchor="center")  # 🔹 Centrado en la pantalla
        nombres_visuales = {1: "VDFD", 2: "VDM", 3: "VAD"}
        nombre_mostrar = nombres_visuales.get(project_id, f"{project_id}")
        title_label = ctk.CTkLabel(form_container, text=f"Configuración del Proyecto {nombre_mostrar}",
                                   font=("Arial", 22, "bold"), text_color="white")
        title_label.pack(pady=10)

        param_frame = ctk.CTkFrame(form_container, fg_color="gray40", corner_radius=10, width=500, height=400)
        param_frame.pack(pady=10, padx=10, fill="both", expand=False)
        param_frame.pack_propagate(False)

        self.create_common_parameters(param_frame, include_grid, include_entrada)

        button_frame = ctk.CTkFrame(form_container, fg_color="gray30")
        button_frame.pack(pady=20)

        ctk.CTkButton(button_frame, text="Ejecutar en FPGA",
                      command=lambda: [
                          self.run_process(project_id),
                          self.show_screen("results")],
                      fg_color="#28a745", font=("Arial", 18, "bold"), width=150, height=40).pack(side="left", padx=5)

        ctk.CTkButton(button_frame, text="Volver",
                      command=lambda: self.show_screen("selection"),
                      fg_color="#6c757d", font=("Arial", 18, "bold"), width=150, height=40).pack(side="left", padx=5)

        ctk.CTkButton(button_frame, text="Cerrar",
                      command=self.quit,
                      fg_color="#dc3545", font=("Arial", 18, "bold"), width=150, height=40).pack(side="left", padx=5)

        return self.config_screen

    def create_common_parameters(self, parent_frame, include_grid=True, include_entrada=True):

        params = [
            ("X_START", "x_start"),
            ("X_END", "x_end"),
            ("Y_START", "y_start"),
            ("Y_END", "y_end"),
            ("Iteraciones", "iterations")
        ]

        for label, key in params:
            param_frame = ctk.CTkFrame(parent_frame)
            param_frame.pack(pady=5, padx=10, fill="x")

            ctk.CTkLabel(param_frame, text=label, font=("Arial", 16, "bold"), text_color="white").pack(side="left",
                                                                                                       padx=10)
            ctk.CTkEntry(param_frame, textvariable=self.config_params[key], width=100).pack(side="right", padx=10)

        if include_grid:
            grid_params = [("GRID_X", "grid_x"), ("GRID_Y", "grid_y")]
            for label, key in grid_params:
                grid_frame = ctk.CTkFrame(parent_frame)
                grid_frame.pack(pady=5, padx=10, fill="x")

                ctk.CTkLabel(grid_frame, text=label, font=("Arial", 16, "bold"), text_color="white").pack(side="left",
                                                                                                          padx=10)
                ctk.CTkEntry(grid_frame, textvariable=self.config_params[key], width=100).pack(side="right", padx=10)

        if include_entrada:
            signal_frame = ctk.CTkFrame(parent_frame, fg_color="gray30", corner_radius=10)
            signal_frame.pack(pady=10, padx=10, fill="x")

            ctk.CTkLabel(signal_frame, text="Archivo de Entrada:", font=("Arial", 16, "bold"), text_color="white").pack(
                side="left", padx=10)

            self.signal_entry = ctk.CTkEntry(signal_frame, textvariable=self.config_params["signal_damage"], width=200)
            self.signal_entry.pack(side="left", padx=10)

        output_frame = ctk.CTkFrame(parent_frame, fg_color="gray30", corner_radius=10)
        output_frame.pack(pady=10, padx=10, fill="x")

        ctk.CTkLabel(output_frame, text="Archivo de Salida:", font=("Arial", 16, "bold"), text_color="white").pack(
            side="left", padx=10)

        self.output_entry = ctk.CTkEntry(output_frame, textvariable=self.config_params["output_filename"], width=200)
        self.output_entry.pack(side="left", padx=10)

    def run_process(self, project_id=None):

        if project_id is None:
            project_id = self.project_id

        if project_id not in PROJECT_CONFIG:
            self.log_message(f"❌ Proyecto desconocido: {project_id}")
            return

        self.set_results_buttons_state("disabled")

        self.show_screen("results")
        self.clear_console()

        def process_execution():
            project = PROJECT_CONFIG[project_id]
            project_command = project["command"]

            try:
                kernel_file = f"krnl_{project_command}.bin"

                execute_command(self.ssh, "sudo xmutil unloadapp", self.log_message, wait_time=1)
                execute_command(self.ssh, f"sudo xmutil loadapp {project_command}", self.log_message, wait_time=1)
                if hasattr(self, "signal_entry"):
                    selected_file = self.signal_entry.get().strip() or "data_damage_scaled_processed.csv"
                else:
                    selected_file = "data_damage_scaled_processed.csv"
                # selected_file = self.signal_entry.get().strip() or "data_damage_processed.csv"
                output_file = self.output_entry.get().strip() or "result_output.csv"
                config_file = project["config_file"]

                if project_id == 2:
                    command_full = (
                        f"./host_{project_command} {kernel_file} {selected_file} "
                        f"{self.config_params['x_start'].get()} {self.config_params['x_end'].get()} "
                        f"{self.config_params['y_start'].get()} {self.config_params['y_end'].get()} "
                        f"{self.config_params['iterations'].get()}"
                    )
                else:
                    command_full = (
                        f"./host_{project_command} {kernel_file} {selected_file} "
                        f"{self.config_params['grid_x'].get()} {self.config_params['grid_y'].get()} "
                        f"{self.config_params['x_start'].get()} {self.config_params['x_end'].get()} "
                        f"{self.config_params['y_start'].get()} {self.config_params['y_end'].get()} "
                        f"{self.config_params['iterations'].get()}"
                    )

                execute_command(self.ssh, command_full, self.log_message, wait_time=1)

                fetch_output_files(
                    self.ssh, self.fpga_ip.get(), f"./{output_file}", config_file,
                    self.log_message, project_id, selected_file
                )

                if not os.path.exists(f"./{output_file}") or os.stat(f"./{output_file}").st_size == 0:
                    self.log_message(f"❌ Error: No se encontró el archivo de salida {output_file}.")
                    messagebox.showerror("Error", f"No se encontró el archivo de salida: {output_file}")
                    return

                self.local_path_damage = f"./{output_file}"
                self.local_config_path = config_file

                self.after(1, lambda: self.display_damage_image(f"./{output_file}"))
                self.after(1, lambda: self.display_config_used(config_file))

            except Exception as e:
                self.log_message(f"❌ Error durante la ejecución en FPGA: {e}")

            # Habilitar los botones al finalizar la ejecución
            self.after(1, lambda: self.set_results_buttons_state("normal"))

        threading.Thread(target=process_execution, daemon=True).start()

    def create_results_screen(self):
        print("Creando pantalla de resultados...")

        results_screen = ctk.CTkFrame(self.main_frame, fg_color="gray25", corner_radius=10)
        results_screen.pack(fill="both", expand=True)

        container = ctk.CTkFrame(results_screen, width=1800, height=1000, fg_color="gray30", corner_radius=10)
        container.place(relx=0.5, rely=0.5, anchor="center")
        container.pack_propagate(False)

        self.console_output_results = ctk.CTkTextbox(container, fg_color="black", text_color="lime",
                                                     font=("Courier", 14, "bold"), height=150, width=1700)
        self.console_output_results.pack(fill="x", padx=20, pady=(10, 5))

        title_label = ctk.CTkLabel(container, text="Resultados de Ejecución", font=("Arial", 26, "bold"),
                                   text_color="white")
        title_label.pack(pady=10)

        results_frame = ctk.CTkFrame(container, fg_color="gray40", corner_radius=10)
        results_frame.pack(fill="both", padx=20, pady=10, expand=True)

        results_frame.grid_columnconfigure(0, weight=1)
        results_frame.grid_columnconfigure(1, weight=1)
        results_frame.grid_rowconfigure(0, weight=1)

        self.image_frame = ctk.CTkFrame(results_frame, fg_color="gray30", corner_radius=10, width=600, height=600)
        self.image_frame.grid(row=0, column=0, padx=15, pady=15, sticky="nsew")
        self.image_frame.pack_propagate(False)

        self.config_frame = ctk.CTkFrame(results_frame, fg_color="gray30", corner_radius=10, width=600, height=600)
        self.config_frame.grid(row=0, column=1, padx=15, pady=15, sticky="nsew")
        self.config_frame.pack_propagate(False)

        button_frame = ctk.CTkFrame(container, fg_color="gray30")
        button_frame.pack(side="bottom", fill="x", padx=20, pady=10)

        self.result_buttons = []

        button_back = ctk.CTkButton(button_frame, text="⬅ Volver",
                                    command=lambda: self.show_screen("selection"),
                                    fg_color="#6c757d", font=("Arial", 16, "bold"), width=200)
        button_back.pack(side="left", padx=10)
        self.result_buttons.append(button_back)

        button_popup = ctk.CTkButton(button_frame, text="Mostrar Pop-up",
                                     command=self.show_results_popup_from_button,
                                     fg_color="#007BFF", font=("Arial", 16, "bold"), width=200)
        button_popup.pack(side="left", expand=True)
        self.result_buttons.append(button_popup)

        button_exit = ctk.CTkButton(button_frame, text="Cerrar",
                                    command=self.quit,
                                    fg_color="#dc3545", font=("Arial", 16, "bold"), width=200)
        button_exit.pack(side="right", padx=10)
        self.result_buttons.append(button_exit)

        self.set_results_buttons_state("disabled")

        return results_screen

    def show_results_popup(self, file_path, config_path):

        def create_popup():
            print(f"Creando pop - up de resultados")
            popup = ctk.CTkToplevel(self)
            popup.title("Resultados de Ejecución")
            popup.geometry("1200x800")

            popup_frame = ctk.CTkFrame(popup, fg_color="gray30", corner_radius=10)
            popup_frame.pack(fill="both", expand=True, padx=10, pady=10)

            frame_popup_left = ctk.CTkFrame(popup_frame, fg_color="gray40", corner_radius=10)
            frame_popup_left.pack(side="left", fill="both", expand=True, padx=10, pady=10)

            frame_popup_right = ctk.CTkFrame(popup_frame, fg_color="gray40", corner_radius=10)
            frame_popup_right.pack(side="right", fill="both", expand=True, padx=10, pady=10)

            self.after(100, lambda: self.display_damage_image(file_path, frame_popup_left))
            self.after(100, lambda: self.display_config_used(config_path, frame_popup_right))

        threading.Thread(target=create_popup, daemon=True).start()

    def show_results_popup_from_button(self):
        if hasattr(self, "local_path_damage") and hasattr(self, "local_config_path"):
            self.show_results_popup(self.local_path_damage, self.local_config_path)
        else:
            self.log_message("❌ No hay resultados disponibles para mostrar.")

    def display_damage_image(self, file_path, parent_frame=None):
        if parent_frame is None:
            parent_frame = self.image_frame

        for widget in parent_frame.winfo_children():
            widget.destroy()

        try:
            data = np.loadtxt(file_path, delimiter=",")

            extent = [self.config_params["x_start"].get(),
                      self.config_params["x_end"].get(),
                      self.config_params["y_start"].get(),
                      self.config_params["y_end"].get()]

            fig, ax = plt.subplots(figsize=(6, 5))
            img = ax.imshow(data, cmap="jet", aspect="auto", origin="lower", extent=extent)

            ax.set_title("Índice de Daño")
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.ticklabel_format(style='plain', axis='y')

            fig.colorbar(img, ax=ax)

            # 🔹 Etiqueta para mostrar coordenadas y distancia
            coord_label = ctk.CTkLabel(parent_frame, text="", font=("Arial", 12), text_color="white")
            coord_label.pack()

            points = []  # Para almacenar los puntos (máx. 2)
            markers = []  # Para los puntos dibujados
            line = [None]  # Línea entre puntos

            def on_click(event):
                if event.inaxes != ax:
                    return

                x, y = event.xdata, event.ydata
                points.append((x, y))

                # Dibujar punto rojo pequeño
                marker = ax.plot(x, y, 'ro', markersize=4)[0]
                markers.append(marker)

                # Si hay dos puntos, dibujar línea y calcular distancia
                if len(points) == 2:
                    (x1, y1), (x2, y2) = points

                    # Dibujar línea
                    if line[0]:
                        line[0].remove()
                    line[0] = ax.plot([x1, x2], [y1, y2], 'w--', linewidth=1.5)[0]

                    # Calcular distancia euclidiana
                    distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
                    coord_label.configure(
                        text=f"P1: ({x1:.3f}, {y1:.3f}) m   P2: ({x2:.3f}, {y2:.3f}) m   Distancia: {distance:.3f} m"
                    )

                    fig.canvas.draw()

                    # Reset para permitir medir otra vez
                    points.clear()
                    for m in markers:
                        m.remove()
                    markers.clear()

                fig.canvas.draw()

            fig.canvas.mpl_connect("button_press_event", on_click)

            canvas_widget = FigureCanvasTkAgg(fig, master=parent_frame)
            canvas_widget.draw()
            canvas_widget.get_tk_widget().pack(fill="both", expand=True)

        except Exception as e:
            self.log_message(f"❌ Error al cargar la imagen del índice de daño: {e}")


    def display_config_used(self, config_path, parent_frame=None):

        if parent_frame is None:
            parent_frame = self.config_frame

        for widget in parent_frame.winfo_children():
            widget.destroy()

        ctk.CTkLabel(parent_frame, text="Configuración Utilizada", font=("Arial", 18, "bold"),
                     text_color="white").pack(pady=5)

        try:
            with open(config_path, "r") as file:
                config_lines = file.readlines()

            for line in config_lines:
                if "=" in line:
                    param, value = line.strip().split("=")
                    config_container = ctk.CTkFrame(parent_frame, fg_color="transparent")
                    config_container.pack(fill="x", padx=5, pady=2)

                    ctk.CTkLabel(config_container, text=param, font=("Arial", 16, "bold"), text_color="white").pack(
                        side="left", padx=10)
                    ctk.CTkLabel(config_container, text=value, font=("Arial", 16), text_color="white").pack(
                        side="right", padx=10)

        except Exception as e:
            self.log_message(f"❌ Error al cargar la configuración: {e}")

    def load_file(self, file_var):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")])
        if file_path:
            file_var.set(file_path)
            if file_var == self.fpga_file_path:
                self.fpga_file_label.configure(text=f"{os.path.basename(file_path)}")
            else:
                self.matlab_file_label.configure(text=f"{os.path.basename(file_path)}")

    def load_file_conf(self, file_var):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")])
        if file_path:
            file_var.set(file_path)
            self.log_message(f"Archivo seleccionado: {os.path.basename(file_path)}")

    def load_csv_as_array(self, file_path):
        if not os.path.exists(file_path):
            messagebox.showerror("Error", f"Archivo no encontrado: {file_path}")
            return None

        try:
            data = np.loadtxt(file_path, delimiter=",")
            return data
        except Exception as e:
            messagebox.showerror("Error", f"❌ Error al cargar el archivo {file_path}: {e}")
            return None

    def calculate_error_metrics(self, difference_matrix):
        print(f"Calculando metricas de error")
        if difference_matrix is None or difference_matrix.size == 0:
            messagebox.showerror("Error", "No se puede calcular métricas, la matriz está vacía.")
            return

        self.max_error = np.max(difference_matrix)
        self.min_error = np.min(difference_matrix)
        self.mean_error = np.mean(difference_matrix)
        self.std_error = np.std(difference_matrix)

        metrics = {
            "Error Máximo": self.max_error,
            "Error Mínimo": self.min_error,
            "Error Medio": self.mean_error,
            "Desviación Típica": self.std_error
        }

        ctk.CTkLabel(self.metrics_frame, text="Métricas de Error", font=("Arial", 18, "bold"),
                     text_color="white").pack(pady=10)

        for label, value in metrics.items():
            metric_container = ctk.CTkFrame(self.metrics_frame, fg_color="gray30", corner_radius=5)
            metric_container.pack(fill="x", padx=5, pady=5)

            ctk.CTkLabel(metric_container, text=label, font=("Arial", 16, "bold"), text_color="white").pack(side="left",
                                                                                                            padx=10)
            ctk.CTkLabel(metric_container, text=f"{value:.12f}", font=("Arial", 16), text_color="white").pack(
                side="right", padx=10)

        print(f"Mostrando metricas de error")

    def clear_comparison_screen(self):
        self.clear_graph_frame()
        self.clear_metrics_frame()

    def clear_graph_frame(self):
        for widget in self.graph_frame.winfo_children():
            widget.destroy()

    def clear_metrics_frame(self):
        for widget in self.metrics_frame.winfo_children():
            widget.destroy()

    def display_comparison_results(self, parent_frame):

        def draw_plot():
            fig, ax = plt.subplots(figsize=(6, 5))

            extent = [self.config_params["x_start"].get(),
                      self.config_params["x_end"].get(),
                      self.config_params["y_start"].get(),
                      self.config_params["y_end"].get()]

            img = ax.imshow(self.difference_matrix, cmap="jet", aspect="auto",
                            origin="lower", extent=extent)

            filename_fpga = os.path.basename(self.fpga_file_path.get())
            filename_matlab = os.path.basename(self.matlab_file_path.get())
            ax.set_title(f"Diferencia: {filename_fpga} vs {filename_matlab}")
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.ticklabel_format(style='plain', axis='y')

            fig.colorbar(img, ax=ax)

            # Label para mostrar coordenadas y radio
            result_label = ctk.CTkLabel(parent_frame, text="", font=("Arial", 18, "bold"), text_color="white")
            result_label.pack()

            points = []
            markers = []
            circle_artist = [None]

            def on_click(event):
                if event.inaxes != ax:
                    return

                x, y = event.xdata, event.ydata
                points.append((x, y))
                marker = ax.plot(x, y, 'ro', markersize=4)[0]
                markers.append(marker)

                if len(points) >= 4:
                    # Ajuste de círculo por mínimos cuadrados
                    x_vals = np.array([p[0] for p in points])
                    y_vals = np.array([p[1] for p in points])

                    A = np.column_stack((2 * x_vals, 2 * y_vals, np.ones_like(x_vals)))
                    b = x_vals ** 2 + y_vals ** 2

                    try:
                        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        cx, cy, c = sol
                        r = np.sqrt(cx ** 2 + cy ** 2 + c)
                    except np.linalg.LinAlgError:
                        result_label.configure(text="No se pudo ajustar el círculo.")
                        return

                    # Dibujar círculo
                    if circle_artist[0]:
                        circle_artist[0].remove()

                    from matplotlib.patches import Circle
                    circle = Circle((cx, cy), r, fill=False, edgecolor='red', linestyle='--', linewidth=2.5)
                    circle_artist[0] = ax.add_patch(circle)

                    result_label.configure(
                        text=f"Centro del daño: ({cx:.6f}, {cy:.6f}) m "
                    )

                    # Limpiar puntos
                    for m in markers:
                        m.remove()
                    markers.clear()
                    points.clear()

                    fig.canvas.draw()

            fig.canvas.mpl_connect("button_press_event", on_click)

            canvas_widget = FigureCanvasTkAgg(fig, master=parent_frame)
            canvas_widget.draw()
            canvas_widget.get_tk_widget().pack(fill="both", expand=True)

        self.after(100, draw_plot)

    def display_metrics_on_screen(self, metrics_frame):

        metrics_container = ctk.CTkFrame(metrics_frame, fg_color="gray30", corner_radius=10)
        metrics_container.pack(fill="both", expand=True, padx=10, pady=10)

        ctk.CTkLabel(metrics_container, text="Métricas de Comparación", font=("Arial", 18, "bold"),
                     text_color="white").pack(pady=10)

        for label, value in self.error_metrics.items():
            metric_frame = ctk.CTkFrame(metrics_container, fg_color="gray40", corner_radius=5)
            metric_frame.pack(fill="x", padx=5, pady=5)

            ctk.CTkLabel(metric_frame, text=label, font=("Arial", 16, "bold"), text_color="white").pack(side="left",
                                                                                                        padx=10)
            ctk.CTkLabel(metric_frame, text=f"{value:.4f} ", font=("Arial", 16), text_color="white").pack(side="right",
                                                                                                           padx=10)

        print("✅ Métricas mostradas en la pantalla de comparación en porcentaje.")

    def show_comparison_popup(self):
        self.popup = ctk.CTkToplevel(self)
        self.popup.title("Resultados de Comparación")
        self.popup.geometry("1200x800")

        popup_frame = ctk.CTkFrame(self.popup, fg_color="gray30", corner_radius=10)
        popup_frame.pack(fill="both", expand=True, padx=10, pady=10)

        frame_popup_left = ctk.CTkFrame(popup_frame, fg_color="gray40", corner_radius=10)
        frame_popup_left.pack(side="left", fill="both", expand=True, padx=10, pady=10)

        frame_popup_right = ctk.CTkFrame(popup_frame, fg_color="gray40", corner_radius=10)
        frame_popup_right.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        self.display_comparison_results(frame_popup_left)
        self.display_metrics_on_screen(frame_popup_right)

        print("✅ Pop-up de comparación creado correctamente.")

    def create_settings_screen(self):

        print("Creando pantalla de configuración de la aplicación")

        settings_screen = ctk.CTkFrame(self.main_frame, fg_color="gray25", corner_radius=10)
        settings_screen.pack(fill="both", expand=True)

        container = ctk.CTkFrame(settings_screen, width=1800, height=900, fg_color="gray30", corner_radius=10)
        container.place(relx=0.5, rely=0.5, anchor="center")
        container.pack_propagate(False)

        console_frame = ctk.CTkFrame(container, fg_color="gray20", corner_radius=10)
        console_frame.pack(fill="x", padx=20, pady=(10, 5))

        self.console_output_settings = ctk.CTkTextbox(console_frame, fg_color="black", text_color="lime",
                                                      font=("Courier", 14, "bold"), height=200, width=1700)
        self.console_output_settings.pack(fill="x", padx=10, pady=5)

        title_label = ctk.CTkLabel(container, text="Configuración de la Aplicación", font=("Arial", 26, "bold"),
                                   text_color="white")
        title_label.pack(pady=15)

        file_frame = ctk.CTkFrame(container, fg_color="gray40", corner_radius=10)
        file_frame.pack(fill="x", padx=20, pady=10)

        self.file_path_var = tk.StringVar(value="Ningún archivo seleccionado")

        load_button = ctk.CTkButton(file_frame, text="Cargar Archivo",
                                    command=lambda: self.load_file_conf(self.file_path_var),
                                    fg_color="#007BFF", width=150)
        load_button.pack(side="left", padx=10, pady=5)

        file_label = ctk.CTkLabel(file_frame, textvariable=self.file_path_var, font=("Arial", 14), text_color="white")
        file_label.pack(side="left", padx=10, pady=5)

        send_button = ctk.CTkButton(file_frame, text="Enviar Archivo",
                                    command=self.send_file,
                                    fg_color="#28a745", width=200)
        send_button.pack(side="left", padx=10, pady=5)

        download_frame = ctk.CTkFrame(container, fg_color="gray40", corner_radius=10)
        download_frame.pack(fill="x", padx=20, pady=10)

        ctk.CTkLabel(download_frame, text="Archivo a descargar:", font=("Arial", 16, "bold"),
                     text_color="white").pack(side="left", padx=10)

        self.download_file_var = tk.StringVar(value="output_data.csv")  # Archivo por defecto
        self.download_entry = ctk.CTkEntry(download_frame, textvariable=self.download_file_var, width=300)
        self.download_entry.pack(side="left", padx=10)

        download_button = ctk.CTkButton(download_frame, text="Descargar Archivo",
                                        command=self.download_file,
                                        fg_color="#FFC107", width=200)
        download_button.pack(side="left", padx=10, pady=5)

        command_frame = ctk.CTkFrame(container, fg_color="gray40", corner_radius=10)
        command_frame.pack(fill="x", padx=20, pady=10)

        self.command_var = tk.StringVar()

        send_command_button = ctk.CTkButton(command_frame, text="Enviar Comando",
                                            command=self.send_command,
                                            fg_color="#007BFF", width=150)
        send_command_button.pack(side="left", padx=10, pady=5)

        command_entry = ctk.CTkEntry(command_frame, textvariable=self.command_var, width=400,
                                     font=("Arial", 14), placeholder_text="Escriba un comando...")
        command_entry.pack(side="left", padx=10, pady=5)

        back_frame = ctk.CTkFrame(container, fg_color="gray30")
        back_frame.pack(side="bottom", fill="x", pady=10)

        ctk.CTkButton(back_frame, text="⬅ Volver",
                      command=lambda: self.show_screen("intermediate"),
                      fg_color="#6c757d", width=150).pack(side="left", padx=5)

        ctk.CTkButton(back_frame, text="Cerrar",
                      command=self.quit,
                      fg_color="#dc3545", width=150).pack(side="right", padx=5)

        return settings_screen

    def send_command(self):
        command = self.command_var.get()
        if command and self.ssh:
            response = execute_command(self.ssh, command, self.log_message)
            self.log_message(f"🔹 Respuesta: {response}")
        else:
            self.log_message("⚠ No hay conexión SSH activa o el comando está vacío.")

    def send_file(self):
        if self.file_path_var.get() == "Ningún archivo seleccionado":
            self.log_message("⚠ No se ha seleccionado ningún archivo para enviar.")
            return

        remote_path = "/home/petalinux/" + self.file_path_var.get().split("/")[-1]

        if self.ssh:
            send_file_via_ssh(self.ssh, self.file_path_var.get(), remote_path, self.log_message)
        else:
            self.log_message("❌ No hay conexión SSH activa.")

    def show_comparison_popup_from_button(self):
        fpga_file = self.fpga_file_path.get()
        matlab_file = self.matlab_file_path.get()

        if not fpga_file or not matlab_file:
            messagebox.showerror("Error", "Debe seleccionar ambos archivos antes de mostrar el pop-up.")
        else:
            self.show_comparison_popup()

    def compare_images(self):

        print("Iniciando comparación de imágenes...")
        fpga_file = self.fpga_file_path.get()
        matlab_file = self.matlab_file_path.get()

        if not fpga_file or not matlab_file:
            messagebox.showerror("Error", "Debe seleccionar ambos archivos antes de comparar.")
            return

        data_fpga = self.load_csv_as_array(fpga_file)
        data_matlab = self.load_csv_as_array(matlab_file)

        if data_fpga is None or data_matlab is None:
            return

        if data_fpga.shape != data_matlab.shape:
            messagebox.showerror("Error", "Los archivos tienen dimensiones diferentes y no pueden compararse.")
            return

        max_matlab = np.max(data_matlab)
        max_fpga = np.max(data_fpga)
        min_matlab = np.min(data_matlab)
        min_fpga = np.min(data_fpga)
        print(f"max_matlab = {max_matlab}")
        print(f"max_fpga = {max_fpga}")
        print(f"min_matlab = {min_matlab}")
        print(f"min_fpga = {min_fpga}")

        if max_matlab == 0:
            messagebox.showerror("Error", "Uno de los archivos tiene máximo absoluto cero.")
            return

        data_fpga_norm = data_fpga / max_matlab
        data_matlab_norm = data_matlab / max_matlab

        error_absoluto = (np.abs(data_matlab_norm - data_fpga_norm))
        max_abs = np.max(error_absoluto)
        med_abs = np.mean(error_absoluto)
        std_abs = np.std(error_absoluto)

        print(f"max_abs = {max_abs}")
        print(f"med_abs = {med_abs}")

        self.error_metrics = {
            "Error Máximo ": max_abs,
            "Error Medio ": med_abs,
            "Desviación Típica ": std_abs
        }

        grafica = error_absoluto
        self.difference_matrix = grafica

        self.clear_comparison_screen()
        self.display_comparison_results(self.graph_frame)
        self.display_metrics_on_screen(self.metrics_frame)

        print("✅ Comparación completada. Errores normalizados y mostrados como porcentaje.")

    def clear_console(self):
        for console in [getattr(self, "console_output_settings", None), getattr(self, "console_output_results", None)]:
            if console:  # Solo limpiar si la consola existe
                console.configure(state="normal")  # Habilitar edición
                console.delete("1.0", "end")  # Borrar todo el contenido
                console.configure(state="disabled")  # Bloquear edición nuevamente

    def download_file(self):
        if not self.ssh:
            self.log_message("No hay conexión SSH activa.")
            return

        remote_file = f"/home/petalinux/{self.download_file_var.get().strip()}"  # Ruta en la FPGA
        local_file = f"./{self.download_file_var.get().strip()}"  # Se guarda en el directorio actual

        if not remote_file:
            self.log_message("No se ha especificado un archivo para descargar.")
            return

        self.log_message(f"Descargando archivo: {remote_file}")

        try:
            download_file_from_ssh(self.ssh, remote_file, local_file, self.log_message)
            self.log_message(f"Archivo descargado correctamente en: {local_file}")
        except Exception as e:
            self.log_message(f"Error al descargar archivo: {e}")

    def set_results_buttons_state(self, state):
        for button in getattr(self, "result_buttons", []):
            button.configure(state=state)


# 🔹 Ejecutar la aplicación correctamente
if __name__ == "__main__":
    print("Iniciando aplicación...")
    app = FPGAApp()
    app.mainloop()
