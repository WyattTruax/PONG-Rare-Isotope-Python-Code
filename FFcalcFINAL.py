import numpy as np
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.special import spherical_jn
from tkinter import font
import isotopes

# =========================
# Physics Functions
# =========================
Q_values = np.linspace(1e-5, 5, 1000)
r = np.linspace(1e-5, 30, 5000)
a_charge = 0.53

def R0_from_rms(r_rms):
    term = (5/3) * r_rms**2
    return np.sqrt(term)

def fermi_density_shape(r, R0, a):
    return 1.0 / (1 + np.exp((r - R0)/a))

def compute_rho0(r, R0, a, Z):
    shape = fermi_density_shape(r, R0, a)
    delta_r = r[1] - r[0]
    integral = np.sum(4 * np.pi * r**2 * shape) * delta_r
    return Z / integral

def fermi_density(r, R0, a, rho0):
    return rho0 / (1 + np.exp((r - R0)/a))

def compute_form_factor(q_vals, rho_r, r):
    delta_r = r[1] - r[0]
    F0 = 4 * np.pi * np.sum(r**2 * rho_r) * delta_r
    Fq = []

    for q in q_vals:
        j0 = spherical_jn(0, q * r)
        integrand = r**2 * j0 * rho_r
        fq = 4 * np.pi * np.sum(integrand) * delta_r
        Fq.append(fq)

    Fq = np.array(Fq) / F0
    return Fq

# =========================
# GUI Application
# =========================
class IsotopeViewer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Nuclear Form Factor Viewer")
        self.geometry("1100x600")

        # --- High-DPI scaling ---
        dpi = self.winfo_fpixels('1i')
        scale = dpi / 72.0
        self.tk.call('tk', 'scaling', scale)

        # --- Global font control ---
        default_font = font.nametofont("TkDefaultFont")
        default_font.configure(size=11)
        
        self._build_layout()
        self._populate_list()
        self._setup_plot()

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_close(self):
        plt.close("all")
        self.destroy()
        self.quit()

    def _build_layout(self):
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        # Left panel
        left = ttk.Frame(self, padding=10)
        left.grid(row=0, column=0, sticky="ns")

        ttk.Label(left, text="Search").pack(anchor="w")
        self.search_var = tk.StringVar()
        self.search_var.trace_add("write", self._filter_list)

        search_entry = ttk.Entry(left, textvariable=self.search_var)
        search_entry.pack(fill="x", pady=5)

        self.listbox = tk.Listbox(left, height=25, width=30, font=("Arial", 16))
        self.listbox.pack(fill="y", expand=True)
        self.listbox.bind("<<ListboxSelect>>", self._on_select)

        # Right panel
        right = ttk.Frame(self, padding=10)
        right.grid(row=0, column=1, sticky="nsew")

        self.fig, self.ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def _populate_list(self):
        self.filtered = [iso[0] for iso in isotopes.isotopes]
        self._refresh_listbox()

    def _refresh_listbox(self):
        self.listbox.delete(0, tk.END)
        for name in self.filtered:
            self.listbox.insert(tk.END, name)

    def _filter_list(self, *_):
        term = self.search_var.get().lower()
        self.filtered = [iso[0] for iso in isotopes.isotopes if term in iso[0].lower()]
        self._refresh_listbox()

    def _on_select(self, event):
        if not self.listbox.curselection():
            return
        name = self.listbox.get(self.listbox.curselection())
        index = [iso[0] for iso in isotopes.isotopes].index(name)
        self.update_plot(index)

    def _setup_plot(self):
        self.ax.set_xlabel("Momentum Transfer q [fm⁻¹]")
        self.ax.set_ylabel("|F(q)|")
        #self.ax.set_ylim(1e-4, 1.05)
        self.ax.set_yscale("log")
        self.ax.grid(True, linestyle="--", alpha=0.6)
        # add some padding so title/labels are not cropped inside the Tk container
        self.ax.xaxis.labelpad = 8
        self.ax.yaxis.labelpad = 8
        self.canvas.draw_idle()

    def update_plot(self, index):
        name, Z, A, r_rms = isotopes.isotopes[index]

        R0 = R0_from_rms(r_rms)
        rho0 = compute_rho0(r, R0, a_charge, Z)
        rho = fermi_density(r, R0, a_charge, rho0)
        Fq = compute_form_factor(Q_values, rho, r)

        self.ax.clear()
        self.ax.plot(Q_values, np.abs(Fq), label=name)
        self.ax.set_title(f"{name} | RMS = {r_rms:.3f} fm", pad=14)
        self.ax.set_xlabel("Momentum Transfer q [fm⁻¹]")
        self.ax.set_ylabel("|F(q)|")
        #self.ax.set_ylim(1e-4, 1.05)
        self.ax.set_yscale("log")
        self.ax.grid(True, linestyle="--", alpha=0.6)
        self.ax.legend()
        # ensure layout is adjusted after updating the plot
        try:
            self.fig.canvas.draw_idle()
        except Exception:
            pass
        self.canvas.draw_idle()

# =========================
# Run App
# =========================
if __name__ == "__main__":
    app = IsotopeViewer()
    app.mainloop()
    
