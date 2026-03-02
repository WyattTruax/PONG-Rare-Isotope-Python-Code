# write code summary
# include references here

import numpy as np
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.special import spherical_jn
from tkinter import font
import isotopes
import AIEA_isotopes

# **************************
# global variables
# **************************

Q_values = np.linspace(1e-5, 8, 1000)
rand_radii_global = np.linspace(1e-5, 12, 5000)
a_charge = 0.53

# **************************
# new calculations after 50% error fiasco
# **************************

def calc_gaussian_density(r_rms, Z, rand_radii):
    a = np.sqrt(2/3) * r_rms
    shape = np.exp(-(rand_radii**2) / a**2)
    integral = 4 * np.pi * np.trapezoid(rand_radii**2 * shape, x=rand_radii)
    rho0 = Z / integral
    rho_r = rho0 * np.exp(-(rand_radii**2) / a**2)
    return rho_r

def calc_woods_saxon_density(r_rms, Z, rand_radii):
    term = (5/3) * r_rms**2 - (7/3) * (np.pi**2) * a_charge**2
    if term <= 0:
        # fallback for light nuclei
        R0 = np.sqrt((5/3) * r_rms**2)
    else:
        R0 = np.sqrt(term)
    shape = 1 / (1 + np.exp((rand_radii - R0) / a_charge))
    integral = 4 * np.pi * np.trapezoid(rand_radii**2 * shape, x=rand_radii)
    rho0 = Z / integral
    rho_r = rho0 / (1 + np.exp((rand_radii - R0) / a_charge))
    return rho_r

def normalization_check(rho_R, rand_radii): # checking function, doesn't do anythin needed mathematically
    delta_r = rand_radii[1] - rand_radii[0]
    return 4*np.pi * np.sum(rand_radii**2 * rho_R) * delta_r

# this is a long equation so its broken up into coefficient and integral parts
def calc_pwba_ff(rho_R, Z, rand_radii): #pwba = 4pi/Ze * integral_0^inf(r^2 * rho(R) * sin(qr)/qr) dr
    integrals = []
    for q in Q_values:
        j0 = spherical_jn(0, q * rand_radii)
        integrals.append(np.trapezoid(rand_radii**2 * rho_R * j0, x=rand_radii))

    Fq = (4 * np.pi / Z) * np.array(integrals)
    return Fq

# percent error calculation comparing calculated rms charge radius data to AIEA published data
def per_err(r_rms, published_rms):
    return ((r_rms - published_rms) / published_rms) * 100

# caclculates rms charge radius used above (found at middle top of Fq graphs)
def calc_rms_cr(Fq, Q_values):
    return np.sqrt(-6 * np.gradient(Fq, Q_values**2)[0])

# **************************
# GUI application
# **************************

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
        name, Z, M, r_rms = isotopes.isotopes[index]
        name, AIEA_rms = AIEA_isotopes.AIEA_iso[index]

        # new caclculations
        if M < 14:
            rho_r = calc_gaussian_density(r_rms, Z, rand_radii_global)
        else:
            rho_r = calc_woods_saxon_density(r_rms, Z, rand_radii_global)
        Fq = calc_pwba_ff(rho_r, Z, rand_radii_global)
        
        # here is where we need to calc r_rms using r = sqrt[-6(dF/dq2) at q=0]
        # this is what should be compared to AIEA_rms, not the r_rms which was givne from Paul
        rms_cr = calc_rms_cr(Fq, Q_values)

        # calculates our percent error compared to AIEA data
        # in the end, FRIb data --> form factor --> rms charge radius (rms_cr) --> compare to AIEA rms_cr
        pct_err_AIEA = per_err(rms_cr, AIEA_rms) # within 5% for all isotopes, good for the differences between FRIB and all published data
        pct_err_FRIB = per_err(rms_cr, r_rms) # "exactly" 0 for every isotope, good as the amth should just undo itself

        # caclulates the proton count for the isotope (another check line)
        z_check = normalization_check(rho_r, rand_radii_global)

        self.ax.clear()
        self.ax.plot(Q_values, np.abs(Fq), label=name)
        self.ax.set_title(f"{name} | RMS = {r_rms:.3f} fm | Pct Err AIEA = {pct_err_AIEA:.3f}") # \nPct Err FRIB = {pct_err_FRIB:.3f}"f" | Z(norm check) = {z_check:.3f}")
        self.ax.set_xlabel("Momentum Transfer q [fm⁻¹]")
        self.ax.set_ylabel("|F(q)|")
        self.ax.set_yscale("log")
        self.ax.grid(True, linestyle="--", alpha=0.6)
        self.ax.legend()
        # ensure layout is adjusted after updating the plot
        try:
            self.fig.canvas.draw_idle()
        except Exception:
            pass
        self.canvas.draw_idle()

# **************************
# runs app
# **************************
if __name__ == "__main__":
    app = IsotopeViewer()
    app.mainloop()
