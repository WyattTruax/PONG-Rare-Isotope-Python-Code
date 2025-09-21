import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from scipy.special import spherical_jn
import tkinter as tk

# -- Your isotope data and functions (same as before) --

# Isotope data
isotopes = [
    ("C-12", 6, 12, 2.469),
    ("Fe-56", 26, 56, 3.7503333333333335),
    ("Pb-208", 82, 208, 5.501633333333333),
    ("Li-6", 3, 6, 2.556666666666667), # Lithium-6 is supposed to be Gaussian
    ("O-16", 8, 16, 2.7283333333333333),
    ("Ca-40", 20, 40, 3.4703333333333333),
    ("Ca-48", 20, 48, 3.4605),
    ("H-1", 1, 1, 0.85),
    ("H-2", 1, 2, 2.1055),
    ("H-3", 1, 3, 1.72),
    ("He-3", 2, 3, 1.889),
    ("He-4", 2, 4, 1.6806666666666666666666666666),
    ("Li-6", 3, 6, 2.5566666666666666666),
    ("Li-9", 3, 7, 2.4),
    ("Be-9", 4, 9, 2.5095),
    ("Be-10", 4, 10, 2.45),
    ("Be-11", 4, 11, 2.395),
    ("C-12", 6, 12, 2.46775),
    ("C-13", 6, 13, 2.440),
    ("C-14",6, 14, 2.56),
    ("N-14", 7, 14, 2.548),
    ("N-15", 7, 15, 2.65366666666666666666),
    ("O-17", 8, 17, 2.662),
    ("O-18", 8, 18, 2.727),
    ("F-19", 9, 19, 2.900),
    ("Ne-20", 10, 20 , 3.012),
    ("Ne-22", 10, 22, 2.969),
    ("Na-23", 11, 23, 2.94),
    ("Mg-24", 12, 24, 3.046666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666),
    ("Mg-25", 12, 25, 2.0376666666666666666666666666666666666666666666666666666666),
    ("Mg-26", 12, 26, 3.06),
    ("Al-27", 13, 27, 3.0653333333333333333333),
    ("Si-28", 14, 28,  3.112),
    ("Si-29", 14, 29,  3.153),
    ("Si-30", 14, 30, 3.176),
    ("P-31", 15, 31, 3.188),
    ("S-32", 16, 32, 3.24),
    ("S-34", 16, 34, 3.281),
    ("Cl-35", 17, 35, 3.388),
    ("Cl-37", 17, 37, 3.384),
    ("Ar-36", 18, 36, 3.327),
    ("Ar-40", 18, 40,3.432),
    ("K-39", 19, 29, 3.404),
    ("Ca-40", 20, 40, 3.47033333333333333333333333333333),
    ("Ca-48", 20, 48, 3.470),
    ("Ti-48", 22, 48, 3.655),
    ("Ti-50", 22, 50, 3.573),
    ("V-51", 23, 51, 3.6),
    ("Cr-50", 24, 50, 3.662),
    ("Cr-52", 24, 52, 3.643),
    ("Cr-53", 24, 53, 3.726),
    ("Cr-54", 24, 54, 3.776),
    ("Mn-55", 25, 55, 3.68),
    ("Fe-54", 26, 54, 3.680),
    ("Fe-56", 26, 56, 3.77),
    ("Co-59", 27, 59, 3.775),
    ("Ni-58", 28, 58, 3.772),
]

Q_values = [np.linspace(0, 5, 1000)]
r_max = 30  # fm
r = np.linspace(1e-5, r_max, 5000)
a_charge = 0.53  # fm

def R0_from_rms(r_rms, a):
    term = (5/3)*r_rms**2
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
    Fq = np.array(Fq)
    Fq /= F0
    return Fq

# --- Plot setup ---
fig, ax = plt.subplots(figsize=(8,6))
plt.subplots_adjust(bottom=0.3)

current_isotope_index = 0

def update_plot():
    ax.clear()
    name, Z, A, r_rms = isotopes[current_isotope_index]
    R0 = R0_from_rms(r_rms, a_charge)
    rho0 = compute_rho0(r, R0, a_charge, Z)
    rho = fermi_density(r, R0, a_charge, rho0)
    F_q = compute_form_factor(Q_values[0], rho, r)
    ax.plot(Q_values[0], np.abs(F_q), label=f"{name}")
    ax.semilogy()
    ax.set_title(f"{name} | RMS Radius = {r_rms:.3f} fm", fontsize=15)
    ax.set_xlabel("Momentum Transfer $q$ [fm$^{-1}$]", fontsize=15)
    ax.set_ylabel("Form Factor Magnitude $|F(q)|$", fontsize=15)
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(fontsize=25)
    fig.canvas.draw_idle()

# --- Tkinter popup for isotope selection ---
def select_isotope_dialog():
    root = tk.Tk()
    root.title("Select Isotope")
    
    tk.Label(root, text="Search:").pack(pady=(10,0))
    search_var = tk.StringVar()
    search_entry = tk.Entry(root, textvariable=search_var)
    search_entry.pack(pady=(0,10), padx=10, fill=tk.X)
    
    listbox = tk.Listbox(root, height=15, width=40)
    listbox.pack(padx=10, pady=(0,10))
    
    isotope_names = [iso[0] for iso in isotopes]
    for name in isotope_names:
        listbox.insert(tk.END, name)
    
    def on_search(*args):
        search_term = search_var.get().lower()
        listbox.delete(0, tk.END)
        for name in isotope_names:
            if search_term in name.lower():
                listbox.insert(tk.END, name)
    
    search_var.trace("w", on_search)
    
    def on_ok():
        try:
            selection = listbox.get(listbox.curselection())
            root.selected = selection
            root.destroy()
        except tk.TclError:
            # No selection, do nothing
            pass
    
    ok_button = tk.Button(root, text="OK", command=on_ok)
    ok_button.pack(pady=(0,10))
    
    root.selected = None
    # Center the window on screen (optional)
    root.eval('tk::PlaceWindow . center')
    root.mainloop()
    
    return root.selected

# --- Matplotlib button to open the Tkinter popup ---
button_ax = plt.axes([0.4, 0.05, 0.2, 0.075])
button = Button(button_ax, 'Select Isotope', color='lightgreen', hovercolor='green')

def on_button_clicked(event):
    global current_isotope_index
    selected_name = select_isotope_dialog()
    if selected_name is not None:
        # Find index and update
        names = [iso[0] for iso in isotopes]
        if selected_name in names:
            current_isotope_index = names.index(selected_name)
            update_plot()

button.on_clicked(on_button_clicked)

# Initial plot
update_plot()

plt.show()
