import numpy as np
import matplotlib.pyplot as plt
from scipy.special import spherical_jn
import tkinter as tk
from tkinter import ttk

# -- Full isotope data --
isotopes = [
    ("Hydrogen-1", 1, 1, 0.85),
    ("Hydrogen-2", 1, 2, 2.1055),
    ("Hydrogen-3", 1, 3, 1.72),
    ("Helium-3", 2, 3, 1.889),
    ("Helium-4", 2, 4, 1.6806666666666667),
    ("Lithium-6", 3, 6, 2.556666666666667),
    ("Lithium-7", 3, 7, 2.4),  # Lithium-9 typo fixed to Lithium-7
    ("Beryllium-9", 4, 9, 2.5095),
    ("Beryllium-10", 4, 10, 2.45),
    ("Beryllium-11", 4, 11, 2.395),
    ("Carbon-12", 6, 12, 2.46775),
    ("Carbon-13", 6, 13, 2.44),
    ("Carbon-14", 6, 14, 2.56),
    ("Nitrogen-14", 7, 14, 2.548),
    ("Nitrogen-15", 7, 15, 2.6536666666666667),
    ("Oxygen-16", 8, 16, 2.7283333333333333),
    ("Oxygen-17", 8, 17, 2.662),
    ("Oxygen-18", 8, 18, 2.727),
    ("Fluorine-19", 9, 19, 2.9),
    ("Neon-20", 10, 20, 3.012),
    ("Neon-22", 10, 22, 2.969),
    ("Sodium-23", 11, 23, 2.94),
    ("Magnesium-24", 12, 24, 3.0466666666666666),
    ("Magnesium-25", 12, 25, 2.0376666666666667),
    ("Magnesium-26", 12, 26, 3.06),
    ("Aluminum-27", 13, 27, 3.0653333333333333),
    ("Silicon-28", 14, 28, 3.112),
    ("Silicon-29", 14, 29, 3.153),
    ("Silicon-30", 14, 30, 3.176),
    ("Phosphorus-31", 15, 31, 3.188),
    ("Sulfur-32", 16, 32, 3.24),
    ("Sulfur-34", 16, 34, 3.281),
    ("Chlorine-35", 17, 35, 3.388),
    ("Chlorine-37", 17, 37, 3.384),
    ("Argon-36", 18, 36, 3.327),
    ("Argon-40", 18, 40, 3.432),
    ("Potassium-39", 19, 39, 3.404),  # Fixed A from 29 to 39
    ("Calcium-40", 20, 40, 3.4703333333333333),
    ("Calcium-48", 20, 48, 3.47),
    ("Titanium-48", 22, 48, 3.655),
    ("Titanium-50", 22, 50, 3.573),
    ("Vanadium-51", 23, 51, 3.6),
    ("Chromium-50", 24, 50, 3.662),
    ("Chromium-52", 24, 52, 3.643),
    ("Chromium-53", 24, 53, 3.726),
    ("Chromium-54", 24, 54, 3.776),
    ("Manganese-55", 25, 55, 3.68),
    ("Iron-54", 26, 54, 3.68),
    ("Iron-56", 26, 56, 3.77),
    ("Cobalt-59", 27, 59, 3.775),
    ("Nickel-58", 28, 58, 3.772),
]

Q_values = [np.linspace(0, 2, 1000)]
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


def Form_factor_to_matter_radii_rms(Fq):
    rho_real = []
    rho_real_squared = []
    for i, q in enumerate(q_vals):  # Loop with index
        integrand = r**2 * spherical_jn(0, r*q)  # Only the spherical Bessel function, no Fq[i]
        result = 4 * np.pi * np.trapz(integrand, r)
        rho_real.append(result)
    for rho in rho_real:
        square = rho**2
        rho_real_squared.append(square)
    rho_squared_sum = sum(rho_real_squared)
    divided_rho_squared = rho_squared_sum / len(rho_real_squared)
    rms_matter_radius = np.sqrt(divided_rho_squared)
    return rms_matter_radius






"""
fig, ax = plt.subplots(figsize=(8,6))
plt.subplots_adjust(bottom=0.3, left=0.3)

current_isotope_index = 0

def update_plot():
    ax.clear()
    name, Z, A, r_rms = isotopes[current_isotope_index]
    R0 = R0_from_rms(r_rms, a_charge)
    rho0 = compute_rho0(r, R0, a_charge, Z)
    rho = fermi_density(r, R0, a_charge, rho0)
    F_q = compute_form_factor(Q_values[0], rho, r)
    ax.plot(Q_values[0], np.abs(F_q), label=f"{name}")
    ax.set_title(f"{name} | RMS Radius = {r_rms:.3f} fm")
    ax.set_xlabel("Momentum Transfer $q$ [fm$^{-1}$]")
    ax.set_ylabel("Form Factor Magnitude $|F(q)|$")
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    fig.canvas.draw_idle()

def select_isotope_dialog():
    dialog = tk.Tk()
    dialog.title("Select Isotope")
    isotope_names = [iso[0] for iso in isotopes]
    selected = tk.StringVar(value=isotope_names[0])
    combo = ttk.Combobox(dialog, textvariable=selected, values=isotope_names, state="readonly", width=40)
    combo.pack(padx=10, pady=10)
    def on_ok():
        dialog.quit()
        dialog.destroy()
    def on_quit():
        selected.set("")
        dialog.quit()
        dialog.destroy()
    ok_button = tk.Button(dialog, text="OK", command=on_ok)
    ok_button.pack(pady=(0,5))
    quit_button = tk.Button(dialog, text="Quit", command=on_quit)
    quit_button.pack(pady=(0,10))
    dialog.eval('tk::PlaceWindow . center')
    dialog.mainloop()
    return selected.get() if selected.get() else None

# Main loop: alternate between plot and selector
while True:
    fig, ax = plt.subplots(figsize=(8,6))
    update_plot()
    plt.show()
    selected_name = select_isotope_dialog()
    names = [iso[0] for iso in isotopes]
    if selected_name in names:
        current_isotope_index = names.index(selected_name)
    else:
        break  # Exit if dialog closed without selection or user clicked Quit
"""