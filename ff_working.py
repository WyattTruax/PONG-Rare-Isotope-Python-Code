# Running corrected script and showing plots + printed RMS radii per isotope.
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import spherical_jn

# --- Cleaned constants and parameters ---
rho0_matter = 0.17   # nucleons/fm^3 (comment cleaned)
rho0_charge = 0.08   # protons/fm^3
a_matter = 0.56      # fm
a_charge = 0.53      # fm

# Isotope data: (name, Z, A)
isotopes = [
    ("Carbon-12"   , 6, 12),
    ("Magnesium-33", 12, 33),
    ("Iron-56"     , 26, 56),
    ("Lead-208"    , 82, 208)
]

# Radius values
r = np.linspace(0.0, 15.0, 500)  # fm

# helper: compute form factor for an array of q values (returns array of same length)
def compute_form_factor(q_vals, rho_r, r):
    # q_vals: 1D array of q values
    # rho_r: 1D array over r
    # Use broadcasting to form a 2D integrand [nq, nr] then integrate over r
    q = np.asarray(q_vals)
    r_row = r[np.newaxis, :]          # shape (1, nr)
    q_col = q[:, np.newaxis]          # shape (nq, 1)
    j0 = spherical_jn(0, q_col * r_row)   # shape (nq, nr)
    integrand = j0 * rho_r[np.newaxis, :] * (r_row**2)
    ff = 4.0 * np.pi * np.trapz(integrand, x=r_row, axis=1)  # shape (nq,)
    return ff

def calc_rms_from_ff(q_vals, ff):
    # Normalize form factor at q=0 to 1
    # We'll fit F(q) ~= 1 + c * q^2 for small q and use <r^2> = -6 * c
    q = np.asarray(q_vals)
    ff = np.asarray(ff)
    # choose points near q=0 for the fit
    # ensure the first point is q~0; if not, use first few smallest q values
    nfit = max(5, min(20, len(q)//10))  # use between 5 and 20 points depending on length
    idx = np.argsort(q)[:nfit]
    q_small = q[idx]
    ff_small = ff[idx]
    # normalize
    f0 = np.interp(0.0, q_small, ff_small) if 0.0 not in q_small else ff_small[q_small.tolist().index(0.0)]
    if f0 == 0:
        # fallback: use first element but warn
        f0 = ff_small[0]
    F_small = ff_small / f0
    # Fit linear in q^2: F = a0 + a1 * q^2  (a0 should be ~1)
    coeffs = np.polyfit(q_small**2, F_small, 1)  # [a1, a0]
    a1 = coeffs[0]
    rms_sq = -6.0 * a1
    if rms_sq < 0:
        rms = np.nan
    else:
        rms = np.sqrt(rms_sq)
    return rms, rms_sq, coeffs

# --- Plot densities and compute form factors + RMS per isotope ---
plt.figure(figsize=(10,5))
for name, Z, A in isotopes:
    R0_matter = 1.31 * A**(1/3) - 0.84  # nuclear matter radius in fm (empirical)
    R0_charge = 1.76 * Z**(1/3) - 0.96  # nuclear charge radius in fm (empirical)

    # Fermi (Woods-Saxon-like) distributions
    rho_matter = rho0_matter / (1.0 + np.exp((r - R0_matter) / a_matter))
    rho_charge = rho0_charge / (1.0 + np.exp((r - R0_charge) / a_charge))

    plt.plot(r, rho_matter, label=f"{name} - Matter", linestyle='-')
    plt.plot(r, rho_charge, label=f"{name} - Charge", linestyle='--')

plt.xlim([-0.25, 16])
plt.ylim([-0.01,0.2])
plt.xlabel("Radius r [fm]")
plt.ylabel("Density ρ(r) [units: nucleons or protons / fm^3]")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper right', fontsize='small')
plt.title("Matter and Charge Density for Various Isotopes")
plt.tight_layout()
plt.show()

# compute form factors and RMS and optionally plot form factors
q_vals = np.linspace(0.001, 4.0, 300)  # fm^-1
plt.figure(figsize=(10,5))
for name, Z, A in isotopes:
    R0_matter = 1.31 * A**(1/3) - 0.84
    rho_matter = rho0_matter / (1.0 + np.exp((r - R0_matter) / a_matter))

    ff = compute_form_factor(q_vals, rho_matter, r)
    # normalize so F(0)=1 for RMS extraction
    ff_norm = ff / ff[0]

    rms, rms_sq, coeffs = calc_rms_from_ff(q_vals, ff)
    print(f"{name}: RMS (from matter form factor) = {rms:.4f} fm (r^2 = {rms_sq:.6f} fm^2)")

    plt.plot(q_vals, ff_norm, label=f"{name}")

plt.xlabel("q [fm^-1]")
plt.ylabel("Normalized form factor F(q)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.title("Normalized Matter Form Factors")
plt.tight_layout()
plt.show()
