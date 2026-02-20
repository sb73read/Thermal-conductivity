import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm


# --- Core Physics Engine ---
def calculate_k_eff(k_s, k_g, phi, diameters, fractions, thickness, pressure=101325):
    # Sauter Mean Diameter (D32)
    d32 = 1.0 / np.sum(np.array(fractions) / np.array(diameters))

    # Smoluchowski effect (Knudsen suppression in small gaps)
    # Mean free path of air approx 68nm at 1atm
    mfp = 68e-9 * (101325 / pressure)
    k_g_eff = k_g / (1 + 2 * (mfp / (0.1 * d32)))

    # ZBS Model parameters
    alpha = k_s / k_g_eff
    B = 1.25 * ((1 - phi) / phi) ** (10 / 9)

    # Model components
    sqrt_1_phi = np.sqrt(1 - phi)
    term1 = 1 - sqrt_1_phi

    num = 2 / (1 - B / alpha)
    den = (B / alpha) / ((1 - B / alpha) ** 2)
    ln_term = np.log(alpha / B)

    term2 = sqrt_1_phi * ((num * ln_term) - den)
    k_eff = k_g_eff * (term1 + term2)

    # Wall effect (Layer Thickness)
    d_max = np.max(diameters)
    if thickness < (10 * d_max):
        k_eff *= (1 - np.exp(-thickness / (2 * d32)))

    return k_eff, d32


# --- Streamlit UI ---
st.set_page_config(page_title="Powder Thermal Analyzer", layout="wide")
st.title("🔬 Polydisperse Powder Thermal Conductivity Model")
st.markdown("Estimate the effective thermal conductivity ($k_{eff}$) of powder beds based on PSD and packing.")

with st.sidebar:
    st.header("Material Properties")
    k_solid = st.number_input("Solid Conductivity (W/mK)", value=30.0, help="Conductivity of the fully dense material")
    k_gas = st.number_input("Gas Conductivity (W/mK)", value=0.026, help="Air is ~0.026")
    pressure = st.number_input("Gas Pressure (Pa)", value=101325)

    st.header("Bed Geometry")
    phi = st.slider("Packing Fraction (φ)", 0.3, 0.74, 0.60)
    thickness = st.number_input("Layer Thickness (µm)", value=500.0) * 1e-6

st.subheader("Particle Size Distribution (PSD)")
col1, col2 = st.columns([1, 2])

with col1:
    st.write("Enter your size bins and volume fractions:")
    # Default data
    default_data = [
        {"Size (µm)": 10.0, "Fraction": 0.1},
        {"Size (µm)": 30.0, "Fraction": 0.3},
        {"Size (µm)": 60.0, "Fraction": 0.4},
        {"Size (µm)": 100.0, "Fraction": 0.2},
    ]
    edited_df = st.data_editor(default_data, num_rows="dynamic")

    sizes = np.array([row["Size (µm)"] for row in edited_df]) * 1e-6
    fractions = np.array([row["Fraction"] for row in edited_df])

    # Validation
    if abs(sum(fractions) - 1.0) > 0.01:
        st.warning(f"Fractions sum to {sum(fractions):.2f}. They should sum to 1.0.")

with col2:
    # PSD Visualization
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar([s * 1e6 for s in sizes], fractions, width=10, color='skyblue', edgecolor='black')
    ax.set_xlabel("Diameter (µm)")
    ax.set_ylabel("Volume Fraction")
    ax.set_title("Input Particle Distribution")
    st.pyplot(fig)

# --- Calculation ---
if st.button("Calculate Effective Conductivity", type="primary"):
    if len(sizes) > 0 and abs(sum(fractions) - 1.0) < 0.1:
        k_res, d32_res = calculate_k_eff(k_solid, k_gas, phi, sizes, fractions, thickness, pressure)

        st.divider()
        m1, m2, m3 = st.columns(3)
        m1.metric("Effective $k_{eff}$", f"{k_res:.4f} W/mK")
        m2.metric("Sauter Mean ($D_{32}$)", f"{d32_res * 1e6:.2f} µm")
        m3.metric("Reduction from Solid", f"{((1 - k_res / k_solid) * 100):.1f}%")

        st.info(
            f"The coordination number for this packing ($\phi={phi}$) is estimated at **{13.28 * phi:.2f}** contacts per particle.")
    else:
        st.error("Please ensure fractions sum to 1.0 and at least one size bin exists.")