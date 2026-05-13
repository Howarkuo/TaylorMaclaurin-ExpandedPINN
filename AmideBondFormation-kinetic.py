#A (Amine) + B (Carboxylic Acid derivative) $\rightleftharpoons$ C (Amide Drug) + byproduct
# we will treat it as a simple reversible $A + B \rightleftharpoons C$ reaction where all stoichiometric coefficients ($\nu_A, \nu_B, \nu_C$) are $1$
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# =====================================================================
# 1. System Definition
# =====================================================================
R = 1.0  # Scaled universal gas constant
t = np.linspace(0, 15, 300)  # Time array

def get_rate_constants(T):
    """Calculates forward and backward Arrhenius rate constants."""
    dE_f, dE_b = 1.0, 1.5  # Energy barriers (backward is harder/slower)
    k_f = np.exp(-dE_f / (R * T))
    k_b = np.exp(-dE_b / (R * T))
    return k_f, k_b

def coupling_reaction_odes(y, t, T):
    """ODEs for the reaction A + B <-> C"""
    A, B, C = y
    k_f, k_b = get_rate_constants(T)
    
    rate = k_f * A * B - k_b * C
    
    dA_dt = -rate
    dB_dt = -rate
    dC_dt =  rate
    return [dA_dt, dB_dt, dC_dt]

def calculate_KF(A, B, C, T):
    """
    Calculates the Pseudo-First-Order Rate Functional (K_F) 
    using Equation 9 simplified for A + B <-> C.
    """
    k_f, k_b = get_rate_constants(T)
    # Add a tiny epsilon to A to prevent division by zero at extreme conditions
    epsilon = 1e-8 
    K_F = -k_f * B + k_b * (C / (A + epsilon))
    return K_F

# =====================================================================
# 2. Simulation and K_F Calculation for Different Temperatures
# =====================================================================
# Initial conditions: [A]0 = 1.0, [B]0 = 1.5 (slight excess of B), [C]0 = 0
y0 = [1.0, 1.5, 0.0]  
temperatures = [0.5, 1.0, 1.5]

results = {}

for T in temperatures:
    # 1. Solve the ODEs to get the concentration profiles
    sol = odeint(coupling_reaction_odes, y0, t, args=(T,))
    A_sim, B_sim, C_sim = sol.T
    
    # 2. Calculate K_F dynamically across the time steps
    KF_array = calculate_KF(A_sim, B_sim, C_sim, T)
    
    results[T] = KF_array

# =====================================================================
# 3. Plotting the Rate Functional Dynamics
# =====================================================================
plt.figure(figsize=(9, 6))

colors = {0.5: 'blue', 1.0: 'green', 1.5: 'red'}

for T in temperatures:
    plt.plot(t, results[T], label=f'Temperature = {T}', color=colors[T], linewidth=2.5)

plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.title('Dynamic Rate Functional ($K_F$) for Amide Bond Formation', fontsize=14)
plt.xlabel('Time (t)', fontsize=12)
plt.ylabel('Rate Functional $K_F$ (Speed & Direction of A consumption)', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
