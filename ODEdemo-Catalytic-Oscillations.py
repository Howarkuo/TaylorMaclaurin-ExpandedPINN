#Catalytic Reactions (Michaelis-Menten type)
# Implement the coupled ODEs from Section 5.1.1 (Equations 14-17) for Enzyme (E), Substrate (S), Enzyme-Substrate complex (ES), and Product (P).
# Incorporate the Arrhenius equation (Eq. 6) to make the rate constants ($k_1$, $k_2$, $k_3$) temperature-dependent.

# Chemical Oscillations (Brusselator Dynamics):
# Implement the coupled ODEs from Section 5.3 (Equations 22 and 23) for intermediate species $X$ and $Y$.
# Set up the specific initial conditions ($[A]_0 = 1.0$, $[B]_0 = 2.5$) and observe the oscillating concentration profiles over time.

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# =====================================================================
# Global Simulation Parameters
# =====================================================================
R = 1.0  # Scaled universal gas constant
t = np.linspace(0, 25, 500)  # Integration time interval tau = 0 to 25

# =====================================================================
# 1. Catalytic Reactions (Michaelis-Menten type)
# =====================================================================
def catalytic_odes(y, t, T):
    """
    ODEs for Enzyme (E), Substrate (S), Enzyme-Substrate complex (ES), 
    and Product (P) corresponding to Equations 14-17.
    """
    E, S, ES, P = y
    
    # Reaction energy barriers (Section 5.1.1)
    dE1, dE2, dE3 = 1.0, 1.5, 0.5
    
    # Arrhenius rate constants (Eq. 6), assuming prefactor F = 1.0
    k1 = np.exp(-dE1 / (R * T))
    k2 = np.exp(-dE2 / (R * T))
    k3 = np.exp(-dE3 / (R * T))
    
    # Coupled ODEs (Eqs 14-17)
    dE_dt  = -k1 * E * S + k3 * ES + k2 * ES
    dS_dt  = -k1 * E * S + k2 * ES
    dES_dt =  k1 * E * S - k3 * ES - k2 * ES
    dP_dt  =  k3 * ES
    
    return [dE_dt, dS_dt, dES_dt, dP_dt]

# Initial conditions & Temperature for Catalytic Reactions
y0_cat = [0.5, 1.0, 0.0, 0.0]  # [E]_0=0.5, [S]_0=1, [ES]_0=0, [P]_0=0
T_cat = 1.4                    # Simulating for T = 1.4 (as in Figure 1)

# Solve ODEs
sol_cat = odeint(catalytic_odes, y0_cat, t, args=(T_cat,))
E_sim, S_sim, ES_sim, P_sim = sol_cat.T

# =====================================================================
# 2. Chemical Oscillations (Brusselator Dynamics)
# =====================================================================
def brusselator_odes(y, t, T, A, B):
    """
    ODEs for intermediate species X and Y corresponding to Equations 22-23.
    """
    X, Y = y
    
    # Reaction energy barriers (Section 5.3)
    dE1 = dE2 = dE3 = dE4 = 1.0
    
    # Arrhenius rate constants
    k1 = np.exp(-dE1 / (R * T))
    k2 = np.exp(-dE2 / (R * T))
    k3 = np.exp(-dE3 / (R * T))
    k4 = np.exp(-dE4 / (R * T))
    
    # Coupled ODEs (Eqs 22-23)
    dX_dt = k1 * A - k2 * B * X + k3 * (X**2) * Y - k4 * X
    dY_dt = k2 * B * X - k3 * (X**2) * Y
    
    return [dX_dt, dY_dt]

# Initial conditions & Constants for Oscillations
A_const = 1.0
B_const = 2.5
y0_brus = [1.0, 1.0]  # Starting concentrations for intermediates X and Y
T_brus = 1.4          # Simulating for T = 1.4 (as in Figure 13)

# Solve ODEs
sol_brus = odeint(brusselator_odes, y0_brus, t, args=(T_brus, A_const, B_const))
X_sim, Y_sim = sol_brus.T

# =====================================================================
# Plotting the Results
# =====================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Catalytic Reactions
ax1.plot(t, S_sim,  label='Substrate (S)', color='blue', linewidth=2)
ax1.plot(t, P_sim,  label='Product (P)', color='red', linewidth=2)
ax1.plot(t, ES_sim, label='Enzyme-Substrate (ES)', color='gray', linestyle='--', linewidth=2)
ax1.plot(t, E_sim,  label='Enzyme (E)', color='black', linestyle='-.', linewidth=2)
ax1.set_title('Catalytic Reactions (T = 1.4)', fontsize=14)
ax1.set_xlabel('Time (t)', fontsize=12)
ax1.set_ylabel('Concentration', fontsize=12)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Brusselator Dynamics
ax2.plot(t, X_sim, label='Intermediate (X)', color='blue', linewidth=2)
ax2.plot(t, Y_sim, label='Intermediate (Y)', color='red', linewidth=2)
ax2.set_title('Chemical Oscillations: Brusselator (T = 1.4)', fontsize=14)
ax2.set_xlabel('Time (t)', fontsize=12)
ax2.set_ylabel('Concentration', fontsize=12)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
