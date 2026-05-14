#solving the catalytic ODEs using scipy
#Generating the synthetic "ground truth" data is the crucial first step before training any Physics-Informed Neural Network (PINN).
#Since real-world data is often noisy, the paper specifically mentions adding uniformly distributed noise to the numerical solutions to mimic experimental measurement uncertainty.
import numpy as np
import pandas as pd
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# =====================================================================
# 1. Physical Constants & Simulation Parameters
# =====================================================================
R = 1.0  # Scaled universal gas constant
t = np.linspace(0, 25, 51)  # Time array (delta_t = 0.5 as per the paper)

# Training temperatures specified in Section 4.1
training_temperatures = [0.5, 0.75, 1.0, 1.25, 1.5]

# Initial conditions: [E]0=0.5, [S]0=1.0, [ES]0=0, [P]0=0
y0 = [0.5, 1.0, 0.0, 0.0]

# =====================================================================
# 2. Define the Coupled ODE System
# =====================================================================
def catalytic_odes(y, t, T):
    """
    Michaelis-Menten style catalytic reaction: E + S <-> ES -> E + P
    """
    E, S, ES, P = y
    
    # Reaction energy barriers
    dE1, dE2, dE3 = 1.0, 1.5, 0.5
    
    # Arrhenius rate constants (Prefactor F = 1.0)
    k1 = np.exp(-dE1 / (R * T))
    k2 = np.exp(-dE2 / (R * T))
    k3 = np.exp(-dE3 / (R * T))
    
    # The Governing ODEs
    dE_dt  = -k1 * E * S + k3 * ES + k2 * ES
    dS_dt  = -k1 * E * S + k2 * ES
    dES_dt =  k1 * E * S - k3 * ES - k2 * ES
    dP_dt  =  k3 * ES
    
    return [dE_dt, dS_dt, dES_dt, dP_dt]

# =====================================================================
# 3. Generate Data and Add Measurement Noise
# =====================================================================
all_data = []

for T in training_temperatures:
    # Solve the ODEs for the exact "Ground Truth"
    solution = odeint(catalytic_odes, y0, t, args=(T,))
    
    # Extract exact concentrations
    E_exact, S_exact, ES_exact, P_exact = solution.T
    
    # Add Uniform Noise (-0.01 to 0.01) to mimic measurement uncertainty
    noise_level = 0.01
    E_noisy  = E_exact  + np.random.uniform(-noise_level, noise_level, len(t))
    S_noisy  = S_exact  + np.random.uniform(-noise_level, noise_level, len(t))
    ES_noisy = ES_exact + np.random.uniform(-noise_level, noise_level, len(t))
    P_noisy  = P_exact  + np.random.uniform(-noise_level, noise_level, len(t))
    
    # Ensure concentrations don't drop below absolute zero due to noise
    E_noisy  = np.clip(E_noisy, 0, None)
    S_noisy  = np.clip(S_noisy, 0, None)
    ES_noisy = np.clip(ES_noisy, 0, None)
    P_noisy  = np.clip(P_noisy, 0, None)
    
    # Store the results in a list of dictionaries
    for i in range(len(t)):
        all_data.append({
            'Time': t[i],
            'Temperature': T,
            'S_exact': S_exact[i], 'S_noisy': S_noisy[i],
            'P_exact': P_exact[i], 'P_noisy': P_noisy[i],
            'E_exact': E_exact[i], 'E_noisy': E_noisy[i],
            'ES_exact': ES_exact[i], 'ES_noisy': ES_noisy[i]
        })

# =====================================================================
# 4. Package into a DataFrame & Export
# =====================================================================
df_training = pd.DataFrame(all_data)

# Would typically export this to CSV for ML pipeline:
# df_training.to_csv('catalytic_training_data.csv', index=False)

print("Data generation complete! Here is a preview of the structured dataset:\n")
print(df_training.head())

# =====================================================================
# 5. Visualize Ground Truth vs. Noisy Data (For T=1.0)
# =====================================================================
df_plot = df_training[df_training['Temperature'] == 1.0]

plt.figure(figsize=(10, 6))

# Plot exact curves (Ground Truth)
plt.plot(df_plot['Time'], df_plot['S_exact'], label='Substrate (Exact)', color='blue', linewidth=2)
plt.plot(df_plot['Time'], df_plot['P_exact'], label='Product (Exact)', color='red', linewidth=2)

# Plot noisy scatter points (What the PINN actually sees)
plt.scatter(df_plot['Time'], df_plot['S_noisy'], color='blue', alpha=0.5, label='Substrate (Measured)')
plt.scatter(df_plot['Time'], df_plot['P_noisy'], color='red', alpha=0.5, label='Product (Measured)')

plt.title('Synthetic Training Data: Ground Truth vs. Noisy Measurements (T=1.0)')
plt.xlabel('Time (t)')
plt.ylabel('Concentration')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
