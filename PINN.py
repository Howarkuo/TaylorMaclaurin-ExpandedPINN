# instead of treating the kinetic reate k as a fixed number 
# defined the coefficient of its Taylor expansion as nn.Parameter (trainable variables)
# The neural network will learn these coefficients at the same time it learns the concentration profiles.

import torch
import torch.nn as nn

class ChemicalKineticsPINN(nn.Module):
    def __init__(self, ref_temp):
        super().__init__()
        self.ref_temp = ref_temp
        
        # 1. The Neural Network (Predicts Concentrations [A] and [B])
        # Inputs: time (t), temperature (T) -> Output: [A], [B]
        self.net = nn.Sequential(
            nn.Linear(2, 32),
            nn.Tanh(),
            nn.Linear(32, 2) 
        )
        
        # 2. The Taylor Series Coefficients (The "Augmented" Math Part)
        # We approximate the rate constant k(T) using a 2nd-order Taylor expansion:
        # k(T) = k0 + k1*(T - T_ref) + k2*(T - T_ref)^2
        # These are trainable parameters that the model will optimize!
        self.k0 = nn.Parameter(torch.tensor([0.1]))  # 0th order
        self.k1 = nn.Parameter(torch.tensor([0.01])) # 1st order
        self.k2 = nn.Parameter(torch.tensor([0.001]))# 2nd order

    def get_rate_constant(self, T):
        """Calculates the kinetic parameter using the truncated Taylor series."""
        delta_T = T - self.ref_temp
        k_T = self.k0 + (self.k1 * delta_T) + (self.k2 * (delta_T ** 2))
        return k_T

    def forward(self, t, T):
        # Predict concentration profiles
        inputs = torch.cat([t, T], dim=1)
        concentrations = self.net(inputs)
        return concentrations
