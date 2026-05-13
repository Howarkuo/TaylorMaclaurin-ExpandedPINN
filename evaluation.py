import numpy as np

def evaluate_pinn_metrics(y_true, y_pred):
    """
    Calculates the evaluation metrics for the PINN model.
    
    Parameters:
    y_true (array-like): The experimental or exact test data [Y(t)]
    y_pred (array-like): The PINN model's predicted data [\hat{Y}(t)]
    
    Returns:
    dict: A dictionary containing RMSE, R^2, and nRMSE.
    """
    # Convert inputs to numpy arrays for vector math
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # 1. Root-Mean-Squared Error (RMSE) - Equation 12
    # Formula: sqrt( (1/N_s) * sum( (Y_true - Y_pred)^2 ) )
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    # 2. Coefficient of Determination (R^2) - Equation 13
    # Formula: 1 - ( Sum of Squared Residuals / Total Sum of Squares )
    mean_y_true = np.mean(y_true)
    ss_residuals = np.sum((y_true - y_pred) ** 2)
    ss_total = np.sum((y_true - mean_y_true) ** 2)
    
    # Protect against division by zero if all true values are identical
    if ss_total == 0:
        r2 = 0.0 
    else:
        r2 = 1 - (ss_residuals / ss_total)
        
    # 3. Normalized RMSE (nRMSE) - Equation 21
    # Formula: RMSE / SD_X (where SD_X is the standard deviation of test data)
    sd_x = np.std(y_true)
    
    if sd_x == 0:
        nrmse = 0.0
    else:
        nrmse = rmse / sd_x
        
    # Output the exact results to the console
    print("--- PINN Model Accuracy Metrics ---")
    print(f"RMSE  : {rmse:.5f} (Lower is better)")
    print(f"R^2   : {r2:.5f} (Closer to 1.0 is better)")
    print(f"nRMSE : {nrmse:.5f} (Lower is better, typically < 0.25 is good)")
    print("-" * 35)
    
    return {
        'RMSE': rmse,
        'R2': r2,
        'nRMSE': nrmse
    }

# =====================================================================
# Example Usage 
# =====================================================================

# Synthetic "Ground Truth" test data (e.g., substrate concentration over time)
test_data = [0.80, 0.65, 0.50, 0.35, 0.20, 0.10]

# Synthetic "PINN Predictions" (slightly off from the truth)
predicted_data = [0.78, 0.66, 0.49, 0.37, 0.21, 0.12]

# Run the evaluation function
metrics = evaluate_pinn_metrics(test_data, predicted_data)
