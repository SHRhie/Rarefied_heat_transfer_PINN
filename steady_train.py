import os
import numpy as np
from thermal_model_lbfgs import ThermalModel

P_set = [1e0, 1e-1, 1e-2, 1e-4]

for P_Torr in P_set:
    # Get the number of epochs from the user
    epochs = 200

    # Create the ThermalModel instance with the provided P_Torr
    thermal_model = ThermalModel(
        k=0.8, d_b=0.009, Td=77, T_inf=300,
        L=0.048, Q_bias=0, sigma=5.67e-8, epsilon_2=0.02,
        P_Torr=P_Torr  # Pass the P_Torr value to the model
    )
    print(thermal_model.m2)  
    thermal_model.model.summary()

    # Train the model
    thermal_model.train(epochs)
    # Train lbfgs
    thermal_model.ScipyOptimizer(method='L-BFGS-B',
                          options={'maxiter': 4000,
                                   'maxfun': 50000,
                                   'maxcor': 50,
                                   'maxls': 50,
                                   'ftol': np.finfo(float).eps,
                                   'gtol': np.finfo(float).eps,
                                   'factr': np.finfo(float).eps,
                                   'iprint': 50})

    # Plot the temperature distribution after training
    thermal_model.plot_temperature_distribution()

    # Save the weights
    weight_filename = f"weight_{P_Torr:.1e}_Torr.h5"
    thermal_model.model.save_weights(weight_filename)
    print(f"Weights saved to {weight_filename}")
    #print(ThermalModel.m2)