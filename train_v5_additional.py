import os
import time
import tensorflow as tf
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.optimize

from scipy.special import erfc
from utils_v5 import *

start = time.time()

# 모델 초기화 및 훈련 함수 정의
def initialize_and_train_model(P_Torr, k, T_inf, Tip_grid, rho, c, epsilon_2):
    model = TransientThermalModel(
        k=k, d_b=0.009, Td=77, T_inf=T_inf, L=0.048, Tip_grid=Tip_grid, Q_bias=0, sigma=5.67e-8,
        epsilon_2=0.02, P_Torr=P_Torr, alpha=k / (rho * c), rho=rho, c=c, a_cooler=0.039, b_cooler=-2, time_steps=60
    )

    # 모델 훈련
    model.train(epochs=400)

    # SciPy Optimizer를 사용한 최적화
    model.ScipyOptimizer(method='L-BFGS-B',
                         options={'maxiter': 4000,
                                  'maxfun': 50000,
                                  'maxcor': 50,
                                  'maxls': 50,
                                  'ftol': np.finfo(float).eps,
                                  'gtol': np.finfo(float).eps,
                                  'factr': np.finfo(float).eps,
                                  'iprint': 50})

    # 온도 분포 저장
    filename = f"temperature_distribution_Torr_{P_Torr}_rho_{rho}_c_{c}_k_{k}_epsilon{epsilon_2}.npy"
    np.save(filename, model.predict_temperature()[1])

    # xt_values 한 번만 저장
    if not os.path.exists('xt_values_1000.npy'):
        np.save('xt_values_1000.npy', model.xt_values)

    return model

P_Torr_values = [1e-4, 1e-2, 1e0]
'''
# 1. Training for gas pressure와 k
k_values = [0.2, 0.5, 0.8, 1.1, 1.4]
print("Training for gas pressure와 k")
for P_Torr in P_Torr_values:
    for k in k_values:
        initialize_and_train_model(P_Torr, k, T_inf=300, Tip_grid=0.001, rho=2640, c=800)

# 2. Training for gas pressure와 rho
rho_values = [1500, 2000, 2500, 3000, 3500]
print("Training for gas pressure와 rho")
for P_Torr in P_Torr_values:
    for rho in rho_values:
        initialize_and_train_model(P_Torr, k=0.8, T_inf=300, Tip_grid=0.001, rho=rho, c=800)

# 3. Training for gas pressure와 c
c_values = [600, 700, 800, 900, 1000]
print("Training for gas pressure와 c")
for P_Torr in P_Torr_values:
    for c in c_values:
        initialize_and_train_model(P_Torr, k=0.8, T_inf=300, Tip_grid=0.001, rho=2640, c=c)
'''
# 3. Training for gas pressure와 c
epsilon_values = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
print("Training for gas pressure와 epsilon")
for P_Torr in P_Torr_values:
    for epsilon in epsilon_values:
        initialize_and_train_model(P_Torr, k=0.8, T_inf=300, Tip_grid=0.001, rho=2640, c=800, epsilon_2=epsilon)

# Time
print(f"{time.time()-start:.4f} sec")
