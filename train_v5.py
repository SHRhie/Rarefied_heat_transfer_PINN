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
def initialize_and_train_model(P_Torr, k, T_inf, Tip_grid, rho, c=800):
    model = TransientThermalModel(
        k=k, d_b=0.009, Td=77, T_inf=T_inf, L=0.048, Tip_grid=Tip_grid, Q_bias=0, sigma=5.67e-8,
        epsilon_2=0.02, P_Torr=P_Torr, alpha=k / (rho * c), rho=rho, c=800, a_cooler=0.039, b_cooler=-2, time_steps=60
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
    filename = f"temperature_distribution_Torr_{P_Torr}_Tinf_{T_inf}_k_{k}_Tip_grid_{Tip_grid}.npy"
    np.save(filename, model.predict_temperature()[1])

    # xt_values 한 번만 저장
    if not os.path.exists('xt_values.npy'):
        np.save('xt_values.npy', model.xt_values)

    return model
'''
# 다양한 조건에 대해 모델 훈련 및 저장
P_Torr_values = [1e-4, 1e-3, 1e-2, 1e-1, 1e0]
k_values = [0.2, 0.8, 1.4]
rho = 2640  # 고정된 값

# 1. gas pressure와 k
print("Training for gas pressure와 k")
for P_Torr in P_Torr_values:
    for k in k_values:
        initialize_and_train_model(P_Torr, k, T_inf=300, rho=rho, Tip_grid=0.001)
'''
P_Torr_values = [1e-4]
k_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
rho = 2640  # 고정된 값

# 2. k와 gas pressure
print("Training for k와 gas pressure")
for k in k_values:
    for P_Torr in P_Torr_values:
        initialize_and_train_model(P_Torr, k, T_inf=300, rho=rho, Tip_grid=0.001)

'''
k_values = [0.2, 0.8, 1.4]
rho = 2640  # 고정된 값
grid_values = [0.0005, 0.001, 0.0015, 0.002, 0.003]  # 상대 열용량 범위

# 3. c_values와 k
print("Training for c_values와 k")
for grid in grid_values:
    for k in k_values:
        initialize_and_train_model(P_Torr=1e-4, k=k, T_inf=300, rho=rho, Tip_grid=grid)


k_values = [0.2, 0.8, 1.4]
T_inf_values = [150, 200, 230, 270, 300, 325, 350]
rho = 2640  # 고정된 값

# 4. T_inf_values와 k
print("Training for T_inf_values와 k")
for T_inf in T_inf_values:
    for k in k_values:
        initialize_and_train_model(P_Torr=1e-4, k=k, T_inf=T_inf, rho=rho, Tip_grid=0.001)
'''
# Time
print(f"{time.time()-start:.4f} sec")