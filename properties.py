import os
import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

from scipy.special import erfc
from utils_v5 import *

start = time.time()

mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['mathtext.fontset'] = 'stix'  # 수학 표현에 대한 글꼴 설정
mpl.rcParams['font.size'] = 20  # 기본 글꼴 크기 설정
mpl.rcParams['figure.dpi'] = 150

# 물성치 데이터
materials = [
    {"name": "Borosilicate Glass", "k": 0.8, "rho": 2640, "c": 800, "epsilon": 0.02},
    {"name": "Copper", "k": 400, "rho": 8960, "c": 385, "epsilon": 0.02},
    {"name": "Aluminum", "k": 237, "rho": 2700, "c": 897, "epsilon": 0.04},
    {"name": "Stainless Steel (304)", "k": 16, "rho": 8000, "c": 500, "epsilon": 0.3},
    {"name": "Copper Oxide Coating", "k": 33, "rho": 6400, "c": 380, "epsilon": 0.75},
    {"name": "Gold", "k": 315, "rho": 19300, "c": 129, "epsilon": 0.02},
    {"name": "Silver", "k": 430, "rho": 10500, "c": 235, "epsilon": 0.02},
    {"name": "Titanium", "k": 21.9, "rho": 4500, "c": 523, "epsilon": 0.4},
    {"name": "CFRP", "k": 25, "rho": 1600, "c": 800, "epsilon": 0.7},
    {"name": "Alumina", "k": 30, "rho": 3960, "c": 880, "epsilon": 0.25},
    {"name": "Fiberglass", "k": 0.04, "rho": 2200, "c": 800, "epsilon": 0.8}
]

P_Torr_values = [1e-4, 1e-3, 1e-2, 1e-1, 1e0]
Tip_grid = 0.001
T_inf = 300  # 환경 온도 (고정된 값)

# 모델 초기화 및 훈련 함수 정의
def initialize_and_train_model(material, P_Torr):
    model = TransientThermalModel(
        k=material["k"], d_b=0.009, Td=77, T_inf=T_inf, L=0.048, Tip_grid=Tip_grid, Q_bias=0, sigma=5.67e-8,
        epsilon_2=material["epsilon"], P_Torr=P_Torr, alpha=material["k"] / (material["rho"] * material["c"]),
        rho=material["rho"], c=material["c"], a_cooler=0.039, b_cooler=-2, time_steps=60
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

    # 온도 분포 예측 및 cooldown time 계산
    temperature_profile = model.predict_temperature()[1]
    cooldown_time = np.argmax(temperature_profile <= 77)  # 예시로 특정 온도 이하로 떨어지는 시간 구하기
    return cooldown_time

# 다양한 조건에 대해 모델 훈련 및 cooldown time 저장
cooldown_results = {}

for material in materials:
    cooldown_times = []
    for P_Torr in P_Torr_values:
        cooldown_time = initialize_and_train_model(material, P_Torr)
        cooldown_times.append(cooldown_time)
    cooldown_results[material["name"]] = cooldown_times

# Cooldown time 결과 시각화
plt.figure(figsize=(5, 4))

for material in materials:
    plt.plot(
        P_Torr_values,
        cooldown_results[material["name"]],
        label=material["name"]
    )

plt.xlabel('P_Torr (Pressure)')
plt.ylabel('Cooldown Time')
plt.title('Cooldown Time vs Pressure for Various Materials')
plt.xscale('log')  # 로그 스케일로 x축 설정
plt.legend()
plt.grid(True)
plt.show()

# Time
print(f"{time.time() - start:.4f} sec")
