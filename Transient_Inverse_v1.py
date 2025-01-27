import os
import tensorflow as tf
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import scipy.optimize
from scipy.special import erfc

mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['mathtext.fontset'] = 'stix'  # 수학 표현에 대한 글꼴 설정
mpl.rcParams['font.size'] = 20  # 기본 글꼴 크기 설정
mpl.rcParams['figure.dpi'] = 150

class ADAF(tf.keras.layers.Layer):
    def __init__(self, N_p=5, N_m=5, L=1.0, DTYPE='float32', kernel_regularizer=None):
        super(ADAF, self).__init__()
        self.N_p = N_p
        self.N_m = N_m
        self.L = L
        self.x_i = tf.cast(tf.linspace(0.0, L, N_p + 1), dtype=DTYPE)
        self.DTYPE = DTYPE
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)

    def build(self, input_shape):
        self.w = self.add_weight(name='w', shape=(), initializer='random_normal', regularizer=self.kernel_regularizer, trainable=True, dtype=self.DTYPE)
        self.W_i = self.add_weight(name='W_i', shape=(self.N_p,), initializer='random_normal', regularizer=self.kernel_regularizer, trainable=True, dtype=self.DTYPE)

    def out_an(self, n, x_1, x_2, W_i):
        if n == 0:
            a_n = tf.reduce_sum(W_i) / self.N_p
        else:
            sum_1 = tf.math.sin(n * np.pi / self.L * x_1)
            sum_2 = -tf.math.sin(n * np.pi / self.L * x_2)
            a_n = W_i * (sum_1 + sum_2)
            a_n = tf.reduce_sum(a_n)
            a_n = (2.0 / (n * np.pi)) * a_n
        return a_n

    def out_bn(self, n, x_1, x_2, W_i):
        sum_1 = -tf.math.cos(n * np.pi / self.L * x_1)
        sum_2 = tf.math.cos(n * np.pi / self.L * x_2)
        b_n = W_i * (sum_1 + sum_2)
        b_n = tf.reduce_sum(b_n)
        b_n = (2.0 / (n * np.pi)) * b_n
        return b_n

    def out_g_x_1(self, x):
        x_1 = self.x_i[1:]
        x_2 = self.x_i[:-1]

        g_x = self.out_an(0, x_1, x_2, self.W_i) / 2.0 * tf.math.square(x)
        for n in range(1, self.N_m + 1):
            factor = self.L / (n * np.pi)
            factor = tf.constant(factor, self.DTYPE)
            g_x += tf.math.square(factor) * self.out_an(n, x_1, x_2, self.W_i) * (1.0 - tf.math.cos(x / factor))
        return g_x

    def call(self, inputs):
        return self.w * self.out_g_x_1(inputs)

def get_X0(lb, ub, N_0, DTYPE='float32'):
    t_0 = tf.ones((N_0, 1), dtype=DTYPE) * lb[0]
    x_0 = tf.cast(np.random.uniform(lb[1], ub[1], (N_0, 1)), dtype=DTYPE)
    X_0 = tf.concat([t_0, x_0], axis=1)
    return X_0

def get_XB(lb, ub, N_b, DTYPE='float32'):
    t_b = tf.cast(np.random.uniform(lb[0], ub[0], (N_b, 1)), dtype=DTYPE)
    x_b_0 = tf.ones((N_b, 1), dtype=DTYPE) * lb[1]
    x_b_L = tf.ones((N_b, 1), dtype=DTYPE) * ub[1]
    X_b_0 = tf.concat([t_b, x_b_0], axis=1)
    X_b_L = tf.concat([t_b, x_b_L], axis=1)
    return X_b_0, X_b_L

def get_Xr(lb, ub, N_r, DTYPE='float32'):
    t_r = tf.cast(np.random.uniform(lb[0], ub[0], (N_r, 1)), dtype=DTYPE)
    x_r = tf.cast(np.random.uniform(lb[1], ub[1], (N_r, 1)), dtype=DTYPE)
    X_r = tf.concat([t_r, x_r], axis=1)
    return X_r

class TransientThermalModel:
    def __init__(self, k, d_b, Td, T_inf, L, Tip_grid, Q_bias, sigma, epsilon_2, P_Torr, alpha, rho, c, a_cooler, b_cooler, time_steps, target_time, target_temp, activation='tanh', initializer='glorot_normal'):
        self.k = k
        self.d_b = d_b
        self.Td = Td
        self.T_inf = T_inf
        self.L = L
        self.tip = Tip_grid
        self.Q_bias = Q_bias
        self.sigma = sigma
        self.epsilon_2 = epsilon_2
        self.P_Torr = P_Torr
        self.alpha = k / (rho * c)
        self.rho = rho
        self.c = c
        self.h = self.calculate_h_gc()
        self.time_steps = time_steps
        self.h_gc = self.calculate_h_gc()
        self.lbfgs_step = 0
        self.a_cooler = a_cooler
        self.b_cooler = b_cooler
        self.time_steps = time_steps
        self.target_time = target_time
        self.target_temp = target_temp

        self.k = tf.Variable(0.1, trainable=True, dtype=tf.float32, name="Thermal conductivity")

        self.para = [self.k]

        self.lb = tf.constant([0.0, 0.0], dtype=tf.float32)
        self.ub = tf.constant([self.time_steps, self.L], dtype=tf.float32)
        self.DTYPE = 'float32'
        
        self.Ac = np.pi * (tf.square(self.d_b / 2) - tf.square(self.d_b / 2 - 0.001)) #tf.square(self.d_b / 2) - 
        self.m2 = np.pi * self.d_b * self.h / (self.k * self.Ac)
        self.m = tf.sqrt(self.m2)
        
        self.delta_V = np.pi * tf.square(self.d_b / 2) * self.tip  # 볼륨 요소 크기 (적절히 조정)
        self.delta_A = np.pi * tf.square(self.d_b / 2) + (np.pi * self.d_b * self.tip)

        self.t_values = np.concatenate((np.linspace(0, 0.2 * time_steps, num=1000, endpoint=False),
                                        np.linspace(0.2 * time_steps, 0.5 * time_steps, num=1000, endpoint=False),
                                        np.linspace(0.5 * time_steps, time_steps, num=1000+1)))
        self.x_values = np.concatenate((np.linspace(0, 0.98 * self.L, num=500, endpoint=False),
                                        np.linspace(0.98 * self.L, self.L, num=1000+1))).reshape(-1, 1).astype(np.float32)
        #self.x_values = np.linspace(0, self.L, num=1000).reshape(-1, 1).astype(np.float32)
        self.xt_values = np.array([[t, x[0]] for x in self.x_values for t in self.t_values], dtype=np.float32)
        self.xt_tensor = tf.convert_to_tensor(self.xt_values, dtype=tf.float32)

        self.X_0, self.X_b_0, self.X_b_L, self.X_r = self.data_sampling()

        self.activation = activation
        self.initializer = initializer
        self.model = self.build_model()
        self.trainable_parameters = self.model.trainable_variables + self.para
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2, amsgrad=True)

    def build_model(self):
        X_in = tf.keras.Input(shape=(2,))
        hiddens = tf.keras.layers.Lambda(lambda x: 2.0 * (x - self.lb) / (self.ub - self.lb) - 1.0)(X_in)  # 입력 데이터 전처리
        for _ in range(4):
            hiddens = tf.keras.layers.Dense(20, kernel_initializer=self.initializer)(hiddens)
            hiddens = tf.keras.layers.Activation(self.activation)(hiddens)
        hiddens = tf.keras.layers.Dense(20, kernel_initializer=self.initializer)(hiddens)
        hiddens = ADAF(3, 3)(hiddens)
        hiddens = tf.keras.layers.Activation('tanh')(hiddens)
        prediction = tf.keras.layers.Dense(1)(hiddens)
        prediction = 0.5 * (prediction + 1.0)
        prediction = (self.T_inf - self.Td) * prediction
        prediction += self.Td
        model = tf.keras.Model(X_in, prediction)
        return model

    def data_sampling(self):
        X_0 = get_X0(self.lb, self.ub, 2000)
        X_b_0, X_b_L = get_XB(self.lb, self.ub, 2000)
        X_r = get_Xr(self.lb, self.ub, 10000)
        return X_0, X_b_0, X_b_L, X_r

    def calculate_h_gc(self):
        P_fm = 4e-4  # Pressure of free molecule flow in Torr
        P_Torr_conversion_factor = 133.322  # Conversion factor from Torr to Pa

        if self.P_Torr < P_fm:
            return 1.48 * self.P_Torr * P_Torr_conversion_factor + 3 * self.epsilon_2
        elif P_fm <= self.P_Torr < 1:
            return (1.48 * self.P_Torr * P_Torr_conversion_factor) / (1 + 0.34 * self.P_Torr * P_Torr_conversion_factor) + \
                   3 * self.epsilon_2
        else:
            return 4.35 + 3 * self.epsilon_2

    def analytical_solution_at_L(self, t):
        """
        x = L에서 시간 t에 따른 해석적 해를 계산
        """
        T_inf = self.T_inf
        b = self.b_cooler
        a = self.a_cooler
        h_tilde = a / (self.k * self.Ac)
        return (T_inf + b / a) * tf.exp(h_tilde ** 2 * self.alpha * t) * erfc(h_tilde * tf.sqrt(self.alpha * t)) - b / a

    def compute_derivatives(self):
        with tf.GradientTape(persistent=True) as tape:
            t, x = tf.split(self.X_r, 2, axis=1)
            tape.watch(t)
            tape.watch(x)
            T = self.model(tf.stack([t[:, 0], x[:, 0]], axis=1))
            T_x = tape.gradient(T, x)
        T_t = tape.gradient(T, t)
        T_xx = tape.gradient(T_x, x)
        del tape
        return T, T_t, T_xx, T_x

    def fun_u_I(self):
        return self.T_inf

    def get_u_I(self):
        return self.model(self.X_0)

    def get_b_L(self):
        with tf.GradientTape(persistent=True) as tape:
            t, x = tf.split(self.X_b_L, 2, axis=1)
            tape.watch(t)
            tape.watch(x)
            T = self.model(tf.stack([t[:, 0], x[:, 0]], axis=1))
        T_x = tape.gradient(T, x)
        del tape
        return T, T_x

    def get_dT_dt_b(self):
        with tf.GradientTape(persistent=True) as tape:
            t, x = tf.split(self.X_b_L, 2, axis=1)  # X_b_L 기준으로 사용
            tape.watch(t)
            T = self.model(tf.stack([t[:, 0], x[:, 0]], axis=1))
        dT_dt = tape.gradient(T, t)
        del tape
        return dT_dt

    def get_grad(self):
        with tf.GradientTape(persistent=True) as tape:
            total_loss, pde_loss, initial_condition, boundary_condition_x0, boundary_condition_cooler = self.compute_loss()
        gradients = tape.gradient(total_loss, self.trainable_parameters)
        return gradients, total_loss, pde_loss, initial_condition, boundary_condition_x0, boundary_condition_cooler

    def Q_c(self, T):
        return self.a_cooler * T + self.b_cooler
    
    def compute_loss(self):
        T, dT_dt, d2T_dx2, dT_dx = self.compute_derivatives()

        # PDE loss for the governing equation
        pde_loss = tf.reduce_mean(tf.square(dT_dt - self.alpha * d2T_dx2 + self.m2 * (T - self.T_inf) / (self.rho * self.c)))

        # Initial condition at t=0
        initial_condition = tf.reduce_mean(tf.square(self.model(self.X_0) - self.T_inf))

        # Boundary condition at x=0
        boundary_condition_x0 = tf.reduce_mean(tf.square(self.model(self.X_b_0) - self.T_inf))

        T_b, dT_dx_b = self.get_b_L()
        dT_dt_b = self.get_dT_dt_b()

        # Boundary condition at x=L with the tip grid energy balance
        boundary_condition_cooler = (self.rho * self.c * self.delta_V * dT_dt_b) + \
            self.k * self.Ac * dT_dx_b + self.h * self.delta_A * (T_b - self.T_inf) + (self.a_cooler * T_b + self.b_cooler)
        boundary_condition_cooler = tf.reduce_mean(tf.square(boundary_condition_cooler))

        # Target cooldown loss to keep temperature between 76 and 77 at target time
        target_point = tf.convert_to_tensor([[self.target_time, self.L]], dtype=tf.float32)
        predicted_temp_at_target = self.model(target_point)
        lower_bound_loss = tf.reduce_mean(tf.square(tf.nn.relu(76 - predicted_temp_at_target)))
        upper_bound_loss = tf.reduce_mean(tf.square(tf.nn.relu(predicted_temp_at_target - 77)))

        cooldown_loss = lower_bound_loss + upper_bound_loss

        total_loss = (self.m2) * pde_loss + initial_condition + boundary_condition_x0 + (self.m2) * boundary_condition_cooler + (self.m2) * cooldown_loss
        return total_loss, pde_loss, initial_condition, boundary_condition_x0, boundary_condition_cooler

    def train_step(self):
        gradients, total_loss, pde_loss, initial_condition, boundary_condition_x0, boundary_condition_cooler = self.get_grad()
        self.optimizer.apply_gradients(zip(gradients, self.trainable_parameters))
        return total_loss, pde_loss, initial_condition, boundary_condition_x0, boundary_condition_cooler

    def train(self, epochs):
        loss_values = []
        plt.ion()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        for epoch in range(epochs):
            total_loss, pde_loss, initial_condition, boundary_condition_x0, boundary_condition_cooler = self.train_step()
            loss_values.append(total_loss.numpy())
            if epoch % 10 == 0:
                print(f'Epoch: {epoch}, Total Loss: {total_loss.numpy()}')
                print(f'PDE Loss: {pde_loss.numpy()}')
                print(f'Initial Condition Loss: {initial_condition.numpy()}')
                print(f'Boundary Condition at x=0 Loss: {boundary_condition_x0.numpy()}')
                print(f'Boundary Condition at x=L Loss: {boundary_condition_cooler.numpy()}')
                print(f'Thermal conductivity: {self.para[0].numpy()}')

                ax1.clear()
                ax1.semilogy(loss_values, label='Loss')
                ax1.set_xlim(0, right=epoch)
                ax1.tick_params(axis='both', which='major', direction='out', labelsize=14, top=False, right=False, bottom=True, left=True)
                ax1.set_title('Adam')
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('Loss')
                ax1.legend().remove()

                xL_t_values = tf.convert_to_tensor([[t, self.L] for t in self.t_values], dtype=tf.float32)
                T_pred_at_L = self.model(xL_t_values).numpy().flatten()
                T_analytical_at_L = self.analytical_solution_at_L(self.t_values.astype('float32')).numpy().flatten()

                ax2.clear()
                ax2.plot(self.t_values, T_pred_at_L, label='Predicted Temperature', color='red')
                ax2.plot(self.t_values, T_analytical_at_L, label='Analytical Solution', linestyle='dashed', color='black')
                ax2.set_xlim(0, self.time_steps)
                ax2.set_ylim(50, 300)
                ax2.tick_params(axis='both', which='major', direction='out', labelsize=14, top=False, right=False, bottom=True, left=True)
                ax2.set_title('Temperature Profile')
                ax2.set_xlabel('Time (s)')
                ax2.set_ylabel('Temperature (K)')
                ax2.legend().remove()

                plt.tight_layout()
                plt.draw()
                plt.pause(0.1)

        plt.ioff()

    def callback(self, xr=None):
        if self.lbfgs_step % 10 == 0:
            self.plot_iteration()
            plt.pause(0.1)
        self.lbfgs_step += 1

    def predict_temperature(self):
        T_pred = self.model.predict(self.xt_tensor)
        return self.xt_values, T_pred

    def plot_iteration(self, mix=0, max=60):
        plt.figure()
        # 모델을 사용하여 x = L에서의 온도 예측
        xL_t_values = tf.convert_to_tensor([[t, self.L] for t in self.t_values], dtype=tf.float32)
        T_pred_at_L = self.model(xL_t_values).numpy().flatten()
        # 해석적 해를 사용하여 x = L에서 시간에 따른 온도 계산
        T_analytical_at_L = self.analytical_solution_at_L(self.t_values.astype('float32')).numpy().flatten()
        # 그래프 그리기
        plt.plot(self.t_values, T_pred_at_L, 'r')
        plt.plot(self.t_values, T_analytical_at_L, 'k--')
        plt.xlim(mix, max)
        plt.ylim(50, 300)
        plt.tick_params(axis='both', which='major', direction='out', labelsize=14, top=False, right=False, bottom=True, left=True)        
        plt.xlabel('Time (s)')
        plt.ylabel('Temperature (K)')
        plt.title('Temperature Profile')
        plt.legend().remove()
        plt.tight_layout()
        plt.draw()

    def plot_temperature_distribution(self):
        xt_values, T_pred = self.predict_temperature()
        
        def custom_format2(x, pos):
            if x in [0, 0.008, 0.016, 0.024, 0.032, 0.040, 0.048]:
                return '{:.0f}'.format(x*1000)
            else:
                return ''

        fig, ax = plt.subplots(figsize=(5, 4))

        sc = ax.scatter(xt_values[:, 1], xt_values[:, 0], c=T_pred.flatten(), cmap='coolwarm', marker='o')
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label('Temperature (K)', rotation=-90, labelpad=-45)  # title 회전 및 labelpad 조정
        cbar.ax.tick_params(labelsize=14)
        cbar.ax.yaxis.set_label_position('left')  # label 위치 조정

        ax.set_xticks([
            0, 0.008, 0.016,
            0.024, 0.032, 0.040, 0.048
            ])
        ax.xaxis.set_major_formatter(FuncFormatter(custom_format2))
        ax.set_xlabel(r'$\it{x}$ (mm)')
        ax.set_ylabel(r'$\it{t}$ (s)')
        ax.set_xlim(0.040, 0.048)
        ax.set_ylim(0, 30)
        ax.tick_params(axis='both', which='both', direction='out', labelsize=14, top=False, right=False, bottom=True, left=True)

        plt.tight_layout()
        plt.show()

    def ScipyOptimizer(self, method='L-BFGS-B', **kwargs):
        loss_values = []
        plt.ion()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        def get_weight_tensor():
            weight_list = []
            shape_list = []

            for v in self.trainable_parameters:
                shape_list.append(v.shape)
                weight_list.extend(v.numpy().flatten())
            weight_list = tf.convert_to_tensor(weight_list)

            return weight_list, shape_list

        def set_weight_tensor(weight_list):
            idx = 0
            for v in self.trainable_parameters:
                vs = v.shape

                if len(vs) == 2:
                    sw = vs[0] * vs[1]
                    new_val = tf.reshape(weight_list[idx:idx + sw], (vs[0], vs[1]))
                    idx += sw
                elif len(vs) == 1:
                    new_val = weight_list[idx:idx + vs[0]]
                    idx += vs[0]
                elif len(vs) == 0:
                    new_val = weight_list[idx]
                    idx += 1
                elif len(vs) == 3:
                    sw = vs[0] * vs[1] * vs[2]
                    new_val = tf.reshape(weight_list[idx:idx + sw], (vs[0], vs[1], vs[2]))
                    idx += sw
                elif len(vs) == 4:
                    sw = vs[0] * vs[1] * vs[2] * vs[3]
                    new_val = tf.reshape(weight_list[idx:idx + sw], (vs[0], vs[1], vs[2], vs[3]))
                    idx += sw
                v.assign(tf.cast(new_val, self.DTYPE))

        def get_loss_and_grad(w):
            set_weight_tensor(w)
            with tf.GradientTape() as tape:
                total_loss, _, _, _, _ = self.compute_loss()
            grad = tape.gradient(total_loss, self.trainable_parameters)
            loss = total_loss.numpy().astype(np.float64)
            grad_flat = []
            for g in grad:
                grad_flat.extend(g.numpy().flatten())
            grad_flat = np.array(grad_flat, dtype=np.float64)

            # Plotting the loss and temperature distribution
            loss_values.append(loss)
            if self.lbfgs_step % 10 == 0:
                print(f'Thermal conductivity: {self.para[0].numpy()}')
                ax1.clear()
                ax1.semilogy(loss_values, label='Loss')
                ax1.set_xlim(0, right=self.lbfgs_step)
                ax1.tick_params(axis='both', which='major', direction='out', labelsize=14, top=False, right=False, bottom=True, left=True)
                ax1.set_title('L-BFGS-B')
                ax1.set_xlabel('Iteration')
                ax1.set_ylabel('Loss')
                ax1.legend().remove()

                xL_t_values = tf.convert_to_tensor([[t, self.L] for t in self.t_values], dtype=tf.float32)
                T_pred_at_L = self.model(xL_t_values).numpy().flatten()
                T_analytical_at_L = self.analytical_solution_at_L(self.t_values.astype('float32')).numpy().flatten()

                ax2.clear()
                ax2.plot(self.t_values, T_pred_at_L, label='Predicted Temperature', color='red')
                ax2.plot(self.t_values, T_analytical_at_L, label='Analytical Solution', linestyle='dashed', color='black')
                ax2.set_xlim(0, self.time_steps)
                ax2.set_ylim(50, 300)
                ax2.tick_params(axis='both', which='major', direction='out', labelsize=14, top=False, right=False, bottom=True, left=True)
                ax2.set_title('Temperature Profile')
                ax2.set_xlabel('Time (s)')
                ax2.set_ylabel('Temperature (K)')
                ax2.legend().remove()

                plt.tight_layout()
                plt.draw()
                plt.pause(0.1)

            self.lbfgs_step += 1
            return loss, grad_flat

        x0, _ = get_weight_tensor()

        result = scipy.optimize.minimize(fun=get_loss_and_grad,
                                         x0=x0,
                                         jac=True,
                                         method=method,
                                         **kwargs)

        set_weight_tensor(result.x)
        plt.ioff()

        return result

# 훈련 설정
target_time = 27.65   # 타겟 냉각 시간 (예: 30초)
target_temp = 77  # 타겟 온도 (예: 100K)
epochs = 400       # 훈련 에포크 수

# 모델 초기화
model = TransientThermalModel(
    k=0.8, d_b=0.009, Td=77, T_inf=300, L=0.048, 
    Tip_grid=0.001, Q_bias=0, sigma=5.67e-8, epsilon_2=0.02, 
    P_Torr=1e-4, alpha=1e-6, rho=2640, c=800, a_cooler = 0.039, b_cooler = -2,
    time_steps=50, target_time=target_time, target_temp=target_temp
)

# 모델 훈련
model.train(epochs=epochs)

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
model.plot_iteration(0,30)

# 최적화된 매개변수 출력
print(f"훈련 완료 후 매개변수 값:")
print(f"a_cooler: {model.a_cooler.numpy()}")

# x = L에서의 예측 온도
xL_t_values = tf.convert_to_tensor([[t, model.L] for t in model.t_values], dtype=tf.float32)
T_pred_at_L = model.model(xL_t_values).numpy().flatten()

# 77도 이하로 냉각된 시간을 찾기
cooling_time_index = np.argmax(T_pred_at_L <= 77)
cooling_time = model.t_values[cooling_time_index] if cooling_time_index > 0 else None

if cooling_time is not None:
    print(f"x = L에서 온도가 77K 이하로 냉각된 시간: {cooling_time:.2f} 초")
else:
    print("x = L에서 온도가 77K 이하로 냉각되지 않았습니다.")

# Show the plot
plt.show()