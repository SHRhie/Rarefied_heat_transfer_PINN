import tensorflow as tf
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import scipy.optimize
from scipy.special import erfc
import pickle

def custom_format2(x, pos):
            if x in [0, 0.008, 0.016, 0.024, 0.032, 0.040, 0.048]:
                return '{:.0f}'.format(x*1000)
            else:
                return ''

mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['mathtext.fontset'] = 'stix'  # 수학 표현에 대한 글꼴 설정
mpl.rcParams['font.size'] = 20  # 기본 글꼴 크기 설정
mpl.rcParams['figure.dpi'] = 400

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

class ThermalModel:
    def __init__(self, k, d_b, Td, T_inf, L, Q_bias, sigma, epsilon_2, P_Torr, activation='tanh', initializer='glorot_normal'):
        self.k = k
        self.d_b = d_b
        self.Td = Td
        self.T_b = tf.constant(T_inf, dtype=tf.float32)
        self.T_inf = T_inf
        self.L = L
        self.Q_bias = Q_bias
        self.sigma = sigma
        self.epsilon_2 = epsilon_2
        self.P_Torr = P_Torr
        self.h = self.calculate_h_gc()
        
        self.DTYPE='float32'
        
        self.Ac = np.pi * (tf.square(self.d_b / 2) - tf.square(self.d_b / 2 - 0.001))
        self.m2 = np.pi * self.d_b * self.h / (self.k * self.Ac)
        self.m = np.sqrt(self.m2)

        self.x_values = np.linspace(0, self.L, 10000).reshape(-1, 1).astype(np.float32)
        self.x_tensor = tf.convert_to_tensor(self.x_values, dtype=tf.float32)
        self.x_values_train = np.sort(np.random.uniform(0, self.L, 10000)).reshape(-1, 1).astype(np.float32)
        self.x_tensor_train = tf.convert_to_tensor(self.x_values_train, dtype=tf.float32)

        self.activation = activation
        self.initializer = initializer
        self.model = self.build_model()
        self.trainable_parameters=self.model.trainable_variables
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01, amsgrad=True)

    def build_model(self):
        X_in = tf.keras.Input(shape=(1,))
        hiddens = tf.keras.layers.Lambda(lambda x: 2.*(x - 0.0) / (0.048 - 0.0) - 1.)(X_in)  # 입력 데이터 전처리
        for _ in range(3):
            hiddens = tf.keras.layers.Dense(20, kernel_initializer=self.initializer)(hiddens)
            hiddens = tf.keras.layers.Activation(self.activation)(hiddens)
        prediction = tf.keras.layers.Dense(1)(hiddens)
        prediction = 0.5*(prediction +1.)
        prediction = (self.T_b-self.Td) * prediction
        prediction += self.Td
        model = tf.keras.Model(X_in, prediction)
        return model

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

    def analytical_solution(self, x):
        m = tf.cast(self.m, tf.float32)
        L = tf.cast(self.L, tf.float32)
        return self.T_inf - ((self.T_inf - self.Td) * tf.sinh(m * x) / tf.sinh(m * L))

    def compute_derivatives(self):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.x_tensor_train)
            T = self.model(self.x_tensor_train)
            theta = T - self.T_b
            dtheta_dx = tape.gradient(theta, self.x_tensor_train)
        d2theta_dx2 = tape.gradient(dtheta_dx, self.x_tensor_train)
        return T, d2theta_dx2

    def get_grad(self):
        with tf.GradientTape() as tape:
            total_loss = self.compute_loss()
        g = tape.gradient(total_loss, self.trainable_parameters)
        del tape
        return g, total_loss 

    def compute_loss(self):
        T, d2theta_dx2 = self.compute_derivatives()
        diff_eq_loss = tf.reduce_mean(tf.square(self.m2 * T - d2theta_dx2 - self.m2 * self.T_b))
        bc_loss_end = tf.reduce_mean(tf.square(self.model(tf.constant([[self.L]])) - self.Td))
        bc_loss_start = tf.reduce_mean(tf.square(self.model(tf.constant([[0.]])) - self.T_inf))
        total_loss = diff_eq_loss + self.m2 * self.m2 * bc_loss_end + self.m2 * self.m2 * bc_loss_start
        return total_loss

    def train_step(self):
        gradients, total_loss = self.get_grad()
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return total_loss

    def train(self, epochs):
        self.adam_loss_values = []
        self.predicted_temp = []
        for epoch in range(epochs):
            loss = self.train_step()
            self.adam_loss_values.append(loss.numpy())

            if epoch % 10 == 0:
                print(f'Epoch: {epoch}, Loss: {loss.numpy()}')

        self.predicted_temp = self.model(self.x_tensor).numpy()

    def ScipyOptimizer(self, method='L-BFGS-B', **kwargs):
        self.lbfgs_loss_values = []
        self.lbfgs_step = 0

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
                total_loss = self.compute_loss()
            grad = tape.gradient(total_loss, self.trainable_parameters)
            loss = total_loss.numpy().astype(np.float64) 
            grad_flat = []
            for g in grad:
                grad_flat.extend(g.numpy().flatten())
            grad_flat = np.array(grad_flat, dtype=np.float64)

            self.lbfgs_loss_values.append(loss)
            if self.lbfgs_step % 10 == 0:
                print(f'L-BFGS-B Step: {self.lbfgs_step}, Loss: {loss}')

            self.lbfgs_step += 1
            return loss, grad_flat

        x0, _ = get_weight_tensor()

        result = scipy.optimize.minimize(fun=get_loss_and_grad,
                                         x0=x0,
                                         jac=True,
                                         method=method,
                                         **kwargs)

        set_weight_tensor(result.x)
        self.predicted_temp = self.model(self.x_tensor).numpy()
        return result

class TransientThermalModel:
    def __init__(self, k, d_b, Td, T_inf, L, Q_bias, sigma, epsilon_2, P_Torr, alpha, rho, c, a_cooler, b_cooler, time_steps, activation='tanh', initializer='glorot_normal'):
        self.k = k
        self.d_b = d_b
        self.Td = Td
        self.T_inf = T_inf
        self.L = L
        self.Q_bias = Q_bias
        self.sigma = sigma
        self.epsilon_2 = epsilon_2
        self.P_Torr = P_Torr
        self.alpha = k / (rho * c)
        self.rho = rho
        self.c = c
        self.h = self.calculate_h_gc()
        self.a_cooler = a_cooler
        self.b_cooler = b_cooler
        self.time_steps = time_steps
        self.h_gc = self.calculate_h_gc()
        self.lbfgs_step = 0
        
        self.lb = tf.constant([0.0, 0.0], dtype=tf.float32)
        self.ub = tf.constant([self.time_steps, self.L], dtype=tf.float32)
        self.DTYPE = 'float32'
        
        self.Ac = np.pi * (tf.square(self.d_b / 2) - tf.square(self.d_b / 2 - 0.001))
        self.m2 = np.pi * self.d_b * self.h_gc / (self.k * self.Ac)
        self.m = tf.sqrt(self.m2)
        
        self.delta_V = self.Ac * self.L * 0.02 / 10000  # 볼륨 요소 크기 (적절히 조정)

        self.t_values = np.concatenate((np.linspace(0, 0.2 * time_steps, num=100, endpoint=False),
                                        np.linspace(0.2 * time_steps, 0.5 * time_steps, num=100, endpoint=False),
                                        np.linspace(0.5 * time_steps, time_steps, num=100+1)))
        self.x_values = np.concatenate((np.linspace(0, 0.98 * self.L, num=500, endpoint=False),
                                        np.linspace(0.98 * self.L, self.L, num=1000+1))).reshape(-1, 1).astype(np.float32)
        self.xt_values = np.array([[t, x[0]] for x in self.x_values for t in self.t_values], dtype=np.float32)
        self.xt_tensor = tf.convert_to_tensor(self.xt_values, dtype=tf.float32)

        self.X_0, self.X_b_0, self.X_b_L, self.X_r = self.data_sampling()

        self.activation = activation
        self.initializer = initializer
        self.model = self.build_model()
        self.trainable_parameters = self.model.trainable_variables
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
        X_r = get_Xr(self.lb, self.ub, 100000)
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

    def get_grad(self):
        with tf.GradientTape(persistent=True) as tape:
            total_loss, pde_loss, initial_condition, boundary_condition_x0, boundary_condition_cooler = self.compute_loss()
        gradients = tape.gradient(total_loss, self.trainable_parameters)
        return gradients, total_loss, pde_loss, initial_condition, boundary_condition_x0, boundary_condition_cooler

    def compute_loss(self):
        T, dT_dt, d2T_dx2, dT_dx = self.compute_derivatives()
        pde_loss = tf.reduce_mean(tf.square(dT_dt - self.alpha * d2T_dx2))
        initial_condition = tf.reduce_mean(tf.square(self.model(self.X_0) - self.T_inf))
        boundary_condition_x0 = tf.reduce_mean(tf.square(self.model(self.X_b_0) - self.T_inf))

        T_b, dT_dx_b = self.get_b_L()
        boundary_condition_cooler = self.k * self.Ac * dT_dx_b + (self.a_cooler * T_b + self.b_cooler)
        boundary_condition_cooler = tf.reduce_mean(tf.square(boundary_condition_cooler))

        total_loss = self.m2 * pde_loss + initial_condition + boundary_condition_x0 + self.m2 * boundary_condition_cooler
        return total_loss, pde_loss, initial_condition, boundary_condition_x0, boundary_condition_cooler

    def train_step(self):
        gradients, total_loss, pde_loss, initial_condition, boundary_condition_x0, boundary_condition_cooler = self.get_grad()
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return total_loss

    def train(self, epochs):
        self.adam_loss_values = []
        
        for epoch in range(epochs):
            loss = self.train_step()
            self.adam_loss_values.append(loss.numpy())

            if epoch % 10 == 0:
                print(f'Epoch: {epoch}, Loss: {loss.numpy()}')

    def ScipyOptimizer(self, method='L-BFGS-B', **kwargs):
        self.lbfgs_loss_values = []
        self.lbfgs_step = 0

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

            self.lbfgs_loss_values.append(loss)
            if self.lbfgs_step % 10 == 0:
                print(f'L-BFGS-B Step: {self.lbfgs_step}, Loss: {loss}')

            self.lbfgs_step += 1
            return loss, grad_flat

        x0, _ = get_weight_tensor()

        result = scipy.optimize.minimize(fun=get_loss_and_grad,
                                         x0=x0,
                                         jac=True,
                                         method=method,
                                         **kwargs)

        set_weight_tensor(result.x)
        return result

    def predict_temperature(self):
        T_pred = self.model.predict(self.xt_tensor)
        return self.xt_values, T_pred

    def plot_results(self):
        # Plotting the results with steady-state on the first row and transient on the second row
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))

        # Combined Loss Plot for Steady-State Thermal Model
        combined_steady_loss = self.steady_adam_loss_values + self.steady_lbfgs_loss_values
        transition_index = len(self.steady_adam_loss_values)
        axs[0, 0].semilogy(range(transition_index), self.steady_adam_loss_values, label='Adam Loss (Steady)', color='black')
        axs[0, 0].semilogy(range(transition_index, transition_index + len(self.steady_lbfgs_loss_values)),
                           self.steady_lbfgs_loss_values, label='L-BFGS-B Loss (Steady)', color='black')
        axs[0, 0].axvline(x=transition_index, color='black', linestyle='--', label='Transition')
        #axs[0, 0].set_title('Loss (Steady)')
        axs[0, 0].set_xlabel('Iteration')
        axs[0, 0].set_ylabel('Loss')
        axs[0, 0].set_xlim(0, len(combined_steady_loss))
        axs[0, 0].tick_params(axis='both', which='major', labelsize=14, bottom=True, left=True)
        axs[0, 0].tick_params(axis='both', which='minor', labelsize=14, bottom=False, left=False)
        axs[0, 0].legend().remove()

        # Temperature Profile at x for Steady-State Thermal Model
        axs[0, 1].plot(self.steady_x_values.flatten(), self.steady_predicted_temp.flatten(), label='Predicted Temperature (Steady)', color='red')
        axs[0, 1].plot(self.steady_x_values.flatten(), self.steady_analytical_temp.flatten(), 'k--')
        axs[0, 1].set_xticks([
            0, 0.008, 0.016,
            0.024, 0.032, 0.040, 0.048
            ])
        axs[0, 1].xaxis.set_major_formatter(FuncFormatter(custom_format2))
        axs[0, 1].set_xlabel(r'$\it{x}$ (mm)')
        axs[0, 1].set_ylabel('Temperature (K)')
        axs[0, 1].set_xlim(0, 0.048)
        axs[0, 1].set_ylim(50, 300)
        axs[0, 1].set_title('$\it{T(x)}$')
        axs[0, 1].tick_params(axis='both', which='major', labelsize=14, bottom=True, left=True)
        axs[0, 1].tick_params(axis='both', which='minor', labelsize=14, bottom=False, left=False)
        axs[0, 1].legend().remove()

        # Combined Loss Plot for Transient Thermal Model
        combined_transient_loss = self.adam_loss_values + self.lbfgs_loss_values
        transition_index = len(self.adam_loss_values)
        axs[1, 0].semilogy(range(transition_index), self.adam_loss_values, label='Adam Loss (Transient)', color='black')
        axs[1, 0].semilogy(range(transition_index, transition_index + len(self.lbfgs_loss_values)),
                           self.lbfgs_loss_values, label='L-BFGS-B Loss (Transient)', color='black')
        axs[1, 0].axvline(x=transition_index, color='black', linestyle='--', label='Transition')
        #axs[1, 0].set_title('Loss (Transient)')
        axs[1, 0].set_xlabel('Iteration')
        axs[1, 0].set_ylabel('Loss')
        axs[1, 0].set_xlim(0, len(combined_transient_loss))
        axs[1, 0].tick_params(axis='both', which='major', labelsize=14, bottom=True, left=True)
        axs[1, 0].tick_params(axis='both', which='minor', labelsize=14, bottom=False, left=False)
        axs[1, 0].legend().remove()

        # Temperature Profile at L for Transient Thermal Model
        xL_t_values = tf.convert_to_tensor([[t, self.L] for t in self.t_values], dtype=tf.float32)
        T_pred_at_L = self.model(xL_t_values).numpy().flatten()
        T_analytical_at_L = self.analytical_solution_at_L(self.t_values.astype('float32')).numpy().flatten()

        axs[1, 1].plot(self.t_values, T_pred_at_L, label='Predicted Temperature (Transient)', color='red')
        axs[1, 1].plot(self.t_values, T_analytical_at_L, label='Analytical Solution (Transient)', linestyle='dashed', color='black')
        axs[1, 1].set_xlim(0, self.time_steps)
        axs[1, 1].set_ylim(50, 300)
        axs[1, 1].set_title(r'$\it{T(x=L,t)}$')
        axs[1, 1].set_xlabel('$\it{t}$ (s)')
        axs[1, 1].set_ylabel('Temperature (K)')
        axs[1, 1].tick_params(axis='both', which='major', labelsize=14, bottom=True, left=True)
        axs[1, 1].tick_params(axis='both', which='minor', labelsize=14, bottom=False, left=False)
        axs[1, 1].legend().remove()

        for ax in axs.flat:
            ax.tick_params(axis='both', which='major', direction='out', labelsize=14, top=False, right=False, bottom=True, left=True)

        plt.tight_layout()
        plt.show()
        plt.savefig('results_combined.png')

    def save_results(self, filename):
        results = {
            'adam_loss_values': self.adam_loss_values,
            'lbfgs_loss_values': self.lbfgs_loss_values,
            'steady_adam_loss_values': self.steady_adam_loss_values,
            'steady_lbfgs_loss_values': self.steady_lbfgs_loss_values,
            'steady_predicted_temp': self.steady_predicted_temp,
            'steady_analytical_temp': self.steady_analytical_temp,
            't_values': self.t_values,
            'x_values': self.x_values,
        }
        with open(filename, 'wb') as f:
            pickle.dump(results, f)

    def load_results(self, filename):
        with open(filename, 'rb') as f:
            results = pickle.load(f)
        self.adam_loss_values = results['adam_loss_values']
        self.lbfgs_loss_values = results['lbfgs_loss_values']
        self.steady_adam_loss_values = results['steady_adam_loss_values']
        self.steady_lbfgs_loss_values = results['steady_lbfgs_loss_values']
        self.steady_predicted_temp = results['steady_predicted_temp']
        self.steady_analytical_temp = results['steady_analytical_temp']
        self.t_values = results['t_values']
        self.x_values = results['x_values']

# Helper functions
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

# Example usage for Steady-State Thermal Model
steady_model = ThermalModel(k=0.8, d_b=0.009, Td=77, T_inf=300, L=0.048, Q_bias=0, sigma=5.67e-8, epsilon_2=0.02, P_Torr=1e0)
steady_model.train(epochs=200)
steady_model.ScipyOptimizer(method='L-BFGS-B',
                     options={'maxiter': 4000,
                              'maxfun': 50000,
                              'maxcor': 50,
                              'maxls': 50,
                              'ftol': np.finfo(float).eps,
                              'gtol': np.finfo(float).eps,
                              'factr': np.finfo(float).eps,
                              'iprint': 50})

def calculate_l1_l2_errors(prediction, exact):
    l1_absolute = np.mean(np.abs(prediction - exact))
    l2_relative = np.linalg.norm(prediction - exact, 2) / np.linalg.norm(exact, 2)
    return l1_absolute, l2_relative

# Saving steady-state results
steady_model.steady_adam_loss_values = steady_model.adam_loss_values
steady_model.steady_lbfgs_loss_values = steady_model.lbfgs_loss_values
steady_model.steady_predicted_temp = steady_model.predicted_temp
steady_model.steady_analytical_temp = steady_model.analytical_solution(steady_model.x_tensor).numpy()

# Example usage for Transient Thermal Model
transient_model = TransientThermalModel(
    k=0.8, d_b=0.009, Td=77, T_inf=300, L=0.048, Q_bias=0, sigma=5.67e-8, epsilon_2=0.02, P_Torr=1e-4,
    alpha=1e-6, rho=2640, c=800, a_cooler=0.039, b_cooler=-2, time_steps=30
)
transient_model.train(epochs=400)  # Train with Adam
transient_model.ScipyOptimizer(method='L-BFGS-B',
                     options={'maxiter': 4000,
                              'maxfun': 50000,
                              'maxcor': 50,
                              'maxls': 50,
                              'ftol': np.finfo(float).eps,
                              'gtol': np.finfo(float).eps,
                              'factr': np.finfo(float).eps,
                              'iprint': 50})  # Fine-tune with L-BFGS-B

# Combine results
transient_model.steady_adam_loss_values = steady_model.steady_adam_loss_values
transient_model.steady_lbfgs_loss_values = steady_model.steady_lbfgs_loss_values
transient_model.steady_predicted_temp = steady_model.steady_predicted_temp
transient_model.steady_analytical_temp = steady_model.steady_analytical_temp
transient_model.steady_x_values = steady_model.x_values

# Save the combined results
#transient_model.save_results('combined_results.pkl')

# Calculate L1 and L2 errors for the steady-state model
steady_l1, steady_l2 = calculate_l1_l2_errors(steady_model.steady_predicted_temp, steady_model.steady_analytical_temp)
print(f'Steady-State Model L1 Error: {steady_l1:.6f}, L2 Error: {steady_l2:.6f}')

# Calculate L1 and L2 errors for the transient model at x = L
T_pred_at_L = transient_model.model(tf.convert_to_tensor([[t, transient_model.L] for t in transient_model.t_values], dtype=tf.float32)).numpy().flatten()
T_analytical_at_L = transient_model.analytical_solution_at_L(transient_model.t_values.astype('float32')).numpy().flatten()

transient_l1, transient_l2 = calculate_l1_l2_errors(T_pred_at_L, T_analytical_at_L)
print(f'Transient Model L1 Error: {transient_l1:.6f}, L2 Error: {transient_l2:.6f}')


# Plot all results in a single figure
transient_model.plot_results()
