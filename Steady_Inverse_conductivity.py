import tensorflow as tf
import numpy as np
import pandas as pd
import scipy.optimize
import matplotlib.pyplot as plt

class ThermalModel:
    def __init__(self, d_b, Td, T_inf, L, Q_bias, sigma, epsilon_2, P_Torr, var_start, activation='tanh', initializer='glorot_normal'):
        self.d_b = d_b
        self.Td = Td
        self.T_b = tf.constant(T_inf, dtype=tf.float32)
        self.T_inf = T_inf
        self.L = L
        self.Q_bias = Q_bias
        self.sigma = sigma
        self.epsilon_2 = epsilon_2
        self.P_Torr = P_Torr
        self.h = self.calculate_h_gc() + 3*self.epsilon_2

        self.DTYPE = 'float32'

        self.Ac = np.pi * (tf.square(self.d_b / 2) - tf.square(self.d_b / 2 - 0.001))
        self.k = tf.Variable(var_start, dtype=tf.float32, trainable=True)  # Make k a trainable variable
        self.m2 = np.pi * self.d_b * self.h / (self.k * self.Ac)
        self.m = tf.sqrt(self.m2)

        self.x_values = np.linspace(0, self.L, 10000).reshape(-1, 1).astype(np.float32)
        self.x_tensor = tf.convert_to_tensor(self.x_values, dtype=tf.float32)
        self.x_values_train = np.sort(np.random.uniform(0, self.L, 10000)).reshape(-1, 1).astype(np.float32)
        self.x_tensor_train = tf.convert_to_tensor(self.x_values_train, dtype=tf.float32)

        self.activation = activation
        self.initializer = initializer
        self.model = self.build_model()
        self.trainable_parameters = self.model.trainable_variables + [self.k]  # Include k in trainable parameters
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01, amsgrad=True)

    def build_model(self):
        X_in = tf.keras.Input(shape=(1,))
        hiddens = tf.keras.layers.Lambda(lambda x: 2. * (x - 0.0) / (0.048 - 0.0) - 1.)(X_in)
        for _ in range(3):
            hiddens = tf.keras.layers.Dense(20, kernel_initializer=self.initializer)(hiddens)
            hiddens = tf.keras.layers.Activation(self.activation)(hiddens)
        prediction = tf.keras.layers.Dense(1)(hiddens)
        prediction = 0.5 * (prediction + 1.)
        prediction = (self.T_b - self.Td) * prediction
        prediction += self.Td
        model = tf.keras.Model(X_in, prediction)
        return model

    def calculate_h_gc(self):
        P_fm = 4e-4  # Pressure of free molecule flow in Torr
        P_Torr_conversion_factor = 133.322
        if self.P_Torr < P_fm:
            return 1.48 * self.P_Torr * P_Torr_conversion_factor
        elif P_fm <= self.P_Torr < 1:
            return (1.48 * self.P_Torr * P_Torr_conversion_factor) / (1 + 0.34 * self.P_Torr * P_Torr_conversion_factor)
        else:
            return 4.35

    def analytical_solution(self, x):
        m = tf.cast(self.m, tf.float32)
        L = tf.cast(self.L, tf.float32)
        return self.T_inf - ((self.T_inf - self.Td) * tf.sinh(m * x) / tf.sinh(m * L))

    def compute_loss(self, target_cooling_load):
        T, d2theta_dx2 = self.compute_derivatives()
        diff_eq_loss = tf.reduce_mean(tf.square(self.m2 * T - d2theta_dx2 - self.m2 * self.T_b))
        bc_loss_end = tf.reduce_mean(tf.square(self.model(tf.constant([[self.L]])) - self.Td))
        bc_loss_start = tf.reduce_mean(tf.square(self.model(tf.constant([[0.]])) - self.T_inf))
        cooling_load = self.calculate_cooling_load()
        cooling_load_loss = tf.reduce_mean(tf.square(cooling_load - target_cooling_load))
        total_loss = diff_eq_loss + 1e10*bc_loss_end + 1e10*bc_loss_start +cooling_load_loss
        return total_loss

    def compute_derivatives(self):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.x_tensor_train)
            T = self.model(self.x_tensor_train)
            theta = T - self.T_b
            dtheta_dx = tape.gradient(theta, self.x_tensor_train)
        d2theta_dx2 = tape.gradient(dtheta_dx, self.x_tensor_train)
        return T, d2theta_dx2

    def train(self, target_cooling_load, epochs):
        for epoch in range(epochs):
            loss = self.train_step(target_cooling_load)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss.numpy()}, k: {self.k.numpy()}")

    def train_step(self, target_cooling_load):
        with tf.GradientTape() as tape:
            total_loss = self.compute_loss(target_cooling_load)
        gradients = tape.gradient(total_loss, self.trainable_parameters)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_parameters))
        return total_loss

    def calculate_cooling_load(self):
        predicted_temp = self.model(self.x_tensor).numpy().flatten()
        T_0 = predicted_temp[0]
        T_L = predicted_temp[-1]
        cooling_load = -self.k * self.Ac * self.m * (T_L - T_0) * (np.cosh(self.m * self.L) / np.sinh(self.m * self.L))
        return cooling_load

    def ScipyOptimizer(self, target_cooling_load, method='L-BFGS-B', **kwargs):
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
                total_loss = self.compute_loss(target_cooling_load)
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

# Example usage
material = {"d_b": 0.009, "Td": 77, "T_inf": 300, "L": 0.048, "Q_bias": 0, "sigma": 5.67e-8, "epsilon_2": 0.02, "P_Torr": 1e0}
'''target_cooling_load = 0.25  # W

model = ThermalModel(**material)
model.train(target_cooling_load=target_cooling_load, epochs=200)
model.ScipyOptimizer(target_cooling_load=target_cooling_load, method='L-BFGS-B',
                     options={'maxiter': 4000,
                              'maxfun': 50000,
                              'maxcor': 50,
                              'maxls': 50,
                              'ftol': np.finfo(float).eps,
                              'gtol': np.finfo(float).eps,
                              'factr': np.finfo(float).eps,
                              'iprint': 50})
print(f"Optimized thermal conductivity (k): {model.k.numpy():.4f} W/(m·K)")'''

# List of target cooling loads
target_cooling_loads = [0.55]  # Example values in W

# Placeholder for optimized thermal conductivity values
optimized_k_values = {}

# Initialize the thermal model
model = ThermalModel(**material, var_start=1.)

# Iterate through each target cooling load
for target_cooling_load in target_cooling_loads:
    print(f"\nProcessing target cooling load: {target_cooling_load} W")

    # Train the model for the current target cooling load
    model.train(target_cooling_load=target_cooling_load, epochs=100)

    # Optimize the model using ScipyOptimizer
    model.ScipyOptimizer(
        target_cooling_load=target_cooling_load,
        method='L-BFGS-B',
        options={
            'maxiter': 100,
            'maxfun': 50000,
            'maxcor': 50,
            'maxls': 50,
            'ftol': np.finfo(float).eps,
            'gtol': np.finfo(float).eps,
            'factr': np.finfo(float).eps,
            'iprint': 50
        }
    )

    # Retrieve and store the optimized thermal conductivity (k)
    optimized_k = model.k.numpy()
    optimized_k_values[target_cooling_load] = optimized_k
    print(f"Optimized thermal conductivity (k) for {target_cooling_load} W: {optimized_k:.4f} W/(m·K)")

# Print all results
print("\nSummary of optimized thermal conductivity values:")
for load, k in optimized_k_values.items():
    print(f"Target cooling load: {load} W -> Optimized k: {k:.4f} W/(m·K)")
