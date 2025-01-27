import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

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

        self.x_values = np.linspace(0, self.L, 1000).reshape(-1, 1).astype(np.float32)
        self.x_tensor = tf.convert_to_tensor(self.x_values, dtype=tf.float32)
        self.x_values_train = np.sort(np.random.uniform(0, self.L, 1000)).reshape(-1, 1).astype(np.float32)
        self.x_tensor_train = tf.convert_to_tensor(self.x_values_train, dtype=tf.float32)

        self.activation = activation
        self.initializer = initializer
        self.model = self.build_model()
        self.trainable_parameters=self.model.trainable_variables
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

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
        #prediction = prediction * 1e2
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
            tape.watch(self.trainable_parameters)
            total_loss = self.compute_loss()
        g = tape.gradient(total_loss, self.trainable_parameters)
        del tape
        return g, total_loss 

    def compute_loss(self):
        T, d2theta_dx2 = self.compute_derivatives()
        diff_eq_loss = tf.reduce_mean(tf.square(self.m2 * T - d2theta_dx2 - self.m2 * self.T_b))
        bc_loss_end = tf.reduce_mean(tf.square(self.model(tf.constant([[self.L]])) - self.Td))
        bc_loss_start = tf.reduce_mean(tf.square(self.model(tf.constant([[0.]])) - self.T_inf))
        total_loss = diff_eq_loss + self.m2 *self.m2 * bc_loss_end + self.m2 *self.m2 * bc_loss_start
        return total_loss

    def train_step(self):
        gradients, total_loss = self.get_grad()
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return total_loss

    def train(self, epochs):
        loss_values = []
        plt.ion()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        for epoch in range(epochs):
            loss = self.train_step()
            loss_values.append(loss.numpy())

            if epoch % 10 == 0:
                ax1.clear()
                ax1.semilogy(loss_values, label='Loss')
                ax1.set_title('Epoch: {} Loss: {:.4f}'.format(epoch, loss.numpy()))
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('Loss')
                ax1.legend()

                analytical_temp = self.analytical_solution(self.x_tensor)
                predicted_temp = self.model(self.x_tensor)

                ax2.clear()
                ax2.plot(self.x_values.flatten(), analytical_temp.numpy().flatten(), label='Analytical Solution', linestyle='dashed', color='black')
                ax2.plot(self.x_values.flatten(), predicted_temp.numpy().flatten(), label='Predicted Temperature', marker='', color='red')
                ax2.set_title('Temperature Distribution')
                ax2.set_xlabel('Position along the cold well (m)')
                ax2.set_ylabel('Temperature (K)')
                ax2.set_xlim(0, 0.048)
                ax2.set_ylim(50, 300)
                ax2.legend()

                plt.tight_layout()

                plt.draw()
                plt.pause(0.1)

        plt.ioff()

    def ScipyOptimizer(self, method='L-BFGS-B', **kwargs):
        def get_weight_tensor():
            weight_list = []
            shape_list = []
            
            for v in self.trainable_parameters:
                shape_list.append(v.shape)
                weight_list.extend(v.numpy().flatten())
            weight_list = tf.convert_to_tensor(weight_list)
            
            return weight_list, shape_list

        x0, shape_list = get_weight_tensor()

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
            grad, loss = self.get_grad()
            loss = loss.numpy().astype(np.float64)
            grad_flat = []
            for g in grad:
                grad_flat.extend(g.numpy().flatten())
            grad_flat = np.array(grad_flat, dtype=np.float64)
            self.loss = loss
            return loss, grad_flat

        return scipy.optimize.minimize(fun=get_loss_and_grad,
                                       x0=x0,
                                       jac=True,
                                       method=method,
                                       **kwargs)

    def predict_temperature(self):
        T_pred = self.model.predict(self.x_tensor)
        return self.x_values, T_pred

    def plot_temperature_distribution(self):
        x_values, T_pred = self.predict_temperature()
        analytical_temp = self.analytical_solution(self.x_tensor)
        
        plt.figure(figsize=(6, 6))
        plt.plot(x_values.flatten(), T_pred.flatten(), label='Predicted Temperature', marker='', color='red')
        plt.plot(x_values.flatten(), analytical_temp, 'k--')
        plt.xlabel('Position along the cold well (m)')
        plt.ylabel('Temperature (K)')
        plt.title(f'Temperature Distribution along the Cold Well / {self.P_Torr:.0e} Torr.')
        plt.legend()
        plt.xlim(0, 0.048)
        plt.ylim(50, 300)
        plt.grid(False)
        plt.tight_layout()
        plt.show()
