from pr_bc_csv import flow_data
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
from time import time
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

class Xavier_glorot(tf.keras.initializers.Initializer):
    def __init__(self,inputs=5,outputs=5):
        self.minval = -np.sqrt(6/(inputs+outputs))
    
        self.maxval = np.sqrt(6/(inputs+outputs))
    
    def __call__(self,shape, dtype=None):
    
        return tf.random.uniform(shape, minval=self.minval,maxval=self.maxval,dtype=tf.float32)


class IncompressibleNet(Model):

    def __init__(self):
        super(IncompressibleNet, self).__init__()

        initializer = tf.keras.initializers.GlorotUniform
        # initializer = Xavier_glorot 
        self.h1 = Dense(50, activation='tanh', kernel_initializer=initializer)
        self.h2 = Dense(50, activation='tanh', kernel_initializer=initializer)
        self.h3 = Dense(50, activation='tanh', kernel_initializer=initializer)
        self.h4 = Dense(50, activation='tanh', kernel_initializer=initializer)
        self.h5 = Dense(50, activation='tanh', kernel_initializer=initializer)
        self.u = Dense(3, activation='linear', kernel_initializer=initializer)


    def call(self, pos):
        x = self.h1(pos)  # pos = (x, y)
        x = self.h2(x)
        x = self.h3(x)
        x = self.h4(x)
        x = self.h5(x)
        out = self.u(x)  # u,v,p,sig_xx,sig_xy,sig_yy

        u = out[:, 0:1]
        v = out[:, 1:2]
        p = out[:, 2:3]
    
        return u, v, p


class NSpinn(object):

    def __init__(self):

        self.lr = 0.001
        self.opt = Adam(self.lr)

        # density and viscosity
        self.rho = 1.0
        self.mu = 0.01

        self.flow = IncompressibleNet()
        self.flow.build(input_shape=(None, 2))

        self.train_loss_history = []
        self.iter_count = 0
        self.instant_loss = 0

    def ns_net(self, box):
        x = box[:, 0:1]
        y = box[:, 1:2]

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            tape.watch(y)
            xy_c = tf.concat([x,y], axis=1)
            u, v, p = self.flow(xy_c)
            u_x = tape.gradient(u, x)
            u_y = tape.gradient(u, y)
            v_x = tape.gradient(v, x)
            v_y = tape.gradient(v, y)
            p_x = tape.gradient(p,x)
            p_y = tape.gradient(p,y)

        u_xx = tape.gradient(u_x, x)
        u_yy = tape.gradient(u_y, y)
        v_xx = tape.gradient(v_x, x)
        v_yy = tape.gradient(v_y, y)
        del tape

        r_1 = self.rho * (u*u_x+v*u_y) + p_x - self.mu * (u_xx + u_yy)  ## (u*u_x+v*u_y)는 가속도항, p는 압력항, 중력항은 고려하지 않음, (u_xx + u_yy)는 점성력항
        r_2 = self.rho * (u*v_x+v*v_y) + p_y - self.mu * (v_xx + v_yy)  ## 오일러 방정식은 점성력을 고려하지 않음, NSE는 점성력까지 고려
        r_3 = u_x + v_y ## 질량 보존 법칙을 만족 시키기 위하여 사용

        return r_1, r_2, r_3


    def compute_loss(self,r_1, r_2, r_3,  \
                        left_u_hat, left_v_hat, left_u_sol, left_v_sol, \
                        walls_u_hat, walls_v_hat, walls_u_sol, walls_v_sol, \
                        cylin_u_hat, cylin_v_hat, cylin_u_sol, cylin_v_sol, \
                        right_v_hat, right_p_hat, outlet_p_sol, outlet_v_sol):
        loss_bnd = tf.reduce_mean(tf.square(left_u_hat-left_u_sol))
        loss_bnd1 = tf.reduce_mean(tf.square(left_v_hat-left_v_sol))
        loss_bnd2 = tf.reduce_mean(tf.square(walls_u_hat-walls_u_sol))
        loss_bnd3 = tf.reduce_mean(tf.square(walls_v_hat-walls_v_sol))
        loss_bnd4 = tf.reduce_mean(tf.square(cylin_u_hat-cylin_u_sol))
        loss_bnd5 = tf.reduce_mean(tf.square(cylin_v_hat-cylin_v_sol))
        loss_bnd6 = tf.reduce_mean(tf.square(right_v_hat-outlet_v_sol))
        loss_outlet = tf.reduce_mean(tf.square(right_p_hat-outlet_p_sol))
        loss_col = tf.reduce_mean(tf.square(r_1)) 
        loss_col1 = tf.reduce_mean(tf.square(r_2))
        loss_col2 = tf.reduce_mean(tf.square(r_3)) 

        return loss_bnd+loss_outlet+loss_col+loss_col1+loss_col2+loss_bnd1+loss_bnd2+loss_bnd3+loss_bnd4+loss_bnd5+loss_bnd6


    def save_weights(self, path):
        self.flow.save_weights(path + 'flow.h5')


    def load_weights(self, path):
        self.flow.load_weights(path + 'flow.h5')


    def compute_grad(self, box, left_wall, right_wall, walls, cylin, left_u_sol, left_v_sol, walls_u_sol, walls_v_sol, cylin_u_sol, cylin_v_sol, outlet_p_sol, outlet_v_sol):
        with tf.GradientTape() as tape:
            r_1, r_2, r_3 = self.ns_net(box)
            left_u_hat, left_v_hat, _  = self.flow(left_wall)
            _, right_v_hat, right_p_hat = self.flow(right_wall)
            walls_u_hat, walls_v_hat, _ = self.flow(walls)
            cylin_u_hat, cylin_v_hat, _ = self.flow(cylin)
            loss = self.compute_loss(r_1, r_2, r_3, \
                                     left_u_hat, left_v_hat, left_u_sol, left_v_sol, \
                                    walls_u_hat, walls_v_hat, walls_u_sol, walls_v_sol, \
                                    cylin_u_hat, cylin_v_hat, cylin_u_sol, cylin_v_sol, \
                                    right_v_hat, right_p_hat, outlet_p_sol, outlet_v_sol)

        grads = tape.gradient(loss, self.flow.trainable_variables)

        return loss, grads


    def callback(self, arg=None):
        if self.iter_count % 10 == 0:
            print('iter=', self.iter_count, ', loss=', self.instant_loss)
            self.train_loss_history.append([self.iter_count, self.instant_loss])
        self.iter_count += 1



    def train_with_adam(self,box, left_wall, right_wall, walls, cylin, left_u_sol, left_v_sol, walls_u_sol, walls_v_sol, cylin_u_sol, cylin_v_sol, outlet_p_sol, outlet_v_sol, adam_num):

        @tf.function
        def learn():
            loss, grads = self.compute_grad(box, left_wall, right_wall, walls, cylin, left_u_sol, left_v_sol, walls_u_sol, walls_v_sol, cylin_u_sol, cylin_v_sol, outlet_p_sol, outlet_v_sol)
            self.opt.apply_gradients(zip(grads, self.flow.trainable_variables))

            return loss

        for iter in range(int(adam_num)):

            loss = learn()

            self.instant_loss = loss.numpy()
            self.callback()


    def train_with_lbfgs(self, box, left_wall, right_wall, walls, cylin, left_u_sol, left_v_sol, walls_u_sol, walls_v_sol, cylin_u_sol, cylin_v_sol, outlet_p_sol, outlet_v_sol, lbfgs_num):

        def vec_weight():
            # vectorize weights
            weight_vec = []

            # Loop over all weights
            for v in self.flow.trainable_variables:
                weight_vec.extend(v.numpy().flatten())

            weight_vec = tf.convert_to_tensor(weight_vec)
            return weight_vec
        w0 = vec_weight().numpy()

        def restore_weight(weight_vec):
            # restore weight vector to model weights
            idx = 0
            for v in self.flow.trainable_variables:
                vs = v.shape

                # weight matrices
                if len(vs) == 2:
                    sw = vs[0] * vs[1]
                    updated_val = tf.reshape(weight_vec[idx:idx + sw], (vs[0], vs[1]))
                    idx += sw

                # bias vectors
                elif len(vs) == 1:
                    updated_val = weight_vec[idx:idx + vs[0]]
                    idx += vs[0]

                # assign variables (Casting necessary since scipy requires float64 type)
                v.assign(tf.cast(updated_val, dtype=tf.float32))


        def loss_grad(w):
            # update weights in model
            restore_weight(w)

            loss, grads = self.compute_grad(box, left_wall, right_wall, walls, cylin, left_u_sol, left_v_sol, walls_u_sol, walls_v_sol, cylin_u_sol, cylin_v_sol, outlet_p_sol, outlet_v_sol)
            # vectorize gradients
            grad_vec = []
            for g in grads:
                grad_vec.extend(g.numpy().flatten())

            # gradient list to array
            # scipy-routines requires 64-bit floats
            loss = loss.numpy().astype(np.float64)
            self.instant_loss = loss
            grad_vec = np.array(grad_vec, dtype=np.float64)

            return loss, grad_vec

        return scipy.optimize.minimize(fun=loss_grad,
                                       x0=w0,
                                       jac=True,
                                       method='L-BFGS-B',
                                       callback=self.callback,
                                       options={'maxiter': lbfgs_num,
                                                'maxfun': 50000,
                                                'maxcor': 50,
                                                'maxls': 50,
                                                'ftol': 1.0 * np.finfo(float).eps})



    def predict(self, xy):
        u, v, p = self.flow(xy)
        return u, v, p


    def train(self,  adam_num,lbfgs_num):

        box, left_wall, right_wall, walls, cylin, left_u_sol, left_v_sol, walls_u_sol, walls_v_sol, cylin_u_sol, cylin_v_sol, outlet_p_sol, outlet_v_sol = flow_data()

        # Start timer
        t0 = time()
        self.train_with_adam(box, left_wall, right_wall, walls, cylin, left_u_sol, left_v_sol, walls_u_sol, walls_v_sol, cylin_u_sol, cylin_v_sol, outlet_p_sol, outlet_v_sol, adam_num)
        # Print computation time
        print('\nComputation time of adam: {} seconds'.format(time() - t0))
        t1 = time()
        self.train_with_lbfgs(box, left_wall, right_wall, walls, cylin, left_u_sol, left_v_sol, walls_u_sol, walls_v_sol, cylin_u_sol, cylin_v_sol, outlet_p_sol, outlet_v_sol, lbfgs_num)
        # Print computation time
        print('\nComputation time of L-BFGS-B: {} seconds'.format(time() - t1))

        self.save_weights("/home/hrg5763/Documents/tistory_nse_steady_2/")

        np.savetxt('/home/hrg5763/Documents/tistory_nse_steady_2/loss.txt', self.train_loss_history)
        train_loss_history = np.array(self.train_loss_history)


        plt.plot(train_loss_history[:, 0], train_loss_history[:, 1])
        plt.yscale("log")
        plt.show()


# main
def main():



    adam_num = 50001
    lbfgs_num = 100000
    agent = NSpinn()

    agent.train( adam_num,lbfgs_num)


if __name__=="__main__":
    main()