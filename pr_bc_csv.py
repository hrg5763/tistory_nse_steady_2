import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def flow_data():

    path = '/home/hrg5763/Documents/tistory_nse_steady_2/csv/'
    box = pd.read_csv(path + 'full.csv')
    left_wall = pd.read_csv(path + 'left_wall.csv')
    right_wall = pd.read_csv(path + 'right_wall.csv')
    walls = pd.read_csv(path + 'walls.csv')
    cylin = pd.read_csv(path + 'cylin.csv')

    # collocation points
    box_x = box['Points:0']
    box_x = pd.Series.to_numpy(box_x)
    box_x = box_x.reshape(-1,1)

    box_y = box['Points:1']
    box_y = pd.Series.to_numpy(box_y)
    box_y = box_y.reshape(-1,1)

    box = np.concatenate([box_x,box_y], axis=1)

    left_wall_x = left_wall['Points:0']
    left_wall_x = pd.Series.to_numpy(left_wall_x)
    left_wall_x = left_wall_x.reshape(-1,1)

    left_wall_y = left_wall['Points:1']
    left_wall_y = pd.Series.to_numpy(left_wall_y)
    left_wall_y = left_wall_y.reshape(-1,1)

    left_wall = np.concatenate([left_wall_x,left_wall_y], axis=1)

    right_wall_x = right_wall['Points:0']
    right_wall_x = pd.Series.to_numpy(right_wall_x)
    right_wall_x = right_wall_x.reshape(-1,1)

    right_wall_y = right_wall['Points:1']
    right_wall_y = pd.Series.to_numpy(right_wall_y)
    right_wall_y = right_wall_y.reshape(-1,1)

    right_wall = np.concatenate([right_wall_x,right_wall_y], axis=1)

    walls_x = walls['Points:0']
    walls_x = pd.Series.to_numpy(walls_x)
    walls_x = walls_x.reshape(-1,1)

    walls_y = walls['Points:1']
    walls_y = pd.Series.to_numpy(walls_y)
    walls_y = walls_y.reshape(-1,1)

    walls = np.concatenate([walls_x,walls_y], axis=1)

    cylin_x = cylin['Points:0']
    cylin_x = pd.Series.to_numpy(cylin_x)
    cylin_x = cylin_x.reshape(-1,1)

    cylin_y = cylin['Points:1']
    cylin_y = pd.Series.to_numpy(cylin_y)
    cylin_y = cylin_y.reshape(-1,1)

    cylin = np.concatenate([cylin_x,cylin_y], axis=1)

    # boundary conditions

    left_u_sol = np.ones((len(left_wall_x), 1))
    left_v_sol = np.zeros((len(left_wall_x), 1))

    walls_u_sol = np.zeros((len(walls_x), 1))
    walls_v_sol = np.zeros((len(walls_x), 1))

    cylin_u_sol = np.zeros((len(cylin_x), 1))
    cylin_v_sol = np.zeros((len(cylin_x), 1))

    outlet_p_sol = np.zeros((len(right_wall_x), 1))
    outlet_v_sol = np.zeros((len(right_wall_x), 1))

    # convert all to tensors
    box = tf.convert_to_tensor(box, dtype=tf.float32)
    left_wall = tf.convert_to_tensor(left_wall, dtype=tf.float32)
    right_wall = tf.convert_to_tensor(right_wall, dtype=tf.float32)
    walls = tf.convert_to_tensor(walls, dtype=tf.float32)
    cylin = tf.convert_to_tensor(cylin, dtype=tf.float32)
    left_u_sol = tf.convert_to_tensor(left_u_sol, dtype=tf.float32)
    left_v_sol = tf.convert_to_tensor(left_v_sol, dtype=tf.float32)
    walls_u_sol = tf.convert_to_tensor(walls_u_sol, dtype=tf.float32)
    walls_v_sol = tf.convert_to_tensor(walls_v_sol, dtype=tf.float32)
    cylin_u_sol = tf.convert_to_tensor(cylin_u_sol, dtype=tf.float32)
    cylin_v_sol = tf.convert_to_tensor(cylin_v_sol, dtype=tf.float32)
    outlet_p_sol = tf.convert_to_tensor(outlet_p_sol, dtype=tf.float32)
    outlet_v_sol = tf.convert_to_tensor(outlet_v_sol, dtype=tf.float32)




    # plt.figure(figsize=(20,5))
    # plt.scatter(box[:, 0], box[:, 1], s=0.1, c='blue')
    # plt.scatter(left_wall[:, 0], left_wall[:, 1], s=0.1, c='yellow')
    # plt.scatter(right_wall[:, 0], right_wall[:, 1], s=0.1, c='red')
    # plt.scatter(walls[:, 0], walls[:, 1], s=0.1, c='black')
    # plt.scatter(cylin[:, 0], cylin[:, 1], s=0.1, c='purple')
    # plt.show()

    # print(box.shape)

    return box, left_wall, right_wall, walls, cylin, left_u_sol, left_v_sol, walls_u_sol, walls_v_sol, cylin_u_sol, cylin_v_sol, outlet_p_sol, outlet_v_sol

# flow_data()