from pr import *
from pr_bc_csv import *
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import os
from matplotlib import cm
from matplotlib.ticker import LinearLocator
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation


op = pd.read_csv('/home/yh98/Documents/transient_prac/karman/tistory_nse_steady/csv/cylinder_xy.csv')



x_data = op['Points:0']
y_data = op['Points:1']
x_data = pd.Series.to_numpy(x_data)
y_data = pd.Series.to_numpy(y_data)
x_data = x_data.reshape(-1,1)
y_data = y_data.reshape(-1,1)
xy_data = np.concatenate([x_data,y_data], axis=1)



agent = NSpinn()
agent.load_weights('/home/yh98/Documents/transient_prac/karman/tistory_nse_steady/')

# result = agent.predict(xy_col)
result = agent.predict(xy_data)
u=result[0]
v=result[1]
p=result[2]

u=np.array(u)
v=np.array(v)
p=np.array(p)
u = u.reshape(u.shape[0],)
v = v.reshape(v.shape[0],)
p = p.reshape(p.shape[0],)

plt.figure(figsize=(20,5))
plt.title(f"Velocity ")
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.scatter(xy_data[:, 0],xy_data[:, 1],c=u,cmap='jet',s=0.5)
plt.colorbar(label="$u$ [m/s]")
# plt.clim([0,1])
plt.show()
