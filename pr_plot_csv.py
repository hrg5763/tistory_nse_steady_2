from pr_csv import *
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


op = pd.read_csv('/home/hrg5763/Documents/tistory_nse_steady_2/csv/full.csv')
of = pd.read_csv('/home/hrg5763/Documents/tistory_nse_steady_2/openfoam.csv')

x_data_of = of['Points:0']
y_data_of = of['Points:1']
r_data_of = of['Result']
x_data_of = pd.Series.to_numpy(x_data_of)
y_data_of = pd.Series.to_numpy(y_data_of)
r_data_of = pd.Series.to_numpy(r_data_of)
x_data_of = x_data_of.reshape(-1,1)
y_data_of = y_data_of.reshape(-1,1)
r_data_of = r_data_of.reshape(r_data_of.shape[0],)
print(r_data_of.shape)
xy_data_of = np.concatenate([x_data_of,y_data_of], axis=1)

x_data = op['Points:0']
y_data = op['Points:1']
x_data = pd.Series.to_numpy(x_data)
y_data = pd.Series.to_numpy(y_data)
x_data = x_data.reshape(-1,1)
y_data = y_data.reshape(-1,1)
xy_data = np.concatenate([x_data,y_data], axis=1)



agent = NSpinn()
agent.load_weights('/home/hrg5763/Documents/tistory_nse_steady_2/')

# result = agent.predict(xy_col)
result = agent.predict(xy_data_of)
u=result[0]
v=result[1]
p=result[2]

u = np.array(u)
v = np.array(v)
p = np.array(p)
u = u.reshape(u.shape[0],)
v = v.reshape(v.shape[0],)
p = p.reshape(p.shape[0],)
print(u.shape, "u")
plt.figure(figsize=(20,5))
plt.title(f"Velocity ")
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.scatter(xy_data_of[:, 0],xy_data_of[:, 1],c=u,cmap='jet',s=0.5)
plt.colorbar(label="$u$ [m/s]")
# plt.clim([0,1])
plt.show()

plt.figure(figsize=(20,5))
plt.title(f"Velocity ")
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.scatter(xy_data_of[:, 0],xy_data_of[:, 1],c= abs(r_data_of - u) ,cmap='jet',s=0.5)
plt.colorbar(label="$u$ [m/s]")
# plt.clim([0,1])
plt.show()
