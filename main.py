import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

input_type = 'BGR'
general_LAB_flat = []
general_colors = []

for i in range(2):
  image_BGR = cv2.imread('imgs/img' + str(i) + '.jpg')
  image = image_BGR

  conversion = cv2.COLOR_BGR2LAB if input_type == 'BGR' else cv2.COLOR_RGB2LAB
  image_LAB = cv2.cvtColor(image, conversion)

  y,x,z = image_LAB.shape
  LAB_flat = np.reshape(image_LAB, [y*x,z])

  colors = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if input_type == 'BGR' else image
  colors = np.reshape(colors, [y*x,z])/255.

  general_LAB_flat.append(LAB_flat)
  general_colors.append(colors)

general_LAB_flat = np.vstack(general_LAB_flat)
general_colors = np.vstack(general_colors)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xs=general_LAB_flat[:,2], ys=general_LAB_flat[:,1], zs=general_LAB_flat[:,0], s=0.1,  c=general_colors, lw=0)
ax.set_xlabel('A')
ax.set_ylabel('B')
ax.set_zlabel('L')

plt.show()
