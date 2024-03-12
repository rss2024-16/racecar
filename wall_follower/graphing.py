import numpy as np
import matplotlib.pyplt as plt

stats = np.load('1710120317.436326.npy',allow_pickle=True)

look_ahead = stats['look_ahead']
front_wall = stats['front_wall']
average_dist = stats['average_dist']
min_dist = stats['following_wall']

plt.figure()
plt.plot([x for x in range(len(average_dist))],average_dist)
plt.show()