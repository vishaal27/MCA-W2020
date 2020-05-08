import pickle
import numpy as np
import matplotlib.pyplot as plt

with open('final_surf_blobs.json', 'rb') as json_file:
		blobs = pickle.load(json_file)

plot = []

for key in blobs:
	plot.append(len(blobs[key]))

print(np.mean(plot), np.min(plot), np.max(plot))
plt.plot(plot)
plt.savefig('./surf_blobs_distribution.jpg')