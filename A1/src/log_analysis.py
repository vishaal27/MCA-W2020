import json
import numpy as np
import matplotlib.pyplot as plt

with open('final_log_blobs.json', 'r') as json_file:
		blobs = json.load(json_file)

plot = []

for key in blobs:
	plot.append(len(blobs[key]))

print(np.mean(plot), np.min(plot), np.max(plot))
plt.plot(plot)
plt.savefig('./log_blobs_distribution.jpg')