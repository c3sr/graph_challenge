"""

Convert a CSV file into a heatmap.

Consume the binned neighbor list length from src/2dhist.cu to produce a heat map

"""


from __future__ import print_function
import pandas as pd
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

FONTSIZE=14

df = pd.read_csv(sys.argv[1])
arr = df.values

arr = np.delete(arr, 0, axis=1) # delete y labels
arr = np.where(arr > 0, np.log2(arr), 0)

fig, ax = plt.subplots()
im = ax.imshow(arr)

ax.xaxis.tick_top()
locs, labels = ax.get_xticks(), ax.get_xticklabels()
ax.set_xlabel('log2 |N(j)|', fontsize=FONTSIZE)
ax.set_ylabel('log2 |N(i)|', fontsize=FONTSIZE)
ax.set_title('E(i,j) Counts by Neighbor List Length', fontsize=FONTSIZE, pad=30)


cbar = plt.colorbar(im)
cbar.ax.set_ylabel("log2 Edge Count", fontsize=FONTSIZE)


fig.tight_layout()
plt.show()