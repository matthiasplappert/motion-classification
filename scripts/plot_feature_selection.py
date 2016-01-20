import csv

import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams

rcParams['figure.figsize'] = 12, 5
rcParams['text.usetex']=True
#rcParams['text.latex.unicode']=True

ys = []
max_y = None
with open('/Users/matze/Studium/Bachelorarbeit/Evaluation/features/final-dataset/smoothed_normalized_wasserstein/_final__results.csv', 'rb') as f:
    reader = csv.reader(f, delimiter=';')
    for row_idx, row in enumerate(reader):
        if row_idx == 0:
            continue
        measure = float(row[2])
        ys.append(measure)

xs = np.arange(1, len(ys)+1)
plt.plot(xs, ys)
#plt.legend(['fully-connected', 'left-to-right without $\Delta$ constraint', 'left-to-right with $\Delta = 1$', 'left-to-right with $\Delta = 2$'], loc='upper right')
plt.xticks(xs)
plt.xlabel('round')
plt.ylabel('measure')
plt.grid(True)
plt.axis([1, len(ys), 0, 1.4e+11])
plt.show()
