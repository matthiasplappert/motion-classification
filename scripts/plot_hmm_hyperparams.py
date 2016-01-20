import csv

import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams

rcParams['figure.figsize'] = 12, 5
rcParams['text.usetex']=True
#rcParams['text.latex.unicode']=True

topologies = {}
max_y = None
with open('/Users/matze/Desktop/hmm/hyperparams/_results.csv', 'rb') as f:
    reader = csv.reader(f, delimiter=';')
    for row_idx, row in enumerate(reader):
        if row_idx == 0:
            continue
        measure = float(row[2])
        n_states, topology = row[3].split(', ')
        if max_y is None or max_y < measure:
            max_y = measure
        if topology not in topologies:
            topologies[topology] = ([], [])
        topologies[topology][0].append(int(n_states))
        topologies[topology][1].append(measure)

data = []
data.append(topologies['full'])
data.append(topologies['left-to-right-full'])
data.append(topologies['left-to-right-1'])
data.append(topologies['left-to-right-2'])
for data, (xs, ys) in topologies.iteritems():
    plt.plot(xs, ys)
plt.legend(['fully-connected', 'left-to-right without $\Delta$ constraint', 'left-to-right with $\Delta = 1$', 'left-to-right with $\Delta = 2$'], loc='upper right')
plt.xticks(range(3, 21))
plt.xlabel('number of states')
plt.ylabel('score')
plt.grid(True)
plt.axis([3, 20, 25000, 60000])
plt.show()
