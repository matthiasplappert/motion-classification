import csv

import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams

rcParams['figure.figsize'] = 6, 5
#rcParams['text.usetex']=True
#rcParams['text.latex.unicode']=True

times = []
scores = []
stds = []
chains = []
loglikelihoods = []
max_y = None
with open('/Users/matze/Desktop/hmm/fhmm/_results.csv', 'rb') as f:
    reader = csv.reader(f, delimiter=';')
    for row_idx, row in enumerate(reader):
        if row_idx == 0:
            continue
        scores.append(float(row[2]))
        chains.append(int(row[3]))
        times.append(float(row[4]))
        loglikelihoods.append(float(row[5]))
        stds.append(float(row[6]))
loglikelihoods = np.array(loglikelihoods)
stds = np.array(stds)

print loglikelihoods

plt.plot(chains, loglikelihoods)
plt.xticks(range(1, 5))
plt.xlabel('number of chains')
plt.ylabel('standard deviation of loglikelihoods')
plt.fill_between(chains, loglikelihoods+stds, loglikelihoods-stds, facecolor='blue', alpha=0.5)

plt.grid(True)
plt.show()

