import csv

import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams

rcParams['figure.figsize'] = 10, 6
rcParams['text.usetex']=True
#rcParams['text.latex.unicode']=True

penalties = {'l1': ([], []), 'l2': ([], [])}
with open('/Users/matze/Studium/Bachelorarbeit/Documents/thesis/data/decision-makers/log-regression.csv', 'rb') as f:
    reader = csv.reader(f, delimiter=';')
    for row_idx, row in enumerate(reader):
        if row_idx == 0:
            continue
        score = float(row[2])
        penalty = row[3]
        C = float(row[4])
        penalties[penalty][0].append(C)
        penalties[penalty][1].append(score)

plt.plot(penalties['l1'][0], penalties['l1'][1])
plt.plot(penalties['l2'][0], penalties['l2'][1])
plt.legend(['L1 regularization', 'L2 regularization'], loc='lower right')
plt.xticks(penalties['l1'][0])
plt.xlabel('C')
plt.ylabel('score')
plt.xscale('log')
plt.grid(True)
plt.axis([penalties['l1'][0][0], penalties['l1'][0][-1], 0.6, 1])
plt.show()
