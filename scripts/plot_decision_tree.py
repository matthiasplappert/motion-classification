import csv

import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams

rcParams['figure.figsize'] = 10, 6
rcParams['text.usetex']=True
#rcParams['text.latex.unicode']=True

data = {'gini': ([], []), 'entropy': ([], [])}
with open('/Users/matze/Studium/Bachelorarbeit/Documents/thesis/data/decision-makers/decision-tree.csv', 'rb') as f:
    reader = csv.reader(f, delimiter=';')
    for row_idx, row in enumerate(reader):
        if row_idx == 0:
            continue
        score = float(row[2])
        criterion = row[3]
        depth = float(row[4])
        data[criterion][0].append(depth)
        data[criterion][1].append(score)

plt.plot(data['gini'][0], data['gini'][1])
plt.plot(data['entropy'][0], data['entropy'][1])
plt.legend(['Gini impurity criterion', 'information gain criterion'], loc='lower right')
#plt.xticks(np.arange(1, 40, 5))
plt.xlabel('maximum depth')
plt.ylabel('score')
plt.grid(True)
plt.axis([1, 40, 0, 1])
plt.show()
