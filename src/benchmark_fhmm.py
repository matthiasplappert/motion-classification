import line_profiler
import numpy as np

import toolkit.dataset as dataset
from toolkit.hmm.impl_hmmlearn import GaussianFHMM
import hmmlearn.fhmmc

print('Loading dataset ...')
X, y, _, _ = dataset.load_manifest('../data/dataset_small.json')  # TODO: make configurable
fhmm = GaussianFHMM(n_states=5, n_chains=2)
target_indexes = np.where(y[:, 0] == 1)[0]
X_train = [v for i, v in enumerate(X) if i in target_indexes]
assert len(X_train) == len(target_indexes)

print('Fitting model ...')
fhmm.fit(X_train)

# Benchmark
profile = line_profiler.LineProfiler(hmmlearn.fhmmc._forward)
profile.runcall(fhmm.loglikelihood, X_train[0])
profile.print_stats()
