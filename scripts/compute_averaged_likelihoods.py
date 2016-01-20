import os
import csv

import numpy as np


def compute_wasserstein(pos_ll_means, pos_ll_stds, neg_ll_means, neg_ll_stds, k=23):
	n_labels = len(pos_ll_means)
	pos_vars = np.square(pos_ll_stds)
	neg_vars = np.square(neg_ll_stds)
	ds = []
	for idx in xrange(n_labels):
		m1 = pos_ll_means[idx]
		m2 = neg_ll_means[idx]
		var1 = pos_vars[idx]
		var2 = neg_vars[idx]
		d = np.sqrt(abs(m1 - m2) + (var1 + var2 - 2 * np.sqrt(var1 * var2)))
		ds.append(d)
	measure = np.median(ds) / float(k)
	return measure


def compute_from_path(ll_path, label_path, n_features=None, print_all=False, label_names=None):
	lls = np.loadtxt(ll_path, delimiter=';')
	y = np.loadtxt(label_path, delimiter=';', dtype=int)
	assert lls.shape == y.shape
	n_samples, n_labels = y.shape

	pos_ll_means = []
	pos_ll_stds = []
	neg_ll_means = []
	neg_ll_stds = []
	for label_idx in xrange(n_labels):
		pos_rows = np.where(y[:, label_idx] == 1)[0]
		neg_rows = np.where(y[:, label_idx] == 0)[0]
		assert np.size(pos_rows) + np.size(neg_rows) == n_samples
		pos_lls = lls[pos_rows, label_idx]
		neg_lls = lls[neg_rows, label_idx]
		
		pos_ll_means.append(np.mean(pos_lls))
		pos_ll_stds.append(np.std(pos_lls))
		neg_ll_means.append(np.mean(neg_lls))
		neg_ll_stds.append(np.std(neg_lls))

	if print_all:
		for idx, (pos_mean, pos_std) in enumerate(zip(pos_ll_means, pos_ll_stds)):
			label_name = ''
			if label_names is not None:
				label_name = label_names[idx]
			print('%s: %f +- %f' % (label_name, pos_mean, pos_std))
		print('')
		for idx, (neg_mean, neg_std) in enumerate(zip(neg_ll_means, neg_ll_stds)):
			label_name = ''
			if label_names is not None:
				label_name = label_names[idx]
			print('%s: %f +- %f' % (label_name, neg_mean, neg_std))
		print('')

	print('median pos loglikelihood: %f +- %f' % (np.median(pos_ll_means), np.median(pos_ll_stds)))
	print('median neg loglikelihood: %f +- %f' % (np.median(neg_ll_means), np.median(neg_ll_stds)))
	if n_features is not None:
		print('wasserstein: %f' % compute_wasserstein(pos_ll_means, pos_ll_stds, neg_ll_means, neg_ll_stds, k=n_features))

# root_path = '/Users/matze/Desktop/hmm/init/'
# ll_filename = '007_uniform_k-means_full_loglikelihoods.csv'
# label_filename = '007_uniform_k-means_full_labels.csv'
# label_names = ['walk','turn-right','turn-left','speed-normal','direction-forward','speed-slow','speed-fast','bend-left','bend-right','push-recovery','direction-backward','direction-left','direction-right','direction-circle','direction-counter-clockwise','direction-clockwise','direction-slalom','run','direction-upward','kick','foot-right','throw','hand-right','hand-left','bow','deep','slight','high','foot-left','low','squat','punch','stomp','jump','gold','putting','drive','tennis','smash','forehand','wave','hand-both','guitar','violin','stir','wipe','dance','waltz','chachacha']
# compute_from_path(os.path.join(root_path, ll_filename), os.path.join(root_path, label_filename), n_features=22, print_all=True, label_names=label_names)

label_names = ['walk','turn-right','turn-left','speed-normal','direction-forward','speed-slow','speed-fast','bend-left','bend-right','push-recovery','direction-backward','direction-left','direction-right','direction-circle','direction-counter-clockwise','direction-clockwise','direction-slalom','run','direction-upward','kick','foot-right','throw','hand-right','hand-left','bow','deep','slight','high','foot-left','low','squat','punch','stomp','jump','gold','putting','drive','tennis','smash','forehand','wave','hand-both','guitar','violin','stir','wipe','dance','waltz','chachacha']
root_path = '/Users/matze/Desktop/hmm/fhmm/'
# n_features = []
# with open(os.path.join(root_path, '_final__results.csv'), 'rb') as f:
# 	reader = csv.reader(f, delimiter=';')
# 	for idx, row in enumerate(reader):
# 		if idx == 0:
# 			continue
# 		n_features.append(int(row[3]))

file_prefixes = []
for name in os.listdir(root_path):
	split = name.split('_loglikelihoods.csv')
	if len(split) == 2:
		file_prefixes.append(split[0])
sorted(file_prefixes)
n_features = [22 for _ in xrange(len(file_prefixes))]
assert len(file_prefixes) == len(n_features)

for idx, (prefix, n) in enumerate(zip(file_prefixes, n_features)):
	ll_filename = '%s_loglikelihoods.csv' % prefix
	label_filename = '%s_labels.csv' % prefix
	print(idx)
	compute_from_path(os.path.join(root_path, ll_filename), os.path.join(root_path, label_filename), n_features=n, print_all=True, label_names=label_names)
	print('')
