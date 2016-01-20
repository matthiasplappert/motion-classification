import numpy as np
from sklearn.utils.validation import check_array


class NotFittedError(ValueError, AttributeError):
    pass


def check_feature_array(array, n_features=None):
    array = check_array(array, ensure_2d=True, allow_nd=False)
    if n_features is not None and array.shape[1] != n_features:
        raise ValueError('feature array must have exactly %d features' % n_features)
    return array


def check_is_fitted(estimator, attributes, msg=None, all_or_any=all):
    # Based on sklearn.util.validation.check_is_fitted but also ensures
    # that the attribute is not None
    if msg is None:
        msg = ("This %(name)s instance is not fitted yet. Call 'fit' with "
               "appropriate arguments before using this method.")

    if not hasattr(estimator, 'fit'):
        raise TypeError("%s is not an estimator instance." % estimator)

    if not isinstance(attributes, (list, tuple)):
        attributes = [attributes]

    if not all_or_any([hasattr(estimator, attr) for attr in attributes]):
        raise NotFittedError(msg % {'name': type(estimator).__name__})

    if not all_or_any([getattr(estimator, attr) for attr in attributes]):
        raise NotFittedError(msg % {'name': type(estimator).__name__})


def check_multilabel_array(array, n_labels=None, force_binary=True):
    array = check_array(array, ensure_2d=True, allow_nd=False, dtype=int)
    if n_labels is not None and array.shape[1] != n_labels:
        raise ValueError('multilabel array must have exactly %d labels' % n_labels)
    if force_binary:
        count_ones = np.count_nonzero(array == 1)
        count_zeros = np.count_nonzero(array == 0)
        if np.size(array) != count_ones + count_zeros:
            raise ValueError('multilabel array must be binary')
    return array


def pad_sequences(X):
    # Find longest sequence
    n_samples_max = 0
    for X_curr in X:
        n_samples_curr = X_curr.shape[0]
        if n_samples_curr > n_samples_max:
            n_samples_max = n_samples_curr

    # Adjust length of all sequences to be equal
    for idx, X_curr in enumerate(X):
        n_samples_curr = X_curr.shape[0]
        delta_samples = n_samples_max - n_samples_curr
        assert delta_samples >= 0
        if delta_samples > 0:
            fill_array = np.zeros((delta_samples, X_curr.shape[1]))
            X[idx] = np.append(X_curr, fill_array, axis=0)
            assert X[idx].shape[0] == n_samples_max
    X = np.asarray(X)
    return X
