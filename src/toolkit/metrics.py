import numpy as np

from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
from tabulate import tabulate

from util import check_multilabel_array, check_feature_array


def multilabel_tp_fp_tn_fn_scores(y_true, y_pred):
    y_true = check_multilabel_array(y_true)
    y_pred = check_multilabel_array(y_pred)
    n_labels = y_true.shape[1]
    if y_true.shape != y_pred.shape:
        raise ValueError('y_true and y_pred must have equal shapes')

    # Each confusion matrix has the following structure:
    # [[tp, fn],[fp, tn]]
    # When unrolling, the indexes are as follows:
    tp_idx, fn_idx, fp_idx, tn_idx = (0, 1, 2, 3)
    matrix = np.zeros((4, n_labels), dtype=int)
    for label_idx in range(n_labels):
        matrix[:, label_idx] = confusion_matrix(y_true[:, label_idx], y_pred[:, label_idx], labels=[1, 0]).ravel()
    return matrix[tp_idx], matrix[fp_idx], matrix[tn_idx], matrix[fn_idx]


def multilabel_accuracy(y_true, y_pred):
    y_true = check_multilabel_array(y_true)
    y_pred = check_multilabel_array(y_pred)
    n_labels = y_true.shape[1]
    if y_true.shape != y_pred.shape:
        raise ValueError('y_true and y_pred must have equal shapes')

    accuracy = np.zeros(n_labels)
    for label_idx in range(n_labels):
        accuracy[label_idx] = accuracy_score(y_true[:, label_idx], y_pred[:, label_idx])
    return accuracy


def multilabel_classification_report(y_true, y_pred, fmt='.3f', target_names=None):
    y_true = check_multilabel_array(y_true)
    y_pred = check_multilabel_array(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError('y_true and y_pred must have equal shapes')
    n_labels = y_true.shape[1]
    if target_names is not None and len(target_names) != n_labels:
        raise ValueError('target_names must specify a name for all %d labels' % n_labels)

    # Collect stats
    precision, recall, f1_score, support = precision_recall_fscore_support(y_true, y_pred)
    tp, fp, tn, fn = multilabel_tp_fp_tn_fn_scores(y_true, y_pred)
    accuracy = multilabel_accuracy(y_true, y_pred)

    # Generate data for table, where each row represents a label
    headers = ['', 'precision', 'recall', 'f1-score', 'accuracy', 'support', 'TP', 'TN', 'FP', 'FN']
    data = []
    for label_idx in range(n_labels):
        target_name = str(label_idx) if target_names is None else target_names[label_idx]
        row = [target_name, precision[label_idx], recall[label_idx], f1_score[label_idx], accuracy[label_idx],
               support[label_idx], tp[label_idx], tn[label_idx], fp[label_idx], fn[label_idx]]
        data.append(row)

    # Calculate summaries for all values
    summary = ['avg / total', np.average(precision), np.average(recall), np.average(f1_score), np.average(accuracy),
               np.sum(support), np.sum(tp), np.sum(tn), np.sum(fp), np.sum(fn)]
    data.append(summary)

    return tabulate(data, headers=headers, floatfmt=fmt)


def multilabel_prediction_report(y_true, y_pred, scores=None, score_fmt='.3f'):
    y_true = check_multilabel_array(y_true)
    y_pred = check_multilabel_array(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError('y_true and y_pred must have equal shapes')
    if scores is not None and y_true.shape != scores.shape:
        raise ValueError('y_true and scores must have equal shapes')

    # Format likelihoods so that they are properly aligned
    if scores is not None:
        printable_scores = _printable_scores(scores, score_fmt=score_fmt)
        assert len(printable_scores) == len(y_true)
    else:
        printable_scores = None

    # Generate table
    headers = ['', 'label', 'prediction', 'scores']
    data = []
    for idx, pred in enumerate(y_pred):
        label = y_true[idx]
        correct = (np.all(label == pred))
        symbol = '+' if correct else '-'
        row = [symbol, str(label), str(pred), '-' if printable_scores is None else ('[%s]' % printable_scores[idx])]
        data.append(row)
    return tabulate(data, headers=headers, floatfmt='.0f')


def multilabel_loglikelihood_report(y_true, loglikelihoods, fmt='.0f'):
    y_true = check_multilabel_array(y_true)
    loglikelihoods = check_feature_array(loglikelihoods)
    if y_true.shape != loglikelihoods.shape:
        raise ValueError('y_true and loglikelihoods must have equal shapes')

    # Format likelihoods so that they are properly aligned
    printable_loglikelihoods = _printable_scores(loglikelihoods, score_fmt=fmt)
    assert len(printable_loglikelihoods) == len(loglikelihoods)

    # Generate table
    headers = ['label', 'loglikelihoods']
    data = []
    for idx, label in enumerate(y_true):
        row = [str(label), '[' + printable_loglikelihoods[idx] + ']']
        data.append(row)
    return tabulate(data, headers=headers, floatfmt='.0f')


def multilabel_loglikelihood_summary(y_true, loglikelihoods):
    y_true = check_multilabel_array(y_true)
    loglikelihoods = check_feature_array(loglikelihoods)
    if y_true.shape != loglikelihoods.shape:
        raise ValueError('y_true and loglikelihoods must have equal shapes')
    n_labels = y_true.shape[1]

    # Calculate stats
    summary = np.zeros((n_labels, 4))
    for label_idx in xrange(n_labels):
        pos_indexes = np.where(y_true[:, label_idx] == 1)[0]
        neg_indexes = np.where(y_true[:, label_idx] == 0)[0]

        summary[label_idx][0] = np.average(loglikelihoods[:, label_idx][pos_indexes])
        summary[label_idx][1] = np.std(loglikelihoods[:, label_idx][pos_indexes])
        summary[label_idx][2] = np.average(loglikelihoods[:, label_idx][neg_indexes])
        summary[label_idx][3] = np.std(loglikelihoods[:, label_idx][neg_indexes])
    return summary


def multilabel_loglikelihood_summary_report(y_true, loglikelihoods, fmt='.2f', target_names=None):
    n_labels = y_true.shape[1]
    if target_names is not None and len(target_names) != n_labels:
        raise ValueError('target_names must specify a name for all %d labels' % n_labels)

    stats = multilabel_loglikelihood_summary(y_true, loglikelihoods)
    avg_stats = np.average(stats, axis=0)
    assert avg_stats.shape == (4,)

    # Generate table
    headers = ['', 'mean positive', 'stddev positive', 'mean negative', 'stddev negative']
    data = []
    for label_idx in xrange(n_labels):
        target_name = str(label_idx) if target_names is None else target_names[label_idx]
        curr_stats = stats[label_idx]
        row = [target_name, curr_stats[0], curr_stats[1], curr_stats[2], curr_stats[3]]
        data.append(row)

    # Calculate summaries for all values
    summary_row = ['avg', avg_stats[0], avg_stats[1], avg_stats[2], avg_stats[3]]
    data.append(summary_row)

    return tabulate(data, headers=headers, floatfmt=fmt)


def distances_report(distances, fmt='.2f', target_names=None):
    distances = check_feature_array(distances)
    if distances.shape[0] != distances.shape[1]:
        raise ValueError('distances must be square')
    n_samples = distances.shape[1]
    if target_names is not None and len(target_names) != n_samples:
        raise ValueError('target_names must specify a name for all %d labels' % n_samples)

    # Generate table
    if target_names is None:
        target_names = range(n_samples)
    headers = [''] + target_names
    data = []
    for sample_idx in xrange(n_samples):
        target_name = target_names[sample_idx]
        row = [target_name] + list(distances[sample_idx])
        data.append(row)

    # Calculate summaries for all values
    averages = np.average(distances, axis=1)
    summary_row = ['avg'] + list(averages)
    data.append(summary_row)

    return tabulate(data, headers=headers, floatfmt=fmt)


def _printable_scores(scores, score_fmt='.3f'):
    printable_scores = tabulate(scores.tolist(), floatfmt=score_fmt, tablefmt='plain').splitlines()
    return printable_scores
