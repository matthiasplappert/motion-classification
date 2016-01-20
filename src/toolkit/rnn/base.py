import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, TimeDistributedDense
from keras.layers.recurrent import LSTM, GRU
from keras.layers.normalization import BatchNormalization
import keras.optimizers as opt

from ..util import check_is_fitted, pad_sequences


class Classifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_batch_size=4, n_epochs=1000):
        self.n_batch_size = n_batch_size
        self.n_epochs = n_epochs
        self.model_ = None
        self.n_labels_ = None

    def fit(self, X, y):
        assert isinstance(X, list)  #TODO: this should not be an assert
        assert len(y) > 0
        assert len(X) == len(y)

        X = pad_sequences(X)
        print X.shape, y.shape

        n_features = X.shape[2]
        self.n_labels_ = y.shape[1]
        print n_features, self.n_labels_

        model = Sequential()
        model.add(GRU(n_features, 128))
        model.add(Dropout(0.1))
        model.add(BatchNormalization(128))
        model.add(Dense(128, self.n_labels_))
        model.add(Activation('sigmoid'))

        sgd = opt.SGD(lr=0.005, decay=1e-6, momentum=0., nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, class_mode='categorical')

        model.fit(X, y, batch_size=self.n_batch_size, nb_epoch=self.n_epochs, show_accuracy=True)
        self.model_ = model

    def predict_proba(self, X):
        check_is_fitted(self, 'model_', 'n_labels_')
        X = pad_sequences(X)
        y_pred = self.model_.predict(X, batch_size=self.n_batch_size)
        return y_pred