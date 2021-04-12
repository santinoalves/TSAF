from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout

from tensorflow.keras.wrappers.scikit_learn import KerasRegressor


class TSAFlstm(KerasRegressor):
    '''Extends keras regressor to sequence input - single output NN'''


    def score(self, x, y, **kwargs):
        x = x.reshape(x.shape[0], x.shape[1], 1)
        return super(LSTMRegressor, self).score(x, y, **kwargs)

    '''Extends the keras regressor to sequence input - single output LSTM'''

    def predict(self, x, **kwargs):
        x = x.reshape(x.shape[0], x.shape[1], 1)
        return super(LSTMRegressor, self).predict(x, **kwargs)

    def fit(self, x, y, **kwargs):
        '''Constructs a new model with `build_fn` & fit the model to `(x, y)` where x is a sequence of the time series.'''
        x = x.reshape(x.shape[0], x.shape[1], 1)
        from tensorflow.keras.callbacks import EarlyStopping
        stopper = EarlyStopping(monitor='val_loss', patience=5)
        params = kwargs.copy()
        params.update(callbacks=[stopper])

        return super(LSTMRegressor, self).fit(x, y, **params)


def generate_model(n_neurons=64):
    lstm = LSTM_Model().generate_model
    regressor = LSTMRegressor(lstm)
    return regressor


class LSTM_Model(object):
    def __init__(self, n_neurons=64, stateful=False):
        self.neurons = n_neurons
        self.stateful = stateful

    def generate_model(self):
        model = Sequential()
        model.add(LSTM(self.neurons, stateful=self.stateful, unit_forget_bias=True, unroll=True, recurrent_dropout=0.2))
        # model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model
