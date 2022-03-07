from tensorflow.compat.v1.keras.layers import Dense, Input, Layer
from tensorflow.compat.v1.keras.models import Model
from tensorflow.compat.v1.keras.optimizers import Adam
from tensorflow.compat.v1 import disable_eager_execution
import tensorflow as tf
import tensorflow.compat.v1.keras.backend as K
disable_eager_execution()


class AE:
    def __init__(self, n_components, origin_dim):
        self.latent_dim = n_components
        self.origin_dim = origin_dim
        self.inputs = Input(shape=(origin_dim,))
    
    def _build_net(self):
        self.encoder = self._encoder()
        self.decoder = self._decoder()
        outputs = self.decoder(self.encoder(self.inputs))
        self.model = Model(inputs=self.inputs, outputs=outputs, name='AE')
        
    def _encoder(self):
        hidden = Dense(256, activation='relu')(self.inputs)
        hidden = Dense(128, activation='relu')(hidden)
        hidden = Dense(64, activation='relu')(hidden)
        e_out = Dense(self.latent_dim, activation='relu')(hidden)
        return Model(inputs=self.inputs, outputs=e_out, name='encoder')

    def _decoder(self):
        # decoder
        d_in = Input(shape=(self.latent_dim,))
        hidden = Dense(64, activation='relu')(d_in)
        hidden = Dense(128, activation='relu')(hidden)
        hidden = Dense(256, activation='relu')(hidden)
        d_out = Dense(self.origin_dim, activation='sigmoid')(hidden)
        return Model(inputs=d_in, outputs=d_out, name='decoder')

    def fit_transform(self, x, **kwargs):
        self._build_net()
        self.model.compile(optimizer=Adam(0.0001), loss='mae')
        hist = self.fit(x, **kwargs)
        print(hist.history['loss'][-1])
        return self.transform(x)
    
    def fit(self, x, **kwargs):
      return self.model.fit(x, x, **kwargs)
     
    def transform(self, x):
      return self.encoder.predict(x)
