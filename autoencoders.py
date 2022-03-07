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
        self.model = None
        self.encoder = None
        self.decoder = None
    
    def build_net(self):
        # encoder
        e_in = Input(shape=(self.origin_dim,))
        hidden = Dense(256, activation='relu')(e_in)
        hidden = Dense(128, activation='relu')(hidden)
        hidden = Dense(64, activation='relu')(hidden)
        e_out = Dense(self.latent_dim, activation='relu')(hidden)
        self.encoder = Model(inputs=e_in, outputs=e_out, name='encoder')

        # decoder
        d_in = Input(shape=(self.latent_dim,))
        hidden = Dense(64, activation='relu')(d_in)
        hidden = Dense(128, activation='relu')(hidden)
        hidden = Dense(256, activation='relu')(hidden)
        d_out = Dense(self.origin_dim, activation='sigmoid')(hidden)
        self.decoder = Model(inputs=d_in, outputs=d_out, name='decoder')
        
        # AE
        outputs = self.decoder(self.encoder(e_in))
        self.model = Model(inputs=e_in, outputs=outputs, name='AE')

    def fit_transform(self, x, **kwargs):
        self.build_net()
        self.model.compile(optimizer=Adam(0.00001), loss='mae')
        hist = self.fit(x)
        print(hist.history['loss'][-1])
        return self.encoder.predict(x, **kwargs)
    
    def fit(self, x, **kwargs):
      self.model.fit(x, x, **kwargs)
     
    def transform(self, x):
      return self.encoder.predict(x)
