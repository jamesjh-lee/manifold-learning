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

class Sampling(Layer):
    def call(self,inputs):
        mu, sigma = inputs
        batch = tf.shape(mu)[0]
        dim = tf.shape(mu)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return mu + sigma * epsilon

class VAE:
    def __init__(self, n_components, origin_dim):
        self.latent_dim = n_components
        self.origin_dim = origin_dim
        self.inputs = Input(shape=(self.origin_dim), dtype=tf.float32)

        
    def _build_net(self):
        self.encoder = self._encoder()
        self.decoder = self._decoder()
        outputs = self.decoder(self.encoder(self.inputs)[-1])
        self.model = Model(inputs=self.inputs, outputs=outputs, name='VAE')
    
    def vae_loss(self, x, y):
        marginal_likelihood = tf.reduce_sum(x * tf.math.log(y) + (1 - x) * tf.math.log(1 - y), 1)
        KL_divergence = 0.5 * tf.reduce_sum(tf.square(self.mu) + tf.square(self.sigma) - tf.math.log(1e-8 + tf.square(self.sigma)) - 1, 1)
        marginal_likelihood = tf.reduce_mean(marginal_likelihood)
        KL_divergence = tf.reduce_mean(KL_divergence)
        ELBO = marginal_likelihood - KL_divergence
        return -ELBO
    
    def _encoder(self):
        hidden = Dense(256, activation='relu')(self.inputs)
        hidden = Dense(128, activation='relu')(hidden)
        hidden = Dense(64, activation='relu')(hidden)
        self.mu = Dense(self.latent_dim)(hidden)
        self.sigma = 1e-6 + tf.math.softplus(Dense(self.latent_dim)(hidden))
        self.z = Sampling()([self.mu, self.sigma])
        return Model(inputs=self.inputs, outputs=[self.mu, self.sigma, self.z], name='VAE-encoder')
    
    def _decoder(self):
        d_in = Input(shape=(self.latent_dim))
        hidden = Dense(64, activation='relu')(d_in)
        hidden = Dense(128, activation='relu')(hidden)
        hidden = Dense(256, activation='relu')(hidden)
        d_out = Dense(self.origin_dim, activation='sigmoid')(hidden)
        return Model(inputs=d_in, outputs=d_out, name='VAE-decoder')
    
    def fit_transform(self, x, **kwargs):
        self._build_net()
        self.model.compile(optimizer=Adam(0.00001), loss=self.vae_loss, metrics=['mae'])
        hist = self.fit(x, **kwargs)
        print(hist.history['loss'][-1], hist.history['mae'][-1])
        return self.transform(x)
    
    def fit(self, x, **kwargs):
        return self.model.fit(x, x, **kwargs)
    
    def transform(self, x):
        return self.encoder.predict(x)[0]
