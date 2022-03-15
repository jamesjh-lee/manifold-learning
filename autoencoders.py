from tensorflow.compat.v1.keras.layers import Dense, Input, Layer, Reshape, Flatten, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.compat.v1.keras.models import Model
from tensorflow.compat.v1.keras.optimizers import get
from tensorflow.compat.v1 import disable_eager_execution
import tensorflow as tf
import tensorflow.compat.v1.keras.backend as K
import numpy as np
disable_eager_execution()

def downsample(x, rate=2):
    return Dense(int(x.shape[-1]/2), activation='relu')(x)

def upsample(x, rate=2):
    return Dense(int(x.shape[-1]*2), activation='relu')(x)


class AE:
    def __init__(self, latent_dim, **kwargs):
        self.latent_dim = latent_dim
        self.hyperparameter = kwargs
        self.optimizer = get(kwargs['optimizer'])
        try:
            self.learning_rate = kwargs['learning_rate']
        except KeyError:
            self.learning_rate = 1e-4

    def _build_net(self, origin_dim):
        self.inputs = Input(shape=(origin_dim))
        self.encoder = self.__encoder(origin_dim)
        self.decoder = self.__decoder(origin_dim)
        outputs = self.decoder(self.encoder(self.inputs))
        self.model = Model(inputs=self.inputs, outputs=outputs, name='AE')

    def __encoder(self, origin_dim, layers=2):
        hidden = Dense(1024, activation='relu')(self.inputs)
        for i in range(layers):
            hidden = downsample(hidden)
        outputs = Dense(self.latent_dim, activation='relu')(hidden)
        return Model(inputs=self.inputs, outputs=outputs, name='Encoder')

    def __decoder(self, origin_dim, layers=2):
        inputs = Input(shape=(self.latent_dim))
        for i in range(layers):
            if i == 0:
                hidden = upsample(inputs)
                continue
            hidden = upsample(hidden)
        hidden = Dense(1024, activation='relu')(hidden)
        outputs = Dense(origin_dim, activation='sigmoid')(hidden)
        return Model(inputs=inputs, outputs=outputs, name='Decoder')

    def fit(self, x, **kwargs):
        self._build_net(x.shape[-1])
        self.optimizer._hyper['learning_rate'] = self.learning_rate
        self.model.compile(self.optimizer, loss=self.hyperparameter['loss'], metrics=['binary_crossentropy'])
        hist = self.model.fit(x, x, **kwargs)
        print(len(hist.history['loss']), hist.history['loss'][-1])

    def predict(self, x, **kwargs):
        return self.encoder.predict(x)

    def fit_transform(self, x, **kwargs):
        self.fit(x, **kwargs)
        return self.predict(x, **kwargs)


class Sampling(Layer):
    def call(self,inputs):
        mu, sigma = inputs
        batch = tf.shape(mu)[0]
        dim = tf.shape(mu)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return mu + sigma * epsilon


class VAE:
    def __init__(self, latent_dim, **kwargs):
        self.latent_dim = latent_dim
        self.hyperparameter = kwargs
        self.optimizer = get(kwargs['optimizer'])
        try:
            self.learning_rate = kwargs['learning_rate']
        except KeyError:
            self.learning_rate = 1e-4
        try:
            self.beta = kwargs['beta']
        except KeyError:
            self.beta = 0.5

    def _build_net(self, origin_dim):
        self.inputs = Input(shape=(origin_dim))
        self.encoder = self.__encoder(origin_dim)
        self.decoder = self.__decoder(origin_dim)
        outputs = self.decoder(self.encoder(self.inputs)[-1])
        self.model = Model(inputs=self.inputs, outputs=outputs, name='AE')

    def __encoder(self, origin_dim, layers=2):
        hidden = Dense(1024, activation='relu')(self.inputs)
        for i in range(layers):
            hidden = downsample(hidden)
        self.mu = Dense(self.latent_dim, activation='relu')(hidden)
        self.sigma = Dense(self.latent_dim, activation='softplus')(hidden)
        z = Sampling()([self.mu, self.sigma])
        return Model(inputs=self.inputs, outputs=[self.mu, self.sigma, z], name='vEncoder')

    def __decoder(self, origin_dim, layers=2):
        inputs = Input(shape=(self.latent_dim))
        for i in range(layers):
            if i == 0:
                hidden = upsample(inputs)
                continue
            hidden = upsample(hidden)
        hidden = Dense(1024, activation='relu')(hidden)
        outputs = Dense(origin_dim, activation='softplus')(hidden)
        return Model(inputs=inputs, outputs=outputs, name='vDecoder')

    def vae_loss(self, x, y):
        marginal_likelihood = tf.reduce_sum(x * tf.math.log(y) + (1 - x) * tf.math.log(1 - y), 1)
        KL_divergence = self.beta * tf.reduce_sum(tf.square(self.mu) + tf.square(self.sigma) - tf.math.log(1e-8 + tf.square(self.sigma)) - 1, 1)
        marginal_likelihood = tf.reduce_mean(marginal_likelihood)
        KL_divergence = tf.reduce_mean(KL_divergence)
        ELBO = marginal_likelihood - KL_divergence
        return -ELBO

    def fit(self, x, **kwargs):
        self._build_net(x.shape[-1])
        self.optimizer._hyper['learning_rate'] = self.learning_rate
        self.model.compile(self.optimizer, loss=self.vae_loss, metrics=['binary_crossentropy'])

        # data argumentation
        #x = np.concatenate([x for i in range(5)], axis=0)
        hist = self.model.fit(x, x, **kwargs)
        print(len(hist.history['loss']), hist.history['loss'][-1])

    def predict(self, x, **kwargs):
        return self.encoder.predict(x)[0]

    def fit_transform(self, x, **kwargs):
        self.fit(x, **kwargs)
        return self.predict(x, **kwargs)


class ConvolutionalVAE:
    def __init__(self, latent_dim, **kwargs):
        self.latent_dim = latent_dim
        self.hyperparameter = kwargs
        self.optimizer = get(kwargs['optimizer'])
        try:
            self.learning_rate = kwargs['learning_rate']
        except KeyError:
            self.learning_rate = 1e-4
        try:
            self.beta = kwargs['beta']
        except KeyError:
            self.beta = 0.9

    def _build_net(self, origin_dim):
        self.inputs = Input(shape=(origin_dim))
        self.encoder = self.__encoder(origin_dim)
        self.decoder = self.__decoder(origin_dim)
        outputs = self.decoder(self.encoder(self.inputs)[-1])
        self.model = Model(inputs=self.inputs, outputs=outputs, name='AE')

    def __encoder(self, origin_dim, layers=2):
        hidden = Reshape((8,8,1))(self.inputs)
        hidden = Conv2D(16, (3, 3), activation='relu', padding='same')(hidden)
        hidden = MaxPooling2D((2, 2), padding='same')(hidden)
        hidden = Conv2D(8, (3, 3), activation='relu', padding='same')(hidden)
        hidden = MaxPooling2D((2, 2), padding='same')(hidden)
        self.h1 = hidden.shape
        hidden = Flatten()(hidden)
        self.h2 = hidden.shape[-1]
        hidden = downsample(hidden)
        self.mu = Dense(self.latent_dim, activation='relu')(hidden)
        self.sigma = Dense(self.latent_dim, activation='softplus')(hidden)
        z = Sampling()([self.mu, self.sigma])
        return Model(inputs=self.inputs, outputs=[self.mu, self.sigma, z], name='vEncoder')

    def __decoder(self, origin_dim, layers=2):
        inputs = Input(shape=(self.latent_dim))
        hidden = upsample(inputs)
        hidden = Dense(self.h2, activation='relu')(hidden)
        hidden = Reshape(self.h1[1:])(hidden)
        hidden = Conv2D(8, (3, 3), activation='relu', padding='same')(hidden)
        hidden = UpSampling2D((2, 2))(hidden)
        hidden = Conv2D(16, (3, 3), activation='relu', padding='same')(hidden)
        hidden = UpSampling2D((2, 2))(hidden)
        hidden = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(hidden)
        outputs = Flatten()(hidden)
        return Model(inputs=inputs, outputs=outputs, name='vDecoder')

    def vae_loss(self, x, y):
        marginal_likelihood = tf.reduce_sum(x * tf.math.log(y) + (1 - x) * tf.math.log(1 - y), 1)
        KL_divergence = self.beta * tf.reduce_sum(tf.square(self.mu) + tf.square(self.sigma) - tf.math.log(1e-8 + tf.square(self.sigma)) - 1, 1)
        marginal_likelihood = tf.reduce_mean(marginal_likelihood)
        KL_divergence = tf.reduce_mean(KL_divergence)
        ELBO = marginal_likelihood - self.beta * KL_divergence
        return -ELBO

    def fit(self, x, **kwargs):
        self._build_net(x.shape[-1])
        self.optimizer._hyper['learning_rate'] = self.learning_rate
        self.model.compile(self.optimizer, loss=self.vae_loss, metrics=['binary_crossentropy'])
        hist = self.model.fit(x, x, **kwargs)
        print(len(hist.history['loss']), hist.history['loss'][-1])

    def predict(self, x, **kwargs):
        return self.encoder.predict(x)[0]

    def fit_transform(self, x, **kwargs):
        self.fit(x, **kwargs)
        return self.predict(x, **kwargs)
