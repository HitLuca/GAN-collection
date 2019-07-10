from typing import List

import numpy as np
from keras import backend as K, Model
from keras.layers.merge import _Merge


def set_model_trainable(model: Model, trainable: bool) -> Model:
    model.trainable = trainable
    for l in model.layers:
        l.trainable = trainable
    return model


def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)


def apply_lr_decay(models: List[Model], lr_decay_factor: float) -> None:
    for model in models:
        lr_tensor = model.optimizer.lr
        lr = K.get_value(lr_tensor)
        K.set_value(lr_tensor, lr * lr_decay_factor)


def vae_loss(z_mean, z_log_var, real, predicted):
    def loss(y_true, y_pred):
        crossentropy_loss = K.mean(K.binary_crossentropy(real, predicted))
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return K.mean(crossentropy_loss + kl_loss)

    return loss


def gradient_penalty_loss(_, y_pred, averaged_samples, gradient_penalty_weight: int):
    gradients = K.gradients(y_pred, averaged_samples)[0]
    gradients_sqr = K.square(gradients)
    gradients_sqr_sum = K.sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)
    return K.mean(gradient_penalty)


class RandomWeightedAverage(_Merge):
    def __init__(self, batch_size, **kwargs):
        super().__init__(**kwargs)
        self._batch_size = batch_size

    def _merge_function(self, inputs):
        weights = K.random_uniform((self._batch_size, 1, 1, 1))
        averaged_inputs = (weights * inputs[0]) + ((1 - weights) * inputs[1])
        return averaged_inputs


def sampling(args):
    z_mean, z_log_var = args
    batch_size = K.shape(z_mean)[0]
    latent_dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch_size, latent_dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def mean_gaussian_negative_log_likelihood(y_true, y_pred):
    nll = 0.5 * np.log(2 * np.pi) + 0.5 * K.square(y_pred - y_true)
    axis = tuple(range(1, len(K.int_shape(y_true))))
    return K.mean(K.sum(nll, axis=axis), axis=-1)
