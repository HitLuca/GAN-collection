from functools import partial

from keras import Model
from keras.layers import *
from keras.optimizers import Adam

from gan_collection.utils.gan_utils import set_model_trainable, wasserstein_loss, \
    gradient_penalty_loss, RandomWeightedAverage


def build_generator(latent_dim: int, classes_n: int, resolution: int, filters: int = 32, kernel_size: int = 3,
                    channels: int = 3) -> Model:
    image_size = 4
    filters *= int(resolution / image_size / 2)

    latent_input = Input((latent_dim,))
    conditional_input = Input((classes_n,))

    generated = Concatenate()([latent_input, conditional_input])

    generated = Dense(image_size * image_size * 16)(generated)
    generated = Activation('relu')(generated)

    generated = Reshape((image_size, image_size, 16))(generated)

    while image_size != resolution:
        generated = UpSampling2D()(generated)
        generated = Conv2D(filters, kernel_size, padding='same')(generated)
        generated = Activation('relu')(generated)
        generated = Conv2D(filters, kernel_size, padding='same')(generated)
        generated = Activation('relu')(generated)

        filters = int(filters / 2)
        image_size *= 2

    generated = Conv2D(channels, kernel_size, padding='same', activation='sigmoid')(generated)

    generator = Model([latent_input, conditional_input], generated, name='generator')
    print(generator.summary())
    return generator


def build_critic(resolution: int, classes_n: int, filters: int = 32, kernel_size: int = 3, channels: int = 3) -> Model:
    image_size = resolution

    critic_inputs = Input((resolution, resolution, channels))
    class_inputs = Input((classes_n,))
    criticized = critic_inputs

    while image_size != 4:
        criticized = Conv2D(filters, kernel_size, padding='same')(criticized)
        criticized = LeakyReLU(0.2)(criticized)
        criticized = Conv2D(filters, kernel_size, padding='same')(criticized)
        criticized = LeakyReLU(0.2)(criticized)
        criticized = MaxPooling2D()(criticized)

        filters *= 2
        image_size = int(image_size / 2)

    criticized = Flatten()(criticized)
    criticized = Concatenate()([criticized, class_inputs])

    criticized = Dense(1)(criticized)

    critic = Model([critic_inputs, class_inputs], criticized, name='critic')
    print(critic.summary())
    return critic


def build_generator_model(generator: Model, critic: Model, latent_dim: int, classes_n: int,
                          generator_lr: float) -> Model:
    set_model_trainable(generator, True)
    set_model_trainable(critic, False)

    noise_samples = Input((latent_dim,))
    class_samples = Input((classes_n,))

    generated_samples = generator([noise_samples, class_samples])

    generated_criticized = critic([generated_samples, class_samples])

    generator_model = Model([noise_samples, class_samples], generated_criticized, name='generator_model')
    generator_model.compile(optimizer=Adam(generator_lr, beta_1=0.5, beta_2=0.9), loss=wasserstein_loss)
    return generator_model


def build_critic_model(generator: Model, critic: Model, latent_dim: int, resolution: int, classes_n: int,
                       batch_size: int, critic_lr: float,
                       gradient_penalty_weight: int, channels: int = 3) -> Model:
    set_model_trainable(generator, False)
    set_model_trainable(critic, True)

    noise_samples = Input((latent_dim,))
    class_samples = Input((classes_n,))

    real_samples = Input((resolution, resolution, channels))

    generated_samples = generator([noise_samples, class_samples])
    generated_criticized = critic([generated_samples, class_samples])
    real_criticized = critic([real_samples, class_samples])

    averaged_samples = RandomWeightedAverage(batch_size)([real_samples, generated_samples])
    averaged_criticized = critic([averaged_samples, class_samples])

    partial_gp_loss = partial(gradient_penalty_loss,
                              averaged_samples=averaged_samples,
                              gradient_penalty_weight=gradient_penalty_weight)
    partial_gp_loss.__name__ = 'gradient_penalty'

    critic_model = Model([real_samples, noise_samples, class_samples],
                         [real_criticized, generated_criticized, averaged_criticized], name='critic_model')

    critic_model.compile(optimizer=Adam(critic_lr, beta_1=0.5, beta_2=0.9),
                         loss=[wasserstein_loss, wasserstein_loss, partial_gp_loss])
    return critic_model
