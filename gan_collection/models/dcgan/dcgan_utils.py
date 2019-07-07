from keras import Model, Input
from keras.layers import Dense, LeakyReLU, Reshape, Conv2D, Flatten, Conv2DTranspose, BatchNormalization, \
    Dropout, UpSampling2D, MaxPooling2D
from keras.optimizers import Adam

from gan_collection.utils.gan_utils import set_model_trainable


def build_generator(latent_dim: int, resolution: int, filters: int = 32, kernel_size: int = 3,
                    channels: int = 3) -> Model:
    image_size = 4

    filters *= int(resolution / image_size / 2)
    generator_inputs = Input((latent_dim,))
    generated = generator_inputs

    generated = Dense(image_size * image_size * 16)(generated)
    generated = LeakyReLU(0.2)(generated)

    generated = Reshape((image_size, image_size, 16))(generated)

    while image_size != resolution:
        generated = UpSampling2D()(generated)
        generated = Conv2D(filters, kernel_size, padding='same')(generated)
        generated = LeakyReLU(0.2)(generated)

        filters = int(filters / 2)
        image_size *= 2

    generated = Conv2D(channels, kernel_size, padding='same', activation='sigmoid')(generated)

    generator = Model(generator_inputs, generated, name='generator')
    print(generator.summary())
    return generator


def build_discriminator(resolution: int, filters: int = 32, kernel_size: int = 3, channels: int = 3) -> Model:
    image_size = resolution

    discriminator_inputs = Input((resolution, resolution, channels))
    discriminated = discriminator_inputs

    while image_size != 4:
        discriminated = Conv2D(filters, kernel_size, padding='same')(discriminated)
        discriminated = LeakyReLU(0.2)(discriminated)
        discriminated = MaxPooling2D()(discriminated)

        filters *= 2
        image_size = int(image_size / 2)

    discriminated = Flatten()(discriminated)

    discriminated = Dense(1, activation='sigmoid')(discriminated)

    discriminator = Model(discriminator_inputs, discriminated, name='discriminator')
    print(discriminator.summary())
    return discriminator


def build_generator_model(generator: Model, discriminator: Model, latent_dim: int, generator_lr: float) -> Model:
    set_model_trainable(generator, True)
    set_model_trainable(discriminator, False)

    noise_samples = Input((latent_dim,))
    generated_samples = generator(noise_samples)

    generated_discriminated = discriminator(generated_samples)

    generator_model = Model(noise_samples, generated_discriminated, name='generator_model')
    generator_model.compile(optimizer=Adam(generator_lr, beta_1=0.5, beta_2=0.9), loss='binary_crossentropy')
    return generator_model


def build_discriminator_model(generator: Model, discriminator: Model, latent_dim: int, resolution: int,
                              discriminator_lr: float, channels: int = 3) -> Model:
    set_model_trainable(generator, False)
    set_model_trainable(discriminator, True)

    noise_samples = Input((latent_dim,))
    real_samples = Input((resolution, resolution, channels))

    generated_samples = generator(noise_samples)
    generated_discriminated = discriminator(generated_samples)
    real_discriminated = discriminator(real_samples)

    discriminator_model = Model([real_samples, noise_samples],
                                [real_discriminated, generated_discriminated], name='discriminator_model')

    discriminator_model.compile(optimizer=Adam(discriminator_lr, beta_1=0.5, beta_2=0.9),
                                loss=['binary_crossentropy', 'binary_crossentropy'])
    return discriminator_model
