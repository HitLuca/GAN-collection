import os

import numpy as np
from keras.utils import plot_model
from numpy import ndarray

from gan_collection.models.abstract_gan.abstract_gan_model import AbstractGAN
from gan_collection.models.dcgan import dcgan_utils
from gan_collection.utils.utils import plot_save_losses, plot_save_latent_space, plot_save_samples

generator_lr = 0.0005
discriminator_lr = 0.0005
batch_size = 128
n_generator = 1
n_discriminator = 1


class DCGAN(AbstractGAN):
    def __init__(self, run_dir: str, outputs_dir: str, model_dir: str, generated_datasets_dir: str, resolution: int,
                 channels: int, epochs: int, output_save_frequency: int, model_save_frequency: int,
                 loss_save_frequency: int, latent_space_save_frequency: int, dataset_generation_frequency: int,
                 dataset_size: int, latent_dim: int):

        super().__init__(run_dir=run_dir, outputs_dir=outputs_dir, model_dir=model_dir,
                         generated_datasets_dir=generated_datasets_dir, resolution=resolution, channels=channels,
                         epochs=epochs, output_save_frequency=output_save_frequency,
                         model_save_frequency=model_save_frequency, loss_save_frequency=loss_save_frequency,
                         latent_space_save_frequency=latent_space_save_frequency,
                         dataset_generation_frequency=dataset_generation_frequency, dataset_size=dataset_size,
                         latent_dim=latent_dim)
        self._generator_lr = generator_lr
        self._discriminator_lr = discriminator_lr
        self._batch_size = batch_size
        self._n_generator = n_generator
        self._n_discriminator = n_discriminator

        self._losses = [[], []]

        self._build_models()

    def _build_models(self) -> None:
        self._generator = dcgan_utils.build_generator(self._latent_dim, self._resolution)
        self._discriminator = dcgan_utils.build_discriminator(self._resolution)

        self._generator_model = dcgan_utils.build_generator_model(self._generator,
                                                                  self._discriminator,
                                                                  self._latent_dim,
                                                                  self._generator_lr)

        self._discriminator_model = dcgan_utils.build_discriminator_model(self._generator,
                                                                          self._discriminator,
                                                                          self._latent_dim,
                                                                          self._resolution,
                                                                          self._discriminator_lr)

    def _save_models_architectures(self) -> None:
        plot_model(self._generator, to_file=self._run_dir + 'generator.png')
        plot_model(self._discriminator, to_file=self._run_dir + 'discriminator.png')

    def train(self, dataset: ndarray, *_) -> list:
        ones = np.ones((self._batch_size, 1))
        zeros = np.zeros((self._batch_size, 1))

        while self._epoch < self._epochs:
            self._epoch += 1
            discriminator_losses = []
            for _ in range(self._n_discriminator):
                indexes = np.random.randint(0, dataset.shape[0], self._batch_size)
                real_samples = dataset[indexes]
                noise = np.random.normal(0, 1, (self._batch_size, self._latent_dim))
                inputs = [real_samples, noise]

                discriminator_losses.append(
                    self._discriminator_model.train_on_batch(inputs, [ones, zeros])[0])
            discriminator_loss = np.mean(discriminator_losses)

            generator_losses = []
            for _ in range(self._n_generator):
                noise = np.random.normal(0, 1, (self._batch_size, self._latent_dim))
                inputs = noise

                generator_losses = self._generator_model.train_on_batch(inputs, ones)
            generator_loss = np.mean(generator_losses)

            generator_loss = float(generator_loss)
            discriminator_loss = float(discriminator_loss)

            self._losses[0].append(generator_loss)
            self._losses[1].append(discriminator_loss)

            print("%d [D loss: %+.6f] [G loss: %+.6f]" % (
                self._epoch, discriminator_loss, generator_loss))

            if self._epoch % self._loss_save_frequency == 0 and self._loss_save_frequency > 0:
                self._save_losses()

            if self._epoch % self._output_save_frequency == 0 and self._output_save_frequency > 0:
                self._save_outputs()

            if self._epoch % self._latent_space_save_frequency == 0 and self._latent_space_save_frequency > 0:
                self._save_latent_space()

            if self._epoch % self._model_save_frequency == 0 and self._model_save_frequency > 0:
                self._save_models()

            if self._epoch % self._dataset_generation_frequency == 0 and self._dataset_generation_frequency > 0:
                self._generate_dataset()

        self._generate_dataset()
        self._save_losses()
        self._save_models()
        self._save_outputs()
        self._save_latent_space()

        return self._losses

    def _save_outputs(self) -> None:
        noise = np.random.normal(0, 1, (self._outputs_rows * self._outputs_columns, self._latent_dim))
        generated_samples = self._generator.predict(noise)

        plot_save_samples(generated_samples, self._outputs_rows, self._outputs_columns, self._resolution,
                          self._channels, self._outputs_dir, self._epoch)

    def _save_latent_space(self) -> None:
        latent_space_inputs = np.zeros((self._latent_space_rows * self._latent_space_columns, self._latent_dim))

        for i, v_i in enumerate(np.linspace(-1.5, 1.5, self._latent_space_rows, True)):
            for j, v_j in enumerate(np.linspace(-1.5, 1.5, self._latent_space_columns, True)):
                latent_space_inputs[i * self._latent_space_rows + j, :2] = [v_i, v_j]

        generated_data = self._generator.predict(latent_space_inputs)

        plot_save_latent_space(generated_data, self._latent_space_rows, self._latent_space_columns,
                               self._resolution, self._channels, self._outputs_dir, self._epoch)

    def _save_losses(self) -> None:
        plot_save_losses(self._losses[:2], ['generator', 'discriminator'], self._outputs_dir, 'gan_loss')

    def _save_models(self) -> None:
        root_dir = self._model_dir + str(self._epoch) + '/'
        os.mkdir(root_dir)
        self._discriminator_model.save(root_dir + 'discriminator_model.h5')
        self._generator_model.save(root_dir + 'generator.h5')
        self._generator.save(root_dir + 'generator.h5')
        self._discriminator.save(root_dir + 'discriminator.h5')

    def _generate_dataset(self) -> None:
        z_samples = np.random.normal(0, 1, (self._dataset_size, self._latent_dim))
        generated_dataset = self._generator.predict(z_samples)
        np.save(self._generated_datasets_dir + ('/%d_generated_data' % self._epoch), generated_dataset)
