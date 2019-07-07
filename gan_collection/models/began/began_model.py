import os

import keras.backend as K
import numpy as np
from keras.utils import plot_model
from numpy import ndarray

from gan_collection.models.abstract_gan.abstract_gan_model import AbstractGAN
from gan_collection.models.began import began_utils
from gan_collection.utils.utils import plot_save_samples, plot_save_latent_space, plot_save_losses

gamma = 0.5
lambda_k = 0.001
batch_size = 16
initial_lr = 1e-4
min_lr = 1e-5
lr_decay_rate = 0.9
k = 0
loss_exponent = 1


class BEGAN(AbstractGAN):
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

        self._gamma = gamma
        self._lambda_k = lambda_k
        self._batch_size = batch_size
        self._initial_lr = initial_lr
        self._min_lr = min_lr
        self._lr_decay_rate = lr_decay_rate
        self._k = k
        self._loss_exponent = loss_exponent

        self._lr = self._initial_lr
        self._losses = [[], [], [], []]
        self._global_step = 0

        self._build_models()
        self._save_models_architectures()

    def _build_models(self) -> None:
        self._generator = began_utils.build_decoder(self._latent_dim, self._resolution)
        self._discriminator = began_utils.build_discriminator(self._latent_dim, self._resolution)

        self._discriminator_model = began_utils.build_discriminator_model(self._discriminator, self._resolution,
                                                                          self._initial_lr, self._loss_exponent)

        self._generator_model = began_utils.build_generator_model(self._generator, self._discriminator,
                                                                  self._latent_dim, self._initial_lr,
                                                                  self._loss_exponent)

    def _save_models_architectures(self) -> None:
        plot_model(self._generator, to_file=self._run_dir + 'generator.png')
        plot_model(self._discriminator, to_file=self._run_dir + 'discriminator.png')

    def train(self, dataset: ndarray, *_) -> list:
        zeros = np.zeros(self._batch_size)
        zeros_2 = np.zeros(self._batch_size * 2)

        batches_per_epoch = dataset.shape[0] // self._batch_size
        dataset_indexes = np.arange(len(dataset))

        last_m_global = np.inf
        lr_decay_step = 0

        while self._epoch < self._epochs:
            self._epoch += 1
            np.random.shuffle(dataset_indexes)

            self._lr = max(self._initial_lr * (self._lr_decay_rate ** lr_decay_step), self._min_lr)
            K.set_value(self._generator_model.optimizer.lr, self._lr)
            K.set_value(self._discriminator_model.optimizer.lr, self._lr)

            m_history = []
            k_history = []
            generator_losses = []
            discriminator_losses = []
            for batch in range(batches_per_epoch):
                self._global_step += 1
                k_model = np.ones(self._batch_size) * self._k

                indexes = dataset_indexes[batch * self._batch_size:(batch + 1) * self._batch_size]
                real_samples = dataset[indexes]

                noise_decoder = np.random.uniform(-1, 1, (self._batch_size, self._latent_dim))
                noise_generator = np.random.uniform(-1, 1, (self._batch_size * 2, self._latent_dim))

                generated_samples = self._generator.predict(noise_decoder)

                discriminator_loss_real, discriminator_loss_generated = self._discriminator_model.predict(
                    [real_samples, generated_samples, k_model])
                self._discriminator_model.train_on_batch([real_samples, generated_samples, k_model], [zeros, zeros])

                generator_loss = self._generator_model.train_on_batch(noise_generator, zeros_2)

                discriminator_loss_real = float(np.mean(discriminator_loss_real))
                discriminator_loss_generated = float(np.mean(discriminator_loss_generated))

                discriminator_loss = float(discriminator_loss_real - self._k * discriminator_loss_generated)

                self._update_k(discriminator_loss_real, generator_loss)

                m_value = discriminator_loss_real + np.abs(self._gamma * discriminator_loss_real - generator_loss)
                m_history.append(m_value)
                generator_losses.append(generator_loss)
                discriminator_losses.append(discriminator_loss)
                k_history.append(self._k)

                self._losses[0].append(generator_loss)
                self._losses[1].append(discriminator_loss)
                self._losses[2].append(self._k)
                self._losses[3].append(m_value)

            generator_loss = float(np.mean(generator_losses))
            discriminator_loss = float(np.mean(discriminator_losses))
            k = float(np.mean(k_history))
            m_global = float(np.mean(m_history))

            if last_m_global <= m_global:  # decay LearningRate
                lr_decay_step += 1
            last_m_global = m_global

            print("%d %d [D loss: %+.6f] [G loss: %+.6f] [K: %+.6f] [M: %+.6f]" %
                  (self._epoch, self._global_step, discriminator_loss, generator_loss, k, m_global))
            print("[learning_rate: %+.6f]" % self._lr)

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
        noise = np.random.uniform(-1, 1, (self._outputs_rows * self._outputs_columns, self._latent_dim))
        generated_samples = self._generator.predict(noise)

        plot_save_samples(generated_samples, self._outputs_rows, self._outputs_columns, self._resolution,
                          self._channels, self._outputs_dir, self._epoch)

    def _save_latent_space(self) -> None:
        latent_space_inputs = np.zeros((self._latent_space_rows * self._latent_space_columns, self._latent_dim))

        for i, v_i in enumerate(np.linspace(-1, 1, self._latent_space_rows, True)):
            for j, v_j in enumerate(np.linspace(-1, 1, self._latent_space_columns, True)):
                latent_space_inputs[i * self._latent_space_rows + j, :2] = [v_i, v_j]

        generated_data = self._generator.predict(latent_space_inputs)

        plot_save_latent_space(generated_data, self._latent_space_rows, self._latent_space_columns,
                               self._resolution, self._channels, self._outputs_dir, self._epoch)

    def _save_losses(self) -> None:
        plot_save_losses(self._losses[:2], ['generator', 'discriminator'], self._outputs_dir, 'gan_loss')
        plot_save_losses(self._losses[2:3], ['k'], self._outputs_dir, 'k')
        plot_save_losses(self._losses[3:4], ['m_value'], self._outputs_dir, 'm_value')

    def _save_models(self) -> None:
        root_dir = self._model_dir + str(self._epoch) + '/'
        os.mkdir(root_dir)
        self._discriminator_model.save(root_dir + 'discriminator_model.h5')
        self._generator_model.save(root_dir + 'generator.h5')
        self._generator.save(root_dir + 'generator.h5')
        self._discriminator.save(root_dir + 'discriminator.h5')

    def _generate_dataset(self) -> None:
        z_samples = np.random.uniform(-1, 1, (self._dataset_size, self._latent_dim))
        generated_dataset = self._generator.predict(z_samples)
        np.save(self._generated_datasets_dir + ('/%d_generated_data' % self._epoch), generated_dataset.astype)

    def _update_k(self, discriminator_loss_real: float, discriminator_loss_generated: float) -> None:
        self._k = self._k + self._lambda_k * (
                self._gamma * discriminator_loss_real - discriminator_loss_generated)
        self._k = np.clip(self._k, 0.0, 1.0)
