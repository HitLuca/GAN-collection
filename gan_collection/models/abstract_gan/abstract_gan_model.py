import abc

from numpy import ndarray


class AbstractGAN(abc.ABC):
    def __init__(self, run_dir: str, outputs_dir: str, model_dir: str, generated_datasets_dir: str,
                 resolution: int, channels: int, epochs: int, output_save_frequency: int,
                 model_save_frequency: int, loss_save_frequency: int,
                 latent_space_save_frequency: int, dataset_generation_frequency: int, dataset_size: int,
                 latent_dim: int, latent_space_rows: int = 6, latent_space_columns: int = 6, outputs_rows: int = 6,
                 outputs_columns: int = 6):
        self._run_dir = run_dir
        self._outputs_dir = outputs_dir
        self._model_dir = model_dir
        self._generated_datasets_dir = generated_datasets_dir

        self._resolution = resolution
        self._channels = channels
        self._epochs = epochs
        self._output_save_frequency = output_save_frequency
        self._model_save_frequency = model_save_frequency
        self._loss_save_frequency = loss_save_frequency
        self._latent_space_save_frequency = latent_space_save_frequency
        self._latent_dim = latent_dim

        self._dataset_generation_frequency = dataset_generation_frequency
        self._dataset_size = dataset_size

        self._latent_space_rows = latent_space_rows
        self._latent_space_columns = latent_space_columns

        self._outputs_rows = outputs_rows
        self._outputs_columns = outputs_columns

        self._epoch = 0

    @abc.abstractmethod
    def _build_models(self) -> None:
        pass

    @abc.abstractmethod
    def train(self, dataset: ndarray, classes: ndarray) -> list:
        pass

    @abc.abstractmethod
    def _save_models_architectures(self) -> None:
        pass

    @abc.abstractmethod
    def _save_outputs(self) -> None:
        pass

    @abc.abstractmethod
    def _save_latent_space(self) -> None:
        pass

    @abc.abstractmethod
    def _save_losses(self) -> None:
        pass

    @abc.abstractmethod
    def _save_models(self) -> None:
        pass

    @abc.abstractmethod
    def _generate_dataset(self) -> None:
        pass
