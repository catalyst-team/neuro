from typing import Any, Mapping
import logging

from generator_coords import CoordsGenerator

import torch
from torch.utils.data import DataLoader

from catalyst.dl.runner import Runner
from catalyst.utils.tools.typing import Device, Model

logger = logging.getLogger(__name__)


class NeuroRunner(Runner):
    """
    Deep Learning Runner for different NeuroRunner.
    """

    def __init__(
        self,
        model: Model = None,
        device: Device = None,
        input_key: Any = "features",
        output_key: Any = "logits",
        input_target_key: str = "targets",
        n_samples=None,
        subvolume_shape=None,
        volume_shape=None,
        batch_size=None,
    ):
        """
        Args:
            model: neuro models
            device: runner's device
            input_batch_keys: list of strings of keys for batch elements,
                e.g. ``input_batch_keys = ["features", "targets"]`` and your
                DataLoader returns 2 tensors (images and targets)
                when state.input will be
                ``{"features": batch[0], "targets": batch[1]}``
            n_samples
            subvolume_shape
            volume_shape
        """
        super(Runner).__init__(model=model, device=device)
        self.input_key = input_key
        self.output_key = output_key
        self.target_key = input_target_key
        self.subvolume_shape = subvolume_shape
        self.volume_shape = volume_shape
        self.generator = CoordsGenerator(
            self.volume_shape, self.subvolume_shape
        )
        self.n_samples = n_samples
        self.batch_size = batch_size

        if isinstance(self.input_key, str):
            # when model expects value
            self._process_input = self._process_input_str
        elif isinstance(self.input_key, (list, tuple)):
            # when model expects tuple
            self._process_input = self._process_input_list
        elif self.input_key is None:
            # when model expects dict
            self._process_input = self._process_input_none
        else:
            raise NotImplementedError()

        if isinstance(output_key, str):
            # when model returns value
            self._process_output = self._process_output_str
        elif isinstance(output_key, (list, tuple)):
            # when model returns tuple
            self._process_output = self._process_output_list
        elif self.output_key is None:
            # when model returns dict
            self._process_output = self._process_output_none
        else:
            raise NotImplementedError()

    def _crop_data(self, batch, coords):
        output = {}
        for key, _ in batch.items():
            if key == self.input_key:
                x = torch.zeros(
                    1,
                    self.subvolume_shape[0],
                    self.subvolume_shape[1],
                    self.subvolume_shape[2],
                )

                x[
                    0,
                    :self.subvolume_shape[0],
                    :self.subvolume_shape[1],
                    :self.subvolume_shape[2],
                ] = batch[key][
                    0,
                    coords[0][0]:coords[0][1],
                    coords[1][0]:coords[1][1],
                    coords[2][0]:coords[2][1],
                ]
                output[self.input_key] = x.unsqueeze(0)
            elif key == self.target_key:
                y = torch.zeros(
                    1,
                    batch[key].size()[1],
                    self.subvolume_shape[0],
                    self.subvolume_shape[1],
                    self.subvolume_shape[2],
                )
                y[
                    0,
                    :,
                    :self.subvolume_shape[0],
                    :self.subvolume_shape[1],
                    :self.subvolume_shape[2],
                ] = batch[key][
                    0,
                    :,
                    coords[0][0]:coords[0][1],
                    coords[1][0]:coords[1][1],
                    coords[2][0]:coords[2][1],
                ]
                output[key] = y
        return output

    def _run_batch(self, batch: Mapping[str, Any], coords) -> None:
        """
        Inner method to run train step on specified data batch,
        with batch callbacks events.
        Args:
            batch (Mapping[str, Any]): dictionary with data batches
                from DataLoader.
            coords: random coordinates
        """
        self.state.global_step += self.state.batch_size
        batch = self._crop_data(batch, coords)
        batch = self._batch2device(batch, self.device)
        self.state.input = batch

        self._run_event("on_batch_start")
        self._handle_batch(batch=batch)
        self._run_event("on_batch_end")

    def _run_loader(self, loader: DataLoader) -> None:
        """
        Inner method to pass whole DataLoader through Runner,
        with loader callbacks events.
        Args:
            loader (DataLoader): dataloader to iterate
        """
        self.state.batch_size = (
            loader.batch_sampler.batch_size
            if loader.batch_sampler is not None
            else loader.batch_size
        )
        self.state.global_step = (
            self.state.global_step or
            self.state.global_epoch * len(loader) * self.state.batch_size
        )

        for i, batch in enumerate(loader):
            for k, coords in enumerate(
                self.generator.get_coordinates(self.n_samples)
            ):
                self.state.loader_step = i + 1 + k
                self._run_batch(batch, coords)
                if self.state.need_early_stop:
                    self.state.need_early_stop = False
                    break


__all__ = ["NeuroRunner"]
