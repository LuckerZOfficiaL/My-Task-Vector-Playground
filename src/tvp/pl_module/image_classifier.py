import logging
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import hydra
import omegaconf
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import torchmetrics
from torch.optim import Optimizer

from nn_core.common import PROJECT_ROOT
from nn_core.model_logging import NNLogger

from tvp.data.datamodule import MetaData
from tvp.data.datasets.common import maybe_dictionarize
from tvp.utils.utils import torch_load, torch_save

from src.tvp.modules.cosine_annealing_lr_scheduler import CosineAnnealingLRScheduler

pylogger = logging.getLogger(__name__)


class ImageClassifier(pl.LightningModule):
    logger: NNLogger

    def __init__(self, encoder, classifier, metadata: Optional[MetaData] = None, *args, **kwargs) -> None:
        super().__init__()

        # Populate self.hparams with args and kwargs automagically!
        # We want to skip metadata since it is saved separately by the NNCheckpointIO object.
        # Be careful when modifying this instruction. If in doubt, don't do it :]
        self.save_hyperparameters(logger=False, ignore=("metadata",))

        self.sparsity_percentile = 0.1

        self.metadata = metadata
        self.num_classes = classifier.out_features

        metric = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes, top_k=1)
        self.train_acc = metric.clone()
        self.val_acc = metric.clone()
        self.test_acc = metric.clone()

        self.encoder = encoder
        self.classification_head = classifier
        # self.encoder.create_tv_mask() # call this to create the TV sparsity mask in the encoder, the mask is applied to the gradient to prevent pruned weights from updating

        self.max_train_steps = None

        self.elapsed_train_steps = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Method for the forward pass.

        'training_step', 'validation_step' and 'test_step' should call
        this method in order to compute the output predictions and the loss.

        Returns:
            output_dict: forward output containing the predictions (output logits ecc...) and the loss if any.
        """
        embeddings = self.encoder(x)

        logits = self.classification_head(embeddings)

        return logits

    def _step(self, batch: Dict[str, torch.Tensor], split: str) -> Mapping[str, Any]:
        batch = maybe_dictionarize(batch, self.hparams.x_key, self.hparams.y_key)

        x = batch[self.hparams.x_key]
        gt_y = batch[self.hparams.y_key]

        logits = self(x)
        loss = F.cross_entropy(logits, gt_y)
        preds = torch.softmax(logits, dim=-1)

        metrics: torchmetrics.Accuracy = getattr(self, f"{split}_acc")
        metrics.update(preds, gt_y)

        self.log_dict(
            {
                f"acc/{split}": metrics,
                f"loss/{split}": loss,
            },
            on_epoch=True,
        )

        if split == "train":
            self.elapsed_train_steps += 1
            self.log_dict(
                {
                    f"elapsed_train_steps": self.elapsed_train_steps,
                    f"epoch": self.current_epoch,
                    f"lr": self.trainer.optimizers[0].param_groups[0]["lr"],
                },
                on_step=True,
            )

        return {"logits": logits.detach(), "loss": loss}

    def freeze_head(self):
        self.classification_head.weight.requires_grad_(False)
        self.classification_head.bias.requires_grad_(False)

    def training_step(self, batch: Any, batch_idx: int) -> Mapping[str, Any]:
        result = self._step(batch=batch, split="train")        
        return result
    
    
    """
    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.encoder.reset_weights_by_percentile(percentile=self.sparsity_percentile)
    """

    """
    def on_train_end(self):
        self.encoder.reset_weights_by_percentile(percentile=self.sparsity_percentile)
    """

    def _set_cosine_annealing_lr_scheduler_total_steps(self):
        if "lr_scheduler" in self.hparams:

            lr_schedulers = self.lr_schedulers()

            if type(lr_schedulers) == list:
                lr_scheduler = lr_schedulers[0]
            else:
                lr_scheduler = lr_schedulers

            if lr_scheduler.__class__.__name__ == "CosineAnnealingLRScheduler":

                lr_scheduler: CosineAnnealingLRScheduler = lr_scheduler

                lr_scheduler.total_steps = self.max_train_steps

    def on_train_start(self):
        self._set_cosine_annealing_lr_scheduler_total_steps()
        
    """
    def on_after_backward(self): # after backprop, we apply the binary mask element-wise to the gradient to prevent some weights from updating, maintaining TV sparsity
        if self.encoder.tv_mask is not None:
            for name, param in self.encoder.model.named_parameters():
                if name in self.encoder.tv_mask and param.grad is not None:
                    param.grad *= self.encoder.tv_mask[name].to(param.device)
    """

    def validation_step(self, batch: Any, batch_idx: int) -> Mapping[str, Any]:
        return self._step(batch=batch, split="val")

    def test_step(self, batch: Any, batch_idx: int) -> Mapping[str, Any]:
        return self._step(batch=batch, split="test")

    def configure_optimizers(self):
        opt = hydra.utils.instantiate(self.hparams.optimizer, params=self.parameters(), _convert_="partial")
        if "lr_scheduler" not in self.hparams:
            return [opt]
        
        scheduler = CosineAnnealingLRScheduler(
            optimizer=opt,
            base_lr=self.hparams.optimizer.lr,
            warmup_length=self.hparams.lr_scheduler.warmup_length,
            total_steps=self.hparams.lr_scheduler.total_steps
        )
        scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        }
        return [opt], [scheduler_config]

        

    def __call__(self, inputs):
        return self.forward(inputs)

    def save(self, filename):
        print(f"Saving image classifier to {filename}")
        torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f"Loading image classifier from {filename}")
        return torch_load(filename)

    """def on_train_epoch_end(self):
        self.log_epoch_end_metrics("train")

    def on_validation_epoch_end(self):
        self.log_epoch_end_metrics("val")

    def on_test_epoch_end(self):
        self.log_epoch_end_metrics("test")

    def log_epoch_end_metrics(self, split):
        print(f"Epoch {self.current_epoch} ended.")
        metrics = getattr(self, f"{split}_acc")
        accuracy = metrics.compute()
        tv_sparsity = self.encoder.get_tv_sparsity()
        print(f"{split.capitalize()} Accuracy: {accuracy}")
        print(f"TV Sparsity: {tv_sparsity}")
        self.log(f"acc/{split}_epoch_end", accuracy, on_epoch=True, prog_bar=True)
        metrics.reset()"""