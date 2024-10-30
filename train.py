import torch
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
from pytorch_lightning.loggers import WandbLogger
from torch import nn
import lightning as pl
from model import ConformerEncoderForPretrain
from dataset import get_loaders
from transformers import get_cosine_schedule_with_warmup
import wandb
import yaml
from datetime import datetime

now = datetime.now()
millis_string = now.strftime("%Y-%m-%d %H:%M:%S.%f")
project_name = 'ASR'
experiment_name = f"{project_name}_{millis_string}"
config = yaml.safe_load(open('config.yml'))

train_config = config['train_params']
train_loader, valid_loader = get_loaders(batch_size=train_config['batch_size'])

max_steps = config['max_steps']
warmup_steps = 0.1 * max_steps
class Module(pl.LightningModule):
    def __init__(self):
        super(Module, self).__init__()
        self.model = ConformerEncoderForPretrain(config)
        self.train_loader = train_loader
        self.valid_loader = valid_loader

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=train_config['learning_rate'],
                                      weight_decay=train_config['weight_decay'],
                                      fused = True, betas = (0.9, 0.999))
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=max_steps
        )
        sch = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1,
        }
        return [optimizer], [sch]

    def configure_model(self) -> None:
        self.model = ConformerEncoderForPretrain(config)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self.train_loader

    def val_dataloader(self) -> TRAIN_DATALOADERS:
        return self.valid_loader

    def compute_grad_norm(self):
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm

    def training_step(self, batch, batch_idx):
        loss = self.model(**batch)
        grad_norm = self.compute_grad_norm()
        self.log_dict({
            'train_loss': loss.item(),
            'train_grad_norm': grad_norm
        }, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.model(**batch)
        grad_norm = self.compute_grad_norm()
        self.log_dict({
            'valid_loss': loss.item(),
            'valid_grad_norm': grad_norm
        }, prog_bar=True, sync_dist=True)
        return loss

logger = WandbLogger(project=project_name, name=experiment_name)
checkpoint_callback = pl.pytorch.callbacks.ModelCheckpoint(
    monitor='valid_loss',
    dirpath='./checkpoints',
    mode="min",
    filename='{epoch:02d}-{train_loss:.2f}-{valid_loss:.2f}',
    every_n_train_steps=20000,
    save_weights_only=False,
    save_top_k = 5)

learning_rate_monitor = pl.pytorch.callbacks.LearningRateMonitor(logging_interval="step",
                                                                 log_momentum=True,
                                                                 log_weight_decay=True)

trainer = pl.Trainer(
    devices='auto',
    callbacks=[checkpoint_callback, learning_rate_monitor],
    enable_checkpointing=True,
    log_every_n_steps=15,
    num_nodes=1,
    accelerator="gpu",
    max_steps=max_steps,
    logger=logger,
    precision="bf16-mixed",
    accumulate_grad_batches=4,
    val_check_interval = 15000
)

module = Module()
trainer.fit(module)





