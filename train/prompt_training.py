import collections.abc
import os
import time
from typing import Optional, List

import torch
import torch.distributed
import transformers
from packaging import version
from torch.optim import SGD
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers.file_utils import is_datasets_available
from transformers.integrations import (
    is_comet_available,
    is_optuna_available,
    is_ray_available,
    is_tensorboard_available,
    is_wandb_available,
)
from transformers.optimization import AdamW, get_scheduler
from transformers.trainer_callback import (
    DefaultFlowCallback,
    ProgressCallback,
)
from transformers.trainer_callback import TrainerState
from transformers.trainer_utils import TrainOutput, EvalLoopOutput
from transformers.utils import logging

from trainer import Trainer

logger = logging.get_logger(__name__)


class PromptTrainer(Trainer):
    """
    this is a trainer for p-tuning
    """

    def __init__(self, args, model, tokenizer, data_collator=None, train_dataset=None, eval_dataset=None,
                 compute_metrics=None, tb_writer=None):
        super(PromptTrainer, self).__init__(model=model, args=args, data_collator=data_collator,
                                            train_dataset=train_dataset, eval_dataset=eval_dataset,
                                            compute_metrics=compute_metrics, tb_writer=tb_writer)
        self.tokenizer = tokenizer
        self.args = args
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tb_writer = tb_writer
        self.data_collator = data_collator
        self.compute_metrics = compute_metrics

    def train(self, model_path=None:
        """
        this is a training process for prompt tuning
        support:    multi-gpu training
                    custom parameters for training
                    custom lr scheduler
                    custom optimizer
                    custom data_collator
                    custom compute_metrics
                    logging with tensorboard
        :param **kwargs:
        :param model_path:
        :return:
        """
        # Data loading

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Based on Transformers' default one, we add fixing layer option where the bottom n layers' parameters
        are fixed and only the top layers will be fine-tuned.
        """
        if self.args.hf_inference_model:
            return

        if self.optimizer is None:
            params = {}
            for n, p in self.model.named_parameters():
                if self.args.fix_layers > 0:
                    if 'encoder.layer' in n:
                        try:
                            layer_num = int(n[n.find('encoder.layer') + 14:].split('.')[0])
                        except ValueError:
                            raise ValueError(f"Unexpected error when going through {n}, check its name")

                        if layer_num >= self.args.fix_layers:
                            print('yes', n)
                            params[n] = p
                        else:
                            print('no ', n)

                    elif 'embeddings' in n:
                        print('no ', n)
                    else:
                        print('yes', n)
                        params[n] = p
                else:
                    params[n] = p
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in params.items() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in params.items() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
            if self.args.optimizer == 'adam':
                self.optimizer = AdamW(
                    optimizer_grouped_parameters,
                    lr=self.args.learning_rate,
                    betas=(self.args.adam_beta1, self.args.adam_beta2),
                    eps=self.args.adam_epsilon,
                )
            elif self.args.optimizer == 'sgd':
                self.optimizer = SGD(
                    optimizer_grouped_parameters,
                    lr=self.args.learning_rate
                )
            else:
                raise NotImplementedError

        if self.lr_scheduler is None:
            self.lr_scheduler = get_scheduler(
                self.args.lr_scheduler_type,
                optimizer=self.optimizer,
                num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
            )
