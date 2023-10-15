import os
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import Trainer, TrainingArguments


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

    def get_train_dataloader(self):
        """
        this function is used to get the train dataloader
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        train_sampler = self.args.train_sampler(self.train_dataset)
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.args.train_batch_size,
                                           sampler=train_sampler, collate_fn=self.data_collator,
                                           drop_last=self.args.dataloader_drop_last)
