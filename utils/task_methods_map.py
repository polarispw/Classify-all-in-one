"""
This file contains a dict that maps task name to their corresponding methods
To customize a new task, add a new item to this dict and implement the methods
"""
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
)
from peft import (
    PeftModelForSequenceClassification, PromptTuningConfig, PromptEncoderConfig, LoraConfig
)

from data.dataset import CLSDataset
from models.modeling_bert import SIMBert
from train.bert_trainer import BertTrainer

task_methods_map = {
    "fine-tune": {
        "dataset": CLSDataset,
        "metric": None,
        "data_collator": DataCollatorWithPadding,
        "tokenizer": AutoTokenizer,
        "model": AutoModelForSequenceClassification,
        "trainer": BertTrainer,
    },

    "pre-train": {
        "dataset": CLSDataset,
        "metric": None,
        "data_collator": DataCollatorWithPadding,
        "tokenizer": AutoTokenizer,
        "model": SIMBert,
        "trainer": BertTrainer,
    },

    "prompt-tuning": {
        "dataset": CLSDataset,
        "metric": None,
        "data_collator": DataCollatorWithPadding,
        "tokenizer": AutoTokenizer,
        "model": PeftModelForSequenceClassification,
        "peft_config": PromptTuningConfig,
        "trainer": Trainer,
    },

    "p-tuning": {
        "dataset": CLSDataset,
        "metric": None,
        "data_collator": DataCollatorWithPadding,
        "tokenizer": AutoTokenizer,
        "model": PeftModelForSequenceClassification,
        "peft_config": PromptEncoderConfig,
        "trainer": Trainer,
    },

    "lora": {
        "dataset": CLSDataset,
        "metric": None,
        "data_collator": DataCollatorWithPadding,
        "tokenizer": AutoTokenizer,
        "model": PeftModelForSequenceClassification,
        "peft_config": LoraConfig,
        "trainer": Trainer,
    },

}
