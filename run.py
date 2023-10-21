import logging
import os
import sys
from datetime import datetime

import torch
from filelock import FileLock
from transformers import AutoTokenizer
from transformers import HfArgumentParser, set_seed

from configs.args_list import CLSModelArguments, CLSDatasetArguments, CLSTrainingArguments
from data.dataset import CLSDataset
from data.data_collator import DataCollatorForCLS
from models.modeling_bert import BertLikeModel4CLSFT, BertLikeModel4SSIMPT
from train.base_trainer import CLSTrainer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def main():
    # parse arguments
    parser = HfArgumentParser((CLSModelArguments, CLSDatasetArguments, CLSTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script, and it's the path to a json file,
        # then parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.name_or_path,
        cache_dir=model_args.cache_dir,
    )
    if "gpt2" in model_args.name_or_path:
        tokenizer.sep_token_id = tokenizer.eos_token_id
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # create model class
    task_class_dict = {
        "fine-tune": BertLikeModel4CLSFT,
        "pre-train": BertLikeModel4SSIMPT,
    }
    model = task_class_dict[training_args.task_type](model_args)
    tokenizer.model_type = model.config.model_type

    # Get datasets
    from datasets import load_dataset
    raw_datasets = load_dataset("glue", "mrpc")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokenized_datasets = raw_datasets.map(lambda example:
                                          tokenizer(example["sentence1"], example["sentence2"], truncation=True),
                                          batched=True)

    # Initialize Trainer
    trainer = CLSTrainer(
        model,
        training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=DataCollatorForCLS(tokenizer=tokenizer),
        tokenizer=tokenizer,
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train()

    # Output Dic
    final_result = {
        'time': str(datetime.today()),
        'output_dir': training_args.output_dir
    }

    if trainer.is_world_process_zero():
        with FileLock('log.lock'):
            with open(training_args.log_file, 'a') as f:
                final_result.update(vars(model_args))
                final_result.update(vars(training_args))
                final_result.update(vars(data_args))
                if 'evaluation_strategy' in final_result:
                    final_result.pop('evaluation_strategy')
                f.write(str(final_result) + '\n')

    logger.info('****** Output Dir *******')
    logger.info(training_args.output_dir)

    return final_result


if __name__ == "__main__":
    main()
