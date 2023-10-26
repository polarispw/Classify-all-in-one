import json
import logging
import os
import sys
from dataclasses import asdict
from datetime import datetime

from filelock import FileLock
from peft import get_peft_model
from transformers import AutoTokenizer
from transformers import HfArgumentParser

from config.args_list import CLSModelArguments, CLSDatasetArguments, CLSTrainingArguments
from utils.task_methods_map import task_methods_map

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def main():
    # Parse arguments
    parser = HfArgumentParser((CLSModelArguments, CLSDatasetArguments, CLSTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    task_type = training_args.task_type

    # Setup logging
    logging.basicConfig(
        filename="tmp_log.txt",
        filemode='a',
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

    # Create tokenizer
    tokenizer = task_methods_map[task_type]['tokenizer'].from_pretrained(
        model_args.name_or_path,
        cache_dir=model_args.cache_dir,
    )

    # Load dataset
    data_manager = task_methods_map[task_type]['dataset'](args=data_args, tokenizer=tokenizer)
    # split and tokenized automatically
    # preprocess the dataset according to the input
    # tokenized_datasets = dataset.tokenize_dataset(col_names=data_args.col_names)

    data_collator = task_methods_map[task_type]['data_collator'](tokenizer=tokenizer)

    # dataset for test running
    from datasets import load_dataset
    raw_datasets = load_dataset("glue", "mrpc")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokenized_datasets = raw_datasets.map(lambda example:
                                          tokenizer(example["sentence1"], example["sentence2"], truncation=True),
                                          batched=True)
    print(tokenized_datasets,
          len(tokenized_datasets['train'][0]['input_ids']),
          len(tokenized_datasets['train'][0]['sentence1']),
          len(tokenized_datasets['train'][0]['sentence2']),  sep="\n")
    # Load metric
    metric = task_methods_map[task_type]['metric']

    # Create model
    if task_type == 'pre-train':
        model = task_methods_map[task_type]['model'](model_args)
    elif task_type == 'fine-tune':
        model = task_methods_map[task_type]['model'].from_pretrained(
            model_args.name_or_path,
            cache_dir=model_args.cache_dir,
        )
    else:
        model = task_methods_map[task_type]['model'].from_pretrained(
            model_args.name_or_path,
            cache_dir=model_args.cache_dir,
        )
        peft_config = task_methods_map[task_type]['peft_config'](
            task_type="SEQ_CLS",
            num_virtual_tokens=training_args.num_virtual_tokens,
            encoder_hidden_size=training_args.encoder_hidden_size,
        )
        model = get_peft_model(model, peft_config)

    # Initialize Trainer
    trainer = task_methods_map[task_type]['trainer'](
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=metric,
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train()

    final_result = {
        'time': str(datetime.today()),
        'output_dir': training_args.output_dir
    }

    # Inference
    if training_args.do_eval:
        ...
        # eval_result = trainer.evaluate()

    if training_args.do_predict:
        ...
        # predict_result = trainer.predict(tokenized_datasets["test"])

    # Save args and logs
    if trainer.is_world_process_zero():
        with FileLock('log.lock'):
            with open(training_args.log_file, 'a') as f:
                # transfer model_args, data_args, training_args to dict
                final_result["model_args"] = asdict(model_args)
                final_result["data_args"] = asdict(data_args)
                final_result["training_args"] = asdict(training_args)
                final_result = json.dumps(final_result, indent=4)
                f.write(str(final_result) + '\n')
                with open("tmp_log.txt", 'r') as tmp:
                    f.write(tmp.read())

    logger.info('****** Output Dir *******')
    logger.info(training_args.output_dir)

    return final_result


if __name__ == "__main__":
    main()
    logging.shutdown()
    os.remove("tmp_log.txt")
