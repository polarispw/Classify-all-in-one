from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification
from transformers import Trainer
import numpy as np
import evaluate

from configs.args_list import CLSTrainingArguments


raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
print(tokenized_datasets["train"][0])


def compute_metrics(eval_preds):
    metric = evaluate.load("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


training_args = CLSTrainingArguments(output_dir="../archive",
                                     evaluation_strategy="epoch",
                                     max_steps=500,
                                     learning_rate=2e-5,
                                     logging_steps=10,
                                     save_steps=500,
                                     )
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2, cache_dir="../model_cache")

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

# visualize the training logs by tensorboard
# tensorboard --logdir .\archive\logs\
