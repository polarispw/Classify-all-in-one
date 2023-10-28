# NLP project template

## PLMs for classification tasks

### Features:

- support various models: encoders, decoders, seq2seq
- support functions like full parameter training, fine-tuning, p-tuning
- customize the structure of model, invisible to higher levels like data processor or trainer
- support various metrics
- customize lr scheduler, optimizer,
- convenient dataset for text in json files, support multi-labels and multi-text content
- easily run with config.json and CLI
- auto archive of logs, results and checkpoints, support tensorboard
- support multi-GPU and low memory for LLMs PEFT
- utils for data pre-process, visualization and a chatgpt api

```python
Project/
|---archive/	# dir of logs, checkpoints and results
|---configs/
	|---config.json	# default hyper-parameters
    |---args_list.py	# arg_parser class
|---data/
	|---preprocess.py	# any2json, profile the datasets: seq_len, labels 
    |---augmentation.py	# data augmentaion
	|---dataset.py
|---data_lib/	# dir of datasets
|---models/
	|---base_model.py	# abstract for calling 
    |---modeling_[model_name].py	# implement and customization of models
    |---...
|---solver/
    |---evaluator.py	# calculate loss
    |---lr_scheduler.py	# schedule learning rate
    |---optimizer.py	# library of optimizers
|---train/
	|---trainer.py	# basic trainer
	|---parallel_trainer.py	# train on multi-GPU
	|---...
|---test/
	|---test_[dataset_name].py	# various by datasets
|---utils/
	|---chatgpt_api.py	# access to chatgpt
    |---save.py	# create dir for every exp; save logs in CLI, checkpoints and tensorboard files
    |---...
|---scripts/
	|---train.sh	# receive hyper-parameters and call run.py for training
	|---test.sh	# receive hyper-parameters and call run.py for test
run.py
requirements.txt
```
