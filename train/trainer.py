"""
The Trainer class, to easily train a 🤗 Transformers from scratch or finetune it on a new task.
"""

import collections.abc
import os
import time
from typing import Optional, List, Union, Dict, Callable, Tuple

import torch
import torch.distributed
import transformers
from packaging import version
from torch.optim import SGD
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    DataCollator,
    EvalPrediction,
    TrainerCallback,
    TrainingArguments)
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

DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

# Check if Pytorch version >= 1.6 to switch between Native AMP and Apex
if version.parse(torch.__version__) < version.parse("1.6"):
    raise EnvironmentError("Version of pytorch is lower than 1.6")
else:
    _use_native_amp = True

if version.parse(torch.__version__) < version.parse("1.2"):
    _use_ddp_no_sync = False
else:
    _use_ddp_no_sync = True

if is_datasets_available():
    pass

if is_tensorboard_available():
    from transformers.integrations import TensorBoardCallback

    DEFAULT_CALLBACKS.append(TensorBoardCallback)

if is_wandb_available():
    from transformers.integrations import WandbCallback

    DEFAULT_CALLBACKS.append(WandbCallback)

if is_comet_available():
    from transformers.integrations import CometCallback

    DEFAULT_CALLBACKS.append(CometCallback)

if is_optuna_available():
    pass

if is_ray_available():
    pass

logger = logging.get_logger(__name__)
logger.setLevel(logging.INFO)

""" 
The above part is copied from MeZO
"""


def default_dev_objective(metrics):
    """
    Objective used for picking the best model on development sets
    """
    if "eval_mnli/acc" in metrics:
        return metrics["eval_mnli/acc"]
    elif "eval_mnli-mm/acc" in metrics:
        return metrics["eval_mnli-mm/acc"]
    elif "eval_f1" in metrics:
        return metrics["eval_f1"]
    elif "eval_mcc" in metrics:
        return metrics["eval_mcc"]
    elif "eval_pearson" in metrics:
        return metrics["eval_pearson"]
    elif "eval_acc" in metrics:
        return metrics["eval_acc"]

    raise Exception("No metric founded for {}".format(metrics))


class Trainer(transformers.Trainer):
    """
    Adding some functions based on Transformers' Trainer class.
    """

    def __init__(
            self,
            model: Union[PreTrainedModel, torch.nn.Module] = None,
            args: TrainingArguments = None,
            data_collator: Optional[DataCollator] = None,
            train_dataset: Optional[Dataset] = None,
            eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
            tokenizer: Optional[PreTrainedTokenizerBase] = None,
            model_init: Optional[Callable[[], PreTrainedModel]] = None,
            compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
            callbacks: Optional[List[TrainerCallback]] = None,
            optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
            preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ):
        super().__init__(model, args, data_collator, train_dataset, eval_dataset, tokenizer, model_init,
                         compute_metrics, callbacks, optimizers, preprocess_logits_for_metrics)

        self.warmup_ratio = 0.1
        self._past = None
        self.dev_objective = None
        self.best_dir = None

    def train(self, model_path=None, dev_objective=None, **kwargs):
        """
        Main training entry point.

        The training logic is directly borrowed from transformers.Trainer (version 3.0.2).
        Add early stopping.
        :param dev_objective:
        :param model_path:
        """
        self.best_dir = None
        self.objective = -float("inf")
        self.dev_objective = dev_objective if dev_objective is not None else default_dev_objective

        """ Prepare dataloader and hyperparameters for update steps"""
        # return a DataLoader using self.train_dataset and self.data_collator
        train_dataloader = self.get_train_dataloader()

        # calculate update steps to do in one epoch
        num_update_steps_per_epoch = len(train_dataloader) // self.args.gradient_accumulation_steps
        if num_update_steps_per_epoch == 0:
            num_update_steps_per_epoch = 1

        # calculate the total number of epochs to train, based on args.max_steps or args.num_train_epochs
        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            num_train_epochs = (self.args.max_steps // num_update_steps_per_epoch +
                                int(self.args.max_steps % num_update_steps_per_epoch > 0))
        else:
            t_total = int(len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs)
            num_train_epochs = self.args.num_train_epochs

        """ Prepare optimizer and schedule (linear warmup and decay) """
        self.create_optimizer_and_scheduler(num_training_steps=t_total)
        optimizer = self.optimizer
        scheduler = self.lr_scheduler

        # Check if saved optimizer or scheduler states exist
        if (
                model_path is not None
                and os.path.isfile(os.path.join(model_path, "optimizer.pt"))
                and os.path.isfile(os.path.join(model_path, "scheduler.pt"))
        ):
            # Load in optimizer and scheduler states
            optimizer.load_state_dict(
                torch.load(os.path.join(model_path, "optimizer.pt"), map_location=self.args.device)
            )
            scheduler.load_state_dict(torch.load(os.path.join(model_path, "scheduler.pt")))

        """ Get model prepared for training """
        model = self.model

        # Multi-gpu training (should be after apex fp16 initialization)
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Distributed training (should be after apex fp16 initialization)
        if self.args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.args.local_rank],
                output_device=self.args.local_rank,
                find_unused_parameters=True,
            )

        """ Start training """
        total_train_batch_size = (
                self.args.train_batch_size
                * self.args.gradient_accumulation_steps
                * (torch.distributed.get_world_size() if self.args.local_rank != -1 else 1)
        )
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", self.num_examples(train_dataloader))
        logger.info("  Num Epochs = %d", num_train_epochs)
        logger.info("  Instantaneous batch size per device = %d", self.args.per_device_train_batch_size)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", total_train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        self.state = TrainerState()
        self.state.global_step = 0
        start_time = time.time()
        self.epoch = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0

        if self.args.gradient_checkpointing:
            model.gradient_checkpointing_enable()

        # Check if continuing training from a checkpoint
        if model_path is not None:
            # set global_step to global_step of last saved checkpoint from model path
            try:
                self.state.global_step = int(model_path.split("-")[-1].split("/")[0])
                epochs_trained = self.state.global_step // (
                        len(train_dataloader) // self.args.gradient_accumulation_steps)
                steps_trained_in_current_epoch = self.state.global_step % (
                        len(train_dataloader) // self.args.gradient_accumulation_steps
                )

                logger.info("  Continuing training from checkpoint, will skip to saved global_step")
                logger.info("  Continuing training from epoch %d", epochs_trained)
                logger.info("  Continuing training from global step %d", self.state.global_step)
                logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
            except ValueError:
                self.state.global_step = 0
                logger.info("  Starting fine-tuning.")

        tr_loss = torch.tensor(0.0).to(self.args.device)
        logging_loss_scalar = 0.0
        
        metrics = None
        for epoch in range(epochs_trained, int(num_train_epochs)):
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)

            epoch_iterator = tqdm(train_dataloader, desc="Iteration for training", disable=True)

            # Reset the past mems state at the beginning of each epoch if necessary.
            if self.args.past_index >= 0:
                self._past = None

            for step, inputs in enumerate(epoch_iterator):

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                # standard, non-ZO optimization
                tr_loss += self.training_step(model, inputs)

                if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                        # last step in epoch but step is always smaller than gradient_accumulation_steps
                        self.args.gradient_accumulation_steps >= len(epoch_iterator) == (step + 1)
                ):
                    optimizer.step()
                    scheduler.step()

                    model.zero_grad()
                    self.state.global_step += 1
                    self.epoch = epoch + (step + 1) / len(epoch_iterator)

                    if (self.args.logging_steps > 0 and self.state.global_step % self.args.logging_steps == 0) or (
                            self.state.global_step == 1 and self.args.logging_first_step
                    ):
                        logs = {}
                        tr_loss_scalar = tr_loss.item()
                        logs["loss"] = (tr_loss_scalar - logging_loss_scalar) / self.args.logging_steps
                        # backward compatibility for pytorch schedulers
                        logs["learning_rate"] = (
                            scheduler.get_last_lr()[0]
                            if version.parse(torch.__version__) >= version.parse("1.4")
                            else scheduler.get_lr()[0]
                        )
                        logging_loss_scalar = tr_loss_scalar

                        self.log(logs)
                        logger.info(str(logs))

                if 0 < self.args.max_steps < self.state.global_step:
                    epoch_iterator.close()
                    break

                if self.args.evaluate_during_training and self.state.global_step % self.args.eval_steps == 0:
                    output = self.evaluate()
                    metrics = output.metrics
                    objective = self.dev_objective(metrics)
                    if objective > self.objective:
                        logger.info("Best dev result: {}".format(objective))
                        self.objective = objective
                        # self.save_model(self.args.output_dir)
                        # Now we save this to (CPU) memory instead of disk <-- much faster
                        self.best_model_ckpt = {k: v.detach().cpu() for k, v in model.state_dict().items()}

            if 0 < self.args.max_steps < self.state.global_step:
                # train_iterator.close()
                break

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed!)\n\n")
        return TrainOutput(self.state.global_step, tr_loss / self.state.global_step, metrics), self.objective

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Based on Transformers' default one, we add fixing layer option
        where the bottom n layers' parameters will not be fine-tuned.
        """
        # no need when doing inference
        if self.args.hf_inference_model:
            return

        # No existing optimizer, create the optimizer
        if self.optimizer is None:

            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = []

            # layer wise lr: calculate the lr_factor for each layer
            num_layer = self.model.config.num_hidden_layers
            base_factor = 1.0
            lr_factor_list = []
            for i in range(num_layer):
                lr_factor_list.append(base_factor)
                base_factor *= self.args.lr_layer_decay_rate
            lr_factor_list = lr_factor_list[::-1]

            # group the parameters
            for n, p in self.model.named_parameters():
                layer_num = int(n[n.find('encoder.layer') + 14:].split('.')[0])
                weight_norm = not any(nd in n for nd in no_decay) and not ('embeddings' in n)
                if p.requires_grad:
                    optimizer_grouped_parameters.append(
                        {
                            "params": [p],
                            "weight_decay": self.args.weight_decay if weight_norm else 0.0,
                            "lr": self.args.learning_rate * lr_factor_list[layer_num],
                        }
                    )

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

        # No existing scheduler, create the scheduler
        if self.lr_scheduler is None:
            self.lr_scheduler = get_scheduler(
                self.args.lr_scheduler_type,
                optimizer=self.optimizer,
                num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
            )

    def evaluate(self,
                 eval_dataset: Optional[Dataset] = None,
                 ignore_keys: Optional[List[str]] = None,
                 metric_key_prefix: str = "eval", ) -> EvalLoopOutput:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are
        task-dependent (pass it to the init :obj:`compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (`Dataset`, *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns
                not accepted by the `model.forward()` method are automatically removed. It must implement the `__len__`
                method.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)

        Returns:
            Return output instead of output.metrics (so there is also logits)
        """
        if eval_dataset is not None and not isinstance(eval_dataset, collections.abc.Sized):
            raise ValueError("eval_dataset must implement __len__")

        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        output = self.prediction_loop(eval_dataloader, description="Evaluation")

        self.log(output.metrics)
        logger.info(output.metrics)

        return output
