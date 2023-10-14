import os
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import Trainer, TrainingArguments

class MyTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tb_writer = SummaryWriter()

    def evaluate(self, eval_dataset=None):
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        output = self.prediction_loop(eval_dataloader, description="Evaluation")
        self.log(output.metrics)
        self.tb_writer.add_scalar("eval_loss", output.metrics["eval_loss"], self.state.global_step)
        return output

    def train(self, model_path=None):
        train_dataloader = self.get_train_dataloader()
        eval_dataloader = self.get_eval_dataloader(self.eval_dataset)

        if model_path is not None:
            self.model = self.model.from_pretrained(model_path)

        self.global_step = 0
        self.epoch = 0
        self.total_flos = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0

        tr_loss = torch.tensor(0.0).to(self.args.device)
        logging_loss = torch.tensor(0.0).to(self.args.device)
        self.model.zero_grad()

        train_iterator = trange(
            epochs_trained, int(self.args.num_train_epochs), desc="Epoch", disable=not self.is_local_master()
        )
        for epoch in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=not self.is_local_master())
            for step, inputs in enumerate(epoch_iterator):
                self.global_step += 1
                self.total_flos += self.floating_point_ops(inputs)

                loss = self.training_step(model, inputs)
                tr_loss += loss
                if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    len(epoch_iterator) <= self.args.gradient_accumulation_steps
                    and (step + 1) == len(epoch_iterator)
                ):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / len(epoch_iterator)

                    if (self.args.logging_steps > 0 and self.state.global_step % self.args.logging_steps == 0) or (
                        self.global_step == 1 and self.args.logging_first_step
                    ):
                        logs: Dict[str, float] = {}
                        tr_loss_scalar = tr_loss.item()
                        logs["loss"] = (tr_loss_scalar - logging_loss) / self.args.logging_steps
                        # backward compatibility for pytorch schedulers
                        logs["learning_rate"] = (
                            self.lr_scheduler.get_last_lr()[0]
                            if version.parse(torch.__version__) >= version.parse("1.4")
                            else self.lr_scheduler.get_lr()[0]
                        )
                        logging_loss = tr_loss_scalar

                        self.log(logs)

                        if self.args.evaluate_during_training:
                            self.evaluate()

                        if self.args.save_steps > 0 and self.state.global_step % self.args.save_steps == 0:
                            output_dir = os.path.join(self.args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}")
                            self.save_model(output_dir)
                            self.tb_writer.add_scalar("loss", (tr_loss - logging_loss) / self.args.logging_steps, self.state.global_step)
                            logging_loss = tr_loss

                            if 0 < self.args.max_steps < self.state.global_step:
                                train_iterator.close()
                                break

                        if self.tb_writer:
                            self.tb_writer.close()

    def training_step(self, model, inputs):
        model.train()
        inputs = self._prepare_inputs(inputs)

        loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        loss.backward()

        return loss.detach()

    def compute_loss(self, model, inputs):
        outputs = model(**inputs)
        loss = outputs.loss
        return loss

    def log(self, logs):
        if self.state.is_local_process_zero:
            self.tb_writer.add_scalar("lr", self._get_lr()[0], self.state.global_step)
            for k, v in logs.items():
                self.tb_writer.add_scalar(k, v, self.state.global_step)

    def _get_lr(self):
        return [group["lr"] for group in self.optimizer.param_groups]

    def save_model(self, output_dir):
        if self.is_world_process_zero():
            os.makedirs(output_dir, exist_ok=True)
            self.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)

            torch.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
            torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

    def _prepare_inputs(self, inputs):
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.args.device)

        return inputs
