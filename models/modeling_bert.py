from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import PreTrainedModel, AutoModel, AutoConfig, AutoTokenizer
from transformers.modeling_outputs import SequenceClassifierOutput

from config.args_list import CLSModelArguments


class CLSBert(PreTrainedModel):
    """
    The Layer-wise Bert class is a BERT-like model with MLP classifier for sequence classification.
    Supports layer wise lr and freezing certain layers.
    """

    def __init__(self, args: CLSModelArguments):
        super(CLSBert, self).__init__(AutoConfig.from_pretrained(args.model_name_or_path))

        # you can change the attributes init in ModelConfig here before loading the model
        self.name_or_path = args.model_name_or_path
        self.cache_dir = args.cache_dir
        self.max_position_embeddings = args.max_seq_length

        self.num_labels = args.num_labels
        self.problem_type = args.problem_type

        self.base = AutoModel.from_pretrained(self.name_or_path, cache_dir=self.cache_dir)

        self.layer1_width = self.config.hidden_size // 2
        self.layer2_width = self.config.hidden_size // 4
        self.classifier = nn.Sequential(
            torch.nn.Dropout(0.1),
            torch.nn.Linear(self.config.hidden_size, self.layer1_width),
            torch.nn.ReLU(),
            torch.nn.Linear(self.layer1_width, self.layer2_width),
            torch.nn.ReLU(),
            torch.nn.Linear(self.layer2_width, self.num_labels)
        )

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):

        outputs = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        # here we sum the last hidden state of all tokens together for cls input
        cls_input = outputs.last_hidden_state.sum(dim=1)
        # cls_input = outputs[1]
        logits = self.classifier(cls_input)

        # here we compute the loss
        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)

            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class SIMBert(PreTrainedModel):
    """
    This class is used to build a model that has a BERT-like architecture to do supervised SimCSE pre-training.
    We use base + MLP to get the sentence embedding, inferring from the SimCSE paper.
    """

    def __init__(self, args: CLSModelArguments):
        super(SIMBert, self).__init__(AutoConfig.from_pretrained(args.model_name_or_path))

        # you can change the attributes init in ModelConfig here before loading the model
        self.name_or_path = args.model_name_or_path
        self.cache_dir = args.cache_dir
        self.max_position_embeddings = args.max_seq_length
        self.num_labels = args.num_labels
        self.problem_type = args.problem_type

        self.base = AutoModel.from_pretrained(self.name_or_path, cache_dir=self.cache_dir)

        self.loss = nn.CrossEntropyLoss()
        self.scale = 20.0

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                token_type_ids: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                head_mask: Optional[torch.Tensor] = None,
                inputs_embeds: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None, ):
        outputs = self.base(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds,
                            output_attentions=output_attentions,
                            output_hidden_states=output_hidden_states,
                            return_dict=return_dict)

        # here we sum the last hidden state of all tokens together as pooled output
        pooled_output = outputs.last_hidden_state.sum(dim=1)

        # labels = torch.tensor([0, 1, 2, 3]).to(pooled_output.device)
        logits, loss = self.simcse_sup_loss(pooled_output, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def simcse_sup_loss(self, output_hidden_states: torch.Tensor, labels: torch.Tensor):
        """
        This function is used to compute the supervised loss for SimCSE.
        We consider the output_hidden_states in a batch with the same label as pos pairs, and others as neg pairs.
        :param output_hidden_states:
        :param labels:
        :return:
        """
        # get pos pairs from input labels
        label2idx = {}
        for idx, label in enumerate(labels):
            if label not in label2idx:
                label2idx[label] = [idx]
            else:
                label2idx[label].append(idx)

        # calculate cosine similarity
        sim = F.cosine_similarity(output_hidden_states.unsqueeze(1), output_hidden_states.unsqueeze(0), dim=-1)
        scaled_sim = sim / self.scale

        pos_probs = torch.where(labels.unsqueeze(1) == labels.unsqueeze(0), scaled_sim, torch.zeros_like(scaled_sim))
        neg_probs = torch.where(labels.unsqueeze(1) != labels.unsqueeze(0), scaled_sim, torch.zeros_like(scaled_sim))
        pos_probs = torch.sub(pos_probs, torch.eye(labels.shape[0], device=scaled_sim.device) / self.scale)

        neg_probs = torch.add(neg_probs, torch.eye(labels.shape[0], device=scaled_sim.device) / self.scale)
        pos_probs = torch.sum(pos_probs, dim=0)
        neg_probs = torch.sum(neg_probs, dim=0)
        pred_probs = torch.stack([pos_probs, neg_probs], dim=0).transpose(0, 1)
        loss = self.loss(pred_probs, torch.zeros_like(labels))
        return pos_probs - neg_probs, loss


if __name__ == "__main__":
    print(torch.cuda.is_available())
    my_model = SIMBert(CLSModelArguments('bert-base-uncased', cache_dir='../model_cache')).to('cuda:0')
    my_model.train()

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    test_inputs = tokenizer(["Hello, my dog is cute",
                             "Hello, my dog is cute",
                             "Dimension where cosine similarity is computed",
                             "Dimension where cosine similarity is computed"],
                            padding=True,
                            truncation=True,
                            return_tensors="pt").to('cuda')

    loss, sim = my_model(**test_inputs)
    print(loss)
    loss.backward()
    print("finish")
