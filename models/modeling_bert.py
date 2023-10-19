import torch
import torch.nn as nn
from typing import Optional, List

from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import PreTrainedModel, AutoModel, AutoConfig, AutoTokenizer
from transformers.modeling_outputs import SequenceClassifierOutput

from configs.args_list import CLSModelArguments


class CLSBertLikeModel(PreTrainedModel):
    """
    This class is used to build a model that has a BERT-like architecture for sequence classification.
    """
    def __init__(self, args: CLSModelArguments):
        super(CLSBertLikeModel, self).__init__(AutoConfig.from_pretrained(args.name_or_path))

        # you can change the attributes init in ModelConfig here before loading the model
        self.name_or_path = args.name_or_path
        self.cache_dir = args.cache_dir
        self.max_position_embeddings = args.max_seq_length
        self.num_labels = args.num_labels
        self.problem_type = args.problem_type

        self.base = AutoModel.from_pretrained(self.name_or_path, cache_dir=self.cache_dir)

        self.classifier_width = 2 * self.config.hidden_size
        self.classifier = nn.Sequential(
            torch.nn.Linear(self.config.hidden_size, self.classifier_width),
            torch.nn.Dropout(0.2),
            torch.nn.ReLU(),
            torch.nn.Linear(self.classifier_width, self.config.hidden_size),
            torch.nn.Dropout(0.1),
            torch.nn.ReLU(),
            torch.nn.Linear(self.config.hidden_size, self.num_labels)
        )

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
                return_dict: Optional[bool] = None,):

        outputs = self.base(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds,
                            output_attentions=output_attentions,
                            output_hidden_states=output_hidden_states,
                            return_dict=return_dict,)

        # here we sum the last hidden state of all tokens together for cls input
        cls_input = outputs.last_hidden_state.sum(dim=1)
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


if __name__ == "__main__":
    my_model = CLSBertLikeModel(CLSModelArguments('bert-base-uncased', cache_dir='../model_cache'))

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    test_inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    print(test_inputs)

    cls_res = my_model(**test_inputs)
    print(cls_res)
