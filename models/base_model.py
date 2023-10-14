"""Custom models for few-shot learning specific operations."""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, BertOnlyMLMHead
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel

from modeling_roberta import RobertaPreTrainedModel, RobertaModel, RobertaLMHead

logger = logging.getLogger(__name__)


def resize_token_type_embeddings(model, new_num_types: int, random_segment: bool):
    """
    Resize the segment (token type) embeddings for BERT
    """
    if hasattr(model, 'bert'):
        old_token_type_embeddings = model.bert.embeddings.token_type_embeddings
    else:
        raise NotImplementedError
    new_token_type_embeddings = nn.Embedding(new_num_types, old_token_type_embeddings.weight.size(1))
    if not random_segment:
        new_token_type_embeddings.weight.data[
        :old_token_type_embeddings.weight.size(0)] = old_token_type_embeddings.weight.data

    model.config.type_vocab_size = new_num_types
    if hasattr(model, 'bert'):
        model.bert.embeddings.token_type_embeddings = new_token_type_embeddings
    else:
        raise NotImplementedError


def base_finetuning_forward(
        model,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        mask_pos=None,
        labels=None,
        sfc_input_ids=None,
        sfc_attention_mask=None,
        sfc_mask_pos=None
):

    if mask_pos is not None:
        mask_pos = mask_pos.squeeze()

    model_fn = model.get_model_fn()
    # Encode everything
    if token_type_ids is not None:
        outputs = model_fn(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
    else:
        outputs = model_fn(
            input_ids,
            attention_mask=attention_mask,
        )

    # Get <mask> token representation
    sequence_output = outputs[0]
    if mask_pos is not None:
        sequence_mask_output = sequence_output[torch.arange(sequence_output.size(0)), mask_pos]
    else:
        sequence_mask_output = sequence_output[:, 0]  # <cls> representation
        # sequence_mask_output = sequence_output.mean(dim=1) # average representation

    logits = model.classifier(sequence_mask_output)

    loss = None
    if labels is not None:
        if model.model_args.l2_loss:
            coords = torch.nn.functional.one_hot(labels.squeeze(), model.config.num_labels).float()
            loss = nn.MSELoss()(logits.view(-1, logits.size(-1)), coords)
        else:
            loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), labels.view(-1))

    output = (logits,)

    return ((loss,) + output) if loss is not None else output


class BertModelForPromptFinetuning(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        print(config)
        self.model_type = config.model_type
        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(config)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        self.init_weights()

        # These attributes should be assigned once the model is initialized
        self.model_args, self.data_args, self.label_word_list = None, None, None

        # For regression
        self.lb, self.ub = 0.0, 1.0

        # For auto label search.
        self.return_full_softmax = None

    def get_model_fn(self):
        return self.bert

    def get_lm_head_fn(self):
        return self.cls

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, mask_pos=None, labels=None):
        return base_finetuning_forward(self, input_ids, attention_mask, token_type_ids, mask_pos, labels)


class RobertaModelForPromptFinetuning(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        logger.warning("By default for RoBERTa models the input embeddings and the output embeddings are NOT tied!!!!")
        self.num_labels = config.num_labels
        print(config)
        self.model_type = config.model_type
        self.roberta = RobertaModel(config)
        self.lm_head = RobertaLMHead(config)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        self.init_weights()

        # These attributes should be assigned once the model is initialized
        self.model_args, self.data_args, self.label_word_list = None, None, None

    def tie_emb(self):
        output_embeddings = self.lm_head.decoder
        self._tie_or_clone_weights(output_embeddings, self.roberta.get_input_embeddings())

    def get_model_fn(self):
        return self.roberta

    def get_lm_head_fn(self):
        return self.lm_head

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, mask_pos=None, labels=None):
        return base_finetuning_forward(self, input_ids, attention_mask, token_type_ids, mask_pos, labels)


class GPT2ModelForPromptFinetuning(GPT2LMHeadModel):

    def __init__(self, config):
        super().__init__(config)
        raise NotImplementedError("Need to check if the lm head is properly loaded and whether it is tied.")
        self.num_labels = config.num_labels
        print(config)
        self.model_type = config.model_type
        # self.transformer = GPT2Model(config)
        # self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        self.init_weights()

        # These attributes should be assigned once the model is initialized
        self.model_args, self.data_args, self.label_word_list = None, None, None

    def get_model_fn(self):
        return self.transformer

    def get_lm_head_fn(self):
        return self.lm_head

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, mask_pos=None, labels=None):
        return base_finetuning_forward(self, input_ids, attention_mask, token_type_ids, mask_pos, labels)


MODEL_TYPES = {
    "bert": BertModelForPromptFinetuning,
    "roberta": RobertaModelForPromptFinetuning,
    "gpt2": GPT2ModelForPromptFinetuning,
}
