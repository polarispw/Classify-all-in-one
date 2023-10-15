import dataclasses
import json
import logging
from dataclasses import dataclass
from typing import List, Optional, Union

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase
from transformers.data.processors.utils import InputFeatures

from format import datasets_mapping, load_datasets_from_json
from processors import processors_mapping

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class MyDataCollatorWithPadding:
    """
    Implements padding for LM-BFF inputs, return inputs as a dictionary of tensors in batch.
    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features):
        mask_pos = []
        standard_features = []

        for item in features:
            standard_item = {}
            for field in ["input_ids", "label", "attention_mask", "token_type_ids"]:
                if getattr(item, field) is not None:
                    standard_item[field] = getattr(item, field)
            standard_features.append(standard_item)
            mask_pos.append(item.mask_pos)

        batch = self.tokenizer.pad(
            standard_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        if any(mask_pos):
            batch["mask_pos"] = torch.tensor(mask_pos)

        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]

        return batch


@dataclass(frozen=True)
class CLSInputFeatures(InputFeatures):
    """
    Inherit from Transformers' InputFeatuers.
    """
    input_ids: List[int]
    attention_mask: Optional[List[int]] = None
    token_type_ids: Optional[List[int]] = None
    label: Optional[Union[int, float]] = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self)) + "\n"


def tokenize_multipart_input(
        input_data,
        max_length,
        tokenizer,
        truncate_head=False,
):
    def enc(text):
        return tokenizer.encode(text, add_special_tokens=False)

    input_text_list = input_data['texts']
    input_ids = []
    attention_mask = []
    token_type_ids = []  # Only for BERT
    mask_pos = None  # Position of the mask token

    if tokenizer.cls_token_id is not None:
        input_ids = [tokenizer.cls_token_id]
        attention_mask = [1]
        token_type_ids = [0]
    else:
        input_ids = []
        attention_mask = []
        token_type_ids = []

    for sent_id, input_text in enumerate(input_text_list):
        if input_text is None:
            logging.warning(f"Missing input of sentence {sent_id}")
            continue
        if pd.isna(input_text) or input_text is None:
            # Empty input
            input_text = ''
        input_tokens = enc(input_text) + [tokenizer.sep_token_id]
        input_ids += input_tokens
        attention_mask += [1 for i in range(len(input_tokens))]
        token_type_ids += [sent_id for i in range(len(input_tokens))]

    if 'T5' in type(tokenizer).__name__:  # T5 does not have CLS token
        input_ids = input_ids[1:]
        attention_mask = attention_mask[1:]
        token_type_ids = token_type_ids[1:]

    # Truncate
    if len(input_ids) > max_length:
        if truncate_head:
            input_ids = input_ids[-max_length:]
            attention_mask = attention_mask[-max_length:]
            token_type_ids = token_type_ids[-max_length:]
        else:
            # Default is to truncate the tail
            input_ids = input_ids[:max_length]
            attention_mask = attention_mask[:max_length]
            token_type_ids = token_type_ids[:max_length]

    result = {'input_ids': input_ids, 'attention_mask': attention_mask, 'label': input_data['labels']}
    if 'BERT' in type(tokenizer).__name__:
        # Only provide token type ids for BERT
        result['token_type_ids'] = token_type_ids

    return result


class CLSTaskDataset(Dataset):
    """
    This is a dataset for CLS task
    """
    def __init__(self, args, tokenizer, mode:str = "train"):
        self.args = args
        self.mode = mode

        self.task_name = args.task_name
        self.data_desc = datasets_mapping[args.task_name]

        self.tokenizer = tokenizer
        self.processor = processors_mapping[args.task_name]

        # Get label list and (for prompt) label word list
        if mode == "train":
            self.data_list, _, _ = load_datasets_from_json(self.data_desc)
        elif mode == "dev":
            _, self.data_list, _ = load_datasets_from_json(self.data_desc)
        elif mode == "test":
            _, _, self.data_list = load_datasets_from_json(self.data_desc)
        self.label_list = self.data_desc.label_list
        self.num_labels = len(self.label_list)

    def __getitem__(self, i):
        """
        Returns a list of processed "InputFeatures".
        """
        # Prepare features
        inputs = tokenize_multipart_input(
            input_data=self.data_list[i],
            max_length=self.args.max_length,
            tokenizer=self.tokenizer,
            truncate_head=False,
        )
        features = CLSInputFeatures(**inputs)
        return features

    def __len__(self):
        return len(self.data_list)

    def get_labels(self):
        return self.label_list


if __name__ == "__main__":
    from transformers import AutoTokenizer
    from configs.args_list import ModelArguments

    test_args = ModelArguments("./")
    test_args.task_name = "llm"
    test_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    dataset = CLSTaskDataset(args=test_args, tokenizer=test_tokenizer, mode="train")
    print(dataset[0])
    print(dataset[1])
    print(dataset[2])
    print(dataset[3])
    print(dataset[4])
