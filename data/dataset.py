import dataclasses
import json
import logging
from dataclasses import dataclass
from typing import List, Optional, Union

import pandas as pd
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase
from transformers.data.processors.utils import InputFeatures

from file_processor import datasets_description_mapping, load_datasets_from_json
from .processors import processors_mapping

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class CLSDataCollatorWithPadding:
    """
    Implements padding for LM-BFF inputs, return inputs as a dictionary of tensors in batch.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features):
        standard_features = []

        for item in features:
            standard_item = {}
            for field in ["input_ids", "attention_mask", "token_type_ids", "label"]:
                if getattr(item, field) is not None:
                    standard_item[field] = getattr(item, field)
            standard_features.append(standard_item)

        batch = self.tokenizer.pad(
            standard_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

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

    result = {'input_ids': input_ids,
              'attention_mask': attention_mask,
              'label': int(input_data['labels'][0])
              }

    if 'BERT' in type(tokenizer).__name__:
        # Only provide token type ids for BERT
        result['token_type_ids'] = token_type_ids

    return result


class CLSTaskDataset(Dataset):
    """
    This is a dataset for CLS task
    """
    def __init__(self, args, tokenizer, mode: str = "train"):
        self.args = args

        self.data_desc = datasets_description_mapping[args.task_name]
        self.processor = processors_mapping[args.task_name]

        self.tokenizer = tokenizer

        # Get data in list
        if mode == "train":
            self.data_list, _, _ = load_datasets_from_json(self.data_desc)
        elif mode == "dev":
            _, self.data_list, _ = load_datasets_from_json(self.data_desc)
        elif mode == "test":
            _, _, self.data_list = load_datasets_from_json(self.data_desc)

        # Get label list and do mapping
        self.label_list = self.data_desc.label_list
        self.label_to_id = {label: i for i, label in enumerate(self.label_list)}
        self.id_to_label = {i: label for i, label in enumerate(self.label_list)}

    def __getitem__(self, i):
        """
        Returns a list of processed "InputFeatures".
        """
        # Prepare features
        inputs = tokenize_multipart_input(
            input_data=self.data_list[i],
            max_length=self.args.max_seq_length,
            tokenizer=self.tokenizer,
            truncate_head=False,
        )
        features = CLSInputFeatures(**inputs)
        return features

    def __len__(self):
        return len(self.data_list)


if __name__ == "__main__":
    from transformers import AutoTokenizer
    from configs.args_list import CLSModelArguments

    test_args = CLSModelArguments("./")
    test_args.task_name = "llm"
    test_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    dataset = CLSTaskDataset(args=test_args, tokenizer=test_tokenizer, mode="train")
    print(dataset[0])

    data_collator = CLSDataCollatorWithPadding(tokenizer=test_tokenizer)
    data_list = []
    for i in range(8):
        data_list.append(dataset[i])
    print(data_collator.__call__(data_list))
