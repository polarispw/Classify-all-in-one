"""
Here includes data_collators for different tasks, which can be directly passed to the trainer
Support: data_collator for CLS finetune
         data_collator for SimCSE pretrain
         data_collator for p-tuning
"""
from dataclasses import dataclass
from typing import Union, Optional, List, Any, Dict


from transformers import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy


@dataclass
class DataCollatorForCLS:
    """
       Data collator that will dynamically pad the inputs received.

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
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = self.tokenizer.pad(
            features,
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

