"""
this is base dataset for all CLS tasks
"""
import os
from typing import List

import datasets
from datasets import load_dataset, DatasetDict, Dataset

from configs.args_list import CLSDatasetArguments


class CLSDataset:
    """
    This is a dataset class for CLS task
    """

    def __init__(self, args: CLSDatasetArguments, tokenizer):
        # check the file
        self.data_path = args.data_path
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data path {self.data_path} does not exist.")
        elif ".csv" in self.data_path:
            self.file_type = "csv"
        elif ".json" in self.data_path:
            self.file_type = "json"
        elif ".txt" in self.data_path:
            self.file_type = "txt"
        else:
            raise NotImplementedError(f"File type {self.file_type} is not supported.")

        self.tokenizer = tokenizer
        self.random_seed = args.seed

        self.raw_datasets = None
        self.tokenized_datasets = None
        self.balanced_subset = None

    def load_dataset(self) -> DatasetDict:
        if self.file_type == "csv":
            self.raw_datasets = load_dataset("csv", data_files=self.data_path)
        elif self.file_type == "json":
            self.raw_datasets = load_dataset("json", data_files=self.data_path)
        elif self.file_type == "txt":
            self.raw_datasets = load_dataset("text", data_files=self.data_path)

        return self.raw_datasets

    def sample_balanced_subsets(self, num_for_each_class: int = None) -> DatasetDict:
        """
        This function is used to sample balanced subsets from the **train** dataset.
        :return:
        """
        # get the num of data in each class
        num_data = {}
        if self.raw_datasets is None:
            self.load_dataset()

        for label in self.raw_datasets["train"]["label"]:
            if label not in num_data:
                num_data[label] = 1
            else:
                num_data[label] += 1

        # randomly sample the min_num_data from each class
        min_num_data = min(num_data.values()) if num_for_each_class is None else min(num_for_each_class,
                                                                                     min(num_data.values()))
        sampled_data = []
        for label in num_data:
            sampled_data.append(self.raw_datasets["train"].filter(lambda example: example["label"] == label).shuffle(
                seed=self.random_seed).select(range(min_num_data)))

        # concatenate the sampled data
        self.balanced_subset = DatasetDict({"train": datasets.concatenate_datasets(sampled_data)})
        return self.balanced_subset

    def tokenize_dataset(self, col_names: List) -> DatasetDict:
        """
        This function is used to tokenize the dataset.
        Func of tokenizer is also implemented in data_collator, which is recommended.
        If you tokenize the dataset here, just use DefaultDataCollator in trainer to avoid tokenization again.
        :param col_names: title of columns to be tokenized
        :return:  dataset
        """
        if self.raw_datasets is None:
            self.load_dataset()

        self.tokenized_datasets = self.raw_datasets
        for col_name in col_names:
            self.tokenized_datasets = self.tokenized_datasets.map(lambda example: self.tokenizer(example[col_name]),
                                                                  batched=True)
        return self.tokenized_datasets


if __name__ == "__main__":
    from transformers import AutoTokenizer

    my_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    my_dataset = CLSDataset(args=CLSDatasetArguments(data_path="../data_lib/fake_reviews/fake reviews dataset.csv"),
                            tokenizer=my_tokenizer)

    balanced_dataset = my_dataset.sample_balanced_subsets(32)["train"]
    tokenized_dataset = my_dataset.tokenize_dataset(["text_"])["train"]
    print(my_dataset.raw_datasets)
    print(balanced_dataset)
    print(tokenized_dataset[0])
