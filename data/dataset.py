"""
Here includes DataManager Class for different tasks
"""
import os
from typing import List, Dict

import datasets
from datasets import load_dataset, DatasetDict, Dataset

from config.args_list import CLSDatasetArguments


class CLSDataManager:
    """
    This is a dataset class for base CLS task
    """

    def __init__(self, args, tokenizer):
        # check the file
        self.args = args
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
        self.random_seed = args.rand_seed

    def load_dataset(self):
        if self.file_type == "csv":
            raw_datasets = load_dataset("csv", data_files=self.data_path)
        elif self.file_type == "json":
            raw_datasets = load_dataset("json", data_files=self.data_path)
        elif self.file_type == "txt":
            raw_datasets = load_dataset("text", data_files=self.data_path)
        else:
            raise NotImplementedError(f"File type {self.file_type} is not supported.")

        if len(raw_datasets) == 1:
            split_key = list(raw_datasets.keys())[0]
            raw_datasets = raw_datasets[split_key]

        return raw_datasets, type(raw_datasets)

    def load_and_split_dataset(self,
                               raw_dataset: Dataset = None,
                               train_ratio: float = 0.8) -> DatasetDict:
        """
        This function is used to split the dataset into train and test.
        """
        if raw_dataset is None:
            raw_dataset = self.load_dataset()

        if isinstance(raw_dataset, DatasetDict):
            raise TypeError("DatasetDict is already split, check the key of it")

        raw_dataset = raw_dataset.train_test_split(train_size=train_ratio, seed=self.random_seed)
        return raw_dataset

    def sample_balanced_subsets(self,
                                raw_dataset: Dataset,
                                target_col_name: str = "label",
                                num_for_each_class: int = None) -> Dataset:
        """
        This function is used to sample balanced subsets from dataset.
        """
        # get the num of data in each class
        num_data = {}
        # if no key named "train", change here to fit your dataset or use dataset.train_test_split() to split it
        for label in raw_dataset[target_col_name]:
            if label not in num_data:
                num_data[label] = 1
            else:
                num_data[label] += 1

        # randomly sample the min_num_data from each class
        min_num_data = min(num_data.values()) if num_for_each_class is None else min(num_for_each_class,
                                                                                     min(num_data.values()))
        sampled_data = []
        for label in num_data:
            sampled_data.append(raw_dataset.filter(lambda example: example[target_col_name] == label).shuffle(
                seed=self.random_seed).select(range(min_num_data)))

        # rank the sampled data in turn of label, for SIMCSE pretrain
        ranked_data = []
        for i in range(min_num_data):
            for l in range(len(sampled_data)):
                ranked_data.append(sampled_data[l][i])

        balanced_subset = datasets.Dataset.from_list(ranked_data)
        return balanced_subset

    def tokenize_dataset(self,
                         raw_dataset: Dataset,
                         col_names: List,
                         max_seq_length: int = 512) -> Dataset:
        """
        This function is used to tokenize the dataset.
        """
        tokenized_dataset = None
        for col_name in col_names:
            tokenized_dataset = raw_dataset.map(lambda example: self.tokenizer(example[col_name],
                                                                               truncation=True,
                                                                               padding="max_length",
                                                                               max_length=max_seq_length),
                                                batched=True)
        return tokenized_dataset

    def collate_for_model(self,
                          raw_dataset: Dataset,
                          feature2input: Dict,
                          max_seq_length: int = 512) -> Dataset:
        """
        This function is used to prepare the dataset for model inputs
        feature2input: a dict that maps feature name to model's input parameters' name
                        {"labels": "your_label", "input_ids": ["sentence1", sentence2, ...], ...}
        """
        # map the labels
        raw_dataset = raw_dataset.rename_column(feature2input["labels"], "labels")

        # merge the features
        for feature in feature2input:
            if feature != "labels":
                raw_dataset = self.merge_columns(self,
                                                 raw_dataset=raw_dataset,
                                                 col_names=feature2input[feature],
                                                 new_col_name="text")

        # tokenize the merged features
        tokenized_dataset = self.tokenize_dataset(raw_dataset=raw_dataset,
                                                  col_names=["text"],
                                                  max_seq_length=max_seq_length)

        return tokenized_dataset

    @staticmethod
    def merge_columns(self,
                      raw_dataset: Dataset,
                      col_names: List,
                      new_col_name: str,
                      remain_old_columns: bool = False) -> Dataset:
        """
        This function is used to merge columns into one column.
        """
        merged_dataset = raw_dataset.map(lambda example:
                                         {new_col_name: " ".join([example[col_name] for col_name in col_names])})
        if not remain_old_columns:
            merged_dataset = merged_dataset.remove_columns(col_names)

        return merged_dataset

    @staticmethod
    def save_to_json(self,
                     dataset: [DatasetDict, Dataset],
                     save_path: str,
                     **to_json_kwargs):
        """
        This function is used to save the dataset to json file.
        """
        if isinstance(dataset, Dataset):
            dataset.to_json(save_path, **to_json_kwargs)
        elif isinstance(dataset, DatasetDict):
            for split in dataset:
                file_name = save_path.split(".json")[0] + f"_{split}.json"
                dataset[split].to_json(file_name, **to_json_kwargs)
        else:
            raise TypeError(f"Dataset type {type(dataset)} is not supported.")


if __name__ == "__main__":
    from transformers import AutoTokenizer

    my_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    args = CLSDatasetArguments(data_path="../data_lib/chatgpt_review/chatgpt_reviews.csv")
    my_dataset = CLSDataManager(args=args, tokenizer=my_tokenizer)

    r_dataset, typ = my_dataset.load_dataset()
    print(r_dataset, typ, sep="\n")

    s_dataset = my_dataset.load_and_split_dataset(raw_dataset=r_dataset)
    print(s_dataset)

    # b_dataset = my_dataset.sample_balanced_subsets(raw_dataset=s_dataset["train"],
    #                                                target_col_name="rating",
    #                                                num_for_each_class=16)
    # print(b_dataset, b_dataset.num_rows, sep="\n")
    #
    # t_dataset = my_dataset.tokenize_dataset(raw_dataset=s_dataset["train"],
    #                                         col_names=["title", "review"])
    # print(t_dataset)
    #
    # m_dataset = my_dataset.merge_columns(raw_dataset=s_dataset["train"],
    #                                      col_names=["title", "review"],
    #                                      new_col_name="text")
    # print(m_dataset)
    #
    # my_dataset.save_to_json(dataset=s_dataset, save_path="../data_lib/chatgpt_review/chatgpt_reviews.json")
    # reload = load_dataset("json", data_files="../data_lib/chatgpt_review/chatgpt_reviews_train.json")
    # print(reload)

    input_dataset = my_dataset.collate_for_model(raw_dataset=s_dataset["train"],
                                                 feature2input={"input_ids": ["title", "review"],
                                                                "labels": "rating"})
    print(input_dataset)
