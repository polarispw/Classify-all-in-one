"""
transfer other formats to json and conduct division
"""
import json

import pandas as pd
from dataclasses import dataclass

@dataclass
class DatasetDescription:
    """
    This class is used to describe the dataset.
    """
    task_name: str
    data_dir: str
    train_file: str
    dev_file: str
    test_file: str
    header: bool
    label_list: list
    text_columns: list # if in .json, text_colums should be keys
    label_columns: list # if in .json, label_colums should be str(label)
    train_sample_num: int
    dev_sample_num: int
    test_sample_num: int


datasets_mapping = {
    "detector": DatasetDescription(
        task_name="mr",
        data_dir="../data_lib/mr/",
        train_file="../data_lib/mr/train.json",
        dev_file="../data_lib/mr/dev.json",
        test_file="../data_lib/mr/test.json",
        header=False,
        label_list=["0", "1"],
        text_columns=[1],
        label_columns=[0],
        train_sample_num=5331,
        dev_sample_num=2665,
        test_sample_num=2665,
    ),
    "llm": DatasetDescription(
        task_name="sst-5",
        data_dir="../data_lib/chatgpt_reviews.csv",
        train_file="../data_lib/chatgpt_reviews_train.json",
        dev_file="../data_lib/chatgpt_reviews_dev.json",
        test_file="../data_lib/chatgpt_reviews_test.json",
        header=True,
        label_list=["1", "2", "3", "4", "5"],
        text_columns=[1, 2],
        label_columns=[3],
        train_sample_num=0,
        dev_sample_num=0,
        test_sample_num=0,
    ),
    "nlp": DatasetDescription(
        task_name="subj",
        data_dir="../data_lib/subj/",
        train_file="../data_lib/subj/train.json",
        dev_file="../data_lib/subj/dev.json",
        test_file="../data_lib/subj/test.json",
        header=False,
        label_list=["0", "1"],
        text_columns=[1],
        label_columns=[0],
        train_sample_num=5331,
        dev_sample_num=2665,
        test_sample_num=2665,
    ),
}


def load_data_from_rawfile(data_desc: DatasetDescription):
    """
    This function is used to load data from files: .csv, .json, .txt
    And will return a list of data, each item is a dict of {labels, texts}
    according to the data description.
    :param data_desc:
    :return: [{labels: , texts: }, ...]
    """
    data = []
    data_dir = data_desc.data_dir
    file_format = data_dir.split('.')[-1]
    if file_format == 'csv':
        data_file = pd.read_csv(data_dir, header=None).values.tolist()
        if data_desc.header:
            data_file = data_file[1:]
        for line in data_file:
            item = {"labels": [str(line[idx]) for idx in data_desc.label_columns],
                    "texts": [str(line[idx]) for idx in data_desc.text_columns]}
            data.append(item)

    elif file_format == 'json':
        with open(data_dir, 'r', encoding='utf-8') as f:
            data_file = json.load(f)
            if data_desc.header:
                data_file = data_file[1:]
            for line in data_file:
                item = {"labels": [k for k in data_desc.label_columns],
                        "texts": [str(line[k]) for k in data_desc.text_columns]}
                data.append(item)

    elif file_format == 'txt':
        with open(data_dir, 'r', encoding='utf-8') as f:
            data_file = f.readlines()
            if data_desc.header:
                data_file = data_file[1:]
            for line in data_file:
                item = {"labels": [str(line[idx]) for idx in data_desc.label_columns],
                        "texts": [str(line[idx]) for idx in data_desc.text_columns]}
                data.append(item)

    else:
        raise Exception("file format not supported.")

    return data


def random_divide_dataset(data_desc, ratio=None):
    """
    This function is used to randomly divide a dataset into train/dev/test sets and save them into files.
    :param data_desc:
    :param ratio:
    :return: datasets
    """
    # Load data
    data = load_data_from_rawfile(data_desc)

    # Randomly shuffle data
    if ratio is None:
        ratio = [0.8, 0.1, 0.1]
    train_sample_num = int(len(data) * ratio[0])
    dev_sample_num = int(len(data) * ratio[1])
    test_sample_num = int(len(data) * ratio[2])

    import random
    random.shuffle(data)

    # Divide data
    train_data = data[:train_sample_num]
    dev_data = data[train_sample_num:train_sample_num + dev_sample_num]
    test_data = data[train_sample_num + dev_sample_num:train_sample_num + dev_sample_num + test_sample_num]
    print(len(data))
    # Save data as json files
    data_dir = data_desc.data_dir
    train_file = data_dir.rsplit('.', 1)[0] + '_train.json'
    dev_file = data_dir.rsplit('.', 1)[0] + '_dev.json'
    test_file = data_dir.rsplit('.', 1)[0] + '_test.json'
    print(train_file, dev_file, test_file)
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False)
    with open(dev_file, 'w', encoding='utf-8') as f:
        json.dump(dev_data, f, ensure_ascii=False)
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False)
    print(f"finish writing in {data_dir.rsplit('.', 1)[0]}")
    return train_data, dev_data, test_data


def load_datasets_from_json(data_desc: DatasetDescription):
    """
    This function is used to load data from json files.
    :param data_desc:
    :return: [{labels, texts}, ...], [...], [...]
    """
    with open(data_desc.train_file, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open(data_desc.dev_file, 'r', encoding='utf-8') as f:
        dev_data = json.load(f)
    with open(data_desc.test_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    return train_data, dev_data, test_data


if __name__ == "__main__":
    test_dataset = load_data_from_rawfile(datasets_mapping["llm"])
    print(test_dataset[0])
    random_divide_dataset(datasets_mapping["llm"])
    train_data, dev_data, test_data = load_datasets_from_json(datasets_mapping["llm"])
    print(train_data[0])
    print(len(train_data), len(dev_data), len(test_data))
