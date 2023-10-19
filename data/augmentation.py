"""
this file contains data augmentation methods
"""

import argparse
import json
import random
from os import wait

from transformers import pipeline
from file_processor import datasets_mapping, load_data_from_rawfile
from utils.chatgpt_api import gpt_35_api_stream


class Augmenter(object):
    """
    this class is used to augment the data
    it includes methods: substitution by pretrained model, back translation, and prompting chatgpt
    """

    def __init__(self, args):
        self.model = args.model
        self.tokenizer = args.tokenizer
        self.method = args.method
        self.metric = args.metric
        self.data_dir = args.rawdata_dir
        self.cache_dir = args.cache_dir
        self.inplace = args.inplace

        # load data and shuffle
        self.data_list = load_data_from_rawfile(datasets_mapping[args.task])
        random.shuffle(self.data_list)

        self.augmented_data_list = []
        self.num = args.num
        self.label = args.label
        self.to_do_list = {self.label[i]: self.num[i] for i in range(len(self.label))}
        self.eval_score = 0

    def calculate_distribution(self):
        """
        this function is used to calculate the distribution of labels
        """
        res = {}
        for data in self.data_list:
            for label in data["labels"]:
                if label not in res:
                    res[label] = 1
                else:
                    res[label] += 1
        for k, v in res.items():
            res[k] = v / len(self.data_list)
        return res

    def substitution(self):
        """
        this function is used to substitute some words to [MASK] token in the sentence
        and then use the pre-trained model to predict the [MASK]
        supporting models: bert, roberta
        """
        mask_filler = pipeline("fill-mask",
                               model=self.model,
                               tokenizer=self.tokenizer,
                               cache_dir=self.cache_dir)
        for data in self.data_list:
            for label in data["labels"]:
                if self.to_do_list[label] > 0:
                    new_data = {"labels": data["labels"], "texts": []}
                    for text in data["texts"]:
                        text_ids = self.tokenizer.tokenize(text)

                        # randomly choose some words to mask
                        mask_ids = []
                        for i in range(len(text_ids)):
                            if text_ids[i] != "[CLS]" and text_ids[i] != "[SEP]" and text_ids[i] != "[PAD]":
                                mask_ids.append(i)
                        mask_num = int(len(mask_ids) * 0.15)
                        mask_ids = random.sample(mask_ids, mask_num)
                        for mask_id in mask_ids:
                            text_ids[mask_id] = "[MASK]"
                        text = self.tokenizer.convert_tokens_to_string(text_ids)

                        # predict the masked words
                        sub_res = mask_filler(text)[0]["sequence"]

                        # evaluate the augmentation
                        self.evaluate(text, sub_res)

                        new_data["texts"].append(sub_res)
                    self.augmented_data_list.append(new_data)
                    self.to_do_list[label] -= 1
                else:
                    continue

        return self.augmented_data_list

    def back_translation(self):
        """
        this function is used to translate the sentence to another language and then translate back
        supporting models: bert, roberta
        """
        translator1 = pipeline("translation_en_to_fr",
                               model=self.model,
                               tokenizer=self.tokenizer,
                               cache_dir=self.cache_dir)
        translator2 = pipeline("translation_fr_to_en",
                               model=self.model,
                               tokenizer=self.tokenizer,
                               cache_dir=self.cache_dir)
        for data in self.data_list:
            for label in data["labels"]:
                if self.to_do_list[label] > 0:
                    new_data = {"labels": data["labels"], "texts": []}
                    for text in data["texts"]:
                        # translate to another language
                        new_text = translator1(text)
                        # translate back
                        new_text = translator2(new_text)
                        self.evaluate(text, new_text)
                        new_data["texts"].append(new_text)
                    self.augmented_data_list.append(new_data)
                    self.to_do_list[label] -= 1
                else:
                    continue

        return self.augmented_data_list

    def prompting_chatgpt(self):
        """
        this function is used to use ICL ability of chatgpt to generate the sentence
        free api is limited to 60 requests per hour
        """
        for data in self.data_list:
            for label in data["labels"]:
                if self.to_do_list[label] > 0:
                    new_data = {"labels": data["labels"], "texts": []}
                    for text in data["texts"]:
                        message = [{'role': 'user',
                                    'content': f'rewrite the following sentence: {text}'}]
                        if gpt_35_api_stream(message):
                            new_data["texts"].append(message[-1]["content"])
                        else:
                            raise Exception("chatgpt api error")
                        wait(60)   # wait for 1 minute
                    self.augmented_data_list.append(new_data)
                    self.to_do_list[label] -= 1
                else:
                    continue

        return self.augmented_data_list

    def get_augmented_data(self):
        """
        this function is used to get the augmented data
        """
        if self.method == "substitution":
            self.data_list.append(self.substitution())
        elif self.method == "back_translation":
            self.data_list.append(self.back_translation())
        elif self.method == "prompting_chatgpt":
            self.data_list.append(self.prompting_chatgpt())
        else:
            raise ValueError("the augmentation method is not supported")

        self.save()

        return

    def evaluate(self, text1, text2):
        """
        this function is used to evaluate the augmentation quality
        """
        pass

    def save(self):
        """
        this function is used to save the augmented data to json file
        """
        self.data_list.append(self.augmented_data_list)
        if self.inplace:
            augmented_file = datasets_mapping[self.args.task].train_file
        else:
            augmented_file = datasets_mapping[self.args.task].train_file.rsplit('.', 1)[0] + '_augmented.json'
        with open(augmented_file, 'w', encoding='utf-8', newline='\n') as f:
            json.dump(self.data_list, f, ensure_ascii=False, indent=4)
        print(f"finish writing in {augmented_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="llm", type=str, help="the task name")
    parser.add_argument("--model", default="bert-base-uncased", type=str, help="the model name")
    parser.add_argument("--cache_dir", default="../model_cache", type=str, help="the cache dir of model")
    parser.add_argument("--tokenizer", default="bert-base-uncased", type=str, help="the tokenizer name")
    parser.add_argument("--method", default="substitution", type=str, help="the augmentation method")
    parser.add_argument("--metric", default="CosineSimilarity", type=str, help="metric to evaluate the augmentation")
    parser.add_argument("--data_dir", default="data", type=str, help="the data path")
    parser.add_argument("--num", default=None, type=list, help="the numbers of augmented data")
    parser.add_argument("--label", default=None, type=list, help="the labels of augmented data")
    parser.add_argument("--inplace", default=False, type=bool, help="whether to do changes inplace")
    args = parser.parse_args()

    test_augmenter = Augmenter(args)
    print(test_augmenter.calculate_distribution())
