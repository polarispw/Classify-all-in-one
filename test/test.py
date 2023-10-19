from datasets import load_dataset

path = r"E:\Github\NLP-project-template\data_lib\fake_reviews\fake reviews dataset.csv"
data_files = {"train": [path]}
my_dataset = load_dataset("csv", data_files=data_files, split="train")

print(my_dataset[0])
print(my_dataset.features)
