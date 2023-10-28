"""
This file contains metrics for different datasets
"""
import evaluate
import numpy as np

accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")


def compute_acc_f1(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    acc = accuracy.compute(predictions=predictions, references=labels)['accuracy']
    f1_score = f1.compute(predictions=predictions, references=labels, average='macro')['f1']
    return {"accuracy": f"{acc:.6f}", "f1": f"{f1_score:.6f}"}
