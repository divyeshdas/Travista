import torch
import time
from nltk.translate.bleu_score import sentence_bleu


def compute_bleu(reference, prediction):
    return sentence_bleu([reference.split()], prediction.split())


def compute_perplexity(loss):
    return torch.exp(torch.tensor(loss)).item()


def compute_accuracy(reference, prediction):
    return int(reference.strip() == prediction.strip())


def measure_inference_time(model_func, sentence):
    start = time.time()
    model_func(sentence)
    end = time.time()
    return end - start