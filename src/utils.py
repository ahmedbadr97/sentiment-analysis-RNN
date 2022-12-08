import json
import random

import numpy as np
import torch
from IPython.display import display_html
from itertools import chain, cycle


def display_side_by_side(*args, titles=cycle([''])):
    html_str = ''
    for df, title in zip(args, chain(titles, cycle(['</br>']))):
        html_str += '<th style="text-align:center"><td style="vertical-align:top">'
        html_str += f'<h2 style="text-align: center;">{title}</h2>'
        html_str += df.to_html().replace('table', 'table style="display:inline"')
        html_str += '</td></th>'
    display_html(html_str, raw=True)


def cosine_similarity(embedding, valid_size=16, valid_window=100, device='cpu'):
    """ Returns the cosine similarity of validation words with words in the embedding matrix.
        Here, embedding should be a PyTorch embedding module.
    """

    # Here we're calculating the cosine similarity between some random words and
    # our embedding vectors. With the similarities, we can look at what words are
    # close to our random words.

    # sim = (a . b) / |a||b|

    embed_vectors = embedding.weight

    # magnitude of embedding vectors, |b|
    magnitudes = embed_vectors.pow(2).sum(dim=1).sqrt().unsqueeze(0)

    # pick N words from our ranges (0,window) and (1000,1000+window). lower id implies more frequent
    valid_examples = np.array(random.sample(range(valid_window), valid_size // 2))
    valid_examples = np.append(valid_examples,
                               random.sample(range(1000, 1000 + valid_window), valid_size // 2))
    valid_examples = torch.LongTensor(valid_examples).to(device)

    valid_vectors = embedding(valid_examples)
    similarities = torch.mm(valid_vectors, embed_vectors.t()) / magnitudes

    return valid_examples, similarities


def closest_words(embedding, int2word, colsest_n=6):
    valid_examples, valid_similarities = cosine_similarity(embedding)
    _, closest_idxs = valid_similarities.topk(colsest_n)

    valid_examples, closest_idxs = valid_examples.to('cpu'), closest_idxs.to('cpu')
    for ii, valid_idx in enumerate(valid_examples):
        closest_words = [int2word[idx.item()] for idx in closest_idxs[ii]][1:]
        print(int2word[valid_idx.item()] + " | " + ', '.join(closest_words))
    print("...\n")


def read_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()


def save_json(data: dict, file_path):
    with open(file_path, 'w') as json_file:
        json_obj = json.dumps(data)
        json_file.write(json_obj)


def load_json(file_path):
    with open(file_path, "r") as file:
        word2int = json.load(file)
    return word2int
