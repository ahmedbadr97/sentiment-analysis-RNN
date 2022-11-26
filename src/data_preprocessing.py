from string import punctuation
from collections import Counter

import numpy as np


def remove_punctuations(txt):
    new_txt = []
    for char in txt:
        if char not in punctuation:
            if char == '\n':
                char = ' \n '
            new_txt.append(char)

    return "".join(new_txt)


def get_noise_words(reviews_txt, labels_txt, threshold):
    # calculate the importance of each word using pos
    noisy_words = set()
    positive_words_cnt = Counter()
    negative_words_cnt = Counter()
    reviews_vocab = set()
    reviews = reviews_txt.split('\n')[:-1]
    labels = labels_txt.split('\n')[:-1]
    for i in range(len(reviews)):
        words = reviews[i].split(' ')
        for word in words:
            if labels[i] == 'positive':
                positive_words_cnt[word] += 1
            else:
                negative_words_cnt[word] += 1
            reviews_vocab.add(word)
    for word in reviews_vocab:
        word_ratio = np.log((positive_words_cnt[word] + 1) / (negative_words_cnt[word] + 1))
        if abs(word_ratio) < threshold:
            noisy_words.add(word)
    return noisy_words


def remove_noise(txt: str, filtering_ratio, prob_threshold=0.5, min_freq=2):
    # we will split the txt to lines  to keep the \n

    # till -1 to drop last empty line after the last \n

    word_counts = Counter(txt.split())
    total_count = sum(word_counts.values())

    word_weight = {word: count / total_count for word, count in word_counts.items()}
    p_drop = {word: 1 - np.sqrt(1.0 / (word_weight[word] * filtering_ratio)) for word in word_counts}

    noisy_words = set()
    lines = txt.split('\n')[:-1]

    new_txt = []
    for line in lines:
        word_split_txt = line.split()

        for word in word_split_txt:
            if p_drop[word] <= prob_threshold and word_counts[word] >= min_freq:
                new_txt.append(word)
                new_txt.append(" ")
            else:
                noisy_words.add(word)
        new_txt.append('\n')

    # till -1 to drop last empty line after the last \n
    return "".join(new_txt[:-1]), list(noisy_words)
