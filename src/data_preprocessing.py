from string import punctuation
from collections import Counter

import numpy as np

# r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""
# range(start, stop, step)

from pandas import DataFrame


def remove_punctuations(txt):
    new_txt = []
    punctuations = {'.', '?', '!', '#', '$', '%', '&', '(', ')', '*', ',', '+', '-', '/', ':', ';', '<', '=', '>',
                    '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~', '"', "'"}
    punctuation_map = {}
    for p in punctuations:
        punctuation_map[p] = f" {p} "
    for char in txt:

        if char not in punctuations:
            if char == '\n':
                char = ' \n '
            new_txt.append(char.lower())

    return "".join(new_txt)


def get_head_mid_tail(df: DataFrame, n: int):
    size = df.shape[0]
    mid_idx = size // 2
    indices = [i for i in range(n)] + [i for i in range(mid_idx - (n // 2), mid_idx + (n // 2))] + [i for i in
                                                                                                    range(size - n,
                                                                                                          size)]
    return df.iloc[indices]


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


def remove_noise(txt: str, filtering_ratio, prob_threshold=0.5, min_freq=2, min_rev_freq=None):
    # we will split the txt to lines  to keep the \n

    # till -1 to drop last empty line after the last \n

    word_counts = Counter(txt.split())
    total_count = sum(word_counts.values())
    word_rev_count = Counter()

    word_weight = {word: count / total_count for word, count in word_counts.items()}
    p_drop = {word: 1 - np.sqrt(1.0 / (word_weight[word] * filtering_ratio)) for word in word_counts}

    noisy_words = set()
    lines = txt.split('\n')

    new_txt = []

    if min_rev_freq is not None:
        for line in lines:
            word_split_set = set(line.split())
            for word in word_split_set:
                word_rev_count[word] += 1

    for line in lines:
        word_split_txt = line.split()

        for word in word_split_txt:
            not_noise = p_drop[word] <= prob_threshold and word_counts[word] >= min_freq
            if min_rev_freq is not None:
                not_noise &= word_rev_count[word] >= min_rev_freq

            if not_noise:
                new_txt.append(word)
                new_txt.append(" ")
            else:
                noisy_words.add(word)
        new_txt.append('\n')

    # till -1 to drop last empty line after the last \n
    return "".join(new_txt[:-1]), list(noisy_words), p_drop


def remove_common_words(reviews: str, labels: str, threshold: float, min_freq: int):
    """
    remove common words that appear in both positive reviews and negative reviews, so it doesn't have weight for deciding
    whether this review is positive or negative
    :param reviews: reviews txt line separated
    :param labels: labels txt line separated
    :param threshold: threshold of removing common word from 0.0 to 4.0
    :param min_freq: minimum frequency for the word to keep
    :return: (clean_reviews_txt,pos_negative_ratio_dict,removed words)
    """

    reviews_list = reviews.split('\n')
    labels_list = labels.split('\n')
    positive_words_cnt = Counter()
    negative_words_cnt = Counter()
    all_words_cnt = Counter()

    # calculate word importance
    for i in range(len(reviews_list)):
        words = reviews_list[i].split(" ")
        for word in words:
            if labels_list[i] == 'positive':
                positive_words_cnt[word] += 1
            else:
                negative_words_cnt[word] += 1
            all_words_cnt[word] += 1
    # %%
    words_pos_neg_ratio = Counter()
    for word, cnt in all_words_cnt.items():
        if cnt >= min_freq:
            words_pos_neg_ratio[word] = np.log((positive_words_cnt[word] + 1) / (negative_words_cnt[word] + 1))

    # remove non-important words

    new_txt = []
    noisy_words = set()
    for review in reviews_list:
        word_split_txt = review.split()

        for word in word_split_txt:
            if all_words_cnt[word] >= min_freq and abs(words_pos_neg_ratio[word]) >= threshold:
                new_txt.append(word)
                new_txt.append(" ")
            else:
                noisy_words.add(word)
        new_txt.append('\n')

    return "".join(new_txt), words_pos_neg_ratio, list(noisy_words)
