import os
import random

import numpy as np

from collections import Counter
import torch
import json


class Word2VecDataset:

    def __init__(self, txt: str, batch_size, window_size=5, no_noise_outputs=25, word2int: dict = None):

        self.window_size = window_size
        self.no_noise_outputs = no_noise_outputs
        self.batch_size = batch_size
        words = txt.split()
        self.no_words = len(words)

        self.words_counter = Counter(words)
        # add * to be a padding word, and it will have the index zero after sorting by count
        self.words_counter['*'] = np.Inf
        words_list = sorted(self.words_counter, key=self.words_counter.get, reverse=True)
        # setting its freq to zero again to not select it in frequency distribution while choosing noise words
        self.words_counter['*'] = 0

        if word2int is None:
            self.word2int = {}
            self.int2word = {}
            for idx, word in enumerate(words_list):
                self.word2int[word] = idx
                self.int2word[idx] = word
        else:
            self.word2int = word2int
            for word, int_value in word2int.items():
                self.int2word[int_value] = word

        self.no_unique_words = len(words_list)

        self.int_txt = [self.word2int[word] for word in txt.split()]
        # vector of vocab size each index will have the probability of selecting the word of this index
        # we will use unigram distribution power 3/4
        freq_array = np.array([self.words_counter[word] for word in words_list], dtype=float)
        self.noise_distribution = torch.tensor((freq_array ** 0.75) / np.sum(freq_array ** 0.75))
        self.no_batches = self.no_words // self.batch_size

    def __iter__(self):

        indices_distribution = [i for i in range(self.no_words)]
        random.shuffle(indices_distribution)
        word_idx_iter = iter(indices_distribution)

        for batch_idx in range(self.no_batches):

            batch_x, batch_y, batch_noise = [], [], []

            for _ in range(self.batch_size):
                word_idx = next(word_idx_iter)

                int_word = self.int_txt[word_idx]

                # max for avoiding negative indices
                r_window_size = random.randint(1, self.window_size + 1)
                start_idx = max(0, word_idx - r_window_size)

                # min for avoiding out of bounds indices
                end_idx = min(self.no_words - 1, word_idx + r_window_size)

                # end_idx+1 because python slicing end not inclusive
                y = self.int_txt[start_idx:word_idx] + self.int_txt[word_idx + 1:end_idx + 1]
                # extend input vector to be like input
                x = [int_word for _ in range(len(y))]

                # torch.multinomial takes array of probability  of selecting each index the array and no of samples (indices)
                # you want to take which is the indices of the selected words
                noise_output = torch.multinomial(self.noise_distribution, len(y) * self.no_noise_outputs,
                                                 replacement=True).view(len(y),
                                                                        self.no_noise_outputs).tolist()

                batch_x.extend(x)
                batch_y.extend(y)
                batch_noise.extend(noise_output)

            # noise output
            # noise_dist_target = {}

            # first set the portability of selecting the right output to be 0
            # for word_int in y:
            #     noise_dist_target[word_int] = self.noise_distribution[word_int]
            #     self.noise_distribution[word_int] = 0.0

            #  reset the portability of selecting the right output to be its value
            # for word_int in y:
            #     self.noise_distribution[word_int] = noise_dist_target[word_int]

            yield torch.tensor(batch_x, dtype=torch.long), torch.tensor(batch_y, dtype=torch.long), torch.tensor(
                batch_noise)

    def __len__(self):
        return self.no_batches

    def save_word2int(self, path):
        with open(os.path.join(path, "word2int.json"), 'w') as json_file:
            json_obj = json.dumps(self.word2int)
            json_file.write(json_obj)


class SentimentAnalysisDataset:
    def __init__(self, reviews_txt: str, labels_txt, word2int: dict, batch_size, def_review_len=800):

        self.word2int = word2int
        self.int2word = {}
        self.review_len = def_review_len
        self.batch_size = batch_size
        for word, int_value in word2int.items():
            self.int2word[int_value] = word

        txt_reviews_list = reviews_txt.split('\n')
        self.int_rev_list = []
        self.no_batches = len(txt_reviews_list) // batch_size

        for review in txt_reviews_list:
            new_review = [0 for _ in range(def_review_len)]
            review_words = review.split()
            int_review = []
            # change review from words to int
            for word in review_words:
                if word in self.word2int:
                    int_review.append(self.word2int[word])
            """
            we have the new_words array of zeros with the fixed review length that me made
            the goal is to trim the reviews that have words more than the fixed length and left pad the reviews that 
            have lesser length
            """
            curr_rev_len = min(len(int_review), def_review_len)
            rev_idx = curr_rev_len - 1
            new_rev_idx = def_review_len - 1
            for _ in range(curr_rev_len):
                new_review[new_rev_idx] = int_review[rev_idx]
                new_rev_idx -= 1
                rev_idx -= 1
            self.int_rev_list.append(new_review)
        self.labels_list = []
        for label in labels_txt.split('\n')[:-1]:
            if label == 'positive':
                self.labels_list.append(1)
            else:
                self.labels_list.append(0)

    def __iter__(self):
        random_indices = [i for i in range(len(self.int_rev_list))]
        random.shuffle(random_indices)
        idx_iter = iter(random_indices)
        for _ in range(self.no_batches):
            batch_x = []
            batch_y = []
            for i in range(self.batch_size):
                idx = next(idx_iter)
                batch_x.append(self.int_rev_list[idx])
                batch_y.append(self.labels_list[idx])
            yield torch.tensor(batch_x, dtype=torch.long), torch.tensor(batch_y, dtype=torch.int)

    def __len__(self):
        return self.no_batches
