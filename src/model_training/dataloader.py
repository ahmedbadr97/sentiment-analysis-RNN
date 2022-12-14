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

        # save word2int and int2word dict
        self.int2word = {}
        if word2int is None:
            self.word2int = {}
            for idx, word in enumerate(words_list):
                self.word2int[word] = idx
                self.int2word[idx] = word
        else:
            self.word2int = word2int
            for word, int_value in word2int.items():
                self.int2word[int_value] = word

        self.no_unique_words = len(words_list)

        # save txt reviews as int , where each word has its own int value
        self.int_txt = [self.word2int[word] for word in txt.split()]

        # vector of vocab size each index will have the probability of selecting the word of this index
        # we will use uni-gram distribution power 3/4
        freq_array = np.array([self.words_counter[word] for word in words_list], dtype=float)
        self.noise_distribution = torch.tensor((freq_array ** 0.75) / np.sum(freq_array ** 0.75))
        self.no_batches = self.no_words // self.batch_size

    def __iter__(self):
        # walk on the txt randomly not word by word , first generate indices from 0 to the no of words in the txt
        # then shuffle
        indices_distribution = [i for i in range(self.no_words)]
        random.shuffle(indices_distribution)
        word_idx_iter = iter(indices_distribution)

        for batch_idx in range(self.no_batches):
            # create batch from input, target , noise words
            batch_x, batch_y, batch_noise = [], [], []
            # get noise outputs for all inputs in the batch
            # each input word will be repeated 2* window size each one of them needs n_noise output * batch_size
            batch_noise_tensor = torch.multinomial(self.noise_distribution,
                                                   self.batch_size * self.no_noise_outputs * self.window_size * 2,
                                                   replacement=True).view(
                self.batch_size, -1)

            for i in range(self.batch_size):
                word_idx = next(word_idx_iter)

                int_word = self.int_txt[word_idx]

                # max for avoiding negative indices
                r_window_size = random.randint(1, self.window_size)
                start_idx = max(0, word_idx - r_window_size)

                # min for avoiding out of bounds indices
                end_idx = min(self.no_words - 1, word_idx + r_window_size)

                # end_idx+1 because python slicing end not inclusive
                y = self.int_txt[start_idx:word_idx] + self.int_txt[word_idx + 1:end_idx + 1]
                # extend input vector to be like input
                x = [int_word for _ in range(len(y))]

                # get the actual window size *2 because window size is randomly generated from 1 to window size
                noise_output = batch_noise_tensor[i][:len(y) * self.no_noise_outputs].view(len(y),
                                                                                           self.no_noise_outputs).tolist()

                batch_x.extend(x)
                batch_y.extend(y)
                batch_noise.extend(noise_output)

            yield torch.tensor(batch_x, dtype=torch.long), torch.tensor(batch_y, dtype=torch.long), torch.tensor(
                batch_noise)

    def __len__(self):
        return self.no_batches

    def save_word2int(self, path):
        with open(os.path.join(path, "word2int.json"), 'w') as json_file:
            json_obj = json.dumps(self.word2int)
            json_file.write(json_obj)


class SentimentAnalysisDataset:
    def __init__(self, reviews_list: str, labels_list, word2int: dict, batch_size, def_review_len=800):

        self.word2int = word2int
        self.int2word = {}
        self.review_len = def_review_len
        self.batch_size = batch_size
        for word, int_value in word2int.items():
            self.int2word[int_value] = word

        self.int_rev_list = []
        self.no_batches = len(reviews_list) // batch_size
        rev_len_sum = 0.0

        for review in reviews_list:
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

            rev_len_sum += len(int_review)
            self.int_rev_list.append(new_review)

        self.avg_rev_len = int(float(rev_len_sum) / len(self.int_rev_list))

        self.labels_list = []
        for label in labels_list:
            if label == 'positive':
                self.labels_list.append(1)
            else:
                self.labels_list.append(0)
        self.rev_indices = [i for i in range(len(self.int_rev_list))]

    def __iter__(self):

        random.shuffle(self.rev_indices)
        idx_iter = iter(self.rev_indices)
        for _ in range(self.no_batches):
            batch_x = []
            batch_y = []
            for i in range(self.batch_size):
                idx = next(idx_iter)
                batch_x.append(self.int_rev_list[idx])
                batch_y.append(self.labels_list[idx])
            yield torch.tensor(batch_x, dtype=torch.long), torch.tensor(batch_y, dtype=torch.float)

    def __len__(self):
        return self.no_batches
