import torch
import torch.nn as nn


class SkipGram(nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super().__init__()
        self.input_embedding = nn.Embedding(vocab_size, embedding_size)
        self.output_embedding = nn.Embedding(vocab_size, embedding_size)

        self.input_embedding.weight.data.uniform_(-1, 1)
        self.output_embedding.weight.data.uniform_(-1, 1)

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

    def forward(self, input_words, output_words, noise_words):

        # batch_size*window_size
        # input_words = input_words.view(-1)
        # output_words = output_words.view(-1)
        # noise_words = noise_words.view(-1)

        in_words_emb = self.input_embedding(input_words)
        out_words_emb = self.output_embedding(output_words)
        noise_words_emb = self.output_embedding(noise_words)
        return in_words_emb, out_words_emb, noise_words_emb


class NegativeSamplingLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, in_words_emb, out_words_emb, noise_words_emb):
        # batchxWindow is the batch_size multiplied by windows size
        batch_x_window, no_noise_words, emb_size = noise_words_emb.shape
        # view as row vector
        in_words_emb = in_words_emb.view(batch_x_window, emb_size, 1)
        # view as col vector
        out_words_emb = out_words_emb.view(batch_x_window, 1, emb_size)

        expected_output_loss = torch.bmm( out_words_emb,in_words_emb).sigmoid().log()
        expected_output_loss = expected_output_loss.squeeze()

        noise_loss = torch.bmm(noise_words_emb.neg(),in_words_emb).sigmoid().log()
        # sum the losses over the sample of noise vectors
        noise_loss=noise_loss.squeeze().sum(1)

        # calculate the mean for the batch
        return -(expected_output_loss + noise_loss).mean()
