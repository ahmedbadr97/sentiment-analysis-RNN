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

        expected_output_loss = torch.bmm(out_words_emb, in_words_emb).sigmoid().log()
        expected_output_loss = expected_output_loss.squeeze()

        noise_loss = torch.bmm(noise_words_emb.neg(), in_words_emb).sigmoid().log()
        # sum the losses over the sample of noise vectors
        noise_loss = noise_loss.squeeze().sum(1)

        # calculate the mean for the batch
        return -(expected_output_loss + noise_loss).mean()


class SentimentAnalysis(nn.Module):
    def __init__(self, word2int: dict, embedding_dim=256, hidden_nodes=256, n_layers=2, embedding_layer=None,
                 dropout_prop=0.5):
        super().__init__()

        self.word2int = word2int
        self.n_layers = n_layers
        self.hidden_nodes = hidden_nodes

        self.embedding_layer = nn.Embedding(len(word2int),
                                            embedding_dim) if embedding_layer is None else embedding_layer
        self.lstm = nn.LSTM(input_size=embedding_dim, num_layers=n_layers, hidden_size=hidden_nodes,
                            dropout=dropout_prop, batch_first=True)
        self.dropout = nn.Dropout(dropout_prop)
        self.fc = nn.Linear(hidden_nodes, 1)
        self.sigmoid = nn.Sigmoid()
        self.input_size = len(word2int)

    def _init_hidden(self, batch_size):
        param_iter = self.parameters()
        next(param_iter)
        lstm_weights = next(param_iter)

        hidden = (
            lstm_weights.new(self.n_layers, batch_size, self.hidden_nodes).zero_(),
            lstm_weights.new(self.n_layers, batch_size, self.hidden_nodes).zero_()
        )
        return hidden

    def forward(self, x: torch.Tensor, hidden=None):
        batch_size = x.shape[0]
        if hidden is None:
            hidden = self._init_hidden(batch_size)
        embedding = self.embedding_layer(x)
        lstm_out, lstm_hidden = self.lstm(embedding, hidden)

        # to (batch_size*seq_len,hidden_nodes)
        fc_input = lstm_out.contiguous().view(-1, self.hidden_nodes)
        fc_input = self.dropout(fc_input)
        output = self.fc(fc_input)

        output = self.sigmoid(output)  # (batch_size*seq_len,1)
        # we want it to be (batch,seq_len) to get the last output of the sequence which is the last word in the review
        # that our output depend on
        output = output.view(batch_size, -1)
        # select only the last output of the whole sequence
        output = output[:, -1]

        return output, hidden

