from torch import optim

from .models import SkipGram
from .dataloader import Word2VecDataset
from .utils import cosine_similarity
from .models import NegativeSamplingLoss
import traintracker
from traintracker import TrackerMod


def train(model: SkipGram, epochs, skip_gram_data: Word2VecDataset, device='cpu', **kwargs):
    if 'optimizer' not in kwargs:
        optimizer = optim.Adam(model.parameters(), lr=0.003)
    else:
        optimizer = kwargs['optimizer']
    if 'criterion' not in kwargs:
        criterion = NegativeSamplingLoss()
    else:
        criterion = kwargs['criterion']

    # train for some number of epochs
    steps = 0
    # every 10%
    print_every = int(len(skip_gram_data) * 0.1)
    # (n_train_batches, train_batch_size)
    train_data_size = skip_gram_data.no_batches, skip_gram_data.batch_size
    hyperparameters = {"batch size": skip_gram_data.batch_size, "optimizer": optimizer,
                       "embedding_size": model.embedding_size, "window_size": skip_gram_data.window_size,
                       "no noise samples": skip_gram_data.no_noise_outputs}
    train_tracker = traintracker.TrainTracker(model=model, tracker_mod=TrackerMod.TRAIN_ONLY,
                                              train_data_size=train_data_size,
                                              train_data_dir=kwargs['train_data_dir'], hyperparameters=hyperparameters,
                                              weights_dir=kwargs['weights_dir'])
    for e in range(epochs):
        train_tracker.train()

        # get our input, target batches
        for input_words, target_words, noise_words in skip_gram_data:
            steps += 1

            inputs_words, targets_words, noise_words = input_words.to(device), target_words.to(device), noise_words.to(
                device)

            # input, output, and noise vectors
            in_words_emb, out_words_emb, noise_words_emb = model.forward(inputs_words, targets_words, noise_words)

            # negative sampling loss
            loss = criterion(in_words_emb, out_words_emb, noise_words_emb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_tracker.step(loss.item())

            # loss stats
            if steps % print_every == 0:
                print("Epoch: {}/{}".format(e + 1, epochs))
                print("Loss: ", loss.item())  # avg batch loss at this point in training
                valid_examples, valid_similarities = cosine_similarity(model.input_embedding, device=device)
                _, closest_idxs = valid_similarities.topk(6)

                valid_examples, closest_idxs = valid_examples.to('cpu'), closest_idxs.to('cpu')
                for ii, valid_idx in enumerate(valid_examples):
                    closest_words = [skip_gram_data.int2word[idx.item()] for idx in closest_idxs[ii]][1:]
                    print(skip_gram_data.int2word[valid_idx.item()] + " | " + ', '.join(closest_words))
                print("...\n")
        train_tracker.end_epoch()
