import torch.nn
from torch import optim, nn

from .models import SkipGram
from .dataloader import Word2VecDataset, SentimentAnalysisDataset
from .utils import cosine_similarity
from .models import NegativeSamplingLoss, SentimentAnalysis
import traintracker
from traintracker import TrackerMod


def skipgram_train(model: SkipGram, epochs, skip_gram_data: Word2VecDataset, device='cpu', **kwargs):
    if 'optimizer' not in kwargs:
        optimizer = optim.Adam(model.parameters(), lr=0.003)
    else:
        optimizer = kwargs['optimizer']
    if 'criterion' not in kwargs:
        criterion = NegativeSamplingLoss()
    else:
        criterion = kwargs['criterion']

    # train tracker is a package that tracks the change of the hyperparameters during training and if it changed from
    # the last saved one it saves the new hyperparameters and tracks the epoch  number of changing the hyperparmater
    steps = 0
    # every 10%
    print_every = int(len(skip_gram_data) * 0.1)
    # (n_train_batches, train_batch_size)
    train_data_size = skip_gram_data.no_batches, skip_gram_data.batch_size
    # pass the hyperparameter you want to track to train tracker as dictionary
    hyperparameters = {"batch size": skip_gram_data.batch_size, "optimizer": optimizer,
                       "embedding_size": model.embedding_size, "window_size": skip_gram_data.window_size,
                       "no noise samples": skip_gram_data.no_noise_outputs}
    if "notes" in kwargs:
        hyperparameters['notes'] = kwargs['notes']
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
            # prints avg loss , time remaining
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
        # prints time taken and the train and test loss and if the test loss decreased it saves the model weights
        train_tracker.end_epoch()


def sentiment_model_test(model: SentimentAnalysis, test_data: SentimentAnalysisDataset,
                         criterion, train_tracker: traintracker.TrainTracker, device='cpu'):
    model.eval()
    train_tracker.valid()
    hidden = None
    avg_test_loss = 0
    with torch.no_grad():
        for review_batch, output_batch in test_data:
            review_batch, output_batch = review_batch.to(device), output_batch.to(device)

            predicted_output, hidden = model(review_batch, hidden)
            loss = criterion(predicted_output, output_batch)
            avg_test_loss = train_tracker.step(loss.item())

    return avg_test_loss


def sentiment_model_train(model: SentimentAnalysis, epochs, train_data: SentimentAnalysisDataset,
                          test_data: SentimentAnalysisDataset, device='cpu', **kwargs):
    if 'optimizer' not in kwargs:
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    else:
        optimizer = kwargs['optimizer']
    if 'criterion' not in kwargs:
        criterion = torch.nn.BCELoss()
    else:
        criterion = kwargs['criterion']

    if 'grad_clip' not in kwargs:
        grad_clip = 5
    else:
        grad_clip = kwargs['grad_clip']

    hyperparameters = {"batch_size": train_data.batch_size, "optimizer": optimizer,
                       "review length": train_data.review_len, "grad clip": grad_clip}
    if "notes" in kwargs:
        hyperparameters['notes'] = kwargs['notes']

    train_data_size = train_data.no_batches, train_data.batch_size
    test_data_size = test_data.no_batches, test_data.batch_size
    train_tracker = traintracker.TrainTracker(model=model, tracker_mod=TrackerMod.TRAIN_TEST,
                                              train_data_size=train_data_size, test_data_size=test_data_size,
                                              train_data_dir=kwargs['train_data_dir'], hyperparameters=hyperparameters,
                                              weights_dir=kwargs['weights_dir'])
    train_losses, valid_losses = [], []
    print("Testing before training")
    test_tracker = traintracker.TrainTracker(model, tracker_mod=TrackerMod.TEST_ONLY, test_data_size=test_data_size)
    test_loss = sentiment_model_test(model, test_data, criterion, test_tracker, device=device)
    print(f"Test Loss :{round(test_loss, 3)}")

    for e in range(epochs):
        model.train()
        train_tracker.train()
        hidden = None
        for review_batch, output_batch in train_data:
            review_batch, output_batch = review_batch.to(device), output_batch.to(device)

            predicted_output, hidden = model(review_batch, hidden)
            loss = criterion(predicted_output, output_batch)

            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            train_tracker.step(loss.item())

            optimizer.zero_grad()
            hidden = tuple(h.data for h in hidden)

        valid_loss = sentiment_model_test(model, test_data=test_data, criterion=criterion,
                                          train_tracker=train_tracker, device=device)

        train_loss, valid_loss = train_tracker.end_epoch()
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
