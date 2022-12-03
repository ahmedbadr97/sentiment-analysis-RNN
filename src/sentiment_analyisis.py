from . import SentimentAnalysis as SentimentAnalysisModel
from . import utils
import torch
from . import data_preprocessing

model = None


def load_model(weights_path: str, word2dict_path, device='cpu', parameters: dict = None):
    global model

    word2dict = utils.load_json(word2dict_path)
    if parameters is None:
        model = SentimentAnalysisModel(word2dict, hidden_nodes=256, n_layers=2, embedding_dim=300)
    else:
        model = SentimentAnalysisModel(word2dict, **parameters)
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device)
    return model


def predict(review: str):
    global model
    if model is None:
        raise Exception("please load the model first")
    review = data_preprocessing.remove_punctuations(review)

    review_words = review.split()
    int_review = []

    for word in review_words:
        if word in model.word2int:
            int_review.append(model.word2int[word])

    model.eval()
    out,hidden=model(torch.tensor([int_review],dtype=torch.long))

    return out.item()



