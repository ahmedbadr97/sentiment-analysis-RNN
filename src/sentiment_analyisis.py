from . import SentimentAnalysis as SentimentAnalysisModel
from . import utils
import torch
from . import data_preprocessing

model = None
current_device = 'cpu'


def load_model(weights_path: str, word2dict_path, device='cpu', parameters: dict = None):
    global model
    global current_device
    current_device = device
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
    global current_device
    if model is None:
        raise Exception("please load the model first")
    review = data_preprocessing.remove_punctuations(review)

    review_words = review.split()
    int_review = []

    for word in review_words:
        if word in model.word2int:
            int_review.append(model.word2int[word])
    review_tensor = torch.tensor([int_review], dtype=torch.long, device=current_device)

    model.eval()
    out, hidden = model(review_tensor)

    return out.item()
