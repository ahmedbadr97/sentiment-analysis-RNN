import os.path

from .models import SentimentAnalysis as SentimentAnalysisModel
from . import utils
import torch
from .data_preprocessing import remove_punctuations
import gdown

_model = None
_current_device = 'cpu'
_default_weights_url = "https://drive.google.com/u/1/uc?id=1xree_w9VKqNwCP9djoN5_xBi2XH3XdfW&export=download"
_default_word2int_url = "https://drive.google.com/u/1/uc?id=1Uvo7JjsYkAborlgCfruk1SsIjw4g_1xR&export=download"


def load_model(weights_path: str = None, word2int_dict_path=None, device='cpu', parameters: dict = None):
    global _model
    global _current_device
    _current_device = device

    # download weights and word2int map if it isn't passed
    if word2int_dict_path is None or weights_path is None:
        # create a model_weights folder to download the weights and word2dict in if it isn't exist
        if not os.path.exists('./model_weights'):
            os.mkdir('./model_weights')
    if word2int_dict_path is None:
        word2int_dict_path = "./model_weights/int2word.json"
        if not os.path.exists(word2int_dict_path):
            gdown.download(_default_word2int_url, word2int_dict_path)
    if weights_path is None:
        weights_path = "./model_weights/latest_weights.pt"
        if not os.path.exists(weights_path):
            gdown.download(_default_weights_url, weights_path)

    word2dict = utils.load_json(word2int_dict_path)
    if parameters is None:
        _model = SentimentAnalysisModel(word2dict, hidden_nodes=256, n_layers=2, embedding_dim=256)
    else:
        _model = SentimentAnalysisModel(word2dict, **parameters)
    state_dict = torch.load(weights_path, map_location="cpu")
    _model.load_state_dict(state_dict)
    _model.to(device)
    return _model


def predict(review: str):
    global _model
    global _current_device
    if _model is None:
        raise Exception("please load the model first")
    review = remove_punctuations(review)

    review_words = review.split()
    int_review = []

    for word in review_words:
        if word in _model.word2int:
            int_review.append(_model.word2int[word])
    review_tensor = torch.tensor([int_review], dtype=torch.long, device=_current_device)

    _model.eval()
    out, hidden = _model(review_tensor)

    return out.item()
