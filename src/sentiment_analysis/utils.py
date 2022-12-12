import json


def load_json(file_path):
    with open(file_path, "r") as file:
        word2int = json.load(file)
    return word2int


def remove_punctuations(txt):
    """
    remove punctuations and change all characters to lowercase
    :param txt: reviews txt
    :return: cleaned text preserving the newline character
    """
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
