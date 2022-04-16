import numpy as np
from scipy import io


def str_to_num(char):
    char = char.lower()
    if char.isalpha():
        num = ord(char) - 86
    else:
        num = int(char)
    return num


def lire_alpha_digit(path, chars=None):
    """
    Given the path of the alpha digit database, and a string of characters this
    function return the matrices of all images corresponding to the characters
    present in the string

    Args:

        path (str): path of the file containing the alphadigits database
        chars (str): string containing all the characters for which we want to
                     extract the matrices for training

    Returns:
            mats (array): array of the matrices flattened
            inidices (array): array of the indices of the labels

    """
    obj = io.loadmat(path)
    mats, indices = [], []
    if isinstance(chars, str) and chars.isalnum():
        chars = sorted(set(chars))
        for char in chars:
            index = str_to_num(char)
            for mat in obj['dat'][index]:
                mats.append(mat.flatten())
                indices.append(index)
        mats, indices = np.array(mats), np.array(indices)
    return mats, indices


indices_to_labels = {}
for char in '0123456789abcdefghifjklmnopqrstuvwxyz':
    indices_to_labels[str_to_num(char)] = char

labels_to_indices = {key: value for key, value in indices_to_labels.items()}
