import numpy as np

def encode_labels(label_string: str, classes:list) -> np.ndarray:
    """
    Convert a pipe-separated label string to a multi-hot binary vector

    Args:
        label string: e.g. 'Atelectasis|Effusion' or 'No Finding'
        classes: ordered list of 14 disease class names

    Returns:
        np.ndarray of shape (14,) with 0s and 1s
    """
    vec = np.zeros(len(classes), dtype=np.float32)
    if label_string == 'No Finding':
        return vec # all zeros
    for label in label_string.split('|'):
        label = label.strip() # removes whitespace form the beginning and end of the string
        if label in classes:
            vec[classes.index(label)] = 1.0
    return vec



