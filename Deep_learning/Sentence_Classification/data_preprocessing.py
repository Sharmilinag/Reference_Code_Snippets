from constants import puncts, contraction_dict
import re
import numpy as np

# ## Download from : https://nlp.stanford.edu/projects/glove/
EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'

def remove_punctuations(x):
    x = str(x)
    for punct in puncts:
        if punct in x:
            x = x.replace(punct, f' {punct} ')
    return x


# ## Converts numbers to #
# ## Most embeddings have preprocessed their text like this.
def clean_numbers(x):
    if bool(re.search(r'\d', x)):
        x = re.sub('[0-9]{5,}', '#####', x)
        x = re.sub('[0-9]{4}', '####', x)
        x = re.sub('[0-9]{3}', '###', x)
        x = re.sub('[0-9]{2}', '##', x)
    return x


# ## Replace contractions
def _get_contractions(contraction_dict):
    contraction_re = re.compile('(%s)' % '|'.join(contraction_dict.keys()))
    return contraction_dict, contraction_re


contractions, contractions_re = _get_contractions(contraction_dict)


def replace_contractions(text):
    def replace(match):
        return contractions[match.group(0)]

    return contractions_re.sub(replace, text)


# ## GLoVE word embedding
def load_glove_index():
    def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')[:300]

    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))
    return embeddings_index


glove_embedding_index = load_glove_index()


