import multiprocessing
from gensim.models import Word2Vec
import numpy as np

def train_wordvec(corpus_list, dim):    
    params = {'size': dim, 'window': 40, 'min_count': 5, 
              'workers': max(1, multiprocessing.cpu_count() - 10), 'sample': 1E-3,}
    model_wv = Word2Vec(corpus_list, **params)
    return model_wv
    
def get_X_wv(corpus_list, model):
    '''
    transform recipes to vector
    params: corpus_list: list(list(string, string))
    params: model: trained word2vec model from gensim
    return: X_wv: np.2darrray
    return: fn_wv: list of feature names
    '''
    vec_num = model.vector_size
    X_wv = np.zeros((len(corpus_list),vec_num))
    for i, recipe in enumerate(corpus_list):
        len_recipe = len(recipe)
        vector = np.zeros((len_recipe,vec_num))
        count = 0
        for j in range(len_recipe):
            try:
                vector[j] = model.wv.__getitem__(recipe[j])
                count += 1
            except KeyError:
                pass
        # average the word vectors, eliminated the rare words
        X_wv[i] = vector.mean(axis=0)*len_recipe/count
        fn_wv = ['vec_'+str(ele) for ele in range(vec_num)]
    return X_wv, fn_wv

def get_wordvec(corpus_list, model, pooling = 'average'):
    '''
    new version
    transform recipes to vector
    params: corpus_list: list(list(string, string))
    params: model: trained word2vec model from gensim
    return: X_wv: np.2darrray
    return: fn_wv: list of feature names
    '''
    vec_num = model.vector_size
    known_words = list(model.wv.vocab.keys())
    X_wv = np.zeros((len(corpus_list),vec_num))
    for i, recipe in enumerate(corpus_list):
        # only keep known words
        recipe_known = [word for word in recipe if word in known_words]
        len_recipe = len(recipe_known)
        vector = np.zeros((len_recipe,vec_num))
        # loop every words in recipe
        for j in range(len_recipe):
            vector[j] = model.wv.__getitem__(recipe_known[j])

        # average the word vectors, eliminated the rare words
        if pooling == 'average':
            X_wv[i] = vector.mean(axis=0)
        if pooling == 'max':
            X_wv[i] = vector.max(axis=0)
            
        fn_wv = ['vec_'+str(ele) for ele in range(vec_num)]
    return X_wv, fn_wv


def padding(X, max_length, pad = 0):
    if type(X[0]) == list:
        listoflist = X
        return [padding(listt, max_length) for listt in listoflist]
    else:
        listt = X
        diff_len =  max_length - len(listt)
        if diff_len<0:
            return listt[:max_length]
        elif diff_len == 0:
            return listt
        else:
            return listt+[pad]*diff_len