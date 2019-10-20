import os
import copy
import pandas as pd
import numpy as np
import multiprocessing
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfTransformer

from utils.path import dir_HugeFiles
from utils.words import make_corpus, get_wordcount_list, make_corpus_0
from utils.save import make_dir, save_pickle, load_pickle, auto_save_csv, print_time

import gensim.downloader as api
info = api.info()  # show info about available models/datasets
from gensim.models import Word2Vec, FastText, KeyedVectors
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from utils.nDoc2Vec import nDoc2Vec # modify gensim doc2vec
from glove import Corpus, Glove # implemented by stanford nlp

class doc2vec_wrap():
    def __init__(self, params = {}):
        params_default = {'main':
                          {'vector_size': 300, 'window': 5, 'min_count': 5, 
                          'learning_rate':0.025,
                          'epochs': 40, 
                          'workers': max(1, multiprocessing.cpu_count() - 10), 
                          'sample': 1E-5,
                          'dm':0,
                          'dm_concat':0},
                         'pretrained': '../data/vocab.bin'}
        params_default['main'].update(params['main'])
        #params_default['pretrained']=params['pretrained']
        self.params = params_default
        
    def fit(self, corpus_list):
        params = self.params
        id_corpus = [TaggedDocument(doc, [i]) for i, doc in enumerate(corpus_list)]
        gensim_model = nDoc2Vec(**params['main'])
        gensim_model.build_vocab(id_corpus)
        gensim_model.intersect_word2vec_format(params['pretrained'], lockf= 1.0, binary = True)#trainable
        gensim_model.train(id_corpus, total_examples=gensim_model.corpus_count, epochs=gensim_model.epochs)
        self.gensim_model = gensim_model
        self.vector_size = params['main']['vector_size']
    def getdocvector(self, doc):
        return self.gensim_model.infer_vector(doc)
    
class glove_wrap:
    def __init__(self, params ={}):
        params_default = {'size': 300, 'window': 40, 'min_count': 5, 
                          'learning_rate':0.05,
                          'epochs': 40, 
                          'workers': max(1, multiprocessing.cpu_count() - 10), 
                          'sample': 1E-3}
        params_default.update(params)
        self.params = params_default
    def getvector(self, word):
        return self.glove_model.word_vectors[self.glove_model.dictionary[word]]
    def fit(self, corpus_list):
        params = self.params
        corpus = Corpus()
        corpus.fit(corpus_list, window= params['window'])
        glove_model = Glove(no_components=params['size'], learning_rate=params['learning_rate'])
        glove_model.fit(corpus.matrix, epochs=params['epochs'], 
                        no_threads= max(1, multiprocessing.cpu_count()- 10), 
                        verbose=False)
        glove_model.add_dictionary(corpus.dictionary)
        self.glove_model = glove_model
        self.vocab = glove_model.dictionary
        self.vector_size = params['size']

class gensim_wrap:
    def __init__(self, knowns, model, params = {}):
        params_default = {'size': 300, 'window': 40, 'min_count': 5,
                      'workers': max(1, multiprocessing.cpu_count() - 10), 'sample': 1E-3}
        params_default.update(params)            
        self.params = params_default
        self.model = model
        self.knowns = knowns
    def getvector(self, word):
        return self.gensim_model.wv.__getitem__(word)
    def fit(self, corpus_list):
        params = self.params
        gensim_model = self.model(corpus_list, **params)
        self.gensim_model = gensim_model
        self.vocab = gensim_model.wv.vocab
        self.vector_size = params['size']
    
class pretrained_wrap():
    def __init__(self, knowns, modelname):   
        gensim_model = api.load(modelname)
        self.vector_size = gensim_model.vector_size
        len_vocab = len(gensim_model.wv.vocab)
        new_dict = dict()
        for i, word in enumerate(knowns):
            if word in gensim_model.wv.vocab:
                new_dict[word] = gensim_model.wv.__getitem__(word)
            else:
                new_dict[word] = np.zeros(self.vector_size)
        self.new_dict = new_dict
        self.vocab = dict(zip(new_dict.keys(), range(len(knowns))))
    def getvector(self, word):
        return self.new_dict[word]
    def fit(self, corpus_list):
        # will change nothing
        return self
        
class salvador_wrap(gensim_wrap):
    '''
    default: load the pretrained model from (Salvador, 2017) on CVPR
    other usuage: can used to load any other pretrained word embedding model in .bin file 
    '''
    def __init__(self, path = '../data/vocab.bin'):    
        gensim_model = KeyedVectors.load_word2vec_format(path, binary = True)
        self.gensim_model = gensim_model
        self.vocab = gensim_model.wv.vocab
        self.vector_size = gensim_model.vector_size


def get_wordvec(corpus_list, model, pooling = 'average'):
    '''
    new new version, associated with gensim_wrap and glove_wrap
    transform recipes to vector
    params: corpus_list: list(list(string, string))
    params: model: trained word2vec model from gensim
    return: X_wv: np.2darrray
    return: fn_wv: list of feature names
    '''
    if hasattr(model, 'getvector'):
        vec_num = model.vector_size
        known_words = list(model.vocab.keys())#change
        X_wv = np.zeros((len(corpus_list),vec_num))
        for i, recipe in enumerate(corpus_list):
            # only keep known words
            recipe_known = [word for word in recipe if word in known_words]
            len_recipe = len(recipe_known)
            vector = np.zeros((len_recipe,vec_num))
            # loop every words in recipe
            for j in range(len_recipe):
                vector[j] = model.getvector(recipe_known[j])#change

            # average the word vectors, eliminated the rare words
            if pooling == 'average':
                X_wv[i] = vector.mean(axis=0)
            if pooling == 'max':
                X_wv[i] = vector.max(axis=0)

            fn_wv = ['vec_'+str(ele) for ele in range(vec_num)]
        return X_wv, fn_wv
    
    elif hasattr(model, 'getdocvector'):
        vec_num = model.vector_size
        X_wv = np.zeros((len(corpus_list),vec_num))
        for i in range(X_wv.shape[0]):
            X_wv[i] = model.getdocvector(corpus_list[i])
        fn_wv = ['vec_'+str(ele) for ele in range(vec_num)]
        return X_wv, fn_wv

    
        
class fixed_makedata():
    def __init__(self, dic, ls, tag='GI', tags = [], add_skip_thoughts = False):
        keys =  ['name', 'ingredients', 'directions']
        keys = [k+'_UNK2_none' for k in keys]
        if tags:
            keys = tags
        self.corpus_list = make_corpus_0(dic, keys , ids= ls)[1]
        self.knowns = self.known(self.corpus_list)
        self.y = np.array([v[tag] for i, v in dic.items() if i in ls]) # store to list
        self.ls = np.array(range(len(self.y))) # reindex
        self.tag = tag
        self.method = ''
        fields = ['cholesterol','calcium','sodium','potassium','vitamin c',
                      'iron','thiamin','niacin','vitamin b6','magnesium','folate','vitamin a',
                      'protein','sugars','total carbohydrates','total fat', 'dietary fiber','saturated fat']
        dw = ['dry_weight']
        nutritions = [n+'/'+dw[0] for n in fields] + dw + ['calorie/dry_weight']
        df = pd.DataFrame.from_dict(dic, orient = 'index')
        self.nu = df.iloc[sorted(ls),:][nutritions].values # should sort it
        self.nu_fn = nutritions
        if add_skip_thoughts:
            X = load_pickle(add_skip_thoughts)
            self.skip = X
            self.skip_fn = ['sk_'+str(ele) for ele in range(X.shape[1])]

    def change_y(self, dic, ls, tag = 'GI'):
        self.y = np.array([v[tag] for i, v in dic.items() if i in ls]) # store to list
        self.tag = tag
        
    def replace(self, selected_index):
        new = copy.deepcopy(self)
        new.corpus_list = np.array([new.corpus_list[i] for i in selected_index])
        new.nu = np.array([new.nu[i] for i in selected_index])
        new.y = new.y[selected_index]
        new.ls = np.array(range(len(new.y)))
        if hasattr(new, 'skip'):
            new.skip = np.array([new.skip[i] for i in selected_index])
        return new
    
    def known(self, corpus_list):
        recipes = []
        for recipe in corpus_list:
            recipes+=recipe
        knowns = list(set(recipes))
        return knowns
    
    def add_train_test(self, train_index, test_index):
        self.train_index = train_index
        self.test_index = test_index
        return self