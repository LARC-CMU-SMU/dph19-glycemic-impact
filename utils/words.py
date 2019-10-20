import re
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from collections import Counter
from scipy.sparse import csr_matrix
    
def make_corpus_0(dic, keys, ids =[], application=''):
    '''
    Args:
        dic from df, with all recipes info
        keys: list, for example, _name, _ingredients
        ids: list/Series of ids we want
    '''
    # if non specific, take all ids in dic
    if ids ==[]:
        ids = dic.keys()
    corpus_llist ,corpus_list, corpus = [], [] ,[]
    if not application:
        for i, v in dic.items():
            if i in ids:
                onerecipe = sum([clean_wordcount(v[key]) for key in keys],[])
                corpus_llist.append(onerecipe)
                onerecipe = sum(onerecipe,[])
                corpus_list.append(onerecipe)
                corpus.append(' '.join(onerecipe))
        return corpus_llist, corpus_list, corpus
    else:
        recipes = []
        for i, v in dic.items():
            recipe=[]
            for k in keys:
                recipe += v[k]
            if application == 'sent': ## corpus_llist
                recipes.append(recipe)
            elif application == 'word':# corpus_list
                recipes.append(sum(recipe,[]))
        return recipes

# older version, also useful
def make_corpus(dic, keys, ids =[]):
    '''
    Args:
        dic from df, with all recipes info
        keys: list, for example, _name, _ingredients
        ids: list/Series of ids we want
    '''
    # if non specific, take all ids in dic
    if ids ==[]:
        ids = dic.keys()
    corpus_list, corpus = [], []
    for i, v in dic.items():
        if i in ids:
            onerecipe = sum([clean_wordcount(v[key]) for key in keys],[])
            onerecipe = sum(onerecipe, [])
            corpus_list.append(onerecipe)
            corpus.append(' '.join(onerecipe))
    return corpus_list, corpus

def get_wordcount(corpus):
    if type(corpus)==list:
        return get_wordcount_list(corpus)
    vectorizer = CountVectorizer(min_df = 5, token_pattern= r"(?u)\b\w\w+\b|!|\?|\"|\'", stop_words = 'english')
    X_word_count = vectorizer.fit_transform(corpus).toarray()
    fn_word_count = vectorizer.get_feature_names()
    return X_word_count, fn_word_count

def get_wordcount_list(corpus_list, higherthan = 5):
    fn = []
    for recipe in corpus_list:
        for w in recipe:
            if not w in fn:
                fn.append(w)
    fn_dict = dict(zip(fn,range(0,len(fn))))
    row, col, data = [], [], []
    for i, recipe in enumerate(corpus_list):
        for w, times in dict(Counter(recipe)).items():
            row.append(i)
            col.append(fn_dict[w])
            data.append(times)
    X = csr_matrix((data,(row,col)), shape=(len(corpus_list),len(fn)))
    idx = np.where(X.sum(axis=0)>higherthan)[1] # higherthan 5
    X = X[:,idx].toarray()
    fn = np.array(fn)[idx].tolist()
    
    # drop punctuations and UNK
    # ? and ! is not found in fn
    words = [word for word in ['UNK',"(",")",",",".","!",'?'] if word in fn ]
    words_idx = np.array([fn.index(word) for word in words])
    X = np.delete(X, words_idx, 1)
    fn = np.delete(np.array(fn), words_idx, 0).tolist()
    
    return X, fn

def clean_str(row):
    '''
    Args:
        :str a sentence or food name
    Return:
        :list
    '''
    # add space before punctuation
    if type(row) == str:
        line = re.sub('([.,!?()])', r' \1 ',row)
        line = re.sub('\s{2,}', ' ', line)
        line = line.split(' ')
        line = [ele for ele in line if ele!='']
        return [line]
def clean_list(listofstr):
    '''
    Args:
        :listofstr a sentence or food name
    Return:
        :list
    '''
    listoflist = [clean_wordcount (ele) for ele in listofstr]
    listoflist = [ele for ele in listoflist if ele != [''] and ele !=[[]]]
    return sum(listoflist, [])
    
def clean_wordcount(row):
   
    if type(row) == str:
        return clean_str(row)
    elif type(row) == list:
        return clean_list(row)
    else:
        print('Error in wordcount module')
        
def parse_section(row, row2):
    # parse row1
    row = str(row)
    row = row.split(';')
    row = filter(None, row)
    full = [ele.split('->') for ele in row]
    # clean row2
    row2 = [ele for ele in row2 if ele not in ['Home','Recipes']]
    # combine
    full += row2
    tags = list(set(sum(full, [])))
    dict_sec = {'categories': full, 'tags': tags}
    return dict_sec

def nigram_transformer(row, ngram_list):
    '''
    params: row: e.g. 'grain rice and some milk'
    params: ngram_list = ['hot dogs', 'wild rice']
    output: grain_rice and some milk
    '''
    for strr in ngram_list:
        if strr in row:
            row = re.sub(strr, underscore(strr), row)
    return row

def underscore(strr):
    if ' ' in strr:
        return re.sub(' ','_', strr)
    else:
        return strr

def replace_UNK(recipe, knowns, space):
    if type(recipe[0]) == list:
        return [replace_UNK(sent, knowns, space) for sent in recipe]
    else:
        ans = [word if word in knowns else 'UNK' for word in recipe]
        if space:
            return ' '.join(ans)
        else:
            return ans 
