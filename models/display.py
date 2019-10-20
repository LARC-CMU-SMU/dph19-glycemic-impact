import pandas as pd
import numpy as np
from models.nested_validation import average_score
from utils.save import make_dir, save_pickle, load_pickle, auto_save_csv


def pickle2df(results, pickle_path):
    mean_score = average_score(results)
    df = pd.DataFrame.from_dict(mean_score, orient = 'index').mean(axis = 0)
    df = round(df,3)
    df['model1'] = results[0]['model1']
    df['model2'] = results[0]['model2']
    df['method'] = results[0]['method']
    df['tag'] = results[0]['tag']
    df['pickle_path'] = pickle_path
    df = df.to_frame().T
    auto_save_csv(df)
    return df

def FindSimilar(model, the_food, num = 10):
    '''
    model = word2vec trained using gensim
    the_food = any string
    num = the number of similar word presented
    '''
    print( "-"*10 + the_food + " is similar to"+"-"*10)
    try:
        res = model.wv.most_similar(the_food ,topn = num)
        for item in res:
            print(item[0]+","+str(item[1]))
    except KeyError:
        print('cannot find in dict')