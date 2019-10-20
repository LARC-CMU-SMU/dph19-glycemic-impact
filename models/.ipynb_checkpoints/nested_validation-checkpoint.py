from sklearn.metrics import precision_score, f1_score, recall_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from utils.words import get_wordcount_list
from utils.save import print_time
import copy
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from models.features import get_wordvec
from pebble import concurrent
from concurrent.futures import TimeoutError
from sklearn import preprocessing

allmetrics = ['%s_%s'%(name, metric) for name in ['test', 'train'] for metric in ['f1','precision','recall','accuracy']]

def inputs_generater(corp, model1, method = ['wc','nb','l2','scale']):
    '''generates X, y for the clf model
    Args:
        corp: e.g. object makedata
        model1: e.g. object gensim_wrap
    Returns:
        data: X_train, X_test, y_train, y_test, fn
    ''' 
    if not hasattr(corp, 'test_index'):
        train_index, test_index, y_train, y_test = train_test_split(corp.ls, 
                                                                    corp.y, 
                                                                    test_size = 0.25, 
                                                                    stratify = corp.y,
                                                                    random_state =2018)
    else: 
        train_index, test_index = corp.train_index, corp.test_index
        del corp.train_index, corp.test_index
    # preparing corpus
    train, test = corp.replace(train_index), corp.replace(test_index)

    
    corpus_list, fn = [], []
    if 'wc' in method:
        X, feature_name = get_wordcount_list(corp.corpus_list,  higherthan = 0)  
        if 'nb' in method:
            y_highGI, y_lowGI = train.y.astype(bool), ~train.y.astype(bool)
            alpha = 1
            p = X[train_index][y_lowGI].sum(axis = 0) + alpha
            sum_p = p.sum() + alpha
            q = X[train_index][y_highGI].sum(axis = 0) + alpha
            sum_q = q.sum() + alpha
            count_ratio = np.log(p/sum_p) - np.log(q/sum_q)
            X = np.multiply(X, count_ratio)    
        elif 'scale' in method:
            X = preprocessing.scale(X, axis = 0) #norm to feature  
        corpus_list.append(X)
        fn += feature_name
        
    if 'wv' in method and model1:
        X, feature_name = get_wordvec(corp.corpus_list, model1)
        if 'scale' in method:
            X =  preprocessing.scale(X, axis = 0) #norm to feature
        corpus_list.append(X)
        fn+=feature_name
        
    if 'nu' in method:
        X, feature_name = corp.nu, corp.nu_fn
        if 'scale' in method:
            X =  preprocessing.scale(X, axis = 0) #norm to feature
        corpus_list.append(X)
        fn+=feature_name
        
    if 'skip' in method:
        X, feature_name = corp.skip, corp.skip_fn
        if 'scale' in method:
            X =  preprocessing.scale(X, axis = 0) #norm to feature
        corpus_list.append(X)
        fn+=feature_name
        
    corpus_list = np.hstack(corpus_list)
    X_train, X_test = corpus_list[train_index], corpus_list[test_index]
        
    data = X_train, X_test, train.y, test.y, fn
    return data


class clf_running:
    def __init__(self, data, model1, model2, method, threshold = 0.5, print_ = False):
        '''generates X, y for the clf model
        Args:
            data: X_train, X_test, y_train, y_test, fn
            model2: sklearn model with predefined parameters
            threshold: probability threshold, float
        Returns:
            result: dict
        '''
        X_train, X_test, y_train, y_test, fn = data
        if model2.__class__ == 'lightgbm.sklearn.LGBMClassifier':
            trained_model = model2.fit(X_train, y_train,
                                       eval_set = [(np.vstack(X_test), y_test)],
                                       eval_metric ='logloss',
                                       early_stopping_rounds = 100)
        else:
            trained_model = model2.fit(X_train, y_train)
            
        result ={'model2': model2}
        if hasattr(trained_model, 'coef_'):
            #"warning: feature number inconsistent"
            assert len(fn) == len(trained_model.coef_[0])
            result['f_importance'] = importance(trained_model, fn)
        self.result = result
        
        if hasattr(trained_model, 'predict_proba'):
            self.y_trains = trained_model.predict_proba(X_train)[:,1], y_train, 'train'
            self.y_tests = trained_model.predict_proba(X_test)[:,1], y_test, 'test'
        elif hasattr(trained_model, 'decision_function'):
            self.y_trains = trained_model.decision_function(X_train), y_train, 'train'
            self.y_tests = trained_model.decision_function(X_test), y_test, 'test'
        else:
             print('cannot tune probability threshold')
        
        self.metrics = {'recall': recall_score, 'precision': precision_score, 'f1': f1_score, 'accuracy': accuracy_score}
        self.result = self.tune(threshold)
    
    def tune(self, threshold = None):
        result = self.result
        for y_pred, y_true, name in [self.y_trains, self.y_tests]:
            if threshold:
                y_pred = [1 if ele>threshold else 0 for ele in y_pred]
            new = {'%s_%s'%(name, metric): function(y_true,y_pred) for metric, function in self.metrics.items()}
            result.update(new)
        return result
    
class clf_y_pred(clf_running):
    def tune(self, threshold = None):
        result = self.result
        for y_pred, y_true, name in [self.y_trains, self.y_tests]:
            if threshold:
                y_pred = [1 if ele>threshold else 0 for ele in y_pred]
            new = {'%s_%s'%(name, metric): function(y_true,y_pred) for metric, function in self.metrics.items()}
            new['y_pred'] = y_pred
            result.update(new)
        return result
    
    
    
def clf_running_search(data, model1, model2, method,
                       num_grid = 20, print_ = False):
    '''generates X, y for the clf model
    Args:
        data: X_train, X_test, y_train, y_test, fn
        model2: sklearn model with predefined parameters
    Returns:
        best_result['threshold']
    '''
    if hasattr(model2, 'predict_proba'):
        if print_:
            print('grid searching: predict_proba; can ignore the UndefinedMetricWarning')
        #thresholds = [i/num_grid for i in range(1, num_grid, 1)]
        thresholds = [0.01*i for i in list(range(45, 55, 1))]
        
    elif hasattr(model2, 'decision_function'):
        if print_:
            print('grid searching: decision_function; can ignore the UndefinedMetricWarning')
        positive_score = model.decision_function(X_truth['train'])
        lower, upper = min(positive_score), max(positive_score)   
        thresholds = [round(lower + i/num_grid*(upper-lower),2) for i in range(1, num_grid, 1)]+[0]
    else:
        print('cannot tune probability threshold')
    best_result, best_f1 = {'threshold':thresholds[0]}, 0.0
    prediction = clf_running(data, model1, model2, method)
    for t in thresholds:
        result = prediction.tune(t)
        if best_f1 < result['test_f1']:
            best_result.update({'threshold':t})
            best_result.update(result)
            best_f1 = result['test_f1']
    return best_result['threshold']
    
def objective(params, fixed_threshold = False):
    print(params['p2'])
    model1 = params['model1']
    method = params['method']
    model2 = params['model2'](**params['p2'])
    
    timed_object = cv_evaluate(params['corp'], model1, model2, method,
                               folds = 5, fixed_threshold = fixed_threshold)
    try:
        results = timed_object.result()
        print([result['test_f1'] for result in results])
        average_f1 = np.array([result['test_f1'] for result in results]).mean()
    except TimeoutError:
        average_f1 = 0.0
        print('Time exceeds limit')
    # may not handle other error...
    #except Exception as error:
    #    print("Function raised %s" % error)
    #    print(error.traceback)
    print(average_f1)
    return (-1)*average_f1

def objective_RFECV(params, fixed_threshold = False):
    print(params['p2'])
    model1 = params['model1']
    method = params['method']
    model2 = params['model2'](**params['p2'])
    model2_fselection = RFECV(model2, step = 1, cv = 5, scoring = 'f1')
    #model1, method = None, None
    timed_object = cv_evaluate(params['corp'], model1, model2_fselection, method,
                               folds = 1, fixed_threshold = fixed_threshold)
    try:
        results = timed_object.result()
        print([result['test_f1'] for result in results])
        average_f1 = np.array([result['test_f1'] for result in results]).mean()
    except TimeoutError:
        average_f1 = 0.0
        print('Time exceeds limit')
    print(average_f1)
    return (-1)*average_f1


@concurrent.process(timeout=60*60*5) # used to be 1.5hr
def cv_evaluate(corp, model1, model2, method, folds = 5, fixed_threshold = False):
    print_time()
    results = []
    ss = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 2019)
    nfold = 0
    for train_index, test_index in ss.split(corp.ls, corp.y):
        data = inputs_generater(corp.replace(train_index), model1, method)
        '''
        3. run and find the best prob threshold
        '''
        if fixed_threshold:
            threshold = 0.5
        else:
            threshold = clf_running_search(data, model1, model2, method)
        '''
        4. run again and evaluate on the real test set
        ''' 
        data = inputs_generater(corp.add_train_test(train_index, test_index), model1, method)
        result = clf_running(data, model1, model2, method, threshold).result
        results.append(result)
        nfold +=1
        if nfold>=folds:
            break
    return results
    
def average_score(results, tt_metrics = allmetrics):
    '''
    Args:
        results: output of outer_CV
        tt_metrics = ['test']
    '''
    mean_score = dict()
    metrics = [ele for ele in tt_metrics if ele in results[0].keys()]
    for i, fold in enumerate(results):
        for k, v in fold.items():
            mean_score[i] = dict(zip(metrics, [fold[k] for k in metrics]))
    return mean_score

def importance(model, col):
    coefficients = pd.concat([pd.DataFrame(col),pd.DataFrame(np.transpose(model.coef_))], axis = 1)
    coefficients.columns =['feature','importance']
    coefficients = coefficients.sort_values(by=['importance'])
    # return coefficients.head(10)
    return coefficients

def f_selection(corp, model1, estimator, method):
    corpus_list, fn = [], []
    # exposed the test set whilefe feature selection by applying the NB weight
    if 'wc' in method:
        X, feature_name = get_wordcount_list(corp.corpus_list,  higherthan = 0)  
        if 'nb' in method:
            y_highGI, y_lowGI = corp.y.astype(bool), ~corp.y.astype(bool)
            alpha = 1
            p = X[y_lowGI].sum(axis = 0) + alpha
            sum_p = p.sum() + alpha
            q = X[y_highGI].sum(axis = 0) + alpha
            sum_q = q.sum() + alpha
            count_ratio = np.log(p/sum_p) - np.log(q/sum_q)
            X = np.multiply(X, count_ratio)    
        if 'l2' in method:
            X = preprocessing.normalize(X, axis = 1)
        if 'scale' in method:
            X = preprocessing.scale(X, axis = 0) #norm to feature  
        corpus_list.append(X)
        fn += feature_name
        
    if 'wv' in method and model1:
        X, feature_name = get_wordvec(corp.corpus_list, model1)
        if 'scale' in method:
            X =  preprocessing.scale(X, axis = 0) #norm to feature
        corpus_list.append(X)
        fn+=feature_name
        
    if 'nu' in method:
        X, feature_name = corp.nu, corp.nu_fn
        if 'scale' in method:
            X =  preprocessing.scale(X, axis = 0) #norm to feature
        corpus_list.append(X)
        fn+=feature_name
        
    if 'skip' in method:
        X, feature_name = corp.skip, corp.skip_fn
        if 'scale' in method:
            X =  preprocessing.scale(X, axis = 0) #norm to feature
        corpus_list.append(X)
        fn+=feature_name
        
    corpus_list = np.hstack(corpus_list)    
    X, y = corpus_list, corp.y
    ss = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 2019)
    selector = RFECV(estimator, step = 1, cv = ss, scoring = 'f1')
    selector = selector.fit(X, y)
    selected = np.nonzero(selector.support_)[0]
    return selected