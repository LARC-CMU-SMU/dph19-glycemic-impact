import numpy as np
import copy

def fmin2(fn, space, max_evals = 5):
    np.random.seed(5)
    rd = np.random.randint(30, size=1000)
    historical = []
    j, num, best_loss = 0, 0, 0
    p2 = space['p2']
    while num < max_evals:
        new_p2 = {}
        for k, v in p2.items():
            if type(v) == list:
                rand_idx = int(rd[j]%len(v))
                new_p2[k] = v[rand_idx]
                j+=1
            else:
                new_p2[k] = v
            if j == 900:
                j = 0
        if new_p2 not in historical:
            new_space = space
            new_space.update({'p2':new_p2})
            historical.append(new_p2)
            neg_f1 = fn(new_space)
            if neg_f1 < best_loss:
                best = copy.deepcopy(new_space)
            best_loss = min(neg_f1,best_loss)
            print(neg_f1, best_loss)
            num+=1
    return best

p2_lgbm = {'class_weight':'balanced',
      'boosting':'gbrt',
      'num_leaves': [32, 64, 128, 256, 512],
      'max_depth':[4,8,16],
      'learning_rate':[0.1,0.15],
      'gamma':[0.45, 0.55, 0.65],
      'n_estimators':2000,
      'lambda_l2':[0.1,1,3,10,50],
      'feature_fraction':[0.5, 0.75],
      'bagging_fraction':[0.5,0.75, 0.9],
      'bagging_freq':[5,10,20],
      'subsample':[1,0.9,0.7],
     } 

p2_lr = {'class_weight':'balanced',
      'solver': 'liblinear', #hp.choice('solver', ['liblinear','saga']),
      'penalty':'l2', #hp.choice('penalty',['l1','l2']),
      'C': [0.01, 0.05, 0.1, 0.5, 1.0, 3.0, 10, 100, 500, 1000]
    }

def fmin2_model1(fn, space, max_evals = 5, field = 'model1'):
    '''
    can only random search one field at a time
    same as fmin2 if set field to p2 
    '''
    historical = []
    j, num, best_loss = 0, 0, 0
    model1s = space[field]
    while num < max_evals:
        new_space = []
        new_space = space
        new_space.update({'model1': model1s[num]})
        print(new_space)
        
        neg_f1 = fn(new_space)
        if neg_f1 < best_loss:
            best = copy.deepcopy(new_space)
            best_num = copy.deepcopy(num)
        best_loss = min(neg_f1,best_loss)
        # best_num+1 represents the id of trained model, starts from 1 to 8
        print(neg_f1, best_loss, best_num+1)
        
        num+=1
    return best