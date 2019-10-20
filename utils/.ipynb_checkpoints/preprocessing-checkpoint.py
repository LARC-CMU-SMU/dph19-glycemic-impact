#from utils.path import dir_HugeFiles #, dir_json, dir_save
import numpy as np
import pandas as pd
import pickle
import json
import re
import os
from utils.save import make_dir

def load(dir_save):
    # if exist, then load
    if os.path.isfile(dir_save):
        print('exist')
        with open(dir_save, 'rb') as f:
            data = pickle.load(f)
            return data
    else:
        print('cannot find')

def preprocessing(dir_save, dirname):
    '''
    Args:
        dir_save: The path toward a pickle file that saves data after preprocessing
        dir_json: The path of a directory, should end with /. has .json files inside
    '''
    # if exist, then load
    if os.path.isfile(dir_save):
        print('file exists, should just load, not preprocessing')
    
    else:
        # step 1: just combine all json
        print('load data')
        data = {"recipes":[]}
        files = os.listdir(dirname)
        len_files = len(files)
        for i, filename in enumerate(files):
            filepath = dirname + filename
            if i<len_files:
                with open(filepath, 'r') as f:
                    one_recipe = json.load(f)
                data['recipes'].append(one_recipe)
            if i%5000 == 0:
                print("{:.2f} % finished".format(100*i/len_files))
        print("{:.2f} % finished".format(100))
        
        
        # step two: clean up the data
        print('data preprocessing')
        for k, v in enumerate(data['recipes']):
            target = 'nutritions'
            if target in v.keys():
                row = v[target]
                nutrition = parse_nutrition(row)
                for k2, v2 in nutrition.items():
                    key_to_add = '_'+k2
                    data['recipes'][k][key_to_add] = v2[0] # remove the unit

            target = 'time'
            if target in v.keys():
                row = v[target]
                time = parse_time(row)
                for k2, v2 in time.items():
                    key_to_add = '_'+k2
                    data['recipes'][k][key_to_add] = v2
            
            target = 'sections'
            if target in v.keys():
                row = v[target]
                row2 = v['categories']
                section = parse_section(row, row2)
                for k2, v2 in section.items():
                    key_to_add = '_'+k2
                    data['recipes'][k][key_to_add] = v2

            targets = ['calorie','servings','followers_count']
            for target in targets:
                if target in v.keys():
                    row = v[target]
                    key_to_add = '_'+target
                    if type(row)==float:
                        value = row
                        data['recipes'][k][key_to_add] = value
                    elif type(row)==str:
                        value = float(re.findall(r'\d+', row)[0])
                        data['recipes'][k][key_to_add] = value
                    else:
                        data['recipes'][k][key_to_add] = np.nan
                        
            target = 'name'
            key_to_add = '_'+target
            if target in v.keys():
                row = v[target]
                row = clean_line(row)
                data['recipes'][k][key_to_add] = row
                
            target ='ingredients'
            key_to_add = '_'+target
            key_to_add2 = '_'+target+'2'
            if target in v.keys():
                row = v[target]
                listofline = [clean_line(line) for line in row]
                listofline = [line for line in listofline if line!='']
                if len(' '.join(listofline)) == 0:
                    data['recipes'][k][key_to_add] = np.nan
                else:
                    data['recipes'][k][key_to_add] = listofline
            
            target ='directions'
            key_to_add = '_'+target
            if target in v.keys():
                row = v[target]
                line = ' '.join(row)
                # if empty in the beginning
                if len(line) == 0:
                    data['recipes'][k][key_to_add] = np.nan
                else:
                    line = clean_line(line)
                    # add space before punctuation
                    line = re.sub('([.,!?()])', r' \1 ', line)
                    line = re.sub('\s{2,}', ' ', line)
                    # split by sentence
                    listofline = re.split('(?<=[.?!]) +',line)
                    listofline = [line for line in listofline if line!='']
                    # if empty after preprocessing
                    if len(' '.join(listofline)) == 0:
                        data['recipes'][k][key_to_add] = np.nan 
                    else:
                        data['recipes'][k][key_to_add] = listofline
            
        # step 3: save
        print('data saving')
        df = pd.DataFrame.from_dict(data['recipes'])
        make_dir(dir_save)
        with open(dir_save, 'wb') as f:
            pickle.dump(df, f,protocol=pickle.HIGHEST_PROTOCOL)
        return df

def parse_nutrition(row):
    dict_nutrition = dict()
    if type(row) == str:   
        for str1 in row.split('=='):
            try:
                str_forw, str_back = re.split('\\: ', str1)
                str_unit = "".join(re.findall("[a-zA-Z]", str_back))
                val_float = float("".join(re.findall("(?![a-zA-Z]).", str_back)))
                dict_nutrition[str_forw.lower()] = [val_float, str_unit]
            except ValueError:
                continue
    return dict_nutrition


def parse_time(row):
    row = str(row)
    row = row.replace('d','1440')
    row = row.replace('h','60')
    row = row.replace('m','1')
    row = row.split(' ')
    ans = 0
    row_len = len(row)
    if row_len ==1:
        ans = np.nan
    try:
        for i in range(int(row_len/2)):
            ans = ans + int(row[2*i])*int(row[2*i+1])
    except ValueError:
        ans = np.nan
    dict_time = {'time': ans}
    return dict_time

def parse_section(row, row2):
    # parse row1
    row = str(row)
    row = row.split(';')
    row = filter(None, row)
    full = [ele.split('->') for ele in row]
    # clean row2
    row2 = [ele for ele in row2 if ele not in ['Home','Recipes']]
    # combine
    full.append(row2)
    full = [clean_list(ele) for ele in full]
    tags = list(set(sum(full, [])))
    dict_sec = {'categories': full, 'tags': tags}
    return dict_sec

def clean_line(line):
    '''
    Args:
        line: a string, such as food name, sentences...
    '''
    # all lowercase
    line = str(line)
    line = line.lower()
    # only reserve number and alphabets
    line = re.sub(r'[^a-z0-9+()/?!.,]', ' ', line)
    # remove extra spaces
    line = re.sub(' +',' ',line).strip()
    return line

def clean_list(row):
    '''
    Args:
        row: list of lines
    '''
    return [clean_line(ele) for ele in row]

def clean_ny(row):
    '''
    Args:
        row: list of lines
    '''
    return [ele for ele in row if type(ele)==str and len(ele)>0]