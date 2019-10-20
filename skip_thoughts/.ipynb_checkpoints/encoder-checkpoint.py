from model import UniSkip, Encoder
from data_loader import DataLoader
from vocab import load_dictionary
from config import *
from torch import nn
import numpy as np

from torch.autograd import Variable
import torch

class UsableEncoder:
    
    def __init__(self, loc, WORD_DICT):
        # BEST_MODEL = "../../dir_HugeFiles/prev_model/skip-best-loss10.237"
        print("Preparing the DataLoader. Loading the word dictionary")
        # WORD_DICT = '../dir_HugeFiles/instructions/skip_inst/skip_instruction.csv.pkl'
        self.d = DataLoader(sentences=[''], word_dict=load_dictionary(WORD_DICT))
        self.encoder = None
        
        print("Loading encoder from the saved model at {}".format(loc))
        model = UniSkip()
        model.load_state_dict(torch.load(loc, map_location=lambda storage, loc: storage))
        self.encoder = model.encoder
        if USE_CUDA:
            self.encoder.cuda(CUDA_DEVICE)
            print('using cuda')
    
    def encode(self, text):
        def chunks(l, n):
            """Yield successive n-sized chunks from l."""
            for i in range(0, len(l), n):
                yield l[i:i + n]
        
        ret = []
        
        for chunk in chunks(text, 100):
            #print("encoding chunk of size {}".format(len(chunk)))
            indices = [self.d.convert_sentence_to_indices(sentence) for sentence in chunk]
            indices = torch.stack(indices)
            indices, _ = self.encoder(indices)
            indices = indices.view(-1, self.encoder.thought_size)
            indices = indices.data.cpu().numpy()
            
            ret.extend(indices)
        ret = np.array(ret)
        
        return ret
    
'''
from utils.path import dir_HugeFiles
path = dir_HugeFiles+'sk_corpus/full_0420_85'
encoder = UsableEncoder(path+'/skip-best-loss17.292-epoch694500',WORD_DICT = path+'.csv.pkl')
'''