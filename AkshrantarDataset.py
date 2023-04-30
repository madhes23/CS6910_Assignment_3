# %%
import torch
import pandas as pd
from sklearn.utils import shuffle 

# %%
class AksharantarDataset():
    def __init__(self, lang, start_token='<', end_token='>', pad_token=' ', unk_token='~'):
        self.start_token = start_token
        self.end_token = end_token
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.lang = lang
        self.dir_path = 'aksharantar_sampled/'

        #contructing dictionaries
        train_path = self.dir_path + self.lang + '/'+self.lang+'_train.csv'
        df = pd.read_csv(train_path, header=None)
        train_source, train_target = df[0].tolist(), df[1].tolist()
        english_chars = list(set(''.join(train_source) + start_token + end_token + pad_token + unk_token))
        target_chars = list(set(''.join(train_target) + start_token + end_token + pad_token + unk_token))

        self.en_i2l, self.en_l2i = {}, {}
        self.tr_i2l, self.tr_l2i = {}, {}

        for i, x in enumerate(english_chars):
            self.en_l2i[x] = i
            self.en_i2l[i] = x
        for i, x in enumerate(target_chars):
            self.tr_l2i[x] = i
            self.tr_i2l[i] = x

    #TODO: convert the lower_bound -> equalize_length
    def preprocess(self, strings, lower_bound = -1):
        """
        Adds start and end token and adds padding

        lower_bound : ensures that each strings is padded atleast upto lower bound length
        """
        res = []
        max_len = len(max(strings, key=len)) + 2  #2 is added, because we added start and end token to each word
        if(lower_bound != -1):
            max_len = max(max_len, lower_bound)

        for item in strings:
            temp = self.start_token + item + self.end_token
            temp = temp.ljust(max_len, self.pad_token)
            res.append(temp)
        return res


    def string_to_tensor(self, strings, l2i_dict):
        """
        replaces the chareceters of the sting with corrospong ix (by refering l2i_dict) and returns as int tensor
        """
        res = torch.zeros(len(strings), len(strings[0]))
        
        for i in range(len(strings)):
            for j in range(len(strings[i])):
                if strings[i][j] not in l2i_dict :
                    res[i][j] = l2i_dict[self.unk_token]
                else:
                    res[i][j] = l2i_dict[strings[i][j]] 
        return res.type(torch.LongTensor)


    def load_data(self, set, batch_size, is_shuffle = True, num_batches = -1, padding_lower_bound = -1):
        """
        Parameter:
        set: a string to define which set of data to load, set can be any one of the following ["train", "test", "valid"]
        batch_size: batch size in which the data be loaded
        num_batches: returns this many batches of size batch_size
                     if -1 : returns the entire data in batches 
                     else only returns that many number of batches
        Returns:
        
        """
        path = self.dir_path + self.lang + '/'+self.lang+'_' + set +'.csv'
        df = pd.read_csv(path, header=None)

        source, target = df[0].tolist(), df[1].tolist()
        if(is_shuffle == True):
            source, target = shuffle(source, target)

        res = []
        batch_count = 0
        for i in range(0, len(source), batch_size):
            source_batch, target_batch = source[i:i+batch_size], target[i:i+batch_size]
            source_batch, target_batch = self.preprocess(source_batch, padding_lower_bound), self.preprocess(target_batch, padding_lower_bound)
            source_batch, target_batch = self.string_to_tensor(source_batch, self.en_l2i), self.string_to_tensor(target_batch, self.tr_l2i)
            source_batch, target_batch = source_batch.transpose(0,1), target_batch.transpose(0,1)
            res.append((source_batch, target_batch))
            batch_count += 1
            if(num_batches != -1 and batch_count == num_batches):
                break
        return res


    def tensor_to_string(self, output, string_type = "target"):
        """
        output shape: target_seq_length * N
        string_type : can take values of ["source", "target"]
                      if source: the ouput is encoded form of english
                      if target: the output is encoded form of target language
        """
        if(string_type == "target"):
            i2l_dict = self.tr_i2l
        elif(string_type == "source"):
            i2l_dict = self.en_i2l
        
        res = []
        for j in range(output.shape[1]):
            temp = ""
            for i in range(output.shape[0]):
                temp += i2l_dict[output[i,j].item()]
            
            res.append(temp)
        return res