# %%
import torch
import pandas as pd

# %%
class AksharantarDataset():
    def __init__(self, lang, start_token='<', end_token='>', pad_token=' ', unk_token='~'):
        self.start_token = start_token
        self.end_token = end_token
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.lang = lang

        #contructing dictionaries
        train_path = 'aksharantar_sampled/' + self.lang + '/'+self.lang+'_train.csv'
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

    
    def preprocess(self, strings):
        """Adds start and end token and adds padding"""
        res = []
        max_len = len(max(strings, key=len))

        for item in strings:
            temp = self.start_token + item + self.end_token
            temp = temp.ljust(max_len+2, self.pad_token) #2 is added, because we added start and end token to each word
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


    def load_data(self, set, batch_size):
        """
        Parameter:
        set: a string to define which set of data to load, set can be any one of the following ["train", "test", "valid"]
        batch_size: batch size in which the data be loaded

        Returns:
        
        """
        path = 'aksharantar_sampled/' + self.lang + '/'+self.lang+'_' + set +'.csv'
        df = pd.read_csv(path, header=None)

        source, target = df[0].tolist(), df[1].tolist()

        res = []
        for i in range(0, len(source), batch_size):
            source_batch, target_batch = source[i:i+batch_size], target[i:i+batch_size]
            source_batch, target_batch = self.preprocess(source_batch), self.preprocess(target_batch)
            source_batch, target_batch = self.string_to_tensor(source_batch, self.en_l2i), self.string_to_tensor(target_batch, self.tr_l2i)
            source_batch, target_batch = source_batch.transpose(0,1), target_batch.transpose(0,1)
            res.append((source_batch, target_batch))
        return res



# %%
data = AksharantarDataset("tam")
print(len(data.tr_i2l))