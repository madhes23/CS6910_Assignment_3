# %%
import torch
import pandas as pd
from sklearn.utils import shuffle 

# %%
class AksharantarDataset():
    """
    This Class handles the data related operations for Aksharantar Dataset
    Init Parameters:
    ---------------
    lang : str (one of the languages in AkshrantarDataset)
    start_token : starting token used in pre-processing the data
    end_token : ending token used in pre-processing the data
    unk_token : token used for unknown values in the data
    """
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
        self.tar_i2l, self.tar_l2i = {}, {}

        for i, x in enumerate(english_chars):
            self.en_l2i[x] = i
            self.en_i2l[i] = x
        for i, x in enumerate(target_chars):
            self.tar_l2i[x] = i
            self.tar_i2l[i] = x


    def preprocess(self, strings, upper_bound = -1):
        """
        Parameters:
        ----------
        strings: a list of strings to be preprocessed
        uppper_bound: If set a value, all the strings in the list will have a constant lenght of this value

        Returns:
        -------
        res : a list of strings after tokenization (and slicing in case of upper_bound is applied)
        """
        res = []
        max_len = len(max(strings, key=len)) + 2 #2 is added, because we added start and end token to each word

        if(upper_bound == -1):
            for item in strings:
                temp = self.start_token + item + self.end_token
                temp = temp.ljust(max_len, self.pad_token)
                res.append(temp)
            return res
        else:
            for item in strings:
                temp = self.start_token + item + self.end_token
                if(len(temp) < upper_bound):
                    temp = temp.ljust(upper_bound, self.pad_token)
                elif(len(temp) > upper_bound):
                    print(temp)
                    to_slice = len(temp) - upper_bound + 1
                    print(to_slice)
                    temp = temp[:-to_slice]
                    temp += self.end_token
                res.append(temp)
            return res


    def string_to_tensor(self, strings, l2i_dict):
        """
        Replaces the chareceters of the sting with corrospong ix (by refering l2i_dict) and returns as int tensor
        
        Paramteters:
        ----------
        strings: list of preprocessed strings
        l2i_dict : letter2index dictionary to be used for converting string to tensor

        Returns:
        ------
        res : torch.Tensor of shape (len(strings), len(strings[0]))
        """
        res = torch.zeros(len(strings), len(strings[0]))
        
        for i in range(len(strings)):
            for j in range(len(strings[i])):
                if strings[i][j] not in l2i_dict :
                    res[i][j] = l2i_dict[self.unk_token]
                else:
                    res[i][j] = l2i_dict[strings[i][j]] 
        return res.type(torch.LongTensor)


    def load_data(self, set, batch_size, is_shuffle = True, num_batches = -1, padding_upper_bound = -1):
        """
        Parameters:
        -----------
        set: a string to define which set of data to load, set can be any one of the following ["train", "test", "valid"]
        batch_size: batch size in which the data be loaded
        num_batches: returns this many batches of size batch_size
                     if -1 : returns the entire data in batches 
                     else only returns that many number of batches
        
        Returns:
        -------
        A list of tuples (source, target), where source and target is a torch.Tensor
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
            source_batch, target_batch = self.preprocess(source_batch, padding_upper_bound), self.preprocess(target_batch, padding_upper_bound)
            source_batch, target_batch = self.string_to_tensor(source_batch, self.en_l2i), self.string_to_tensor(target_batch, self.tar_l2i)
            source_batch, target_batch = source_batch.transpose(0,1), target_batch.transpose(0,1)
            res.append((source_batch, target_batch))
            batch_count += 1
            if(num_batches != -1 and batch_count == num_batches):
                break
        return res


    def tensor_to_string(self, output, string_type = "target"):
        """
        Converts tensors to strings, where each string is stored in column wise in the tensors by default

        Parameters:
        ----------
        output shape: target_seq_length * N
        string_type : can take values of ["source", "target"]
                      if source: the ouput is encoded form of english
                      if target: the output is encoded form of target language
        
        Returns:
        A list of strings after convertion (and removing the tokens)
        """
        if(string_type == "target"):
            i2l_dict = self.tar_i2l
        elif(string_type == "source"):
            i2l_dict = self.en_i2l
        
        res = []
        for j in range(output.shape[1]):
            temp = ""
            for i in range(1, output.shape[0]): #starting from index 1, because 0th index is always start_token
                if(i2l_dict[output[i,j].item()] == self.end_token):
                    break
                temp += i2l_dict[output[i,j].item()]
            res.append(temp)
        return res