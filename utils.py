import torch


def data_loader(source, target, batch_size):
    """
    returns: list of list which contains padded (source, target) pairs
    """
    res = []
    for i in range(0, len(source), batch_size):
        batch = []
        src_batch = source[i:i+batch_size]
        tar_batch = target[i:i+batch_size]
        scr_max_len = len(max(src_batch, key=len))
        tar_max_len = len(max(tar_batch, key=len))

        for j in range(len(src_batch)):
            padded_scr, padded_tar = src_batch[j].ljust(scr_max_len), tar_batch[j].ljust(tar_max_len)
            batch.append((padded_scr, padded_tar))

        res.append(batch)
    
    return res


def preprocess(strings, start_token, end_token, pad_token):
    """Adds start and end token and adds padding"""
    res = []
    max_len = len(max(strings, key=len))

    for item in strings:
        temp = start_token + item + end_token
        temp = temp.ljust(max_len+2, pad_token) #2 is added, because we added start and end token to each word
        res.append(temp)

    return res


def string_to_tensor(strings, l2i_dict):
    """
    replaces the chareceters of the sting with corrospong ix (by refering l2i_dict) and returns as int tensor
    """
    res = torch.zeros(len(strings), len(strings[0]))
    
    for i in range(len(strings)):
        for j in range(len(strings[i])):
            if strings[i][j] not in l2i_dict :
                continue #ignoring the charecters that are not in the dictionary
            res[i][j] = l2i_dict[strings[i][j]]
        
    return res.type(torch.LongTensor)