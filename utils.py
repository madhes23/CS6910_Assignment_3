import torch
import matplotlib.pyplot as plt
import numpy as np

# def data_loader(source, target, batch_size):
#     """
#     returns: list of list which contains padded (source, target) pairs
#     """
#     res = []
#     for i in range(0, len(source), batch_size):
#         batch = []
#         src_batch = source[i:i+batch_size]
#         tar_batch = target[i:i+batch_size]
#         scr_max_len = len(max(src_batch, key=len))
#         tar_max_len = len(max(tar_batch, key=len))

#         for j in range(len(src_batch)):
#             padded_scr, padded_tar = src_batch[j].ljust(scr_max_len), tar_batch[j].ljust(tar_max_len)
#             batch.append((padded_scr, padded_tar))

#         res.append(batch)
#     return res


def preprocess(strings, start_token, end_token, pad_token):
    """Adds start and end token and adds padding"""
    res = []
    max_len = len(max(strings, key=len))

    for item in strings:
        temp = start_token + item + end_token
        temp = temp.ljust(max_len+2, pad_token) #2 is added, because we added start and end token to each word
        res.append(temp)

    return res


def string_to_tensor(strings, l2i_dict, unk_token):
    """
    replaces the chareceters of the sting with corrospong ix (by refering l2i_dict) and returns as int tensor
    """
    res = torch.zeros(len(strings), len(strings[0]))
    
    for i in range(len(strings)):
        for j in range(len(strings[i])):
            if strings[i][j] not in l2i_dict :
                res[i][j] = l2i_dict[unk_token]
            else:
                res[i][j] = l2i_dict[strings[i][j]]
        
    return res.type(torch.LongTensor)


def plot_graphs(training_errors, validation_errors, training_accuracy, validation_accuracy):
    """
    Plots a Error and Accuracy graphs for training and validation data over the epochs

    Params:
    -----
    training_errors: list containing error (ie loss values) of the training data over the epochs
    validation_errors: list containing error (ie loss values) of the validation data over the epochs
    training_accuracy: list containing accuracy of the training data over the epochs
    validation_accuracy: list containing accuracy of the validation data over the epochs

    Returns:
    -----
    fig: matplot figure object 
    """
    x = np.arange(len(training_errors))
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(x, training_errors, label = "Training Error")
    ax1.plot(x, validation_errors, label = "Validation Error")
    ax1.set_title("Errors")
    ax1.legend(fontsize= 'small', loc='upper right')
    ax2.plot(x, training_accuracy, label = "Training Accuracy")
    ax2.plot(x, validation_accuracy, label = "Validation Accuracy")
    ax2.set_title("Accuracy")
    ax2.legend(fontsize = 'small',loc='upper right')
    return fig