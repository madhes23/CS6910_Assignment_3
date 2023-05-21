import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import seaborn as sns

def plot_graphs(training_errors, validation_errors, training_accuracy, validation_accuracy):
    """
    Plots a Error and Accuracy graphs for training and validation data over the epochs

    Parameters:
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


def plot_attention_heatmap(attention_matrix, source, target, ax = None, normalize = True):
    """
    attention_matrix : torch.Tensor of shape (max_len-1, max_len-1) 
    source : source string
    target : target string
    ax : axis in which to draw the plot, otherwise use the currently-active axis.
    
    Returns:
    ax : matplotlib Axis
    """
    attention_matrix = attention_matrix[:len(target), :len(source)] #only using upto the end_token
    if(normalize == True):
        row_sum = attention_matrix.sum(1, keepdim = True)
        attention_matrix = attention_matrix / row_sum
    attention_matrix = attention_matrix.detach().cpu().numpy()

    ax = sns.heatmap(attention_matrix, cmap="Blues", cbar=False, ax= ax, fmt=".2f")
    ax.set_xticks(np.arange(len(source)) + 0.5)
    ax.set_xticklabels(list(source), rotation=0, fontproperties=fm.FontProperties(fname='C:/Windows/Fonts/Nirmala.ttf'))
    ax.set_yticks(np.arange(len(target)) + 0.5)
    ax.set_yticklabels(list(target), rotation=0, fontproperties=fm.FontProperties(fname='C:/Windows/Fonts/Nirmala.ttf'))
    return ax
    

def plot_attention_heatmaps(attention_matrixes, source_strings, target_strings, normalize = True):
    """
    This function can plot 12 attention heatmaps, which 3 heatmap per row
    attention_matrixes: torch.Tensor of shape (12, max_len-1, max_len-1) where k is number of attention_matrices
    source_strings: a list of 12 source strings
    target_strings: a list of 12 target strings
    """
    fig, axes = plt.subplots(4, 3, figsize=(12, 6)) #change here to change the number of heatmaps visualized
    for i, ax in enumerate(axes.flat):
        ax = plot_attention_heatmap(attention_matrixes[i], source_strings[i], target_strings[i], ax=ax, normalize = normalize)
    fig.tight_layout()
    fig.set_size_inches(12, 18)
    fig.suptitle("Sample Attention Heatmaps")
    return fig