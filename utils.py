import matplotlib.pyplot as plt
import numpy as np

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