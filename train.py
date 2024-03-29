from SequenceLearning import *
import argparse
import torch.optim as optim
import wandb
import matplotlib.pyplot as plt
from Utils import plot_graphs


# Model hyperparameters (for both attention and vanilla models)
batch_size = 128
input_size_encoder = english_char_count
input_size_decoder = target_char_count
output_size = target_char_count
MAX_LEN = 35


parser = argparse.ArgumentParser(description="The command line interface to interact with the repository")
parser.add_argument('-bs', '--batch_size', metavar="", required=False, type=int, default=128, help="Batch size")
parser.add_argument('-e', '--epochs', metavar="", required=True, type=int, default=10, help="Number of Epochs to run the training")
parser.add_argument('-lr', '--learning_rate', metavar="", required=True, type=float, help="Learning rate to be used in the optimizer")
parser.add_argument('-emb', '--embedding_size', metavar="", required=True, type=int, help="Size of the Embedding vectors")
parser.add_argument('-el', '--encoder_layers', metavar="", required=True, type=int, help="Number of the layers used in the Encoder RNN Cells")
parser.add_argument('-dl', '--decoder_layers', metavar="", required=True, type=int, help="Number of the layers used in the Decoder RNN Cells")
parser.add_argument('-ep', '--enc_dropout', metavar="", required=True, type=float, help="Dropout probability in Encoder RNN while training")
parser.add_argument('-dp', '--dec_dropout', metavar="", required=True, type=float, help="Dropout probability in Decoder RNN while training")
parser.add_argument('-hs', '--hidden_size', metavar="", required=True, type=int, help="Size of the Hidden state vectors")
parser.add_argument('-D', '--bi_directional', action='store_true', help="If this arg is passed, bi directional RNN Cells will be used")
parser.add_argument('-at', '--attention', action='store_true', help="If this arg is passed, attention decoder will be used")
parser.add_argument('-rnn', '--rnn_class', metavar="", required=True, type=str, choices=["LSTM", "GRU", "RNN"], help="Type of RNN Class to be used in RNN Cells of Encoder, Decoder")
parser.add_argument('-wp', '--wandb_project', metavar="", required=False, type=str, default="Assignment3_CLI", help="W&B Project for logging and monitoring")
parser.add_argument('-we', '--wandb_entity', metavar="", required=False, type=str, default="madhes23", help="W&B entity")
args = parser.parse_args()


if(__name__ == '__main__'):
    if(args.rnn_class == "LSTM"):
        rnn=nn.LSTM
    elif(args.rnn_class == "GRU"):
        rnn=nn.GRU
    elif(args.rnn_class == "RNN"):
        rnn=nn.RNN

    if args.bi_directional is None:
        bi_directional = False
    else:
        bi_directional = True
    
    enc = Encoder(english_char_count, 
                args.embedding_size, args.hidden_size, 
                num_layers=args.encoder_layers, 
                bi_dir=bi_directional,
                p=args.enc_dropout,
                rnn_class=rnn).to(device)
    
    if args.attention is None:
        dec = Decoder(target_char_count, args.embedding_size, args.hidden_size, target_char_count, 
                    num_layers=args.decoder_layers, 
                    bi_dir=bi_directional, 
                    p = args.dec_dropout,
                    rnn_class=rnn).to(device)
        print("Loading the data...")
        train = data.load_data("train", batch_size)
        valid = data.load_data("valid", batch_size)
        test = data.load_data(  "test", batch_size)

    else:
        dec = AttnDecoder(args.embedding_size, args.hidden_size, output_size=target_char_count, 
                            num_layers=args.decoder_layers,
                            bi_dir = bi_directional,
                            rnn_class= rnn,
                            max_length=MAX_LEN).to(device)
        print("Loading the data...")
        train = data.load_data("train", batch_size, padding_upper_bound=MAX_LEN)
        valid = data.load_data("valid", batch_size, padding_upper_bound=MAX_LEN)
        test = data.load_data(  "test", batch_size, padding_upper_bound=MAX_LEN)


    model = Seq2Seq(enc, dec).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    print("Training...")
    loss, val_loss, acc, val_acc = model.learn(train, valid, args.epochs, optimizer)
    fig = plot_graphs(loss, val_loss, acc, val_acc)
    plt.show()

    #logging in WandB
    print("Syncing loss and accuracies to WandB: ")
    run_name = f"cli_{args.encoder_layers}_enc_{args.decoder_layers}_dec_{args.hidden_size}_hs_"
    if(bi_directional == True):
        run_name += "bidir_"
    if(dec.used_attn == True):
        run_name += "attn_"
    if(len(loss) != args.epochs):
        run_name += "early_stop"
        print("Early Stopping is performed!!")

    wandb.init(project = args.wandb_project, entity = args.wandb_entity, name = run_name)
    for i in range(len(loss)):
        wandb.log({"tr_err":loss[i],
                    "tr_acc" : acc[i],
                    "val_err" : val_loss[i],
                    "val_acc" : val_acc[i],
                    "epoch":(i+1)})
    
    wandb.finish()
