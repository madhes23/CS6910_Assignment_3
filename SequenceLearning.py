import torch
import torch.nn as nn
import torch.nn.functional as F
from AkshrantarDataset import AksharantarDataset
import random
from tqdm import tqdm
from Utils import plot_graphs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

language = 'tam'
start_token = '<'
end_token = '>'
pad_token = ' '
unk_token = '~'

data = AksharantarDataset(language, start_token, end_token, pad_token, unk_token)
criterion = nn.CrossEntropyLoss(ignore_index=data.tar_l2i[pad_token])
target_char_count = len(data.tar_l2i)
english_char_count = len(data.en_l2i)

#main classes for the Seq2Seq laearning problem
class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers = 1, p = 0, bi_dir = False, rnn_class = nn.GRU):
        """
        Init Parameters:
        ----------------
        input_size : english_char_count
        embedding_size : size of each embedding vector
        hidden_size : size of hidden state vector
        num_layers : number of recurrent layers of RNN
        p : dropout probability
        bi_dir : flag to set bidirection in the RNN cell used
        rnn_class: type of RNN to be used in the encoder

        Input:
        ------
        x : torch.Tensor of shape (seq_length, N)
            where seq_length - len of longest string in the batch
            N - batch size
        
        Output:
        ------
        outputs: torch.Tensor of shape (seq_len, N, hidden_size * D), where D = 2 if bi_dir = True else 1
        hidden: torch.Tensor of shape (num_layers * D, N, hidden_size)
        
        cell: torch.Tensor of shape (num_layers * D, N, hidden_size) if(rnn_class == "LSTM")
        """
        super(Encoder, self).__init__()
        self.dropout = nn.Dropout(p)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn_class= rnn_class
        self.rnn = rnn_class(embedding_size, hidden_size, num_layers, dropout=p, bidirectional = bi_dir)


    def forward(self, x):
        # x shape: (seq_length, N) where N is batch size

        embedding = self.dropout(self.embedding(x))
        # embedding shape: (seq_length, N, embedding_size)

        if(self.rnn_class.__name__ == "LSTM"):
            outputs, (hidden, cell) = self.rnn(embedding)
            # outputs shape: (seq_length, N, hidden_size)
        else:
            outputs, hidden = self.rnn(embedding)
        
        if(self.rnn_class.__name__ == "LSTM"):
            return outputs, hidden, cell
        else:
            return outputs, hidden


class Decoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers = 1, p = 0, bi_dir = False, rnn_class = nn.GRU):
        """
        Init Parameters:
        ---------------
        input_size: target_char_count
        embedding_size: size of each embedding vector
        hidden_size: size of hidden state vector
        output_size: number of output features in fully connected layer, here target_char_count
        num_layers : number of recurrent layers of RNN
        p : dropout probability
        bi_dir : flag to set bidirection in the RNN cell used
        rnn_class: type of RNN to be used in the encoder

        Input:
        -----
        x: torch.Tensor of shape (N)
        hidden: torch.Tensor of shape (num_layers * D, N, hidden_size), where D = 2 if bi_dir = True else 1
        cell: torch.Tensor of shape (num_layers * D, N, hidden_size)

        Outputs:
        predications: torch.Tensor of shape (N, target_char_count)
        hidden: torch.Tensor of shape (num_layers * D, N, hidden_size)
        
        cell: torch.Tensor of shape (num_layers * D, N, hidden_size) if(rnn_class == "LSTM")
        """
        super(Decoder, self).__init__()
        self.dropout = nn.Dropout(p)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.used_attn = False

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn_class = rnn_class
        self.rnn = rnn_class(embedding_size, hidden_size, num_layers, dropout=p, bidirectional = bi_dir)

        self.D = 1
        if(bi_dir == True):
            self.D = 2
        self.fc = nn.Linear(hidden_size * self.D, output_size)


    def forward(self, x, hidden, cell = None, encoder_outputs = None):
        #cell is set to none, for GRU and RNN

        # x shape: (N) where N is for batch size, we want it to be (1, N), seq_length
        # is 1 here because we are sending in a single word and not a sentence
        # print(x.shape, hidden.shape, cell.shape)
        x = x.unsqueeze(0)

        embedding = self.dropout(self.embedding(x))
        # embedding shape: (1, N, embedding_size)
        
        if(self.rnn_class.__name__ == "LSTM"):
            outputs, (hidden, cell) = self.rnn(embedding, (hidden, cell))
            # outputs shape: (1, N, hidden_size * D)
        else:
            outputs, hidden = self.rnn(embedding, hidden)
            

        predictions = self.fc(outputs)

        # predictions shape: (1, N, length_target_vocabulary) to send it to
        # loss function we want it to be (N, length_target_vocabulary) so we're
        # just gonna remove the first dim
        predictions = predictions.squeeze(0)

        if(self.rnn_class.__name__ == "LSTM"):
            return predictions, hidden, cell
        else:
            return predictions, hidden


class AttnDecoder(nn.Module):
    """
    Init Parameters:
    ---------------
    embedding_size: size of each embedding vector
    hidden_size: size of hidden state vector
    output_size: number of output features in fully connected layer, here target_char_count
    num_layers : number of recurrent layers of RNN
    p : dropout probability
    max_length : maximum length of the input word (from encoder) for which this decoder is able to handle - by using attention mechanism
    bi_dir : flag to set bidirection in the RNN cell used
    rnn_class: type of RNN to be used in the encoder

    Input:
    -----
    input: torch.Tensor of shape (N)
    hidden: torch.Tensor of shape (num_layers * D, N, hidden_size), where D = 2 if bi_dir = True else 1
    cell: torch.Tensor of shape (num_layers * D, N, hidden_size) if rnn_class == "LSTM"
    encoder_outputs: torch.Tensor of shape (seq_len, N, hidden_size * D)

    Output:
    ------
    prob : torch.Tensor of shape (N, target_char_count)
    hidden : torch.Tensor of shape (num_layers * D, N, hidden_size), where D = 2 if bi_dir = True else 1
    attn_weights : torch.Tensor of shape (1, N, max_len)

    cell : torch.Tensor of shape (num_layers * D, N, hidden_size) if rnn_class == "LSTM"
    """
    def __init__(self, embedding_size, hidden_size, output_size, num_layers = 1, dropout_p=0.1, max_length=32, bi_dir = False, rnn_class = nn.GRU):
        super(AttnDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size= embedding_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.num_layers = num_layers
        self.used_attn = True
        self.D = 1
        if(bi_dir == True):
            self.D = 2
        self.rnn_class = rnn_class

        self.embedding = nn.Embedding(output_size, embedding_size)
        self.attn = nn.Linear((self.D * num_layers * hidden_size) + embedding_size, max_length)
        self.attn_combine = nn.Linear((self.D * hidden_size) + embedding_size, hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.rnn = self.rnn_class(hidden_size, hidden_size, num_layers, dropout = dropout_p, bidirectional = bi_dir)
        self.out = nn.Linear(hidden_size * self.D, output_size)

    def forward(self, input, hidden, cell = None, encoder_outputs = None):
        embedded = self.embedding(input)
        embedded = self.dropout(embedded).unsqueeze(0)
        # print("embed ",embedded.shape)
        # (1, N, es)

        #IDEA: cat for each in 0th dim of hidd for handling different num_layers in encoder and decoder
        temp = torch.cat((embedded, hidden[0].unsqueeze(0)), 2)
        for i in range(hidden.shape[0]-1):
            temp = torch.cat((temp, hidden[i].unsqueeze(0)), 2)

        # print("temp ", temp.shape) # (1, N, (d*nl).hs+es)

        temp = self.attn(temp)
        # print("after attn ", temp.shape) # (1, N, max)

        attn_weights = F.softmax(
            temp, dim=1)
        # print("attn_weights :",attn_weights.shape) # (1, N, max)
        # print("ecn_op: ", encoder_outputs.shape)
        # print("attn wei: ", attn_weights.transpose(0,1).shape)
        # print("enc_opts ", encoder_outputs.transpose(0,1).shape)
        
        attn_applied = torch.bmm(attn_weights.transpose(0,1),
                                 encoder_outputs.transpose(0,1))
        # attn_applied (N, 1, hs)

        attn_applied = attn_applied.transpose(0,1) #(1, N, d.hs)
        
        # print("attn appld ", attn_applied.shape) # (1, N, d.hs)

        output = torch.cat((embedded, attn_applied), 2)
        # print("outpt after cat ",output.shape) # (1, N, (D)hs+es)

        output = self.attn_combine(output)
        # print("after atn comb: ",output.shape) # (1, N, hs)
        
        output = F.relu(output)
        if(self.rnn_class.__name__ == "LSTM"):
            output, (hidden, cell) = self.rnn(output, (hidden, cell)) 
        else:
            output, hidden = self.rnn(output, hidden)

        # print("out ", output.shape, "hid ", hidden.shape)

        prob = self.out(output).squeeze(0) #(1, N, op)
        # print("prob ", prob.shape)

        # output = F.log_softmax(self.out(output[0]), dim=1)
        # print("attn ", attn_weights.shape)
        if(self.rnn_class.__name__ == "LSTM"):
            return prob, hidden, cell, attn_weights
        else:
            return prob, hidden, attn_weights


class Seq2Seq(nn.Module):
    """
    This Class combines the Encoder and Decoder Classes seen above. 
    Init Parameters:
    ---------------
    encoder: Encoder class object
    decoder: Decoder class object

    Input:
    -----
    source : torch.Tensor of shape (source seq_len, N) where source seq_len = len(longest word in the batch) if attention is not used, 
                                                                                                        else seq_len = max_length 
    target : torch.Tensor of shape (target seq_len, N)
    teacher_forcing : A boolean value to indicate if the teacher forcing should be used.
    
    Output:
    ------
    outputs : torch.Tensor of shape(target seq_len, batch_size, target_char_count)
    """
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.attn_dec = decoder.used_attn
        self.encoder_layers = encoder.num_layers
        self.decoder_layers = decoder.num_layers
        self.D = decoder.D #we set bidiretion as common in both encoder and decoder, so no need to check for D value seperately
        self.enc_to_dec = nn.Linear(self.encoder_layers*self.D, self.decoder_layers*self.D)
        self.rnn_class = decoder.rnn_class #we use same rnn in both encoder and decoder

    def forward(self, source, target, teacher_forcing = False):
        batch_size = source.shape[1] 
        target_len = target.shape[0]

        # print("source shape ", source.shape)
        # print("target shape ", target.shape)
        # print("N : ", batch_size)
        # print("tar len : ", target_len)

        outputs = torch.zeros(target_len, batch_size, target_char_count)
        # print("outputs shape : ", outputs.shape)

        
        if(self.rnn_class.__name__ == "LSTM"):
            enc_ops, hidden, cell = self.encoder(source)
        else:
            enc_ops, hidden = self.encoder(source)
        
        if(self.encoder_layers > self.decoder_layers): # take only the top layers
            hidden = hidden[-self.D * self.decoder_layers: , : , : ]
        elif(self.encoder_layers < self.decoder_layers): #repeat the top most layer to decoder_layer number of times (without the loss of bidirectional info)
            last = hidden[-self.D * 1: , : , :]
            # print("last : ", last.shape)
            hidden = last.repeat(self.decoder_layers, 1, 1)
            # print("hidden after ", hidden.shape)        


        if(self.rnn_class.__name__ == "LSTM"):
            if(self.encoder_layers > self.decoder_layers): # take only the top layers
                cell = cell[-self.D * self.decoder_layers: , : , : ]
            elif(self.encoder_layers < self.decoder_layers): #repeat the top most layer to decoder_layer number of times (without the loss of bidirectional info)
                last = cell[-self.D * 1: , : , :]
                cell = last.repeat(self.decoder_layers, 1, 1)


        # Get the first input (start_token) and input it to the Decoder
        x = target[0]
        outputs[:, :, data.tar_l2i[start_token]] = 1 #setting prob = 1 for starting token 
        # print("target len :", target_len)
        overall_attn = None
        for t in range(1, target_len):
            # Use previous hidden, cell as context from encoder at start
            if(self.rnn_class.__name__ == "LSTM"):
                if(self.attn_dec == True):
                    output, hidden, cell, attention = self.decoder(x, hidden, cell, encoder_outputs= enc_ops)
                    # print(attention.shape)
                    if(overall_attn == None):
                        overall_attn = attention
                    else:
                        overall_attn = torch.cat((overall_attn, attention), dim=0)
                else:
                    output, hidden, cell = self.decoder(x, hidden, cell, encoder_outputs= enc_ops)
            else:
                if(self.attn_dec == True):
                    output, hidden, attention = self.decoder(x, hidden, encoder_outputs=enc_ops)
                else:
                    output, hidden = self.decoder(x, hidden, encoder_outputs=enc_ops)

            # Store next output prediction
            outputs[t] = output

            # Get the best word the Decoder predicted (index in the vocabulary)
            best_guess = output.argmax(1)

            # With probability of teacher_force_ratio we take the actual next word
            # otherwise we take the word that the Decoder predicted it to be.
            # Teacher Forcing is used so that the model gets used to seeing
            # similar inputs at training and testing time, if teacher forcing is 1
            # then inputs at test time might be completely different than what the
            # network is used to. This was a long comment.
            x = target[t] if teacher_forcing == True else best_guess
        # print("OUTPUTS: ", outputs)

        # print(attention.shape)
        # print(overall_attn.shape)
        if(self.attn_dec == True):
            return outputs, overall_attn
        else:
            return outputs


    def get_attention_matrix(self, source, target):
        if(self.attn_dec == False):
            print("Invalid request")
            return
    
        self.eval()
        with torch.no_grad():
            _, attention = self(source, target)
            attention.to(device)
        self.train()
        attention =  attention.permute(1,0,2) #shape : (N, max_len-1, max_len)
        return attention[:, :, 1:] #shape: (N, max_len -1, max_len-1)


    def calc_accuracy(self, output, target):
        """
        Compares the tensors, and calculates the wordwise accuracy. 
        Note: We do not care what is the ouput after we see the end_token
        hence comparing only upto the end_token
        Parameters:
        ---------
        output: torch.Tensor of shape (seq_len, N)
        target: torch.Tensor of shape (seq_len, N)
        Returns:
        -------
        word-wise comparition accuracy[in the range 0-100%] between output vs target matrix
        """
        # batch_size = 32
        seq_len = output.shape[0]
        N = output.shape[1]
        matched_strings = 0
        with torch.no_grad():
            for j in range(N):
                current_word_matched = True
                for i in range(seq_len):
                    if(target[i][j] == data.tar_l2i[pad_token]): #we dont care whatever prediction in the pad_token place
                        break
                    if(output[i][j] != target[i][j]): #compare the predictions of charecters, start and end_token places
                        current_word_matched = False
                        break
                if(current_word_matched == True):
                    matched_strings += 1
        return matched_strings*100 / N 
    

    def calc_evaluation_metrics(self, src_tar_pair, path_to_store_predictions = None):
        """
        This function computes evaluation metrics, including loss and accuracy, 
        and also provides an option to store the predictions in a file.

        Parameter:
        --------
        src_tar_pair: a list of tuples, where each tuple consists of two tensors (source, target)
                      source:tensors of shape (seq_len * N)
                      target:tensors of shape (seq_len * N) where source seq_len = len(longest word in the batch) if attention is not used, 
                                                                                                                    else seq_len = max_length 
                    Note: seq_len might not be same across all the batches if attn is not used
        path_to_store_predictions: str - path of a file to store the predicted values
        
        Returns:
        --------
        loss, accuracy
        """
        num_batches = len(src_tar_pair)
        self.eval() #change model to evaluation mode
        with torch.no_grad():
            acc = 0
            running_loss = 0
            for source, target in src_tar_pair:
                source = source.to(device)
                target = target.to(device)
                
                if(self.attn_dec == True):
                    output, _ = self(source, target)
                else:
                    output = self(source, target)
                
                output = output.to(device)

                if(path_to_store_predictions != None):
                    source_text = data.tensor_to_string(source, "source")
                    target_text = data.tensor_to_string(target)
                    ouput_text = data.tensor_to_string(output.argmax(2))

                    with open(path_to_store_predictions, 'a', encoding="utf-8") as f:
                        f.write(f'Source,Target,Predicted,PredictionStatus\n')
                        for en, crt, pred in zip(source_text, target_text, ouput_text):
                            prediction_status = "wrong"
                            if(crt == pred):
                                prediction_status = "correct"
                            f.write(f'{en},{crt},{pred},{prediction_status}\n')

                acc += self.calc_accuracy(output.argmax(2), target)

                output = output.reshape(-1, output.shape[2])
                target = target.reshape(-1)

                loss = criterion(output, target)
                running_loss += loss.item()
        self.train() #change model back to training mode
        return running_loss/num_batches, acc/num_batches


    def learn(self, train, valid, num_epochs, optimizer):
        loss_list, acc_list, val_loss_list, val_acc_list = [], [], [], []
        teacher_forcing_ratio = 0.5
        early_stoping_patience = 5
        # print_batches = 1

        for epoch in range(num_epochs):
            running_loss = 0
            running_accuracy = 0
            total_batches = len(train)
                

            for inp_data, target in tqdm(train, desc=f"[Epoch {epoch+1:3d}/{num_epochs}] ", leave = False):        
            # for inp_data, target in train:
                inp_data = inp_data.to(device)
                target = target.to(device)

                teacher_forcing = False
                if(epoch < 0.5*num_epochs): #for inital epochs, batch wise tfr
                    teacher_forcing = True if random.random() < teacher_forcing_ratio else False
                    
                if(self.attn_dec == True):
                    output, _ = self(inp_data, target, teacher_forcing)
                else:
                    output = self(inp_data, target, teacher_forcing)
                
                output = output.to(device)
                
                # if(epoch == num_epochs-1 and print_batches != 0):
                #     print(data.tensor_to_string(output.argmax(2)))
                #     print(data.tensor_to_string(target))
                #     print_batches -= 1

                running_accuracy += self.calc_accuracy(output.argmax(2), target)

                output = output.reshape(-1, output.shape[2])
                target = target.reshape(-1)

                optimizer.zero_grad()
                loss = criterion(output, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1)
                optimizer.step()
                running_loss += loss.item()

            val_loss, val_accuracy= self.calc_evaluation_metrics(valid)

            print(f"[Epoch {epoch+1:3d}/{num_epochs}] \t Loss: {(running_loss/total_batches):.3f}\t Acc: {(running_accuracy/total_batches):2.2f} \t Val Loss: {val_loss:2.3f}\t Val Acc: {val_accuracy:2.2f}")
            loss_list.append(running_loss/total_batches)
            acc_list.append(running_accuracy/total_batches)
            val_loss_list.append(val_loss)
            val_acc_list.append(val_accuracy)

            # if(epoch+1 >= early_stoping_patience): #TODO: uncomment this
            #     if(val_acc_list[-1] < 10): #breaking if the validation acc hasnt incresed above 10 even after 'early_stoping_patience' epochs
            #         break

            #     temp_acc_list = val_acc_list[-early_stoping_patience : ] #breaking if the validation data starts to overfit
            #     over_fitting = True
            #     for i in range(1,len(temp_acc_list)):
            #         if(round(temp_acc_list[i-1], 1) <= round(temp_acc_list[i], 1)):
            #             over_fitting = False
            #             break
            #     if(over_fitting == True):
            #         break


        return loss_list, val_loss_list, acc_list, val_acc_list