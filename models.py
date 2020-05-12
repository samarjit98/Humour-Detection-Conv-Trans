import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionModel(nn.Module):

    def __init__(self, 
                num_chars=93, 
                embedding_dim=128, 
                hidden_dim=100, 
                labels=2):
        super(AttentionModel, self).__init__()
        
        self.embedding = nn.Embedding(num_chars, embedding_dim)
        self.convnet = nn.Sequential(
                                nn.Conv1d(embedding_dim, embedding_dim, 3, padding=1),
                                nn.MaxPool1d(3),
                                nn.BatchNorm1d(embedding_dim),
                                nn.Dropout(),
                                nn.Conv1d(embedding_dim, embedding_dim, 3, padding=1),
                                nn.MaxPool1d(3),
                                nn.BatchNorm1d(embedding_dim),
                                nn.Dropout(),
                            )
        self.bilstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)
        self.W_s1 = nn.Linear(2*hidden_dim, 100)
        self.W_s2 = nn.Linear(100, 30)
        self.fc_layer = nn.Linear(30*2*hidden_dim, 100)
        self.label = nn.Linear(100, labels)

    def attention_net(self, lstm_output):
        attn_weight_matrix = self.W_s2(torch.tanh(self.W_s1(lstm_output)))
        attn_weight_matrix = attn_weight_matrix.permute(0, 2, 1)
        attn_weight_matrix = F.softmax(attn_weight_matrix, dim=2)

        return attn_weight_matrix


    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x = self.convnet(x)
        x = x.permute(0, 2, 1)
        x = x.permute(1, 0, 2)
        output, _ = self.bilstm(x)
        output = output.permute(1, 0, 2)
        attn_weight_matrix = self.attention_net(output)
        hidden_matrix = torch.bmm(attn_weight_matrix, output)
        fc_out = self.fc_layer(hidden_matrix.view(-1, hidden_matrix.size()[1]*hidden_matrix.size()[2]))
        logits = self.label(fc_out)

        return logits

class AttentionModel2(nn.Module):

    def __init__(self, 
                num_chars=93, 
                embedding_dim=128, 
                hidden_dim=100, 
                labels=2):
        super(AttentionModel2, self).__init__()
        
        self.embedding = nn.Embedding(num_chars, embedding_dim)
        self.convnet = nn.Sequential(
                                nn.Conv1d(embedding_dim, embedding_dim, 3, padding=1),
                                nn.MaxPool1d(3),
                                nn.BatchNorm1d(embedding_dim),
                                nn.Dropout(),
                                nn.Conv1d(embedding_dim, embedding_dim, 3, padding=1),
                                nn.MaxPool1d(3),
                                nn.BatchNorm1d(embedding_dim),
                                nn.Dropout(),
                            )
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.label = nn.Linear(hidden_dim, labels)

    def attention_net(self, lstm_output, final_state):
        hidden = final_state.squeeze(0)
        attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return new_hidden_state

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x = self.convnet(x)
        x = x.permute(0, 2, 1)
        x = x.permute(1, 0, 2)
        output, (final_hidden_state, final_cell_state) = self.lstm(x) 
        output = output.permute(1, 0, 2) 
        attn_output = self.attention_net(output, final_hidden_state)
        logits = self.label(attn_output)

        return logits

class CNN(nn.Module):

    def __init__(self, 
                num_chars=38, 
                embedding_dim=128, 
                labels=2):
        super(CNN, self).__init__()

        self.embedding = nn.Embedding(num_chars, embedding_dim)
        self.convnet1 = nn.Sequential(
                                nn.Conv1d(embedding_dim, embedding_dim, 3, padding=1),
                                nn.BatchNorm1d(embedding_dim),
                                nn.ReLU(),
                                nn.MaxPool1d(3),
                                nn.Dropout(),
                                nn.Conv1d(embedding_dim, embedding_dim, 3, padding=1),
                                nn.BatchNorm1d(embedding_dim),
                                nn.ReLU(),
                                nn.MaxPool1d(3),
                                nn.Dropout(),
                            )
        self.attn1 = nn.TransformerEncoderLayer(128, 8, 
                                                dim_feedforward=512, 
                                                dropout=0.5, 
                                                activation='relu')
        self.convnet2 = nn.Sequential(
                                nn.Conv1d(embedding_dim, embedding_dim, 3, padding=1),
                                nn.BatchNorm1d(embedding_dim),
                                nn.ReLU(),
                                nn.MaxPool1d(3),
                                nn.Dropout(),
                            )
        self.attn2 = nn.TransformerEncoderLayer(128, 8, 
                                                dim_feedforward=512, 
                                                dropout=0.5, 
                                                activation='relu')
        self.convnet3 = nn.Sequential(
                                nn.Conv1d(embedding_dim, embedding_dim, 3, padding=1),
                                nn.BatchNorm1d(embedding_dim),
                                nn.ReLU(),
                                nn.MaxPool1d(3),
                                nn.Dropout(),
                            )
        self.label = nn.Linear(2*embedding_dim, labels)

    def forward(self, x):
        this_batch = x.shape[0]
        x = self.embedding(x)

        x = x.permute(0, 2, 1)
        x = self.convnet1(x)
        x = x.permute(0, 2, 1)

        x = x.permute(1, 0, 2)
        x = self.attn1(x)
        x = x.permute(1, 0, 2)

        x = x.permute(0, 2, 1)
        x = self.convnet2(x)
        x = x.permute(0, 2, 1)

        x = x.permute(1, 0, 2)
        x = self.attn2(x)
        x = x.permute(1, 0, 2)

        x = x.permute(0, 2, 1)
        x = self.convnet3(x)
        x = x.permute(0, 2, 1)

        x = x.reshape((this_batch, -1))

        return self.label(x)

class CNN2(nn.Module):

    def __init__(self, 
                num_chars=38, 
                embedding_dim=128, 
                labels=2):
        super(CNN2, self).__init__()

        self.embedding = nn.Embedding(num_chars, embedding_dim)
        self.convnet1 = nn.Sequential(
                                nn.Conv1d(embedding_dim, embedding_dim, 3, padding=1),
                                nn.BatchNorm1d(embedding_dim),
                                nn.ReLU(),
                                nn.MaxPool1d(3),
                                nn.Dropout(),
                                nn.Conv1d(embedding_dim, embedding_dim, 3, padding=1),
                                nn.BatchNorm1d(embedding_dim),
                                nn.ReLU(),
                                nn.MaxPool1d(3),
                                nn.Dropout(),
                            )
        self.attn1_1 = nn.TransformerEncoderLayer(128, 8, 
                                                dim_feedforward=512, 
                                                dropout=0.5, 
                                                activation='relu')
        self.attn1_2 = nn.TransformerEncoderLayer(128, 8, 
                                                dim_feedforward=512, 
                                                dropout=0.5, 
                                                activation='relu')
        self.convnet2 = nn.Sequential(
                                nn.Conv1d(embedding_dim, embedding_dim, 3, padding=1),
                                nn.BatchNorm1d(embedding_dim),
                                nn.ReLU(),
                                nn.MaxPool1d(3),
                                nn.Dropout(),
                            )
        self.attn2 = nn.TransformerEncoderLayer(128, 8, 
                                                dim_feedforward=512, 
                                                dropout=0.5, 
                                                activation='relu')
        self.convnet3 = nn.Sequential(
                                nn.Conv1d(embedding_dim, embedding_dim, 3, padding=1),
                                nn.BatchNorm1d(embedding_dim),
                                nn.ReLU(),
                                nn.MaxPool1d(3),
                                nn.Dropout(),
                            )
        self.label = nn.Linear(2*embedding_dim, labels)

    def forward(self, x):
        this_batch = x.shape[0]
        x = self.embedding(x)

        x = x.permute(0, 2, 1)
        x = self.convnet1(x)
        x = x.permute(0, 2, 1)

        x = x.permute(1, 0, 2)
        x = self.attn1_1(x)
        x = self.attn1_2(x)
        x = x.permute(1, 0, 2)

        x = x.permute(0, 2, 1)
        x = self.convnet2(x)
        x = x.permute(0, 2, 1)

        x = x.permute(1, 0, 2)
        x = self.attn2(x)
        x = x.permute(1, 0, 2)

        x = x.permute(0, 2, 1)
        x = self.convnet3(x)
        x = x.permute(0, 2, 1)

        x = x.reshape((this_batch, -1))

        return self.label(x)

'''
class CNN2(nn.Module):

    def __init__(self, 
                num_chars=38, 
                embedding_dim=128, 
                labels=2):
        super(CNN2, self).__init__()

        self.embedding = nn.Embedding(num_chars, embedding_dim)
        self.convnet1 = nn.Sequential(
                                nn.Conv1d(embedding_dim, embedding_dim, 3, padding=1),
                                nn.BatchNorm1d(embedding_dim),
                                nn.ReLU(),
                                nn.MaxPool1d(3),
                                nn.Dropout(),
                                nn.Conv1d(embedding_dim, embedding_dim, 3, padding=1),
                                nn.BatchNorm1d(embedding_dim),
                                nn.ReLU(),
                                nn.MaxPool1d(3),
                                nn.Dropout(),
                            )
        self.attn1_1 = nn.TransformerEncoderLayer(128, 8, 
                                                dim_feedforward=512, 
                                                dropout=0.5, 
                                                activation='relu')
        self.attn1_2 = nn.TransformerEncoderLayer(128, 8, 
                                                dim_feedforward=512, 
                                                dropout=0.5, 
                                                activation='relu')
        self.convnet2 = nn.Sequential(
                                nn.Conv1d(embedding_dim, embedding_dim, 3, padding=1),
                                nn.BatchNorm1d(embedding_dim),
                                nn.ReLU(),
                                nn.MaxPool1d(3),
                                nn.Dropout(),
                            )
        self.attn2 = nn.TransformerEncoderLayer(128, 8, 
                                                dim_feedforward=512, 
                                                dropout=0.5, 
                                                activation='relu')
        self.convnet3 = nn.Sequential(
                                nn.Conv1d(embedding_dim, embedding_dim, 3, padding=1),
                                nn.BatchNorm1d(embedding_dim),
                                nn.ReLU(),
                                nn.MaxPool1d(3),
                                nn.Dropout(),
                            )
        self.label = nn.Linear(2*embedding_dim, labels)

    def forward(self, x):
        this_batch = x.shape[0]
        x = self.embedding(x)

        x = x.permute(0, 2, 1)
        x = self.convnet1(x)
        x = x.permute(0, 2, 1)

        x = x.permute(1, 0, 2)
        x = self.attn1_1(x)
        x = self.attn1_2(x)
        x = x.permute(1, 0, 2)

        x = x.permute(0, 2, 1)
        x = self.convnet2(x)
        x = x.permute(0, 2, 1)

        x = x.permute(1, 0, 2)
        x = self.attn2(x)
        x = x.permute(1, 0, 2)

        x = x.permute(0, 2, 1)
        x = self.convnet3(x)
        x = x.permute(0, 2, 1)

        x = x.reshape((this_batch, -1))

        return self.label(x)
'''

'''
The following are implemented by Himadyuti Bhanja
'''

class CharRNN(nn.Module):
    def __init__(self, 
                num_chars=38, 
                embedding_dim=128, 
                labels=2, 
                n_layers=1):

        super(CharRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_dim = embedding_dim
        self.embedding = nn.Embedding(num_chars, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, 
                            embedding_dim, 
                            n_layers, 
                            batch_first = True)
        self.label = nn.Linear(embedding_dim, labels)

    def forward(self, x):
        this_batch = x.shape[0]
        x = self.embedding(x)   
        x, _ = self.rnn(x)
        return self.label(x[:, -1, :])


class CharLSTM(nn.Module):
    def __init__(self, 
                num_chars=38, 
                embedding_dim=128, 
                labels=2, 
                n_layers=2, 
                hidden_dim = 128):

        super(CharLSTM, self).__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(num_chars, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, 
                                hidden_dim, 
                                n_layers, 
                                dropout = 0.2, 
                                batch_first = True)
        self.label = nn.Linear(hidden_dim, labels)

    def forward(self, x):
        this_batch = x.shape[0]
        x = self.embedding(x)   
        x, _ = self.lstm(x)
        return self.label(x[:, -1, :])


class CharGRU(nn.Module):
    def __init__(self, 
                num_chars=38, 
                embedding_dim=128, 
                labels=2, 
                n_layers=2, 
                hidden_dim = 128):

        super(CharGRU, self).__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(num_chars, embedding_dim)
        self.gru = nn.GRU(embedding_dim, 
                            hidden_dim, 
                            n_layers, 
                            dropout = 0.2, 
                            batch_first = True)
        self.label = nn.Linear(hidden_dim, labels)

    def forward(self, x):
        this_batch = x.shape[0]
        x = self.embedding(x)   
        x, _ = self.gru(x)
        return self.label(x[:, -1, :])
        

class SubwordLSTM(nn.Module):
    def __init__(self, 
                num_chars=38, 
                embedding_dim=128, 
                labels=2, 
                n_layers=2):

        super(SubwordLSTM, self).__init__()
        self.n_layers = n_layers
        self.hidden_dim = embedding_dim
        self.embedding = nn.Embedding(num_chars, embedding_dim)
        self.convnet = nn.Sequential(
                                nn.Conv1d(embedding_dim, embedding_dim, 3, padding=1),
                                nn.ReLU(),
                                nn.MaxPool1d(3),
                                nn.Dropout(0.2)
                            )
        self.lstm = nn.LSTM(embedding_dim, 
                            embedding_dim, 
                            n_layers, 
                            dropout = 0.2, 
                            batch_first = True)
        self.label = nn.Linear(embedding_dim, labels)

    def forward(self, x):
        this_batch = x.shape[0]
        x = self.embedding(x)  
        x = x.permute(0, 2, 1)
        x = self.convnet(x)
        x = x.permute(0, 2, 1) 
        x, _ = self.lstm(x)
        return self.label(x[:, -1, :])


