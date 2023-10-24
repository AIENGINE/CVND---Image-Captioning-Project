import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, captions_shape, num_layers=1):
        super(DecoderRNN, self).__init__()
        
        self.dict_size = vocab_size
        self.lstm_layers = num_layers
        self.dropout_prob = 0.5
        self.hidden_dim = hidden_size
        self.embedding_dim = embed_size # output feature shape from CNN
        self.feature_embeddings = nn.Embedding(self.dict_size, self.embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, num_layers=num_layers, dropout=self.dropout_prob, batch_first=True)
        # self.dropout = nn.Dropout(self.dropout_prob)
        self.linear = nn.Linear(self.hidden_dim, self.dict_size)

        self.batch_dim = captions_shape[0]
        self.hidden_state = self.init_hidden(self.batch_dim)
    
    def init_hidden(self, batch_dim):

        # The axes dimensions are (n_layers, batch_size, hidden_dim)
        return (torch.zeros(self.lstm_layers, batch_dim, self.hidden_dim),
                torch.zeros(self.lstm_layers, batch_dim, self.hidden_dim))
    
    def forward(self, features, captions):
        embeddings = self.feature_embeddings(captions)
        print(embeddings.shape)
        
        lstm_out, self.hidden_state = self.lstm(features.unsqueeze(dim=1), self.hidden_state)
        print(f"lstm_out shape: {lstm_out.shape} h and c: {self.hidden_state[0].shape}, {self.hidden_state[1].shape}")
        lstm_out, self.hidden_state = self.lstm(embeddings.reshape(self.batch_dim, -1, self.embedding_dim), self.hidden_state)
        print(f"lstm_out shape: {lstm_out.shape} h and c: {self.hidden_state[0].shape}, {self.hidden_state[1].shape}")
        
        linear_out = self.linear(lstm_out.reshape(-1, self.hidden_dim))

        return linear_out.reshape(self.batch_dim, -1, self.dict_size)
        



    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        pass