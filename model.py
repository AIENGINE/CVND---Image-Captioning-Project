import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


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
    def __init__(self, embed_size, hidden_size, vocab_size, feature_shape, num_layers=1):
        super(DecoderRNN, self).__init__()
        
        self.dict_size = vocab_size
        self.lstm_layers = num_layers
        self.dropout_prob = 0.5
        self.hidden_dim = hidden_size
        self.embedding_dim = embed_size # output feature shape from CNN
        self.feature_embeddings = nn.Embedding(self.dict_size, self.embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, dropout=self.dropout_prob, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(self.dropout_prob)
        self.linear = nn.Linear(self.hidden_dim, self.dict_size)

        self.batch_dim = feature_shape
        # self.hidden_state = self.init_hidden(self.batch_dim)
        self.init_weights_linear_layer()

    def init_weights_linear_layer(self):
        ''' Initialize weights for fully connected layer '''
        self.feature_embeddings.weight.data.uniform_(-0.1, 0.1)
        # Set bias tensor to all zeros
        self.linear.bias.data.fill_(0)
        # FC weights as random uniform
        self.linear.weight.data.uniform_(-1, 1)

    
    def init_hidden(self, batch_dim):

        # The axes dimensions are (n_layers, batch_size, hidden_dim)
        return (torch.zeros(self.lstm_layers, batch_dim, self.hidden_dim).cuda(),
                torch.zeros(self.lstm_layers, batch_dim, self.hidden_dim).cuda())
    
    def forward(self, features, captions):
        
        # to concat image tensor 1 char has to be dropped
        captions = captions[:, :-1] 
        embeddings = self.feature_embeddings(captions)
        # print(embeddings.shape)
        hc = self.init_hidden(self.batch_dim)
        feature_embed = torch.cat((features.unsqueeze(dim=1), embeddings), dim=1)
        lstm_out, (h, c) = self.lstm(feature_embed, hc)
       
        # lstm_out, (h, c) = self.lstm(features.unsqueeze(dim=1), hc)
        # print(f"lstm_out shape: {lstm_out.shape} h and c: {self.hidden_state[0].shape}, {self.hidden_state[1].shape}")
        # lstm_out, (h, c) = self.lstm(embeddings.reshape(self.batch_dim, -1, self.embedding_dim), (h, c))
        # print(f"lstm_out shape: {lstm_out.shape} h and c: {self.hidden_state[0].shape}, {self.hidden_state[1].shape}")
       
        linear_out_drop = self.dropout(lstm_out)
        linear_out_drop = self.linear(linear_out_drop.reshape(-1, self.hidden_dim))
        x = linear_out_drop.reshape(self.batch_dim, -1, self.dict_size)

        return x


    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        print(f"input dim: {inputs.shape}")
        sent_idx = []
        hc = states
        top_word_idx = torch.zeros(1, 20)
        for idx in range(max_len):
        # while True:

            lstm_out, hc = self.lstm(inputs, hc)
            # print(f"lstm_out dim: {lstm_out.shape}, hidden shape: {hc[0].shape}, {hc[1].shape}")
            linear_out = self.linear(lstm_out.reshape(-1, self.hidden_dim))
            # tag_scores = F.log_softmax(linear_out, dim=1)
            pred, top_word_idx = torch.max(linear_out, dim=1)
            print(f"pred score: {pred}, word idx: {top_word_idx}")
            inputs = self.feature_embeddings(top_word_idx)
            inputs = inputs.unsqueeze(dim=1)
            top_word_idx = top_word_idx.squeeze().to("cpu").tolist()
            # if top_word_idx == 1:
            sent_idx.append(top_word_idx)
                # break    

        # print(f"top word idx shape: {top_word_idx.shape}")
        # top_word_idx = top_word_idx.squeeze(dim=0)
        # top_word_idx = top_word_idx.to("cpu")
        # top_word_idx = top_word_idx.tolist()

        # print(f"top word idx shape: {len(top_word_idx)}")
        print(f"len of sent idx: {len(sent_idx)}")
        return sent_idx

