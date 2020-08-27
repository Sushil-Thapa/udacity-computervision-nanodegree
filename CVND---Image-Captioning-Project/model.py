import torch
import torch.nn as nn
import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size,
                            hidden_size,
                            num_layers,
                            batch_first= True,
                            dropout = 0)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def init_hidden(self, batch_size):
        """ Initialize a hidden state
        Dimensions: (num_layers, batch_size, hidden_dim)
        """
        return (torch.zeros((1, batch_size, self.hidden_size), device=device), \
                torch.zeros((1, batch_size, self.hidden_size), device=device))
    
    def forward(self, features, captions):
        self.hidden = self.init_hidden(features.shape[0])

        fx = features.unsqueeze(1)
        cx = self.embed(captions[:,:-1])

        x = torch.cat((fx, cx), dim=1)

        x, _ = self.lstm(x, self.hidden)
        x = self.fc(x)
        return x


    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        output_list, sent_len, index = [] , 0, None
        hidden = self.init_hidden(inputs.shape[0])  

        with torch.no_grad():
            while sent_len < max_len:
                out, hidden = self.lstm(inputs, hidden)
                out = self.fc(out)

                out = out.squeeze(1)
                
                out = out.argmax(dim=1)
                output_list.append(out.item())

                inputs = self.embed(out.unsqueeze(0))

                sent_len += 1
                if index == 1:
                    break
                
        return output_list