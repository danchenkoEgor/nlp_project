import numpy as np 
from joblib import load
import torch
from torch import nn

def padding(review_int: list, seq_len: int) -> np.array:
    features = np.zeros((len(review_int), seq_len), dtype = int)
    for i, review in enumerate(review_int):
        if len(review) <= seq_len:
            zeros = list(np.zeros(seq_len - len(review)))
            new = zeros + review
        else:
            new = review[: seq_len]
        features[i, :] = np.array(new)
            
    return features

device = torch.device('cpu')
def func(sentence):

    EMBEDDING_DIM = 64
    HIDDEN_DIM = 64
    N_LAYERS = 2
    SEQ_LEN = 256

    vocab_to_int = load('models/vocab.joblib')
    VOCAB_SIZE = len(vocab_to_int)+1 

    reviews_int = [vocab_to_int[word] for word in sentence.split() if vocab_to_int.get(word)]

    features = padding([reviews_int], SEQ_LEN)
    features = torch.tensor(features)

    model = RNNNet(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, SEQ_LEN, N_LAYERS)
    model.load_state_dict(torch.load('models/rnn_model_epoch_2.pt', map_location=device))
    model.eval()
    output = model(features)

    sigmoid = nn.Sigmoid()

    return sigmoid(output).detach().numpy()


class RNNNet(nn.Module):    
    '''
    vocab_size: int, размер словаря (аргумент embedding-слоя)
    emb_size:   int, размер вектора для описания каждого элемента последовательности
    hidden_dim: int, размер вектора скрытого состояния
    batch_size: int, размер batch'а

    '''
    
    def __init__(self, 
                 vocab_size: int, 
                 emb_size: int, 
                 hidden_dim: int, 
                 seq_len: int, 
                 n_layers: int = 1) -> None:
        super().__init__()
        
        self.seq_len  = seq_len 
        self.emb_size = emb_size 
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(vocab_size, self.emb_size)
        self.rnn_cell  = nn.RNN(self.emb_size, self.hidden_dim, batch_first=True, num_layers=n_layers, bidirectional=True)
        self.linear    = nn.Sequential(
            nn.Dropout(),
            nn.Linear(self.hidden_dim * self.seq_len*2, 256),
            nn.Dropout(),
            nn.Linear(256, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.embedding(x.to(device))
        output, _ = self.rnn_cell(x)

        output = output.contiguous().view(output.size(0), -1)
        out = self.linear(output.squeeze(0))
        return out