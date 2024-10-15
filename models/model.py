import torch
import math
import copy
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_max_pool as gmp, global_mean_pool as gmean
from rdkit import Chem
import networkx as nx
import numpy as np
from torch_geometric.loader import DataLoader
from torch_geometric import data as DATA
from configs import *
from concurrent.futures import ThreadPoolExecutor
class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, embedding_dim)

        position = torch.arange(0., max_len).unsqueeze(1)  # [max_len, 1], 位置编码
        div_term = torch.exp(torch.arange(0., embedding_dim, 2) * -(math.log(10000.0) / embedding_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # 增加维度
        self.register_buffer('pe', pe)  # 内存中定一个常量，模型保存和加载的时候，可以写入和读出

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)  # Embedding + PositionalEncoding
        return self.dropout(x)
    
def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value, mask=None, dropout=None):  # q,k,v: [batch, h, seq_len, d_k]
    d_k = query.size(-1)  # query的维度
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # 打分机制 [batch, h, seq_len, seq_len]

    p_atten = F.softmax(scores, dim=-1)  # 对最后一个维度归一化得分, [batch, h, seq_len, seq_len]

    if dropout is not None:
        p_atten = dropout(p_atten)

    return torch.matmul(p_atten, value), p_atten  # [batch, h, seq_len, d_k]

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, embedding_dim, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert embedding_dim % h == 0

        self.d_k = embedding_dim // h  # 将 embedding_dim 分割成 h份 后的维度
        self.h = h  # h 指的是 head数量
        self.linears = clones(nn.Linear(embedding_dim, embedding_dim), 4)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):  # q,k,v: [batch, seq_len, embedding_dim]

        if mask is not None:
            mask = mask.unsqueeze(1)  # [batch, seq_len, 1]
        nbatches = query.size(0)

        # 1. Do all the linear projections(线性预测) in batch from embeddding_dim => h x d_k
        # [batch, seq_len, h, d_k] -> [batch, h, seq_len, d_k]
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]

        # 2. Apply attention on all the projected vectors in batch.
        # atten:[batch, h, seq_len, d_k], p_atten: [batch, h, seq_len, seq_len]
        attn, p_atten = attention(query, key, value, mask=mask, dropout=self.dropout)
        # get p_atten
        # res.append(p_atten.cpu().detach().numpy())

        # 3. "Concat" using a view and apply a final linear.
        # [batch, h, seq_len, d_k]->[batch, seq_len, embedding_dim]
        attn = attn.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        out=self.linears[-1](attn)
        return out

class NaiveNet(nn.Module):
    """
        CNN only
    """

    def __init__(self, input_size=None):
        super(NaiveNet, self).__init__()
        self.NaiveCNN = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=8, kernel_size=7, stride=2, padding=0),  # [bs, 8, 72]
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Conv1d(in_channels=8, out_channels=32, kernel_size=3, stride=1, padding=1),  # [bs 32 72]
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, padding=0),  # [bs 32 36]
            nn.Dropout(p=0.2),
            nn.Conv1d(in_channels=32, out_channels=input_size, kernel_size=3, stride=1, padding=1),  # [bs 128 36]
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, padding=0)  # [bs 128 18]
        )
        # self.NaiveBiLSTM = nn.LSTM(input_size=128,hidden_size=128,batch_first=True,bidirectional=True)
        in_features_1 = (input_size - 7) // 2 + 1
        in_features_2 = (in_features_1 - 2) // 2 + 1
        in_features_3 = (in_features_2 - 2) // 2 + 1
        self.Flatten = nn.Flatten()
        self.SharedFC = nn.Sequential(nn.Linear(in_features=input_size * in_features_3, out_features=input_size-Fea),
                                      nn.ReLU(),
                                      nn.Dropout()
                                      )

    def forward(self, x):
        x = self.NaiveCNN(x)
        output = self.Flatten(x)  # flatten output
        outs = self.SharedFC(output)
        return outs

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


class Model(nn.Module):
    def __init__(self, vocab_size, emb_dim, part_num, features_num,  p_drop, h, hidden_size, outputs_size):
        super(Model, self).__init__()

        #embedding
        self.word_embeddings = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.weight1 = nn.Parameter(torch.randn(emb_dim))
        self.weight2 = nn.Parameter(torch.randn(emb_dim))
        self.word_embeddings.weight.data.uniform_(-1., 1.)
        self.part_num = part_num
        self.use_gpu = torch.cuda.is_available()
        self.dropout = nn.Dropout(p=p_drop)

        self.position = PositionalEncoding(emb_dim, p_drop)
        self.atten = MultiHeadedAttention(h, emb_dim)  # self-attention-->建立一个全连接的网络结构
        self.norm = nn.LayerNorm(emb_dim)
        self.first_linear = nn.Linear(emb_dim, hidden_size)
        self.second_linear = nn.Linear(hidden_size*2, hidden_size)
        self.cnn = NaiveNet(input_size=hidden_size + features_num)
        self.init_weights()
        self.hidden2label = nn.Sequential(
            # nn.Linear(64 , 32),
            nn.Linear(hidden_size, outputs_size)
        )
        self.num_task = outputs_size
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.avg_pool2 = nn.AdaptiveAvgPool1d(0)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
    def init_weights(self):
        init_range = 0.1
        self.first_linear.bias.data.zero_()
        self.first_linear.weight.data.uniform_(-init_range, init_range)
        self.second_linear.bias.data.zero_()
        self.second_linear.weight.data.uniform_(-init_range, init_range)


    def subsequence_embedding(self, inputs):  # torch.Size([batch, part_num, part_len-3+1])
        outputs = []
        for part_idx in range(inputs.shape[1]):
            output = self.word_embeddings(inputs[:, part_idx, :])  # torch.Size([batch, part_len-3+1, emb_dim])
            output = torch.transpose(output, dim0=2, dim1=1)  # torch.Size([batch, emb_dim, part_len-3+1])
            output = self.avg_pool(output)  # torch.Size([batch, emb_dim, 1])
            outputs.append(output)
        outputs = torch.cat(outputs, dim=2)  # torch.Size([batch, emb_dim, part_num])

        return outputs
    
    def connect_embedding(self, inputs0, inputs1, inputs2):
        embed0 = self.subsequence_embedding(inputs0)
        embed1 = self.subsequence_embedding(inputs1)
        embed2 = self.subsequence_embedding(inputs2)
        return embed0, embed1, embed2

    def average(self, embed0, embed1, embed2):
        result = []
        for i in range(embed0.shape[0]):
            a = embed0[i,:,:].unsqueeze(2)
            b = embed1[i,:,:].unsqueeze(2)
            c = embed2[i,:,:].unsqueeze(2)
            d = torch.cat((a,b),axis=2)
            e = torch.cat((d,c),axis=2)
            final = self.max_pool(e)
            result.append(final)
        result = torch.cat(result,dim=2)
        result = torch.transpose(result,0,2)
        result = torch.transpose(result,1,2)
        return result

    def attention(self, lstm_out, final_state):
        hidden = final_state.view(-1, self.hidden_dim, 1)
        attn_weights = torch.bmm(lstm_out, hidden).squeeze(2)  # torch.Size([batch, part_num])
        soft_attn_weights = F.softmax(attn_weights, 1)
        context = torch.bmm(lstm_out.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)  # torch.Size([batch, hidden_dim])
        return context

    def forward(self, inputs, features):# torch.Size([batch, part_num, part_len-3+1]), torch.Size([batch, f_num])
        #embeddings
        inputs0, inputs1, inputs2 = inputs
        embed0, embed1, embed2 = self.connect_embedding(inputs0, inputs1, inputs2)
        embed = self.average(embed0, embed1, embed2)  #torch.size[128, 128, 64]
        embed = torch.transpose(embed,1,2)   #torch.size[128, 64, 128]

        #Transformer
        embeded = self.position(embed)  # 2. PosionalEncoding [batch, seq_len, embedding_dim]
        inp_atten = self.atten(embeded, embeded, embeded)   #torch.size[128, 64, 128]
        inp_atten = self.norm(inp_atten + embeded)  #torch.size[128, 64, 128]
        inp_atten= self.norm(inp_atten)  #torch.size[128, 64, 128]
        b_avg = inp_atten.sum(1) / (embeded.shape[1] + 1e-5)  # [batch, embedding_dim]
        trans_out = self.first_linear(b_avg).squeeze() #[batch, hidden_size]
        if len(trans_out.shape) == 1:
            trans_out = trans_out.unsqueeze(0)
        #cnn
        logits_feature = torch.cat([trans_out, features], dim=-1)     # torch.Size([batch, hidden_size+features_dim])
        cnn_in = logits_feature.unsqueeze(1)  #[batch, 1, hidden_size]
        cnn_out = self.cnn(cnn_in)   #[batch, 256]
        trans_cnn_output = torch.cat((trans_out,cnn_out),-1)    #[batch, 256+hidden_size]
        model_output = self.second_linear(trans_cnn_output) #[batch, 256]

        #add features
        # logits_feature = torch.cat([model_output, features], dim=1)     # torch.Size([batch, 256+features_dim])
        logits = self.hidden2label(model_output)  # torch.Size([batch, output_size])
        return  logits
