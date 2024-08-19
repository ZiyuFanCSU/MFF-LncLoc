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
        self.SharedFC = nn.Sequential(nn.Linear(in_features=input_size * in_features_3, out_features=input_size-15),
                                      nn.ReLU(),
                                      nn.Dropout()
                                      )

    def forward(self, x):
        x = self.NaiveCNN(x)
        output = self.Flatten(x)  # flatten output
        outs = self.SharedFC(output)
        return outs

def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)

    c_size = mol.GetNumAtoms()

    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])

    return c_size, features, edge_index

class GCNNet(torch.nn.Module):
    def __init__(self, n_output=2,num_features_xd=78, dropout=0.2,emb_dim = 32):

        super(GCNNet, self).__init__()
          
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        # SMILES1 graph branch
        self.n_output = n_output
        self.drug1_conv1 = GCNConv(num_features_xd, num_features_xd*2)
        # self.drug1_conv2 = GCNConv(num_features_xd, num_features_xd*2)
        # self.drug1_conv3 = GCNConv(num_features_xd*2, num_features_xd * 4)
        self.drug1_fc_g1 = torch.nn.Linear(num_features_xd*2, emb_dim)
        self.final = torch.nn.Linear(emb_dim + num_features_xd*2, emb_dim)
  

    def forward(self,input):
        SMILES_ATCG = { "Adenosine":"C1=NC(=C2C(=N1)N(C=N2)C3C(C(C(O3)CO)O)O)N",
                        "Thymidine":"CC1=CN(C(=O)NC1=O)C2CC(C(O2)CO)O",
                        "Cytidine":"C1=CN(C(=O)N=C1N)C2C(C(C(O2)CO)O)O",
                        "Guanosine":"C1=NC2=C(N1C3C(C(C(O3)CO)O)O)N=C(NC2=O)N"} 
        smile_graph = {}
        for smile in ["Adenosine","Thymidine","Cytidine","Guanosine"]:
            g = smile_to_graph(SMILES_ATCG[smile])
            smile_graph[smile] = g 
        smiles = ["Adenosine","Thymidine","Cytidine","Guanosine"]
        features_list = []
        edge_index_list = []
        batch = []
        batchLL = []

        for smile in smiles:
            g = smile_graph[smile]
            batchLL.append(g[0])
            for i in g[1]:
                features_list.append(i)
            for j in g[2]:
                edge_index_list.append(j)
        
        for _ in range(int(batchLL[0])):
            batch.append(0)
        for _ in range(int(batchLL[1])):
            batch.append(1)
        for _ in range(int(batchLL[2])):
            batch.append(2)
        for _ in range(int(batchLL[3])):
            batch.append(3)

        GCNData = DATA.Data(x=torch.Tensor(features_list),
                            edge_index= torch.LongTensor(edge_index_list).transpose(1, 0),batch = torch.LongTensor(batch))
        # data1 = DataLoader(GCNData, batch_size=218, shuffle=None)

        # for batch_data in data1:
        x1, edge_index1, batch1 = GCNData.x.to(input.device), GCNData.edge_index.to(input.device), GCNData.batch.to(input.device)
        x1 = self.drug1_conv1(x1, edge_index1)
        # x1 = self.drug1_conv2(x1, edge_index1)
        # x1 = self.drug1_conv3(x1, edge_index1)
        x1 = self.dropout(self.relu(x1))
        x1 = gmp(x1, batch1)      
        x1 = self.relu(self.drug1_fc_g1(x1))

        return x1


class Model(nn.Module):
    def __init__(self, vocab_size, emb_dim, part_num, features_num,  p_drop, h, hidden_size, outputs_size):
        super(Model, self).__init__()

        #embedding
        self.word_embeddings2 = GCNNet(emb_dim = emb_dim)
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
    
    def smiles_embeddings(self,input):
        smi_embedding = self.word_embeddings2(input)
        SMI_em = {}
        SMI_em['A'] = smi_embedding[0]
        SMI_em['T'] = smi_embedding[1]
        SMI_em['C'] = smi_embedding[2]
        SMI_em['G'] = smi_embedding[3]
        import json
        file_path = VOCAB_PATH0
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        swapped_dict = {value: key for key, value in data.items()}
        sd = []

        for i in swapped_dict.keys():
            v = []
            if i != 0:
                for j in swapped_dict[i]:
                    v.append(SMI_em[j])
                sd.append(torch.mean(torch.cat(v, dim=0).view(TOKEN_LEN,PART_NUM), dim=0))
            else:
                sd.append(torch.randn(PART_NUM, requires_grad=True, device=smi_embedding.device))
        sd = torch.stack(sd)
        smi_embed = []  

        for num_1 in input:
            temp_1 = []
            for num_2 in num_1:
                temp_2 = [sd[int(num_3)] for num_3 in num_2]
                temp_1.append(torch.stack(temp_2, dim=0))
            smi_embed.append(torch.stack(temp_1, dim=0))
        smi_embed = torch.stack(smi_embed, dim=0)

        return smi_embed


    def subsequence_embedding(self, inputs):  # torch.Size([batch, part_num, part_len-3+1])
        outputs = []
        # output_smiles = torch.mean(self.smiles_embeddings(inputs), dim=2)
        for part_idx in range(inputs.shape[1]):
            # output = F.one_hot(inputs[:, part_idx, :], num_classes=65).float()
            output = self.word_embeddings(inputs[:, part_idx, :])  # torch.Size([batch, part_len-3+1, emb_dim])
            output = torch.transpose(output, dim0=2, dim1=1)  # torch.Size([batch, emb_dim, part_len-3+1])
            # output = self.cnn_layer(output)
            output = self.avg_pool(output)  # torch.Size([batch, emb_dim, 1])
            outputs.append(output)
        outputs = torch.cat(outputs, dim=2)  # torch.Size([batch, emb_dim, part_num])
        # weighted_tensor1 = output_smiles * self.weight1.view(1, PART_NUM, 1)
        # weighted_tensor2 = outputs * self.weight2.view(1, PART_NUM, 1)
        # output_tensor = weighted_tensor1 + weighted_tensor2
        
        return outputs
    
    def connect_embedding(self, inputs0, inputs1, inputs2):
        embed0 = self.subsequence_embedding(inputs0)
        embed1 = self.subsequence_embedding(inputs1)
        embed2 = self.subsequence_embedding(inputs2)
        return embed0, embed1, embed2

    # def connect_embedding(self, inputs0, inputs1, inputs2):
    #     with ThreadPoolExecutor() as executor:
    #         future0 = executor.submit(self.subsequence_embedding, inputs0)
    #         future1 = executor.submit(self.subsequence_embedding, inputs1)
    #         future2 = executor.submit(self.subsequence_embedding, inputs2)
            
    #         embed0 = future0.result()
    #         embed1 = future1.result()
    #         embed2 = future2.result()
            
    #     return embed0, embed1, embed2


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
