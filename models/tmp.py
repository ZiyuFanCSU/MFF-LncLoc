import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, vocab_size, emb_dim, part_num, features_num, outputs_size):
        super(Model, self).__init__()

        self.hidden_dim = 128
        self.batch_size = 128
        self.use_gpu = torch.cuda.is_available()

        self.word_embeddings = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.word_embeddings.weight.data.uniform_(-1., 1.)

        self.num_layers = 1
        self.dropout = 0.3
        self.bilstm = nn.LSTM(emb_dim, self.hidden_dim // 2, batch_first=True, num_layers=self.num_layers,
                              dropout=self.dropout, bidirectional=True)

        self.hidden2label = nn.Sequential(
            nn.Linear(16 + features_num, 64),
            nn.Linear(64, outputs_size)
        )
        self.hidden = self.init_hidden()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.Linear = nn.Linear(self.hidden_dim, 16)

    def subsequence_embedding(self, inputs):  # torch.Size([batch, part_num, part_len-3+1])
        outputs = []
        for part_idx in range(inputs.shape[1]):
            output = self.word_embeddings(inputs[:, part_idx, :])  # torch.Size([batch, part_len-3+1, emb_dim])
            output = torch.transpose(output, dim0=2, dim1=1)  # torch.Size([batch, emb_dim, part_len-3+1])
            output = self.avg_pool(output)  # torch.Size([batch, emb_dim, 1])
            outputs.append(output)
        outputs = torch.cat(outputs, dim=2)  # torch.Size([batch, emb_dim, part_num])
        return outputs

    def init_hidden(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        if self.use_gpu:
            hidden_state = Variable(torch.zeros(2 * self.num_layers, batch_size, self.hidden_dim // 2).cuda())
            cell_state = Variable(torch.zeros(2 * self.num_layers, batch_size, self.hidden_dim // 2).cuda())
        else:
            hidden_state = Variable(torch.zeros(2 * self.num_layers, batch_size, self.hidden_dim // 2))
            cell_state = Variable(torch.zeros(2 * self.num_layers, batch_size, self.hidden_dim // 2))
        return (hidden_state, cell_state)

    def attention(self, lstm_out, final_state):
        hidden = final_state.view(-1, self.hidden_dim, 1)
        attn_weights = torch.bmm(lstm_out, hidden).squeeze(2)  # torch.Size([batch, part_num])
        soft_attn_weights = F.softmax(attn_weights, 1)
        context = torch.bmm(lstm_out.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(
            2)  # torch.Size([batch, hidden_dim])
        return context


    def forward(self, inputs, features):  # torch.Size([batch, part_num, part_len-3+1]), torch.Size([batch, f_num])
        outputs = self.subsequence_embedding(inputs)  # torch.Size([batch, emb_dim, part_num])
        outputs = torch.transpose(outputs, dim0=2, dim1=1)  # torch.Size([batch, part_num, emb_dim])
        hidden = self.init_hidden(outputs.size()[0])
        lstm_out, hidden = self.bilstm(outputs, hidden)  # torch.Size([batch, part_num, emb_dim])
        final_hidden_state, final_cell_state = hidden  # h_n, c_n: torch.Size([2*num_layer, batch, hidden_dim//2])
        attn_out = self.attention(lstm_out, final_hidden_state)  # torch.Size([batch, hidden_dim])
        # linear_out = self.Linear(attn_out)
        # logits_feature = torch.cat([linear_out, features], dim=1)     # torch.Size([batch, hidden_dim+features_dim])
        # logits = self.hidden2label(logits_feature)  # torch.Size([hidden_dim+features_dim, output_size])
        return attn_out

