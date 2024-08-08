import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, GINConv, GATConv, global_add_pool
from torch_geometric.nn.inits import uniform

from utils import obtain_act, obtain_norm, infomax_corruption, sce_loss
from transformers import BertModel, BertTokenizer, RobertaTokenizer, RobertaModel, AutoTokenizer, AutoModel, AutoModelForMaskedLM,RobertaTokenizer, RobertaModel, T5Tokenizer, T5Model,T5EncoderModel
from sentence_transformers import SentenceTransformer, util


class LinearPred(nn.Module):

    def __init__(self, in_dim, emb_dim, out_dim, num_layer, act='relu'):
        super().__init__()

        self.linears = nn.ModuleList()
        self.acts = nn.ModuleList()
        self.num_layer = num_layer

        for i in range(self.num_layer):
            if i == 0:
                self.linears.append(nn.Linear(in_dim, emb_dim))
            elif i == self.num_layer - 1:
                self.linears.append(nn.Linear(emb_dim, out_dim))
            else:
                self.linears.append(nn.Linear(emb_dim, emb_dim))
            if i != self.num_layer - 1:
                self.acts.append(obtain_act(act))
            else:
                self.acts.append(obtain_act(None))

            # Initialize parameters
            nn.init.xavier_uniform_(self.linears[-1].weight)
            if self.linears[-1].bias is not None:
                self.linears[-1].bias.data.fill_(0.0)

    def forward(self, x):
        
        for i in range(self.num_layer):
            x = self.acts[i](self.linears[i](x))
        
        return x


class Encoder(nn.Module):

    def __init__(self, in_dim, emb_dim, num_layer, kernel='gcn', drop_ratio=0,
                 act='relu', norm='batchnorm', concat=True, last_act=True,aggr='mean'):
        super().__init__()

        self.num_layer = num_layer
        self.emb_dim = [in_dim] + [emb_dim] * num_layer
        # just try
        # self.emb_dim = [in_dim] + [emb_dim] * (num_layer - 1) + [768]

        self.drop_ratio = drop_ratio
        self.norm = norm
        self.concat = concat

        self.encs = torch.nn.ModuleList()
        self.acts = torch.nn.ModuleList()
        if self.norm:
            self.norms = torch.nn.ModuleList()

        for i in range(self.num_layer):
            if kernel == 'gcn':
                conv = GCNConv(self.emb_dim[i], self.emb_dim[i + 1], normalize=True, add_self_loops=True)
            elif kernel == 'gin':
                conv = GINConv(LinearPred(self.emb_dim[i], self.emb_dim[i + 1], self.emb_dim[i + 1], 1))
            elif kernel == 'gin2':
                conv = GINConv(LinearPred(self.emb_dim[i], self.emb_dim[i + 1], self.emb_dim[i + 1], 2))
            self.encs.append(conv)
            if i == self.num_layer - 1 and not last_act:
                act = None
            self.acts.append(obtain_act(act))

            if self.norm:
                self.norms.append(obtain_norm(self.norm)(self.emb_dim[i + 1]))
        
    def forward(self, x, edge_index, edge_weight=None, batch=None):

        xs = []
        for i in range(self.num_layer):
            x = self.encs[i](x, edge_index, edge_weight)
            x = self.norms[i](x) if self.norm else x
            x = F.dropout(self.acts[i](x), self.drop_ratio, training=self.training)
            xs.append(x)
        
        if batch is not None:
            xs = [global_add_pool(x, batch) for x in xs]
        
        if self.concat:
            x = torch.concat(xs, dim=1)
        else:
            x = xs[-1]

        return x


class GraphCL(nn.Module):

    def __init__(self, in_dim, emb_dim, num_layer, kernel='gcn', drop_ratio=0,
                 act='relu', norm=None):
        super().__init__()

        self.lin_emb_dim = emb_dim * num_layer
        self.encoder = Encoder(in_dim, emb_dim, num_layer, kernel, drop_ratio, act, norm)
        self.proj_head = LinearPred(self.lin_emb_dim, emb_dim, emb_dim, 2)
    

    def forward(self, x, edge_index, edge_weigt=None, batch=None):

        h = self.encoder(x, edge_index, edge_weigt, batch)
        h = self.proj_head(h)

        return h
    
    
    def get_loss(self, x, x_aug, batch_size):

        T = 0.2
        n_samples, _ = x.size()
        x_abs = x.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1)
        if n_samples <= batch_size:
            batch_size = n_samples
            sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
            sim_matrix = torch.exp(sim_matrix / T)
            pos_sim = sim_matrix[range(batch_size), range(batch_size)]
            loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
            loss = - torch.log(loss).mean()
        else:
            n_loop = n_samples // batch_size + 1
            losses = []
            for i in range(n_loop):
                start = i*batch_size
                end = (i + 1)*batch_size if i != n_loop - 1 else n_samples
                n_sim = batch_size if i != n_loop - 1 else end - start
                sim_matrix = torch.einsum('ik,jk->ij', x[start:end], x_aug) / torch.einsum('i,j->ij', x_abs[start:end], x_aug_abs)
                sim_matrix = torch.exp(sim_matrix / T)
                pos_sim = sim_matrix[range(n_sim), range(n_sim)]
                loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
                losses.append(-torch.log(loss))
            
            loss = torch.concat(losses).mean()

        return loss
    

class GraphInfoMax(nn.Module):

    EPS = 1e-15

    def __init__(self, in_dim, emb_dim, num_layer, kernel='gcn', drop_ratio=0,
                 act='relu', norm=None):
        super().__init__()

        self.emd_dim = emb_dim

        self.encoder = Encoder(in_dim, emb_dim, num_layer, kernel, drop_ratio, act, norm, concat=False, last_act=False)
        # self.pred_head = LinearPred(emb_dim, emb_dim, 2, 2)

        self.weight = nn.Parameter(torch.empty(emb_dim, emb_dim))
        # just try
        uniform(self.emd_dim, self.weight)
        # self.weight = nn.Parameter(torch.empty(768, 768))
        # uniform(768, self.weight)
        

    def forward(self, x, edge_index, edge_weigt=None, batch=None):

        pos_h = self.encoder(x, edge_index, edge_weigt, batch)

        x_cor = infomax_corruption(x, batch)
        neg_h = self.encoder(x_cor, edge_index, edge_weigt, batch)

        summary = torch.sigmoid(pos_h.mean(dim=0))

        return pos_h, neg_h, summary
    
    def discriminate(self, h, summary):

        summary = summary.t() if summary.dim() > 1 else summary
        value = torch.matmul(h, torch.matmul(self.weight, summary))
        return torch.sigmoid(value)
    
    def get_loss(self, pos_h, neg_h, summary):

        pos_loss = -torch.log(self.discriminate(pos_h, summary) + self.EPS).mean()
        neg_loss = -torch.log(1 - self.discriminate(neg_h, summary) + self.EPS).mean()

        return pos_loss + neg_loss
    

    # def predict(self, x, edge_index, edge_weigt=None, batch=None):
        
    #     h, _, _ = self.forward(x, edge_index, edge_weigt, batch)
    #     pred = F.softmax(self.pred_head(h), dim=-1)

    #     return pred

class GraphMAE(nn.Module):

    EPS = 1e-15

    def __init__(self, in_dim, emb_dim, num_layer, kernel='gcn', drop_ratio=0,
                 act='relu', norm=None, concat=False, mask_ratio=0.5, replace_ratio=0):
        super().__init__()

        self.emb_dim = emb_dim if not concat else num_layer * emb_dim
        self.mask_ratio = mask_ratio
        self.replace_ratio = replace_ratio

        self.encoder = Encoder(in_dim, emb_dim, num_layer, kernel, drop_ratio, act, norm, 
                               concat=False, last_act=True, aggr='mean')
        self.decoder = Encoder(emb_dim, in_dim, 1, kernel, drop_ratio, act, norm=None, 
                               concat=False, last_act=False, aggr='mean')

        self.encoder_mask_token = nn.Parameter(torch.zeros(1, in_dim))
        self.encoder_to_decoder = nn.Linear(self.emb_dim, emb_dim, bias=False)


    def forward(self, x, edge_index, edge_weigt=None, batch=None):
        # Mask
        mask_x, mask_nodes = self.encding_mask(x)
        h = self.encoder(mask_x, edge_index, edge_weigt, batch)
        h = self.encoder_to_decoder(h)

        # Re-mask
        h[mask_nodes] = 0
        recon = self.decoder(h, edge_index, edge_weigt, batch)

        return x[mask_nodes], recon[mask_nodes]


    def encding_mask(self, x):

        num_nodes = x.shape[0]

        # random masking
        perm = torch.randperm(num_nodes, device=x.device)
        num_mask_nodes = int(self.mask_ratio * num_nodes)
        mask_nodes = perm[: num_mask_nodes]
        # keep_nodes = perm[num_mask_nodes: ]

        if self.replace_ratio > 0:
            num_noise_nodes = int(self.replace_ratio * num_mask_nodes)
            perm_mask = torch.randperm(num_mask_nodes, device=x.device)
            token_nodes = mask_nodes[perm_mask[: int((1 - self.replace_ratio) * num_mask_nodes)]]
            noise_nodes = mask_nodes[perm_mask[-int(self.replace_ratio * num_mask_nodes):]]
            noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]

            out_x = x.clone()
            out_x[token_nodes] = 0.0
            out_x[noise_nodes] = x[noise_to_be_chosen]
        else:
            out_x = x.clone()
            token_nodes = mask_nodes
            out_x[mask_nodes] = 0.0

        out_x[token_nodes] += self.encoder_mask_token

        return out_x, mask_nodes
    

    def get_loss(self, x, edge_index, edge_weigt=None, batch=None):

        mask_x, recon_x = self.forward(x, edge_index, edge_weigt, batch)
        loss = sce_loss(mask_x, recon_x, alpha=2)

        return loss
    
    def embed(self, x, edge_index, edge_weigt=None, batch=None):

        return self.encoder(x, edge_index, edge_weigt, batch)

class TextModel(nn.Module):
    def __init__(self, encoder):
        super(TextModel, self).__init__()
        self.encoder = encoder
        if self.encoder == 'Bert' or  self.encoder == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.textmodel = BertModel.from_pretrained('bert-base-uncased')


        if self.encoder == 'Roberta' or  self.encoder == 'roberta' :
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
            self.textmodel = RobertaModel.from_pretrained('roberta-base')
        if self.encoder == 'SentenceBert':
            self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/multi-qa-distilbert-cos-v1")
            self.textmodel = AutoModel.from_pretrained("sentence-transformers/multi-qa-distilbert-cos-v1")
        if self.encoder == 'SimCSE':
            self.tokenizer = AutoTokenizer.from_pretrained('princeton-nlp/sup-simcse-bert-base-uncased')
            self.textmodel = AutoModel.from_pretrained('princeton-nlp/sup-simcse-bert-base-uncased')
        if self.encoder == 'e5':
            self.tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-base-v2')
            self.textmodel = AutoModel.from_pretrained('intfloat/e5-base-v2')
        if self.encoder  == 't5':
            self.tokenizer = T5Tokenizer.from_pretrained("t5-large")
            self.textmodel = T5EncoderModel.from_pretrained("t5-large")
       

    
    def forward(self, input):        
        inputs = self.tokenizer(input, return_tensors='pt', truncation=True, padding=True).to(self.textmodel.device)

        with torch.no_grad():
            outputs = self.textmodel(**inputs)

        text_embedding = outputs[0][:,0,:].squeeze()
        return text_embedding
