import numpy
import torch
import torch.nn as nn
from dgl import DGLError
from dgl import function as fn
from dgl.utils import expand_as_pair
import torch.nn.functional as Fn


class EncoderRoGCN(nn.Module):
    def __init__(self, in_features, embedding_range, hidden_layer=2,
                 activation=None,
                 self_loop=True, ent_drop=0, rel_drop=0):
        super().__init__()
        assert hidden_layer >= 1
        self.activation = activation
        self.self_loop = self_loop
        self.layers = nn.ModuleList()
        if hidden_layer == 1:
            self.layers.append(EncoderRGCNLayer(norm='both', ent_drop=ent_drop, rel_drop=rel_drop,
                                                embedding_range=embedding_range, activation=activation,
                                                self_loop=self_loop,
                                                h_dim=in_features))
        # layer one
        else:
            self.layers.append(EncoderRGCNLayer(norm='both', ent_drop=ent_drop, rel_drop=rel_drop,
                                                embedding_range=embedding_range, activation=activation,
                                                self_loop=self_loop,
                                                h_dim=in_features))
            if hidden_layer > 2:
                for _ in range(hidden_layer - 2):
                    self.layers.append(
                        EncoderRGCNLayer(norm='both', ent_drop=ent_drop, rel_drop=rel_drop,
                                         embedding_range=embedding_range, activation=activation, self_loop=self_loop,
                                         h_dim=in_features
                                         ))
            self.layers.append(EncoderRGCNLayer(norm='both', ent_drop=ent_drop, rel_drop=rel_drop,
                                                embedding_range=embedding_range, activation=activation,
                                                self_loop=self_loop,
                                                h_dim=in_features
                                                ))

    # forward for embedding
    # block 为不同层gcn的采样
    # when mode is not sample. means whole graph will pass here
    def forward(self, blocks, x, weight, mode='sample'):
        if mode == 'sample':
            for indice, layer in enumerate(self.layers):
                x = layer(blocks[indice], x, weight)
        else:
            for layer in self.layers:
                x = layer(blocks, x, weight, train=False)
        return x


# conv layer
class EncoderRGCNLayer(nn.Module):
    def __init__(self,
                 h_dim,
                 embedding_range=None,
                 norm='none',
                 activation=None,
                 allow_zero_in_degree=False,
                 attention_mechanism=False,
                 self_loop=True,
                 ent_drop=0,
                 rel_drop=0):

        super().__init__()

        if norm not in ('none', 'both', 'right'):
            raise DGLError('Invalid norm value. Must be either "none", "both" or "right".'
                           ' But got "{}".'.format(norm))
        self._norm = norm

        self.self_loop = self_loop

        self._allow_zero_in_degree = allow_zero_in_degree

        self.pi = 3.14159265358979323846

        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(h_dim, h_dim))
            nn.init.xavier_uniform_(self.loop_weight,
                                    gain=nn.init.calculate_gain('relu'))

        if embedding_range is None:
            self.register_parameter('embedding_range', None)
        else:
            self.register_parameter('embedding_range', embedding_range)

        self.activation = activation

        self.ent_dropout = nn.Dropout(p=ent_drop)
        self.rel_dropout = nn.Dropout(p=rel_drop)
        # TODO add attention aggregation
        self.attention_mechanism = attention_mechanism

    def forward(self, graph, feat, weight):
        with graph.local_scope():
            feat_src, feat_dst = expand_as_pair(feat, graph)
            graph.srcdata['h'] = feat_src
            graph.dstdata['h'] = feat_dst
            if self._norm == 'both':
                degs = graph.out_degrees().float().clamp(min=1)
                norm = torch.pow(degs, -0.5)
                shp = norm.shape + (1,) * (feat_src.dim() - 1)
                norm = torch.reshape(norm, shp)
                feat_src = feat_src * norm
            rel_weight_feature = weight[graph.edata['etype']]  # get rel weight
            graph.edata['r'] = self.rel_dropout(rel_weight_feature)
            # gcn
            graph.srcdata['h'] = feat_src
            graph.apply_edges(fn.u_mul_e('h', 'r', 'm'))
            graph.update_all(fn.copy_e('m', 'm'),
                             fn.mean(msg='m', out='mix'))
            rst = graph.dstdata['mix']
            # self_loop
            self_loop_h = torch.matmul(graph.dstdata['h'], self.loop_weight)
            rst = self_loop_h + rst
            # need to check if norm is needed
            if self._norm != 'none':
                degs = graph.in_degrees().float().clamp(min=1)
                if self._norm == 'both':
                    norm = torch.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (feat_dst.dim() - 1)
                norm = torch.reshape(norm, shp)
                rst = rst * norm
            self.ent_dropout(rst)
            if self.activation is not None:
                rst = self.activation(rst)
            return rst


class ScorePredictor(nn.Module):
    def __init__(self, dist_func, sw=None):
        super(ScorePredictor, self).__init__()
        self.dist_func = dist_func
        self.sw = sw

    def forward(self, edge_subgraph, x):
        with edge_subgraph.local_scope():
            edge_subgraph.ndata['x'] = x
            edge_subgraph.apply_edges(self.edge_func)
            return edge_subgraph.edata['score']

    def edge_func(self, edges):
        head = edges.src['x']
        tail = edges.dst['x']
        score = self.dist_func(head, tail, mode='single')
        if self.sw is not None:
            sw = edges.data[self.sw]
            score = score * sw
        return {'score': score}


def dist_func(head, tail):
    score = head * tail
    score = score.norm(dim=0)
    return score


class GCN(nn.Module):
    def __init__(self, whole_graph, nentity, nrelation, hidden_dim, graph_layer_num=2,
                 ent_ini=None, ent_drop=0, rel_drop=0, self_loop=True, graph_activation=None, freeze=False):
        super(GCN, self).__init__()
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.graph_layer_num = graph_layer_num

        self.whole_graph = whole_graph

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False
        )
        # entity embedding init :
        if ent_ini is None:
            self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))
            nn.init.xavier_uniform_(
                tensor=self.entity_embedding,
                gain=graph_activation
            )
            self.entity_dim = hidden_dim
            self.relation_dim = hidden_dim
        else:
            embed_li = []
            hidden_dim = len(ent_ini[0])
            for i in range(whole_graph.number_of_nodes()):
                embed_li.append(ent_ini[i])
            embed_li = numpy.asarray(embed_li)
            embed_li = torch.from_numpy(embed_li).float()
            self.entity_dim = hidden_dim
            self.relation_dim = hidden_dim
            self.ent_ini = nn.Embedding.from_pretrained(embed_li, freeze=freeze)
        self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
        nn.init.xavier_uniform_(
            tensor=self.relation_embedding,
            gain=graph_activation
        )
        if graph_activation == 'relu':
            act_fun = Fn.relu
        elif graph_activation == 'gelu':
            act_fun = Fn.gelu
        elif graph_activation == 'sigmoid':
            act_fun = Fn.sigmoid
        elif graph_activation == 'elu':
            act_fun = Fn.elu
        elif graph_activation == 'glu':
            act_fun = Fn.glu
        elif graph_activation == 'none':
            act_fun = None
        else:
            raise ValueError(
                'the graph activation function should be [relu gelu sigmoid elu glu or none] but instead of {}'
                    .format(graph_activation))

        self.whole_graph.ndata['feature'] = self.entity_embedding
        self.Encoder = EncoderRoGCN(self.entity_dim, self.embedding_range, self.hidden_dim,
                                    activation=act_fun,
                                    self_loop=self_loop, ent_drop=ent_drop, rel_drop=rel_drop)
        self.predictor = ScorePredictor(dist_func)

    def compute_loss(self, pos_score, neg_score, args):
        print(neg_score.shape)
        # Margin loss
        if args.adversarial_temperature != 0:
            negative_score = (Fn.softmax(neg_score * args.adversarial_temperature).detach()
                              * Fn.logsigmoid(-neg_score)).sum()
        else:
            negative_score = Fn.logsigmoid(-neg_score).sum()
        positive_score = Fn.logsigmoid(pos_score).squeeze()

        positive_sample_loss = - positive_score.mean()
        negative_sample_loss = - negative_score.mean()
        # neg_loss, pos_loss = (self.margin_loss(negative_score.view(n_edges, -1), pos_score[etype].unsqueeze(1)))
        neg_loss = negative_sample_loss
        pos_loss = positive_sample_loss
        return pos_loss, neg_loss

    def get_loss(self, positive_graph, negative_graph, blocks, x, args):
        x = self.Encoder(blocks, x, self.relation_embedding)
        positive_score, negative_score = self.pred(positive_graph, negative_graph, x)
        return self.compute_loss(positive_score, negative_score, args)

    def forward(self, blocks, x):
        '''
        Forward function that calculate the score of a batch of triples.
        combine negative graph score => negative score / positive score => positive score
        '''
        x = self.Encoder(blocks, x, self.relation_embedding)
        return x

    def pred(self, positive_graph, negative_graph, x):
        positive_score = self.predictor(positive_graph, x)
        negative_score = self.predictor(negative_graph, x)
        return positive_score, negative_score

    @staticmethod
    def train_step(model, optimizer, iterator, args):
        '''
        A single train step. Apply back-propation and return the loss
        '''
        model.train()

        input_nodes, positive_graph, negative_graph, blocks = next(iterator)
        if args.cuda:
            blocks = [b.to(torch.device('cuda:0')) for b in blocks]
            positive_graph = positive_graph.to(torch.device('cuda:0'))
            negative_graph = negative_graph.to(torch.device('cuda:0'))

        optimizer.zero_grad()
        input_features = blocks[0].srcdata['feature']
        pos_loss, neg_loss = model.get_loss(positive_graph, negative_graph, blocks, input_features, args)
        # if args.subsampling_weight:
        #     neg_subsampling_weight = negative_graph.edata['sw']
        #     pos_subsampling_weight = positive_graph.edata['sw']
        #     positive_sample_loss = - (subsampling_weight * positive_score).sum() / subsampling_weight.sum()
        #     negative_sample_loss = - (subsampling_weight * negative_score).sum() / subsampling_weight.sum()
        # else :
        loss = (pos_loss + neg_loss) / 2
        if args.regularization != 0.0:
            # Use L3 regularization for ComplEx and DistMult
            regularization = args.regularization * (
                    args.ent_regularization * model.entity_embedding.norm(p=3) ** 3 +
                    args.rel_regularization * model.relation_embedding.norm(p=3).norm(p=3) ** 3
            )
            loss = loss + regularization
            regularization_log = {'regularization': regularization.item()}
        else:
            regularization_log = {}

        loss.backward()

        optimizer.step()

        log = {
            **regularization_log,
            'positive_sample_loss': pos_loss.item(),
            'negative_sample_loss': neg_loss.item(),
            'loss': loss.item()
        }

        return log


def main(args):
    # load graph data
    import warnings
    warnings.filterwarnings('ignore')
