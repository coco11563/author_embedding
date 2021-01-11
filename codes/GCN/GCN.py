import torch
import torch.nn as nn
from dgl import DGLError
from dgl import function as fn
from dgl.utils import expand_as_pair


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
                x = layer(blocks, x, weight, train = False)
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

def main(args):
    # load graph data
    import warnings
    warnings.filterwarnings('ignore')
