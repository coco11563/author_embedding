"""
Inductive Representation Learning on Large Graphs
Paper: http://papers.nips.cc/paper/6703-inductive-representation-learning-on-large-graphs.pdf
Code: https://github.com/williamleif/graphsage-simple
Simple reference implementation of GraphSAGE.
"""
import argparse
import pickle

import dgl
import numpy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from dgl.data import register_data_args
from dgl.nn.pytorch.conv import SAGEConv
from tqdm import tqdm

from codes.utils.datareader import init_embedding, read_valid_author, init_triples_with_trace
from codes.utils.embed_opt_utils import embed_opt
from codes.utils.node_samplr.randomwalk import RandomWalkNodeDataLoader, RandomWalkMultiLayerFullNeighborSampler
from codes.utils.valid_utils import evaluator


class GraphSAGE(nn.Module):
    def __init__(self,
                 g,
                 nentity, # input feat
                 n_hidden, # hidden dimension
                 n_classes, # Output feature size; i.e, the number of dimensions of :math:`h_i^{(l+1)}`.
                 n_layers, # gcn layer
                 activation, # activation str format
                 dropout,
                 aggregator_type,
                 dist_func,
                 ent_ini, # num of training nodes
                 in_feats = 64,
                 freeze = False):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.freeze = freeze
        if activation == 'relu':
            act_fun = Fn.relu
        elif activation == 'gelu':
            act_fun = Fn.gelu
        elif activation == 'sigmoid':
            act_fun = Fn.sigmoid
        elif activation == 'elu':
            act_fun = Fn.elu
        elif activation == 'glu':
            act_fun = Fn.glu
        elif activation == 'none':
            act_fun = None
        else:
            raise ValueError(
                'the graph activation function should be [relu gelu sigmoid elu glu or none] but instead of {}'
                    .format(activation))
        self.activation = act_fun
        # input layer
        self.layers.append(SAGEConv(in_feats, n_hidden, aggregator_type, feat_drop=dropout, activation=act_fun))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(SAGEConv(n_hidden, n_hidden, aggregator_type, feat_drop=dropout, activation=act_fun))
        # output layer
        self.layers.append(SAGEConv(n_hidden, n_classes, aggregator_type, feat_drop=dropout, activation=act_fun)) # activation None
        self.predictor = ScorePredictor(dist_func)
        if ent_ini is None:
            self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))
            nn.init.xavier_uniform_(
                tensor=self.entity_embedding,
                gain=activation
            )
            self.entity_dim = in_feats
        else:
            embed_li = []
            hidden_dim = len(ent_ini[0])
            for i in range(nentity):
                embed_li.append(ent_ini[i])
            embed_li = numpy.asarray(embed_li)
            embed_li = torch.from_numpy(embed_li).float()
            self.entity_embedding = nn.Parameter(embed_li, requires_grad= not freeze)
            # self.entity_dim = hidden_dim
            # self.entity_embedding = nn.Embedding.from_pretrained(embed_li, freeze=freeze)
        self.g = g
        self.g.ndata['feature'] = self.entity_embedding

    def forward(self, graph, eval = False):
        if eval :
            h = graph.srcdata['feature']
        else:
            h = graph[0].srcdata['feature']
        # print(h)
        for l, layer in enumerate(self.layers):
            if eval :
                h = layer(graph, h)
            else :
                h = layer(graph[l], h)
        return h

    def compute_loss(self, score, labels, args):
        # print(score.shape)
        # Margin loss
        # gpu problem triggered
        lbd = args.lambda_parameter
        neg_mask = (labels == -1)
        pos_mask = (labels == 1)
        # print(neg_mask.shape)
        # print(torch.ones_like(score[neg_mask]).shape)
        # print(torch.ones_like(score[pos_mask]).shape)
        neg_score = score[neg_mask]
        pos_score = score[pos_mask]

        # print('\n')
        # print('neg_score', neg_score.detach().mean())
        # print('pos_score', pos_score.detach().mean())
        if args.adversarial_temperature != 0:
            negative_score = (Fn.softmax(neg_score * args.adversarial_temperature).detach()
                              * Fn.logsigmoid(neg_score - lbd)).mean() # neg_socre > 0
        else:
            negative_score = Fn.logsigmoid(neg_score - lbd).mean()
        positive_score = - Fn.logsigmoid(lbd - pos_score).mean()
        return positive_score, negative_score

    def get_loss(self, tuples, labels, blocks, args, sw = None):
        x = self.forward(blocks)
        score = self.pred(tuples, x, sw)
        return self.compute_loss(score, labels, args)

    def pred(self, tuples, x, sw):
        score = self.predictor(tuples, x, sw)
        return score

    def regular_loss(self, args):
        reg_loss = None
        for p in self.parameters() :
            if reg_loss is None :
                reg_loss = args.L3_regularization * p.norm(p=3) ** 3
            else:
                reg_loss += args.L3_regularization * p.norm(p=3) ** 3
        return reg_loss

    def evaluate(self, g, eval_func, eval_node, eval_label):
        # self.eval()
        x = self.forward(g, eval=True)
        x = x.detach()
        return x, eval_func(x[eval_node], eval_label)

class ScorePredictor(nn.Module):

    def __init__(self, dist_func):
        super(ScorePredictor, self).__init__()
        self.dist_func = dist_func

    def forward(self, tuples, x, sw):
        tuple_feature = x[tuples]
        score = self.score_func(tuple_feature[:,0], tuple_feature[:, 1])
        if sw is not None :
            score = score * sw
        return score

    def score_func(self, v1, v2):
        return self.dist_func(v1, v2)

def dist_func(head, tail):
    # score = Fn.cosine_similarity(head, tail)
    # print(head.shape, tail.shape)
    score = torch.nn.functional.pairwise_distance(head, tail)
    return score

def main(args):
    # load and preprocess dataset
    sys_path = "/home/xiaomeng/jupyter_base/author_embedding"  # the model check point and log will save in this path

    import sys

    sys.path.append(sys_path)
    import warnings
    warnings.filterwarnings("ignore")
    torch.multiprocessing.set_sharing_strategy('file_system')

    global pool
    if args.gpu == -1 :
        use_cuda = False
    else:
        use_cuda = True
    node_indice, indice_node, start_indice, triples, rel_set, true_trace = init_triples_with_trace('../../data/graph/author_community.pkl', reverse=True)
    edges = numpy.array(triples)
    num_to_generate = edges.shape[0]
    choices = np.random.uniform(size=num_to_generate)

    train_p = 0.4
    valid_p = 0.1
    test_p = 0.5

    train_flag = choices <= train_p
    test_flag = (choices > train_p) & (choices <= train_p + test_p)
    validation_flag = choices > (train_p + test_p)

    test = edges[test_flag]
    train = edges[train_flag]
    valid = edges[validation_flag]
    train_dst = train[:,2]
    train_src = train[:,0]
    num_nodes = len(node_indice.items())

    print("# entities: {}".format(num_nodes))
    print("# relations: {}".format(1))
    print("# training edges: {}".format(train.shape[0]))
    print("# validation edges: {}".format(valid.shape[0]))
    print("# testing edges: {}".format(test.shape[0]))

    embedding_dict = init_embedding('../../data/graph/author_w2v_embedding.pkl', node_indice)

    valid_author = read_valid_author('../../data/classification/test_author_text_corpus.txt', node_indice)



    node_li = []
    label_li = []
    for k, v in valid_author.items():
        node_li.append(k)
        label_li.append(v)

    eval_node = torch.from_numpy(np.asarray(node_li))
    eval_label = np.asarray(label_li)
    cat_dst = numpy.concatenate((train_dst, train_src))
    cat_src = numpy.concatenate((train_src, train_dst))
    g = dgl.graph((cat_src, cat_dst), num_nodes = num_nodes)
    with open('graph_dump_sage.pkl', 'wb') as f :
        pickle.dump(g,f)
    # remove isolated nodes
    nid = g.nodes()[g.out_degrees() != 0]
    model = GraphSAGE(g,
                      num_nodes,
                      args.n_hidden,
                      args.n_class,
                      args.n_layers,
                      "relu",
                      args.dropout,
                      args.aggregator_type,
                      dist_func=dist_func,
                      ent_ini=embedding_dict,
                      freeze=False
                      )
    print(model)
    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    sampler = RandomWalkMultiLayerFullNeighborSampler(args.n_hidden, train[:,(0,2)], 4, neg_num=12, true_tuple=true_trace,
                                                  length=4, windows=2, restart_prob=0)

    if use_cuda :
        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)


    dataloader = RandomWalkNodeDataLoader(
        g, nid, sampler,
        batch_size=521, shuffle=True, drop_last=False, num_workers=0)

    best_epoch = 0
    best_nmi = 0
    best_ari = 0
    best_mif1 = 0
    best_maf1 = 0
    epoch = 0
    it = tqdm(dataloader)
    while True:
        for path_tuple, label, subsampling_ws, _, output_nodes, blocks in it:

            # print(g.ndata['feature'])
            it.set_description("GEN %i" % epoch)
            model_state_file = '/home/xiaomeng/jupyter_base/author_embedding_mac/codes/GCN/mds/model_state_gcn_{}.pth'.format(epoch)
            model.train()
            if use_cuda :
                blocks = [b.to(torch.device('cuda:{}'.format(args.gpu))) for b in blocks]
                path_tuple = path_tuple.to(torch.device('cuda:{}'.format(args.gpu)))
                label = label.to(torch.device('cuda:{}'.format(args.gpu)))
                # input_nodes = input_nodes.to(torch.device('cuda'))
                subsampling_ws = subsampling_ws.to(torch.device('cuda:{}'.format(args.gpu)))
            pos_loss, neg_loss = model.get_loss(path_tuple, label, blocks, args, subsampling_ws)
            # pos_loss, neg_loss = model.get_loss(path_tuple, label, blocks, input_nodes, args, None)
            loss = pos_loss - neg_loss
            if args.L3_regularization != 0.0:
                # Use L3 regularization for ComplEx and DistMult
                reg_loss = model.regular_loss(args)
                loss = loss + reg_loss
            else:
                reg_loss = 0
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print(
            #     "Epoch {:04d} | Loss {:.4f} | pos_loss {:.4f} | neg_loss {:.4f}".
            #         format(epoch, loss.item(), pos_loss.item(), neg_loss.item()))

            # eval
            model.eval()
            if use_cuda :
                model.cpu()
            embed, (NMI, ARI, MiF1, MaF1) = model.evaluate(g, evaluator ,eval_node, eval_label)
            if MiF1 < best_mif1 or MaF1 < best_maf1:
                if epoch >= args.n_epochs:
                    break
            else:
                best_epoch = epoch
                best_nmi = NMI
                best_ari = ARI
                best_mif1 = MiF1
                best_maf1 = MaF1
            with open(model_state_file, 'wb') as f :
                torch.save({'state_dict': model.state_dict(), 'epoch': epoch},
                           f)
            it.set_postfix(loss=loss.item(), bestgen=best_epoch)
            # it.set_postfix({"Best NMI" : best_nmi,"Best ARI" : best_ari, "Best MICRO-F1" : best_mif1, "Best Epoch" :
            #     best_epoch, "loss" : loss.item(), "pos_loss" : pos_loss.item(), "neg_loss" : neg_loss.item()})
            # tqdm.write('Epoch {} '.format(epoch))
            tqdm.write("NMI {:.4f} | ARI {:.4f} | MICRO-F1 {:.4f} | MACRO-F1 {:.4f} ".format(NMI, ARI, MiF1, MaF1))
            tqdm.write("Best NMI {:.4f} | Best ARI {:.4f} | Best MICRO-F1 {:.4f} | Best MACRO-F1 {:.4f} ".format(best_nmi, best_ari, best_mif1, best_maf1))
            tqdm.write("Loss {:.4f} | pos_loss {:.4f} | neg_loss {:.4f}".format(loss.item(), pos_loss.item(), neg_loss.item()))
            # print(
            #     "Best NMI {:.4f} | Best ARI {:.4f} | Best MICRO-F1 {:.4f} | Best MACRO-F1 {:.4f} | Best Epoch {} | Loss {:.4f} | pos_loss {:.4f} | neg_loss {:.4f}".
            #     format(best_nmi, best_ari, best_mif1, best_maf1, best_epoch, loss.item(), pos_loss.item(), neg_loss.item()))
            test_dict = embed_opt(embed, torch.range(0, num_nodes - 1, dtype=torch.long).cuda(), indice_node)
            # dump for next eval!
            with open('/home/xiaomeng/jupyter_base/author_embedding_mac/codes/GCN//emb/gcn_embed_{}.pkl'.format(epoch), 'wb') as f:
                pickle.dump(test_dict, f)
                # pickle.dump(test_dict, 'rgcn_embed.pkl')
            if use_cuda:
                model.cuda()
            epoch += 1
            if epoch > args.n_epochs :
                break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GraphSAGE')
    register_data_args(parser)
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="dropout probability")
    parser.add_argument("--gpu", type=int, default=0,
                        help="gpu")
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=200,
                        help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=64,
                        help="number of hidden gcn units")
    parser.add_argument("--n-class", type=int, default=64,
                        help="number of hidden dimension")
    parser.add_argument("--n-layers", type=int, default=1,
                        help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Weight for L2 loss")
    parser.add_argument("--aggregator-type", type=str, default="mean",
                        help="Aggregator type: mean/gcn/pool/lstm")
    parser.add_argument('-a', '--adversarial_temperature', default=0, type=float)
    parser.add_argument('--L3-regularization', default=0, type=float)
    parser.add_argument('--lambda-parameter', default=1.5, type=float)
    args = parser.parse_args()
    print(args)

    main(args)