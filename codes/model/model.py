import argparse
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import RelGraphConv

from codes.utils.embed_opt_utils import embed_opt
from codes.utils.valid_utils import evaluator

sys.path.append('/home/xiaomeng/jupyter_base/author_embedding')

from codes.utils.sample_utils import *
from codes.utils.model_utils import BaseRGCN
from codes.utils.gcn_utils import get_adj_and_degrees, build_test_graph, generate_sampled_graph_and_labels, calc_mrr
from codes.utils.datareader import *

class EmbeddingLayer(nn.Module):
    def __init__(self, num_nodes, h_dim, embed_dict, freeze_option = True):
        super(EmbeddingLayer, self).__init__()
        self.embed_dict = embed_dict
        if self.embed_dict is not None :
            embed_li = []
            for i in range(num_nodes) :
                embed_li.append(embed_dict[i])
            embed_li = np.asarray(embed_li)
            embed_li = torch.from_numpy(embed_li).float()
            self.embedding = torch.nn.Embedding.from_pretrained(embed_li, freeze=freeze_option)
        else :
            print("init from random embedding")
            self.embedding = torch.nn.Embedding(num_nodes, h_dim)

    def forward(self, g, h, r, norm):
        return self.embedding(h.squeeze())

class RGCN(BaseRGCN):
    def build_input_layer(self):
        return EmbeddingLayer(self.num_nodes, self.h_dim, self.embed_dict, self.freeze_embedding)

    def build_hidden_layer(self, idx):
        act = F.relu if idx < self.num_hidden_layers - 1 else None
        return RelGraphConv(self.h_dim, self.h_dim, self.num_rels, self.regularizer,
                self.num_bases, activation=act, self_loop=True, layer_norm=True,
                dropout=self.dropout)

class LinkPredict(nn.Module):
    def __init__(self, in_dim, h_dim, num_rels, regularizer = "bdd", num_bases=-1,
                 num_hidden_layers=1, dropout=0, use_cuda=False, reg_param=0, embed_dict = None, freeze_embedding = False):
        super(LinkPredict, self).__init__()
        self.rgcn = RGCN(in_dim, h_dim, h_dim, num_rels * 2, num_bases,
                         num_hidden_layers, dropout, use_cuda,
                         embed_dict = embed_dict,
                         freeze_embedding = freeze_embedding,
                         regularizer = regularizer)
        self.freeze = freeze_embedding
        self.reg_param = reg_param
        self.w_relation = nn.Parameter(torch.Tensor(num_rels, h_dim))
        nn.init.xavier_uniform_(self.w_relation,
                                gain=nn.init.calculate_gain('relu'))

    def calc_score(self, embedding, triplets):
        # DistMult
        s = embedding[triplets[:,0]]
        r = self.w_relation[triplets[:,1]]
        o = embedding[triplets[:,2]]
        score = torch.sum(s * r * o, dim=1)
        # score = torch.sum(s * o, dim=1)
        return score

    def forward(self, g, h, r, norm):
        return self.rgcn.forward(g, h, r, norm)

    def regularization_loss(self, embedding):
        if not self.freeze :
            return torch.mean(embedding.pow(2)) + torch.mean(self.w_relation.pow(2))
        else :
            return torch.mean(self.w_relation.pow(2))

    def get_loss(self, g, embed, triplets, labels):
        # triplets is a list of data samples (positive and negative)
        # each row in the triplets is a 3-tuple of (source, relation, destination)
        score = self.calc_score(embed, triplets)
        predict_loss = F.binary_cross_entropy_with_logits(score, labels)
        reg_loss = self.regularization_loss(embed)
        return predict_loss + self.reg_param * reg_loss

def node_norm_to_edge_norm(g, node_norm):
    g = g.local_var()
    # convert to edge norm
    g.ndata['norm'] = node_norm
    g.apply_edges(lambda edges : {'norm' : edges.dst['norm']})
    return g.edata['norm']

def main(args):
    # load graph data
    import warnings
    warnings.filterwarnings('ignore')
    global pool
    node_indice, indice_node, start_indice, triples, rel_set = init_triples('../../data/graph/author_community.pkl')
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

    train_data = train
    valid_data = valid
    test_data = test

    num_rels = len(rel_set)
    num_nodes = len(node_indice.items())

    print("# entities: {}".format(num_nodes))
    print("# relations: {}".format(num_rels))
    print("# training edges: {}".format(train.shape[0]))
    print("# validation edges: {}".format(valid.shape[0]))
    print("# testing edges: {}".format(test.shape[0]))

    embedding_dict = init_embedding('../../data/graph/author_w2v_embedding.pkl', node_indice)

    valid_author = read_valid_author('../../data/classification/test_author_text_corpus.txt', node_indice)

    node_li = []
    label_li = []
    for k,v in valid_author.items() :
        node_li.append(k)
        label_li.append(v)

    eval_node = torch.from_numpy(np.asarray(node_li))
    eval_label = np.asarray(label_li)

    for k,v in embedding_dict.items() :
        print(v.shape)
        break
    # check cuda
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(args.gpu)
        eval_node = eval_node.cuda()
    # create model
    if args.pretrain is False :
        embedding_dict = None

    model = LinkPredict(num_nodes,
                        args.n_hidden,
                        num_rels,
                        num_bases=args.n_bases,
                        num_hidden_layers=args.n_layers,
                        dropout=args.dropout,
                        use_cuda=use_cuda,
                        reg_param=args.regularization,
                        # embed_dict=embedding_dict,
                        embed_dict=embedding_dict,
                        freeze_embedding=args.freeze)

    # validation and testing triplets
    valid_data = torch.LongTensor(valid_data)
    test_data = torch.LongTensor(test_data)

    # build test graph
    test_graph, test_rel, test_norm = build_test_graph(
        num_nodes, num_rels, train_data)
    test_deg = test_graph.in_degrees(
                range(test_graph.number_of_nodes())).float().view(-1,1)
    test_node_id = torch.arange(0, num_nodes, dtype=torch.long).view(-1, 1)
    test_rel = torch.from_numpy(test_rel)
    test_norm = node_norm_to_edge_norm(test_graph, torch.from_numpy(test_norm).view(-1, 1))

    if use_cuda:
        model.cuda()

        # test_graph = test_graph.to('cuda:' + str(args.gpu))
        # test_node_id = test_node_id.cuda()
        # test_rel = test_rel.cuda()
        # test_norm = test_norm.cuda()

        # test_data = test_data.cuda()
        # valid_data = valid_data.cuda()

    # build adj list and calculate degrees for sampling
    adj_list, degrees = get_adj_and_degrees(num_nodes, train_data)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    model_state_file = 'model_state.pth'
    forward_time = []
    backward_time = []
    if args.edge_sampler == "neighbor" :
        pool = MultiThreadSamplerPool(train_data, args.graph_batch_size, args.graph_split_size,
                    num_rels, adj_list, degrees, args.negative_sample,
                    args.edge_sampler, max_runtime=args.n_epochs, max_worker=64)
    print('init sampler thread pool')
    # training loop
    print("start training loop...")

    epoch = 0
    best_nmi = 0
    best_ari = 0
    best_mif1 = 0
    best_maf1 = 0

    while True:
        if epoch % 1 == 0:
            # perform validation on CPU because full graph is too large
            model.eval()
            print("start eval")
            model.cpu()
            embed = model(test_graph, test_node_id, test_rel, test_norm)
            embed = embed.detach()
            eval_embed = embed[eval_node]
            eval_embed = eval_embed.numpy()
            NMI, ARI, MICRO_F1, MACRO_F1 = evaluator(eval_embed, eval_label)

            # if NMI < best_nmi or ARI < best_ari or MICRO_F1 < best_mif1 or MACRO_F1 < best_maf1:
            if MICRO_F1 < best_mif1 or MACRO_F1 < best_maf1:
                if epoch >= args.n_epochs:
                    break
            else:
                best_epoch = epoch
                best_nmi = NMI
                best_ari = ARI
                best_mif1 = MICRO_F1
                best_maf1 = MACRO_F1
                # best_mrr = mrr
                # torch.save({'state_dict': model.state_dict(), 'epoch': epoch},
                #            model_state_file)
            test_dict = embed_opt(embed, torch.range(0, num_nodes - 1, dtype=torch.long).cuda(), indice_node)

            with open('rgcn_embed_{}.pkl'.format(epoch), 'wb') as f:
                pickle.dump(test_dict, f)
                # pickle.dump(test_dict, 'rgcn_embed.pkl')
            if use_cuda :
                model.cuda()
        model.train()
        if args.edge_sampler == "neighbor" :
            # 防止队列为空
            while pool.q.empty():
                pass
            # perform edge neighborhood sampling to generate training graph and data
            g, node_id, edge_type, node_norm, data, labels = pool.get()
        # perform edge neighborhood sampling to generate training graph and data
        else :
            g, node_id, edge_type, node_norm, data, labels = \
                generate_sampled_graph_and_labels(
                    train_data, args.graph_batch_size, args.graph_split_size,
                    num_rels, adj_list, degrees, args.negative_sample,
                    args.edge_sampler)
            print("Done edge sampling")
        # set node/edge feature
        node_id = torch.from_numpy(node_id).view(-1, 1).long()
        edge_type = torch.from_numpy(edge_type)
        edge_norm = node_norm_to_edge_norm(g, torch.from_numpy(node_norm).view(-1, 1))
        data, labels = torch.from_numpy(data), torch.from_numpy(labels)
        deg = g.in_degrees(range(g.number_of_nodes())).float().view(-1, 1)
        if use_cuda:
            node_id, deg = node_id.cuda(), deg.cuda()
            edge_type, edge_norm = edge_type.cuda(), edge_norm.cuda()
            data, labels = data.cuda(), labels.cuda()
            g = g.to(args.gpu)

        t0 = time.time()
        embed = model(g, node_id, edge_type, edge_norm)
        loss = model.get_loss(g, embed, data, labels)
        t1 = time.time()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm) # clip gradients
        optimizer.step()
        t2 = time.time()

        forward_time.append(t1 - t0)
        backward_time.append(t2 - t1)
        print("Epoch {:04d} | Loss {:.4f} | Best NMI {:.4f} | Best ARI {:.4f} | Best MICRO-F1 {:.4f} | Best MACRO-F1 {:.4f} | Forward {:.4f}s | Backward {:.4f}s | Best Epoch {}".
              format(epoch, loss.item(), best_nmi, best_ari, best_mif1, best_maf1, forward_time[-1], backward_time[-1], best_epoch))

        optimizer.zero_grad()

        # validation

            # if use_cuda:
            #     model.cuda()
        epoch += 1
        # if epoch % args.evaluate_every == 0:
        #     # perform validation on CPU because full graph is too large
        #
        #     #     model.cpu()
        #     model.eval()
        #     print("start eval")
        #     mrr = calc_mrr(embed, model.w_relation, torch.LongTensor(train_data),
        #                          test_data, valid_data, hits=[1, 3, 10], eval_bz=args.eval_batch_size,
        #                          eval_p=args.eval_protocol)
        #     # save best model
        #     if mrr < best_mrr:
        #         if epoch >= args.n_epochs:
        #             break
        #     else:
        #         best_mrr = mrr
        #         torch.save({'state_dict': model.state_dict(), 'epoch': epoch},
        #                    model_state_file)

    print("training done")
    print("Mean forward time: {:4f}s".format(np.mean(forward_time)))
    print("Mean Backward time: {:4f}s".format(np.mean(backward_time)))

    print("\nstart testing:")

    # use best model checkpoint
    checkpoint = torch.load(model_state_file)
    # if use_cuda:
    #     model.cpu() # test on CPU
    model.eval()
    model.load_state_dict(checkpoint['state_dict'])
    print("Using best epoch: {}".format(checkpoint['epoch']))
    embed = model(test_graph, test_node_id, test_rel, test_norm)
    # calc_mrr(embed, model.w_relation, torch.LongTensor(train_data), valid_data,
    #                test_data, hits=[1, 3, 10], eval_bz=args.eval_batch_size, eval_p=args.eval_protocol)
    eval_embed = embed[eval_node]
    eval_embed = eval_embed.detach().cpu().numpy()
    evaluator(eval_embed, eval_label)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RGCN')
    parser.add_argument("--dropout", type=float, default=0.1,
            help="dropout probability")
    parser.add_argument("--n-hidden", type=int, default=64,
            help="number of hidden units")
    parser.add_argument("--gpu", type=int, default=1,
            help="gpu")
    parser.add_argument("--lr", type=float, default=1e-4,
            help="learning rate")
    parser.add_argument("--n-bases", type=int, default=4,
            help="number of weight blocks for each relation")
    parser.add_argument("--n-layers", type=int, default=2,
            help="number of propagation rounds")
    parser.add_argument("--n-epochs", type=int, default=60000,
            help="number of minimum training epochs")
    parser.add_argument("-d", "--dataset", type=str, default="FB15k-237",
            help="dataset to use")
    parser.add_argument("--eval-batch-size", type=int, default=500,
            help="batch size when evaluating")
    parser.add_argument("--eval-protocol", type=str, default="raw",
            help="type of evaluation protocol: 'raw' or 'filtered' mrr")
    parser.add_argument("--regularization", type=float, default=0.01,
            help="regularization weight")
    parser.add_argument("--grad-norm", type=float, default=1.0,
            help="norm to clip gradient to")
    parser.add_argument("--graph-batch-size", type=int, default=30000,
            help="number of edges to sample in each iteration")
    parser.add_argument("--graph-split-size", type=float, default=0.5,
            help="portion of edges used as positive sample")
    parser.add_argument("--negative-sample", type=int, default=10,
            help="number of negative samples per positive sample")
    parser.add_argument("--evaluate-every", type=int, default=100,
            help="perform evaluation every n epochs")
    parser.add_argument("--edge-sampler", type=str, default="neighbor",
            help="type of edge sampler: 'uniform' or 'neighbor'")
    parser.add_argument("--freeze", type=bool, default=False)
    parser.add_argument('--pretrain', type=bool, default=True)
    args = parser.parse_args()
    print(args)
    main(args)
