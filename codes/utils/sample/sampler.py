import dgl
import numpy
from dglds import load_data
from dgl.dataloading.negative_sampler import _BaseNegativeSampler
from dgl import backend as F
import torch

from EdgeDataLoader import EdgeDataLoader
# from utils.dglds import load_data, build_sub_graph
from codes.utils.sample.dglds import build_sub_graph


class NegativeSampler(object):
    def __init__(self, g, k):
        # caches the probability distribution
        self.weights = g.in_degrees().float() ** 0.75
        self.k = k

    def __call__(self, g, eids):
        src, _ = g.find_edges(eids)

        src = src.repeat_interleave(self.k)

        dst = self.weights.multinomial(len(src), replacement=True)

        return src, dst
# 原始的sampler
class OriUniform(_BaseNegativeSampler):
    def __init__(self, k):
        self.k = k

    def _generate(self, g, eids, canonical_etype):
        _, _, vtype = canonical_etype
        shape = F.shape(eids)
        dtype = F.dtype(eids)
        ctx = F.context(eids)
        shape = (shape[0] * self.k,)
        src, _ = g.find_edges(eids, etype=canonical_etype)
        src = F.repeat(src, self.k, 0)
        dst = F.randint(shape, dtype, ctx, 0, g.number_of_nodes(vtype))
        return src, dst

# 修改的以metapath采样的Uniform sampler
class Uniform(_BaseNegativeSampler):
    def __init__(self, k, type_dict, type_set, etype_dict):
        self.k = k
        self.ntype_dict = type_dict # nid => ntype
        self.type_set = dict()  # ntype => node seq
        for k, v in type_set.items() :
            self.type_set[k] = torch.as_tensor(list(v))
        self.etype_dict = etype_dict # eid => etype 也可以定义为一种映射函数 g.edata['type']
    # 采样edge
    def _generate(self, g, eids, canonical_etype):
        dtype = F.dtype(eids)
        ctx = F.context(eids)
        # find 起始点
        src, _dst = g.find_edges(eids, etype=canonical_etype)
        etype = self.etype_dict[eids]
        src = F.repeat(src, self.k, 0)
        etype = F.repeat(etype, self.k, 0)
        dsts = None
        # dst 应该在set里选哈
        for i in _dst :
            '''
            没有过滤掉负采样中的正例，并且在负采样中没有去除正确的原dst 同时没有为这条边生成采样系数权值subsampling weight
            同时这里的采样只取true src与negative dst，暨tail batch， 应该根据mode来进行正负采样
            这里在UniformBaseOnTriples进行修正
            '''
            nid = i.numel()
            ntype = self.ntype_dict[nid]
            node_set = self.type_set[ntype]
            node_limit = len(node_set)
            # uniform sampling
            dst = F.randint((1, 2 * self.k), dtype, ctx, 0, node_limit)
            dst = node_set[dst]
            if dsts is None :
                dsts = dst
            else :
                dsts = torch.cat((dsts, dst), dim = 1)
        return (src, dsts.squeeze(dim = 0)), etype


'''
上面的采样器没有过滤掉负采样中的正例，并且在负采样中没有去除正确的原dst 同时没有为这条边生成采样系数权值subsampling weight
同时这里的采样只取true src与negative dst，暨tail batch， 应该根据mode来进行正负采样
这里在UniformBaseOnTriples进行修正
1.采样的负例中不存在正例
2.针对每一个采样的例子返回一个子采样权值
3.采样分为head、tail两种
'''
class UniformBaseOnTriples(_BaseNegativeSampler):
    def __init__(self, k, whole_true, etype_dict, mode, base_num = 4):
        self.k = k
        self.whole_true = whole_true
        self.base_num = base_num
        self.count = self.count_frequency(whole_true, base_num)
        self.true_head, self.true_tail = self.get_true_head_and_tail(whole_true)
        self.mode = mode
        self.etype_dict = etype_dict
    def _generate(self, g, eids, canonical_etype) :

        srcs, dsts = g.find_edges(eids, etype=canonical_etype)
        neg_sample = None
        nsrcs, ndsts = srcs.numpy(), dsts.numpy()

        etype = self.etype_dict[eids]
        netype = etype.numpy()
        _, _, vtype = canonical_etype
        subsampling_w = []
        for indice, src in enumerate(nsrcs) :
            negative_sample_size = 0
            rel = netype[indice]
            subsampling_weight = self.count[(src, rel)] + self.count[(ndsts[indice], -rel - 1)]
            subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))
            negative_sample_list = []
            while negative_sample_size < self.k:
                negative_sample = numpy.random.randint(g.number_of_nodes(vtype), size=self.k * 2)
                if self.mode == 'head-batch':
                    mask = numpy.in1d(
                        negative_sample,
                        self.true_head[(rel, ndsts[indice])], # sample pure head (not true)
                        assume_unique=True,
                        invert=True
                    )
                elif self.mode == 'tail-batch':
                    mask = numpy.in1d(
                        negative_sample,
                        self.true_tail[(src, rel)], # sample pure tail (not true)
                        assume_unique=True,
                        invert=True
                    )
                else:
                    raise ValueError('Training batch mode %s not supported' % self.mode)
                negative_sample = negative_sample[mask]
                negative_sample_list.append(negative_sample)
                negative_sample_size += negative_sample.size
                subsampling_w.append(subsampling_weight)
            # negative sample
            negative_sample = numpy.concatenate(negative_sample_list)[:self.k]
            # add positive sample to the head
            # if self.mode == 'head-batch' :
            #     negative_sample = numpy.append(numpy.array(ndsts[indice]), negative_sample)
            # elif self.mode == 'tail-batch' :
            #     negative_sample = numpy.append(numpy.array(src), negative_sample)
            # else:
            #     raise ValueError('Training batch mode %s not supported' % self.mode)

            if neg_sample is None :
                neg_sample  = negative_sample
            else :
                neg_sample = numpy.concatenate((neg_sample, negative_sample))
        if self.mode == 'head-batch':
            dsts = F.repeat(dsts, self.k, 0)
            srcs = torch.from_numpy(neg_sample)
        elif self.mode == 'tail-batch':
            srcs = F.repeat(srcs, self.k, 0)
            dsts = torch.from_numpy(neg_sample)
        else:
            raise ValueError('Training batch mode %s not supported' % self.mode)
        subsampling_w = torch.cat(subsampling_w, dim = -1)
        etype = F.repeat(etype, self.k, 0)
        subsampling_w = F.repeat(subsampling_w, self.k, 0)
        return (srcs, dsts), etype, subsampling_w

    @staticmethod
    def count_frequency(triples, start=4):
        '''
        Get frequency of a partial triple like (head, relation) or (relation, tail)
        The frequency will be used for subsampling like word2vec
        '''
        count = {}
        for head, relation, tail in triples:
            if (head, relation) not in count:
                count[(head, relation)] = start
            else:
                count[(head, relation)] += 1

            if (tail, -relation - 1) not in count:
                count[(tail, -relation - 1)] = start
            else:
                count[(tail, -relation - 1)] += 1
        return count

    @staticmethod
    def get_true_head_and_tail(triples):
        '''
        Build a dictionary of true triples that will
        be used to filter these true triples for negative sampling
        '''

        true_head = {}
        true_tail = {}

        for head, relation, tail in triples:
            if (head, relation) not in true_tail:
                true_tail[(head, relation)] = []
            true_tail[(head, relation)].append(tail)
            if (relation, tail) not in true_head:
                true_head[(relation, tail)] = []
            true_head[(relation, tail)].append(head)

        for relation, tail in true_head:
            true_head[(relation, tail)] = numpy.array(list(set(true_head[(relation, tail)])))
        for head, relation in true_tail:
            true_tail[(head, relation)] = numpy.array(list(set(true_tail[(head, relation)])))

        return true_head, true_tail

class Subsampling(Uniform) :
    def _generate(self, g, eids, canonical_etype):
        pass

if __name__ == '__main__':
    data = load_data('wn18')
    num_nodes = data.num_nodes
    train_data = data.train
    valid_data = data.valid
    test_data = data.test
    num_rels = data.num_rels
    nentity = num_nodes
    nrelation = num_rels
    whole_graph = data._g
    train_graph = build_sub_graph(whole_graph, whole_graph.edata['train_mask'], False)
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
    e_id = train_graph.edges(form='eid')
    # only one etype
    canonical_etype = train_graph.canonical_etypes[0]

    etype_dict = train_graph.edata['etype']
    src, dst = train_graph.find_edges(e_id, etype=canonical_etype)
    rel = etype_dict[e_id]
    train_data_triple = torch.cat((src.unsqueeze(1), rel.unsqueeze(1), dst.unsqueeze(1)), dim=1).numpy()
    # train_triples = []
    # for i, j, k in train_data_triple:
    #     train_triples.append((i, j, k))

    '''
    all data sample scripts
    '''
    # train_triples = []
    # for i, j, k in train_data:
    #     train_triples.append((i, j, k))
    # test_triples = []
    # for i, j, k in test_data:
    #     test_triples.append((i, j, k))
    # valid_triples = []
    # for i, j, k in valid_data:
    #     valid_triples.append((i, j, k))
    # # All true triples
    # all_true_triples = train_triples + valid_triples + test_triples
    # All true triples
    # all_true_triples = train_triples
    train_data_set = EdgeDataLoader(
        train_graph, e_id, sampler,
        negative_sampler=UniformBaseOnTriples(128, train_data_triple, etype_dict, mode='head-batch'),
        # negative_sampler=dgl.dataloading.negative_sampler.Uniform(128),
        batch_size=100,
        shuffle=True,
        drop_last=False,
        num_workers=4
    )
    for input_nodes, pair_graph, neg_pair_graph, blocks in train_data_set :
        print(neg_pair_graph)
        break