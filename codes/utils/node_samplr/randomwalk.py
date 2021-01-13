import inspect

import dgl
import dgl.sampling.randomwalks as walk
import numpy
import torch
from dgl import backend as F
from dgl.dataloading import DataLoader, Mapping, utils, Collator, NID, distributed, subgraph, sampling, BlockSampler, \
    EID, transform
from dgl.dataloading.dataloader import _tensor_or_dict_to_numpy, _locate_eids_to_exclude, assign_block_eids
from dgl.dataloading.pytorch import _pop_blocks_storage, _restore_blocks_storage
from dgl.distributed import DistGraph
from torch.utils.data import WeightedRandomSampler


class RandomWalkSampler:
    def __init__(self, tuples, base_num,
                 metapath=None, length=None, prob=None,
                 restart_prob=None, windows=2,
                 true_tuple=None, neg_num=0):
        self.metapath = metapath
        self.length = length
        self.prob = prob
        self.restart_prob = restart_prob
        self.window_size = windows
        self.neg_num = neg_num
        self.true_tuple = self.get_true_head_and_tail(true_tuple)
        self.count = self.count_frequency(tuples, base_num)

    """
    目前的问题主要在于 采样时会采很多次入度多的点
    """

    def sampler(self, g, nids):
        traces, type = walk.random_walk(g, nodes=nids,
                                        metapath=self.metapath, length=self.length,
                                        prob=self.prob, restart_prob=self.restart_prob)
        print('trace is', traces)
        num_nodes = g.number_of_nodes()
        tuples = []
        labels = []
        node_set = set()
        subsampling_ws = []
        for path in traces.numpy():
            srcs, dsts, subsampling_w = self.trace_sampler(path, node_set)
            if self.neg_num > 0 :
                neg_src, neg_dst, neg_subsampling_w = self.negative_sampling(srcs, dsts, num_nodes, node_set)
                srcs = torch.from_numpy(numpy.array(srcs))
                dsts = torch.from_numpy(numpy.array(dsts))
                pos_sample = torch.cat((srcs.reshape(-1, 1), dsts.reshape(-1, 1)), 1)
                neg_sample = torch.cat((neg_src.reshape(-1, 1), neg_dst.reshape(-1, 1)), 1)
                pos_label = torch.ones_like(srcs)  # n
                neg_label = torch.ones_like(neg_src) * -1
                label = torch.cat((pos_label, neg_label))  # n
                tuple = torch.cat((pos_sample, neg_sample))
                sub_w = torch.cat((subsampling_w, neg_subsampling_w))
            else :
                srcs = torch.from_numpy(numpy.array(srcs))
                dsts = torch.from_numpy(numpy.array(dsts))
                tuple = torch.cat((srcs.reshape(-1, 1), dsts.reshape(-1, 1)), 1)
                label = torch.ones_like(srcs)  # n
                sub_w = subsampling_w
            tuples.append(tuple)
            labels.append(label)
            subsampling_ws.append(sub_w)
        return torch.cat(tuples), torch.cat(labels), torch.cat(subsampling_ws), torch.from_numpy(numpy.array(list(node_set)))

    """
    输入 一组 trace
    返回 对应的 src dst 以及sub sampling weight
    """

    def trace_sampler(self, traces, node_set):
        srcs = []
        dsts = []
        subsampling_w = []
        for start in range(self.length):
            # for all node but not last one
            src = traces[start]
            for i in range(self.window_size):
                # sample every window_size
                window_s = i + 1
                next_indice = start + window_s
                if next_indice <= self.length:
                    dst = traces[next_indice]
                    subsampling_weight = self.count[src] + self.count[- dst - 1]
                    subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))
                    subsampling_w.append(subsampling_weight)
                    srcs.append(src)
                    dsts.append(dst)
                    node_set.add(src)
                    node_set.add(dst)
                    # tuples.append((src, dst))
        subsampling_w = torch.cat(subsampling_w, dim=-1)
        return srcs, dsts, subsampling_w

    """         
    输入 tuples 
    ---
    返回 负采样/子采样权值
    """

    def negative_sampling(self, nsrcs, ndsts, node_limit, node_set):
        subsampling_w = []
        neg_sample = None
        for indice, src in enumerate(nsrcs):
            negative_sample_size = 0
            negative_sample_list = []
            subsampling_weight = self.count[src] + self.count[- ndsts[indice] - 1]
            subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))

            while negative_sample_size < self.neg_num:
                # TODO 这里的负采样是全局负采样，对于有关系的需要考虑 metapath
                # 这里都是tail负采样
                negative_sample = numpy.random.randint(node_limit, size=self.neg_num * 2)
                # 排除正例
                mask = numpy.in1d(
                    negative_sample,
                    self.true_tuple[src],  # sample pure tail (not true)
                    assume_unique=True,
                    invert=True)
                negative_sample = negative_sample[mask]
                negative_sample_list.append(negative_sample)
                negative_sample_size += negative_sample.size
                subsampling_w.append(subsampling_weight)

            negative_sample = numpy.concatenate(negative_sample_list)[:self.neg_num]

            if neg_sample is None:
                neg_sample = negative_sample
            else:
                neg_sample = numpy.concatenate((neg_sample, negative_sample))
        subsampling_w = torch.cat(subsampling_w, dim=-1)
        subsampling_w = F.repeat(subsampling_w, self.neg_num, 0)
        pos_src = torch.tensor(nsrcs, dtype=torch.long)
        pos_src = F.repeat(pos_src, self.neg_num, 0)
        dsts = torch.from_numpy(neg_sample)
        for i in dsts:
            num = i.item()
            node_set.add(num)
        return pos_src, dsts, subsampling_w

    @staticmethod
    def count_frequency(tuples, base_num):
        count = {}
        for head, tail in tuples:
            if head not in count:
                count[head] = base_num
            else:
                count[head] += 1

            if (- tail - 1) not in count:
                count[- tail - 1] = base_num
            else:
                count[- tail - 1] += 1
        return count

    @staticmethod
    def get_true_head_and_tail(true_trace):
        ret = dict()
        for k, v in true_trace.items():
            ret[k] = numpy.array(list(v))
        return ret


class RandomWalkMultiLayerNeighborSampler(BlockSampler):
    def __init__(self, fanouts, tuples, base_num, replace=False, return_eids=False,
                 metapath=None, length=None, prob=None,
                 restart_prob=None, windows=2,
                 true_tuple=None, neg_num=0):
        super().__init__(len(fanouts), return_eids)
        self.random_walk_sampler = RandomWalkSampler(tuples, base_num, metapath=metapath, length=length, prob=prob,
                 restart_prob=restart_prob, windows=windows,
                 true_tuple=true_tuple, neg_num=neg_num)

        self.fanouts = fanouts
        self.replace = replace

    def sample_frontier(self, block_id, g, seed_nodes):
        fanout = self.fanouts[block_id]
        if isinstance(g, distributed.DistGraph):
            if fanout is None:
                # TODO(zhengda) There is a bug in the distributed version of in_subgraph.
                # let's use sample_neighbors to replace in_subgraph for now.
                frontier = distributed.sample_neighbors(g, seed_nodes, -1, replace=False)
            else:
                frontier = distributed.sample_neighbors(g, seed_nodes, fanout, replace=self.replace)
        else:
            if fanout is None:
                frontier = subgraph.in_subgraph(g, seed_nodes)
            else:
                frontier = sampling.sample_neighbors(g, seed_nodes, fanout, replace=self.replace)
        return frontier

    def sample_blocks(self, g, seed_nodes, exclude_eids=None) :
        print('seed is ', seed_nodes)
        blocks = []
        exclude_eids = (
            _tensor_or_dict_to_numpy(exclude_eids) if exclude_eids is not None else None)
        tuples, labels, subsampling_ws, seed_nodes = self.random_walk_sampler.sampler(g, seed_nodes)
        for block_id in reversed(range(self.num_layers)):
            frontier = self.sample_frontier(block_id, g, seed_nodes)

            # Removing edges from the frontier for link prediction training falls
            # into the category of frontier postprocessing
            if exclude_eids is not None:
                parent_eids = frontier.edata[EID]
                parent_eids_np = _tensor_or_dict_to_numpy(parent_eids)
                located_eids = _locate_eids_to_exclude(parent_eids_np, exclude_eids)
                if not isinstance(located_eids, Mapping):
                    # (BarclayII) If frontier already has a EID field and located_eids is empty,
                    # the returned graph will keep EID intact.  Otherwise, EID will change
                    # to the mapping from the new graph to the old frontier.
                    # So we need to test if located_eids is empty, and do the remapping ourselves.
                    if len(located_eids) > 0:
                        frontier = transform.remove_edges(frontier, located_eids)
                        frontier.edata[EID] = F.gather_row(parent_eids, frontier.edata[EID])
                else:
                    # (BarclayII) remove_edges only accepts removing one type of edges,
                    # so I need to keep track of the edge IDs left one by one.
                    new_eids = parent_eids.copy()
                    for k, v in located_eids.items():
                        if len(v) > 0:
                            frontier = transform.remove_edges(frontier, v, etype=k)
                            new_eids[k] = F.gather_row(parent_eids[k], frontier.edges[k].data[EID])
                    frontier.edata[EID] = new_eids

            block = transform.to_block(frontier, seed_nodes)

            if self.return_eids:
                assign_block_eids(block, frontier)

            seed_nodes = {ntype: block.srcnodes[ntype].data[NID] for ntype in block.srctypes}

            # Pre-generate CSR format so that it can be used in training directly
            block.create_formats_()
            blocks.insert(0, block)
        return tuples, labels, subsampling_ws, blocks

class RandomWalkMultiLayerFullNeighborSampler(RandomWalkMultiLayerNeighborSampler):
    def __init__(self, n_layers, tuples, base_num, replace=False, return_eids=False,
                 metapath=None, length=None, prob=None,
                 restart_prob=None, windows=2,
                 true_tuple=None, neg_num=0
                 ):
        super().__init__([None] * n_layers, tuples, base_num, replace=replace, return_eids=return_eids,
                 metapath=metapath, length=length, prob=prob,
                 restart_prob=restart_prob, windows=windows,
                 true_tuple=true_tuple, neg_num=neg_num)

# remove dist graph support
class RandomWalkNodeCollator(Collator):
    def __init__(self, g, nids, block_sampler):
        self.g = g
        self._is_distributed = isinstance(g, DistGraph)
        assert not self._is_distributed
        if not isinstance(nids, Mapping):
            assert len(g.ntypes) == 1, \
                "nids should be a dict of node type and ids for graph with multiple node types"
        self.nids = nids
        self.block_sampler = block_sampler

        # for heterogeneous graph
        if isinstance(nids, Mapping):
            self._dataset = utils.FlattenedDict(nids)
        else:
            self._dataset = nids

    @property
    def dataset(self):
        return self._dataset

    def collate(self, items):
        if isinstance(items[0], tuple):
            items = utils.group_as_dict(items)

        if isinstance(items, dict):
            items = utils.prepare_tensor_dict(self.g, items, 'items')
        else:
            items = utils.prepare_tensor(self.g, items, 'items')

        # items is the chosen id
        tuples, labels, subsampling_ws, blocks = self.block_sampler.sample_blocks(self.g, items)
        output_nodes = blocks[-1].dstdata[NID]
        input_nodes = blocks[0].srcdata[NID]

        return tuples, labels, subsampling_ws, input_nodes, output_nodes, blocks


class RandomWalkNodeDataLoader:
    collator_arglist = inspect.getfullargspec(RandomWalkNodeCollator).args
    def __init__(self, g, nids, block_sampler, **kwargs):
        collator_kwargs = {}
        dataloader_kwargs = {}
        for k, v in kwargs.items():
            if k in self.collator_arglist:
                collator_kwargs[k] = v
            else:
                dataloader_kwargs[k] = v

        assert not isinstance(g, DistGraph)
        # block sampler is the neighbor sampler
        self.collator = _NodeCollator(g, nids, block_sampler, **collator_kwargs)
        self.dataloader = DataLoader(self.collator.dataset,
                                         collate_fn=self.collator.collate,
                                         **dataloader_kwargs)
        self.is_distributed = False

    def __iter__(self):
        """Return the iterator of the data loader."""
        if self.is_distributed:
            # Directly use the iterator of DistDataLoader, which doesn't copy features anyway.
            return iter(self.dataloader)
        else:
            return _NodeDataLoaderIter(self)

    def __len__(self):
        """Return the number of batches of the data loader."""
        return len(self.dataloader)


class _NodeCollator(RandomWalkNodeCollator):
    def collate(self, items):
        tuples, labels, subsampling_ws, input_nodes, output_nodes, blocks = super().collate(items)
        _pop_blocks_storage(blocks, self.g)
        return tuples, labels, subsampling_ws, input_nodes, output_nodes, blocks


class _NodeDataLoaderIter:
    def __init__(self, node_dataloader):
        self.node_dataloader = node_dataloader
        self.iter_ = iter(node_dataloader.dataloader)

    def __next__(self):
        tuples, labels, subsampling_ws, input_nodes, output_nodes, blocks = next(self.iter_)
        _restore_blocks_storage(blocks, self.node_dataloader.collator.g)
        return tuples, labels, subsampling_ws, input_nodes, output_nodes, blocks

if __name__ == '__main__':
    u, v = torch.tensor([0, 0, 0, 1, 1, 2, 3, 3]), torch.tensor([1, 2, 3, 3, 0, 0, 0, 1])
    g = dgl.graph((u, v))
    g.ndata['p'] = 1 / g.out_degrees()
    true_trace = {2: {0}, 0: {1, 2, 3}, 1: {3, 0}, 3: {0, 1}}
    tuples = torch.cat((u.view((-1, 1)), v.view(-1, 1)), 1)
    base_num = 1
    sampler = RandomWalkMultiLayerNeighborSampler([None], tuples.numpy(), base_num, neg_num=0, true_tuple=true_trace, length=2, windows=1)
    train_nid = g.nodes()
    dataloader = RandomWalkNodeDataLoader(
        g, train_nid, sampler,
        batch_size=1, shuffle=True, drop_last=False, num_workers=4)
    for tuple, label, subsampling_ws, input_nodes, output_nodes, blocks in dataloader:
        print(tuple)
        print(label)
        print(input_nodes)
        print(output_nodes)
        print(blocks)
        break