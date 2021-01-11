from dgl.dataloading import EdgeCollator, inspect
from dgl.dataloading.pytorch import _EdgeDataLoaderIter, _EdgeCollator
from dgl.distributed import DistGraph
from torch.utils.data import DataLoader
from dgl.base import NID, EID
from dgl import transform
from dgl import utils
from dgl import backend as F
from dgl.convert import heterograph
import numpy as np
from collections.abc import Mapping

class EdgeDataLoader:
    collator_arglist = inspect.getfullargspec(EdgeCollator).args
    def __init__(self, g, eids, block_sampler, **kwargs):
        collator_kwargs = {}
        dataloader_kwargs = {}
        for k, v in kwargs.items():
            if k in self.collator_arglist:
                collator_kwargs[k] = v
            else:
                dataloader_kwargs[k] = v
        self.collator = EdgeCollatorWithEdgeType(g, eids, block_sampler, **collator_kwargs)

        assert not isinstance(g, DistGraph), \
                'EdgeDataLoader does not support DistGraph for now. ' \
                + 'Please use DistDataLoader directly.'
        self.dataloader = DataLoader(
            self.collator.dataset, collate_fn=self.collator.collate, **dataloader_kwargs)

    def __iter__(self):
        """Return the iterator of the data loader."""
        return _EdgeDataLoaderIter(self)

    def __len__(self):
        """Return the number of batches of the data loader."""
        return len(self.dataloader)

class EdgeCollatorWithEdgeType(_EdgeCollator) :
    def collate(self, items):
        if self.negative_sampler is None:
            input_nodes, pair_graph, blocks = self.collate_func(items)
            _pop_subgraph_storage(pair_graph, self.g)
            _pop_blocks_storage(blocks, self.g_sampling)
            return input_nodes, pair_graph, blocks
        else:
            input_nodes, pair_graph, neg_pair_graph, blocks = self.collate_func(items)
            _pop_subgraph_storage(pair_graph, self.g)
            _pop_subgraph_storage(neg_pair_graph, self.g)
            _pop_blocks_storage(blocks, self.g_sampling)
            return input_nodes, pair_graph, neg_pair_graph, blocks

    def collate_func(self, items):
        if self.negative_sampler is None:
            return self._collate(items)
        else:
            return self._collate_with_negative_sampling(items)

    def _collate(self, items):
        if isinstance(items[0], tuple):
            # returns a list of pairs: group them by node types into a dict
            items = utils.group_as_dict(items)
            items = utils.prepare_tensor_dict(self.g_sampling, items, 'items')
        else:
            items = utils.prepare_tensor(self.g_sampling, items, 'items')

        pair_graph = self.g.edge_subgraph(items)
        seed_nodes = pair_graph.ndata[NID]

        exclude_eids = _find_exclude_eids(
            self.g,
            self.exclude,
            items,
            reverse_eid_map=self.reverse_eids,
            reverse_etype_map=self.reverse_etypes)

        blocks = self.block_sampler.sample_blocks(
            self.g_sampling, seed_nodes, exclude_eids=exclude_eids)
        input_nodes = blocks[0].srcdata[NID]

        return input_nodes, pair_graph, blocks




    # TODO add edge type while doing negative sampling
    def _collate_with_negative_sampling(self, items):
        if isinstance(items[0], tuple):
            # returns a list of pairs: group them by node types into a dict
            items = utils.group_as_dict(items)
            items = utils.prepare_tensor_dict(self.g_sampling, items, 'items')
        else:
            items = utils.prepare_tensor(self.g_sampling, items, 'items')

        pair_graph = self.g.edge_subgraph(items, preserve_nodes=True)
        induced_edges = pair_graph.edata[EID]

        neg_srcdst, edge_type, subsampling_w = self.negative_sampler(self.g, items)
        # neg_srcdst, edge_type = self.negative_sampler(self.g, items)
        # neg_srcdst = self.negative_sampler(self.g, items)
        if not isinstance(neg_srcdst, Mapping):
            assert len(self.g.etypes) == 1, \
                'graph has multiple or no edge types; '\
                'please return a dict in negative sampler.'
            neg_srcdst = {self.g.canonical_etypes[0]: neg_srcdst}
        # Get dtype from a tuple of tensors
        dtype = F.dtype(list(neg_srcdst.values())[0][0])
        neg_edges = {
            etype: neg_srcdst.get(etype, (F.tensor([], dtype), F.tensor([], dtype)))
            for etype in self.g.canonical_etypes}
        neg_pair_graph = heterograph(
            neg_edges, {ntype: self.g.number_of_nodes(ntype) for ntype in self.g.ntypes})

        pair_graph, neg_pair_graph = transform.compact_graphs([pair_graph, neg_pair_graph])
        pair_graph.edata[EID] = induced_edges
        num = 0
        for type in neg_pair_graph.canonical_etypes :
            if isinstance(edge_type, dict) :
                neg_pair_graph.edata[type]['etype'] = edge_type[type]
                neg_pair_graph.edata[type]['sw'] = subsampling_w # add sumpling weight
            else :
                neg_pair_graph.edata['etype'] = edge_type[num:num + neg_pair_graph.number_of_edges(type)]
                neg_pair_graph.edata['sw'] = subsampling_w[num:num + neg_pair_graph.number_of_edges(type)] # add sumpling weight
                num += neg_pair_graph.number_of_edges(type)

        seed_nodes = pair_graph.ndata[NID]

        exclude_eids = _find_exclude_eids(
            self.g,
            self.exclude,
            items,
            reverse_eid_map=self.reverse_eids,
            reverse_etype_map=self.reverse_etypes)

        blocks = self.block_sampler.sample_blocks(
            self.g_sampling, seed_nodes, exclude_eids=exclude_eids)

        input_nodes = blocks[0].srcdata[NID]

        return input_nodes, pair_graph, neg_pair_graph, blocks



# The following code is a fix to the PyTorch-specific issue in
# https://github.com/dmlc/dgl/issues/2137
#
# Basically the sampled blocks/subgraphs contain the features extracted from the
# parent graph.  In DGL, the blocks/subgraphs will hold a reference to the parent
# graph feature tensor and an index tensor, so that the features could be extracted upon
# request.  However, in the context of multiprocessed sampling, we do not need to
# transmit the parent graph feature tensor from the subprocess to the main process,
# since they are exactly the same tensor, and transmitting a tensor from a subprocess
# to the main process is costly in PyTorch as it uses shared memory.  We work around
# it with the following trick:
#
# In the collator running in the sampler processes:
# For each frame in the block, we check each column and the column with the same name
# in the corresponding parent frame.  If the storage of the former column is the
# same object as the latter column, we are sure that the former column is a
# subcolumn of the latter, and set the storage of the former column as None.
#
# In the iterator of the main process:
# For each frame in the block, we check each column and the column with the same name
# in the corresponding parent frame.  If the storage of the former column is None,
# we replace it with the storage of the latter column.

def _pop_subframe_storage(subframe, frame):
    for key, col in subframe._columns.items():
        if key in frame._columns and col.storage is frame._columns[key].storage:
            col.storage = None

def _pop_subgraph_storage(subg, g):
    for ntype in subg.ntypes:
        if ntype not in g.ntypes:
            continue
        subframe = subg._node_frames[subg.get_ntype_id(ntype)]
        frame = g._node_frames[g.get_ntype_id(ntype)]
        _pop_subframe_storage(subframe, frame)
    for etype in subg.canonical_etypes:
        if etype not in g.canonical_etypes:
            continue
        subframe = subg._edge_frames[subg.get_etype_id(etype)]
        frame = g._edge_frames[g.get_etype_id(etype)]
        _pop_subframe_storage(subframe, frame)

def _pop_blocks_storage(blocks, g):
    for block in blocks:
        for ntype in block.srctypes:
            if ntype not in g.ntypes:
                continue
            subframe = block._node_frames[block.get_ntype_id_from_src(ntype)]
            frame = g._node_frames[g.get_ntype_id(ntype)]
            _pop_subframe_storage(subframe, frame)
        for ntype in block.dsttypes:
            if ntype not in g.ntypes:
                continue
            subframe = block._node_frames[block.get_ntype_id_from_dst(ntype)]
            frame = g._node_frames[g.get_ntype_id(ntype)]
            _pop_subframe_storage(subframe, frame)
        for etype in block.canonical_etypes:
            if etype not in g.canonical_etypes:
                continue
            subframe = block._edge_frames[block.get_etype_id(etype)]
            frame = g._edge_frames[g.get_etype_id(etype)]
            _pop_subframe_storage(subframe, frame)

def _restore_subframe_storage(subframe, frame):
    for key, col in subframe._columns.items():
        if col.storage is None:
            col.storage = frame._columns[key].storage

def _restore_subgraph_storage(subg, g):
    for ntype in subg.ntypes:
        if ntype not in g.ntypes:
            continue
        subframe = subg._node_frames[subg.get_ntype_id(ntype)]
        frame = g._node_frames[g.get_ntype_id(ntype)]
        _restore_subframe_storage(subframe, frame)
    for etype in subg.canonical_etypes:
        if etype not in g.canonical_etypes:
            continue
        subframe = subg._edge_frames[subg.get_etype_id(etype)]
        frame = g._edge_frames[g.get_etype_id(etype)]
        _restore_subframe_storage(subframe, frame)

def _restore_blocks_storage(blocks, g):
    for block in blocks:
        for ntype in block.srctypes:
            if ntype not in g.ntypes:
                continue
            subframe = block._node_frames[block.get_ntype_id_from_src(ntype)]
            frame = g._node_frames[g.get_ntype_id(ntype)]
            _restore_subframe_storage(subframe, frame)
        for ntype in block.dsttypes:
            if ntype not in g.ntypes:
                continue
            subframe = block._node_frames[block.get_ntype_id_from_dst(ntype)]
            frame = g._node_frames[g.get_ntype_id(ntype)]
            _restore_subframe_storage(subframe, frame)
        for etype in block.canonical_etypes:
            if etype not in g.canonical_etypes:
                continue
            subframe = block._edge_frames[block.get_etype_id(etype)]
            frame = g._edge_frames[g.get_etype_id(etype)]
            _restore_subframe_storage(subframe, frame)



def assign_block_eids(block, frontier):
    """Assigns edge IDs from the original graph to the block.

    See also
    --------
    BlockSampler
    """
    for etype in block.canonical_etypes:
        block.edges[etype].data[EID] = frontier.edges[etype].data[EID][
            block.edges[etype].data[EID]]
    return block

def _tensor_or_dict_to_numpy(ids):
    if isinstance(ids, Mapping):
        return {k: F.zerocopy_to_numpy(v) for k, v in ids.items()}
    else:
        return F.zerocopy_to_numpy(ids)

def _locate_eids_to_exclude(frontier_parent_eids, exclude_eids):
    """Find the edges whose IDs in parent graph appeared in exclude_eids.

    Note that both arguments are numpy arrays or numpy dicts.
    """
    if isinstance(frontier_parent_eids, Mapping):
        result = {
            k: np.isin(frontier_parent_eids[k], exclude_eids[k]).nonzero()[0]
            for k in frontier_parent_eids.keys() if k in exclude_eids.keys()}
        return {k: F.zerocopy_from_numpy(v) for k, v in result.items()}
    else:
        result = np.isin(frontier_parent_eids, exclude_eids).nonzero()[0]
        return F.zerocopy_from_numpy(result)

def _find_exclude_eids_with_reverse_id(g, eids, reverse_eid_map):
    if isinstance(eids, Mapping):
        eids = {g.to_canonical_etype(k): v for k, v in eids.items()}
        exclude_eids = {
            k: F.cat([v, F.gather_row(reverse_eid_map[k], v)], 0)
            for k, v in eids.items()}
    else:
        exclude_eids = F.cat([eids, F.gather_row(reverse_eid_map, eids)], 0)
    return exclude_eids

def _find_exclude_eids_with_reverse_types(g, eids, reverse_etype_map):
    exclude_eids = {g.to_canonical_etype(k): v for k, v in eids.items()}
    reverse_etype_map = {
        g.to_canonical_etype(k): g.to_canonical_etype(v)
        for k, v in reverse_etype_map.items()}
    exclude_eids.update({reverse_etype_map[k]: v for k, v in exclude_eids.items()})
    return exclude_eids

def _find_exclude_eids(g, exclude_mode, eids, **kwargs):
    if exclude_mode is None:
        return None
    elif exclude_mode == 'reverse_id':
        return _find_exclude_eids_with_reverse_id(g, eids, kwargs['reverse_eid_map'])
    elif exclude_mode == 'reverse_types':
        return _find_exclude_eids_with_reverse_types(g, eids, kwargs['reverse_etype_map'])
    else:
        raise ValueError('unsupported mode {}'.format(exclude_mode))
