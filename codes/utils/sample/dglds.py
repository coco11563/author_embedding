import os

import dgl
import numpy as np
import torch
from dgl.data import DGLDataset
from dgl.data.knowledge_graph import _read_dictionary, _read_triplets_as_list, build_knowledge_graph, WN18Dataset, \
    FB15kDataset, FB15k237Dataset
from dgl.data.utils import save_graphs, save_info


class NSFCDataSet(DGLDataset):
    def __init__(self, reverse=False, name='NSFC-F01', raw_dir='../data', force_reload=True, verbose=True,
                 init_type_dict=False):
        path = os.getcwd()
        self.reverse = reverse
        self.name_ = name
        self.type_dict = None
        if init_type_dict:
            self.type_dict = dict()
        # raw_dir=None, hash_key=(), force_reload=False, verbose=False):
        super(NSFCDataSet, self).__init__(self.name_, url="", raw_dir=os.path.join(path, raw_dir),
                                          force_reload=force_reload, verbose=verbose)

    def __getitem__(self, idx):
        r"""Gets the graph object

       Parameters
       -----------
       idx: int
           Item index, FB15k237Dataset has only one graph object

       Return
       -------
       :class:`dgl.DGLGraph`

           The graph contains

           - ``edata['e_type']``: edge relation type
           - ``edata['train_edge_mask']``: positive training edge mask
           - ``edata['val_edge_mask']``: positive validation edge mask
           - ``edata['test_edge_mask']``: positive testing edge mask
           - ``edata['train_mask']``: training edge set mask (include reversed training edges)
           - ``edata['val_mask']``: validation edge set mask (include reversed validation edges)
           - ``edata['test_mask']``: testing edge set mask (include reversed testing edges)
           - ``ndata['ntype']``: node type. All 0 in this dataset
       """
        assert idx == 0, "This dataset has only one graph"
        return self._g

    def __len__(self):
        r"""The number of graphs in the dataset."""
        return super(NSFCDataSet, self).__len__()

    # TODO build hetero graph
    def process(self):
        """
       The original knowledge base is stored in triplets.
       This function will parse these triplets and build the DGLGraph.
       """
        root_path = self.raw_path
        entity_path = os.path.join(root_path, 'entities.dict')
        relation_path = os.path.join(root_path, 'relations.dict')
        train_path = os.path.join(root_path, 'train.txt')
        valid_path = os.path.join(root_path, 'valid.txt')
        test_path = os.path.join(root_path, 'test.txt')
        # hash => index
        self.entity_dict = _read_dictionary(entity_path)
        # type constrain ====  edit in 2020 10 15 21点40分
        self.type_dict = dict()
        self.type_constrain_dict = dict()
        for value, index in self.entity_dict.items():
            if not value.__contains__('#'):  # apply id
                self.type_dict[index] = 'applyid'
                if self.type_constrain_dict.__contains__('applyid'):
                    self.type_constrain_dict['applyid'].add(index)
                else:
                    self.type_constrain_dict['applyid'] = set()
                    self.type_constrain_dict['applyid'].add(index)
            elif value.__contains__('kw#'):
                self.type_dict[index] = 'kw'
                if self.type_constrain_dict.__contains__('kw'):
                    self.type_constrain_dict['kw'].add(index)
                else:
                    self.type_constrain_dict['kw'] = set()
                    self.type_constrain_dict['kw'].add(index)
            elif value.__contains__('application'):
                self.type_dict[index] = 'application'
                if self.type_constrain_dict.__contains__('application'):
                    self.type_constrain_dict['application'].add(index)
                else:
                    self.type_constrain_dict['application'] = set()
                    self.type_constrain_dict['application'].add(index)
            elif value.__contains__('ros#'):
                self.type_dict[index] = 'ros'
                if self.type_constrain_dict.__contains__('ros'):
                    self.type_constrain_dict['ros'].add(index)
                else:
                    self.type_constrain_dict['ros'] = set()
                    self.type_constrain_dict['ros'].add(index)
            else:
                raise Exception('wrong type {} name is {}'.format(index, value))

        # type constrain done
        self.relation_dict = _read_dictionary(relation_path)
        train = np.asarray(_read_triplets_as_list(train_path, self.entity_dict, self.relation_dict))
        valid = np.asarray(_read_triplets_as_list(valid_path, self.entity_dict, self.relation_dict))
        test = np.asarray(_read_triplets_as_list(test_path, self.entity_dict, self.relation_dict))
        num_nodes = len(self.entity_dict)
        num_rels = len(self.relation_dict)
        if self.verbose:
            print("# entities: {}".format(num_nodes))
            print("# relations: {}".format(num_rels))
            print("# training edges: {}".format(train.shape[0]))
            print("# validation edges: {}".format(valid.shape[0]))
            print("# testing edges: {}".format(test.shape[0]))

        # for compatability
        self._train = train
        self._valid = valid
        self._test = test

        self._num_nodes = num_nodes
        self._num_rels = num_rels
        # build graph
        g, data = build_knowledge_graph(num_nodes, num_rels, train, valid, test, reverse=self.reverse)
        etype, ntype, train_edge_mask, valid_edge_mask, test_edge_mask, train_mask, val_mask, test_mask = data
        g.edata['train_edge_mask'] = train_edge_mask
        g.edata['valid_edge_mask'] = valid_edge_mask
        g.edata['test_edge_mask'] = test_edge_mask
        g.edata['train_mask'] = train_mask
        g.edata['val_mask'] = val_mask
        g.edata['test_mask'] = test_mask
        g.edata['etype'] = etype
        g.ndata['ntype'] = ntype
        print(type(g))
        self._g = g

    def has_cache(self):
        graph_path = os.path.join(self.save_path,
                                  self.save_name + '.bin')
        info_path = os.path.join(self.save_path,
                                 self.save_name + '.pkl')
        if os.path.exists(graph_path) and \
                os.path.exists(info_path):
            return True

        return False

    def save(self):
        """save the graph list and the labels"""
        graph_path = os.path.join(self.save_path,
                                  self.save_name + '.bin')
        info_path = os.path.join(self.save_path,
                                 self.save_name + '.pkl')
        save_graphs(str(graph_path), self._g)
        save_info(str(info_path), {'num_nodes': self.num_nodes,
                                   'num_rels': self.num_rels})

    def has_cache(self):
        r"""Overwrite to realize your own logic of
       deciding whether there exists a cached dataset.

       By default False.
       """
        return True

    def load(self):
        raise Exception('this dataset need generate constrain dict, load cannot perform this action')

    @property
    def num_nodes(self):
        return self._num_nodes

    @property
    def num_rels(self):
        return self._num_rels

    @property
    def save_name(self):
        return self.name + '_dgl_graph'

    @property
    def train(self):
        # deprecate_property('dataset.train', 'g.edata[\'train_mask\']')
        return self._train

    @property
    def valid(self):
        # deprecate_property('dataset.valid', 'g.edata[\'val_mask\']')
        return self._valid

    @property
    def test(self):
        # deprecate_property('dataset.test', 'g.edata[\'test_mask\']')
        return self._test


def load_data(dataset):
    r"""Load knowledge graph dataset for RGCN link prediction tasks

    It supports three datasets: wn18, FB15k and FB15k-237

    Parameters
    ----------
    dataset: str
        The name of the dataset to load.

    Return
    ------
    The dataset object.
    """
    if dataset == 'wn18':
        return WN18Dataset()
    elif dataset == 'FB15k':
        return FB15kDataset()
    elif dataset == 'FB15k-237':
        return FB15k237Dataset()
    elif dataset == 'NSFC':
        return NSFCDataSet(name='NSFC-F01')


def build_sub_graph(g, mask, preserve_nodes=False):
    set = torch.arange(g.number_of_edges())[mask]
    edges = set
    sub_g = g.edge_subgraph(edges,
                            preserve_nodes=preserve_nodes)
    return sub_g


def one_shot_iterator(dataloader):
    '''
        Transform a PyTorch Dataloader into python iterator
        '''
    while True:
        for data in dataloader:
            yield data

if __name__ == '__main__':
    path = os.getcwd()
    # with open('/home/shaow/jupyter_base/KGE_RGCN_PAPER_CODE/data/win18/entities.dict') as f:
    #     print(f)
    # a = NSFCDataSet()
    # print(a)
    B = NSFCDataSet(name='NSFC-F01')
    # print(B.entity_dict)
    # print(B._g)
    g = B._g
    train_mask = g.edata['train_mask']
    train_g = build_sub_graph(g, train_mask)
    e_id = train_g.edges(form='eid')
    train_g.ndata['feature'] = torch.randn((train_g.number_of_nodes(), 128))
    # print(train_g)
    # for i in train_g.etypes:
    #     e_id[i] = train_g.edges(etype=i, form='eid')
    # print(e_id)
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
    dataloader = dgl.dataloading.EdgeDataLoader(
        train_g, e_id, sampler,
        negative_sampler=dgl.dataloading.negative_sampler.Uniform(256),
        batch_size=1024,
        shuffle=True,
        drop_last=False,
        num_workers=32)
    count = 0
    for input_nodes, positive_graph, negative_graph, blocks in dataloader:
        count += 1
    print(count)
    etype = g.edata['etype']
    import dgl.function as Fn


    def message_func(edges):
        return {'dstntype': edges.data['etype'] + 1}


    g.apply_edges(Fn.copy_e('etype', 'srcntype'))
    g.apply_edges(message_func)
    print(g.ndata['ntype'])
