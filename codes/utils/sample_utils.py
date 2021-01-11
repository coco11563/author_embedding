"""
Utility functions for link prediction
Most codes is adapted from authors' implementation of RGCN link prediction:
https://github.com/MichSchli/RelationPrediction

"""
from concurrent.futures.thread import ThreadPoolExecutor
from multiprocessing import Queue, Manager
from multiprocessing import Lock

import numpy as np
import tensorflow as tf
import dgl
import threading
from queue import PriorityQueue
from multiprocessing import Pool
#######################################################################
#
# Utility function for building training and testing graphs
#
#######################################################################

def get_adj_and_degrees(num_nodes, triplets):
    """ Get adjacency list and degrees of the graph
    """
    adj_list = [[] for _ in range(num_nodes)]
    for i,triplet in enumerate(triplets):
        adj_list[triplet[0]].append([i, triplet[2]])
        adj_list[triplet[2]].append([i, triplet[0]])

    degrees = np.array([len(a) for a in adj_list])
    adj_list = [np.array(a) for a in adj_list]
    return adj_list, degrees

def sample_edge_neighborhood(adj_list, degrees, n_triplets, sample_size):
    """Sample edges by neighborhood expansion.

    This guarantees that the sampled edges form a connected graph, which
    may help deeper GNNs that require information from more than one hop.
    """
    edges = np.zeros((sample_size), dtype=np.int32)

    #initialize
    sample_counts = np.array([d for d in degrees])
    picked = np.array([False for _ in range(n_triplets)])
    seen = np.array([False for _ in degrees])

    for i in range(0, sample_size):
        weights = sample_counts * seen

        if np.sum(weights) == 0:
            weights = np.ones_like(weights)
            weights[np.where(sample_counts == 0)] = 0

        probabilities = (weights) / np.sum(weights)
        chosen_vertex = np.random.choice(np.arange(degrees.shape[0]),
                                         p=probabilities)
        chosen_adj_list = adj_list[chosen_vertex]
        seen[chosen_vertex] = True

        chosen_edge = np.random.choice(np.arange(chosen_adj_list.shape[0]))
        chosen_edge = chosen_adj_list[chosen_edge]
        edge_number = chosen_edge[0]

        while picked[edge_number]:
            chosen_edge = np.random.choice(np.arange(chosen_adj_list.shape[0]))
            chosen_edge = chosen_adj_list[chosen_edge]
            edge_number = chosen_edge[0]

        edges[i] = edge_number
        other_vertex = chosen_edge[1]
        picked[edge_number] = True
        sample_counts[chosen_vertex] -= 1
        sample_counts[other_vertex] -= 1
        seen[other_vertex] = True

    return edges

def sample_edge_uniform(adj_list, degrees, n_triplets, sample_size):
    """Sample edges uniformly from all the edges."""
    all_edges = np.arange(n_triplets)
    return np.random.choice(all_edges, sample_size, replace=False)


def sample_edge_random_walk(adj_list, degrees, n_triplets, sample_size):
    edges = np.zeros((sample_size), dtype=np.int32)

    # initialize
    sample_counts = np.array([d for d in degrees])
    picked = np.array([False for _ in range(n_triplets)])
    seen = np.array([False for _ in degrees])

    for i in range(0, sample_size):
        weights = sample_counts * seen

        if np.sum(weights) == 0:
            weights = np.ones_like(weights)
            weights[np.where(sample_counts == 0)] = 0

        probabilities = (weights) / np.sum(weights)
        chosen_vertex = np.random.choice(np.arange(degrees.shape[0]),
                                         p=probabilities)
        chosen_adj_list = adj_list[chosen_vertex]
        seen[chosen_vertex] = True

        chosen_edge = np.random.choice(np.arange(chosen_adj_list.shape[0]))
        chosen_edge = chosen_adj_list[chosen_edge]
        edge_number = chosen_edge[0]

        while picked[edge_number]:
            chosen_edge = np.random.choice(np.arange(chosen_adj_list.shape[0]))
            chosen_edge = chosen_adj_list[chosen_edge]
            edge_number = chosen_edge[0]

        edges[i] = edge_number
        other_vertex = chosen_edge[1]
        picked[edge_number] = True
        sample_counts[chosen_vertex] -= 1
        sample_counts[other_vertex] -= 1
        seen[other_vertex] = True

    return edges


def generate_sampled_graph_and_labels(triplets, sample_size, split_size,
                                      num_rels, adj_list, degrees,
                                      negative_rate, sampler="uniform"):
    """Get training graph and signals
    First perform edge neighborhood sampling on graph, then perform negative
    sampling to generate negative samples
    """
    # perform edge neighbor sampling
    if sampler == "uniform":
        edges = sample_edge_uniform(adj_list, degrees, len(triplets), sample_size)
    elif sampler == "neighbor":
        edges = sample_edge_neighborhood(adj_list, degrees, len(triplets), sample_size)
    else:
        raise ValueError("Sampler type must be either 'uniform' or 'neighbor' or 'randomwalk'.")

    # relabel nodes to have consecutive node ids
    edges = triplets[edges]
    src, rel, dst = edges.transpose()
    uniq_v, edges = np.unique((src, dst), return_inverse=True)
    src, dst = np.reshape(edges, (2, -1))
    relabeled_edges = np.stack((src, rel, dst)).transpose()

    # negative sampling
    samples, labels = negative_sampling(relabeled_edges, len(uniq_v),
                                        negative_rate)

    # further split graph, only half of the edges will be used as graph
    # structure, while the rest half is used as unseen positive samples
    split_size = int(sample_size * split_size)
    graph_split_ids = np.random.choice(np.arange(sample_size),
                                       size=split_size, replace=False)
    src = src[graph_split_ids]
    dst = dst[graph_split_ids]
    rel = rel[graph_split_ids]

    # build DGL graph
    # print("# sampled nodes: {}".format(len(uniq_v)))
    # print("# sampled edges: {}".format(len(src) * 2))

    g, rel, norm = build_graph_from_triplets(len(uniq_v), num_rels,
                                             (src, rel, dst))
    return g, uniq_v, rel, norm, samples, labels

def comp_deg_norm(g):
    g = g.local_var()
    in_deg = g.in_degrees(range(g.number_of_nodes())).float().numpy()
    norm = 1.0 / in_deg
    norm[np.isinf(norm)] = 0
    return norm

def build_graph_from_triplets(num_nodes, num_rels, triplets):
    """ Create a DGL graph. The graph is bidirectional because RGCN authors
        use reversed relations.
        This function also generates edge type and normalization factor
        (reciprocal of node incoming degree)
    """
    g = dgl.DGLGraph()
    g.add_nodes(num_nodes)
    src, rel, dst = triplets
    src, dst = np.concatenate((src, dst)), np.concatenate((dst, src))
    rel = np.concatenate((rel, rel + num_rels))
    edges = sorted(zip(dst, src, rel))
    dst, src, rel = np.array(edges).transpose()
    g.add_edges(src, dst)
    norm = comp_deg_norm(g)
    # print("# nodes: {}, # edges: {}".format(num_nodes, len(src)))
    return g, rel, norm

def build_test_graph(num_nodes, num_rels, edges):
    src, rel, dst = edges.transpose()
    print("Test graph:")
    return build_graph_from_triplets(num_nodes, num_rels, (src, rel, dst))

def negative_sampling(pos_samples, num_entity, negative_rate):
    size_of_batch = len(pos_samples)
    num_to_generate = size_of_batch * negative_rate
    neg_samples = np.tile(pos_samples, (negative_rate, 1))
    labels = np.zeros(size_of_batch * (negative_rate + 1), dtype=np.float32)
    labels[: size_of_batch] = 1
    values = np.random.randint(num_entity, size=num_to_generate)
    choices = np.random.uniform(size=num_to_generate)
    subj = choices > 0.5
    obj = choices <= 0.5
    neg_samples[subj, 0] = values[subj]
    neg_samples[obj, 2] = values[obj]

    return np.concatenate((pos_samples, neg_samples)), labels
class MultiThreadSamplerPool(threading.Thread) :
    def __init__(self, train_data, graph_batch_size, graph_split_size,
                num_rels, adj_list, degrees, negative_sample,
                edge_sampler, max_runtime ,max_worker = 16, max_seq = 128):
        super().__init__()
        # self.executor = ThreadPoolExecutor(max_worker)
        # self.lock = Lock()
        # self.q = Queue(maxsize=max_seq)
        self.max_runtime = max_runtime
        self.data = train_data
        self.graph_batch_size = graph_batch_size
        self.graph_split_size = graph_split_size
        self.num_rels = num_rels
        self.adj_list = adj_list
        self.degrees = degrees
        self.negative_sample = negative_sample
        self.edge_sampler = edge_sampler
        self.max_worker = max_worker
        self.manager = Manager()

        self.setDaemon(True)
        # 父进程创建Queue，并传给各个子进程：
        self.q = self.manager.Queue(maxsize=max_seq)
        self.ctx = self.manager.Lock()
        self.pool = Pool(processes=self.max_worker)
        self.count = 0
        self.start()

    def run(self):
        while self.count <= self.max_runtime:
            while self.q.full() :
                pass
            # task = MultiThreadSampler(self.data, self.graph_batch_size, self.graph_split_size,
            #                           self.num_rels, self.adj_list, self.degrees, self.negative_sample,
            #                           self.edge_sampler, self.q, self.lock)
            self.pool.apply_async(self.process, args=(self.q, self.data, self.graph_batch_size, self.graph_split_size,
                                               self.num_rels, self.adj_list, self.degrees, self.negative_sample,
                                               self.edge_sampler))
            self.count += 1
            # print(self.q.qsize())
            # self.q.put(ret.get())
            # self.q.put(ret.get())

    def get(self):
        ret = self.q.get()
        return ret

    @staticmethod
    def process(q, triplets, sample_size, split_size,
                                      num_rels, adj_list, degrees,
                                      negative_rate, sampler="uniform") :
        q.put(generate_sampled_graph_and_labels(
            triplets, sample_size, split_size,
            num_rels, adj_list, degrees,
            negative_rate, sampler))

class MultiThreadEvaluatingPool(threading.Thread) :
    def __init__(self):
        super(MultiThreadEvaluatingPool, self).__init__()