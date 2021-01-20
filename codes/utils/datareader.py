import pickle

import dgl
import numpy


def init_triples(community_path):
    node_indice = dict()
    indice_node = dict()
    start_indice = 0
    triples = set()
    rel_set = set()
    with open(community_path, 'rb') as f:
        entity_dict = pickle.load(file=f)
        for node, inform in entity_dict.items():
            if node_indice.__contains__(node):
                indice = node_indice[node]
            else:
                indice = start_indice
                node_indice[node] = indice
                start_indice += 1
            neighbor = inform['neighbors']
            neighbor_map_list = []
            for i in neighbor:
                if node_indice.__contains__(i):
                    indice_ = node_indice[i]
                else:
                    indice_ = start_indice
                    node_indice[i] = indice_
                    start_indice += 1
                neighbor_map_list.append(indice_)
            indice = numpy.repeat([indice], len(neighbor_map_list))

            rel = inform['relations']
            for i in rel:
                rel_set.add(i)

            assert len(rel) == len(indice) and len(indice) == len(neighbor_map_list)
            for index, head in enumerate(indice):
                triples.add((head, rel[index], neighbor_map_list[index]))
    # inverse index
    for k, v in node_indice.items():
        indice_node[v] = k

    return node_indice, indice_node, start_indice, list(triples), rel_set

def init_triples_with_trace(community_path, reverse = True):
    node_indice = dict()
    indice_node = dict()
    start_indice = 0
    triples = set()
    rel_set = set()
    true_trace = dict()
    with open(community_path, 'rb') as f:
        entity_dict = pickle.load(file=f)
        for node, inform in entity_dict.items():
            if node_indice.__contains__(node):
                indice = node_indice[node]
            else:
                indice = start_indice
                node_indice[node] = indice
                start_indice += 1
            neighbor = inform['neighbors']
            neighbor_map_list = []
            for i in neighbor:
                if node_indice.__contains__(i):
                    indice_ = node_indice[i]
                else:
                    indice_ = start_indice
                    node_indice[i] = indice_
                    start_indice += 1
                neighbor_map_list.append(indice_)
            if reverse :
                if true_trace.__contains__(indice) :
                    nodeset = true_trace[indice]
                    for i in neighbor_map_list :
                        nodeset.add(i)
                    true_trace[indice] = nodeset
                else :
                    true_trace[indice] = set(neighbor_map_list)
            else :
                true_trace[indice] = set(neighbor_map_list)
            if reverse :
                for i in neighbor_map_list :
                    if true_trace.__contains__(i) :
                        node_set = true_trace[i]
                        node_set.add(indice)
                        true_trace[i] = node_set
                    else :
                        node_set = set()
                        node_set.add(indice)
                        true_trace[i] = node_set

            indice = numpy.repeat([indice], len(neighbor_map_list))
            rel = inform['relations']
            for i in rel:
                rel_set.add(i)


            assert len(rel) == len(indice) and len(indice) == len(neighbor_map_list)
            for index, head in enumerate(indice):
                triples.add((head, rel[index], neighbor_map_list[index]))
    # inverse index
    for k, v in node_indice.items():
        indice_node[v] = k

    return node_indice, indice_node, start_indice, list(triples), rel_set, true_trace

def init_embedding(embedding_path, node_indice):
    with open(embedding_path, 'rb') as f:
        entity_embedding = pickle.load(file=f)
        embedding_dict = dict()
        for k, v in node_indice.items():
            embedding = entity_embedding[k]
            embedding_dict[v] = embedding
    return embedding_dict


def read_valid_author(test_author_path, node_indice) :
    valid_author = dict()
    with open(test_author_path, 'r') as f:
        line = f.readline()
        while line:
            line_sp = line.split('\t')
            assert len(line_sp) == 3
            node = line_sp[0]
            label = int(line_sp[-1])
            indice = node_indice[node]
            valid_author[indice] = label
            line = f.readline()
    return valid_author


if __name__ == '__main__':
    # node_indice, indice_node, start_indice, triples, rel_set = init_triples('../../data/graph/author_community.pkl')
    node_indice, indice_node, start_indice, triples, rel_set, trace = init_triples_with_trace('../../data/graph/author_community.pkl')
    print(trace)
    # train = numpy.array(triples)
    # train_dst = train[:, 1]
    # train_src = train[:, 0]
    # g = dgl.graph((train_dst, train_src), num_nodes = len(indice_node))
    # print(g)
    # print(len(indice_node))
    # print(g.nodes())
    # print(len(node_indice.items()))
    # print(len(rel_set))
    # print(rel_set)
    # embedding_dict = init_embedding('../../data/graph/author_w2v_embedding.pkl', node_indice)
    # print(len(embedding_dict.items()))
    # for k,v in embedding_dict.items() :
    #     print(v.shape)
    #     break
    # valid_author = read_valid_author('../../data/classification/test_author_text_corpus.txt', node_indice)
    # print(len(valid_author.items()))
    # for i in valid_author :
    #     print(i)
    #     assert embedding_dict.__contains__(i)
    #     break