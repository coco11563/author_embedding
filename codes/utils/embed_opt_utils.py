import pickle

# from codes.model.model import build_test_graph, node_norm_to_edge_norm
# from codes.utils.datareader import init_triples, init_embedding, read_valid_author
import numpy as np
import torch


def model_reproduce(pth_path, model, *args) :
    checkpoint = torch.load(pth_path)
    model.eval()
    model.load_state_dict(checkpoint['state_dict'])
    print(model.parameters())
    embed = model(*args)
    return embed

def embed_opt(embed, index, indice_node) :
    opt_embed = embed[index]
    eval_embed = opt_embed.detach().cpu().numpy()
    ret_dict = dict()
    for ind, i in enumerate(index) :
        i_ = i.item()
        nid = indice_node[i_]
        ret_dict[nid] = eval_embed[ind]
    return ret_dict


if __name__ == '__main__':
    node_indice, indice_node, start_indice, triples, rel_set = init_triples('../../data/graph/author_community.pkl')
    embedding_dict = init_embedding('../../data/graph/author_w2v_embedding.pkl', node_indice)
    valid_author = read_valid_author('../../data/classification/test_author_text_corpus.txt', node_indice)
    node_li = []
    label_li = []
    for k, v in valid_author.items():
        node_li.append(k)
        label_li.append(v)

    eval_node = torch.from_numpy(np.asarray(node_li))
    eval_label = np.asarray(label_li)

    num_rels = len(rel_set)
    num_nodes = len(node_indice.items())

    model = LinkPredict(num_nodes,
                        64,
                        num_rels,
                        num_bases=4,
                        num_hidden_layers=2,
                        dropout=0,
                        use_cuda=1,
                        reg_param=0.01,
                        embed_dict=embedding_dict)

    model_pth = '/home/xiaomeng/jupyter_base/author_embedding/codes/model/model_state.pth'

    edges = np.array(triples)
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

    test_graph, test_rel, test_norm = build_test_graph(
        num_nodes, num_rels, train)
    test_deg = test_graph.in_degrees(
        range(test_graph.number_of_nodes())).float().view(-1, 1)
    test_node_id = torch.arange(0, num_nodes, dtype=torch.long).view(-1, 1)
    test_rel = torch.from_numpy(test_rel)
    test_norm = node_norm_to_edge_norm(test_graph, torch.from_numpy(test_norm).view(-1, 1))

    model.cuda(1)
    test_graph = test_graph.to('cuda:' + str(1))
    test_node_id = test_node_id.cuda()
    test_rel = test_rel.cuda()
    test_norm = test_norm.cuda()

    embed = model_reproduce(model_pth, model, test_graph, test_node_id, test_rel, test_norm)
    #
    # eval_embed = embed[eval_node]
    # eval_embed = eval_embed.detach().cpu().numpy()
    # evaluator(eval_embed, eval_label)

    ret_dict = embed_opt(embed, eval_node, indice_node)
    with open( 'rgcn_embed.pkl', 'wb') as f:
        pickle.dump(ret_dict, f)
    print(ret_dict)

