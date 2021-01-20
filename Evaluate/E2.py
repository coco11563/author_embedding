import argparse

from sklearn import metrics
import sklearn.preprocessing as preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.cluster import KMeans
import numpy as np

import sys

sys.path.append('/home/qiaozy/Author_profiling')
# from bert_pytorch.dataset import WordVocab

import pickle
import tqdm

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn as nn
from torch.optim import Adam


def load_data(corpus_path, encoding="utf-8"):
    author_labels = []
    authors = []
    with open(corpus_path, "r", encoding=encoding) as f:
        # for i, line in enumerate(tqdm.tqdm(f, desc="Loading Dataset")):
        for i, line in enumerate(f):
            author, words, label = line.replace("\n", "").split("\t")
            authors.append(author)
            author_labels.append(int(label))

    return authors, author_labels


def load_embedding(authors, embs):
    with open(embs, 'rb') as f:
        author_embeddings = pickle.load(f)
    embeddings = []
    for a in authors:
        embeddings.append(author_embeddings[a])
    return embeddings


class ClassificationDataset(Dataset):
    def __init__(self, data, author_labels):
        self.author_labels = author_labels
        self.data = data
        self.corpus_lines = len(self.data)

    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, item):
        output = {"input": self.data[item],
                  "label": self.author_labels[item]}

        return {key: torch.tensor(value) for key, value in output.items()}


class CModel(nn.Module):
    def __init__(self, hidden=64, label_num=4):
        super().__init__()

        self.linear = nn.Linear(hidden, hidden)
        self.linear1 = nn.Linear(hidden, label_num)
        self.softmax = nn.LogSoftmax(dim=-1)

        self.criterion = nn.NLLLoss()

    def forward(self, data):
        author_embeddings = data["input"]
        author_output = self.softmax(self.linear1(self.linear(author_embeddings)))
        loss = self.criterion(author_output, data["label"])

        return loss, author_output


class Trainer:
    def __init__(self, hidden=64, label_num: int = 4,
                 lr: float = 1e-3, weight_decay: float = 1e-7, with_cuda: bool = True, log_freq: int = 10):

        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:1" if cuda_condition else "cpu")

        self.model = CModel(hidden, label_num).to(self.device)

        self.log_freq = log_freq

        self.optim = Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

    def train(self, epoch, data_loader):
        str_code = "train"
        # data_iter = tqdm.tqdm(enumerate(data_loader),
        #                       desc="EP_%s:%d" % (str_code, epoch),
        #                       total=len(data_loader),
        #                       bar_format="{l_bar}{r_bar}")
        data_iter = enumerate(data_loader)
        avg_loss = 0.0
        total_correct = 0
        total_element = 0

        pre_label = []
        ture_label = []

        self.model.train()

        for i, data in data_iter:
            # 0. batch_data will be sent into the device(GPU or cpu)
            data = {key: value.to(self.device) for key, value in data.items()}

            loss, author_output = self.model(data)

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            avg_loss += loss.item()

            correct = author_output.argmax(dim=-1).eq(data["label"]).sum().item()
            avg_loss += loss.item()
            total_correct += correct
            total_element += data["label"].nelement()

            ture_label.extend(data["label"].cpu().detach().numpy())
            pre_label.extend(author_output.argmax(dim=-1).cpu().detach().numpy())

            post_fix = {
                "epoch": epoch,
                "iter": i,
                "avg_loss": avg_loss / (i + 1),
                "avg_acc": total_correct / total_element * 100,
                "loss": loss.item()
            }

            # if i % self.log_freq == 0:
            #    data_iter.write(str(post_fix))
        MicroF1 = f1_score(ture_label, pre_label, average='micro')
        MacroF1 = f1_score(ture_label, pre_label, average='macro')
        total_acc = total_correct * 100.0 / total_element
        # print("EP%d_%s, avg_loss=" % (epoch, str_code), avg_loss / len(data_iter), 'Micro_F1=', MicroF1, 'Macro_F1=',
        #       MacroF1)
        return total_acc

    def evaluator(self, embs, label):
        estimator = KMeans(n_clusters=4)
        estimator.fit(embs)
        label_pred = estimator.labels_

        print('NMI:%.4f' % metrics.normalized_mutual_info_score(label, label_pred))
        print('ARI:%.4f' % metrics.adjusted_rand_score(label, label_pred))

        scaler = preprocessing.StandardScaler()
        X = scaler.fit_transform(embs)
        X = np.mat(X)
        train_X, test_X, train_y, test_y = train_test_split(X, label, test_size=0.6)
        model = LogisticRegression()
        model.fit(train_X, train_y)
        pred_y = model.predict(test_X)

        print('Micro-F1: %.4f' % f1_score(test_y, pred_y, average='micro'))
        print('Macro-F1: %.4f' % f1_score(test_y, pred_y, average='macro'))

    def predict(self, epoch, data_loader, str_code):
        # str_code = "train"
        # data_iter = tqdm.tqdm(enumerate(data_loader),
        #                       desc="EP_%s:%d" % (str_code, epoch),
        #                       total=len(data_loader),
        #                       bar_format="{l_bar}{r_bar}")
        data_iter = enumerate(data_loader)
        avg_loss = 0.0
        total_correct = 0
        total_element = 0

        ture_label = []
        pre_label = []

        self.model.eval()

        for i, data in data_iter:
            # 0. batch_data will be sent into the device(GPU or cpu)
            data = {key: value.to(self.device) for key, value in data.items()}

            _, author_output = self.model(data)

            correct = author_output.argmax(dim=-1).eq(data["label"]).sum().item()
            total_correct += correct
            total_element += data["label"].nelement()

            ture_label.extend(data["label"].cpu().detach().numpy())
            pre_label.extend(author_output.argmax(dim=-1).cpu().detach().numpy())

            post_fix = {
                "epoch": epoch,
                "iter": i,
                "avg_acc": total_correct / total_element * 100,
            }

            # if i % self.log_freq == 0:
            #    data_iter.write(str(post_fix))

        total_acc = total_correct * 100.0 / total_element
        MicroF1 = f1_score(ture_label, pre_label, average='micro')
        MacroF1 = f1_score(ture_label, pre_label, average='macro')
        # print("EP%d_%s, " % (epoch, str_code), 'Micro_F1=', MicroF1, 'Macro_F1=', MacroF1)

        # self.evaluator(np.array(all_embeddings), np.array(labels))
        return total_acc, MicroF1, MacroF1


def train(args,
          train_data, train_y, valid_data, valid_y, test_data, test_y):

    # print("Loading Train Dataset")
    train_dataset = ClassificationDataset(train_data, train_y)
    # print("Loading valid Dataset")
    valid_dataset = ClassificationDataset(valid_data, valid_y)
    # print("Loading valid Dataset")
    test_dataset = ClassificationDataset(test_data, test_y)

    # print("Creating Dataloader")
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    valid_data_loader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    # print("Creating BERT Trainer")
    trainer = Trainer(hidden=64, label_num=4)

    # print("Training Start")
    best_valid_acc = -1
    result = ""
    test_acc = trainer.predict(-1, test_data_loader, "test")
    for epoch in range(args.epochs):
        train_acc = trainer.train(epoch, train_data_loader)
        valid_acc, _, _ = trainer.predict(epoch, valid_data_loader, "valid")
        test_acc, MicroF1, MacroF1 = trainer.predict(epoch, test_data_loader, "test")

        if valid_acc > best_valid_acc:
            result = [MicroF1, MacroF1]
            best_valid_acc = valid_acc

    # print("result:", str(result))
    return result

if __name__ == '__main__':
    best_Mi = [-1, -1, -1, -1, -1, -1]
    best_Ma = [-1, -1, -1, -1, -1, -1]
    best_A_epoch = [-1, -1, -1]
    best_I_epoch = [-1, -1, -1]
    authors, label = load_data('/home/xiaomeng/jupyter_base/author_embedding/data/test_author_text_corpus.txt')


    for i in range(200) :
        epoch = i
        print('epoch : {}'.format(epoch))
        parser = argparse.ArgumentParser()
        author_embedding = load_embedding(authors,
                                          '/home/xiaomeng/jupyter_base/author_embedding/codes/GCN/gcn_embed_{}.pkl'.format(
                                              epoch))




        parser.add_argument("-c", "--dataset", default= '/home/xiaomeng/jupyter_base/author_embedding/data/test_author_text_corpus.txt', type=str, help="classification dataset address")
        parser.add_argument("-e", "--epochs", type=int, default=10, help="number of epochs")
        # parser.add_argument("-v", "--embs", type=str, default='/home/xiaomeng/jupyter_base/author_embedding/codes/GCN/gcn_embed_{}.pkl'.format(epoch),help="researcher embeddings")
        parser.add_argument("-w", "--num_workers", type=int, default=5, help="dataloader worker size")
        parser.add_argument("-s", "--seq_len", type=int, default=200, help="maximum sequence len")
        parser.add_argument("-b", "--batch_size", type=int, default=32, help="number of batch_size")

        args = parser.parse_args()
        train_data, test_data, train_y, test_y = train_test_split(author_embedding, label, test_size=0.6)
        train_data, valid_data, train_y, valid_y = train_test_split(train_data, train_y, test_size=0.25)
        result_2 = train(args, train_data, train_y, valid_data, valid_y, test_data, test_y)

        MicroF1 = result_2[0]
        MacroF1 = result_2[1]
        if best_Mi[4] < MicroF1:
            print('best_MI 30/10/60 found in round {}'.format(epoch))
            best_Mi[4] = result_2[0]
            best_Mi[5] = result_2[1]
            best_I_epoch[2] = epoch

        if best_Ma[5] < MacroF1:
            print('best_MA 30/10/60 found in round {}'.format(epoch))
            best_Ma[4] = result_2[0]
            best_Ma[5] = result_2[1]
            best_A_epoch[2] = epoch
        print('best  MI:' ,best_Mi)
        print('best  MA:' ,best_Ma)