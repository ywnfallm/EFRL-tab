import os

import torch
from torchmeta.utils.prototype import get_prototypes
from train.metric_based import get_accuracy
from utils import MetricLogger
import numpy as np
import argparse
import torch
import torch.nn as nn

parser = argparse.ArgumentParser(description = 'STUNT')
parser.add_argument('--data_name', default = 'income', type = str)
parser.add_argument('--shot_num', default = 1, type=int)
parser.add_argument('--load_path', default = '', type=str)
parser.add_argument('--seed', default = 0, type = int)
parser.add_argument('--trainseed', default = 0, type = int)
args = parser.parse_args()

# if args.data_name == 'income':
#     input_size = 105
#     output_size = 2
#     hidden_dim = 1024

tabularsInfo = {"kr-vs-kp": {"classNum": 2, "00": 73, "10": 36, "20": 72, "30": 108},
                    "cmc": {"classNum": 3, "00": 24, "10": 9, "20": 16, "30": 23},
                    "credit-approval": {"classNum": 2, "00": 51, "10": 15, "20": 24, "30": 33},
                    "credit-g": {"classNum": 2, "00": 61, "10": 20, "20": 33, "30": 46},
                    "splice": {"classNum": 3, "00": 287, "10": 60, "20": 120, "30": 180},
                    "tic-tac-toe": {"classNum": 2, "00": 27, "10": 9, "20": 18, "30": 27},
                    "electricity": {"classNum": 2, "00": 14, "10": 8, "20": 9, "30": 10},
                    "eucalyptus": {"classNum": 5, "00": 91, "10": 19, "20": 24, "30": 29},
                    "sick": {"classNum": 2, "00": 52, "10": 27, "20": 48, "30": 69},
                    "vowel": {"classNum": 11, "00": 27, "10": 12, "20": 14, "30": 16},
                    "analcatdata_dmft": {"classNum": 6, "00": 21, "10": 4, "20": 8, "30": 12},
                    "adult": {"classNum": 2, "00": 108, "10": 14, "20": 22, "30": 30},
                    "ilpd": {"classNum": 2, "00": 11, "10": 10, "20": 11, "30": 12},
                    "nomao": {"classNum": 2, "00": 174, "10": 118, "20": 147, "30": 176},
                    "PhishingWebsites": {"classNum": 2, "00": 68, "10": 30, "20": 60, "30": 90},
                    "cylinder-bands": {"classNum": 2, "00": 172, "10": 35, "20": 52, "30": 69},
                    "bank-marketing": {"classNum": 2, "00": 51, "10": 16, "20": 25, "30": 34},
                    "dresses-sales": {"classNum": 2, "00": 165, "10": 12, "20": 23, "30": 34},
                    "connect-4": {"classNum": 3, "00": 126, "10": 42, "20": 84, "30": 126},
                    "MiceProtein": {"classNum": 8, "00": 77, "10": 77, "20": 77, "30": 77},
                    "car": {"classNum": 4, "00": 21, "10": 6, "20": 12, "30": 18},
                    "Internet-Advertisements": {"classNum": 2, "00": 3113, "10": 1558, "20": 3113, "30": 4668},
                    "dna": {"classNum": 3, "00": 360, "10": 180, "20": 360, "30": 540},
                    "churn": {"classNum": 2, "00": 33, "10": 20, "20": 24, "30": 28}}
datasetName = args.data_name[:-2]
datasetbhStr = args.data_name[-2:]
if (datasetbhStr <= '00'):
    datasetbh = '00'
elif (datasetbhStr <= '10'):
    datasetbh = '10'
elif (datasetbhStr <= '20'):
    datasetbh = '20'
else:
    datasetbh = '30'

input_size = tabularsInfo[datasetName][datasetbh]
output_size = tabularsInfo[datasetName]["classNum"]
hidden_dim = 1024

class MLPProto(nn.Module):
    def __init__(self, in_features, out_features, hidden_sizes, drop_p = 0.):
        super(MLPProto, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_sizes = hidden_sizes
        self.drop_p = drop_p

        self.encoder = nn.Sequential(
            nn.Linear(in_features, hidden_sizes, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_sizes, hidden_sizes, bias=True)
        )

    def forward(self, inputs):
        embeddings = self.encoder(inputs)
        return embeddings

model = MLPProto(input_size, hidden_dim, hidden_dim)
model.load_state_dict(torch.load(args.load_path))

train_x = np.load('./data/'+args.data_name+'/xtrain.npy')
train_y = np.load('./data/'+args.data_name+'/ytrain.npy')
test_x = np.load('./data/'+args.data_name+'/xtest.npy')
test_y = np.load('./data/'+args.data_name+'/ytest.npy')
train_idx = np.load('./data/'+args.data_name+'/index{}/train_idx_{}.npy'.format(args.shot_num, args.seed))

few_train = model(torch.tensor(train_x[train_idx]).float())
support_x = few_train.detach().numpy()
support_y = train_y[train_idx]
few_test = model(torch.tensor(test_x).float())
query_x = few_test.detach().numpy()
query_y = test_y

def get_accuracy(prototypes, embeddings, targets):

    sq_distances = torch.sum((prototypes.unsqueeze(1)
        - embeddings.unsqueeze(2)) ** 2, dim=-1)
    _, predictions = torch.min(sq_distances, dim=-1)
    return torch.mean(predictions.eq(targets).float()) * 100.

train_x = torch.tensor(support_x.astype(np.float32)).unsqueeze(0)
train_y = torch.tensor(support_y.astype(np.int64)).unsqueeze(0).type(torch.LongTensor)
val_x = torch.tensor(query_x.astype(np.float32)).unsqueeze(0)
val_y = torch.tensor(query_y.astype(np.int64)).unsqueeze(0).type(torch.LongTensor)
prototypes = get_prototypes(train_x, train_y, output_size)
acc = get_accuracy(prototypes, val_x, val_y).item()

print(args.seed, acc)

out_dir = 'result/{}_seed{}_{}shot'.format(args.data_name, args.trainseed, args.shot_num)
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
out_file = '{}/test'.format(out_dir)
with open(out_file, 'a+') as f:
    f.write('seed: '+str(args.seed)+', test: '+str(acc))
    f.write('\n')