import torch
from torchvision import transforms

from torchmeta.transforms import ClassSplitter, Categorical

from data.generaldataset import GeneralDataset

def get_meta_dataset(P, dataset, only_test=False):
    datasetName = dataset[:-2]
    datasetbhStr = dataset[-2:]

    if (datasetbhStr >= '00'):
        if (datasetbhStr <= '00'):
            datasetbh = '00'
        elif (datasetbhStr <= '10'):
            datasetbh = '10'
        elif (datasetbhStr <= '20'):
            datasetbh = '20'
        elif (datasetbhStr <= '30'):
            datasetbh = '30'
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()

    tabularsInfo = {"kr-vs-kp":{"classNum":2,"00":73,"10":36,"20":72,"30":108},
                    "cmc":{"classNum":3,"00":24,"10":9,"20":16,"30":23},
                    "credit-approval":{"classNum":2,"00":51,"10":15,"20":24,"30":33},
                    "credit-g":{"classNum":2,"00":61,"10":20,"20":33,"30":46},
                    "splice":{"classNum":3,"00":287,"10":60,"20":120,"30":180},
                    "tic-tac-toe":{"classNum":2,"00":27,"10":9,"20":18,"30":27},
                    "electricity":{"classNum":2,"00":14,"10":8,"20":9,"30":10},
                    "eucalyptus":{"classNum":5,"00":91,"10":19,"20":24,"30":29},
                    "sick":{"classNum":2,"00":52,"10":27,"20":48,"30":69},
                    "vowel":{"classNum":11,"00":27,"10":12,"20":14,"30":16},
                    "analcatdata_dmft":{"classNum":6,"00":21,"10":4,"20":8,"30":12},
                    "adult":{"classNum":2,"00":108,"10":14,"20":22,"30":30},
                    "ilpd":{"classNum":2,"00":11,"10":10,"20":11,"30":12},
                    "nomao":{"classNum":2,"00":174,"10":118,"20":147,"30":176},
                    "PhishingWebsites":{"classNum":2,"00":68,"10":30,"20":60,"30":90},
                    "cylinder-bands":{"classNum":2,"00":172,"10":35,"20":52,"30":69},
                    "bank-marketing":{"classNum":2,"00":51,"10":16,"20":25,"30":34},
                    "dresses-sales":{"classNum":2,"00":165,"10":12,"20":23,"30":34},
                    "connect-4":{"classNum":3,"00":126,"10":42,"20":84,"30":126},
                    "MiceProtein":{"classNum":8,"00":77,"10":77,"20":77,"30":77},
                    "car":{"classNum":4,"00":21,"10":6,"20":12,"30":18},
                    "Internet-Advertisements":{"classNum":2,"00":3113,"10":1558,"20":3113,"30":4668},
                    "dna":{"classNum":3,"00":360,"10":180,"20":360,"30":540},
                    "churn":{"classNum":2,"00":33,"10":20,"20":24,"30":28}}

    if (not (datasetName in tabularsInfo.keys())):
        raise NotImplementedError()

    meta_train_dataset = GeneralDataset(dataset=dataset,
                                tabular_size=tabularsInfo[datasetName][datasetbh],
                                seed=P.seed,
                                source='train',
                                shot=P.num_shots,
                                tasks_per_batch=P.batch_size,
                                test_num_way=P.num_ways,
                                query=P.num_shots_test)

    meta_val_dataset = GeneralDataset(dataset=dataset,
                              tabular_size=tabularsInfo[datasetName][datasetbh],
                              seed=P.seed,
                              source='val',
                              shot=1,
                              tasks_per_batch=P.test_batch_size,
                              test_num_way=2,
                              query=P.num_shots_test) # query=30

    return meta_train_dataset, meta_val_dataset

# def get_meta_dataset(P, dataset, only_test=False):
#
#     if dataset == 'income':
#         meta_train_dataset = Income(tabular_size = 105,
#                                     seed=P.seed,
#                                     source='train',
#                                     shot=P.num_shots,
#                                     tasks_per_batch=P.batch_size,
#                                     test_num_way = P.num_ways,
#                                     query = P.num_shots_test)
#
#         meta_val_dataset = Income(tabular_size = 105,
#                                     seed=P.seed,
#                                     source='val',
#                                     shot=1,
#                                     tasks_per_batch=P.test_batch_size,
#                                     test_num_way = 2,
#                                     query = 30)
#
#     else:
#         raise NotImplementedError()
#
#     return meta_train_dataset, meta_val_dataset
