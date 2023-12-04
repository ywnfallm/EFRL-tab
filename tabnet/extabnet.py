from pytorch_tabnet.tab_model import TabNetClassifier

import torch
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split

import numpy as np
import os
import sys

if __name__ == '__main__':
    dataset_name = sys.argv[1]
    seed = int(sys.argv[2])
    np.random.seed(seed)

    resultdir = './result'
    with open(f'../data/{dataset_name}.npy', 'rb') as f:
        data = np.load(f)

    data, target = data[:, :-1], data[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, stratify=target,
                                                        random_state=seed)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2 / 0.8,
                                                          stratify=y_train, random_state=seed)

    cat_idxs = []
    cat_dims = []

    clf = TabNetClassifier(cat_idxs=cat_idxs,
                           cat_dims=cat_dims,
                           cat_emb_dim=1,
                           optimizer_fn=torch.optim.Adam,
                           optimizer_params=dict(lr=2e-2),
                           scheduler_params={"step_size": 50,  # how to use learning rate scheduler
                                             "gamma": 0.9},
                           scheduler_fn=torch.optim.lr_scheduler.StepLR,
                           mask_type='entmax'  # "sparsemax"
                           )

    max_epochs = 1000 if not os.getenv("CI", False) else 2

    clf.fit(
        X_train=X_train, y_train=y_train,
        eval_set=[(X_train, y_train), (X_valid, y_valid)],
        eval_name=['train', 'valid'],  # eval_metric=['auc'],
        max_epochs=max_epochs, patience=50,
        batch_size=1024, virtual_batch_size=128,
        num_workers=0,
        weights=1,
        drop_last=False
    )

    preds = clf.predict(X_test)
    testacc = accuracy_score(y_test, preds)
    resultfile = f'{resultdir}/{dataset_name}_seed{seed}.txt'
    with open(resultfile, 'a') as f:
        f.write(f'tabnet, {dataset_name}, seed{seed}, test_acc, {testacc}\n')

    # seed = 0
    # np.random.seed(seed)
    # resultdir = './result'
    #
    # datasetNames = ['dresses-sales', 'cylinder-bands', 'eucalyptus', 'analcatdata_dmft', 'tic-tac-toe',
    #                 'vowel', 'credit-g', 'cmc', 'ilpd',
    #                 'credit-approval', 'kr-vs-kp', 'car', 'splice', 'sick',
    #                 'dna', 'churn', 'Internet-Advertisements', 'PhishingWebsites', 'nomao',
    #                 'electricity', 'adult', 'bank-marketing', 'connect-4']
    # # embNums = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16',
    # #            '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30']
    #
    # embNums = ['00',]
    #
    # for name in range(len(datasetNames)):
    #     for emb in range(len(embNums)):
    #
    #         dataset_name = f'{datasetNames[name]}{embNums[emb]}'
    #
    #         with open(f'../data/{dataset_name}.npy', 'rb') as f:
    #             data = np.load(f)
    #
    #         data, target = data[:, :-1], data[:, -1]
    #         X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, stratify=target,
    #                                                             random_state=seed)
    #         X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2 / 0.8,
    #                                                               stratify=y_train, random_state=seed)
    #
    #         cat_idxs = []
    #         cat_dims = []
    #
    #         clf = TabNetClassifier(cat_idxs=cat_idxs,
    #                                cat_dims=cat_dims,
    #                                cat_emb_dim=1,
    #                                optimizer_fn=torch.optim.Adam,
    #                                optimizer_params=dict(lr=2e-2),
    #                                scheduler_params={"step_size": 50,  # how to use learning rate scheduler
    #                                                  "gamma": 0.9},
    #                                scheduler_fn=torch.optim.lr_scheduler.StepLR,
    #                                mask_type='entmax'  # "sparsemax"
    #                                )
    #
    #         max_epochs = 10 if not os.getenv("CI", False) else 2
    #
    #         clf.fit(
    #             X_train=X_train, y_train=y_train,
    #             eval_set=[(X_train, y_train), (X_valid, y_valid)],
    #             eval_name=['train', 'valid'], # eval_metric=['auc'],
    #             max_epochs=max_epochs, patience=50,
    #             batch_size=1024, virtual_batch_size=128,
    #             num_workers=0,
    #             weights=1,
    #             drop_last=False
    #         )
    #
    #         preds = clf.predict(X_test)
    #         testacc = accuracy_score(y_test, preds)
    #         with open(f'{resultdir}/{dataset_name}_seed{seed}.txt','a') as f:
    #             f.write(f'tabnet, {dataset_name}, seed_{seed}, test_acc, {testacc}\n')

