import os.path
import sys
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, ConfusionMatrixDisplay, classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.optim import Adam
from torch.utils.data import DataLoader

from scarf.loss import NTXent
from scarf.model import SCARF

from example.dataset import ExampleDataset
from example.utils import dataset_embeddings, fix_seed, train_epoch

if __name__ == '__main__':
    dataset_name = sys.argv[1]
    seed = int(sys.argv[2])  # seed = 1234
    resultdir = './result'

    ###Data
    with open(f'../data/{dataset_name}.npy', 'rb') as f:
        data = np.load(f)
    data, target = data[:, :-1], data[:, -1]
    train_data, test_data, train_target, test_target = train_test_split(
        data,
        target,
        test_size=0.2,
        stratify=target,
        random_state=seed
    )

    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)

    # to torch dataset
    train_ds = ExampleDataset(
        train_data,
        train_target
    )
    test_ds = ExampleDataset(
        test_data,
        test_data
    )

    ###Training
    batch_size = 128
    epochs = 1000  # 1000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = SCARF(
        input_dim=train_ds.shape[1],
        emb_dim=16,
        corruption_rate=0.6,
    ).to(device)
    optimizer = Adam(model.parameters(), lr=0.001)
    ntxent_loss = NTXent()

    # loss_history = []

    for epoch in range(1, epochs + 1):
        epoch_loss = train_epoch(model, ntxent_loss, train_loader, optimizer, device, epoch)
        # loss_history.append(epoch_loss)

    ###Evaluate embeddings
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # get embeddings for training and test set
    train_embeddings = dataset_embeddings(model, train_loader, device)
    test_embeddings = dataset_embeddings(model, test_loader, device)

    clf = LogisticRegression()

    # embeddings dataset: train the classifier on the embeddings
    clf.fit(train_embeddings, train_target)
    vanilla_predictions = clf.predict(test_embeddings)

    testacc = accuracy_score(test_target, vanilla_predictions)
    print(f'testacc on the embeddings = {testacc}')

    resultfile = f'{resultdir}/{dataset_name}_seed{seed}.txt'
    with open(resultfile, 'a') as f:
        f.write(f'scarf, {dataset_name},seed{seed},test_acc,{testacc}\n')



    #
    seed = 1234
    # fix_seed(seed)
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
    # resultdir = './result'
    #
    # for name in range(len(datasetNames)):
    #     for emb in range(len(embNums)):
    #
    #         dataset_name = f'{datasetNames[name]}{embNums[emb]}'
    #
    #         ###Data
    #         with open(f'../data/{dataset_name}.npy', 'rb') as f:
    #             data = np.load(f)
    #         data, target = data[:, :-1], data[:, -1]
    #         train_data, test_data, train_target, test_target = train_test_split(
    #             data,
    #             target,
    #             test_size=0.2,
    #             stratify=target,
    #             random_state=seed
    #         )
    #
    #         scaler = StandardScaler()
    #         train_data = scaler.fit_transform(train_data)
    #         test_data = scaler.transform(test_data)
    #
    #         # to torch dataset
    #         train_ds = ExampleDataset(
    #             train_data,
    #             train_target
    #         )
    #         test_ds = ExampleDataset(
    #             test_data,
    #             test_data
    #         )
    #
    #         ###Training
    #         batch_size = 128
    #         epochs = 10 # 1000
    #         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #
    #         train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    #
    #         model = SCARF(
    #             input_dim=train_ds.shape[1],
    #             emb_dim=16,
    #             corruption_rate=0.6,
    #         ).to(device)
    #         optimizer = Adam(model.parameters(), lr=0.001)
    #         ntxent_loss = NTXent()
    #
    #         # loss_history = []
    #
    #         for epoch in range(1, epochs + 1):
    #             epoch_loss = train_epoch(model, ntxent_loss, train_loader, optimizer, device, epoch)
    #             # loss_history.append(epoch_loss)
    #
    #         ###Evaluate embeddings
    #         train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    #         test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    #
    #         # get embeddings for training and test set
    #         train_embeddings = dataset_embeddings(model, train_loader, device)
    #         test_embeddings = dataset_embeddings(model, test_loader, device)
    #
    #         clf = LogisticRegression()
    #
    #         # embeddings dataset: train the classifier on the embeddings
    #         clf.fit(train_embeddings, train_target)
    #         vanilla_predictions = clf.predict(test_embeddings)
    #
    #         testacc = accuracy_score(test_target, vanilla_predictions)
    #         print(f'testacc on the embeddings = {testacc}')
    #
    #         logfilepath = f'{resultdir}/{dataset_name}_seed{seed}.txt'
    #         with open(logfilepath, 'a') as f:
    #             f.write(f'{dataset_name},seed_{seed},test_acc,{testacc}\n')


