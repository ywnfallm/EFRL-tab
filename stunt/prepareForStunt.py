import numpy as np
from sklearn.cluster import KMeans
import os
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    # process all datasets

    datasetNames = ['dresses-sales', 'cylinder-bands', 'eucalyptus', 'analcatdata_dmft', 'tic-tac-toe',
                    'vowel', 'credit-g', 'cmc', 'ilpd',
                    'credit-approval', 'kr-vs-kp', 'car', 'splice', 'sick',
                    'dna', 'churn', 'Internet-Advertisements', 'PhishingWebsites', 'nomao',
                    'electricity', 'adult', 'bank-marketing', 'connect-4',
                    ]
    databh = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16',
              '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30']


    for name in range(len(datasetNames)):
        for bh in range(31):
            seed = 0
            np.random.seed(seed)
            dataset = datasetNames[name] + databh[bh]
            shot = 5

            # make data directory if necessary
            datadir = f'./data/{dataset}'
            if not os.path.exists(datadir):
                os.mkdir(datadir)

            # make data index directory if necessary
            dataindexdir = f'{datadir}/index{shot}'
            if not os.path.exists(dataindexdir):
                os.mkdir(dataindexdir)

            # 1. split train-test set , save xtrain.npy, ytrain.npy, xtest.npy, ytest.npy in data/<dataset>/
            openfile = f'../data/{dataset}.npy'
            with open(openfile, 'rb') as f:
                sample = np.load(f)

            data, target = sample[:, :-1], sample[:, -1]
            xtrain, xtest, ytrain, ytest = train_test_split(data, target, test_size=0.2, stratify=target,
                                                            random_state=seed)

            np.save(datadir + '/xtrain.npy', xtrain)
            np.save(datadir + '/ytrain.npy', ytrain)
            np.save(datadir + '/xtest.npy', xtest)
            np.save(datadir + '/ytest.npy', ytest)

            # 2. generate pseudo_val_y, save to val_y.npy in data/<dataset>/
            train_x, val_x, train_y, val_y = train_test_split(xtrain, ytrain, test_size=0.2 / 0.8, stratify=ytrain,
                                                              random_state=seed)

            np.save(datadir + '/train_x.npy', train_x)
            np.save(datadir + '/val_x.npy', val_x)

            n_clusters = len(np.unique(val_y))
            model = KMeans(n_clusters=n_clusters)
            model.fit(val_x)
            labels = model.predict(val_x)

            np.save(datadir + '/val_y.npy', labels)

            # 3. save train_idx_<seed>.npy for dataset, seed=0,...,9
            for i in range(10):
                np.random.seed(i)

                classes = np.unique(ytrain).shape[0]
                all = []

                for j in range(classes):
                    indall = np.array(np.where(ytrain == j)).reshape(-1)
                    ind = np.random.choice(indall, shot, replace=False)
                    all.append(ind)

                train_idx = np.concatenate(all)

                trainidxdir = f'{dataindexdir}/train_idx_{i}.npy'
                np.save(trainidxdir, train_idx)

# seed = 0
    # np.random.seed(seed)
    # dataset = 'ilpd00'
    # shot = 5
    #
    # # make data directory if necessary
    # datadir = f'./data/{dataset}'
    # if not os.path.exists(datadir):
    #     os.mkdir(datadir)
    #
    # # make data index directory if necessary
    # dataindexdir = f'{datadir}/index{shot}'
    # if not os.path.exists(dataindexdir):
    #     os.mkdir(dataindexdir)
    #
    # # 1. split train-test set , save xtrain.npy, ytrain.npy, xtest.npy, ytest.npy in data/<dataset>/
    # openfile = f'../data/{dataset}.npy'
    # with open(openfile, 'rb') as f:
    #     sample = np.load(f)
    #
    # data, target = sample[:,:-1], sample[:, -1]
    # xtrain, xtest, ytrain, ytest = train_test_split(data, target, test_size=0.2, stratify=target, random_state=seed)
    #
    # np.save(datadir + '/xtrain.npy', xtrain)
    # np.save(datadir + '/ytrain.npy', ytrain)
    # np.save(datadir + '/xtest.npy', xtest)
    # np.save(datadir + '/ytest.npy', ytest)
    #
    # # 2. generate pseudo_val_y, save to val_y.npy in data/<dataset>/
    # train_x, val_x, train_y, val_y = train_test_split(xtrain, ytrain, test_size=0.2/0.8, stratify=ytrain, random_state=seed)
    #
    # np.save(datadir + '/train_x.npy', train_x)
    # np.save(datadir + '/val_x.npy', val_x)
    #
    # n_clusters = len(np.unique(val_y))
    # model = KMeans(n_clusters=n_clusters)
    # model.fit(val_x)
    # labels = model.predict(val_x)
    #
    # np.save(datadir + '/val_y.npy', labels)
    #
    # # 3. save train_idx_<seed>.npy for dataset, seed=0,...,9
    # for i in range(10):
    #     np.random.seed(i)
    #
    #     classes = np.unique(ytrain).shape[0]
    #     all = []
    #
    #     for j in range(classes):
    #         indall = np.array(np.where(ytrain == j)).reshape(-1)
    #         ind = np.random.choice(indall, shot, replace=False)
    #         all.append(ind)
    #
    #     train_idx = np.concatenate(all)
    #
    #     trainidxdir = f'{dataindexdir}/train_idx_{i}.npy'
    #     np.save(trainidxdir, train_idx)

