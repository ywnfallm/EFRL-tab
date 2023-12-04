from pytorch_tabnet.pretraining import TabNetPretrainer
import torch
import openml
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
np.random.seed(0)
import os

def emb_data(dataname, data, isCategory, target, mybatch):
    # 使用tabnet自监督模型获取输入的嵌入表示(无干扰，连续属性不标准化)，保存为 datasetname_embi.npy，i表示嵌入维数
    # 随机打乱数据集顺序
    shuffled_index = np.random.permutation(data.index)
    data = data.iloc[shuffled_index]
    # split数据集
    train_indices, valid_indices = train_test_split(range(len(data)), test_size=0.2, random_state=False)
    # 模型输入准备
    categorical_columns = [col for index, col in enumerate(data.columns) if isCategory[index]]
    categorical_dims = {col: data[col].nunique() for col in categorical_columns}
    features = [col for col in data.columns if col not in [target]]
    cat_idxs = [i for i, f in enumerate(features) if f in categorical_columns]
    cat_dims = [categorical_dims[f] for i, f in enumerate(features) if f in categorical_columns]
    X_train = data[features].values[train_indices]
    X_valid = data[features].values[valid_indices]
    max_epochs = 1000 if not os.getenv("CI", False) else 2
    # TabNetPretrainer
    for emb in range(3): # 连续属性未标准化嵌入
        unsupervised_model = TabNetPretrainer(
                cat_idxs=cat_idxs,
                cat_dims=cat_dims,
                cat_emb_dim=emb+1, # 3,
                optimizer_fn=torch.optim.Adam,
                optimizer_params=dict(lr=2e-2),
                mask_type='entmax',  # "sparsemax",
                n_shared_decoder=1,  # nb shared glu for decoding
                n_indep_decoder=1,  # nb independent glu for decoding
            )
        unsupervised_model.fit(
                X_train=X_train,
                eval_set=[X_valid],
                max_epochs=max_epochs, patience=50,
                batch_size=mybatch[0], virtual_batch_size=mybatch[1],
                num_workers=0,
                drop_last=False,
                pretraining_ratio=0.8,
            )
        # Make reconstruction from a dataset
        _, embedded_X = unsupervised_model.predict(data[features].values)
        print(f'保存连续属性未标准化、无干扰的tabnet自监督嵌入表示数据集为{dataname}_emb{emb+1}_pert4.npy')
        np.save(f'./data/{dataname}_emb{emb+1}_pert4.npy', np.hstack([embedded_X, data[target].values.reshape(-1,1)]))
    # 连续属性标准化
    for index, col in enumerate(data.columns):
        if not isCategory[index]:
            # 注意：这里修改DataFrame的值会影响DataFrame.values数组的结果
            data[col] = StandardScaler().fit_transform(data[col].values.reshape(-1,1))
    for emb in range(3): # 连续属性标准化嵌入
        unsupervised_model = TabNetPretrainer(
                cat_idxs=cat_idxs,
                cat_dims=cat_dims,
                cat_emb_dim=emb+1, # 3,
                optimizer_fn=torch.optim.Adam,
                optimizer_params=dict(lr=2e-2),
                mask_type='entmax',  # "sparsemax",
                n_shared_decoder=1,  # nb shared glu for decoding
                n_indep_decoder=1,  # nb independent glu for decoding
            )
        unsupervised_model.fit(
                X_train=X_train,
                eval_set=[X_valid],
                max_epochs=max_epochs, patience=50,
                batch_size=mybatch[0], virtual_batch_size=mybatch[1],
                num_workers=0,
                drop_last=False,
                pretraining_ratio=0.8,
            )
        # Make reconstruction from a dataset
        _, embedded_X = unsupervised_model.predict(data[features].values)
        print(f'保存连续属性标准化、无干扰的tabnet自监督嵌入表示数据集为{dataname}_emb{emb+1}_pert4_norm.npy')
        np.save(f'./data/{dataname}_emb{emb+1}_pert4_norm.npy', np.hstack([embedded_X, data[target].values.reshape(-1,1)]))

if __name__ == '__main__':
    datasetIDs = [3, 23, 29, 31, 46,
                  50, 151, 188, 38, 307,
                  469, 1590, 1480, 1486, 4534,
                  6332, 1461, 23381, 40668,
                  40975, 40978, 40670, 40701]
    datasetNames = ['kr-vs-kp', 'cmc', 'credit-approval', 'credit-g', 'splice',
                    'tic-tac-toe', 'electricity', 'eucalyptus', 'sick', 'vowel',
                    'analcatdata_dmft', 'adult', 'ilpd', 'nomao', 'PhishingWebsites',
                    'cylinder-bands', 'bank-marketing', 'dresses-sales', 'connect-4',
                    'car', 'Internet-Advertisements', 'dna', 'churn']
    datasetTargets = ['class', 'Contraceptive_method_used', 'class', 'class', 'Class',
                      'Class', 'class', 'Utility', 'Class', 'Class',
                      'Prevention', 'class', 'Class', 'Class', 'Result',
                      'band_type', 'Class', 'Class', 'class',
                      'class', 'class', 'class', 'class']
    dataBatch = [(512, 128), (128, 64), (128, 64), (128, 64), (256, 128),
                 (128, 64), (2048, 256), (32, 16), (512, 128), (32, 16),
                 (32, 16), (2048, 256), (128, 64), (2048, 256), (1024, 128),
                 (64, 32), (2048, 256), (64, 32), (2048, 256),
                 (128, 64), (512, 128), (256, 128), (512, 128)]

    for ind in range(len(datasetIDs)):

        data, _, isCat, feaName = openml.datasets.get_dataset(datasetIDs[ind]).get_data(dataset_format="dataframe")

        # 保存原始数据文件 datasetname_origin.csv
        # print(f'保存源数据集为{datasetNames[ind]}_origin.csv')
        # data.to_csv(f'./data/{datasetNames[1]}_origin.csv', index=False)

        # 预处理步骤：无效列处理(删除单值列和无值列)；缺失值处理(分类属性用"VV_likely"替换，连续属性用均值替换)。数据文件保存为 datasetname.csv
        # 无效列处理：删除单值列和无值列
        datacolumns = data.shape[1]
        for index, col in enumerate(data.columns):
            if data[col].nunique() < 2:
                # print(f'{datasetNames[ind]}中属性{col}为单值列或无值列，清除该列！')
                data.drop(columns=col, inplace=True)
                isCat.pop(len(isCat)-(datacolumns-index))
                feaName.remove(col)
        # 缺失值处理：分类属性用"VV_likely"替换，连续属性用均值替换
        for index, col in enumerate(data.columns):
            if isCat[index]:
                data[col] = data[col].astype('object')
                data[col] = data[col].fillna("VV_likely")
            else:
                data[col] = data[col].fillna(data[col].mean())
        # print(f'保存无效列处理和缺失值处理后的数据集为{datasetNames[ind]}.csv')
        # data.to_csv(f'./data/{datasetNames[ind]}.csv', index=False)

        # 准备经过one-hot编码的数据集，保存为 datasetname_onehot.npy
        dl = []
        for index, col in enumerate(data.columns):
            if isCat[index]:
                if col == datasetTargets[ind]:
                    lenc = LabelEncoder()
                    dl.append(lenc.fit_transform(data[col].values.reshape(-1, 1)).reshape(-1,1))
                else:
                    ohe = OneHotEncoder()
                    dl.append(ohe.fit_transform(data[col].values.reshape(-1, 1)).toarray())
            else:
                dl.append(data[col].values.reshape(-1, 1))
        # print(f'保存one-hot编码后数据集为{datasetNames[ind]}_onehot.npy')
        # np.save(f'./data/{datasetNames[ind]}_onehot.npy', np.hstack(dl))

        # 使用tabnet自监督模型获取输入的嵌入表示(无干扰，连续属性不标准化)，保存为 datasetname_embi.npy，i表示嵌入维数
        # 分类属性需要 LabelEncoder
        for index, col in enumerate(data.columns):
            if isCat[index]:
                lenc = LabelEncoder()
                data[col] = lenc.fit_transform(data[col].values)
        emb_data(datasetNames[ind], data, isCat, datasetTargets[ind], dataBatch[ind])
