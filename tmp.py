import os

import pandas as pd
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
from utils import tabularModel, titanicDataset


def main():
    # %%
    writer = SummaryWriter('nn_structure')
    train_data = pd.read_csv(os.path.join('data', 'titanic', 'test.csv'))

    # %%

    train_data.head(10)

    # %%

    dataset = titanicDataset(os.path.join('data', 'titanic', 'train.csv'))
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset,
                                                                 [int(0.8 * len(dataset)),
                                                                  len(dataset) - int(0.8 * len(dataset))])
    # test_dataset = titanicDataset(os.path.join('data', 'titanic', 'test.csv'))

    # %%

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=16, shuffle=True, num_workers=4)
    valid_dataloader = DataLoader(valid_dataset,
                                  batch_size=len(valid_dataset), shuffle=False, num_workers=4)

    # %%

    categorical_columns = ['Pclass', 'Sex', 'Embarked', 'binned_family']
    categories = [3, 2, 3, 4]

    # %%

    continuous_columns = ['Fare', 'Age']

    # %%

    embedding_size = list(zip(categories, [128] * 4))

    # %%

    model = tabularModel(embedding_size, categorical_columns, continuous_columns).double()

    # %%

    optim = torch.optim.Adagrad(model.parameters(), lr=1e-3, weight_decay=0.95)

    # %%
    lossfunc = nn.CrossEntropyLoss()
    # %%

    num_epoch = 300

    # %%
    x, y = next(iter(train_dataloader))
    writer.add_graph(model, x)

    # %%


if __name__ == '__main__':
    print("Excecuting!")
    main()
