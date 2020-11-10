import os

import pandas as pd
from torch.utils.data import DataLoader

import torch

from utils import tabularModel, titanicDataset


def main():
    # %%

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
    model.train()
    for epoch in range(num_epoch):
        for i, (x, y) in enumerate(train_dataloader):
            optim.zero_grad()
            pred = model(x)
            loss = lossfunc(pred, y.squeeze().long())
            loss.backward()
            optim.step()

            if i % 2 == 0:
                with torch.no_grad():
                    model.eval()
                    x, y = next(iter(valid_dataloader))
                    pred = model(x)
                    valid_loss = lossfunc(pred, y.squeeze().long())
                    accu = (pred.argmax(-1) == y.squeeze()).sum() // len(valid_dataset)
                print("Epoch: {} | Train loss: {:.4f} | Valid loss: {:.4f} | Valid Accu: {:.2f}".format(
                    epoch, loss.item(), valid_loss.item(), accu.item()
                ))
                model.train()

    # %%


if __name__ == '__main__':
    print("Excecuting!")
    main()
