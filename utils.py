import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset


class titanicDataset(Dataset):
    def __init__(self, path, train=True):
        super(titanicDataset, self).__init__()
        raw_data = pd.read_csv(path)
        features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']
        self.train = train
        if train:
            label = ['Survived']
            self.y = raw_data.loc[:, label]
            self.y = self.y.to_numpy()

        def binning(x):
            if x == 0:
                return 0.0
            elif 1 <= x <= 2:
                return 1.0
            elif 3 <= x <= 5:
                return 2.0
            else:
                return 3.0

        binned_family = (raw_data.loc[:, 'SibSp'] + raw_data.loc[:, 'Parch']).apply(binning)
        self.x = raw_data.loc[:, features]
        self.x['binned_family'] = binned_family

        self.x['Sex'] = (self.x['Sex'] == 'male').astype(float)

        def cat2vec(x):
            if x == 'S':
                return 0.0
            elif x == 'C':
                return 1.0
            elif x == 'Q':
                return 2.0
            else:
                return 0.0

        self.x['Embarked'] = self.x['Embarked'].apply(cat2vec)

        self.x['Pclass'] = self.x['Pclass'].astype(float) - 1.0
        self.x['Age'] = self.x['Age'].fillna(float(int(self.x['Age'].mean())))
        self.x['Fare'] = self.x['Fare'].fillna(self.x['Fare'].mean())

        # as torch dataloader doesn't support returning pd dataframe, convert to a dictionary
        self.x = self.x.to_dict()
        self.columns = list(self.x.keys())

    def __len__(self):
        return len(self.x['Age'])

    def __getitem__(self, index):
        out_x = {}
        for col in self.columns:
            out_x[col] = self.x[col][index]
        if self.train:
            return out_x, self.y[index]
        else:
            return out_x


class tabularModel(nn.Module):
    def __init__(self, embedding_size=None, categorical_columns=None, continuous_columns=None, fc_size=[256, 64],
                 dropout=[0.6, 0.3]):
        super().__init__()
        self.categorical_columns = categorical_columns
        self.continuous_columns = continuous_columns
        if embedding_size is not None and categorical_columns is not None:
            try:
                assert len(embedding_size) == len(categorical_columns)
                self.embeddings = nn.ModuleList([nn.Embedding(categories, size) for categories, size in embedding_size])
                n_emb = sum(e.embedding_dim for e in self.embeddings)  # length of all embeddings combined
                self.n_emb, self.n_cont = sum([tmp for _, tmp in embedding_size]), len(continuous_columns)
                self.lin1 = nn.Linear(self.n_emb + self.n_cont, fc_size[0])
                self.lin2 = nn.Linear(fc_size[0], fc_size[1])
                self.lin3 = nn.Linear(fc_size[1], 2)
                self.bn1 = nn.BatchNorm1d(self.n_cont)
                self.bn2 = nn.BatchNorm1d(fc_size[0])
                self.bn3 = nn.BatchNorm1d(fc_size[1])
                self.emb_drop = nn.Dropout(dropout[0])
                self.drops = nn.Dropout(dropout[1])
            except AssertionError:
                print("length of embedding size must be equal to the size of categorical columns!")
        else:
            raise ValueError("Embedding size and categorical columns must be specified!")

    def forward(self, x):
        x_cat = []
        x_cont = []
        for cat in self.categorical_columns:
            x_cat.append(x[cat])
        for cat in self.continuous_columns:
            x_cont.append(x[cat])
        # reshape two tensors to shape (batch_size, num_columns, 1)
        x_cat = torch.cat(x_cat).reshape(len(self.categorical_columns), -1).permute(1, 0).long().to(
            next(self.parameters()).device)
        x_cont = torch.cat(x_cont, 0).reshape(len(self.continuous_columns), -1).permute(1, 0).double().to(
            next(self.parameters()).device)

        x = [e(x_cat[:, i]) for i, e in enumerate(self.embeddings)]
        x = torch.cat(x, 1)
        x = self.emb_drop(x)
        x2 = self.bn1(x_cont)

        x = torch.cat([x, x2], 1)

        x = F.relu(self.lin1(x))
        x = self.drops(x)
        x = self.bn2(x)
        x = F.relu(self.lin2(x))
        x = self.drops(x)
        x = self.bn3(x)
        x = self.lin3(x)

        return x
