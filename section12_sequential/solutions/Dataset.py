import pandas as pd

import torch
from torch.utils.data.dataset import Dataset


class WeatherDataset(Dataset):
    def __init__(self, dataset_file, day_range, split_date, train_test="train"):
        df = pd.read_csv(dataset_file)
        df['Date'] = pd.to_datetime(df['Date'])  # Convert 'Date' column to datetime
        df.set_index('Date', inplace=True)

        mean = df.mean()
        std = df.std()
        df = (df - mean) / std

        self.mean = torch.tensor(mean.to_numpy()).reshape(1, -1)
        self.std = torch.tensor(std.to_numpy()).reshape(1, -1)

        if train_test == "train":
            self.dataset = df[df.index < split_date]
        elif train_test == "test":
            self.dataset = df[df.index >= split_date]
        else:
            ValueError("train_test should be train or test")

        self.day_range = day_range

    def __getitem__(self, index):
        end_index = index + self.day_range
        current_series = self.dataset.iloc[index:end_index]

        day_tensor = torch.LongTensor(current_series.index.day.to_numpy())
        month_tensor = torch.LongTensor(current_series.index.month.to_numpy())
        data_values = torch.FloatTensor(current_series.values)

        return day_tensor, month_tensor, data_values

    def __len__(self):
        return len(self.dataset) - self.day_range
