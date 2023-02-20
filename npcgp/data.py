# Copyright 2017 Hugh Salimbeni
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# From https://github.com/ICL-SML/Doubly-Stochastic-DGP/blob/master/demos/datasets.py

import numpy as np
import os
import csv
import pandas
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile
from ftplib import FTP


class Dataset(object):
    def __init__(self, name, N, D, type, data_path="/data/"):
        self.data_path = data_path
        self.name, self.N, self.D = name, N, D
        assert type in ["regression", "classification", "multiclass", "mo_regression"]
        self.type = type

    def csv_file_path(self, name):
        """Path to data."""
        return "{}{}.csv".format(self.data_path, name)

    def read_data(self):
        """Read raw data from CSV."""
        data = pandas.read_csv(
            self.csv_file_path(self.name), header=None, delimiter=","
        ).values
        return {"X": data[:, :-1], "Y": data[:, -1, None]}

    def download_data(self):
        """Downloads data from source, must be implemented for new datasets."""
        NotImplementedError

    def get_data(self, seed=0, split=0, prop=0.9):
        """Normalize and split data in train and test sets."""
        path = self.csv_file_path(self.name)
        if not os.path.isfile(path):
            self.download_data()

        full_data = self.read_data()
        split_data = self.split(full_data, seed, split, prop)
        split_data = self.normalize(split_data, "X")

        if "regression" in self.type:
            split_data = self.normalize(split_data, "Y")

        return split_data

    def split(self, full_data, seed, split, prop):
        """Split data into train and test sets."""
        ind = np.arange(self.N)

        np.random.seed(seed + split)
        np.random.shuffle(ind)

        n = int(self.N * prop)

        X = full_data["X"][ind[:n], :]
        Xs = full_data["X"][ind[n:], :]

        Y = full_data["Y"][ind[:n], :]
        Ys = full_data["Y"][ind[n:], :]

        return {"X": X, "Xs": Xs, "Y": Y, "Ys": Ys}

    def normalize(self, split_data, X_or_Y):
        """Standardize inputs and outputs to N(0, 1)."""
        m = np.average(split_data[X_or_Y], 0)[None, :]
        s = np.std(split_data[X_or_Y], 0)[None, :] + 1e-6

        split_data[X_or_Y] = (split_data[X_or_Y] - m) / s
        split_data[X_or_Y + "s"] = (split_data[X_or_Y + "s"] - m) / s

        split_data.update({X_or_Y + "_mean": m.flatten()})
        split_data.update({X_or_Y + "_std": s.flatten()})
        return split_data


uci_base = "https://archive.ics.uci.edu/ml/machine-learning-databases/"


class Airline(Dataset):
    def __init__(self):
        self.name, self.N, self.D = "airline", 800000, 8
        self.type = "regression"

    def download_data(self):
        raise Warning("Data not fetchable via wget.")

    def read_data(self):
        """
        2021 Vincent Dutordoir.

        Script adapted from James Hensman
        https://github.com/jameshensman/VFF/blob/master/experiments/airline/airline_vff_additive.py

        Data pickle file can be downloaded from:
        https://drive.google.com/file/d/1CnA6FYb8jNUckJt4VLz_ONA1KgV-bXwK/view?usp=sharing
        Returns the Airline delay dataset, containing a total of 5929413 rows.
        Each datapoint has 8 features.
        All features are rescaled to [-1, 1] and the target is normalized to be N(0, 1) distributed.
        :param n: int
            total dataset size (train + test size)
            n_train = 2/3 * n = train size
            n_test = 1/3 * n = test size
            Defaults to None, which corresponds to returning all rows (5929413 in total).
        :return:
            X: [n_train, 8], Y: [n_train, 1]
            XT: [n_test, 8], YT: [n_test, 1]

        NOTE: Pandas 0.24.2 must be used to unpickle, as versions 1.x do not support pickles this old.
        """
        # Import the data
        data = pandas.read_pickle("data/airline.pickle")

        # Convert time of day from hhmm to minutes since midnight
        data.ArrTime = 60 * np.floor(data.ArrTime / 100) + np.mod(data.ArrTime, 100)
        data.DepTime = 60 * np.floor(data.DepTime / 100) + np.mod(data.DepTime, 100)

        # Pick out the data
        Y = data["ArrDelay"].values
        names = [
            "Month",
            "DayofMonth",
            "DayOfWeek",
            "plane_age",
            "AirTime",
            "Distance",
            "ArrTime",
            "DepTime",
        ]
        X = data[names].values

        # Consider first 800k datapoints
        n_train = 700000
        n_test = 100000
        XT = X[n_train : (n_train + n_test)]
        YT = Y[n_train : (n_train + n_test)]
        X = X[:n_train]
        Y = Y[:n_train]

        # Normalize Y scale and offset
        Ymean = Y.mean()
        Ystd = Y.std()
        Y = (Y - Ymean) / Ystd
        Y = Y.reshape(-1, 1)
        YT = (YT - Ymean) / Ystd
        YT = YT.reshape(-1, 1)

        # Normalize X on [-1, 1]
        Xmin, Xmax = X.min(0), X.max(0)
        X = (X - Xmin) / (Xmax - Xmin)
        X = 2 * (X - 0.5)
        XT = (XT - Xmin) / (Xmax - Xmin)
        XT = 2 * (XT - 0.5)

        return {"X": X, "Y": Y, "Xs": XT, "Ys": YT, "Y_std": np.array([Ystd])}


class Energy(Dataset):
    def __init__(self):
        self.name, self.N, self.D = "energy", 768, 8
        self.type = "regression"

    def download_data(self):
        url = "https://code.datasciencedojo.com/datasciencedojo/datasets/raw/master/Energy%20Efficiency/ENB2012_data.xlsx"

        data = pandas.read_excel(url, engine="openpyxl").values
        # note this uses y1 as the target!!
        data = data[:, :-1]

        with open(self.csv_file_path(self.name), "w") as f:
            csv.writer(f).writerows(data)


class EnergyMO(Dataset):
    def __init__(self):
        self.name, self.N, self.D, self.O = "energy_mo", 768, 8, 2
        self.type = "mo_regression"

    def download_data(self):
        url = "https://code.datasciencedojo.com/datasciencedojo/datasets/raw/master/Energy%20Efficiency/ENB2012_data.xlsx"
        data = pandas.read_excel(url, engine="openpyxl").values

        with open(self.csv_file_path(self.name), "w") as f:
            csv.writer(f).writerows(data)

    def read_data(self):
        data = pandas.read_csv(
            self.csv_file_path(self.name), header=None, delimiter=","
        ).values
        return {"X": data[:, :-2], "Y": data[:, -2:]}


class Kin8mn(Dataset):
    def __init__(self):
        self.name, self.N, self.D = "kin8nm", 8192, 8
        self.type = "regression"

    def download_data(self):

        url = "https://www.openml.org/data/get_csv/3626/dataset_2175_kin8nm.csv"

        data = pandas.read_csv(url, header=0)
        data = data.values

        with open(self.csv_file_path(self.name), "w") as f:
            csv.writer(f).writerows(data)


class NavalMO(Dataset):
    def __init__(self):
        self.name, self.N, self.D, self.O = "naval_mo", 11934, 16, 2
        self.type = "mo_regression"

    def download_data(self):

        url = "{}{}".format(uci_base, "00316/UCI%20CBM%20Dataset.zip")

        with urlopen(url) as zipresp:
            with ZipFile(BytesIO(zipresp.read())) as zfile:
                zfile.extractall("/tmp/")

        data = pandas.read_fwf("/tmp/UCI CBM Dataset/data.txt", header=None).values

        with open(self.csv_file_path(self.name), "w") as f:
            csv.writer(f).writerows(data)

    def read_data(self):
        data = pandas.read_csv(
            self.csv_file_path(self.name), header=None, delimiter=","
        ).values
        return {"X": data[:, :-2], "Y": data[:, -2:]}


class Power(Dataset):
    def __init__(self):
        self.name, self.N, self.D = "power", 9568, 4
        self.type = "regression"

    def download_data(self):
        url = "{}{}".format(uci_base, "00294/CCPP.zip")
        with urlopen(url) as zipresp:
            with ZipFile(BytesIO(zipresp.read())) as zfile:
                zfile.extractall("/tmp/")

        data = pandas.read_excel(
            "/tmp/CCPP//Folds5x2_pp.xlsx", engine="openpyxl"
        ).values

        with open(self.csv_file_path(self.name), "w") as f:
            csv.writer(f).writerows(data)


class Protein(Dataset):
    def __init__(self):
        self.name, self.N, self.D = "protein", 45730, 9
        self.type = "regression"

    def download_data(self):

        url = "{}{}".format(uci_base, "00265/CASP.csv")

        data = pandas.read_csv(url).values

        data = np.concatenate([data[:, 1:], data[:, 0, None]], 1)

        with open(self.csv_file_path(self.name), "w") as f:
            csv.writer(f).writerows(data)


class HouseElectric(Dataset):
    def __init__(self):
        self.name, self.N, self.D = "houseelectric", 2049280, 11
        self.type = "regression"

    def download_data(self):

        url = "https://github.com/treforevans/uci_datasets/raw/master/uci_datasets/houseelectric/data.csv.gz"

        data = pandas.read_csv(url, compression="gzip", header=None, sep=",").values
        with open(self.csv_file_path(self.name), "w") as f:
            csv.writer(f).writerows(data)


class Polymer(Dataset):
    def __init__(self):
        self.data_path = "data"
        self.name, self.N, self.D, self.O = "polymer", 60, 10, 4
        self.type = "mo_regression"

        self.f_name = os.path.join(self.data_path, "polymer.dat")

    def download_data(self):
        ftp_addr = "ftp.cis.upenn.edu"
        ftp = FTP(ftp_addr)
        ftp.login()
        ftp.cwd("pub/ungar/chemdata/")
        with open(self.f_name, "wb") as fp:
            ftp.retrbinary("RETR polymer.dat", fp.write)
        ftp.quit()

    def read_data(self):
        try:
            data = np.loadtxt(self.f_name)

        except OSError:
            self.download_data()
            data = np.loadtxt(self.f_name)

        return {
            "X": data[:41, :10],
            "Y": data[:41, 10:],
            "Xs": data[41:, :10],
            "Ys": data[41:, 10:],
        }

    def get_data(self):
        return {**self.read_data(), "Y_std": np.ones(4)}


class Datasets(object):
    def __init__(self, data_path="/data/"):
        if not os.path.isdir(data_path):
            os.mkdir(data_path)

        datasets = []

        datasets.append(Airline())
        datasets.append(Energy())
        datasets.append(EnergyMO())
        datasets.append(Kin8mn())
        datasets.append(NavalMO())
        datasets.append(Power())
        datasets.append(Protein())
        datasets.append(Polymer())
        datasets.append(HouseElectric())

        self.all_datasets = {}
        for d in datasets:
            d.data_path = data_path
            self.all_datasets.update({d.name: d})
