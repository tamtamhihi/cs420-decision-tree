from typing import Tuple
from pandas.core.frame import DataFrame
from env import INPUT_DATA, TRAIN_SPLITS
from pandas import read_csv
from typing import List, Tuple
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from dataset import TrainTestData

# Read input and split train-test in different proportions.
def read_input() -> DataFrame:
    data: DataFrame = read_csv(INPUT_DATA, header=None)
    data = shuffle(data)
    return data

def split_train_test(data: DataFrame, train_split: int) -> TrainTestData:
    data = shuffle(data)
    train_set, test_set = train_test_split(data, test_size = (100 - train_split) / 100)
    train_X, train_y = split_X_y(train_set)
    test_X, test_y = split_X_y(test_set)
    return TrainTestData(train_X, train_y, test_X, test_y)

def split_X_y(data: DataFrame) -> Tuple[DataFrame, DataFrame]:
    return data.iloc[:,:-1], data.iloc[:,-1]