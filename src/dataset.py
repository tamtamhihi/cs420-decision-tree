from pandas.core.frame import DataFrame

class TrainTestData:
    def __init__(self, train_X, train_y, test_X, test_y) -> None:
        self.train_X = train_X
        self.train_y = train_y
        self.test_X = test_X
        self.test_y = test_y

        def __map__(value):
            if value == 'b':
                return 0
            elif value == 'x':
                return 1
            else:
                return 2

        self.train_X = self.train_X.applymap(__map__)
        self.test_X = self.test_X.applymap(__map__)
        