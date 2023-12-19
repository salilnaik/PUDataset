import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

class puDataset:
    def __init__(self, X: np.ndarray, y: np.ndarray, c: float) -> None:
        assert len(X) == len(y), "X and y must be the same length"
        assert c <= 1 and c >= 0, "c must be <= 1 and >= 0"
        self.X = X
        self.y = y
        self.c = c
        self.s = None
        self.X_test = None
        self.y_test = None
        
    def catToBin(self, positive_class) -> np.ndarray:
        assert positive_class in self.y, "y must contain at least one instance of positive class"
        negative = np.where(self.y != positive_class)
        self.y[negative] = 0
        return self.y

    def removeLabels(self) -> None:
        self.X, self.y = shuffle(self.X, self.y)
        self.s = self.y.copy()
        positive = np.where(self.y == 1)[0]
        if(self.c == 1):
            unlabeled = positive
        else:
            unlabeled = positive[int(len(positive)*self.c):]
        self.s[unlabeled] = 0
        self.X, self.y, self.s = shuffle(self.X, self.y, self.s)
        
    def trainTestSplit(self, test_size: float | int) -> None:
        assert type(test_size) == float or type(test_size) == int, "test_size must be of type int or float"
        if(type(test_size) == float):
            assert 0 <= test_size <= 1, "test_size must be >= 0 and <= 1"
        else:
            assert 0 <= test_size < len(self.X), "test_size must be >= 0 and < len(dataset)"
        self.X, self.X_test, self.y, self.y_test = train_test_split(self.X, self.y)
    
    def getTrainData(self) -> tuple[np.ndarray, np.ndarray]:
        return self.X, self.y
    
    def getTestData(self) -> tuple[np.ndarray, np.ndarray]:
        assert self.X_test and self.y_test, "Test dataset not created"
        return self.X_test, self.y_test
    
    def getPriorLabels(self) -> np.ndarray:
        return self.c