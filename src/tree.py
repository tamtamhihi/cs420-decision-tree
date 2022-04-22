from pandas.core.frame import DataFrame
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import graphviz
from env import TREE_VISUALIZE_PATH
import matplotlib.pyplot as plt

class DecisionTreeWrapper:
    def __init__(self, data_X: DataFrame, data_y: DataFrame, train_split: int, max_depth=None):
        self.train_split = train_split
        self.model = DecisionTreeClassifier(criterion='entropy', max_depth=max_depth)
        self.model.fit(data_X, data_y)
        self.max_depth = max_depth
        self.actual_depth = self.model.get_depth()
    def visualize(self):
        self.visualizer = export_graphviz(self.model, out_file=None)
        graph = graphviz.Source(self.visualizer)
        filepath = TREE_VISUALIZE_PATH + f'-{self.train_split}'
        if self.max_depth != None:
            filepath += f'-depth-{self.max_depth}'
        graph.render(filepath)
    def predict_and_evaluate(self, test_X: DataFrame, test_y: DataFrame, verbose=True):
        prediction = self.model.predict(test_X)
        report = classification_report(test_y, prediction, labels=self.model.classes_, zero_division=0)
        report_dict = classification_report(test_y, prediction, labels=self.model.classes_, output_dict=True, zero_division=0)
        if verbose:
            print(report)
        self.accuracy = report_dict['accuracy']
        matrix = confusion_matrix(test_y, prediction)
        plot = ConfusionMatrixDisplay(matrix, display_labels=self.model.classes_)
        plot.plot()
        if verbose:
            filepath = f'matrix-{self.train_split}.jpg'
            plt.savefig(filepath)