from dataset import TrainTestData
from input import read_input, split_train_test
from tree import DecisionTreeWrapper
from env import TRAIN_SPLITS, PLOT_TREE, CHOSEN_PROP_INDEX, DEPTHS
from typing import List

def prompt():
    input('Press enter to continue...')

if __name__ == '__main__':
    print('-------------------')
    print('Reading input...')
    dataset = read_input()
    N = len(dataset)

    train_test_data: List[TrainTestData] = []
    trees = []

    for train_split in TRAIN_SPLITS:
        print('-------------------')
        print(f'%% Train {train_split}% - Test {100 - train_split}%')
        data = split_train_test(dataset, train_split)
        train_test_data.append(data)
        num_train = len(data.train_X)
        num_train_expected = N * train_split // 100
        num_test = len(data.test_X)
        num_test_expected = N - num_train_expected
        print(f'Train: {num_train} - Expected train: {num_train_expected} - Test: {num_test}  - Expected test: {num_test_expected}')
        print('Fitting...')
        tree = DecisionTreeWrapper(data.train_X, data.train_y, train_split)
        trees.append(tree)
        if PLOT_TREE:
            print('Visualizing...')
            tree.visualize()
        print('Evaluating...')
        tree.predict_and_evaluate(data.test_X, data.test_y)
        print('Accuracy: {:.2f}'.format(tree.accuracy))
        prompt()
    
    print('-------------------')
    print('Exploring max depth for decision tree...')
    print(f'Chosen dataset: Train {TRAIN_SPLITS[CHOSEN_PROP_INDEX]}% - Test {100 - TRAIN_SPLITS[CHOSEN_PROP_INDEX]}%')

    prompt()

    print(f'%%%% max_depth = None: Actual depth = {trees[CHOSEN_PROP_INDEX].actual_depth}')
    accuracy = dict()
    chosen_data = train_test_data[CHOSEN_PROP_INDEX]
    
    for depth in DEPTHS:
        print(f'%%%% max_depth = {depth} running...')
        tree_depth = DecisionTreeWrapper(chosen_data.train_X, chosen_data.train_y, TRAIN_SPLITS[CHOSEN_PROP_INDEX], depth)
        tree_depth.predict_and_evaluate(chosen_data.test_X, chosen_data.test_y, verbose=False)
        tree_depth.visualize()
        accuracy[depth] = tree_depth.accuracy
    print()
    print('{:<15} {}'.format('Max_depth', 'Accuracy'))
    print('{:<15} {:.2f}'.format('None', trees[CHOSEN_PROP_INDEX].accuracy))
    for depth in DEPTHS:
        print('{:<15} {:.2f}'.format(depth, accuracy[depth]))