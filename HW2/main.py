import argparse
from model import DecisionTree
from data_module import AnnealDataModule

def main():
    parser = argparse.ArgumentParser(add_help=True)

    parser.add_argument('--max_depth', type=int, default=None, help='The maximum depth of the tree.\n The default value is infinite.')
    parser.add_argument('--min_samples_split', type=int, default=2, help='The minimum number of samples required to split and internal node.\n The default is 2.')
    parser.add_argument('--min_samples_leaf', type=int, default=1, help='The minimum number of samples required to be at a leaf node.\n The default value is 1.')
    parser.add_argument('--saved_model_path', type=str, help='Where the trained model will be stored.')

    args = parser.parse_args()

    DataModule = AnnealDataModule('data/anneal_train.csv', 'data/anneal_test.csv')

    data = DataModule.get_dataset()
    X = data['X_train']
    y = data['y_train']

    model = DecisionTree(args.max_depth, args.min_samples_split, args.min_samples_leaf)

    model.fit_model(X, y)

    model.save_model(args.saved_model_path)

    return

if __name__ == '__main__':
    main()