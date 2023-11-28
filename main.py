from argparse import ArgumentParser
from experiment import *

parser = ArgumentParser()
parser.add_argument('-dataset', default='housing', type=str,
                    choices=['adult', 'wine', 'titanic', 'autompg'], help='Name of dataset.')
parser.add_argument('-stl_epochs', default=100,
                    help='Number of STL training epochs.', type=int)
parser.add_argument('-mtl_epochs', default=100,
                    help='Number of MTL training epochs.', type=int)
parser.add_argument('-regression', default=True, action='store_false',
                    help='Whether the task is regression or (binary) classification.')
parser.add_argument('-runs', type=int, default=5,
                    help='Number of runs.')
parser.add_argument('-es_patience', default=3, type=int,
                    help='Early Stopping patience.')
parser.add_argument('-pl_patience', default=1, type=int,
                    help='Reduce learning rate on plateau patience.')
parser.add_argument('-verbose', default=False, action='store_true',
                    help='Print training process.')
parser.add_argument('-tune_arch', default=False, action='store_true',
                    help='Tune the MLP architecture.')
parser.add_argument('-show_full_scores', default=False, action='store_true',
                    help='Prints a Pandas DataFrame with multiple scores in the MTL setting.')
parser.add_argument('-save_plots', default=False, action='store_true',
                    help='Whether to save plots of Accuracy/MSE-Fidelity.')

args = parser.parse_args()

e = Experiment(args)
e.run()
