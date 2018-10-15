import argparse



arg_lists = []
parser = argparse.ArgumentParser()

parser.add_argument('--n_tranformers', type=int, default=2)
parser.add_argument('--n_scalers', type=int, default=3)
parser.add_argument('--n_constructers', type=int, default=2)
parser.add_argument('--n_selecters', type=int, default=4)
parser.add_argument('--n_models', type=int, default=6)
parser.add_argument('--lstm_size', type=int, default=30)
parser.add_argument('--temperature', type=float, default=5.0)
parser.add_argument('--tanh_constant', type=int, default=2.5)
parser.add_argument('--save_dir', type=str, default='logs')
parser.add_argument('--load_path', type=str, default='')
parser.add_argument('--dataset', type=int, default=6)
parser.add_argument('--log_level', type=str, default='INFO', choices=['INFO', 'DEBUG', 'WARN'])
parser.add_argument('--log_dir', type=str, default='logs')
parser.add_argument('--log_step', type=int, default=50)
parser.add_argument('--data_dir', type=str, default='data')
parser.add_argument('--controller_max_step', type=int, default=1000,
                       help='step for controller parameters')


def get_args():
    """Parses all of the arguments above, which mostly correspond to the
    hyperparameters mentioned in the paper.
    """
    args, unparsed = parser.parse_known_args()

    return args, unparsed


