from argparse import ArgumentParser


def get_config():
    parser = ArgumentParser()
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--seed', type=int, default=2, help='random seed')
    parser.add_argument('--n_input', type=int, default=9, help='n_input')
    parser.add_argument('--n_output', type=int, default=6, help='n_output')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=.99, help='momentum of the som update')
    parser.add_argument('-b', '--batch_size', default=256, type=int, help='mini-batch size (default: 32)')
    parser.add_argument('-e', '--epochs', default=100, type=int, help='number of total epochs (default: 30)')

    args, unknown = parser.parse_known_args()
    return args
