from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser(description='Implementation of many methods for detecting OOD samples.')

    parser.add_argument('--validation-size', default=5000, type=int, help='Number of validation set. (default 5000)')

    args = parser.parse_args()

    return args
