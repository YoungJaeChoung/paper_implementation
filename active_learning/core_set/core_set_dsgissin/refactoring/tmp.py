import argparse


def parse_input_test():
    p = argparse.ArgumentParser()
    p.add_argument('-expidx', '--experiment_index', type=int, help="index of current experiment")
    args = p.parse_args()
    return args


args = parse_input_test()
print(args)
