#!/usr/bin/env python3

import argparse, sys

parser = argparse.ArgumentParser()
parser.add_argument('src', type=str)
parser.add_argument('dst', type=str)
ARGS = vars(parser.parse_args())

with open(ARGS['dst'], 'w') as f_dst:
    for line in open(ARGS['src']):
        tokens = line.strip().split()
        label = tokens[0]
        if float(label) == 1:
            output = '1'
        else:
            output = '-1'
        for token in tokens[1:]:
            dim, val = token.split(':')
            output += ' {dim}:{dim}:{val}'.format(dim=dim, val=val)
        f_dst.write(output + '\n')
