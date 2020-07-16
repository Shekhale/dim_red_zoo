from __future__ import division
from data import load_dataset
import argparse
import numpy as np
import torch

from triplet import train_triplet
from acai import train_acai
from support_func import  sanitize

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        group.add_argument(*args, **kwargs)

    group = parser.add_argument_group('dataset options')
    aa("--database", default="sift")
    aa("--method", type=str, default="triplet")

    group = parser.add_argument_group('Model hyperparameters')
    aa("--dout", type=int, default=16,
       help="output dimension")
    group = parser.add_argument_group('Computation params')
    aa("--seed", type=int, default=1234)
    aa("--device", choices=["cuda", "cpu", "auto"], default="auto")
    aa("--val_freq", type=int, default=10,
       help="frequency of validation calls")
    aa("--print_results", type=int, default=0)
    aa("--batch_size", type=int, default=64)
    aa("--epochs", type=int, default=40)

    args = parser.parse_args()

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(args)

    results_file_name = "/home/shekhale/results/dim_red_zoo/" + args.database + "/train_results_" + args.method + ".txt"
    if args.print_results > 0:
        with open(results_file_name, "a") as rfile:
            rfile.write("\n\n")
            rfile.write("START TRAINING \n")

    print ("load dataset %s" % args.database)
    (_, xb, xq, _) = load_dataset(args.database, args.device, calc_gt=False, mnt=True)

    base_size = xb.shape[0]
    threshold = int(base_size * 0.1)
    perm = np.random.permutation(base_size)
    xv = xb[perm[:threshold]]
    xt = xb[perm[threshold:]]

    print(xb.shape, xt.shape, xv.shape, xq.shape)

    xt = sanitize(xt)
    xv = sanitize(xv)
    xb = sanitize(xb)
    xq = sanitize(xq)

    if args.method == "triplet":
        train_triplet(xt, xv, xq, args, results_file_name)
    else if args.method == "acai":
        train_acai(xt, xv, xq, args, results_file_name)
    else:
        print("Select an available method")
