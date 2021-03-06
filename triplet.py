from __future__ import division
import time
import numpy as np
from torch import nn, optim
import torch.nn.functional as F
import torch

from support_func import  loss_permutation, loss_top_1_in_lat_top_k, normalize_numpy,\
                          get_nearestneighbors, sanitize, forward_pass, Normalize,\
                          repeat, pairwise_NNs_inner


def get_params():
    print("Dimensionality reduction via `triplet` method implies a specification the following parameters (types and default values atttached):\n")
    print("Enter separated by spaces:")
    print("margin (float, 0.0), lambda_uniform (float, 0.01), dint (uint, 1024), r_pos (uint, 5), r_neg (uint, 10)")

    args = input()
    # dt = np.dtype([('margin', np.float64), ('lambda_uniform', np.float64), ('dint', np.uint32),\
    #                ('r_pos', np.uint32), ('r_neg', np.uint32)])
    # args = np.array(args.split(), dtype=dt)
    args = args.split()

    return float(args[0]), float(args[1]), int(args[2]), int(args[3]), int(args[4])


def triplet_optimize(xt, xv, gt_nn, xq, net, args, lambda_uniform, kpos, rank_negative, val_k, margin):
    lr_schedule = [float(x.rstrip().lstrip()) for x in args.lr_schedule.split(",")]
    assert args.epochs % len(lr_schedule) == 0
    lr_schedule = repeat(lr_schedule, args.epochs // len(lr_schedule))
    print("Lr schedule", lr_schedule)

    N = gt_nn.shape[0]

    xt_var = torch.from_numpy(xt).to(args.device)

    # prepare optimizer
    optimizer = optim.SGD(net.parameters(), lr_schedule[0], momentum=args.momentum)
    pdist = nn.PairwiseDistance(2)

    all_logs = []
    for epoch in range(args.epochs):
        # Update learning rate
        args.lr = lr_schedule[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr

        t0 = time.time()

        # Sample positives for triplet
        rank_pos = np.random.choice(kpos, size=N)
        positive_idx = gt_nn[np.arange(N), rank_pos]

        # Sample negatives for triplet
        net.eval()
        print("  Forward pass")
        xl_net = forward_pass(net, xt, 1024)
        print("  Distances")
        I = get_nearestneighbors(xl_net, xl_net, rank_negative, args.device, needs_exact=False)
        negative_idx = I[:, -1]

        # training pass
        print("  Train")
        net.train()
        avg_triplet, avg_uniform, avg_loss = 0, 0, 0
        offending = idx_batch = 0

        # process dataset in a random order
        perm = np.random.permutation(N)

        t1 = time.time()

        for i0 in range(0, N, args.batch_size):
            i1 = min(i0 + args.batch_size, N)
            n = i1 - i0
            data_idx = perm[i0:i1]

            # anchor, positives, negatives
            ins = xt_var[data_idx]
            pos = xt_var[positive_idx[data_idx]]
            neg = xt_var[negative_idx[data_idx]]

            # do the forward pass (+ record gradients)
            ins, pos, neg = net(ins), net(pos), net(neg)

            # triplet loss
            per_point_loss = pdist(ins, pos) - pdist(ins, neg) + margin
            per_point_loss = F.relu(per_point_loss)
            loss_triplet = per_point_loss.mean()
            offending += torch.sum(per_point_loss.data > 0).item()

            # entropy loss
            I = pairwise_NNs_inner(ins.data)
            distances = pdist(ins, ins[I])
            loss_uniform = - torch.log(distances).mean()

            # combined loss
            loss = loss_triplet + lambda_uniform * loss_uniform

            # collect some stats
            avg_triplet += loss_triplet.data.item()
            avg_uniform += loss_uniform.data.item()
            avg_loss += loss.data.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            idx_batch += 1

        avg_triplet /= idx_batch
        avg_uniform /= idx_batch
        avg_loss /= idx_batch

        logs = {
            'epoch': epoch,
            'loss_triplet': avg_triplet,
            'loss_uniform': avg_uniform,
            'loss': avg_loss,
            'offending': offending,
            'lr': args.lr
        }
        all_logs.append(logs)

        t2 = time.time()

        if (epoch + 1) % args.val_freq == 0 or epoch == args.epochs - 1:
            yt = forward_pass(net, xt, 1024)
            yv = forward_pass(net, xv, 1024)
            logs['perm'] = loss_permutation(xt, yt, args, k=val_k, size=10**4)

            logs['train_top1_k'] = loss_top_1_in_lat_top_k(xt, xt, yt, yt, args, 2, val_k, size=10**5, name="TRAIN")
            logs['valid_top1_k'] = loss_top_1_in_lat_top_k(xv, xt, yv, yt, args, 1, val_k, size=10**5, name="VALID")

            yq = forward_pass(net, xq, 1024)
            logs['query_top1_k'] = loss_top_1_in_lat_top_k(xq, xt, yq, yt, args, 1, val_k, size=10**4, name="QUERY_tr")
            logs['query_top1_2k'] = loss_top_1_in_lat_top_k(xq, xt, yq, yt, args, 1, 2*val_k, size=10**4, name="QUERY_tr")

            net.train()

        t3 = time.time()

        print ('epoch %d, times: [hn %.2f s epoch %.2f s val %.2f s]'
               ' lr = %f'
               ' loss = %g = %g + lam * %g, offending %d' % (
            epoch, t1 - t0, t2 - t1, t3 - t2,
            args.lr,
            avg_loss, avg_triplet, avg_uniform, offending
        ))

        logs['times'] = (t1 - t0, t2 - t1, t3 - t2)

    return all_logs


def train_triplet(xt, xv, xq, args, results_file_name):

    margin, lambda_uniform, dint, r_pos, r_neg = get_params()

    print(margin, lambda_uniform, dint, r_pos, r_neg)

    print ("computing training ground truth")
    xt_gt = get_nearestneighbors(xt, xt, r_pos, device=args.device, needs_exact=True)

    print ("build network")

    dim = xt.shape[1]
    dout = args.dout

    net = nn.Sequential(
        nn.Linear(in_features=dim, out_features=dint, bias=True),
        nn.BatchNorm1d(dint),
        nn.ReLU(),
        nn.Linear(in_features=dint, out_features=dint, bias=True),
        nn.BatchNorm1d(dint),
        nn.ReLU(),
        nn.Linear(in_features=dint, out_features=dout, bias=True),
        Normalize()
    )

    net.to(args.device)
    val_k = 2 * args.dout
    all_logs = triplet_optimize(xt, xv, xt_gt, xq, net, args, lambda_uniform, r_pos, r_neg, val_k, margin)

    if args.print_results > 0:
        with open(results_file_name, "a") as rfile:
            rfile.write("\n")
            rfile.write(
                "Triplet, DATABASE %s, xt_size = %d, batch_size = %d, lat_dim = %d, k = %d, lam_u = %.5f, r_pos = %d, r_neg = %d , dint = %d, margin = %.5f \n" %
                (args.database, xt.shape[0], args.batch_size, args.dout, val_k, lambda_uniform, r_pos, r_neg, dint, margin))

            log = all_logs[-1]
            rfile.write(
                "last perm = %.4f, train_top1_k = %.3f,  valid_top1_k = %.3f, query_top1_k = %.3f, query_top1_2k = %.3f \n" %
                (log['perm'], log['train_top1_k'], log['valid_top1_k'], log['query_top1_k'],
                 log['query_top1_2k']))

            rfile.write(
                "last logs: epochs %d, loss_uniform = %.6f, loss_triplet = %.6f, loss = %.6f, offending = %d, times %f %f %f \n" %
                (log['epoch'] + 1, log['loss_uniform'], log['loss_triplet'], log['loss'], log['offending'],
                 log['times'][0], log['times'][1], log['times'][2]))
            rfile.write("------------------------------------------------------ \n")