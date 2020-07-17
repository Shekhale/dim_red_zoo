from __future__ import division
import time
import numpy as np
from torch import nn, optim

import torch

from support_func import Normalize, repeat, pairwise_NNs_inner, \
                         loss_permutation, loss_top_1_in_lat_top_k, forward_pass, get_nearestneighbors


# b_d   b_k_d
def get_dist(x, y, args, loss_type="my", alpha=0):
    y = y.reshape(-1, y.shape[-1])
    y = y.transpose(0,1)
    if alpha == 0:
        alpha = args.dout
    if loss_type == "my":
        return (x ** 2).sum(-1, keepdim=True) + (y ** 2).sum(0, keepdim=True) - 2 * torch.mm(x, y)
    elif loss_type == "my_skpr":
        return torch.exp(torch.mm(x, y))
    elif loss_type == "stud":
        return (1 + ((x ** 2).sum(-1, keepdim=True)
                     + (y ** 2).sum(0, keepdim=True)
                     - 2 * torch.mm(x, y)) / alpha) ** (-(alpha + 1) / 2)
    else:
        return torch.exp(-((x ** 2).sum(-1, keepdim=True) + (y ** 2).sum(0, keepdim=True) - 2 * torch.mm(x, y)))


# b_d   b_k_d
def get_dist_2(x, y, args, loss_type):
    x = x.reshape(x.shape[0], 1, x.shape[1])
    if loss_type == "my":
        return ((x - y) ** 2).sum(-1)
    if loss_type == "my_skpr":
        return torch.exp((x*y).sum(-1))
    elif loss_type == "stud":
        alpha = args.dout
        return (1 + ((x - y) ** 2).sum(-1) / alpha) ** (-(alpha + 1) / 2)
    else:
        return torch.exp(-((x - y) ** 2).sum(-1))


def div_loss(x, y, z0, args):
    x = x - z0
    y = y - z0
    x = x / x.norm(dim=-1, keepdim=True)
    y = y / y.norm(dim=-1, keepdim=True)
    # x = normalize(x - z0)
    # y = normalize(y - z0)
    sk = x*y
    return sk.sum(-1)


def tsne_optimize(xt, xv, xq, net, args,  k, lambda_uniform, lambda_div, val_k, loss_type):

    lr_schedule = [float(x.rstrip().lstrip()) for x in args.lr_schedule.split(",")]
    assert args.epochs % len(lr_schedule) == 0
    lr_schedule = repeat(lr_schedule, args.epochs // len(lr_schedule))
    print("Lr schedule", lr_schedule)

    N = xt.shape[0]

    xt_var = torch.from_numpy(xt).to(args.device)

    print("  Find k_nn")
    k_nn = get_nearestneighbors(xt, xt, k, args.device, needs_exact=True)
    # prepare optimizer
    optimizer = optim.SGD(net.parameters(), lr_schedule[0], momentum=args.momentum)
    pdist = nn.PairwiseDistance(2)

    prob = torch.ones(k)
    for i in range(k):
        prob[i] = - 2 * (i + 1) / args.dout
    prob = torch.exp(prob)
    # print(prob)
    prob = prob[None, :].to(args.device)

    all_logs = []
    for epoch in range(args.epochs):
        # Update learning rate
        args.lr = lr_schedule[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr

        t0 = time.time()

        net.eval()

        print("  Forward pass")
        yt = forward_pass(net, xt, 1024)
        if lambda_div > 0:
            k_nn_lat = get_nearestneighbors(yt, yt, k, args.device, needs_exact=False)
            del yt
        # training pass
        print("  Train")
        net.train()
        avg_tsne, avg_uniform, avg_div, avg_loss = 0, 0, 0, 0

        # process dataset in a random order
        perm = np.random.permutation(N)

        t1 = time.time()
        idx_batch = 0

        for i0 in range(0, N, args.batch_size):
            i1 = min(i0 + args.batch_size, N)
            n = i1 - i0

            data_idx = perm[i0:i1]

            # anchor, positives, negatives
            bxt = xt_var[data_idx]
            b_k_nn = xt_var[k_nn[data_idx]]

            b_k_nn = b_k_nn.reshape(-1, b_k_nn.shape[-1])

            # do the forward pass (+ record gradients)
            bxt, b_k_nn = net(bxt), net(b_k_nn)
            b_k_nn = b_k_nn.reshape(-1, k, b_k_nn.shape[-1])

            # t-sne loss
            if loss_type == "my":
                dist_b_k = get_dist_2(bxt, b_k_nn, args, loss_type)
                prob_lat = dist_b_k.sum(-1)
                loss_tsne = prob_lat.mean()
            else:
                dist_b_bk = get_dist(bxt, b_k_nn, args, loss_type)
                norm_coeff = dist_b_bk.sum(1, keepdim=True)
                dist_b_k = get_dist_2(bxt, b_k_nn, args, loss_type)
                prob_lat = (dist_b_k / norm_coeff)
                kl_div = - (prob * torch.log(prob_lat)).sum(-1)
                loss_tsne = kl_div.mean()

            # entropy loss
            I = pairwise_NNs_inner(bxt.data)
            distances = pdist(bxt, bxt[I])
            loss_uniform = - torch.log(distances).mean()

            # diversity loss
            if lambda_div > 0:
                b_k_nn_lat = xt_var[k_nn_lat[data_idx]]
                b_k_nn_lat = b_k_nn_lat.reshape(-1, b_k_nn_lat.shape[-1])
                b_k_nn_lat = net(b_k_nn_lat)
                b_k_nn_lat = b_k_nn_lat.reshape(-1, k, b_k_nn_lat.shape[-1])
            loss_div = torch.zeros_like(loss_uniform)

            # combined loss
            loss = loss_tsne + lambda_uniform * loss_uniform + lambda_div * loss_div

            # collect some stats
            avg_tsne += loss_tsne.data.item()
            avg_uniform += loss_uniform.data.item()
            avg_div += loss_div.data.item()
            avg_loss += loss.data.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            idx_batch += 1

        avg_tsne /= idx_batch
        avg_uniform /= idx_batch
        avg_div /= idx_batch
        avg_loss /= idx_batch

        logs = {
            'epoch': epoch,
            'loss_tsne': avg_tsne,
            'loss_div': avg_div,
            'loss_uniform': avg_uniform,
            'loss': avg_loss,
            # 'offending': offending,
            'lr': args.lr
        }

        t2 = time.time()
        # maybe perform a validation run
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

        # synthetic logging
        print ('epoch %d, times: [hn %.2f s epoch %.2f s val %.2f s]'
               ' lr = %f'
               ' loss = %g = %g + lam_u * %g + lam_div * %g' % (
            epoch, t1 - t0, t2 - t1, t3 - t2,
            args.lr,
            avg_loss, avg_tsne, avg_uniform, avg_div
        ))

        logs['times'] = (t1 - t0, t2 - t1, t3 - t2)

        all_logs.append(logs)

    return all_logs


def train_tsne(xt, xv, xq, args, results_file_name):


    k = 40
    lambda_div = 0
    lambda_uniform = 40
    loss_type = "my"

    print ("build network")
    dim = xt.shape[1]
    dint, dout = args.dint, args.dout
    net = nn.Sequential(
        nn.Linear(in_features=dim, out_features=dint, bias=True),
        nn.BatchNorm1d(dint),
        nn.ReLU(),
        nn.Linear(in_features=dint, out_features=2*dint, bias=True),
        nn.BatchNorm1d(2*dint),
        nn.ReLU(),
        nn.Linear(in_features=2*dint, out_features=4*dint, bias=True),
        nn.BatchNorm1d(4*dint),
        nn.ReLU(),
        nn.Linear(in_features=4*dint, out_features=dout, bias=True),
        Normalize()
    )

    net.to(args.device)
    val_k = 2 * args.dout
    all_logs = tsne_optimize(xt, xv, xq, net, args, k, lambda_uniform, lambda_div, val_k, loss_type)

    if args.print_results > 0:
        with open(results_file_name, "a") as rfile:
            rfile.write("\n")
            rfile.write("T-SNE, DATABASE %s, xt_size = %d, batch_size = %d, dint = %d, dout = %d, k = %d, lam_u = %.5f, lambda_div = %.7f, loss_type %s \n" %
                (args.database, xt.shape[0], args.batch_size, dint, args.dout, k, lambda_uniform, lambda_div, loss_type))

            log = all_logs[-1]
            rfile.write("last perm = %.4f, train_top1_k = %.3f,  valid_top1_k = %.3f, query_top1_k = %.3f, query_top1_2k = %.3f \n" %
                (log['perm'], log['train_top1_k'], log['valid_top1_k'], log['query_top1_k'], log['query_top1_2k']))
            rfile.write("last logs: epochs %d, loss_tsne = %.6f, loss_uniform = %.6f, loss_dim = %.6f, loss = %.6f, times %f %f %f \n" %
                        (log['epoch'] + 1, log['loss_tsne'], log['loss_uniform'], log['loss_div'], log['loss'],
                         log['times'][0], log['times'][1], log['times'][2]))
