from __future__ import division
from lib.data import load_dataset
import time
import argparse
import numpy as np
from torch import nn, optim
from lib.metrics import get_nearestneighbors, sanitize
from lib.net import Normalize

import torch.nn.functional as F
import torch


from support_func import loss_permutation, loss_top_1_in_lat_top_k, get_weights,\
                         repeat, pairwise_NNs_inner, forward_pass_enc


def swap_halves(x):
    a, b = x.split(x.shape[0]//2)
    return torch.cat([b, a])


def lerp(start, end, weights):
    return start + weights * (end - start)


class Discriminator(nn.Module):
    def __init__(self, x_dim, hidden_dim, mult=1):
        super().__init__()

        self.discr = nn.Sequential(
            nn.Linear(in_features=x_dim, out_features=hidden_dim, bias=True),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim, bias=True),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim, bias=True),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            # nn.Linear(in_features=hidden_dim, out_features=1, bias=True),
            # nn.BatchNorm1d(hidden_dim),
            # nn.Sigmoid()
        )

    def forward(self, x):
        predict = self.discr(x)
        # x = x.reshape(x.shape[0], -1)
        predict = torch.mean(predict, -1)
        return predict


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim, mult, lat_type="my"):
        super().__init__()

        hidden_dim1 = int(mult*hidden_dim)
        hidden_dim2 = int(mult*hidden_dim1)
        self.enc = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=hidden_dim, bias=True),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim1, bias=True),
            nn.BatchNorm1d(hidden_dim1),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim1, out_features=hidden_dim2, bias=True),
            nn.BatchNorm1d(hidden_dim2),
            nn.ReLU()
        )
        self.enc_m = nn.Sequential(nn.Linear(in_features=hidden_dim2, out_features=z_dim, bias=True),
                                   Normalize())

        self.lat_type = lat_type
        self.fc_var = nn.Linear(hidden_dim2, 1)

    def forward(self, x):
        x = self.enc(x)
        y_mu = self.enc_m(x)

        if self.lat_type == "my":
            y_var = y_mu
        elif self.lat_type == "spherical":
            y_var = F.softplus(self.fc_var(x)) + 1
        else:
            raise NotImplemented

        return y_mu, y_var


class Decoder(nn.Module):
    def __init__(self, z_dim, hidden_dim, output_dim, mult):
        super().__init__()
        hidden_dim1 = int(mult*hidden_dim)
        hidden_dim2 = int(mult*hidden_dim1)

        self.dec = nn.Sequential(
            nn.Linear(in_features=z_dim, out_features=hidden_dim2, bias=True),
            nn.BatchNorm1d(hidden_dim2),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim2, out_features=hidden_dim1, bias=True),
            nn.BatchNorm1d(hidden_dim1),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim1, out_features=hidden_dim, bias=True),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=output_dim, bias=True),
            Normalize()
        )

    def forward(self, x):
        predicted = self.dec(x)

        return predicted


# def get_params():
#     print("Dimensionality reduction via `ACAI` method implies a specification the following parameters (types and default values atttached):\n")
#     print("Enter separated by spaces:")


def acai_optimize(xt, xv, xq, Enc, Dec, Discr, args, lambda_triplet, sigma, weights, val_k,\
                  tr_pos, tr_neg, tr_margin, batch_size, epochs, gamma = 0.5, lambda_fake=1):

    N = xt.shape[0]
    xt_var = torch.from_numpy(xt).to(args.device)

    optimizerEnc = optim.Adam(Enc.parameters(), lr=0.0001)
    optimizerDec = optim.Adam(Dec.parameters(), lr=0.0001)
    optimizerDiscr = optim.Adam(Discr.parameters(), lr=0.0001)

    pdist = nn.PairwiseDistance(2)

    gt_nn = get_nearestneighbors(xt, xt, tr_pos, device=args.device, needs_exact=True)

    all_logs = []
    for epoch in range(epochs):
        t0 = time.time()

        # Sample positives for triplet
        rank_pos = np.random.choice(tr_pos, size=N)
        positive_idx = gt_nn[np.arange(N), rank_pos]

        xl_net = forward_pass_enc(Enc, xt, 1024, args.device)
        print("  Distances")
        I = get_nearestneighbors(xl_net, xl_net, tr_neg, args.device, needs_exact=False)
        negative_idx = I[:, -1]


        # model.train()
        Enc.train()
        Dec.train()
        Discr.train()

        perm = np.random.permutation(N)
        # loss of the epoch
        train_loss, loss_uniform, loss_discr = 0, 0, 0
        avg_sim, avg_uniform, avg_div, avg_loss = 0, 0, 0, 0
        avg_recon, avg_kl = 0, 0
        avg_discr, avg_fake, avg_triplet = 0, 0, 0
        idx_batch, offending = 0, 0
        t1 = time.time()
        for i0 in range(0, N, batch_size):
            i1 = min(i0 + batch_size, N)
        # for i0 in range(0, N, args.batch_size):
        #     i1 = min(i0 + args.batch_size, N)
            n = i1 - i0

            data_idx = perm[i0:i1]

            # anchor, positives, negatives
            x = xt_var[data_idx]
            x_weights = torch.from_numpy(weights[data_idx]).to(args.device)


            """ Update Discriminator """

            y, _ = Enc(x)
            x_ = Dec(y)
            discr_l2 = Discr(torch.lerp(x_, x, gamma))
            loss_discr_l2 = (discr_l2 ** 2).mean()

            alpha = torch.rand(n).to(args.device)
            alpha = 0.5 - torch.abs(alpha - 0.5)  # Make interval [0, 0.5]
            alpha = alpha.reshape(n, 1)

            y_mix = torch.lerp(y, swap_halves(y), alpha)
            x_mix = Dec(y_mix)
            discr_mix = Discr(x_mix)

            loss_discr_mix = ((discr_mix - alpha.reshape(-1)) ** 2).mean()
            loss_discr = 0.5 * (loss_discr_mix + loss_discr_l2)

            optimizerDiscr.zero_grad()
            loss_discr.backward()
            optimizerDiscr.step()

            avg_discr += loss_discr.data.item()

            optimizerEnc.zero_grad()
            optimizerDec.zero_grad()

            """ Triplet """
            x_pos = xt_var[positive_idx[data_idx]]
            x_neg = xt_var[negative_idx[data_idx]]

            # do the forward pass (+ record gradients)
            y, _ = Enc(x)
            y_pos, _ = Enc(x_pos)
            y_neg, _ = Enc(x_neg)

            # triplet loss
            per_point_loss = pdist(y, y_pos) - pdist(y, y_neg) + tr_margin
            per_point_loss = F.relu(per_point_loss)
            loss_triplet = per_point_loss.mean()
            offending += torch.sum(per_point_loss.data > 0).item()

            """ Update Encoder to fake Discriminator """
            y, _ = Enc(x)
            x_ = Dec(y)

            alpha = torch.rand(n).to(args.device)
            alpha = 0.5 - torch.abs(alpha - 0.5)  # Make interval [0, 0.5]
            alpha = alpha.reshape(n, 1)

            y_mix = torch.lerp(y, swap_halves(y), alpha)
            x_mix = Dec(y_mix)
            discr_mix = Discr(x_mix)

            loss_ae_fake = (discr_mix ** 2).mean()

            """ Update AUTOENCODER """
            y, y_var = Enc(x)

            if args.lat_type == "my":
                shift = torch.randn_like(y) - 0.5 * torch.ones_like(y)
                y_shifted = y + shift * sigma
            else:
                raise NotImplemented

            x_rec = Dec(y_shifted)

            # reconstruction loss
            recon_error = ((x - x_rec) ** 2).sum(-1)
            loss_recon = (x_weights * recon_error).mean()

            # entropy loss
            I = pairwise_NNs_inner(y.data)
            distances = pdist(y, y[I])
            loss_uniform = - torch.log(distances).mean()

            loss = loss_recon + lambda_triplet * loss_triplet + lambda_fake * loss_ae_fake

            # backward pass
            loss.backward()

            avg_recon += loss_recon.data.item()
            avg_fake += loss_ae_fake.data.item()
            avg_uniform += loss_uniform.data.item()
            avg_triplet += loss_triplet.data.item()
            avg_loss += loss.data.item()

            # update the weights
            optimizerEnc.step()
            optimizerDec.step()
            idx_batch += 1

        avg_sim /= idx_batch
        avg_uniform /= idx_batch
        avg_div /= idx_batch
        avg_loss /= idx_batch
        avg_recon /= idx_batch
        avg_discr /= idx_batch
        avg_fake /= idx_batch
        avg_triplet /= idx_batch
        avg_kl /= idx_batch

        logs = {
            'epoch': epoch,
            'loss_sim': avg_sim,
            'loss_div': avg_div,
            'loss_uniform': avg_uniform,
            'loss_kl': avg_kl,
            'loss_recon': avg_recon,
            'loss_discr': avg_discr,
            'loss_triplet': avg_triplet,
            'loss_fake': avg_fake,
            'loss': avg_loss,
            'offending': offending,
        }

        t2 = time.time()
        if (epoch + 1) % args.val_freq == 0 or epoch == args.epochs - 1:
            yt = forward_pass_enc(Enc, xt, 1024)
            yv = forward_pass_enc(Enc, xv, 1024)
            logs['perm'] = loss_permutation(xt, yt, args, k=val_k, size=10**4)

            logs['train_top1_k'] = loss_top_1_in_lat_top_k(xt, xt, yt, yt, args, 2, val_k, size=10**5, name="TRAIN")
            logs['valid_top1_k'] = loss_top_1_in_lat_top_k(xv, xt, yv, yt, args, 1, val_k, size=10**5, name="VALID")

            yq = forward_pass_enc(Enc, xq, 1024)
            logs['query_top1_k'] = loss_top_1_in_lat_top_k(xq, xt, yq, yt, args, 1, val_k, size=10**4, name="QUERY_tr")
            logs['query_top1_2k'] = loss_top_1_in_lat_top_k(xq, xt, yq, yt, args, 1, 2*val_k, size=10**4, name="QUERY_tr")

            Enc.train()

        t3 = time.time()

        # synthetic logging
        print ('epoch %d, times: [hn %.2f s epoch %.2f s val %.2f s]'
               ' loss = %g = %g + lam_u * %g +lam_f * %g + lambda_triplet * %g, discr = %g, offending = %g,  ' % (
            epoch, t1 - t0, t2 - t1, t3 - t2,
            avg_loss, avg_recon, avg_uniform, avg_fake, avg_triplet, avg_discr, offending
        ))

        logs['times'] = (t1 - t0, t2 - t1, t3 - t2)
        all_logs.append(logs)

    return all_logs

def train_acai(xt, xv, xq, args, results_file_name):

    # lambda_triplet, lambda_fake, tr_margin, tr_pos, tr_neg, sigma, gamma, dint = get_params()
    lambda_triplet, lambda_fake = 1, 1
    tr_margin, tr_pos, tr_neg = 0.1, 10, 25
    sigma, gamma = 0.01, 0.5
    dint, mult = 1024, 1


    print ("build network")
    dim = xt.shape[1]
    ldim = args.dout

    # print(dim, dint, ldim)
    encoder = Encoder(dim, dint, ldim, mult, args.lat_type).to(args.device)
    decoder = Decoder(ldim, dint, dim, mult).to(args.device)
    discriminator = Discriminator(dim, dint).to(args.device)

    weights = np.ones(xt.shape[0])
    if args.smart_weights == 1:
        weights = get_weights(xt, 50, args)

    all_logs = acai_optimize(xt, xv, xq, encoder, decoder, discriminator, args, lambda_triplet, sigma, weights,\
                             tr_pos, tr_neg, tr_margin, args.batch_size, args.epochs, gamma, lambda_fake)

    if args.print_results > 0:
        with open(results_file_name, "a") as rfile:
            rfile.write("\n")
            rfile.write("\n")
            log = all_logs[-1]
            rfile.write("ACAI, DATABASE %s, xt_size = %d, batch_size = %d, lat_dim = %d, dint = %d, epochs %d \n" %
                (args.database, xt.shape[0], args.batch_size, args.dout, dint, log['epoch'] + 1))
            rfile.write("tr_pos = %d, tr_neg = %d, sigma = %.7f, net_mult = %.3f, margin = %.3f,  bs = %d, lambda_fake = %.2f, lambda_triplet = %.2f  \n" %
                        ( tr_pos, tr_neg, sigma, mult, tr_margin, args.batch_size, lambda_fake, lambda_triplet))
            rfile.write("last perm = %.4f, train_top1 = %.3f, valid_top1 = %.3f, query_top1_50 = %.3f, query_top1_100 = %.3f \n" %
                        (log['perm'], log['train_top1'], log['valid_top1'],  log['query_top1_50'], log['query_top1_100']))

            rfile.write(" loss_uniform = %.6f, loss_recon = %.6f,loss_triplet = %.6f, loss = %.6f, offending = %d, times %f %f %f \n" %
                        (log['loss_uniform'], log['loss_recon'], log['loss_triplet'], log['loss'], log['offending'],
                         log['times'][0], log['times'][1], log['times'][2]))
            rfile.write("------------------------------------------------------ \n")


            rfile.write("\n")


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
