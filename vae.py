import time
import numpy as np
from torch import nn, optim
import torch.nn.functional as F
import torch

from support_func import loss_permutation, loss_top_1_in_lat_top_k, forward_pass_enc, get_weights, pairwise_NNs_inner,\
                         Normalize, get_nearestneighbors


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


class VAE(nn.Module):
    def __init__(self, enc, dec, sigma):
        super().__init__()

        self.enc = enc
        self.dec = dec
        self.sigma = sigma

    def forward(self, x):
        # encode
        z_mu = self.enc(x)

        # sample from the distribution having latent parameters z_mu, z_var
        # reparameterize

        eps = torch.randn_like(z_mu)
        x_sample = z_mu + eps * self.sigma

        # decode
        predicted = self.dec(x_sample)
        return predicted, z_mu, x_sample


def vae_optimize(xt, xv, xq, Enc, Dec, args, lambda_triplet, \
                        sigma, weights, tr_pos, tr_neg, tr_margin, lr, val_k):

    N = xt.shape[0]
    xt_var = torch.from_numpy(xt).to(args.device)
    optimizerEnc = optim.Adam(Enc.parameters(), lr=lr)
    optimizerDec = optim.Adam(Dec.parameters(), lr=lr)

    pdist = nn.PairwiseDistance(2)
    gt_nn = get_nearestneighbors(xt, xt, tr_pos, device=args.device, bs=10**5, needs_exact=True)

    all_logs = []
    for epoch in range(args.epochs):
        t0 = time.time()

        # Sample positives for triplet
        rank_pos = np.random.choice(tr_pos, size=N)
        positive_idx = gt_nn[np.arange(N), rank_pos]

        xl_net = forward_pass_enc(Enc, xt, 1024, args.device)
        print("  Distances")
        I = get_nearestneighbors(xl_net, xl_net, tr_neg, args.device, needs_exact=False)
        # I = get_nearestneighbors_partly(xl_net, qt(xl_net), rank_negative, args.device, bs=10**5, needs_exact=False)
        negative_idx = I[:, -1]
        Enc.train()
        Dec.train()
        perm = np.random.permutation(N)
        avg_sim, avg_uniform, avg_div, avg_loss = 0, 0, 0, 0
        avg_recon, avg_kl = 0, 0
        avg_discr, avg_fake, avg_triplet = 0, 0, 0
        idx_batch, offending = 0, 0
        t1 = time.time()
        for i0 in range(0, N, args.batch_size):
            i1 = min(i0 + args.batch_size, N)
            data_idx = perm[i0:i1]

            # anchor, positives, negatives
            x = xt_var[data_idx]
            x_weights = torch.from_numpy(weights[data_idx]).to(args.device)

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



            if args.lat_type == "my":
                shift = torch.randn_like(y) - 0.5 * torch.ones_like(y)
                y_shifted = y + shift * sigma
            # elif args.lat_type == "spherical":
            #     y_var *= sigma
            #     q_y = VonMisesFisher(y, y_var)
            #     # print(type(q_y))
            #     p_y = HypersphericalUniform(args.dout - 1, args.device)
            #     y_shifted = q_y.rsample()
            #     # y_shifted = y
            else:
                raise NotImplemented

            x_rec = Dec(y_shifted)

            # reconstruction loss
            # recon_loss = F.binary_cross_entropy(x_sample, x, size_average=False)
            recon_error = ((x - x_rec) ** 2).sum(-1)
            loss_recon = (x_weights * recon_error).mean()

            # # kl divergence loss
            # if args.lat_type == "my":
            #     loss_kl = torch.zeros(x.shape[0]).mean()
            # elif args.lat_type == "spherical":
            #     loss_kl = torch.distributions.kl.kl_divergence(q_y, p_y).mean()
            # else:
            #     raise NotImplemented

            # loss_kl = 0.5 * torch.sum(torch.exp(z_var) + z_mu ** 2 - 1.0 - z_var)

            # entropy loss
            I = pairwise_NNs_inner(y.data)
            distances = pdist(y, y[I])
            loss_uniform = - torch.log(distances).mean()

            # total loss
            loss = loss_recon + lambda_triplet * loss_triplet
            # loss = loss_recon + lambda_uniform * loss_uniform + lambda_triplet * loss_triplet

            # backward pass
            loss.backward()

            avg_recon += loss_recon.data.item()
            # avg_fake += loss_fake.data.item()
            avg_uniform += loss_uniform.data.item()
            avg_triplet += loss_triplet.data.item()
            # avg_div += loss_div.data.item()
            avg_loss += loss.data.item()
            # avg_kl += loss_kl.data.item()

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

        # maybe perform a validation run
        if (epoch + 1) % args.val_freq == 0 or epoch == args.epochs - 1:
            yv = forward_pass_enc(Enc, xv, 1024)
            yt = forward_pass_enc(Enc, xt, 1024)

            logs['perm'] = loss_permutation(xt, yt, args, k=val_k, size=10**4)

            logs['train_top1'] = loss_top_1_in_lat_top_k(xt, xt, yt, yt, args, 2, val_k, size=10**5, name="TRAIN")
            logs['valid_top1'] = loss_top_1_in_lat_top_k(xv, xt, yv, yt, args, 1, val_k, size=10**5, name="VALID")

            yq = forward_pass_enc(Enc, xq, 1024)
            logs['query_top1_50'] = loss_top_1_in_lat_top_k(xq, xt, yq, yt, args, 1, val_k, size=10**4, name="QUERY_b")
            logs['query_top1_100'] = loss_top_1_in_lat_top_k(xq, xt, yq, yt, args, 1, 2*val_k, size=10**4, name="QUERY_b")

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


def train_vae(xt, xv, xq, args, results_file_name):

    sigma = 0.04
    mult = 1
    lambda_triplet = 1
    tr_pos, tr_neg, tr_margin = 10, 40, 0
    lr = 0.0001
    smart_weights = False
    val_k = 2 * args.dout
    dint = 1024

    print ("build network")
    dim = xt.shape[1]
    ldim = args.dout
    print(dim, dint, ldim)
    encoder = Encoder(dim, dint, ldim, mult, args.lat_type).to(args.device)
    decoder = Decoder(ldim, dint, dim, mult).to(args.device)
    weights = np.ones(xt.shape[0])

    if smart_weights:
        weights = get_weights(xt, 50, args)

    all_logs = vae_optimize(xt, xv, xq, encoder, decoder, args, lambda_triplet,\
                        sigma, weights, tr_pos, tr_neg, tr_margin, lr, val_k)

    if args.print_results > 0:
        with open(results_file_name, "a") as rfile:
            rfile.write("\n")
            rfile.write("\n")
            rfile.write("UAE, DATABASE %s, num_learn = %d, lat_dim = %d, k = 50, lam_tr = %.7f,  \n" %
                        (args.database, xt.shape[0], args.dout, lambda_triplet))
            log = all_logs[-1]
            rfile.write("tr_pos = %d, tr_neg = %d, sigma = %.7f, net_mult = %.3f, margin = %.3f, width = %d, epochs %d,  bs = %d, lr = %.4f  \n" %
                        ( tr_pos, tr_neg, sigma, mult, tr_margin, dint, log['epoch'] + 1, args.batch_size, lr))
            rfile.write("last perm = %.4f, train_top1 = %.3f, valid_top1 = %.3f, query_top1_50 = %.3f, query_top1_100 = %.3f \n" %
                        (log['perm'], log['train_top1'], log['valid_top1'],  log['query_top1_50'], log['query_top1_100']))

            rfile.write(" loss_uniform = %.6f, loss_recon = %.6f,loss_triplet = %.6f, loss = %.6f, offending = %d, times %f %f %f \n" %
                        (log['loss_uniform'], log['loss_recon'], log['loss_triplet'], log['loss'], log['offending'],
                         log['times'][0], log['times'][1], log['times'][2]))
            rfile.write("------------------------------------------------------ \n")