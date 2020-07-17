import time

from support_func import loss_permutation, loss_top_1_in_lat_top_k
import umap


def train_umap(xt, xv, xq, args, results_file_name):
    nl = 10**5
    neig_num = 15
    m_dist = 0.5
    spread = 1 # must be less than m_dist

    t0 = time.time()

    k = 50
    reducer = umap.UMAP(n_components=args.dout, n_epochs=args.epochs, n_neighbors=neig_num,
                        min_dist=m_dist, spread=spread, metric='euclidean', verbose=True)

    t1 = time.time()
    yt = reducer.fit_transform(xt)
    print("fit")

    t1 = time.time()
    yv = reducer.transform(xv)
    yq = reducer.transform(xq)

    t2 = time.time()
    log = {}

    log['perm'] = loss_permutation(xt, yt, args, k=k, size=10 ** 4)

    log['train_top1'] = loss_top_1_in_lat_top_k(xt, xt, yt, yt, args, 2, k, size=10 ** 5, name="TRAIN")
    log['valid_top1'] = loss_top_1_in_lat_top_k(xv, xt, yv, yt, args, 1, k, size=10 ** 5, name="VALID")

    log['query_top1_50'] = loss_top_1_in_lat_top_k(xq, xt, yq, yt, args, 1, k, size=10 ** 4, name="QUERY_b")
    log['query_top1_100'] = loss_top_1_in_lat_top_k(xq, xt, yq, yt, args, 1, 2 * k, size=10 ** 4, name="QUERY_b")

    t3 = time.time()

    print("last perm = %.4f, train_top1 = %.3f, valid_top1 = %.3f, query_top1_50 = %.3f, query_top1_100 = %.3f \n" %
          (log['perm'], log['train_top1'], log['valid_top1'], log['query_top1_50'], log['query_top1_100']))
    print('times: [hn %.2f s epoch %.2f s val %.2f s]' % (t1 - t0, t2 - t1, t3 - t2))

    if args.print_results > 0:
        with open(results_file_name, "a") as rfile:
            rfile.write("\n")
            rfile.write("\n")
            rfile.write("UMAP, DATABASE %s, num_learn = %d, lat_dim = %d, k = 50, m_dist = %.7f, neig_num = %d, epoch = %d, spread = %d  \n" %
                        (args.database, nl, args.dout, m_dist, neig_num, args.epochs, spread))
            # rfile.write("\n")
            rfile.write(
                "last perm = %.4f, train_top1 = %.3f, valid_top1 = %.3f, query_top1_50 = %.3f, query_top1_100 = %.3f \n" %
                (log['perm'], log['train_top1'], log['valid_top1'], log['query_top1_50'],
                 log['query_top1_100']))

            rfile.write("times %f %f %f \n" % (t1 - t0, t2 - t1, t3 - t2))
