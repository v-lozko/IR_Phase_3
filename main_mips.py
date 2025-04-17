"""

updated main_mips.py to accommodate for additional distance metrics. This includes
cosine and Euclidean distances.

"""

import numpy as np
import h5py
from absl import app, flags
import time
from tabulate import tabulate
from clustering import kmeans, k_random, linearlearner, auxiliary
from tqdm import tqdm


# names of the algorithms
AlgorithmRandom = 'random'
AlgorithmKMeans = 'kmeans'
AlgorithmSphericalKmeans = 'kmeans-spherical'
AlgorithmLinearLearner = 'linear-learner'

# name of the dataset and embedding
flags.DEFINE_string('name_dataset', None, 'Name of the dataset.')
flags.DEFINE_string('name_embedding', None, 'Name of the embedding.')

# decide the file format to import
flags.DEFINE_string('format_file', None, 'hdf5 - for the hdf5 file; npy - for the npy files.')

# dataset for hdf5
flags.DEFINE_string('dataset', None, 'Path to the dataset in hdf5 format.')

flags.DEFINE_string('documents_key', 'documents', 'Dataset key for document vectors.')
flags.DEFINE_string('train_queries_key', 'train_queries', 'Dataset key for train queries.')
flags.DEFINE_string('valid_queries_key', 'valid_queries', 'Dataset key for validation queries.')
flags.DEFINE_string('test_queries_key', 'test_queries', 'Dataset key for test queries.')
flags.DEFINE_string('train_neighbors_key', 'train_neighbors', 'Dataset key for train neighbors.')
flags.DEFINE_string('valid_neighbors_key', 'valid_neighbors', 'Dataset key for validation neighbors.')
flags.DEFINE_string('test_neighbors_key', 'test_neighbors', 'Dataset key for test neighbors.')

# docs, queries and neighbors for npy
flags.DEFINE_string('dataset_docs', None, 'Path to the dataset-docs in npy format.')
flags.DEFINE_string('dataset_queries', None, 'Path to the dataset-queries in npy format.')
flags.DEFINE_string('dataset_neighbors', None, 'Path to the dataset-neighbors in npy format.')

# setting environment
flags.DEFINE_float('test_split_percent', 20, 'Percentage of data points in the test set.')
flags.DEFINE_integer('split_seed', 42, 'Seed used when forming train-test splits.')

# linear-learner
flags.DEFINE_integer('learner_nunits', 0, 'Number of hidden units used by the linear-learner, with 0 we drop'
                                             'the hidden layer.')
flags.DEFINE_integer('learner_nepochs', 100, 'Number of epochs used by the linear-learner.')

# algorithm method
flags.DEFINE_enum('algorithm', AlgorithmKMeans,
                  [AlgorithmRandom,
                   AlgorithmKMeans,
                   AlgorithmSphericalKmeans,
                   AlgorithmLinearLearner],
                  'Indexing algorithm.')

flags.DEFINE_integer('nclusters', 1000, 'When `algorithm` is KMeans-based: Number of clusters.')

#define distance metric
flags.DEFINE_enum('distance_metric', 'dot', ['dot', 'cosine', 'euclidean'], 'Distance metric between queries and centroids')

# multi-probing, set the probes
flags.DEFINE_list('ells', [1],
                  'Minimum number of documents to examine.')

# top-k docs
flags.DEFINE_integer('top_k', 1, 'Top-k documents to retrieve per query.')

# flag to skip the clustering algorithm if already computed
flags.DEFINE_integer('compute_clusters', 0, '0 - perform clustering algorithm; '
                                            '1 - take the results already computed.')

FLAGS = flags.FLAGS


def get_final_results(name_method, centroids, x_test, y_test, top_k, clusters_top_k_test=None, gpu_flag=True, model = None):
    """
    Computes the final results, where we have the accuracy of given centroids.

    :param name_method: Name of the method that generated the centroids under consideration.
    :param centroids: The centroids.
    :param x_test, y_test: The test set.
    :param top_k: The number of top documents.
    :param clusters_top_k_test: The clusters where the top documents for each query are located.
    :param gpu_flag: Flag that indicates whether to run the code using the GPU (True) or not (False).
    """
    import os
    os.makedirs('./ells_stat_sig', exist_ok=True)
    os.makedirs('./results', exist_ok=True)

    # compute the score for each query and centroid
    print(name_method, end=' ')
    print('- run prediction with centroids...', end=' ')
    pred = auxiliary.scores_queries_centroids(centroids, x_test, gpu_flag=gpu_flag, model = model)
    print('end, shape: ', pred.shape)

    # save scores computed
    print('Saving results: score for each query and centroid.')
    np.save('./ells_stat_sig/' + name_method + '_' + FLAGS.name_dataset + '_' + FLAGS.name_embedding + '_' +
            FLAGS.distance_metric + '_' + FLAGS.algorithm + '_ells_stat_sig.npy', pred)

    # computation of the final scores
    results_ells = []
    if top_k > 1:
        raise NotImplementedError("top k > 1 is not supported in this version.")
    else:
        for threshold in tqdm(FLAGS.ells):
            k = int(threshold)
            one_pred = auxiliary.computation_top_k_clusters(k, FLAGS.nclusters, pred)

            # DEBUG: Rank of the true cluster
            query_idx = 0  # Pick a query to inspect
            true_cluster = np.where(y_test[query_idx] == 1)[0][0]
            scores_for_query = pred[query_idx]  # shape: [n_clusters]

            if FLAGS.distance_metric == 'euclidean':
                sorted_clusters = np.argsort(scores_for_query)  # lower = better
            else:
                sorted_clusters = np.argsort(-scores_for_query)  # higher = better

            rank = np.where(sorted_clusters == true_cluster)[0][0]

            print(f"[DEBUG] ell={k}, Query {query_idx}")
            print(f"True cluster: {true_cluster}")
            print(f"Predicted top-10: {sorted_clusters[:10]}")
            print(f"True cluster rank: {rank}")
            print("-----")

            res = auxiliary.evaluate_ell_top_one(one_pred, y_test)
            results_ells.append(res)
            print('k = {0}: {1}'.format(k, res))

    # print the final results
    table = ([['n_k', 'acc']] + [[FLAGS.ells[i_c], results_ells[i_c]] for i_c in range(len(FLAGS.ells))])
    print(tabulate(table, headers='firstrow', tablefmt='psql'))

    # save the results
    file_result = open('./results/' + name_method + '_' + FLAGS.name_dataset + '_' + FLAGS.name_embedding + '_' +
                       FLAGS.distance_metric + '_' + FLAGS.algorithm + str(top_k) + '_results.txt', 'w')
    file_result.write(tabulate(table, headers='firstrow', tablefmt='psql'))
    file_result.close()
    print('Results saved.')


def main(_):
    """
    Main function of our algorithm, where all methods to obtain the final results are invoked.
    """
    start = time.time()

    documents = None
    queries = None
    neighbors = None

    # hdf5 format file
    if FLAGS.format_file == 'hdf5':
        raise NotImplementedError("HDF5 input format is not supported in this version. Please use --format_file=npy.")

    # npy format file
    elif FLAGS.format_file == 'npy':
        documents = np.load(FLAGS.dataset_docs)
        queries = np.load(FLAGS.dataset_queries)
        neighbors = np.load(FLAGS.dataset_neighbors)

        assert len(queries) == len(neighbors)

    # run the clustering algorithm or import the clusters already computed
    print('Running the clustering algorithm or importing the clusters already computed.')

    centroids = None
    label_clustering = None

    # compute centroids and labels
    if FLAGS.compute_clusters == 1:

        # (standard or spherical) kmeans algorithm
        if FLAGS.algorithm in [AlgorithmKMeans, AlgorithmSphericalKmeans]:
            spherical = FLAGS.algorithm == AlgorithmSphericalKmeans
            centroids, label_clustering = kmeans.k_means(doc_vectors=documents,
                                                         n_clusters=FLAGS.nclusters,
                                                         flag_spherical=spherical)
            print(f'Obtained centroids with shape: {centroids.shape}')

        # shallow kmeans algorithm
        elif FLAGS.algorithm == AlgorithmRandom:
            centroids, label_clustering = k_random.random_clustering(doc_vectors=documents,
                                                                     n_clusters=FLAGS.nclusters)
            print(f'Obtained centroids with shape: {centroids.shape}')

        # save centroids and label_clustering
        centroids_file = (FLAGS.name_dataset + '_' + FLAGS.name_embedding + '_' + FLAGS.algorithm + '_centroids.npy')
        label_clustering_file = (FLAGS.name_dataset + '_' + FLAGS.name_embedding + '_' + FLAGS.algorithm + '_label_clustering.npy')

        print('Saving clusters got.')
        np.save(centroids_file, centroids)
        np.save(label_clustering_file, label_clustering)

    # load centroids and labels
    else:
        print("DEBUG:", FLAGS.name_dataset, FLAGS.name_embedding, FLAGS.algorithm)

        centroids_file = (FLAGS.name_dataset + '_' + FLAGS.name_embedding + '_' + FLAGS.algorithm + '_centroids.npy')
        label_clustering_file = (FLAGS.name_dataset + '_' + FLAGS.name_embedding + '_' + FLAGS.algorithm + '_label_clustering.npy')

        centroids = np.load(centroids_file)
        label_clustering = np.load(label_clustering_file)

    # data preparation
    print('Data preparation.')
    x_train = None
    y_train = None
    x_val = None
    y_val = None
    x_test = None
    y_test = None
    clusters_top_k_test = None

    if FLAGS.format_file == 'hdf5':
        raise NotImplementedError("HDF5 input format is not supported in this version. Please use --format_file=npy.")

    elif FLAGS.format_file == 'npy':
        if FLAGS.distance_metric in ['dot', 'cosine']:
            label_data = auxiliary.query_true_label(FLAGS.nclusters, label_clustering, neighbors)
            partitioning = auxiliary.train_test_val(queries, label_data, size_split=FLAGS.test_split_percent/100)
        else:
            label_data = auxiliary.query_true_label(FLAGS.nclusters, label_clustering, neighbors, one_hot=False)
            partitioning = auxiliary.train_test_val(queries, label_data, size_split=FLAGS.test_split_percent / 100)
        x_train = partitioning[0]
        y_train = partitioning[1]
        x_val = partitioning[2]
        y_val = partitioning[3]
        x_test = partitioning[4]
        y_test = partitioning[5]

    np.save(FLAGS.name_dataset + '_' + FLAGS.name_embedding + '_' + FLAGS.distance_metric + '_'
            + FLAGS.algorithm + '_y_test.npy', y_test)

    # training linear-learner
    print('Linear Learner.')
    if FLAGS.distance_metric == 'euclidean':
        print("Unique y_train values:", np.unique(y_train))
        print("Max valid cluster index:", np.max(label_clustering))
        learner_model, new_centroids = linearlearner.run_euclidean_learner(
            x_train, y_train, x_val, y_val, centroids,
            n_epochs=FLAGS.learner_nepochs)
    else:
        new_centroids = linearlearner.run_linear_learner(x_train=x_train, y_train=y_train,
                                                         x_val=x_val, y_val=y_val,
                                                        train_queries=queries,
                                                        n_clusters=FLAGS.nclusters,
                                                        n_epochs=FLAGS.learner_nepochs,
                                                        n_units=FLAGS.learner_nunits)

    print(f'Obtained centroids with shape: {new_centroids.shape}')

    if FLAGS.distance_metric == 'euclidean':
        y_test = auxiliary.query_true_label(
            FLAGS.nclusters, label_clustering, partitioning[5], one_hot=True
        )

    # results: baseline
    if FLAGS.distance_metric in ['dot', 'cosine']:
        # results: baseline
        get_final_results('baseline', centroids, x_test, y_test, FLAGS.top_k, clusters_top_k_test, gpu_flag=True)

        # results: linear-learner
        get_final_results('linearlearner', centroids, x_test, y_test, FLAGS.top_k, clusters_top_k_test, gpu_flag=True)
    else:
        # results: baseline
        get_final_results('baseline', centroids, x_test, y_test, FLAGS.top_k, clusters_top_k_test, gpu_flag=True)

        # results: linear-learner
        get_final_results('linearlearner', centroids, x_test, y_test, FLAGS.top_k, clusters_top_k_test,
                          gpu_flag=True, model = learner_model)

    end = time.time()
    print(f'Done in {end - start} seconds.')


if __name__ == '__main__':
    app.run(main)
