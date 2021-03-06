import os
from datetime import datetime
from pathlib import Path

import tensorflow as tf
import numpy as np
import networkx as nx

from scipy.special import expit

from io_utils.embedding import path_to_embedding, save_embedding, read_embedding
from transformation.HistLossConfiguration import HistLossConfiguration


class HistLossTransformer:

    def __init__(self,
                 graph,
                 graph_name,
                 dimension,
                 seed,
                 hist_loss_configuration: HistLossConfiguration,
                 path_to_dumps,
                 use_cached=True,
                 should_stop=None):
        self.graph = graph
        self.graph_name = graph_name
        self.dim = dimension
        self.seed = seed
        self.hist_loss_configuration = hist_loss_configuration
        self.path_to_dumps = path_to_dumps
        self.use_cached = use_cached
        self.should_stop = should_stop

    def fit(self):
        path = path_to_embedding(
            root=self.path_to_dumps,
            method='hist_loss_' + str(self.hist_loss_configuration),
            name=self.graph_name,
            dim=self.dim
        )
        if self.use_cached:
            if Path(path).exists():
                E = read_embedding(path)
                print("Loaded cached embedding from " + path)
                return E
        if self.hist_loss_configuration.linearity == 'linear':
            E = self.run(
                should_stop=self.should_stop
            )
        elif self.hist_loss_configuration.linearity == 'direct':
            E = self.run_direct(
                should_stop=self.should_stop
            )
        elif self.hist_loss_configuration.linearity == 'nonlinear2':
            E = self.run_nonlinear2(
                should_stop=self.should_stop
            )
        elif self.hist_loss_configuration.linearity == 'nonlinear2-reduce':
            E = self.run_nonlinear2_reduce(
                should_stop=self.should_stop
            )
        elif self.hist_loss_configuration.linearity == 'nonlinear3':
            E = self.run_nonlinear3(
                should_stop=self.should_stop
            )
        else:
            raise Exception('Unknown linearity: ' + self.hist_loss_configuration.linearity)
        save_embedding(
            path,
            E=np.array(E),
            normalize=True
        )

    @staticmethod
    def preprocess_embedding(E, method='normalization'):
        with tf.name_scope('preprocess_embedding'):
            if method == 'normalization':
                E_norm = E / tf.reshape(tf.norm(E, axis=1), (tf.shape(E)[0], 1))
                E_corr = tf.matmul(E_norm, E_norm, transpose_b=True)
                return E_corr
            else:
                raise Exception('Unknown method "' + method + '"')

    @staticmethod
    def calc_loss(pos_hist, neg_hist, method):
        with tf.name_scope('calc_loss') as scope:
            if method == 'SIM':
                return tf.reduce_sum(tf.abs(tf.cumsum(neg_hist - pos_hist)))
            elif method == 'ASIM':
                return - tf.reduce_sum(tf.cumsum(neg_hist - pos_hist))
            else:
                raise Exception('Unknown method "' + method + '"')

    @staticmethod
    def calc_hist(samples, bin_num=64, method='NORMAL'):
        with tf.name_scope('calc_hist') as scope:
            delta = 2 / (bin_num - 1)
            grid_row = tf.range(-1, 1 + delta, delta, dtype=tf.float32)
            if method == 'NORMAL':
                grid = tf.reshape(tf.tile(grid_row, [tf.shape(samples)[0]]), (tf.shape(samples)[0], grid_row.shape[0]))
                samples_grid = tf.transpose(
                    tf.reshape(tf.tile(samples, [grid_row.shape[0]]), (grid_row.shape[0], tf.shape(samples)[0])))
                dif = tf.abs(samples_grid - grid)
                mask = dif < delta
                t = tf.diag_part(
                    tf.matmul(
                        tf.to_float(mask),
                        delta - dif,
                        transpose_a=True
                    ) / (
                            delta * tf.to_float(tf.shape(samples)[0])
                    )
                )
                return t
            elif method == 'TF-KDE':
                # something is wrong now
                samples_shape = tf.shape(samples)

                f = lambda x: tf.distributions.Normal(loc=x, scale=delta / 2)

                kde = tf.contrib.distributions.MixtureSameFamily(
                    mixture_distribution=tf.contrib.distributions.Categorical(
                        probs=tf.fill(samples_shape, 1 / tf.to_float(samples_shape[0]))
                    ),
                    components_distribution=f(samples)
                )

                return kde.prob(grid_row) * delta
            raise Exception('Unknown method "' + method + '"')

    @staticmethod
    def get_pos_mask(S):
        with tf.name_scope('get_pos_mask') as scope:
            return S > 0

    @staticmethod
    def get_neg_mask(S):
        with tf.name_scope('get_neg_mask') as scope:
            return tf.logical_and(
                tf.logical_not(HistLossTransformer.get_pos_mask(S)),
                tf.logical_not(tf.eye(tf.shape(S)[0], dtype=tf.bool))
            )

    @staticmethod
    def calc_pos_samples(S, E_corr, method='NORMAL', *, multiplicator=10, threshold=2):
        with tf.name_scope('calc_pos_samples') as scope:
            if method == 'NORMAL':
                pos_mask = HistLossTransformer.get_pos_mask(S)
                samples = tf.boolean_mask(E_corr, mask=pos_mask)
                tf.summary.histogram("pos_samples", samples)
                return samples
            elif method == 'WEIGHTED':
                v = tf.reshape(E_corr, (-1,))
                w = tf.reshape(tf.cast(S * multiplicator, 'int32'), (-1,))
                w = tf.where(w < threshold, tf.zeros_like(w), w)
                m = tf.sequence_mask(w)
                v2 = tf.tile(v[:, None], [1, tf.shape(m)[1]])
                samples = tf.boolean_mask(v2, m)
                tf.summary.histogram("pos_samples", samples)
                return samples
            else:
                raise Exception('Unknown method "' + method + '"')

    @staticmethod
    def calc_neg_samples(S, E_corr, method='NORMAL', *, multiplicator=10, threshold=2):
        with tf.name_scope('calc_neg_samples') as scope:
            if method == 'NORMAL':
                neg_mask = HistLossTransformer.get_neg_mask(S)
                samples = tf.boolean_mask(E_corr, mask=neg_mask)
                tf.summary.histogram("neg_samples", samples)
                return samples
            elif method == 'IGNORE-NEG':
                neg_mask = HistLossTransformer.get_neg_mask(S)
                samples = tf.boolean_mask(E_corr, mask=neg_mask)
                tf.summary.histogram("neg_samples", samples)
                return tf.where(samples < 0, tf.zeros_like(samples), samples)
            elif method == 'WEIGHTED':
                v = tf.reshape(E_corr, (-1,))
                w = tf.reshape(tf.cast(S * multiplicator, 'int32'), (-1,))
                w = tf.where(w > threshold, tf.zeros_like(w), w)
                w = threshold - w
                m = tf.sequence_mask(w)
                v2 = tf.tile(v[:, None], [1, tf.shape(m)[1]])
                samples = tf.boolean_mask(v2, m)
                tf.summary.histogram("neg_samples", samples)
                return samples
            else:
                raise Exception('Unknown method "' + method + '"')

    @staticmethod
    def calc_simmatrix(A, method='NORMAL'):
        with tf.name_scope('calc_simmatrix') as scope:
            if method == 'ID':
                return A
            elif method == 'ADA':
                d = tf.reduce_sum(A, axis=1)
                D = tf.diag(d)
                return tf.matmul(tf.matrix_inverse(D), tf.matmul(A, A))
            raise Exception('Unknown method "' + method + '"')

    @staticmethod
    def np_calc_simmatrix(A, method='NORMAL'):
        if method == 'ID':
            return A
        elif method == 'ADA':
            D = np.zeros(A.shape)
            for i in range(A.shape[0]):
                D[i, i] = np.sum(A[i])

            S = np.dot(np.linalg.inv(D), np.dot(A, A))
            return S
        raise Exception('Unknown method "' + method + '"')

    def run(self,
            patience=100,
            patience_delta=0.001,
            learning_rate=0.1,
            LOG_DIR='./tensorflow_events/',
            should_stop=None,
            save_intermediate=False
            ):

        tf.reset_default_graph()
        bin_num = 64
        N = self.graph.number_of_nodes()

        batch_size = min(800, N)
        print("Batch size:", str(batch_size))

        _A_batched = tf.placeholder(tf.float32, [batch_size, N])
        _batch_indxs = tf.placeholder(tf.int32, [batch_size])
        _neg_sampling_indxs = tf.placeholder(tf.int32, [None])
        _W = tf.Variable(tf.random_uniform([N, self.dim], -1.0, 1.0), name="W")
        _b = tf.Variable(tf.zeros([N, self.dim]), name="b")
        _b_batched = tf.gather(_b, indices=_batch_indxs)

        _E = tf.matmul(_A_batched, _W) + _b_batched
        _E_corr = self.preprocess_embedding(_E)
        _A_batched_square = tf.gather(_A_batched, _batch_indxs, axis=1)
        _neg_samples = self.calc_neg_samples(
            _A_batched_square,
            _E_corr,
            method=self.hist_loss_configuration.calc_neg_method
        )
        _pos_samples = self.calc_pos_samples(
            _A_batched_square,
            _E_corr,
            method=self.hist_loss_configuration.calc_pos_method
        )

        if N <= 1000:
            _neg_samples = tf.gather(_neg_samples, _neg_sampling_indxs)

        _pos_hist = self.calc_hist(_pos_samples, method=self.hist_loss_configuration.calc_hist_method, bin_num=bin_num)
        _neg_hist = self.calc_hist(_neg_samples, method=self.hist_loss_configuration.calc_hist_method, bin_num=bin_num)
        _loss = - self.calc_loss(_neg_hist, _pos_hist, method=self.hist_loss_configuration.loss_method)

        tf.summary.scalar("loss", _loss)
        _optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(_loss)
        _summary = tf.summary.merge_all()

        G = self.graph
        A = nx.adj_matrix(G).todense()
        A = self.np_calc_simmatrix(A, method=self.hist_loss_configuration.simmatrix_method)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            writer = tf.summary.FileWriter(
                os.path.join(LOG_DIR, '{}_{}_{}_{}_{}').format(
                    str(datetime.now().timestamp()),
                    self.dim,
                    self.hist_loss_configuration,
                    self.graph_name,
                    batch_size
                ),
                graph=sess.graph
            )

            prev_loss = 0
            patience_counter = 0

            for epoch in range(1000):
                if epoch % 50 == 1:
                    print('epoch: ' + str(epoch) + ', loss: ' + str(loss))
                    if save_intermediate:
                        E = np.dot(A, W) + b
                        save_embedding('E_' + str(epoch) + '.txt', np.array(E))
                batch_indxs = np.random.choice(a=N, size=batch_size).astype('int32')
                A_batched = A[batch_indxs]
                pos_count = np.count_nonzero(A_batched[:, batch_indxs])
                neg_count = batch_size * N - pos_count
                neg_sampling_indxs = np.random.choice(a=neg_count,
                                                      size=pos_count * 2).astype('int32')

                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                pos_hist, neg_hist, loss, _, summary, W, b = sess.run(
                    [
                        _pos_hist,
                        _neg_hist,
                        _loss,
                        _optimizer,
                        _summary,
                        _W,
                        _b
                    ],
                    feed_dict={
                        _A_batched: A_batched,
                        _batch_indxs: batch_indxs,
                        _neg_sampling_indxs: neg_sampling_indxs,
                    },
                    options=run_options,
                    run_metadata=run_metadata
                )
                writer.add_run_metadata(run_metadata, "step_{}".format(epoch))

                writer.add_summary(summary, epoch)

                if epoch > 0 and prev_loss - loss > patience_delta:
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter > patience:
                    print("\tearly stopping at", epoch)
                    break

                if should_stop:
                    if should_stop():
                        break

                prev_loss = loss
            E = np.dot(A, W) + b
            return E

    def run_direct(self,
            patience=100,
            patience_delta=0.01,
            learning_rate=0.1,
            LOG_DIR='./tensorflow_events/',
            should_stop=None,
            save_intermediate=False
            ):

        tf.reset_default_graph()
        bin_num = 64
        N = self.graph.number_of_nodes()

        batch_size = min(800, N)
        print("Batch size:", str(batch_size))

        _A_batched = tf.placeholder(tf.float32, [batch_size, N])
        _batch_indxs = tf.placeholder(tf.int32, [batch_size])
        _neg_sampling_indxs = tf.placeholder(tf.int32, [None])
        _E = tf.Variable(tf.random_uniform([N, self.dim], -1.0, 1.0), name="E")
        _E_batched = tf.gather(_E, _batch_indxs)

        _E_corr = self.preprocess_embedding(_E_batched)
        _A_batched_square = tf.gather(_A_batched, _batch_indxs, axis=1)
        _neg_samples = self.calc_neg_samples(
            _A_batched_square,
            _E_corr,
            method=self.hist_loss_configuration.calc_neg_method
        )
        _pos_samples = self.calc_pos_samples(
            _A_batched_square,
            _E_corr,
            method=self.hist_loss_configuration.calc_pos_method
        )

        if N <= 1000:
            _neg_samples = tf.gather(_neg_samples, _neg_sampling_indxs)

        _pos_hist = self.calc_hist(_pos_samples, method=self.hist_loss_configuration.calc_hist_method, bin_num=bin_num)
        _neg_hist = self.calc_hist(_neg_samples, method=self.hist_loss_configuration.calc_hist_method, bin_num=bin_num)
        _loss = - self.calc_loss(_neg_hist, _pos_hist, method=self.hist_loss_configuration.loss_method)

        tf.summary.scalar("loss", _loss)
        _optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(_loss)
        _summary = tf.summary.merge_all()

        G = self.graph
        A = nx.adj_matrix(G).todense()
        A = self.np_calc_simmatrix(A, method=self.hist_loss_configuration.simmatrix_method)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            writer = tf.summary.FileWriter(
                os.path.join(LOG_DIR, '{}_{}_{}_{}_{}').format(
                    str(datetime.now().timestamp()),
                    self.dim,
                    self.hist_loss_configuration,
                    self.graph_name,
                    batch_size
                ),
                graph=sess.graph
            )

            prev_loss = 0
            patience_counter = 0

            for epoch in range(10000):
                if epoch % 50 == 1:
                    print('epoch: ' + str(epoch) + ', loss: ' + str(loss))
                    if save_intermediate:
                        save_embedding('E_' + str(epoch) + '.txt', np.array(E))
                batch_indxs = np.random.choice(a=N, size=batch_size).astype('int32')
                A_batched = A[batch_indxs]
                pos_count = np.count_nonzero(A_batched[:, batch_indxs])
                neg_count = batch_size * N - pos_count
                neg_sampling_indxs = np.random.choice(a=neg_count,
                                                      size=pos_count * 2).astype('int32')

                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                pos_hist, neg_hist, loss, _, summary, E = sess.run(
                    [
                        _pos_hist,
                        _neg_hist,
                        _loss,
                        _optimizer,
                        _summary,
                        _E
                    ],
                    feed_dict={
                        _A_batched: A_batched,
                        _batch_indxs: batch_indxs,
                        _neg_sampling_indxs: neg_sampling_indxs,
                    },
                    options=run_options,
                    run_metadata=run_metadata
                )
                writer.add_run_metadata(run_metadata, "step_{}".format(epoch))

                writer.add_summary(summary, epoch)

                if epoch > 0 and prev_loss - loss > patience_delta:
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter > patience:
                    print("\tearly stopping at", epoch)
                    break

                if should_stop:
                    if should_stop():
                        break

                prev_loss = loss
            # E = np.dot(A, W) + b
            return E

    def run_nonlinear2(self,
                      patience=100,
                      patience_delta=0.001,
                      learning_rate=0.1,
                      LOG_DIR='./tensorflow_events/',
                      should_stop=None,
                      save_intermediate=False
                      ):

        tf.reset_default_graph()
        bin_num = 64
        N = self.graph.number_of_nodes()

        batch_size = min(800, N)
        print("Batch size:", str(batch_size))

        _A_batched = tf.placeholder(tf.float32, [batch_size, N])
        _batch_indxs = tf.placeholder(tf.int32, [batch_size])
        _neg_sampling_indxs = tf.placeholder(tf.int32, [None])
        _W_1 = tf.Variable(tf.random_uniform([N, N], -1.0, 1.0), name="W1")
        _b_1 = tf.Variable(tf.zeros([N, N]), name="b1")
        _W_2 = tf.Variable(tf.random_uniform([N, self.dim], -1.0, 1.0), name="W2")
        _b_2 = tf.Variable(tf.zeros([N, self.dim]), name="b2")
        _b_1_batched = tf.gather(_b_1, indices=_batch_indxs)
        _b_2_batched = tf.gather(_b_2, indices=_batch_indxs)

        _E = tf.matmul(
            tf.nn.sigmoid(
                tf.matmul(
                    _A_batched,
                    _W_1
                ) + _b_1_batched),
            _W_2
        ) + _b_2_batched
        _E_corr = self.preprocess_embedding(_E, 'normalization')
        _A_batched_square = tf.gather(_A_batched, _batch_indxs, axis=1)
        _neg_samples = self.calc_neg_samples(
            _A_batched_square,
            _E_corr,
            method=self.hist_loss_configuration.calc_neg_method
        )
        _pos_samples = self.calc_pos_samples(
            _A_batched_square,
            _E_corr,
            method=self.hist_loss_configuration.calc_pos_method
        )

        if N <= 1000:
            _neg_samples = tf.gather(_neg_samples, _neg_sampling_indxs)

        _pos_hist = self.calc_hist(_pos_samples, method=self.hist_loss_configuration.calc_hist_method, bin_num=bin_num)
        _neg_hist = self.calc_hist(_neg_samples, method=self.hist_loss_configuration.calc_hist_method, bin_num=bin_num)
        _loss = - self.calc_loss(_neg_hist, _pos_hist, method=self.hist_loss_configuration.loss_method)

        tf.summary.scalar("loss", _loss)
        _optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(_loss)
        _summary = tf.summary.merge_all()

        G = self.graph
        A = nx.adj_matrix(G).todense()
        A = self.np_calc_simmatrix(A, method=self.hist_loss_configuration.simmatrix_method)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            writer = tf.summary.FileWriter(
                os.path.join(LOG_DIR, '{}_{}_{}_{}_{}').format(
                    str(datetime.now().timestamp()),
                    self.dim,
                    self.hist_loss_configuration,
                    self.graph_name,
                    batch_size
                ),
                graph=sess.graph
            )

            prev_loss = 0
            patience_counter = 0

            for epoch in range(1000):
                if epoch % 50 == 1:
                    print('epoch: ' + str(epoch) + ', loss: ' + str(loss))
                    if save_intermediate:
                        E = np.dot(np.maximum(np.dot(A, W_1) + b_1, 0), W_2) + b_2
                        save_embedding('E_' + str(epoch) + '.txt', np.array(E))
                batch_indxs = np.random.choice(a=N, size=batch_size).astype('int32')
                A_batched = A[batch_indxs]
                pos_count = np.count_nonzero(A_batched[:, batch_indxs])
                neg_count = batch_size * N - pos_count
                neg_sampling_indxs = np.random.choice(a=neg_count,
                                                      size=pos_count * 2).astype('int32')

                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                pos_hist, neg_hist, loss, _, summary, W_1, b_1, W_2, b_2 = sess.run(
                    [
                        _pos_hist,
                        _neg_hist,
                        _loss,
                        _optimizer,
                        _summary,
                        _W_1,
                        _b_1,
                        _W_2,
                        _b_2,
                    ],
                    feed_dict={
                        _A_batched: A_batched,
                        _batch_indxs: batch_indxs,
                        _neg_sampling_indxs: neg_sampling_indxs,
                    },
                    options=run_options,
                    run_metadata=run_metadata
                )
                writer.add_run_metadata(run_metadata, "step_{}".format(epoch))

                writer.add_summary(summary, epoch)

                if epoch > 0 and prev_loss - loss > patience_delta:
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter > patience:
                    print("\tearly stopping at", epoch)
                    break

                if should_stop:
                    if should_stop():
                        break

                prev_loss = loss
            E = np.dot(expit(np.dot(A, W_1) + b_1), W_2) + b_2
            return E

    def run_nonlinear2_reduce(self,
                      patience=100,
                      patience_delta=0.001,
                      learning_rate=0.1,
                      LOG_DIR='./tensorflow_events/',
                      should_stop=None,
                      save_intermediate=False
                      ):

        tf.reset_default_graph()
        bin_num = 64
        N = self.graph.number_of_nodes()

        batch_size = min(800, N)
        print("Batch size:", str(batch_size))

        _A_batched = tf.placeholder(tf.float32, [batch_size, N])
        _batch_indxs = tf.placeholder(tf.int32, [batch_size])
        _neg_sampling_indxs = tf.placeholder(tf.int32, [None])
        _W_1 = tf.Variable(tf.random_uniform([N, N//3], -1.0, 1.0), name="W1")
        _b_1 = tf.Variable(tf.zeros([N, N//3]), name="b1")
        _W_2 = tf.Variable(tf.random_uniform([N//3, self.dim], -1.0, 1.0), name="W2")
        _b_2 = tf.Variable(tf.zeros([N, self.dim]), name="b2")
        _b_1_batched = tf.gather(_b_1, indices=_batch_indxs)
        _b_2_batched = tf.gather(_b_2, indices=_batch_indxs)

        _E = tf.matmul(
            tf.nn.sigmoid(
                tf.matmul(
                    _A_batched,
                    _W_1
                ) + _b_1_batched),
            _W_2
        ) + _b_2_batched
        _E_corr = self.preprocess_embedding(_E, 'normalization')
        _A_batched_square = tf.gather(_A_batched, _batch_indxs, axis=1)
        _neg_samples = self.calc_neg_samples(
            _A_batched_square,
            _E_corr,
            method=self.hist_loss_configuration.calc_neg_method
        )
        _pos_samples = self.calc_pos_samples(
            _A_batched_square,
            _E_corr,
            method=self.hist_loss_configuration.calc_pos_method
        )

        if N <= 1000:
            _neg_samples = tf.gather(_neg_samples, _neg_sampling_indxs)

        _pos_hist = self.calc_hist(_pos_samples, method=self.hist_loss_configuration.calc_hist_method, bin_num=bin_num)
        _neg_hist = self.calc_hist(_neg_samples, method=self.hist_loss_configuration.calc_hist_method, bin_num=bin_num)
        _loss = - self.calc_loss(_neg_hist, _pos_hist, method=self.hist_loss_configuration.loss_method)

        tf.summary.scalar("loss", _loss)
        _optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(_loss)
        _summary = tf.summary.merge_all()

        G = self.graph
        A = nx.adj_matrix(G).todense()
        A = self.np_calc_simmatrix(A, method=self.hist_loss_configuration.simmatrix_method)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            writer = tf.summary.FileWriter(
                os.path.join(LOG_DIR, '{}_{}_{}_{}_{}').format(
                    str(datetime.now().timestamp()),
                    self.dim,
                    self.hist_loss_configuration,
                    self.graph_name,
                    batch_size
                ),
                graph=sess.graph
            )

            prev_loss = 0
            patience_counter = 0

            for epoch in range(1000):
                if epoch % 50 == 1:
                    print('epoch: ' + str(epoch) + ', loss: ' + str(loss))
                    if save_intermediate:
                        E = np.dot(np.maximum(np.dot(A, W_1) + b_1, 0), W_2) + b_2
                        save_embedding('E_' + str(epoch) + '.txt', np.array(E))
                batch_indxs = np.random.choice(a=N, size=batch_size).astype('int32')
                A_batched = A[batch_indxs]
                pos_count = np.count_nonzero(A_batched[:, batch_indxs])
                neg_count = batch_size * N - pos_count
                neg_sampling_indxs = np.random.choice(a=neg_count,
                                                      size=pos_count * 2).astype('int32')

                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                pos_hist, neg_hist, loss, _, summary, W_1, b_1, W_2, b_2 = sess.run(
                    [
                        _pos_hist,
                        _neg_hist,
                        _loss,
                        _optimizer,
                        _summary,
                        _W_1,
                        _b_1,
                        _W_2,
                        _b_2,
                    ],
                    feed_dict={
                        _A_batched: A_batched,
                        _batch_indxs: batch_indxs,
                        _neg_sampling_indxs: neg_sampling_indxs,
                    },
                    options=run_options,
                    run_metadata=run_metadata
                )
                writer.add_run_metadata(run_metadata, "step_{}".format(epoch))

                writer.add_summary(summary, epoch)

                if epoch > 0 and prev_loss - loss > patience_delta:
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter > patience:
                    print("\tearly stopping at", epoch)
                    break

                if should_stop:
                    if should_stop():
                        break

                prev_loss = loss
            E = np.dot(expit(np.dot(A, W_1) + b_1), W_2) + b_2
            return E

    def run_nonlinear3(self,
                      patience=100,
                      patience_delta=0.001,
                      learning_rate=0.1,
                      LOG_DIR='./tensorflow_events/',
                      should_stop=None,
                      save_intermediate=False
                      ):

        tf.reset_default_graph()
        bin_num = 64
        N = self.graph.number_of_nodes()

        batch_size = min(800, N)
        print("Batch size:", str(batch_size))

        _A_batched = tf.placeholder(tf.float32, [batch_size, N])
        _batch_indxs = tf.placeholder(tf.int32, [batch_size])
        _neg_sampling_indxs = tf.placeholder(tf.int32, [None])
        _W_1 = tf.Variable(tf.random_uniform([N, N], -1.0, 1.0), name="W1")
        _b_1 = tf.Variable(tf.zeros([N, N]), name="b1")
        _W_2 = tf.Variable(tf.random_uniform([N, N], -1.0, 1.0), name="W2")
        _b_2 = tf.Variable(tf.zeros([N, N]), name="b2")
        _W_3 = tf.Variable(tf.random_uniform([N, self.dim], -1.0, 1.0), name="W3")
        _b_3 = tf.Variable(tf.zeros([N, self.dim]), name="b3")
        _b_1_batched = tf.gather(_b_1, indices=_batch_indxs)
        _b_2_batched = tf.gather(_b_2, indices=_batch_indxs)
        _b_3_batched = tf.gather(_b_3, indices=_batch_indxs)

        _E = tf.matmul(
            tf.nn.sigmoid(tf.matmul(
                tf.nn.sigmoid(
                    tf.matmul(
                        _A_batched,
                        _W_1
                    ) + _b_1_batched),
                _W_2
            ) + _b_2_batched),
            _W_3
        ) + _b_3_batched
        _E_corr = self.preprocess_embedding(_E)
        _A_batched_square = tf.gather(_A_batched, _batch_indxs, axis=1)
        _neg_samples = self.calc_neg_samples(
            _A_batched_square,
            _E_corr,
            method=self.hist_loss_configuration.calc_neg_method
        )
        _pos_samples = self.calc_pos_samples(
            _A_batched_square,
            _E_corr,
            method=self.hist_loss_configuration.calc_pos_method
        )

        if N <= 1000:
            _neg_samples = tf.gather(_neg_samples, _neg_sampling_indxs)

        _pos_hist = self.calc_hist(_pos_samples, method=self.hist_loss_configuration.calc_hist_method, bin_num=bin_num)
        _neg_hist = self.calc_hist(_neg_samples, method=self.hist_loss_configuration.calc_hist_method, bin_num=bin_num)
        _loss = - self.calc_loss(_neg_hist, _pos_hist, method=self.hist_loss_configuration.loss_method)

        tf.summary.scalar("loss", _loss)
        _optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(_loss)
        _summary = tf.summary.merge_all()

        G = self.graph
        A = nx.adj_matrix(G).todense()
        A = self.np_calc_simmatrix(A, method=self.hist_loss_configuration.simmatrix_method)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            writer = tf.summary.FileWriter(
                os.path.join(LOG_DIR, '{}_{}_{}_{}_{}').format(
                    str(datetime.now().timestamp()),
                    self.dim,
                    self.hist_loss_configuration,
                    self.graph_name,
                    batch_size
                ),
                graph=sess.graph
            )

            prev_loss = 0
            patience_counter = 0

            for epoch in range(1000):
                if epoch % 50 == 1:
                    print('epoch: ' + str(epoch) + ', loss: ' + str(loss))
                    if save_intermediate:
                        E = np.dot(np.maximum(np.dot(np.maximum(np.dot(A, W_1) + b_1, 0), W_2) + b_2, 0), W_3) + b_3
                        save_embedding('E_' + str(epoch) + '.txt', np.array(E))
                batch_indxs = np.random.choice(a=N, size=batch_size).astype('int32')
                A_batched = A[batch_indxs]
                pos_count = np.count_nonzero(A_batched[:, batch_indxs])
                neg_count = batch_size * N - pos_count
                neg_sampling_indxs = np.random.choice(a=neg_count,
                                                      size=pos_count * 2).astype('int32')

                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                pos_hist, neg_hist, loss, _, summary, W_1, b_1, W_2, b_2, W_3, b_3 = sess.run(
                    [
                        _pos_hist,
                        _neg_hist,
                        _loss,
                        _optimizer,
                        _summary,
                        _W_1,
                        _b_1,
                        _W_2,
                        _b_2,
                        _W_3,
                        _b_3,
                    ],
                    feed_dict={
                        _A_batched: A_batched,
                        _batch_indxs: batch_indxs,
                        _neg_sampling_indxs: neg_sampling_indxs,
                    },
                    options=run_options,
                    run_metadata=run_metadata
                )
                writer.add_run_metadata(run_metadata, "step_{}".format(epoch))

                writer.add_summary(summary, epoch)

                if epoch > 0 and prev_loss - loss > patience_delta:
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter > patience:
                    print("\tearly stopping at", epoch)
                    break

                if should_stop:
                    if should_stop():
                        break

                prev_loss = loss
            E = np.dot(expit(np.dot(expit(np.dot(A, W_1) + b_1), W_2) + b_2), W_3) + b_3
            return E


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
