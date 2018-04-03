from datetime import datetime

import tensorflow as tf
import networkx as nx

NAME_SEPARATOR = '@'


class TfGraphLoss:
    def __init__(self):
        pass

    def loss(self, A, E):
        raise NotImplementedError

    def merged_summary(self):
        raise NotImplementedError

    def inference(self, A):
        raise NotImplementedError


class TfHistLoss(TfGraphLoss):
    def __init__(self, W, b, simmatrix_method="ID"):
        super().__init__()
        self.simmatrix_method = simmatrix_method
        self.W = W
        self.b = b

    def inference(self, A):
        E = tf.matmul(A, self.W) + self.b
        E_norm = E / tf.reshape(tf.norm(E, axis=1), (tf.shape(E)[0], 1))
        return E_norm

    def calc_loss(self, pos_hist, neg_hist):
        with tf.name_scope('calc_loss') as scope:
            return tf.reduce_sum(tf.abs(tf.cumsum(neg_hist - pos_hist)))

    def calc_hist(self, samples, bin_num=64):
        with tf.name_scope('calc_hist') as scope:
            delta = 2 / (bin_num - 1)
            grid_row = tf.range(-1, 1 + delta, delta, dtype=tf.float32)
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

    def calc_pos_samples(self, S, E_corr):
        with tf.name_scope('calc_pos_samples') as scope:
            pos_mask = self.get_pos_mask(S)
            return tf.boolean_mask(E_corr, mask=pos_mask)

    def calc_neg_samples(self, S, E_corr):
        with tf.name_scope('calc_neg_samples') as scope:
            neg_mask = self.get_neg_mask(S)
            return tf.boolean_mask(E_corr, mask=neg_mask)

    def get_pos_mask(self, S):
        with tf.name_scope('get_pos_mask') as scope:
            return S > 0

    def get_neg_mask(self, S):
        with tf.name_scope('get_neg_mask') as scope:
            return tf.logical_and(
                tf.logical_not(self.get_pos_mask(S)),
                tf.logical_not(tf.eye(tf.shape(self.get_pos_mask(S))[0], dtype=tf.bool))
            )

    def calc_simmatrix(self, A, method='normal'):
        with tf.name_scope('calc_simmatrix') as scope:
            if method == 'ID':
                return A
            elif method == 'ADA':
                d = tf.reduce_sum(A, axis=1)
                D = tf.diag(d)
                return tf.matmul(tf.matrix_inverse(D), tf.matmul(A, A))

    def loss(self, A, E):
        S = self.calc_simmatrix(A, method=self.simmatrix_method)

        with tf.name_scope('calc_corr') as scope:
            E_norm = E / tf.reshape(tf.norm(E, axis=1), (tf.shape(E)[0], 1))
            E_corr = tf.matmul(E_norm, E_norm, transpose_b=True)

        tf_neg_samples = self.calc_neg_samples(S, E_corr)
        tf_pos_samples = self.calc_pos_samples(S, E_corr)

        tf.summary.histogram("pos_samples", tf_pos_samples)
        tf.summary.histogram("neg_samples", tf_neg_samples)

        tf_pos_hist = self.calc_hist(tf_pos_samples)
        tf_neg_hist = self.calc_hist(tf_neg_samples)

        loss = - self.calc_loss(tf_neg_hist, tf_pos_hist)

        return loss

    def merged_summary(self):
        return tf.summary.merge_all()

class Graph:
    def __init__(self, G: nx.Graph, name=""):
        self.G = G


class Embedding:
    def __init__(self, data):
        self.data = data

    @property
    def dim(self):
        return self.data.shape[1]


class Transformer:
    def __init__(self, name):
        self.name = name

    def run(self, graph: nx.Graph, dim):
        result = self.transform(graph, dim)
        self.save(result)

    def transform(self, graph: nx.Graph, dim):
        raise NotImplementedError

    def save(self, result: Embedding):
        raise NotImplementedError


class TfOptimizerTransformer(Transformer):
    def __init__(self, name, loss: TfGraphLoss):
        super().__init__("TfOptimizerTransformer" + NAME_SEPARATOR + name)
        self.loss = loss

    def transform(self, graph: nx.Graph, dim):
        tf.reset_default_graph()
        N = graph.number_of_nodes()
        simmatrix_method = 'ID'

        A = tf.placeholder(tf.float32, [N, N], name="A")

        W = tf.Variable(tf.random_uniform([N, dim], -1.0, 1.0), name="W")
        b = tf.Variable(tf.zeros([N, dim]), name="b")

        loss = TfHistLoss(W, b, simmatrix_method=simmatrix_method)

        tf.summary.scalar("loss", loss)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)

        merged = loss.merged_summary()

        A = nx.adj_matrix(G).todense()
        with tf.Session() as sess:
            # variables need to be initialized before we can use them
            sess.run(tf.global_variables_initializer())

            writer = tf.summary.FileWriter('../tensorflow_events/' + str(datetime.utcnow()), graph=sess.graph)

            for epoch in range(1000):

                l, _, summary, result = sess.run(
                    [loss, optimizer, merged, loss.inference(A)],
                    feed_dict={A: nx.adj_matrix(graph).todense()}
                )

                writer.add_summary(summary, epoch)

                if epoch % 200 == 0:
                    print("Epoch: ", epoch)
                    print(l)
            print("done")
            save_embedding(
                path_to_embedding(
                    method='hist_loss_emd_ID',
                    name='SBM_sizes_100_100_100_p_in_0.1_p_out_0.01_seed_43',
                    dim=dim
                ), E=E_norm
            )

    def save(self, result):
        pass

