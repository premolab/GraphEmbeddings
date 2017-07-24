import pickle
import time

import lasagne
import networkx as nx
import numpy as np
import theano
from theano import tensor as T


class Graph2vec:
    def __init__(self, graph, d,
                 fit=True, stoc=True,
                 save_intern_embeddings_step=0,
                 save_intern_name_prefix=None,
                 verbose=1):
        self.d = d
        self.stoc = stoc
        self.max_norm = 10
        self.min_cov = 1e-6
        self.verbose = verbose
        self.last_outputtime = 0
        self.save_intern_embeddings_step = save_intern_embeddings_step
        self.save_intern_name_prefix = save_intern_name_prefix \
            if save_intern_name_prefix is not None \
            else './emb/interns/temp2_histloss_'

        if isinstance(graph, nx.Graph):
            self.graph = graph
            self.n = len(graph)
            self.nodes = graph.nodes()
            self.adjacency_matrix = nx.adjacency_matrix(graph, self.nodes).todense().astype("float32")
        else:
            self.n = graph.shape[0]
            self.nodes = list(range(self.n))
            self.adjacency_matrix = graph

        np.fill_diagonal(self.adjacency_matrix, 0)

        self.embedding = np.zeros((self.n, self.d))

        self.scale = T.scalar('scale')
        self.input_var = T.matrix('input_var')  # mini_batch_size to n
        self.adj_minibatch = T.matrix('adj_minibatch')  # mini_batch_size to mini_batch_size

        self.network = self.init_network()

        self.EPOCH_DEBUG_LINE = 'End Epoch {}/{}, ' + \
                                'cost: {:.5f}, ' + \
                                'last grad norm: {:.5f} ' + \
                                'step: {:.6f} took {:.3f} sec, ' + \
                                'max_norm: {:.3f}, ' + \
                                'bucket: {}'
        self.ITER_DEBUG_LINE = '\tIter {}, ' + \
                               'cost: {:.5f}, ' + \
                               'max_norm: {:.3f} took {:.3f} sec'

        if fit:
            res = self.optimize()
            self.embedding = res

    def init_network(self):
        l_in = lasagne.layers.InputLayer(shape=(None, self.n), input_var=self.input_var)
        # Apply 20% dropout to the input data on training:
        l_in_drop = lasagne.layers.DropoutLayer(l_in, p=0.2)
        network = lasagne.layers.DenseLayer(incoming=l_in_drop,
                                            num_units=self.d,
                                            nonlinearity=lambda x: x / self.scale,
                                            W=lasagne.init.Normal())
        return network

    def init_loss(self, deterministic):
        raise NotImplemented()

    def update_step(self, lr, last_loss, train_loss, bucket, volume):
        raise NotImplemented()

    def get_minibatch(self, i, indxs):
        return indxs, self.adjacency_matrix[indxs], self.adjacency_matrix[indxs][:, indxs], i

    def minibatches(self, max_size=200):
        for i in np.random.permutation(self.n):
            if self.stoc:
                mask = self.adjacency_matrix[i, :]
                res = np.random.permutation(np.where(mask)[1]).tolist()
                yield_arr = [
                    self.get_minibatch(i, res[k:min(len(res), k + max_size - 1)] + [i])
                    for k in range(0, len(res), max_size - 1)
                ]
                for elem in yield_arr:
                    yield elem
            else:
                yield self.get_minibatch(i, np.arange(self.n))

    def save(self, name, embedding=None):
        pickle.dump((self.nodes, self.embedding if embedding is None else embedding), open(name, 'wb'))

    @staticmethod
    def load(name):
        return pickle.load(open(name, 'rb'))

    def optimize(self, step=0.1, num_epochs=300):
        mode = 'FAST_RUN'  #

        # preparing ...
        self.loss = self.init_loss(deterministic=False)
        self.determ_loss = self.init_loss(deterministic=True)

        params = lasagne.layers.get_all_params(self.network, trainable=True)

        learning_rate = T.scalar('learning_rate')
        if self.verbose != 0:
            print('Compiling...')
        get_grad_norm = theano.function(inputs=[self.input_var, self.adj_minibatch, self.scale],
                                        outputs=T.grad(self.determ_loss, self.input_var).norm(2),
                                        mode=mode)
        get_emb = theano.function(inputs=[self.input_var, self.scale],
                                  outputs=lasagne.layers.get_output(self.network, deterministic=True),
                                  mode=mode,
                                  )
        # updates = lasagne.updates.nesterov_momentum(self.loss, params, learning_rate=learning_rate, momentum=0.8)
        updates = lasagne.updates.adagrad(self.loss, params, learning_rate=learning_rate)
        train_fn = theano.function(inputs=[self.input_var, self.adj_minibatch, learning_rate, self.scale],
                                   outputs=self.loss,
                                   updates=updates,
                                   mode=mode)
        # pickle.dump((loss, get_grad_norm, get_emb, updates, train_fn), open('./dumps/hist-theano', 'wb'))
        # (loss, get_grad_norm, get_emb, updates, train_fn) = pickle.load(open('./dumps/hist-theano', 'rb'))

        # optimization
        if self.verbose != 0:
            print('Done!\nOptimization...')
        iters = 0
        lr = 0.005
        last_loss = float('inf')
        best_loss = last_loss
        bucket = 0
        volume = 20 if self.stoc else 10

        self.max_norm = np.maximum(self.max_norm, 2 * np.max(np.linalg.norm(self.embedding, axis=1)))
        grad_norm = 0.
        for epoch in range(num_epochs):
            if self.verbose != 0:
                print('epoch: {}'.format(epoch))
            # In each epoch, we do a full pass over the training data:
            train_loss = 0

            train_batches = 0
            start_time = time.time()
            minibatches = self.minibatches()
            for batch in minibatches:
                nodes_idxs, input_var, adj, ego_node_indx = batch

                train_loss += train_fn(input_var, adj, lr, self.max_norm)
                train_batches += 1
                iters += 1
                if iters % 100 == 1:
                    self.debug_print(
                        False, iters, train_loss / train_batches, self.max_norm, time.time() - start_time
                    )
                    embedding = get_emb(self.adjacency_matrix, self.max_norm)
                    self.max_norm = np.maximum(
                        self.max_norm, 2 * np.max(np.linalg.norm(embedding, axis=1))
                    )

                if self.save_intern_embeddings_step != 0 and \
                        (iters % self.save_intern_embeddings_step == 1 or
                                 iters <= self.save_intern_embeddings_step * 2):
                    embedding = get_emb(self.adjacency_matrix, self.max_norm)
                    self.save(self.save_intern_name_prefix + str(iters).zfill(6) + '.tmp_emb', embedding)

            grad_norm = 0.8 * grad_norm + 0.2 * get_grad_norm(input_var, adj, self.max_norm)

            self.debug_print(True, epoch + 1, num_epochs, train_loss / train_batches, grad_norm, lr,
                        time.time() - start_time, self.max_norm, bucket)

            embedding = get_emb(self.adjacency_matrix, self.max_norm)

            lr = self.update_step(lr, last_loss, train_loss, bucket, volume)

            if train_loss > best_loss - 0.005 * train_batches:
                bucket += 1
            else:
                bucket = 0
                best_loss = train_loss
                self.embedding = embedding

            last_loss = train_loss

            if bucket > volume:
                break
            if grad_norm < 1e-3:
                break

        self.embedding = get_emb(self.adjacency_matrix, self.max_norm)

    def debug_print(self, epoch, *args):
        if self.verbose == 0:
            return

        if epoch:
            print(self.EPOCH_DEBUG_LINE.format(*args))
        else:
            print(self.ITER_DEBUG_LINE.format(*args))
