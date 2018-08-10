import numpy as np
import networkx as nx
from pathlib import Path

from io_utils.embedding import path_to_embedding, read_embedding, save_embedding

from keras.layers import Lambda, merge
import keras.regularizers as Reg
from keras.optimizers import SGD
from keras import backend as KBack

from time import time

from keras.layers import Input, Dense
from keras.models import Model, model_from_json


def model_batch_predictor(model, X, batch_size):
    n_samples = X.shape[0]
    counter = 0
    pred = None
    while counter < n_samples // batch_size:
        _, curr_pred = \
            model.predict(X[batch_size * counter:batch_size * (counter + 1),
                          :].toarray())
        if counter:
            pred = np.vstack((pred, curr_pred))
        else:
            pred = curr_pred
        counter += 1
    if n_samples % batch_size != 0:
        _, curr_pred = \
            model.predict(X[batch_size * counter:, :].toarray())
        if counter:
            pred = np.vstack((pred, curr_pred))
        else:
            pred = curr_pred
    return pred


def batch_generator_sdne(X, beta, batch_size, shuffle):
    row_indices, col_indices = X.nonzero()
    sample_index = np.arange(row_indices.shape[0])
    number_of_batches = row_indices.shape[0] // batch_size
    counter = 0
    if shuffle:
        np.random.shuffle(sample_index)
    while True:
        batch_index = \
            sample_index[batch_size * counter:batch_size * (counter + 1)]
        X_batch_v_i = X[row_indices[batch_index], :].toarray()
        X_batch_v_j = X[col_indices[batch_index], :].toarray()
        InData = np.append(X_batch_v_i, X_batch_v_j, axis=1)

        B_i = np.ones(X_batch_v_i.shape)
        B_i[X_batch_v_i != 0] = beta
        B_j = np.ones(X_batch_v_j.shape)
        B_j[X_batch_v_j != 0] = beta
        X_ij = X[row_indices[batch_index], col_indices[batch_index]]
        deg_i = np.sum(X_batch_v_i != 0, 1).reshape((batch_size, 1))
        deg_j = np.sum(X_batch_v_j != 0, 1).reshape((batch_size, 1))
        a1 = np.append(B_i, deg_i, axis=1)
        a2 = np.append(B_j, deg_j, axis=1)
        OutData = [a1, a2, X_ij.T]
        counter += 1
        yield InData, OutData
        if (counter == number_of_batches):
            if shuffle:
                np.random.shuffle(sample_index)
            counter = 0


def get_encoder(node_num, d, K, n_units, nu1, nu2, activation_fn):
    # Input
    x = Input(shape=(node_num,))
    # Encoder layers
    y = [None] * (K + 1)
    y[0] = x  # y[0] is assigned the input
    for i in range(K - 1):
        y[i + 1] = Dense(n_units[i], activation=activation_fn,
                         W_regularizer=Reg.l1_l2(l1=nu1, l2=nu2))(y[i])
    y[K] = Dense(d, activation=activation_fn,
                 W_regularizer=Reg.l1_l2(l1=nu1, l2=nu2))(y[K - 1])
    # Encoder model
    encoder = Model(input=x, output=y[K])
    return encoder


def get_decoder(node_num, d, K,
                n_units, nu1, nu2,
                activation_fn):
    # Input
    y = Input(shape=(d,))
    # Decoder layers
    y_hat = [None] * (K + 1)
    y_hat[K] = y
    for i in range(K - 1, 0, -1):
        y_hat[i] = Dense(n_units[i - 1],
                         activation=activation_fn,
                         W_regularizer=Reg.l1_l2(l1=nu1, l2=nu2))(y_hat[i + 1])
    y_hat[0] = Dense(node_num, activation=activation_fn,
                     W_regularizer=Reg.l1_l2(l1=nu1, l2=nu2))(y_hat[1])
    # Output
    x_hat = y_hat[0]  # decoder's output is also the actual output
    # Decoder Model
    decoder = Model(input=y, output=x_hat)
    return decoder


def get_autoencoder(encoder, decoder):
    # Input
    x = Input(shape=(encoder.layers[0].input_shape[1],))
    # Generate embedding
    y = encoder(x)
    # Generate reconstruction
    x_hat = decoder(y)
    # Autoencoder Model
    autoencoder = Model(input=x, output=[x_hat, y])
    return autoencoder


def graphify(reconstruction):
    [n1, n2] = reconstruction.shape
    n = min(n1, n2)
    reconstruction = np.copy(reconstruction[0:n, 0:n])
    reconstruction = (reconstruction + reconstruction.T) / 2
    reconstruction -= np.diag(np.diag(reconstruction))
    return reconstruction


def loadmodel(filename):
    try:
        model = model_from_json(open(filename).read())
    except:
        print('Error reading file: {0}. Cannot load previous model'.format(filename))
        exit()
    return model


def loadweights(model, filename):
    try:
        model.load_weights(filename)
    except:
        print('Error reading file: {0}. Cannot load previous weights'.format(filename))
        exit()


def savemodel(model, filename):
    json_string = model.to_json()
    open(filename, 'w').write(json_string)


def saveweights(model, filename):
    model.save_weights(filename, overwrite=True)


class SdneTransformer:

    def __init__(self, graph,
                 graph_name,
                 dimension,
                 path_to_dumps,
                 *hyper_dict, use_cached=True, **kwargs):
        """ Initialize the SDNE class
        Args:
            d: dimension of the embedding
            beta: penalty parameter in matrix B of 2nd order objective
            alpha: weighing hyperparameter for 1st order objective
            nu1: L1-reg hyperparameter
            nu2: L2-reg hyperparameter
            K: number of hidden layers in encoder/decoder
            n_units: vector of length K-1 containing #units in hidden layers
                     of encoder/decoder, not including the units in the
                     embedding layer
            rho: bounding ratio for number of units in consecutive layers (< 1)
            n_iter: number of sgd iterations for first embedding (const)
            xeta: sgd step size parameter
            n_batch: minibatch size for SGD
            modelfile: Files containing previous encoder and decoder models
            weightfile: Files containing previous encoder and decoder weights
        """
        self.graph = graph
        self.graph_name = graph_name
        self.dim = dimension
        self.path_to_dumps = path_to_dumps
        self.use_cached = use_cached
        hyper_params = {
            'method_name': 'sdne',
            'actfn': 'relu',
            'modelfile': None,
            'weightfile': None,
            'savefilesuffix': None

        }
        hyper_params.update(kwargs)
        for key in hyper_params.keys():
            self.__setattr__('_%s' % key, hyper_params[key])
        for dictionary in hyper_dict:
            for key in dictionary:
                self.__setattr__('_%s' % key, dictionary[key])

    def fit(self):
        path = path_to_embedding(
            root=self.path_to_dumps,
            method='sdne',
            name=self.graph_name,
            dim=self.dim
        )
        if self.use_cached:
            if Path(path).exists():
                E = read_embedding(path)
                print("Loaded cached embedding from " + path)
                return E

        E = self.learn_embedding()
        save_embedding(
            path,
            E=np.array(E)
        )

    def get_method_name(self):
        return self._method_name

    def get_method_summary(self):
        return '%s_%d' % (self._method_name, self._d)

    def learn_embedding(self):
        graph = self.graph
        if not graph:
            raise Exception('graph is None')
        S = nx.to_scipy_sparse_matrix(graph)
        t1 = time()
        S = (S + S.T) / 2
        self._node_num = graph.number_of_nodes()

        # Generate encoder, decoder and autoencoder
        self._num_iter = self._n_iter
        # If cannot use previous step information, initialize new models
        self._encoder = get_encoder(self._node_num, self._d,
                                    self._K, self._n_units,
                                    self._nu1, self._nu2,
                                    self._actfn)
        self._decoder = get_decoder(self._node_num, self._d,
                                    self._K, self._n_units,
                                    self._nu1, self._nu2,
                                    self._actfn)
        self._autoencoder = get_autoencoder(self._encoder, self._decoder)

        # Initialize self._model
        # Input
        x_in = Input(shape=(2 * self._node_num,), name='x_in')
        x1 = Lambda(
            lambda x: x[:, 0:self._node_num],
            output_shape=(self._node_num,)
        )(x_in)
        x2 = Lambda(
            lambda x: x[:, self._node_num:2 * self._node_num],
            output_shape=(self._node_num,)
        )(x_in)
        # Process inputs
        [x_hat1, y1] = self._autoencoder(x1)
        [x_hat2, y2] = self._autoencoder(x2)
        # Outputs
        x_diff1 = merge([x_hat1, x1],
                        mode=lambda ab: ab[0] - ab[1],
                        output_shape=lambda L: L[1])
        x_diff2 = merge([x_hat2, x2],
                        mode=lambda ab: ab[0] - ab[1],
                        output_shape=lambda L: L[1])
        y_diff = merge([y2, y1],
                       mode=lambda ab: ab[0] - ab[1],
                       output_shape=lambda L: L[1])

        # Objectives
        def weighted_mse_x(y_true, y_pred):
            """ Hack: This fn doesn't accept additional arguments.
                      We use y_true to pass them.
                y_pred: Contains x_hat - x
                y_true: Contains [b, deg]
            """
            return KBack.sum(
                KBack.square(y_pred * y_true[:, 0:self._node_num]),
                axis=-1) / y_true[:, self._node_num]

        def weighted_mse_y(y_true, y_pred):
            """ Hack: This fn doesn't accept additional arguments.
                      We use y_true to pass them.
            y_pred: Contains y2 - y1
            y_true: Contains s12
            """
            min_batch_size = KBack.shape(y_true)[0]
            return KBack.reshape(
                KBack.sum(KBack.square(y_pred), axis=-1),
                [min_batch_size, 1]
            ) * y_true

        # Model
        self._model = Model(input=x_in, output=[x_diff1, x_diff2, y_diff])
        sgd = SGD(lr=self._xeta, decay=1e-5, momentum=0.99, nesterov=True)
        # adam = Adam(lr=self._xeta, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        self._model.compile(
            optimizer=sgd,
            loss=[weighted_mse_x, weighted_mse_x, weighted_mse_y],
            loss_weights=[1, 1, self._alpha]
        )

        self._model.fit_generator(
            generator=batch_generator_sdne(S, self._beta, self._n_batch, True),
            nb_epoch=self._num_iter,
            samples_per_epoch=S.nonzero()[0].shape[0] // self._n_batch,
            verbose=1
        )
        # Get embedding for all points
        self._Y = model_batch_predictor(self._autoencoder, S, self._n_batch)
        t2 = time()
        # Save the autoencoder and its weights
        if self._weightfile is not None:
            saveweights(self._encoder, self._weightfile[0])
            saveweights(self._decoder, self._weightfile[1])
        if self._modelfile is not None:
            savemodel(self._encoder, self._modelfile[0])
            savemodel(self._decoder, self._modelfile[1])
        if self._savefilesuffix is not None:
            saveweights(
                self._encoder,
                'encoder_weights_' + self._savefilesuffix + '.hdf5'
            )
            saveweights(
                self._decoder,
                'decoder_weights_' + self._savefilesuffix + '.hdf5'
            )
            savemodel(
                self._encoder,
                'encoder_model_' + self._savefilesuffix + '.json'
            )
            savemodel(
                self._decoder,
                'decoder_model_' + self._savefilesuffix + '.json'
            )
            # Save the embedding
            np.savetxt('embedding_' + self._savefilesuffix + '.txt', self._Y)
        print("time: {}".format(t2-t1))
        return self._Y

    def get_embedding(self, filesuffix=None):
        return self._Y if filesuffix is None else np.loadtxt(
            'embedding_' + filesuffix + '.txt'
        )

    def get_edge_weight(self, i, j, embed=None, filesuffix=None):
        if embed is None:
            if filesuffix is None:
                embed = self._Y
            else:
                embed = np.loadtxt('embedding_' + filesuffix + '.txt')
        if i == j:
            return 0
        else:
            S_hat = self.get_reconst_from_embed(embed[(i, j), :], filesuffix)
            return (S_hat[i, j] + S_hat[j, i]) / 2

    def get_reconstructed_adj(self, embed=None, node_l=None, filesuffix=None):
        if embed is None:
            if filesuffix is None:
                embed = self._Y
            else:
                embed = np.loadtxt('embedding_' + filesuffix + '.txt')
        S_hat = self.get_reconst_from_embed(embed, node_l, filesuffix)
        return graphify(S_hat)

    def get_reconst_from_embed(self, embed, node_l=None, filesuffix=None):
        if filesuffix is None:
            if node_l is not None:
                return self._decoder.predict(
                    embed,
                    batch_size=self._n_batch)[:, node_l]
            else:
                return self._decoder.predict(embed, batch_size=self._n_batch)
        else:
            try:
                decoder = model_from_json(
                    open('decoder_model_' + filesuffix + '.json').read()
                )
            except:
                print('Error reading file: {0}. Cannot load previous model'.format('decoder_model_'+filesuffix+'.json'))
                exit()
            try:
                decoder.load_weights('decoder_weights_' + filesuffix + '.hdf5')
            except:
                print('Error reading file: {0}. Cannot load previous weights'.format('decoder_weights_'+filesuffix+'.hdf5'))
                exit()
            if node_l is not None:
                return decoder.predict(embed, batch_size=self._n_batch)[:, node_l]
            else:
                return decoder.predict(embed, batch_size=self._n_batch)


if __name__ == '__main__':
    # load Zachary's Karate graph
    edge_f = 'data/karate.edgelist'
    G = None
    res_pre = 'results/testKarate'
    t1 = time()
    embedding = SdneTransformer(d=2, beta=5, alpha=1e-5, nu1=1e-6, nu2=1e-6, K=3,
                                n_units=[50, 15], rho=0.3, n_iter=50, xeta=0.01,
                                n_batch=500,
                                modelfile=['./intermediate/enc_model.json',
                                './intermediate/dec_model.json'],
                                weightfile=['./intermediate/enc_weights.hdf5',
                                 './intermediate/dec_weights.hdf5'])
    embedding.learn_embedding(graph=G)
    print('SDNE:\n\tTraining time: %f' % (time() - t1))

