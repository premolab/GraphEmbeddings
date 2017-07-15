import lasagne
import theano.tensor as T


class DotLayer(lasagne.layers.MergeLayer):
    def __init__(self, incoming1, incoming2, **kwargs):
        super(DotLayer, self).__init__([incoming1, incoming2], **kwargs)
        # optional: check here that self.input_shapes make sense for a dot product
        # self.input_shapes will be populated by the super() call above

    def get_output_shape_for(self, input_shapes):
        # (rows of first input x columns of second input)
        return self.input_shapes[0][0], self.input_shapes[1][0]

    def get_output_for(self, inputs, **kwargs):
        return T.dot(inputs[0], inputs[1].T)
