from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util import nest
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope as vs

import optimizers

class CTRNNCell(tf.nn.rnn_cell.RNNCell):
    """ API Conventions: https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/python/ops/rnn_cell_impl.py
    """
    def __init__(self, num_units, tau, activation=None):
        self._num_units = num_units
        self.tau = tau
        if activation is None:
            self.activation = lambda x: 1.7159 * tf.tanh(2/3*x)
            # from: LeCun et al. 2012: Efficient backprop
        else:
            self.activation = activation


    @property # Function is callable without (), as if it was a property...
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def zero_state(self, batch_size, dtype):
        """Return zero-filled state tensor(s).
        Args:
          batch_size: int, float, or unit Tensor representing the batch size.
          dtype: the data type to use for the state.
        Returns:
          If `state_size` is an int or TensorShape, then the return value is a
          `N-D` tensor of shape `[batch_size x state_size]` filled with zeros.
          If `state_size` is a nested list or tuple, then the return value is
          a nested list or tuple (of the same structure) of `2-D` tensors with
        the shapes `[batch_size x s]` for each s in `state_size`.
        """
        state_size = self.state_size
        if nest.is_sequence(state_size):
            state_size_flat = nest.flatten(state_size)
            zeros_flat = [
                array_ops.zeros(
                    array_ops.stack(_state_size_with_prefix(s, prefix=[batch_size])),
                    dtype=dtype)
                for s in state_size_flat]
            for s, z in zip(state_size_flat, zeros_flat):
                z.set_shape(_state_size_with_prefix(s, prefix=[None]))
            zeros = nest.pack_sequence_as(structure=state_size,
                                        flat_sequence=zeros_flat)
        else:
            zeros_size = _state_size_with_prefix(state_size, prefix=[batch_size])
            zeros = array_ops.zeros(array_ops.stack(zeros_size), dtype=dtype)
            zeros.set_shape(_state_size_with_prefix(state_size, prefix=[None]))

        return zeros


    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            old_c = state[0]
            old_u = state[1]

            with tf.variable_scope('linear'):
                logits = self._linear(inputs + [old_c], output_size=self.output_size, bias=False)

            with tf.variable_scope('applyTau'):
                new_u = (1-1/self.tau)*old_u + 1/self.tau*logits

            new_c = self.activation(new_u)

        return new_c, (new_c, new_u)

    def _linear(self, args, output_size, bias, bias_start=0.0):
        """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.
        Args:
            args: a 2D Tensor or a list of 2D, batch x n, Tensors.
            output_size: int, second dimension of W[i].
            bias: boolean, whether to add a bias term or not.
            bias_start: starting value to initialize the bias; 0 by default.
        Returns:
            A 2D Tensor with shape [batch x output_size] equal to
            sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
        Raises:
            ValueError: if some of the arguments has unspecified or wrong shape.
        """
        if args is None or (nest.is_sequence(args) and not args):
            raise ValueError("`args` must be specified")
        if not nest.is_sequence(args):
            args = [args]

        # Calculate the total size of arguments on dimension 1.
        total_arg_size = 0
        shapes = [a.get_shape() for a in args]
        for shape in shapes:
            if shape.ndims != 2:
                raise ValueError("linear is expecting 2D arguments: %s" % shapes)
            if shape[1].value is None:
                raise ValueError("linear expects shape[1] to be provided for shape %s, "
                                 "but saw %s" % (shape, shape[1]))
            else:
                total_arg_size += shape[1].value

        dtype = [a.dtype for a in args][0]

        # Now the computation.
        scope = vs.get_variable_scope()
        with vs.variable_scope(scope) as outer_scope:
            weights = vs.get_variable(
                'weights', [total_arg_size, output_size], dtype=dtype)
            if len(args) == 1:
                res = math_ops.matmul(args[0], weights)
            else:
                res = math_ops.matmul(array_ops.concat(args, 1), weights)
            if not bias:
                return res
            with vs.variable_scope(outer_scope) as inner_scope:
                inner_scope.set_partitioner(None)
                biases = vs.get_variable(
                    'biases', [output_size],
                    dtype=dtype,
                    initializer=init_ops.constant_initializer(bias_start, dtype=dtype))
            return nn_ops.bias_add(res, biases)

def shape_printer(obj, prefix):
    try:
        print(prefix, obj.shape)
    except AttributeError:
        print(prefix, type(obj))
        for o in obj:
            shape_printer(o, prefix + '\t')


class MultiLayerHandler():
    def __init__(self, layers):
        """ layers: A list of layers """
        self.layers = layers
        self.num_layers = len(layers)

    @property # Function is callable without (), as if it was a property...
    def state_size(self):
        raise NotImplementedError
        # num_units = []
        # for l in self.layers:
        #     num_units += l.state_size
        # return num_units

    @property
    def output_size(self):
        raise NotImplementedError
        # return self.layers[0]._num_units

    def zero_state(self, batch_size, dtype):
        """Return zero-filled state tensor(s).
        Args:
          batch_size: int, float, or unit Tensor representing the batch size.
          dtype: the data type to use for the state.
        Returns:
          If `state_size` is an int or TensorShape, then the return value is a
          `N-D` tensor of shape `[batch_size x state_size]` filled with zeros.
          If `state_size` is a nested list or tuple, then the return value is
          a nested list or tuple (of the same structure) of `2-D` tensors with
        the shapes `[batch_size x s]` for each s in `state_size`.
        """
        raise NotImplementedError
        # """ Returns a zero filled tuple with shapes equivalent to (new_c, new_u)"""
        # zero_states = []
        # for l in self.layers:
        #     zero_states += l.zero_state(batch_size)
        # return zero_states

    def __call__(self, input_IO, state, scope=None, reverse = True):

        with tf.variable_scope(scope or type(self).__name__):
            out_state = []
            outputs = [[],[]]
            if reverse:
                for i_, l in enumerate(reversed(self.layers)): # Start with the top level
                    i = self.num_layers - i_ - 1
                    scope = 'CTRNNCell_' + str(i)

                    cur_state = state[i]
                    if i == 0: # IO level, last executed
                        cur_input = [input_IO[1]] + [state[i+1][0]] 
                        outputs1, state_ = l(cur_input, cur_state, scope=scope)
                    elif i == self.num_layers - 1: # Highest level
                        cur_input = [input_IO[0]] + [state[i-1][0]]
                        outputs3, state_ = l(cur_input, cur_state, scope=scope)
                    else: # Inbetween layers
                        cur_input = [state[i-1][0]] + [state[i+1][0]]
                        outputs2, state_ = l(cur_input, cur_state, scope=scope)
                    out_state += [state_]
                out_state = tuple(reversed(out_state))
            else:
                for i_, l in enumerate(self.layers): # Start with the top level
                    i = i_ 
                    scope = 'CTRNNCell_' + str(i)

                    cur_state = state[i]
                    if i == 0: # IO level, last executed
                        cur_input = [input_IO[1]] + [state[i+1][0]] 
                        outputs1, state_ = l(cur_input, cur_state, scope=scope)
                    elif i == self.num_layers - 1: # Highest level
                        cur_input = [input_IO[0]] + [state[i-1][0]]
                        outputs3, state_ = l(cur_input, cur_state, scope=scope)
                    else: # Inbetween layers
                        cur_input = [state[i-1][0]] + [state[i+1][0]]
                        outputs2, state_ = l(cur_input, cur_state, scope=scope)
                    out_state += [state_]
                out_state = tuple(out_state)
            outputs = [outputs3, outputs1]
            shape_printer(out_state, 'MLH')
            return outputs, out_state

        # with tf.variable_scope(scope or type(self).__name__):
        #     for i, l in enumerate(self.layers):
        #         scope = 'CTRNNCell_' + str(i)
        #         inputs, state = l([inputs], state, scope=scope)
        # return inputs, state


class CTRNNModel(object):
    def __init__(self, num_units, tau, num_steps, input_dim, output_dim, output_dim2, learning_rate=1e-4):
        #with tf.device('/cpu:0'):
        """ Assumptions
            * x is 3 dimensional: [batch_size, num_steps] 
            Args:
            * num_units: list with num_units, with num_units[0] being the IO layer
            * taus: list with tau values (also if it is only one element!)
        """
        self.num_units = num_units
        self.num_layers = len(self.num_units)
        self.tau = tau

        self.input_dim = input_dim
        self.output_dim = output_dim 
        self.activation = lambda x: 1.7159 * tf.tanh(2/3 * x)
        #self.activation = lambda x: tf.sigmoid(x)


        self.cs = tf.placeholder(tf.float32, shape=[None, num_steps, input_dim], name='csPlaceholder')
        self.cs_reshaped = tf.reshape(tf.transpose(self.cs, [1,0,2]), [-1])

        self.x = tf.placeholder(tf.float32, shape=[None, num_steps, self.output_dim], name='inputPlaceholder')
        self.x_reshaped = tf.reshape(tf.transpose(self.x, [1,0,2]), [-1])
        self.y = tf.placeholder(tf.float32, shape=[None, num_steps, output_dim], name='outputPlaceholder')
        self.y_reshaped = tf.reshape(tf.transpose(self.y, [1, 0, 2]), [-1, output_dim])
        #self.y_reshaped = tf.reshape(self.y, [-1])

        self.final_seq = tf.placeholder(tf.float32, shape=[None, output_dim2], name='finalSequence')

        self.direction = tf.placeholder(tf.bool, shape=())
        # True means generating a sentence from cs
        # false means generating a cs from sentence


        init_input_sentence = tf.placeholder(tf.float32, shape=[None, self.num_units[0]], name='initInputSent')
        init_input_cs = tf.placeholder(tf.float32, shape=[None, self.num_units[2]], name='initInputCS')
        init_input = [init_input_cs, init_input_sentence]
        init_state = []
        for i, num_unit in enumerate(self.num_units):
            init_c = tf.placeholder(tf.float32, shape=[None, num_unit], name='initC_' + str(i))
            init_u = tf.placeholder(tf.float32, shape=[None, num_unit], name='initU_' + str(i))
            init_state += [(init_c, init_u)]
        init_state = tuple(init_state)
        self.init_tuple = (init_input, init_state)

        W = tf.get_variable('W', [num_units[0], num_units[0]], tf.float32)

        cells = []
        for i in range(self.num_layers): 
            num_unit = num_units[i]
            tau = self.tau[i]
            if i == 0:
                cells += [CTRNNCell(num_unit, tau=tau, activation=lambda x:tf.matmul(x, W))]
            else:
                cells += [CTRNNCell(num_unit, tau=tau, activation=self.activation)]
        self.cell = MultiLayerHandler(cells) # First cell (index 0) is IO layer


        with tf.variable_scope("scan", reuse = tf.AUTO_REUSE):
            self.rnn_outputs, self.final_states = tf.cond(self.direction, 
        lambda: tf.scan(lambda state, x: self.cell(x, state[1], reverse=True), [tf.transpose(self.cs, [1, 0, 2]), tf.transpose(self.x, [1, 0, 2])], initializer=self.init_tuple), 
        lambda: tf.scan(lambda state, x: self.cell(x, state[1], reverse=False),[tf.transpose(self.cs, [1, 0, 2]), tf.transpose(self.x, [1, 0, 2])],initializer=self.init_tuple))


        state_state = []
        for i in range(self.num_layers):
            state_state += [(self.final_states[i][0][-1], self.final_states[i][1][-1])]
        state_state = tuple(state_state)
        self.state_tuple = (self.rnn_outputs[-1], state_state)


        rnn_outputs_sentence = self.rnn_outputs[1]
        rnn_outputs_sentence = tf.cast(tf.reshape(rnn_outputs_sentence, [-1, num_units[0]]), tf.float32)
        rnn_outputs_sentence = tf.slice(rnn_outputs_sentence, [0, 0], [-1, output_dim])
        rnn_outputs_cs = self.rnn_outputs[0][num_steps-1] #we want the final step only
        rnn_outputs_cs = tf.slice(rnn_outputs_cs, [0, 0], [-1, output_dim2])


        # FOR MSE SEE BELOW
        #####################################
        self.logits_sequence = rnn_outputs_sentence
        self.total_loss_sequence = tf.reduce_sum(tf.square(tf.subtract(self.y_reshaped, self.logits_sequence)))
        tf.summary.scalar('training/total_loss', self.total_loss_sequence)
        #############################################

        self.logits_cs = rnn_outputs_cs
        self.total_loss_cs = tf.reduce_sum(tf.square(tf.subtract(self.final_seq, self.logits_cs)))
        tf.summary.scalar('training/total_loss', self.total_loss_cs)
##########################################################################


        self.total_loss = tf.cond(self.direction, lambda: tf.reduce_sum(tf.square(tf.subtract(self.y_reshaped, self.logits_sequence))), lambda: tf.reduce_sum(tf.square(tf.subtract(self.final_seq, self.logits_cs))))


        self.train_op = optimizers.AMSGrad(learning_rate).minimize(self.total_loss)
        self.TBsummaries = tf.summary.merge_all()


        config = tf.ConfigProto(device_count = {'CPU': 12,'GPU': 0}, allow_soft_placement = True, log_device_placement = False)
        config.gpu_options.per_process_gpu_memory_fraction = 0.3
        config.operation_timeout_in_ms = 50000

        self.saver = tf.train.Saver(max_to_keep=1)
        self.sess = tf.Session(config = config)

    def forward_step_test(self):
        #Inputs_x_t = tf.constant(Inputs_x)
        #Inputs_sentence_t = tf.constant(Inputs_sentence)
        self.Inputs_x_t = tf.placeholder(tf.float32, shape = [1, self.input_dim], name = 'CS_input')
        self.Inputs_sentence_t = tf.placeholder(tf.float32, shape = [1, self.output_dim], name = 'sentence_input')
        Inputs_t = [self.Inputs_x_t, self.Inputs_sentence_t]
        self.direction = tf.placeholder(tf.bool, shape=())

        with tf.variable_scope("test", reuse = tf.AUTO_REUSE):
            init_state = []
            for i, num_unit in enumerate(self.num_units):
                init_c = tf.placeholder(tf.float32, shape=[None, num_unit], name='initC_' + str(i))
                init_u = tf.placeholder(tf.float32, shape=[None, num_unit], name='initU_' + str(i))
                init_state += [(init_c, init_u)]
            State = tuple(init_state)

        with tf.variable_scope("scan", reuse = tf.AUTO_REUSE):
            self.outputs, self.new_state = tf.cond(self.direction, lambda: self.cell(Inputs_t, State, reverse = True), lambda: self.cell(Inputs_t, State, reverse = False))


    def kullback_leibler(self, x, y):
        return tf.reduce_sum(y*tf.log((y+0.000000001)/(x+0.000000001)))

    def zero_state_tuple(self, batch_size):
        """ Returns a tuple og zeros
        """
        zero_input = np.zeros([batch_size, self.num_units[0]], dtype = np.float32)

        zero_state = []
        for i, num_unit in enumerate(self.num_units):
            zero_c = np.zeros([batch_size, self.num_units[i]], dtype = np.float32)
            zero_u = np.zeros([batch_size, self.num_units[i]], dtype = np.float32)
            zero_state += [(zero_c, zero_u)]

        zero_state = tuple(zero_state)
        return (zero_input, zero_state)

    def get_weights(self):
        return [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if v.name.endswith('kernel:0')]
