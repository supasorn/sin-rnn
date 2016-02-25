import tensorflow as tf
from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import seq2seq
import svgwrite
import math

import numpy as np
import random

class Model():
  def __init__(self, args, infer=False):
    self.dim = 1
    self.args = args
    if infer:
      args.batch_size = 1
      args.seq_length = 1

    if args.model == 'rnn':
      cell_fn = rnn_cell.BasicRNNCell
    elif args.model == 'gru':
      cell_fn = rnn_cell.GRUCell
    elif args.model == 'lstm':
      cell_fn = rnn_cell.BasicLSTMCell
    else:
      raise Exception("model type not supported: {}".format(args.model))

    cell = cell_fn(args.rnn_size)

    cell = rnn_cell.MultiRNNCell([cell] * args.num_layers)

    if (infer == False and args.keep_prob < 1): # training mode
      cell = rnn_cell.DropoutWrapper(cell, output_keep_prob = args.keep_prob)

    self.cell = cell

    self.input_data = tf.placeholder(dtype=tf.float32, shape=[None, args.seq_length, self.dim])
    self.target_data = tf.placeholder(dtype=tf.float32, shape=[None, args.seq_length, self.dim])
    self.initial_state = cell.zero_state(batch_size=args.batch_size, dtype=tf.float32)

    self.num_mixture = args.num_mixture
    NOUT = self.num_mixture * (1 + 2 * self.dim) # prob + mu + sig
    # [prob 1-20, dim1 mu, dim1 sig, dim2,... ]

    with tf.variable_scope('rnnlm'):
      output_w = tf.get_variable("output_w", [args.rnn_size, NOUT])
      output_b = tf.get_variable("output_b", [NOUT])

    inputs = tf.split(1, args.seq_length, self.input_data)
    inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

    outputs, states = seq2seq.rnn_decoder(inputs, self.initial_state, cell, loop_function=None, scope='rnnlm')
    output = tf.reshape(tf.concat(1, outputs), [-1, args.rnn_size])
    output = tf.nn.xw_plus_b(output, output_w, output_b)
    self.final_state = states

    # reshape target data so that it is compatible with prediction shape
    flat_target_data = tf.reshape(self.target_data,[-1, self.dim])
    #[x1_data, x2_data, eos_data] = tf.split(1, 3, flat_target_data)
    x_data = flat_target_data

    def tf_normal(x, mu, sig):
        return tf.exp(-tf.square(x - mu) / (2 * tf.square(sig))) / (sig * tf.sqrt(2 * np.pi))

    
    def get_lossfunc(z_pi, z_mu, z_sig, x_data):
      result0 = tf_normal(x_data, z_mu, z_sig) 
      result1 = tf.reduce_sum(result0 * z_pi, 1, keep_dims=True)
      result2 = -tf.log(tf.maximum(result1, 1e-20)) 
      return tf.reduce_sum(result2)

    self.pi = output[:, 0:self.num_mixture]
    max_pi = tf.reduce_max(self.pi, 1, keep_dims=True)
    self.pi = tf.exp(tf.sub(self.pi, max_pi))
    normalize_pi = tf.inv(tf.reduce_sum(self.pi, 1, keep_dims=True))
    self.pi = normalize_pi * self.pi

    output_each_dim = tf.split(1, self.dim, output[:, self.num_mixture:])

    self.mu = []
    self.sig = []
    self.cost = 0

    for i in range(self.dim):
        [o_mu, o_sig] = tf.split(1, 2, output_each_dim[i])
        o_sig = tf.exp(o_sig)

        self.mu.append(o_mu)
        self.sig.append(o_sig)

        lossfunc = get_lossfunc(self.pi, o_mu, o_sig, x_data[:,i:i+1])
        self.cost += lossfunc / (args.batch_size * args.seq_length * self.dim)

    self.mu = tf.concat(1, self.mu)
    self.sig = tf.concat(1, self.sig)

    self.lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), args.grad_clip)
    optimizer = tf.train.AdamOptimizer(self.lr)
    self.train_op = optimizer.apply_gradients(zip(grads, tvars))


  def sample(self, sess, num=1200):

    def get_pi_idx(x, pdf):
      N = pdf.size
      accumulate = 0
      for i in range(0, N):
        accumulate += pdf[i]
        if (accumulate >= x):
          return i
      print 'error with sampling ensemble'
      return -1

    def sample_gaussian_2d(mu, sig):
      x = np.random.multivariate_normal(mu, np.diag(sig) ** 2, 1)
      #return x[0]
      return mu

    pose = np.array([0], dtype=np.float32)
    prev_x = np.zeros((1, 1, self.dim), dtype=np.float32)

    prev_state = sess.run(self.cell.zero_state(1, tf.float32))

    f = open("output.txt", "w")
    f.write(" ".join(["%f" % x for x in pose]) + "\n")

    height = 500
    dims = (1000, height)
    dwg = svgwrite.Drawing("result.svg", size=dims)
    dwg.add(dwg.rect(insert=(0, 0), size=dims,fill='white'))
    p = "M 0,%f " % (height / 2)

    # priming
    for i in range(100):
        feed = {self.input_data: prev_x, self.initial_state:prev_state}
        prev_x[0][0][0] = 25 * (math.sin(i * 2 * np.pi / 100) - math.sin((i-1) * 2 * np.pi / 100))
        #[o_pi, o_mu, o_sig, prev_state] = sess.run([self.pi, self.mu, self.sig, self.final_state],feed)
        prev_state = sess.run(self.final_state,feed)

    for i in xrange(num):
      feed = {self.input_data: prev_x, self.initial_state:prev_state}

      [o_pi, o_mu, o_sig, next_state] = sess.run([self.pi, self.mu, self.sig, self.final_state],feed)
      #idx = get_pi_idx(random.random(), o_pi[0])
      idx = np.argmax(o_pi[0])

      nxt = sample_gaussian_2d(o_mu[0][idx::self.num_mixture], o_sig[0][idx::self.num_mixture])
      pose += nxt  / self.args.data_scale
    
      p += "L %f,%f " % (i+1, height / 2 + pose)
      #print pose
      print idx
      f.write(" ".join(["%f" % x for x in pose]) + "\n")

      prev_x[0][0] = nxt
      prev_state = next_state

    stroke_width = 1
    the_color = "black"
    dwg.add(dwg.path(p).stroke(the_color,stroke_width).fill("none"))
    dwg.save()
    f.close()

