import sys
execfile(sys.path[0] + "/../utils/tensorutils.py")

from utils import DataLoader
from model import Model

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--rnn_size', type=int, default=256,
                     help='size of RNN hidden state')
  parser.add_argument('--num_layers', type=int, default=3,
                     help='number of layers in the RNN')
  parser.add_argument('--model', type=str, default='lstm',
                     help='rnn, gru, or lstm')
  parser.add_argument('--batch_size', type=int, default=50,
                     help='minibatch size')
  parser.add_argument('--seq_length', type=int, default=300,
                     help='RNN sequence length')
  parser.add_argument('--num_epochs', type=int, default=200,
                     help='number of epochs')
  parser.add_argument('--save_every', type=int, default=20,
                     help='save frequency')
  parser.add_argument('--grad_clip', type=float, default=10.,
                     help='clip gradients at this value')
  parser.add_argument('--learning_rate', type=float, default=0.005,
                     help='learning rate')
  parser.add_argument('--decay_rate', type=float, default=0.95,
                     help='decay rate for rmsprop')
  parser.add_argument('--num_mixture', type=int, default=20,
                     help='number of gaussian mixtures')
  parser.add_argument('--data_scale', type=float, default=1,
                     help='factor to scale raw data down by')
  parser.add_argument('--keep_prob', type=float, default=0.8,
                     help='dropout keep probability')
  parser.add_argument('--reprocess', type=int, default=0,
                     help='reprocess input')
  args = parser.parse_args()
  train(args)

def train(args):
    data_loader = DataLoader(args.batch_size, args.seq_length, args.data_scale, reprocess=args.reprocess)
    x, y = data_loader.next_batch()

    with open(os.path.join('save', 'config.pkl'), 'w') as f:
        cPickle.dump(args, f)

    model = Model(args)

    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        ts = TrainingStatus(sess, args.num_epochs, data_loader.num_batches, save_interval = args.save_every, graph_def = sess.graph_def)

        for e in xrange(ts.startEpoch, args.num_epochs):
            sess.run(tf.assign(model.lr, args.learning_rate * (args.decay_rate ** e)))
            data_loader.reset_batch_pointer()
            state = model.initial_state.eval()
            for b in xrange(data_loader.num_batches):
                ts.tic()
                x, y = data_loader.next_batch()
                feed = {model.input_data: x, model.target_data: y, model.initial_state: state}
                summary, train_loss, state, _ = sess.run([model.summary, model.cost, model.final_state, model.train_op], feed)
                print ts.tocBatch(summary, e, b, train_loss)

            ts.tocEpoch(sess, e)


if __name__ == '__main__':
  main()


