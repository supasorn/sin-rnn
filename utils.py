import os
import glob
import cPickle
import numpy as np 
import svgwrite
import random
import math

class DataLoader():
  def __init__(self, batch_size=50, seq_length=300, scale_factor = 1, limit = 500, reprocess = 0):
    self.scale_factor = scale_factor
    self.data_dir = "data/"
    self.batch_size = batch_size
    self.seq_length = seq_length
    self.limit = limit # removes large noisy gaps in the data

    data_file = os.path.join(self.data_dir, "training.cpkl")

    if not (os.path.exists(data_file)) or reprocess:
        print "creating training data cpkl file from raw source"
        self.preprocess(data_file)

    self.load_preprocessed(data_file)
    self.reset_batch_pointer()

  def preprocess(self, data_file):
    sins = []
    # number of examples
    dims = (1000, 100)
    for i in range(100):

        sin = np.zeros((1000, 1), dtype=np.float32) 
        amp = 5 + random.random() * 25
        for j in range(sin.shape[0]):
            sin[j] = amp * math.sin((j) * 2 * np.pi / 100)
        sins.append(sin)

        if i < 10:
            dwg = svgwrite.Drawing("draw/%04d.svg" % i, size=dims)
            dwg.add(dwg.rect(insert=(0, 0), size=dims,fill='white'))
            p = ""
            for j in range(len(sin)):
                p += "%s %f,%f " % ("M" if j == 0 else "L", j, 50 + sin[j][0])
            stroke_width = 1
            the_color = "black"
            dwg.add(dwg.path(p).stroke(the_color,stroke_width).fill("none"))
            dwg.save()

    #print sins
    f = open(data_file,"wb")
    cPickle.dump(sins, f, protocol=2)
    f.close()


  def load_preprocessed(self, data_file):
    f = open(data_file,"rb")
    self.raw_data = cPickle.load(f)
    f.close()

    self.data = []
    counter = 0

    for data in self.raw_data:
      if len(data) >= (self.seq_length+2):
        data *= self.scale_factor
        a = data[:, 0:1]
        diff = a - np.roll(a, 1)
        diff[0] = 0
        #roll[0] = 0
        self.data.append(diff)

        #print a - roll 
        #exit(0)
        counter += int(len(data) / ((self.seq_length+2))) 

    self.num_batches = int(counter / self.batch_size)

  def next_batch(self):
    # returns a randomised, seq_length sized portion of the training data
    x_batch = []
    y_batch = []
    for i in xrange(self.batch_size):
      data = self.data[self.pointer]
      n_batch = int(len(data) / ((self.seq_length+2))) 

      idx = random.randint(1, len(data) - self.seq_length - 1)
      x_batch.append(np.copy(data[idx:idx+self.seq_length]))
      y_batch.append(np.copy(data[idx+1:idx+self.seq_length+1]))

      if random.random() < (1.0/float(n_batch)): 
        self.tick_batch_pointer()
    return x_batch, y_batch

  def tick_batch_pointer(self):
    self.pointer += 1
    if (self.pointer >= len(self.data)):
      self.pointer = 0

  def reset_batch_pointer(self):
    self.pointer = 0

