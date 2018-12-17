import tensorflow as tf
import os
from model import SRCNN
"""
:Author: chenhao
:Date: 2018/11/24
"""
flags = tf.app.flags    # tf.app.falgs.FLAGS    用于命令执行程序时。命令行运行程序时，可传参数运行；没有参数时，用默认值运行
flags.DEFINE_integer("epoch", 15000, "Number of epoch [15000]")
flags.DEFINE_integer("batch_size", 256, "The size of batch images [128]")
flags.DEFINE_integer("image_size", 33, "The size of image to use [33]")
flags.DEFINE_integer("label_size", 9, "The size of label to produce [21]")
flags.DEFINE_float("learning_rate", 1e-5, "The learning rate of gradient descent algorithm [1e-4]")
flags.DEFINE_integer("c_dim", 1, "Dimension of image color. [1]")
flags.DEFINE_integer("scale", 3, "The size of scale factor for preprocessing input image [3]")
flags.DEFINE_integer("stride", 14, "The size of stride to apply input image [14]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Name of checkpoint directory [checkpoint]")
flags.DEFINE_string("sample_dir", "sample", "Name of sample directory [sample]")
flags.DEFINE_boolean("is_train", False, "True for training, False for testing [True]")
FLAGS = flags.FLAGS


def main(_):
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)

    with tf.Session() as sess:
        srcnn = SRCNN(sess,
              image_size=FLAGS.image_size,
              label_size=FLAGS.label_size,
              batch_size=FLAGS.batch_size,
              c_dim=FLAGS.c_dim,
              checkpoint_dir=FLAGS.checkpoint_dir,
              sample_dir=FLAGS.sample_dir)
        srcnn.train(FLAGS)


if __name__ == '__main__':
    tf.app.run()
