import tensorflow as tf
from util import(
    input_setup,
    read_data,
    merge,
    imsave
)
import os
import time
"""
:Author: chenhao
:Date: 2018/11/24
:Content: 图像超分辨率网络模型包
"""
class SRCNN(object):
    def __init__(self,
                 sess,
                 image_size,
                 label_size,
                 batch_size,
                 c_dim,
                 checkpoint_dir,
                 sample_dir):
        self.sess = sess
        self.is_grayscale = (c_dim == 1)
        self.image_size = image_size
        self.label_size = label_size
        self.batch_size = batch_size
        self.c_dim = c_dim
        self.checkpoint_dir = checkpoint_dir
        self.sample_dir = sample_dir
        self.build_model()

    def build_model(self):
        """
        :Coontent: 模型参数设置
        """
        self.images = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.c_dim], name="images")
        self.labels = tf.placeholder(tf.float32, [None, self.label_size, self.label_size, self.c_dim], name="labels")
        self.weights = {
            "w1": tf.Variable(tf.random_normal([9, 9, 1, 64], stddev=1e-3), name="w1"),
            "w2": tf.Variable(tf.random_normal([7, 7, 64, 32], stddev=1e-3), name="w2"),
            "w3": tf.Variable(tf.random_normal([7, 7, 32, 16], stddev=1e-3), name="w3"),
            "w4": tf.Variable(tf.random_normal([5, 5, 16, 1], stddev=1e-3), name="w4")
        }
        self.biases = {
            "b1": tf.Variable(tf.zeros(64), name="b1"),
            "b2": tf.Variable(tf.zeros(32), name="b2"),
            "b3": tf.Variable(tf.zeros(16), name="b3"),
            "b4": tf.Variable(tf.zeros(1), name="b4")
        }
        self.pred = self.model()
        self.loss = tf.losses.mean_squared_error(self.labels, self.pred)
        self.saver = tf.train.Saver()

    def train(self, config):
        if config.is_train:
            input_setup(self.sess, config)
        else:
            nx, ny = input_setup(self.sess, config) # 合并图像块数

        if config.is_train:
            data_path = os.path.join("./", config.checkpoint_dir, "train.h5")
        else:
            data_path = os.path.join("./", config.checkpoint_dir, "test.h5")

        train_data, train_label = read_data(data_path)

        self.train_op = tf.train.GradientDescentOptimizer(config.learning_rate).minimize(self.loss)

        tf.global_variables_initializer().run()

        counter = 0  # 输出判断数
        start_time = time.time()
        # 加载训练数据
        if self.load(config.checkpoint_dir):
            print("[*] Load SUCCESS")
        else:
            print("[!] Load Failed")
        if config.is_train:
            print("Train....")
            batch_index = len(train_data) // config.batch_size
            for ep in range(config.epoch):
                for idx in range(batch_index):
                    batch_images = train_data[idx*config.batch_size:(idx+1)*config.batch_size]
                    batch_labels = train_label[idx*config.batch_size:(idx+1)*config.batch_size]
                    _, err = self.sess.run([self.train_op, self.loss], {self.images: batch_images, self.labels: batch_labels})
                    counter += 1

                    if counter % 10 == 0:
                        print("Epoch: %2d,step: %2d,time: %4.4f,loss: %.8f" % ((ep+1), counter, time.time()-start_time, err))
                    if counter % 500 == 0:
                        self.save(config.checkpoint_dir, counter)
        else:
            print("Test...")
            result = self.pred.eval({self.images: train_data, self.labels: train_label})
            result = merge(result, [nx, ny])
            result = result.squeeze()   # squeese():把 result 的 ? 维度删除
            image_path = os.path.join(os.getcwd(), config.sample_dir, "text_image.png")
            imsave(image_path, result)

    def model(self):
        """
        :Content: 模型结构设置
        """
        conv1 = tf.nn.leaky_relu(tf.nn.conv2d(self.images, self.weights["w1"], strides=[1, 1, 1, 1], padding="VALID") + self.biases["b1"], alpha=0.005)
        conv2 = tf.nn.leaky_relu(tf.nn.conv2d(conv1, self.weights["w2"], strides=[1, 1, 1, 1], padding="VALID") + self.biases["b2"], alpha=0.005)
        conv3 = tf.nn.leaky_relu(tf.nn.conv2d(conv2, self.weights["w3"], strides=[1, 1, 1, 1], padding="VALID") + self.biases["b3"], alpha=0.005)
        conv4 = tf.nn.conv2d(conv3, self.weights["w4"], strides=[1, 1, 1, 1], padding="VALID") + self.biases["b4"]
        return conv4

    def save(self, checkpoint_dir, step):
        """
        :param checkpoint_dir: 文件路径:
        :param step: 运行次数
        :Content: 保存训练数据
        """
        model_name = "SRCNN.model"
        model_dir = "%s_%s" % ("srcnn", self.label_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=step)

    def load(self, checkpoint_dir):
        """
        :Content: 恢复网络模型保存的最新数据
        """
        print("[*] Reading checkpoint...")
        model_dir = "%s_%s" % ("srcnn", self.label_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)    # os.path.basename(path) 返回路径最后的文件名
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False



