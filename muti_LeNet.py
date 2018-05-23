import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
#数据准备
import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot = True)
X_train = mnist.train.images
Y_train = mnist.train.labels

class mutiLeNet:
    def __init__(self):
        self.xs = tf.placeholder(tf.float32, [None, 784])
        self.ys = tf.placeholder(tf.float32, [None, 10])
        self.dp_k = tf.placeholder(tf.float32)
        self.xs_reshape = tf.reshape(self.xs, [-1, 28, 28, 1])
        self.y_out = self.muti_LeNet(self.xs_reshape)

        # key point
        self.train_y_out = tf.nn.softmax(self.y_out)
        # self.train_y_out = self.y_out
        #

        self.loss = self.get_loss()
        self.train_step = tf.train.AdamOptimizer(0.0001).minimize(self.loss)
        self.correct_p = tf.equal(tf.argmax(self.y_out, 1), (tf.argmax(self.ys, 1)))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_p, "float"))
        self.sess = tf.Session()

    def conv(self, input_data, input_size, output_size, f_size=3):
        w = tf.Variable(tf.random_normal([f_size,f_size,input_size,output_size],stddev = 0.1))
        b = tf.Variable(tf.constant(0.1, shape = [output_size]))
        return tf.nn.relu(tf.nn.conv2d(input_data, w, [1,1,1,1], padding = 'SAME') + b)

    def pooling(self, input_data):
        return tf.nn.avg_pool(input_data, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

    def addLayer(self, input_data, input_size, output_size,type=None):
        W = tf.Variable(tf.random_normal([input_size,output_size],stddev = 0.1))
        basis = tf.Variable(tf.constant(0.1,shape = [output_size]))
        re = tf.matmul(input_data, W) + basis
        if type=="relu":
            return tf.nn.relu(re)
        elif type=="softmax":
            return tf.nn.softmax(re)
        elif type=="sigmoid":
            return tf.nn.sigmoid(re)
        else:
            return re


    def LeNet(self,input):
        l_1 = self.conv(input,1,16,5)
        l_1_p = self.pooling(l_1)
        l_2 = self.conv(l_1_p,16,32,5)
        l_2_p = self.pooling(l_2)
        re = tf.reshape(l_2_p,[-1,7*7*32])
        f_1 = self.addLayer(re,7*7*32,128,"relu")
        f_2 = self.addLayer(f_1,128,128,"relu")
        dp = tf.nn.dropout(f_2,self.dp_k)
        out = self.addLayer(dp,128,1)
        #key point 2
        # out = tf.nn.softmax(out)
        return out

    def muti_LeNet(self,input):
        out = tf.stack([self.LeNet(input),
                        self.LeNet(input),
                        self.LeNet(input),
                        self.LeNet(input),
                        self.LeNet(input),
                        self.LeNet(input),
                        self.LeNet(input),
                        self.LeNet(input),
                        self.LeNet(input),
                        self.LeNet(input)],1)
        out = tf.reshape(out,[-1,10])
        # out = tf.nn.sigmoid(out)
        # out = tf.nn.softmax(out)
        # out = self.addLayer(out,10,10,"softmax")
        return out

    def get_loss(self):
        return -tf.reduce_sum(self.ys*tf.log(self.train_y_out))
        # return tf.reduce_sum(tf.square(self.ys-self.y_out))

    def train(self):
        self.sess.run(tf.global_variables_initializer())
        # 训练过程
        print("Begin Train:")
        for i in range(1000):
           batch = mnist.train.next_batch(50)
           self.sess.run(self.train_step,feed_dict={self.xs:batch[0],self.ys:batch[1],self.dp_k:0.8})
           if i%10 == 0:
               print("i=%d,acc=%f,loss=%f"%(i,self.sess.run(self.accuracy,feed_dict={self.xs:batch[0],self.ys:batch[1],self.dp_k:1.0}),self.sess.run(self.loss,feed_dict={self.xs:batch[0],self.ys:batch[1],self.dp_k:1.0})))
        print("Begin Test:")
        testbatch = mnist.test.next_batch(100)
        print("final_acc=%f"%self.sess.run(self.accuracy,feed_dict={self.xs:testbatch[0],self.ys:testbatch[1],self.dp_k:1.0}))

    def save(self):
        saver = tf.train.Saver()
        saver.save(self.sess, "mutiLeNet_model/model.ckpt")

    def restore_model(self):
        saver = tf.train.Saver()
        saver.restore(self.sess, "mutiLeNet_model/model.ckpt")

mutiLeNet_demo = mutiLeNet()
mutiLeNet_demo.train()
mutiLeNet_demo.save()
mutiLeNet_demo.restore_model()
testbatch = mnist.test.next_batch(1)
print(mutiLeNet_demo.sess.run(mutiLeNet_demo.train_y_out,feed_dict={mutiLeNet_demo.xs:testbatch[0],mutiLeNet_demo.ys:testbatch[1],mutiLeNet_demo.dp_k:1.0}))
plt.imshow(np.reshape(testbatch[0][0],[28,28]))
plt.show()

testbatch = mnist.test.next_batch(3)
tutu_x = np.reshape(np.logical_or(testbatch[0][0],testbatch[0][1],dtype=bool),[-1,784])
tutu_y = np.reshape((testbatch[1][0] + testbatch[1][1]),[-1,10])
print(mutiLeNet_demo.sess.run(mutiLeNet_demo.train_y_out,feed_dict={mutiLeNet_demo.xs:tutu_x,
                                                            mutiLeNet_demo.ys:tutu_y,
                                                            mutiLeNet_demo.dp_k:1.0}))
print(tutu_y)
plt.imshow(np.reshape(testbatch[0][0]+testbatch[0][1],[28,28]))
plt.show()
