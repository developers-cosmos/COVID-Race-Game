#!/usr/bin/env python

from __future__ import print_function
import numpy as np
import tensorflow as tf
import cv2

DEBUG = False


class CnnHeadPoseEstimator:

    def __init__(self, YawFilePath, PitchFilePath):
        self._init_yaw_(YawFilePath)
        self._init_pitch_(PitchFilePath)

        all_vars = tf.all_variables()

        print("========== ALL TF VARS ======== ")
        for k in all_vars:
            print(k.name)

        # Add ops to restore the variables.
        #saver = tf.train.Saver()

        p_ckpt = tf.train.get_checkpoint_state(PitchFilePath)
        y_ckpt = tf.train.get_checkpoint_state(YawFilePath)

        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
        # #Allocate only half of the GPU memory
        self._sess = tf.Session()  # (config=tf.ConfigProto(gpu_options=gpu_options))

        tf.train.Saver(({"conv1y_w": self.hy_conv1_weights, "conv1y_b": self.hy_conv1_biases,
                         "conv2y_w": self.hy_conv2_weights, "conv2y_b": self.hy_conv2_biases,
                         "conv3y_w": self.hy_conv3_weights, "conv3y_b": self.hy_conv3_biases,
                         "conv4y_w": self.hy_conv4_weights, "conv4y_b": self.hy_conv4_biases,
                         "dense1y_w": self.hy_dense1_weights, "dense1y_b": self.hy_dense1_biases,
                         "dense2y_w": self.hy_dense2_weights, "dense2y_b": self.hy_dense2_biases,
                         "outy_w": self.hy_out_weights, "outy_b": self.hy_out_biases
                         })).restore(self._sess, y_ckpt.model_checkpoint_path)

        tf.train.Saver(({"conv1p_w": self.hp_conv1_weights, "conv1p_b": self.hp_conv1_biases,
                         "conv2p_w": self.hp_conv2_weights, "conv2p_b": self.hp_conv2_biases,
                         "conv3p_w": self.hp_conv3_weights, "conv3p_b": self.hp_conv3_biases,
                         "conv4p_w": self.hp_conv4_weights, "conv4p_b": self.hp_conv4_biases,
                         "dense1p_w": self.hp_dense1_weights, "dense1p_b": self.hp_dense1_biases,
                         "dense2p_w": self.hp_dense2_weights, "dense2p_b": self.hp_dense2_biases,
                         "outp_w": self.hp_out_weights, "outp_b": self.hp_out_biases
                         })).restore(self._sess, p_ckpt.model_checkpoint_path)

    def _init_yaw_(self, YawFilePath):
        self._num_labels = 27
        # Input data [batch_size, image_size, image_size, channels]
        self.tf_yaw_input_vector = tf.placeholder(tf.float32, shape=(40, 40))

        # Variables.
        # Conv layer
        #[patch_size, patch_size, num_channels, depth]
        self.hy_conv1_weights = tf.Variable(tf.truncated_normal([3, 3, 1, 64], stddev=0.01))
        self.hy_conv1_biases = tf.Variable(tf.zeros([64]))
        # Conv layer
        #[patch_size, patch_size, depth, depth]
        self.hy_conv2_weights = tf.Variable(tf.truncated_normal([3, 3, 64, 128], stddev=0.01))
        self.hy_conv2_biases = tf.Variable(tf.random_normal(shape=[128]))
        # Conv layer
        #[patch_size, patch_size, depth, depth]
        self.hy_conv3_weights = tf.Variable(tf.truncated_normal([3, 3, 128, 256], stddev=0.01))  # was[3, 3, 128, 256]
        self.hy_conv3_biases = tf.Variable(tf.random_normal(shape=[256]))
        # Conv layer
        #[patch_size, patch_size, depth, depth]
        self.hy_conv4_weights = tf.Variable(tf.truncated_normal([3, 3, 256, 512], stddev=0.01))  # was[3, 3, 128, 256]
        self.hy_conv4_biases = tf.Variable(tf.random_normal(shape=[512]))

        # Dense layer
        #[ 5*5 * previous_layer_out , num_hidden] wd1
        # here 5*5 is the size of the image after pool reduction (divide by
        # half 3 times)
        self.hy_dense1_weights = tf.Variable(tf.truncated_normal([10*10*512, 4096], stddev=0.01))  # was [5*5*256, 1024]
        self.hy_dense1_biases = tf.Variable(tf.random_normal(shape=[4096]))
        # Dense layer
        #[ , num_hidden] wd2
        self.hy_dense2_weights = tf.Variable(tf.truncated_normal([4096, 4096], stddev=0.01))
        self.hy_dense2_biases = tf.Variable(tf.random_normal(shape=[4096]))
        # Output layer
        self.hy_out_weights = tf.Variable(tf.truncated_normal([4096, self._num_labels], stddev=0.01))
        self.hy_out_biases = tf.Variable(tf.random_normal(shape=[self._num_labels]))

        # dropout (keep probability)
        #self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")

        # Model
        def model(data):

            X = tf.reshape(data, shape=[-1, 40, 40, 1])
            if(DEBUG == True):
                print("SHAPE X: " + str(X.get_shape()))

            # Convolution Layer 1
            conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(X, self.hy_conv1_weights,
                strides=[1, 1, 1, 1], padding='SAME'), self.hy_conv1_biases))
            if(DEBUG == True):
                print("SHAPE conv1: " + str(conv1.get_shape()))
            # Max Pooling (down-sampling)
            pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1], padding='SAME')
            if(DEBUG == True):
                print("SHAPE pool1: " + str(pool1.get_shape()))
            # Apply Normalization
            norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
            # Apply Dropout
            #norm1 = tf.nn.dropout(norm1, _dropout)

            # Convolution Layer 2
            conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(norm1, self.hy_conv2_weights,
                strides=[1, 1, 1, 1], padding='SAME'), self.hy_conv2_biases))
            if(DEBUG == True):
                print("SHAPE conv2: " + str(conv2.get_shape()))
            # Max Pooling (down-sampling)
            pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1], padding='SAME')
            if(DEBUG == True):
                print("SHAPE pool2: " + str(pool2.get_shape()))
            # Apply Normalization
            norm2 = tf.nn.lrn(pool2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
            # Apply Dropout
            #norm2 = tf.nn.dropout(norm2, _dropout)

            # Convolution Layer 3
            conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(norm2, self.hy_conv3_weights,
                strides=[1, 1, 1, 1], padding='SAME'), self.hy_conv3_biases))
            if(DEBUG == True):
                print("SHAPE conv3: " + str(conv3.get_shape()))
            # Max Pooling (down-sampling)
            #pool3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            #if(DEBUG == True): print("SHAPE pool3: " + str(pool3.get_shape()))
            # Apply Normalization
            #norm3 = tf.nn.lrn(pool3, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
            #print("SHAPE norm3: " + str(norm3.get_shape()))
            # Apply Dropout
            #norm3 = tf.nn.dropout(norm3, _dropout)
            #if(DEBUG == True): print("SHAPE norm3: " + str(norm3.get_shape()))

            # Convolution Layer 4
            conv4 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv3, self.hy_conv4_weights,
                strides=[1, 1, 1, 1], padding='SAME'), self.hy_conv4_biases))
            if(DEBUG == True):
                print("SHAPE conv4: " + str(conv4.get_shape()))

            # Fully connected layer 5
            dense1 = tf.reshape(
                conv4, [-1, self.hy_dense1_weights.get_shape().as_list()[0]])  # Reshape conv3
            if(DEBUG == True):
                print("SHAPE dense1: " + str(dense1.get_shape()))
            dense1 = tf.nn.relu(
                tf.matmul(dense1, self.hy_dense1_weights) + self.hy_dense1_biases)  # Relu

            # Fully connected layer 6
            dense2 = tf.nn.relu(
                tf.matmul(dense1, self.hy_dense2_weights) + self.hy_dense2_biases)  # Relu
            if(DEBUG == True):
                print("SHAPE dense2: " + str(dense2.get_shape()))

            # Output layer 7
            out = tf.matmul(dense2, self.hy_out_weights) + self.hy_out_biases
            if(DEBUG == True):
                print("SHAPE out: " + str(out.get_shape()))

            return out

        # Predictions for the training, validation, and test data.
        logits = model(self.tf_yaw_input_vector)
        self._yaw_prediction = tf.nn.softmax(logits)

    def _init_pitch_(self, PitchFilePath):
        num_labels_pitch = 8
        # Input data [batch_size, image_size, image_size, channels]
        self.tf_pitch_input_vector = tf.placeholder(tf.float32, shape=(40, 40))

        # Variables.
        # Conv layer
        #[patch_size, patch_size, num_channels, depth]
        self.hp_conv1_weights = tf.Variable(tf.truncated_normal([3, 3, 1, 64], stddev=0.01))
        self.hp_conv1_biases = tf.Variable(tf.zeros([64]))
        # Conv layer
        #[patch_size, patch_size, depth, depth]
        self.hp_conv2_weights = tf.Variable(tf.truncated_normal([3, 3, 64, 128], stddev=0.01))
        self.hp_conv2_biases = tf.Variable(tf.random_normal(shape=[128]))
        # Conv layer
        #[patch_size, patch_size, depth, depth]
        self.hp_conv3_weights = tf.Variable(tf.truncated_normal([3, 3, 128, 256], stddev=0.01))  # was[3, 3, 128, 256]
        self.hp_conv3_biases = tf.Variable(tf.random_normal(shape=[256]))
        # Conv layer
        #[patch_size, patch_size, depth, depth]
        self.hp_conv4_weights = tf.Variable(tf.truncated_normal([3, 3, 256, 512], stddev=0.01))  # was[3, 3, 128, 256]
        self.hp_conv4_biases = tf.Variable(tf.random_normal(shape=[512]))

        # Dense layer
        #[ 5*5 * previous_layer_out , num_hidden] wd1
        # here 5*5 is the size of the image after pool reduction (divide by
        # half 3 times)
        self.hp_dense1_weights = tf.Variable(tf.truncated_normal([10*10*512, 4096], stddev=0.01))  # was [5*5*256, 1024]
        self.hp_dense1_biases = tf.Variable(tf.random_normal(shape=[4096]))
        # Dense layer
        #[ , num_hidden] wd2
        self.hp_dense2_weights = tf.Variable(tf.truncated_normal([4096, 4096], stddev=0.01))
        self.hp_dense2_biases = tf.Variable(tf.random_normal(shape=[4096]))
        # Output layer
        self.hp_out_weights = tf.Variable(tf.truncated_normal([4096, num_labels_pitch], stddev=0.01))
        self.hp_out_biases = tf.Variable(tf.random_normal(shape=[num_labels_pitch]))

        # Model
        def model(data, _dropout=1.0):

            X = tf.reshape(data, shape=[-1, 40, 40, 1])
            if(DEBUG == True):
                print("SHAPE X: " + str(X.get_shape()))

            # Convolution Layer 1
            conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(X, self.hp_conv1_weights,
                strides=[1, 1, 1, 1], padding='SAME'), self.hp_conv1_biases))
            if(DEBUG == True):
                print("SHAPE conv1: " + str(conv1.get_shape()))
            # Max Pooling (down-sampling)
            pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1], padding='SAME')
            if(DEBUG == True):
                print("SHAPE pool1: " + str(pool1.get_shape()))
            # Apply Normalization
            norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
            # Apply Dropout
            norm1 = tf.nn.dropout(norm1, _dropout)

            # Convolution Layer 2
            conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(norm1, self.hp_conv2_weights,
                strides=[1, 1, 1, 1], padding='SAME'), self.hp_conv2_biases))
            if(DEBUG == True):
                print("SHAPE conv2: " + str(conv2.get_shape()))
            # Max Pooling (down-sampling)
            pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1], padding='SAME')
            if(DEBUG == True):
                print("SHAPE pool2: " + str(pool2.get_shape()))
            # Apply Normalization
            norm2 = tf.nn.lrn(pool2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
            # Apply Dropout
            norm2 = tf.nn.dropout(norm2, _dropout)

            # Convolution Layer 3
            conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(norm2, self.hp_conv3_weights,
                strides=[1, 1, 1, 1], padding='SAME'), self.hp_conv3_biases))
            if(DEBUG == True):
                print("SHAPE conv3: " + str(conv3.get_shape()))
            # Max Pooling (down-sampling)
            #pool3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            #if(DEBUG == True): print("SHAPE pool3: " + str(pool3.get_shape()))
            # Apply Normalization
            #norm3 = tf.nn.lrn(pool3, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
            #print("SHAPE norm3: " + str(norm3.get_shape()))
            # Apply Dropout
            #norm3 = tf.nn.dropout(norm3, _dropout)
            #if(DEBUG == True): print("SHAPE norm3: " + str(norm3.get_shape()))

            # Convolution Layer 4
            conv4 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv3, self.hp_conv4_weights,
                strides=[1, 1, 1, 1], padding='SAME'), self.hp_conv4_biases))
            print("SHAPE conv4: " + str(conv4.get_shape()))

            # Fully connected layer 4
            dense1 = tf.reshape(
                conv4, [-1, self.hp_dense1_weights.get_shape().as_list()[0]])  # Reshape conv3
            if(DEBUG == True):
                print("SHAPE dense1: " + str(dense1.get_shape()))
            dense1 = tf.nn.relu(
                tf.matmul(dense1, self.hp_dense1_weights) + self.hp_dense1_biases)  # Relu

            # Fully connected layer 5
            dense2 = tf.nn.relu(
                tf.matmul(dense1, self.hp_dense2_weights) + self.hp_dense2_biases)  # Relu
            if(DEBUG == True):
                print("SHAPE dense2: " + str(dense2.get_shape()))

            # Output layer 6
            out = tf.matmul(dense2, self.hp_out_weights) + self.hp_out_biases
            if(DEBUG == True):
                print("SHAPE out: " + str(out.get_shape()))

            return out

        # Predictions for the training, validation, and test data.
        logits = model(self.tf_pitch_input_vector)
        self._pitch_prediction = tf.nn.softmax(logits)

    def return_yaw_probability(self, image):
        h, w = image.shape
        if(h == w and h > 39):
            #batch_data = train_dataset[1, :, :, :]
            feed_dict = {self.tf_yaw_input_vector: image}
            predictions = self._sess.run(
                [self._yaw_prediction], feed_dict=feed_dict)

            return predictions  # It returns a probability distribution
        else:
            raise ValueError(
                'CnnHeadPoseEstimator: the image given as input is not squared or it is smaller than 40px.')

    def return_pitch_probability(self, image):
        h, w = image.shape
        if(h == w and h > 39):
            #batch_data = train_dataset[1, :, :, :]
            feed_dict = {self.tf_pitch_input_vector: image}
            predictions = self._sess.run(
                [self._pitch_prediction], feed_dict=feed_dict)

            return predictions  # It returns a probability distribution
        else:
            raise ValueError(
                'CnnHeadPoseEstimator: the image given as input is not squared or it is smaller than 40px.')
