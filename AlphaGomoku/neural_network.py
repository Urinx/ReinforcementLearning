import numpy as np
import tensorflow as tf
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization, Activation, LeakyReLU, add
from keras.optimizers import SGD, Adam
from keras import regularizers
from keras.callbacks import TensorBoard, Callback

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class LossHistory(Callback):
    def __init__(self):
        self.losses = []
        self.policy_head_losses = []
        self.value_head_losses = []        

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        self.policy_head_losses.append(logs.get('policy_head_loss'))
        self.value_head_losses.append(logs.get('value_head_loss'))

    def plot_loss(self, img_file):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(self.losses)
        ax.plot(self.policy_head_losses)
        ax.plot(self.value_head_losses)
        plt.title('Model loss')
        plt.ylabel('loss')
        plt.xlabel('episode')
        plt.legend(['loss', 'policy_head_loss', 'value_head_loss'], loc='upper right')
        plt.savefig(img_file)
        plt.close(fig)

class NetworkModel:
    def __init__(self):
        pass

    def train(self, states, targets):
        # model_log = TensorBoard(log_dir='./logs')
        return self.model.fit(states, targets, verbose=self.verbose, callbacks=[self.loss_history])

    def pred(self, x):
        return self.model.predict(x)

    def load(self, name):
        return self.model.load_weights('models/{}.h5'.format(name))

    def save(self, name):
        self.model.save_weights('models/{}.h5'.format(name))

    def info(self):
        self.model.summary()


class Residual_CNN(NetworkModel):

    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.conv_layer_filters = 64
        self.conv_layer_kernel_size = (3, 3)
        self.residual_layer_num = 2
        self.value_head_hidden_layer_size = 20

        self.learning_rate = 0.1
        self.momentum = 0.9
        self.reg_const = 0.0001

        self.batch_size = 20
        self.epochs = 1
        self.verbose = True

        self.model = self.build_model()
        self.loss_history = LossHistory()

    def build_model(self):
        """Construct a convolutional neural network with Resnet-style skip connections.

        Network Diagram:                                                                        [value head]
                              |---------------------------------|                   /---C---B---R---F---D---R---D---T
        I-----C-----B-----R---o---C-----B-----R-----C-----B-----M-----R--- ..... ---|
              \___________/     \___________________________________/               \---C---B---R---F---D---S [polich head]
           [Convolutional layer]          [Residual layer]

        I - input
        B - BatchNormalization
        R - Rectifier non-linearity, LeakyReLU
        T - tanh
        C - Conv2D
        F - Flatten
        D - Dense
        M - merge, add
        S - Softmax
        O - output
        """
        main_input = Input(shape=self.input_dim, name='main_input')

        x = self.conv_layer(main_input, self.conv_layer_filters, self.conv_layer_kernel_size)
        for _ in range(self.residual_layer_num):
            x = self.residual_layer(x, self.conv_layer_filters, self.conv_layer_kernel_size)

        vh = self.value_head(x)
        ph = self.policy_head(x)

        model = Model(inputs=main_input, outputs=[vh, ph])
        model.compile(
            loss={'value_head': 'mean_squared_error','policy_head': self.softmax_cross_entropy_with_logits},
            # loss={'value_head': 'mean_squared_error','policy_head': 'categorical_crossentropy'},
            optimizer=SGD(lr=self.learning_rate, momentum=self.momentum),
            loss_weights={'value_head': 0.5, 'policy_head': 0.5}
            )

        return model

    def conv_layer(self, x, filters, kernel_size):
        conv = Conv2D(
            filters = filters,
            kernel_size = kernel_size,
            strides = (1, 1),
            padding = 'same',
            data_format = 'channels_first',
            use_bias = False,
            activation = 'linear',
            kernel_regularizer = regularizers.l2(self.reg_const)
            )(x)
        bn = BatchNormalization(axis=1)(conv)
        lrelu = LeakyReLU()(bn)
        return lrelu

    def residual_layer(self, x, filters, kernel_size):
        conv_1 = self.conv_layer(x, filters, kernel_size)
        conv_2 = Conv2D(
            filters = filters,
            kernel_size = kernel_size,
            strides = (1, 1),
            padding = 'same',
            data_format = 'channels_first',
            use_bias = False,
            activation = 'linear',
            kernel_regularizer = regularizers.l2(self.reg_const)
            )(conv_1)
        bn = BatchNormalization(axis=1)(conv_2)
        merge_layer = add([x, bn])
        lrelu = LeakyReLU()(merge_layer)
        return lrelu

    def value_head(self, x):
        x = self.conv_layer(x, 1, (1, 1))
        x = Flatten()(x)
        x = Dense(
            self.value_head_hidden_layer_size,
            use_bias = False,
            activation = 'linear',
            kernel_regularizer = regularizers.l2(self.reg_const)
            )(x)
        x = LeakyReLU()(x)
        x = Dense(
            1,
            use_bias = False,
            activation = 'tanh',
            kernel_regularizer = regularizers.l2(self.reg_const),
            name = 'value_head'
            )(x)
        return x

    def policy_head(self, x):
        x = self.conv_layer(x, 2, (1, 1))
        x = Flatten()(x)
        x = Dense(
            self.output_dim,
            use_bias = False,
            activation = 'softmax',
            kernel_regularizer = regularizers.l2(self.reg_const),
            name = 'policy_head'
            )(x)
        return x


    def softmax_cross_entropy_with_logits(self, y_true, y_pred):
        p = y_pred
        pi = y_true

        zero = tf.zeros(tf.shape(pi), tf.float32)
        where = tf.equal(pi, zero)

        negatives = tf.fill(tf.shape(pi), -100.0)
        p = tf.where(where, negatives, p)

        loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=pi, logits=p)
        return loss

class Simple_CNN(NetworkModel):
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.l2_const = 1e-4

        self.verbose = True

        self.model = self.build_model()
        self.loss_history = LossHistory()

    def build_model(self):
        """
        Network Diagram:
                                                       2(1x1)       64     1
            32(3x3)     64(3x3)    128(3x3)        /-----C-----F-----D-----D-----T  [value head]
        I-----C-----R-----C-----R-----C-----R-----|
              \_____________________________/      \-----C-----F-----D-----S        [polich head]
                   [Convolutional layer]               4(1x1)       w^2

        I - input
        B - BatchNormalization
        R - ReLU
        T - tanh
        C - Conv2D
        F - Flatten
        D - Dense
        S - Softmax
        """
        main_input = Input(shape=self.input_dim, name='main_input')

        x = self.conv_layer(main_input, 32, (3, 3))
        x = self.conv_layer(x, 64, (3, 3))
        x = self.conv_layer(x, 128, (3, 3))

        vh = self.value_head(x)
        ph = self.policy_head(x)

        model = Model(main_input, [vh, ph])
        model.compile(
            optimizer=Adam(),
            loss=['mean_squared_error', 'categorical_crossentropy']
            )
        return model

    def conv_layer(self, x, filters, kernel_size, padding='same'):
        conv = Conv2D(
            filters = filters,
            kernel_size = kernel_size,
            padding = padding,
            data_format = 'channels_first',
            activation = 'relu',
            kernel_regularizer = regularizers.l2(self.l2_const)
        )(x)
        return conv

    def value_head(self, x):
        x = self.conv_layer(x, 2, (1, 1), 'valid')
        x = Flatten()(x)
        x = Dense(64, kernel_regularizer=regularizers.l2(self.l2_const))(x)
        x = Dense(
            1,
            kernel_regularizer = regularizers.l2(self.l2_const),
            activation = 'tanh',
            name = 'value_head'
            )(x)
        return x

    def policy_head(self, x):
        x = self.conv_layer(x, 4, (1, 1), 'valid')
        x = Flatten()(x)
        x = Dense(
            self.output_dim,
            kernel_regularizer = regularizers.l2(self.l2_const),
            activation = 'softmax',
            name = 'policy_head'
            )(x)
        return x

