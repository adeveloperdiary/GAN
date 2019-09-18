from keras.datasets import mnist
import keras as k
import numpy as np


class AutoEncoder:
    def __init__(self,
                 input_dim,
                 encoder_conv_filters,
                 encoder_conv_kernels,
                 encoder_conv_strides,
                 decoder_conv_t_filters,
                 decoder_conv_t_kernels,
                 decoder_conv_t_strides,
                 z_dim,
                 use_batch_norm=False,
                 use_dropout=False
                 ):
        self.input_dim = input_dim
        self.encoder_conv_filters = encoder_conv_filters
        self.encoder_conv_kernels = encoder_conv_kernels
        self.encoder_conv_strides = encoder_conv_strides
        self.decoder_conv_t_filters = decoder_conv_t_filters
        self.decoder_conv_t_kernels = decoder_conv_t_kernels
        self.decoder_conv_t_strides = decoder_conv_t_strides
        self.z_dim = z_dim
        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout

        self._build()

    def _build(self):
        # ---------- ENCODER ---------------

        encoder_input = k.layers.Input(shape=self.input_dim, name="encoder_input")

        x = encoder_input

        for i in range(len(self.encoder_conv_filters)):
            x = k.layers.Conv2D(filters=self.encoder_conv_filters[i],
                                kernel_size=self.encoder_conv_kernels[i],
                                strides=self.encoder_conv_strides[i],
                                padding='same',
                                name='encoder_conv_' + str(i + 1))(x)

            x = k.layers.LeakyReLU()(x)

            if self.use_batch_norm:
                x = k.layers.BatchNormalization()(x)

            if self.use_dropout:
                x = k.layers.Dropout(rate=0.25)(x)

        shape_before_flattening = k.backend.int_shape(x)[1:]
        print(shape_before_flattening, k.backend.int_shape(x))

        x = k.layers.Flatten()(x)

        encoder_output = k.layers.Dense(self.z_dim, name='encoder_output')(x)

        self.encoder = k.models.Model(encoder_input, encoder_output)

        # ---------- DECODER ---------------

        decoder_input = k.layers.Input(shape=(self.z_dim,), name='decoder_input')

        x = k.layers.Dense(np.prod(shape_before_flattening))(decoder_input)

        x = k.layers.Reshape(shape_before_flattening)(x)

        for i in range(len(self.decoder_conv_t_filters)):
            x = k.layers.Conv2DTranspose(filters=self.decoder_conv_t_filters[i],
                                         kernel_size=self.decoder_conv_t_kernels[i],
                                         strides=self.decoder_conv_t_strides[i],
                                         padding='same',
                                         name='decoder_conv_t' + str(i + 1))(x)

            # if not the last layer.
            if i < len(self.decoder_conv_t_filters) - 1:
                x = k.layers.LeakyReLU()(x)

                if self.use_batch_norm:
                    x = k.layers.BatchNormalization()(x)

                if self.use_dropout:
                    x = k.layers.Dropout(range=0.25)
            else:
                x = k.layers.Activation('sigmoid')(x)

        decoder_output = x

        self.decoder = k.models.Model(decoder_input, decoder_output)

        # ---------- MERGE ---------------

        model_input = encoder_input
        model_output = self.decoder(encoder_output)

        self.model = k.models.Model(model_input, model_output)

        self.encoder.summary()
        self.decoder.summary()


def load_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_train = x_train.reshape(x_train.shape + (1,))

    x_test = x_test.astype('float32') / 255.
    x_test = x_test.reshape(x_test.shape + (1,))

    return (x_train, y_train), (x_test, y_test)


(x_train, y_train), (x_test, y_test) = load_mnist()

ae = AutoEncoder(input_dim=(28, 28, 1),
                 encoder_conv_filters=[32, 64, 64, 64],
                 encoder_conv_kernels=[3, 3, 3, 3],
                 encoder_conv_strides=[1, 2, 2, 1],
                 decoder_conv_t_filters=[64, 64, 64, 32],
                 decoder_conv_t_kernels=[3, 3, 3, 3],
                 decoder_conv_t_strides=[1, 2, 2, 1],
                 z_dim=2)
