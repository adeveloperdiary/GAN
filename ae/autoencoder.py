from keras.datasets import mnist
import keras as k
import numpy as np
import os
import matplotlib.pyplot as plt


class MonitoringCallback(k.callbacks.Callback):
    def __init__(self, initial_epoch, ae):
        self.epoch = initial_epoch
        self.ae = ae

    def on_batch_end(self, batch, logs={}):
        if batch % 60000 == 0:
            # Sample values from standard normal.
            z_new = np.random.normal(size=(1, self.ae.z_dim))

            y_pred = self.ae.decoder.predict(np.array(z_new))

            reconstruction = y_pred[0].squeeze()

            file_path = os.path.join("output", 'images', 'img_' + str(self.epoch).zfill(3) + '_' + str(batch) + '.jpg')

            if len(reconstruction.shape) == 2:
                plt.imsave(file_path, reconstruction, cmap='gray_r')
            else:
                plt.imsave(file_path, reconstruction)

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch += 1


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

    def compile(self, lr):
        self.lr = lr

        optimizer = k.optimizers.Adam(lr=lr)

        def r_loss(y_true, y_pred):
            return k.backend.mean(k.backend.square(y_true - y_pred), axis=[1, 2, 3])

        self.model.compile(optimizer=optimizer, loss=r_loss)

    def _step_decay_schedule(self, initial_lr, decay_factor=0.5, step_size=1):
        def schedule(epoch):
            new_lr = initial_lr * (decay_factor ** np.floor(epoch / step_size))
            return new_lr

        return k.callbacks.LearningRateScheduler(schedule)

    def train(self, x_train, batch_size, epochs, initial_epoch=0, lr_decay=1):

        monitor = MonitoringCallback(initial_epoch, self)
        lr_schedule = self._step_decay_schedule(self.lr, decay_factor=lr_decay)

        checkpoint = k.callbacks.ModelCheckpoint(os.path.join("output", "weights/weights.h5"), save_weights_only=True, verbose=1)

        callback_list = [checkpoint, monitor, lr_schedule]

        self.model.fit(x_train,
                       x_train,
                       batch_size=batch_size,
                       shuffle=True,
                       epochs=epochs,
                       initial_epoch=initial_epoch,
                       callbacks=callback_list)

    def load_weights(self, path):
        self.model.load_weights(filepath=path)


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
                 decoder_conv_t_filters=[64, 64, 32, 1],
                 decoder_conv_t_kernels=[3, 3, 3, 3],
                 decoder_conv_t_strides=[1, 2, 2, 1],
                 z_dim=2)

MODE = 'TEST'

if MODE == 'TRAIN':
    LEARNING_RATE = 0.0005
    BATCH_SIZE = 32
    INITIAL_EPOCH = 0

    ae.compile(lr=LEARNING_RATE)
    ae.train(x_train, batch_size=BATCH_SIZE, epochs=100, initial_epoch=INITIAL_EPOCH)
else:
    ae.load_weights("output/weights/weights.h5")

    n_to_show = 10
    example_idx = np.random.choice(range(len(x_test)), n_to_show)
    example_images = x_test[example_idx]

    z_points = ae.encoder.predict(example_images)

    reconst_images = ae.decoder.predict(z_points)

    fig = plt.figure(figsize=(15, 3))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    for i in range(n_to_show):
        img = example_images[i].squeeze()
        ax = fig.add_subplot(2, n_to_show, i + 1)
        ax.axis('off')
        ax.text(0.5, -0.35, str(np.round(z_points[i], 1)), fontsize=10, ha='center', transform=ax.transAxes)
        ax.imshow(img, cmap='gray_r')

    for i in range(n_to_show):
        img = reconst_images[i].squeeze()
        ax = fig.add_subplot(2, n_to_show, i + n_to_show + 1)
        ax.axis('off')
        ax.imshow(img, cmap='gray_r')

    plt.savefig('myfig.png')
