import numpy as np
import keras as k
import os
import pickle as pkl
import matplotlib.pyplot as plt
from keras.datasets import mnist


def load_camel_dataset():
    x = np.load("/media/4TB/datasets/QuickDraw/full_numpy_bitmap_camel.npy")

    x = (x.astype('float32') - 127.5) / 127.5

    x = x.reshape(x.shape[0], 28, 28, 1)

    return x


def load_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = (x_train.astype('float32') - 127.5) / 127.5
    x_train = x_train.reshape(x_train.shape + (1,))

    return x_train


class GAN():
    def __init__(self,
                 input_dim,
                 discriminator_conv_filters,
                 discriminator_conv_kernels,
                 discriminator_conv_strides,
                 discriminator_bn_momentum,
                 discriminator_activation,
                 discriminator_dropout,
                 discriminator_lr,
                 generator_first_dense_layer_size,
                 generator_upsample,
                 generator_conv_filters,
                 generator_conv_kernels,
                 generator_conv_strides,
                 generator_bn_momemtum,
                 generator_activation,
                 generator_dropout,
                 generator_lr,
                 optimizer,
                 z_dim):
        self.input_dim = input_dim
        self.discriminator_conv_filters = discriminator_conv_filters
        self.discriminator_conv_kernels = discriminator_conv_kernels
        self.discriminator_conv_strides = discriminator_conv_strides
        self.discriminator_bn_momentum = discriminator_bn_momentum
        self.discriminator_activation = discriminator_activation
        self.discriminator_dropout = discriminator_dropout
        self.discriminator_lr = discriminator_lr
        self.generator_first_dense_layer_size = generator_first_dense_layer_size
        self.generator_upsample = generator_upsample
        self.generator_conv_filters = generator_conv_filters
        self.generator_conv_kernels = generator_conv_kernels
        self.generator_conv_strides = generator_conv_strides
        self.generator_bn_momemtum = generator_bn_momemtum
        self.generator_activation = generator_activation
        self.generator_dropout = generator_dropout
        self.generator_lr = generator_lr
        self.optimizer = optimizer
        self.z_dim = z_dim

        self.epoch = 0

        self.d_losses = []
        self.g_losses = []

        self._build_discriminator()
        self._build_generator()

        self._build_gan()

    def get_activation(self, activation):
        if activation == "leaky_relu":
            layer = k.layers.LeakyReLU(alpha=0.2)
        else:
            layer = k.layers.Activation(activation)

        return layer

    def _build_discriminator(self):

        input = k.layers.Input(shape=self.input_dim, name='discriminator_input')

        x = input

        for i in range(len(self.discriminator_conv_filters)):
            x = k.layers.Conv2D(filters=self.discriminator_conv_filters[i],
                                kernel_size=self.discriminator_conv_kernels[i],
                                strides=self.discriminator_conv_strides[i],
                                padding="same",
                                name='discriminator_conv_' + str(i),
                                kernel_initializer=k.initializers.random_normal(mean=0., stddev=0.02))(x)

            if self.discriminator_bn_momentum and i > 0:
                x = k.layers.BatchNormalization(momentum=self.discriminator_bn_momentum)(x)

            x = self.get_activation(self.discriminator_activation)(x)

            if self.discriminator_dropout:
                x = k.layers.Dropout(rate=self.discriminator_dropout)(x)

        x = k.layers.Flatten()(x)

        output = k.layers.Dense(1, activation='sigmoid', kernel_initializer=k.initializers.random_normal(mean=0., stddev=0.02))(x)

        self.discriminator = k.models.Model(input, output)

    def _build_generator(self):
        input = k.layers.Input(shape=(self.z_dim,), name="generator_input")

        x = input

        x = k.layers.Dense(np.prod(self.generator_first_dense_layer_size), kernel_initializer=k.initializers.random_normal(mean=0., stddev=0.02))(x)

        if self.generator_bn_momemtum:
            x = k.layers.BatchNormalization(momentum=self.generator_bn_momemtum)(x)

        x = self.get_activation(self.generator_activation)(x)

        x = k.layers.Reshape(self.generator_first_dense_layer_size)(x)

        if self.generator_dropout:
            x = k.layers.Dropout(rate=self.generator_dropout)(x)

        for i in range(len(self.generator_conv_filters)):

            if self.generator_upsample[i] == 2:
                x = k.layers.UpSampling2D()(x)
                x = k.layers.Conv2D(filters=self.generator_conv_filters[i],
                                    kernel_size=self.generator_conv_kernels[i],
                                    padding="same",
                                    name='generator_conv_' + str(i),
                                    kernel_initializer=k.initializers.random_normal(mean=0., stddev=0.02))(x)
            else:
                x = k.layers.Conv2DTranspose(filters=self.generator_conv_filters[i],
                                             kernel_size=self.generator_conv_kernels[i],
                                             strides=self.generator_conv_strides[i],
                                             padding="same",
                                             name='generator_conv_' + str(i),
                                             kernel_initializer=k.initializers.random_normal(mean=0., stddev=0.02))(x)

            if i < len(self.generator_conv_filters) - 1:

                if self.generator_bn_momemtum:
                    x = k.layers.BatchNormalization(momentum=self.generator_bn_momemtum)(x)

                x = self.get_activation(self.generator_activation)(x)

            else:
                x = k.layers.Activation('tanh')(x)

        self.generator = k.models.Model(input, x)

    def get_optimizer(self, lr):
        if self.optimizer == "adam":
            optimizer = k.optimizers.Adam(lr=lr, beta_1=0.5)
        elif self.optimizer == "rmsprop":
            optimizer = k.optimizers.RMSprop(lr=lr)
        else:
            optimizer = k.optimizers.Adam(lr=lr)

        return optimizer

    def set_trainable(self, model, isTrainable):
        model.trainable = isTrainable
        for layer in model.layers:
            layer.trainable = isTrainable

    def _build_gan(self):

        self.discriminator.compile(optimizer=self.get_optimizer(self.discriminator_lr),
                                   loss="binary_crossentropy", metrics=["accuracy"])

        self.set_trainable(self.discriminator, False)

        model_input = k.layers.Input(shape=(self.z_dim,), name="model_input")
        model_output = self.discriminator(self.generator(model_input))

        self.model = k.models.Model(model_input, model_output)

        self.model.compile(optimizer=self.get_optimizer(self.generator_lr), loss="binary_crossentropy", metrics=["accuracy"])

        self.set_trainable(self.discriminator, True)

    def train_discriminator(self, x_train, batch_size):
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        idx = np.random.randint(0, x_train.shape[0], batch_size)
        true_imgs = x_train[idx]

        noise = np.random.normal(0, 1, (batch_size, self.z_dim))
        gen_imgs = self.generator.predict(noise)

        d_loss_real, d_acc_real = self.discriminator.train_on_batch(true_imgs, valid)
        d_loss_fake, d_acc_fake = self.discriminator.train_on_batch(gen_imgs, fake)

        d_loss = 0.5 * (d_loss_real + d_loss_fake)
        d_acc = 0.5 * (d_acc_real + d_acc_fake)

        return [d_loss, d_loss_real, d_loss_fake, d_acc, d_acc_real, d_acc_fake]

    def train_generator(self, batch_size):

        valid = np.ones((batch_size, 1))  # Why ????

        noise = np.random.normal(0, 1, (batch_size, self.z_dim))
        return self.model.train_on_batch(noise, valid)

    def train(self, x_train, batch_size, epochs, print_every_n_batches=50):

        for epoch in range(epochs):
            d = self.train_discriminator(x_train, batch_size)
            g = self.train_generator(batch_size)

            self.d_losses.append(d)
            self.g_losses.append(g)

            if epoch % print_every_n_batches == 0:
                print("%d [D loss: (%.3f)(R %.3f, F %.3f)] [D acc: (%.3f)(%.3f, %.3f)] [G loss: %.3f] [G acc: %.3f]" % (
                    epoch, d[0], d[1], d[2], d[3], d[4], d[5], g[0], g[1]))
                self.sample_images("output/img")
                self.model.save_weights(os.path.join("output", 'weights/weights.h5'))
                self.save_model("output/model")

            self.epoch += 1

    def save_model(self, run_folder):
        self.model.save(os.path.join(run_folder, 'model.h5'))
        self.discriminator.save(os.path.join(run_folder, 'discriminator.h5'))
        self.generator.save(os.path.join(run_folder, 'generator.h5'))
        pkl.dump(self, open(os.path.join(run_folder, "obj.pkl"), "wb"))

    def sample_images(self, run_folder):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.z_dim))
        gen_imgs = self.generator.predict(noise)

        gen_imgs = 0.5 * (gen_imgs + 1)
        gen_imgs = np.clip(gen_imgs, 0, 1)

        fig, axs = plt.subplots(r, c, figsize=(15, 15))
        cnt = 0

        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(np.squeeze(gen_imgs[cnt, :, :, :]), cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig(os.path.join(run_folder, "sample_%d.png" % self.epoch))
        plt.close()


if __name__ == '__main__':
    # x_train = load_camel_dataset()
    x_train = load_mnist()

    mode = "build"

    gan = GAN(input_dim=(28, 28, 1)
              , discriminator_conv_filters=[64, 64, 128, 128]
              , discriminator_conv_kernels=[5, 5, 5, 5]
              , discriminator_conv_strides=[2, 2, 2, 1]
              , discriminator_bn_momentum=None
              , discriminator_activation='relu'
              , discriminator_dropout=0.4
              , discriminator_lr=0.0008
              , generator_first_dense_layer_size=(7, 7, 64)
              , generator_upsample=[2, 2, 1, 1]
              , generator_conv_filters=[128, 64, 64, 1]
              , generator_conv_kernels=[5, 5, 5, 5]
              , generator_conv_strides=[1, 1, 1, 1]
              , generator_bn_momemtum=0.9
              , generator_activation='relu'
              , generator_dropout=None
              , generator_lr=0.0004
              , optimizer='rmsprop'
              , z_dim=100)

    if mode == 'build':
        gan.discriminator.summary()
        gan.generator.summary()

        BATCH_SIZE = 128
        EPOCHS = 6000
        PRINT_EVERY_N_BATCHES = 500

        gan.train(
            x_train
            , batch_size=BATCH_SIZE
            , epochs=EPOCHS
            , print_every_n_batches=PRINT_EVERY_N_BATCHES
        )
    else:
        gan.load_weights(os.path.join("output", 'weights/weights.h5'))

    fig = plt.figure()
    plt.plot([x[0] for x in gan.d_losses], color='black', linewidth=0.25)

    plt.plot([x[1] for x in gan.d_losses], color='green', linewidth=0.25)
    plt.plot([x[2] for x in gan.d_losses], color='red', linewidth=0.25)
    plt.plot([x[0] for x in gan.g_losses], color='orange', linewidth=0.25)

    plt.xlabel('batch', fontsize=18)
    plt.ylabel('loss', fontsize=16)

    plt.xlim(0, 2000)
    plt.ylim(0, 2)

    plt.show()

    fig = plt.figure()
    plt.plot([x[3] for x in gan.d_losses], color='black', linewidth=0.25)
    plt.plot([x[4] for x in gan.d_losses], color='green', linewidth=0.25)
    plt.plot([x[5] for x in gan.d_losses], color='red', linewidth=0.25)
    plt.plot([x[1] for x in gan.g_losses], color='orange', linewidth=0.25)

    plt.xlabel('batch', fontsize=18)
    plt.ylabel('accuracy', fontsize=16)

    plt.xlim(0, 2000)

    plt.show()
