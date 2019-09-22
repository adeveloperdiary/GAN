import numpy as np
import keras as k
import os
import pickle as pkl
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from keras.utils.training_utils import multi_gpu_model
import tensorflow as tf


def load_cifar(label):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    train_mask = [y[0] == label for y in y_train]
    test_mask = [y[0] == label for y in y_test]

    x_data = np.concatenate([x_train[train_mask], x_test[test_mask]])
    y_data = np.concatenate([y_train[train_mask], y_test[test_mask]])

    x_data = (x_data.astype('float32') - 127.5) / 127.5

    return x_data, y_data


class WGAN():
    def __init__(self,
                 input_dim,
                 critic_conv_filters,
                 critic_conv_kernels,
                 critic_conv_strides,
                 critic_bn_momentum,
                 critic_activation,
                 critic_dropout,
                 critic_lr,
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
        self.critic_conv_filters = critic_conv_filters
        self.critic_conv_kernels = critic_conv_kernels
        self.critic_conv_strides = critic_conv_strides
        self.critic_bn_momentum = critic_bn_momentum
        self.critic_activation = critic_activation
        self.critic_dropout = critic_dropout
        self.critic_lr = critic_lr
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

    def get_activation(self, activation):
        if activation == "leaky_relu":
            layer = k.layers.LeakyReLU(alpha=0.2)
        else:
            layer = k.layers.Activation(activation)

        return layer

    def _build_critic(self):
        input = k.layers.Input(shape=self.input_dim, name='critic_input')

        x = input

        for i in range(len(self.critic_conv_filters)):
            x = k.layers.Conv2D(filters=self.critic_conv_filters[i],
                                kernel_size=self.critic_conv_kernels[i],
                                strides=self.critic_conv_strides[i],
                                padding="same",
                                name='critic_conv_' + str(i),
                                kernel_initializer=k.initializers.random_normal(mean=0., stddev=0.02))(x)

            if self.critic_bn_momentum and i > 0:
                x = k.layers.BatchNormalization(momentum=self.critic_bn_momentum)(x)

            x = self.get_activation(self.critic_activation)(x)

            if self.critic_dropout:
                x = k.layers.Dropout(rate=self.critic_dropout)(x)

        x = k.layers.Flatten()(x)

        # --------- WGAN Changes ----------

        output = k.layers.Dense(1, activation=None,  # No Activation for WGAN
                                kernel_initializer=k.initializers.random_normal(mean=0., stddev=0.02))(x)

        return k.models.Model(input, output)

    def _build_generator(self):
        input = k.layers.Input(shape=(self.z_dim,), name="generator_input")

        x = input

        x = k.layers.Dense(np.prod(self.generator_first_dense_layer_size), kernel_initializer=k.initializers.random_normal(mean=0., stddev=0.02))(
            x)

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

        return k.models.Model(input, x)

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

    # wasserstein loss function
    def wasserstein_loss(self, y_true, y_pred):
        return -k.backend.mean(y_true * y_pred)

    # -------- Need to change the loss function ------
    def _build_gan(self):

        self.critic.compile(optimizer=self.get_optimizer(self.critic_lr),
                            loss=self.wasserstein_loss  # wasserstein loss function
                            # ,metrics=["accuracy"]
                            )

        self.set_trainable(self.critic, False)

        model_input = k.layers.Input(shape=(self.z_dim,), name="model_input")
        model_output = self.critic(self.generator(model_input))

        self.model = k.models.Model(model_input, model_output)

        self.model.compile(optimizer=self.get_optimizer(self.generator_lr),
                           loss=self.wasserstein_loss
                           # ,metrics=["accuracy"]
                           )

        self.set_trainable(self.critic, True)

    def train_critic(self, x_train, batch_size, clip_threshold):
        valid = np.ones((batch_size, 1))
        fake = - np.ones((batch_size, 1))  # Set labels to -1,1

        idx = np.random.randint(0, x_train.shape[0], batch_size)
        true_imgs = x_train[idx]

        noise = np.random.normal(0, 1, (batch_size, self.z_dim))
        gen_imgs = self.generator.predict(noise)

        d_loss_real = self.critic.train_on_batch(true_imgs, valid)
        d_loss_fake = self.critic.train_on_batch(gen_imgs, fake)

        d_loss = 0.5 * (d_loss_real + d_loss_fake)

        # ------ Weight Clipping ------

        for l in self.critic.layers:
            weights = l.get_weights()
            weights = [np.clip(w, -clip_threshold, clip_threshold) for w in weights]
            l.set_weights(weights)

        return [d_loss, d_loss_real, d_loss_fake]

    def train_generator(self, batch_size):

        valid = np.ones((batch_size, 1))

        noise = np.random.normal(0, 1, (batch_size, self.z_dim))
        return self.model.train_on_batch(noise, valid)

    def train(self, x_train, batch_size, epochs, print_every_n_batches=50, n_critic=5, clip_threshold=0.01):

        # with tf.device("/cpu:0"):
        self.critic = self._build_critic()
        self.generator = self._build_generator()

        # self.critic = multi_gpu_model(critic, gpus=2)
        # self.generator = multi_gpu_model(generator, gpus=2)

        self._build_gan()

        for epoch in range(epochs):

            for _ in range(n_critic):
                d_loss = self.train_critic(x_train, batch_size, clip_threshold)

            g_loss = self.train_generator(batch_size)

            self.d_losses.append(d_loss)
            self.g_losses.append(g_loss)

            if epoch % print_every_n_batches == 0:
                print("%d [D loss: (%.3f)(R %.3f, F %.3f)]  [G loss: %.3f] " % (epoch, d_loss[0], d_loss[1], d_loss[2], g_loss))
                self.sample_images("output/img")
                self.model.save_weights(os.path.join("output", 'weights/weights.h5'))
                self.save_model("output/model")

            self.epoch += 1

    def save_model(self, run_folder):
        self.model.save(os.path.join(run_folder, 'model.h5'))
        self.critic.save(os.path.join(run_folder, 'critic.h5'))
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
    # x_train = load_mnist()

    (x_train, y_train) = load_cifar(1)

    mode = "build"

    gan = WGAN(input_dim=(32, 32, 3)
               , critic_conv_filters=[32, 64, 128, 128]
               , critic_conv_kernels=[5, 5, 5, 5]
               , critic_conv_strides=[2, 2, 2, 1]
               , critic_bn_momentum=None
               , critic_activation='leaky_relu'
               , critic_dropout=None
               , critic_lr=0.00005
               , generator_first_dense_layer_size=(4, 4, 128)
               , generator_upsample=[2, 2, 2, 1]
               , generator_conv_filters=[128, 64, 32, 3]
               , generator_conv_kernels=[5, 5, 5, 5]
               , generator_conv_strides=[1, 1, 1, 1]
               , generator_bn_momemtum=0.8
               , generator_activation='leaky_relu'
               , generator_dropout=None
               , generator_lr=0.00005
               , optimizer='rmsprop'
               , z_dim=100)

    if mode == 'build':
        # gan.critic.summary()
        # gan.generator.summary()

        BATCH_SIZE = 512
        EPOCHS = 6000
        PRINT_EVERY_N_BATCHES = 200
        N_CRITIC = 5
        CLIP_THRESHOLD = 0.01

        gan.train(
            x_train
            , batch_size=BATCH_SIZE
            , epochs=EPOCHS
            , print_every_n_batches=PRINT_EVERY_N_BATCHES
            , n_critic=N_CRITIC
            , clip_threshold=CLIP_THRESHOLD
        )
    else:
        gan.load_weights(os.path.join("output", 'weights/weights.h5'))

    gan.sample_images("output/img")

    fig = plt.figure()
    plt.plot([x[0] for x in gan.d_losses], color='black', linewidth=0.25)

    plt.plot([x[1] for x in gan.d_losses], color='green', linewidth=0.25)
    plt.plot([x[2] for x in gan.d_losses], color='red', linewidth=0.25)
    plt.plot(gan.g_losses, color='orange', linewidth=0.25)

    plt.xlabel('batch', fontsize=18)
    plt.ylabel('loss', fontsize=16)

    # plt.xlim(0, 2000)
    # plt.ylim(0, 2)

    plt.show()


    def compare_images(img1, img2):
        return np.mean(np.abs(img1 - img2))


    r, c = 5, 5

    idx = np.random.randint(0, x_train.shape[0], BATCH_SIZE)
    true_imgs = (x_train[idx] + 1) * 0.5

    fig, axs = plt.subplots(r, c, figsize=(15, 15))
    cnt = 0

    for i in range(r):
        for j in range(c):
            axs[i, j].imshow(true_imgs[cnt], cmap='gray_r')
            axs[i, j].axis('off')
            cnt += 1
    fig.savefig(os.path.join("output", "img/real.png"))
    plt.close()

    r, c = 5, 5
    noise = np.random.normal(0, 1, (r * c, gan.z_dim))
    gen_imgs = gan.generator.predict(noise)

    gen_imgs = 0.5 * (gen_imgs + 1)

    fig, axs = plt.subplots(r, c, figsize=(15, 15))
    cnt = 0

    for i in range(r):
        for j in range(c):
            axs[i, j].imshow(np.squeeze(gen_imgs[cnt, :, :, :]), cmap='gray_r')
            axs[i, j].axis('off')
            cnt += 1
    fig.savefig(os.path.join("output", "img/sample.png"))
    plt.close()

    fig, axs = plt.subplots(r, c, figsize=(15, 15))
    cnt = 0

    for i in range(r):
        for j in range(c):
            c_diff = 99999
            c_img = None
            for k_idx, k in enumerate((x_train + 1) * 0.5):

                diff = compare_images(gen_imgs[cnt, :, :, :], k)
                if diff < c_diff:
                    c_img = np.copy(k)
                    c_diff = diff
            axs[i, j].imshow(c_img, cmap='gray_r')
            axs[i, j].axis('off')
            cnt += 1

    fig.savefig(os.path.join("output", "img/sample_closest.png"))
    plt.close()
