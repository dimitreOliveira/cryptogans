# Dependencies
import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf


# Visualization
def plot_multiple_images(images, n_cols, epoch):
    n_cols = n_cols or len(images)
    n_rows = (len(images) - 1) // n_cols + 1
    if images.shape[-1] == 1:
        images = np.squeeze(images, axis=-1)
    plt.figure(figsize=(n_cols, n_rows))
    for index, image in enumerate(images):
        image = ((image + 1) / 2) # scale back
        plt.subplot(n_rows, n_cols, index + 1)
        plt.imshow(image, cmap="binary")
        plt.axis("off")
    plt.savefig(f'{args.images_output_path}epoch_{epoch}.png')


# Processing
def decode_img(inputs):
    file_path = inputs["paths"]
    img = tf.io.read_file(file_path)
    img = tf.io.decode_png(img, channels=args.image_channels)
    img = tf.image.resize(img, (args.image_size, args.image_size))
    return img

def pre_process(img):
    img = (img / 255.) * 2 - 1 # scale images to be between -1 and 1
    return img


# Data
def get_dataset(inputs):
    if type(inputs) == dict:
        dataset = tf.data.Dataset.from_tensor_slices(inputs) # load from paths
    else:
        filepaths = glob.glob(f"{inputs}/*") # get images paths
        dataset = tf.data.Dataset.list_files(filepaths) # load directly from images
    
    dataset = (dataset.map(decode_img, num_parallel_calls=tf.data.AUTOTUNE)
                      .map(pre_process, num_parallel_calls=tf.data.AUTOTUNE)
                      .shuffle(1024)
                      # .batch(batch_size, drop_remainder=True)
                      # .prefetch(tf.data.AUTOTUNE)
              )
    return dataset


# Training
def discriminator_loss(real_output, fake_output):
  real_loss = cross_entropy(tf.ones_like(real_output), real_output)
  fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
  total_loss = real_loss + fake_loss
  return total_loss

def generator_loss(fake_output):
  return cross_entropy(tf.ones_like(fake_output), fake_output)

@tf.function
def train_step(images, batch_size, codings_size):
  noise = tf.random.normal([batch_size, codings_size])

  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    generated_images = generator(noise, training=True)

    real_output = discriminator(images, training=True)
    fake_output = discriminator(generated_images, training=True)

    gen_loss = generator_loss(fake_output)
    disc_loss = discriminator_loss(real_output, fake_output)

  gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
  disc_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

  gen_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))
  disc_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))

def train(dataset, epochs, batch_size, codings_size):
  for epoch in range(epochs):
    for image_batch in dataset:
      train_step(image_batch, batch_size, codings_size)

    print(f"Epoch {epoch+1}/{epochs}")
    if args.plot_images:
        noise = tf.random.normal([batch_size, codings_size])
        display_images = generator(noise, training=False)
        plot_multiple_images(display_images, 8, epoch)


# Model
def generator_fn():
    inputs = tf.keras.layers.Input(shape=(args.codings_size), dtype=tf.float32, name='inputs')
    x = tf.keras.layers.Dense(6 * 6 * 256, use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    x = tf.keras.layers.Reshape([6, 6, 256])(x)

    x = tf.keras.layers.Conv2DTranspose(128, kernel_size=5, strides=1, padding="same", 
                          use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)

    x = tf.keras.layers.Conv2DTranspose(64, kernel_size=5, strides=2, padding="same", 
                          use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)

    outputs = tf.keras.layers.Conv2DTranspose(args.image_channels, kernel_size=5, strides=2, 
                                padding="same", activation="tanh", 
                                use_bias=False)(x)

    return tf.keras.models.Model(inputs=inputs, outputs=outputs, name="generator")

def discriminator_fn():
    inputs = tf.keras.layers.Input(shape=((args.image_size, args.image_size, args.image_channels)), dtype=tf.float32, name='inputs')
    x = tf.keras.layers.Conv2D(64, kernel_size=4, strides=2, padding="same", 
                          activation=tf.keras.layers.LeakyReLU(0.2))(inputs)
    x = tf.keras.layers.Dropout(0.4)(x)

    x = tf.keras.layers.Conv2D(128, kernel_size=4, strides=2, padding="same", 
                          activation=tf.keras.layers.LeakyReLU(0.2))(x)
    x = tf.keras.layers.Dropout(0.4)(x)

    x = tf.keras.layers.Conv2D(256, kernel_size=3, strides=2, padding="same", 
                          activation=tf.keras.layers.LeakyReLU(0.2))(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    return tf.keras.models.Model(inputs=inputs, outputs=outputs, name="discriminator")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="./cryptogans/data/attributes.csv", help="Path to dataset (attributes.csv)")
    parser.add_argument("--images_path", default="./cryptogans/data/images/", help="Path to images")
    parser.add_argument("--model_output_path", default="./models/", help="Path to output the generator model")
    parser.add_argument("--images_output_path", default="./gen_images/", help="Path to output the generatored images during training")
    parser.add_argument("--codings_size", type=int, default=100, help="Size of the latent z vector")
    parser.add_argument("--image_size", type=int, default=24, help="images size")
    parser.add_argument("--image_channels", type=int, default=4, help="images channels")
    parser.add_argument("--batch_size", type=int, default=16, help="Input batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--plot_images", type=bool, default=False, help="Plot images after each epoch during training")    
    args = parser.parse_args()
    print(args)

    if args.images_output_path and (os.path.exists(args.images_output_path) == False):
        print(f"Saving generated images during training at: {args.images_output_path}")
        os.mkdir(args.images_output_path)

    print("Loading the dataset...")
    df = pd.read_csv(args.data_path)
    df.id = df.id.apply(lambda x: f"{args.images_path}punk{x:03d}.png")

    print("Creating TensorFlow dataset...")
    sampling_ds = [get_dataset({"paths": df.id})]
    dataset = (tf.data.Dataset.sample_from_datasets(sampling_ds)
                              .batch(args.batch_size)
                              .prefetch(tf.data.AUTOTUNE))
    
    # Generator
    generator = generator_fn()
    print("Generator architecture:")
    generator.summary()

    # Discriminator
    discriminator = discriminator_fn()
    print("Discriminator architecture:")
    discriminator.summary()

    gen_optimizer = tf.keras.optimizers.RMSprop()
    disc_optimizer = tf.keras.optimizers.RMSprop()
    cross_entropy = tf.keras.losses.BinaryCrossentropy()

    # Training
    print("Training model...")
    train(dataset, args.epochs, args.batch_size, args.codings_size)

    print(f"Saving model at: {args.model_output_path}...")
    generator.save(args.model_output_path)
