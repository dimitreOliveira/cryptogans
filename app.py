import gradio as gr
import tensorflow as tf
import matplotlib.pyplot as plt


n_images = 36
codings_size = 100
generator = tf.keras.models.load_model("./models/")


def post_process(image):
    image = ((image + 1) / 2) # scale back
    return image


def generate(seed):
    noise = tf.random.normal(shape=[n_images, codings_size], seed=seed)
    generated_images = generator(noise, training=False)

    fig = plt.figure(figsize=(10, 10))
    for i in range(generated_images.shape[0]):
        plt.subplot(6, 6, i+1)
        plt.imshow(post_process(generated_images[i, :, :, :]), cmap="binary")
        plt.axis("off")
    return fig


gr.Interface(fn=generate, 
            inputs=[gr.inputs.Slider(label='Seed', minimum=0, maximum=1000, default=42)], 
            outputs=gr.outputs.Image(type="plot"), 
            title="CryptoGAN", 
            description="These CryptoPunks do not exist. Generate your own CryptoPunks").launch()