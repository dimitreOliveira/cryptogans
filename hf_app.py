import gradio as gr
import tensorflow as tf
import matplotlib.pyplot as plt
from huggingface_hub import from_pretrained_keras


n_images = 36
codings_size = 100
generator = from_pretrained_keras("huggan/crypto-gan")


def generate(seed):
    noise = tf.random.normal(shape=[n_images, codings_size], seed=seed)
    generated_images = generator(noise, training=False)

    fig = plt.figure(figsize=(10, 10))
    for i in range(generated_images.shape[0]):
        plt.subplot(6, 6, i+1)
        plt.imshow(generated_images[i, :, :, :])
        plt.axis("off")
    return fig


gr.Interface(fn=generate, 
            inputs=[gr.inputs.Slider(label='Seed', minimum=0, maximum=1000, default=42)], 
            outputs=gr.outputs.Image(type="plot"), 
            title="CryptoGAN", 
            description="These CryptoPunks do not exist.").launch()
