import glob
import random
import numpy as np
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt


@st.cache(allow_output_mutation=True)
def load_model():
    generator = tf.keras.models.load_model("./models/")
    return generator

def generate(n_images, generator, seed=42, codings_size=100):
    noise = tf.random.normal(shape=[n_images, codings_size], seed=seed)
    generated_images = generator(noise, training=False)
    return generated_images.numpy()

def get_captcha(images, indexes):
    fake_idx = []
    fig, axis = plt.subplots(3, 3, figsize=(10, 10))
    axis = axis.flatten()

    for ax_idx, idx in enumerate(indexes):
        if indexes[idx] < n_fake:
            fake_idx.append(idx)
        axis[ax_idx].imshow(images[idx])
        axis[ax_idx].set_title(f"Punk {ax_idx+1}")
        axis[ax_idx].axis('off')
    return fig, fake_idx

def get_challenge():
    seed = random.randint(0, 1000)
    real_idx = np.random.randint(low=0, high=len(real_files), size=n_real)
    real_samples = np.array([plt.imread(real_files[idx]) for idx in real_idx])
    generated_samples = generate(n_fake, generator, seed)
    captcha_samples = np.concatenate((generated_samples, real_samples))

    captcha_idx = [x for x in range(len(captcha_samples))]
    random.shuffle(captcha_idx)

    fig, fake_idx = get_captcha(captcha_samples, captcha_idx)
    correct_answer = [f"Punk {x+1}" for x in fake_idx]
    correct_answer.sort()
    st.session_state.session_captcha = fig
    st.session_state.correct = correct_answer

# Session captcha
if 'session_captcha' not in st.session_state:
    st.session_state['session_captcha'] = None
if 'correct' not in st.session_state:
    st.session_state['correct'] = []

n_real = 6
n_fake = 3
n_samples = n_real + n_fake
real_images_path = "./data/images"
real_files = glob.glob(f"{real_images_path}/*")
generator = load_model()

st.title("CryptoPunk Captcha")
st.header("Are you a human?")
st.subheader("Choose the 3 fake CryptoPunks")

if st.session_state.session_captcha is None:
    get_challenge()

st.pyplot(st.session_state.session_captcha) # Display captcha
options = st.multiselect('Choose',[f"Punk {x+1}" for x in range(n_samples)]) # Selected options
to_verify = st.button("Verify")

if to_verify:
    options.sort()
    verified = st.session_state.correct == options
    if verified:
        st.write("Correct")
    else:
        st.write("Wrong")
        get_challenge() # Refresh challenge
        st.button("Try again?")
