# CryptoGANs
This is a simple TensorFlow implementation of a DCGAN to generate CryptoPunks

<img src="./assets/gen_samples.png" width="350" height="350">

> CryptoPunks generated by this work

This repository was created during the HuggingFace's [HugGAN sprint](https://github.com/huggingface/community-events/tree/main/huggan). You can also checkout its related entries:
- [Model card (HF hub)](https://huggingface.co/huggan/crypto-gan)
- [CryptoPunks generation (HF space)](https://huggingface.co/spaces/huggan/crypto-gan)
- [CryptoPunk Captcha (HF space)](https://huggingface.co/spaces/huggan/cryptopunk-captcha)


# Usage

## Train
```bash
python3 train.py
```

## Inference using Gradio (CryptoPunk generation)
<img src="./assets/gradio_inference.png" width="850" height="300">


### Using your own model
This requires you to have a `models` folder at root with a generator model.
```bash
python3 app.py
```

### Using a model from HuggingFace hub
This will download the generator model from HuggingFace hub and serve it.
```bash
python3 hf_app.py
```

## Inference using Streamlit (CryptoPunk Captcha)
<img src="./assets/streamlit_inference.png" width="350" height="500">

### Using your own model
This requires you to have a `models` folder at root with a generator model.
```bash
streamlit run captcha_app.py
```

### Using a model from HuggingFace hub
This will download the generator model from HuggingFace hub and serve it.
```bash
streamlit run hf_captcha_app.py
```

# References
- [CryptoPunks GAN](https://github.com/teddykoker/cryptopunks-gan)
- [Deep Convolutional Generative Adversarial Network](https://www.tensorflow.org/tutorials/generative/dcgan)