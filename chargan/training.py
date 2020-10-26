"""
Living example for documentation/experimentation
"""
from .preprocessing import CharTokenizer
from .models import CharAutoencoder
from datasets import load_dataset
import numpy as np
import tensorflow as tf

# load AG News and fit a CharTokenizer on it
ag = load_dataset("ag_news")
ct = CharTokenizer()
ct.fit(ag["train"]["text"], progbar=True)

# Create our character-level autoencoder
ae = CharAutoencoder(ct)

# tokenize an example dataset
tok = np.array(ct.tokenize(ag["train"]["text"][:1000]))
tok = tf.keras.preprocessing.sequence.pad_sequences(
    tok, maxlen=100, dtype="int32", padding="pre", truncating="pre", value=0.0
)

# compile our model
scce = tf.keras.losses.SparseCategoricalCrossentropy()
ae.compile("adam", scce)

# fit the autoencoder
ae.fit(tok, tok, epochs=100, batch_size=64)