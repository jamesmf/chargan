"""
Living example for documentation/experimentation
"""
import re
from typing import List
from .preprocessing import CharTokenizer, CharAEGenerator
from .models import CharAutoencoder
from datasets import load_dataset
import numpy as np
import tensorflow as tf


def decoding_test(
    test_strs: List[str], tokenizer: CharTokenizer, autoencoder: CharAutoencoder
):
    """
    Print out an example of how well the autoencoder is doing
    """
    val_gen = CharAEGenerator(
        tokenizer, len(test_strs), min_sample_len=128, max_sample_len=256
    )
    tokenized = next(val_gen(test_strs))[0]
    results = autoencoder(tokenized).numpy()
    for n, ex in enumerate(tokenized):
        print(test_strs[n])
        rev1 = "".join([tokenizer.char_rev.get(i, "?") for i in tokenized[n]])
        print(f"sampled:\n '{rev1}'")
        rev2 = [tokenizer.char_rev.get(i.argmax(), "?") for i in results[n]]
        print("".join(rev2))
        print("*" * 80)


# load AG News and fit a CharTokenizer on it
ag = load_dataset("ag_news")
ct = CharTokenizer()
ct.fit(ag["train"]["text"], progbar=True)

# Create our character-level autoencoder
ae = CharAutoencoder(ct)

# tokenize an example dataset
# tok = np.array(ct.tokenize(ag["train"]["text"][:1000]))
# tok = tf.keras.preprocessing.sequence.pad_sequences(
#     tok, maxlen=100, dtype="int32", padding="pre", truncating="pre", value=0.0
# )

# compile our model
scce = tf.keras.losses.SparseCategoricalCrossentropy()
ae.compile("adam", scce)

bsize = 128
gen = CharAEGenerator(ct, bsize, max_sample_len=128, min_sample_len=8)
# fit the autoencoder
test_strs = ag["test"]["text"][:5]

epochs = 5
for _ in range(0, epochs):
    ae.fit(
        gen(ag["train"]["text"]),
        epochs=1,
        steps_per_epoch=int(len(ag["train"]) / bsize),
    )

    decoding_test(test_strs, ct, ae)