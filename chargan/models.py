from .preprocessing import CharTokenizer
import tensorflow as tf
import tensorflow.keras as krs
from tensorflow.keras.models import Model

# push these down somewhere more useful
CHAR_EMBEDDING_SIZE = 4
CONV_N_FILTERS = 128
CONV_KERNEL_SIZE = 4
POOL_SIZE = 2


class CharEncoder(tf.keras.layers.Layer):
    def __init__(self, voc_size, emb_size, n_filters, kernel_size):
        super().__init__()
        self.emb = krs.layers.Embedding(voc_size, emb_size, name="enc_emb")
        self.conv1 = krs.layers.Conv1D(
            n_filters,
            kernel_size,
            padding="same",
            activation=krs.layers.LeakyReLU(),
            name="enc_conv_1",
        )
        self.mp1 = krs.layers.MaxPooling1D(
            pool_size=POOL_SIZE, name="maxpool_1", padding="same"
        )
        self.conv2 = krs.layers.Conv1D(
            n_filters,
            kernel_size,
            padding="same",
            activation=krs.layers.LeakyReLU(),
            name="enc_conv_2",
        )
        self.mp2 = krs.layers.MaxPooling1D(
            pool_size=POOL_SIZE, name="maxpool_2", padding="same"
        )

    def call(self, inp):
        x = self.emb(inp)
        x = self.mp1(self.conv1(x))
        x = self.mp2(self.conv2(x))
        return x


class CharDecoder(tf.keras.layers.Layer):
    def __init__(self, voc_size, emb_size, n_filters, kernel_size):
        super().__init__()
        self.up1 = krs.layers.UpSampling1D(size=POOL_SIZE, name="upsample_1")
        self.conv1 = krs.layers.Conv1D(
            n_filters,
            kernel_size,
            padding="same",
            activation=krs.layers.LeakyReLU(),
            name="decc_conv_1",
        )
        self.up2 = krs.layers.UpSampling1D(size=POOL_SIZE, name="upsample_2")
        self.conv2 = krs.layers.Conv1D(
            n_filters,
            kernel_size,
            padding="same",
            activation="relu",
            name="dec_conv_2",
        )
        self.mp2 = krs.layers.MaxPooling1D()
        self.out = krs.layers.Dense(voc_size, activation="softmax")

    def call(self, inp):
        x = self.conv1(self.up1(inp))
        x = self.conv2(self.up2(x))
        x = self.out(x)
        return x


class CharAutoencoder(krs.models.Model):
    def __init__(
        self,
        tokenizer: CharTokenizer,
        load_encoder_from: str = None,
        load_decoder_from: str = None,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.encoder = self.get_encoder(load_from=load_encoder_from)
        self.decoder = self.get_decoder(load_from=load_decoder_from)

    def get_encoder(self, load_from: str = None) -> Model:
        if load_from is not None:
            raise NotImplementedError

        voc_size = max(self.tokenizer.char_dict.values()) + 1
        return CharEncoder(
            voc_size, CHAR_EMBEDDING_SIZE, CONV_N_FILTERS, CONV_KERNEL_SIZE
        )

    def get_decoder(self, load_from: str = None) -> Model:
        if load_from is not None:
            raise NotImplementedError
        voc_size = max(self.tokenizer.char_dict.values()) + 1
        return CharDecoder(
            voc_size, CHAR_EMBEDDING_SIZE, CONV_N_FILTERS, CONV_KERNEL_SIZE
        )

    def call(self, inp: tf.Tensor):
        return self.decoder(self.encoder(inp))
