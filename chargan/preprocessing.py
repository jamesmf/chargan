from tqdm import tqdm
import numpy as np
from tokenizers import BertWordPieceTokenizer
from typing import List, Union
import json
from collections import Counter
from typing import Dict, List, Iterable, Generator
from tensorflow.keras.preprocessing.sequence import pad_sequences


class CharTokenizer:
    """
    Simple class for fitting a character-to-index mapping over a dataset
    as a List[str] (works with datasets.Dataset)
    """

    def __init__(
        self,
        load_from: str = None,
    ):
        self.char_dict: Dict[str, int] = {}
        self.char_rev: Dict[int, str] = {}
        if load_from:
            self._load(load_from)

    def fit(self, data: List[str], min_char_freq: int = 1, progbar: bool = False):
        """
        Create a character-level dictionary based on an Iterable of strings
        """
        char_counter: Counter = Counter()
        iterator_: Iterable = data
        if progbar:
            iterator_ = tqdm(data)
        for example in iterator_:
            chars = Counter(example)
            # get counts of characters and tokens
            for char, char_count in chars.items():
                try:
                    char_counter[char] += char_count
                except KeyError:
                    char_counter[char] = char_count

        counts = [k for k, v in char_counter.items() if v >= min_char_freq]
        self.char_rev = {0: "", 1: "?", 2: "?", 3: ""}
        for c in sorted(counts):
            n = len(self.char_rev)
            self.char_rev[n] = c
            self.char_dict[c] = n

    def tokenize_str(self, str_in) -> List[int]:
        """
        Apply the character-to-index map to give a list of ids
        """
        return list(map(lambda x: self.char_dict.get(x, 1), str_in))

    def tokenize(self, inp: Union[str, List[str]]) -> List[int]:
        """
        Tokenize either a string or a list of strings
        """
        if isinstance(inp, str):
            return self.tokenize_str(inp)
        else:
            return [self.tokenize_str(s) for s in inp]

    def save(self, path: str):
        """
        Write a Preprocessor object to a .JSON config
        """
        config = {
            "char_rev": self.char_rev,
            "char_dict": self.char_dict,
            "max_example_len": self.max_example_len,
        }
        with open(path, "w") as f:
            json.dump(config, f)

    def _load(self, path: str):
        """
        Load a Preprocessor object from disk
        """
        with open(path, "rb") as f:
            result = json.load(f)
        for key, value in result.items():
            setattr(self, key, value)


class CharAEGenerator:
    """
    Generates examples for the task of autoencoding character-level text
    """

    def __init__(
        self,
        tokenizer: CharTokenizer,
        batch_size: int,
        max_sample_len: int = 64,
        min_sample_len: int = 4,
    ):
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_sample_len = max_sample_len
        self.min_sample_len = min_sample_len

    def extract_sample(self, inp_str: str) -> str:
        """
        Take a random substring
        """
        samp_len = np.random.randint(self.min_sample_len, self.max_sample_len + 1)
        if len(inp_str) <= samp_len:
            return inp_str
        ind_max = max(len(inp_str) - samp_len, 0)
        r = np.random.randint(0, ind_max + 1)
        return inp_str[r : r + samp_len]

    def __call__(self, inp: List[str]) -> Generator[List[np.ndarray], None, None]:
        """
        Generator function that creates batches of examples of length self.sample_len tokenized
        and ready to feed to the autoencoder
        """
        ind = 0
        while True:
            batch = [self.extract_sample(s) for s in inp[ind : ind + self.batch_size]]
            batch_tokens = self.tokenizer.tokenize(batch)
            mlen = max([len(i) for i in batch_tokens])
            mlen -= mlen % 4  # even number for the down/upsampling to work
            sequences = pad_sequences(
                batch_tokens,
                maxlen=mlen,
                dtype="int32",
                padding="pre",
                truncating="pre",
                value=0.0,
            )
            yield sequences, sequences
            ind += self.batch_size
            if ind >= len(inp):
                ind = 0
