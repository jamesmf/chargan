from tqdm import tqdm
import numpy as np
from tokenizers import BertWordPieceTokenizer
from typing import List, Union
import json
from collections import Counter
from typing import Dict, List, Iterable


class CharTokenizer:
    """
    Simple class for fitting a character-to-index mapping over a dataset
    as an Iterable[str]
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
        self.token_rev = {}
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
