import pickle
from collections import Counter

try:
    import tqdm
except ImportError:  # pragma: no cover - fallback when tqdm missing
    class _TqdmModule:
        @staticmethod
        def tqdm(iterable, *args, **kwargs):
            return iterable

    tqdm = _TqdmModule()  # type: ignore


class TorchVocab(object):
    """Lightweight port of LogBERT's TorchVocab to unpickle vocab artifacts."""

    def __init__(
        self,
        counter,
        max_size=None,
        min_freq=1,
        specials=None,
        vectors=None,
        unk_init=None,
        vectors_cache=None,
    ):
        specials = specials or ["<pad>", "<oov>"]
        self.freqs = counter
        counter = counter.copy()
        min_freq = max(min_freq, 1)

        self.itos = list(specials)
        for tok in specials:
            counter.pop(tok, None)

        max_size = None if max_size is None else max_size + len(self.itos)

        words_and_frequencies = sorted(counter.items(), key=lambda item: item[0])
        words_and_frequencies.sort(key=lambda item: item[1], reverse=True)

        for word, freq in words_and_frequencies:
            if freq < min_freq or (max_size is not None and len(self.itos) >= max_size):
                break
            self.itos.append(word)

        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

        self.vectors = None
        if vectors is not None:
            self.load_vectors(vectors, unk_init=unk_init, cache=vectors_cache)
        else:
            assert unk_init is None and vectors_cache is None

    def __eq__(self, other):
        return (
            self.freqs == other.freqs
            and self.stoi == other.stoi
            and self.itos == other.itos
            and self.vectors == other.vectors
        )

    def __len__(self):
        return len(self.itos)

    # Placeholder for compatibility with original API.
    def load_vectors(self, *args, **kwargs):  # pragma: no cover - not used here
        raise NotImplementedError("Vector loading is not implemented in this port.")

    def vocab_rerank(self):
        self.stoi = {word: i for i, word in enumerate(self.itos)}

    def extend(self, vocab, sort=False):
        words = sorted(vocab.itos) if sort else vocab.itos
        for word in words:
            if word not in self.stoi:
                self.itos.append(word)
                self.stoi[word] = len(self.itos) - 1


class Vocab(TorchVocab):
    def __init__(self, counter, max_size=None, min_freq=1):
        self.pad_index = 0
        self.unk_index = 1
        self.eos_index = 2
        self.sos_index = 3
        self.mask_index = 4
        super().__init__(
            counter,
            specials=["<pad>", "<unk>", "<eos>", "<sos>", "<mask>"],
            max_size=max_size,
            min_freq=min_freq,
        )

    @staticmethod
    def load_vocab(vocab_path: str):
        with open(vocab_path, "rb") as f:
            return pickle.load(f)

    def save_vocab(self, vocab_path):
        with open(vocab_path, "wb") as f:
            pickle.dump(self, f)


class WordVocab(Vocab):
    def __init__(self, texts, max_size=None, min_freq=1):
        counter = Counter()
        for line in tqdm.tqdm(texts):
            if isinstance(line, list):
                words = line
            else:
                words = str(line).replace("\n", "").replace("\t", "").split()
            for word in words:
                counter[word] += 1
        super().__init__(counter, max_size=max_size, min_freq=min_freq)

    @staticmethod
    def load_vocab(vocab_path: str) -> "WordVocab":
        with open(vocab_path, "rb") as f:
            return pickle.load(f)
