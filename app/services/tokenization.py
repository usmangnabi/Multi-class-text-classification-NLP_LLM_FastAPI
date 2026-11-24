# DAY 1 - Step 2: BPE and WordPiece tokenizers written from scratch
import json  # json saves and loads our tokenizer files
from collections import Counter  # Counter counts how often things appear


class BPETokenizer:
    """Byte Pair Encoding: repeatedly merge the most FREQUENT pair of symbols."""

    def __init__(self, num_merges: int = 300):
        self.num_merges = num_merges  # how many merge steps to learn
        self.merges = []  # list of learned merges, in order
        self.vocab = set()  # all known tokens

    def train(self, texts: list):
        word_freq = Counter()  # how often each word appears
        for text in texts:  # go through every text
            for word in text.split():  # go through every word
                word_freq[tuple(word) + ("</w>",)] += 1  # word as letters + end marker
        self.vocab = {sym for word in word_freq for sym in word}  # start vocab = single letters
        for _ in range(self.num_merges):  # repeat the merge step
            pair_freq = Counter()  # how often each neighbour pair appears
            for word, freq in word_freq.items():  # look at every word
                for i in range(len(word) - 1):  # look at every neighbour pair
                    pair_freq[(word[i], word[i + 1])] += freq  # count the pair
            if not pair_freq:  # nothing left to merge
                break  # stop early
            best = pair_freq.most_common(1)[0][0]  # the most frequent pair
            self.merges.append(best)  # remember this merge
            self.vocab.add("".join(best))  # the merged pair becomes a new token
            word_freq = self._merge(word_freq, best)  # apply the merge to all words

    def _merge(self, word_freq: dict, pair: tuple) -> Counter:
        new_freq = Counter()  # new word counts after merging
        for word, freq in word_freq.items():  # for each word
            symbols, i = [], 0  # build the new symbol list
            while i < len(word):  # walk through the word
                if i < len(word) - 1 and (word[i], word[i + 1]) == pair:  # pair found
                    symbols.append(word[i] + word[i + 1])  # merge the two symbols
                    i += 2  # skip both
                else:
                    symbols.append(word[i])  # keep symbol as it is
                    i += 1  # move one step
            new_freq[tuple(symbols)] += freq  # save the new word
        return new_freq  # return updated counts

    def tokenize(self, text: str) -> list:
        tokens = []  # final tokens
        for word in text.split():  # tokenize word by word
            symbols = list(word) + ["</w>"]  # start from letters
            for pair in self.merges:  # apply merges in the learned order
                i = 0  # position in the word
                while i < len(symbols) - 1:  # check every neighbour pair
                    if (symbols[i], symbols[i + 1]) == pair:  # pair matches
                        symbols[i:i + 2] = ["".join(pair)]  # merge them
                    else:
                        i += 1  # move on
            tokens += [s if s in self.vocab else "<unk>" for s in symbols]  # unknown -> <unk>
        return tokens  # list of tokens

    def save(self, path):
        data = {"merges": [list(m) for m in self.merges], "vocab": sorted(self.vocab)}  # what to save
        with open(path, "w") as file:  # open the file for writing
            json.dump(data, file)  # write as JSON

    @classmethod
    def load(cls, path):
        with open(path) as file:  # open the saved file
            data = json.load(file)  # read the JSON
        tokenizer = cls(num_merges=len(data["merges"]))  # create an empty tokenizer
        tokenizer.merges = [tuple(m) for m in data["merges"]]  # restore merges
        tokenizer.vocab = set(data["vocab"])  # restore vocabulary
        return tokenizer  # ready to use


class WordPieceTokenizer:
    """WordPiece: merge the pair with the best SCORE = pair_freq / (freq_a * freq_b)."""

    def __init__(self, num_merges: int = 300):
        self.num_merges = num_merges  # how many merge steps to learn
        self.vocab = set()  # all known tokens

    def train(self, texts: list):
        word_freq = Counter()  # how often each word appears
        for text in texts:  # go through every text
            for word in text.split():  # go through every word
                pieces = (word[0],) + tuple("##" + c for c in word[1:])  # first letter, then ##letters
                word_freq[pieces] += 1  # count the word
        self.vocab = {p for word in word_freq for p in word} | {"[UNK]"}  # start vocab = letters
        for _ in range(self.num_merges):  # repeat the merge step
            unit_freq, pair_freq = Counter(), Counter()  # counts of single pieces and pairs
            for word, freq in word_freq.items():  # look at every word
                for p in word:  # count every piece
                    unit_freq[p] += freq  # piece count
                for i in range(len(word) - 1):  # count every neighbour pair
                    pair_freq[(word[i], word[i + 1])] += freq  # pair count
            if not pair_freq:  # nothing left to merge
                break  # stop early
            best = max(pair_freq, key=lambda p: pair_freq[p] / (unit_freq[p[0]] * unit_freq[p[1]]))  # best score
            new_token = best[0] + best[1].replace("##", "", 1)  # join the pair (drop the 2nd ##)
            self.vocab.add(new_token)  # add the new token
            word_freq = self._merge(word_freq, best, new_token)  # apply merge to all words

    def _merge(self, word_freq: dict, pair: tuple, new_token: str) -> Counter:
        new_freq = Counter()  # new word counts after merging
        for word, freq in word_freq.items():  # for each word
            pieces, i = [], 0  # build the new piece list
            while i < len(word):  # walk through the word
                if i < len(word) - 1 and (word[i], word[i + 1]) == pair:  # pair found
                    pieces.append(new_token)  # use the merged token
                    i += 2  # skip both
                else:
                    pieces.append(word[i])  # keep piece
                    i += 1  # move one step
            new_freq[tuple(pieces)] += freq  # save the new word
        return new_freq  # return updated counts

    def tokenize(self, text: str) -> list:
        tokens = []  # final tokens
        for word in text.split():  # tokenize word by word
            tokens += self._tokenize_word(word)  # add the pieces of this word
        return tokens  # list of tokens

    def _tokenize_word(self, word: str) -> list:
        pieces, start = [], 0  # pieces found so far, and where we are in the word
        while start < len(word):  # until the whole word is covered
            end, found = len(word), None  # try the longest piece first
            while end > start:  # shrink the piece until it is in vocab
                piece = word[start:end]  # candidate piece
                if start > 0:  # pieces inside a word start with ##
                    piece = "##" + piece  # add the ## marker
                if piece in self.vocab:  # known piece
                    found = piece  # use it
                    break  # stop shrinking
                end -= 1  # try a shorter piece
            if found is None:  # nothing matched
                return ["[UNK]"]  # whole word is unknown
            pieces.append(found)  # save the piece
            start = end  # continue after this piece
        return pieces  # pieces for this word

    def save(self, path):
        with open(path, "w") as file:  # open the file for writing
            json.dump({"vocab": sorted(self.vocab)}, file)  # write vocabulary as JSON

    @classmethod
    def load(cls, path):
        with open(path) as file:  # open the saved file
            data = json.load(file)  # read the JSON
        tokenizer = cls()  # create an empty tokenizer
        tokenizer.vocab = set(data["vocab"])  # restore vocabulary
        return tokenizer  # ready to use
