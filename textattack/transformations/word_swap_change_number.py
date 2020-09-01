import numpy as np
from word2number import w2n
from num2words import num2words
from textattack.transformations import Transformation


def idx_to_words(ls, words):
    """Given a list generated from cluster_idx, return a list that contains
    sub-list (the first element being the idx, and the second element being the
    words corresponding to the idx)"""

    output = []
    for cluster in ls:
        word = words[cluster[0]]
        for idx in cluster[1:]:
            word = " ".join([word, words[idx]])
        output.append([cluster, word])
    return output


def cluster_idx(idx_ls):
    """Given a list of idx, return a list that contains sub-lists of adjacent
    idx."""

    if len(idx_ls) < 2:
        return [[i] for i in idx_ls]
    else:
        output = [[idx_ls[0]]]
        prev = idx_ls[0]
        list_pos = 0

        for idx in idx_ls[1:]:
            if idx - 1 == prev:
                output[list_pos].append(idx)
            else:
                output.append([idx])
                list_pos += 1
            prev = idx
        return output


class WordSwapChangeNumber(Transformation):
    def __init__(self, max_change=1, n=3, **kwargs):
        """A transformation that recognize numbers in sentence, and return
        sentences with altered numbers.

        :param max_change: Maximum percent of change (1 being 100%)
        :param n: Numbers of new numbers to generate
        """
        super().__init__(**kwargs)
        self.max_change = max_change
        self.n = n

    def _get_transformations(self, current_text, indices_to_modify):
        words = current_text.words
        num_idx = []
        num_words = []

        # find indexes of alphabetical words
        for idx in indices_to_modify:
            word = words[idx].lower()
            for number in STR_NUM:
                if number in word:
                    if word in ["point", "and"]:
                        if 0 < idx and (idx - 1) in num_idx:
                            num_idx.append(idx)
                    else:
                        num_idx.append(idx)
                    break

            if word.isdigit():
                num_words.append([[idx], word])

        # cluster adjacent indexes to get whole number
        num_idx = cluster_idx(num_idx)
        num_words += idx_to_words(num_idx, words)

        # replace original numbers with new numbers
        transformed_texts = []
        for num_word in num_words:
            idx = num_word[0]
            word = num_word[1]
            replacement_words = self._get_new_number(word)
            for r in replacement_words:
                if r == word:
                    continue
                text = current_text.replace_word_at_index(idx[0], str(r))
                if len(idx) > 1:
                    index = idx[1]
                    for i in idx[1:]:
                        text = text.delete_word_at_index(index)
                transformed_texts.append(text)

        return transformed_texts

    def _get_new_number(self, word):
        """Given a word, try altering the value if the word is a number return
        in digits if word is given in digit, return in alphabetical form if
        word is given in alphabetical form."""

        if word.isdigit():
            num = float(word)
            return self._alter_number(num)
        else:
            try:
                num = w2n.word_to_num(word)
                num_list = self._alter_number(num)
                return [num2words(n) for n in num_list]
            except ValueError:
                return []

    def _alter_number(self, num):
        """helper function of _get_new_number, change number base on
        self.max_change."""
        if num not in [0, 2, 4]:
            change = int(num * self.max_change) + 1
            if num >= 0:
                num_list = np.random.randint(max(num - change, 1), num + change, self.n)
            else:
                num_list = np.random.randint(num - change, min(0, num + change), self.n)
            return num_list
        return []


STR_NUM = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
    "hundred",
    "thousand",
    "million",
    "billion",
    "point",
    "and",
]
