import os

import numpy as np
import torch
from torch import nn as nn

from textattack.shared import logger, utils


class EmbeddingLayer(nn.Module):
    """A layer of a model that replaces word IDs with their embeddings.

    This is a useful abstraction for any nn.module which wants to take word IDs
    (a sequence of text) as input layer but actually manipulate words'
    embeddings.

    Requires some pre-trained embedding with associated word IDs.
    """

    def __init__(
        self,
        n_d=100,
        embedding_matrix=None,
        word_list=None,
        oov="<oov>",
        pad="<pad>",
        normalize=True,
    ):
        super(EmbeddingLayer, self).__init__()
        word2id = {}
        if embedding_matrix is not None:
            for word in word_list:
                assert word not in word2id, "Duplicate words in pre-trained embeddings"
                word2id[word] = len(word2id)

            logger.debug(f"{len(word2id)} pre-trained word embeddings loaded.\n")

            n_d = len(embedding_matrix[0])

        if oov not in word2id:
            word2id[oov] = len(word2id)

        if pad not in word2id:
            word2id[pad] = len(word2id)

        self.word2id = word2id
        self.n_V, self.n_d = len(word2id), n_d
        self.oovid = word2id[oov]
        self.padid = word2id[pad]
        self.embedding = nn.Embedding(self.n_V, n_d)
        self.embedding.weight.data.uniform_(-0.25, 0.25)

        weight = self.embedding.weight
        weight.data[: len(word_list)].copy_(torch.from_numpy(embedding_matrix))
        logger.debug(f"EmbeddingLayer shape: {weight.size()}")

        if normalize:
            weight = self.embedding.weight
            norms = weight.data.norm(2, 1)
            if norms.dim() == 1:
                norms = norms.unsqueeze(1)
            weight.data.div_(norms.expand_as(weight.data))

    def forward(self, input):
        return self.embedding(input)


class GloveEmbeddingLayer(EmbeddingLayer):
    """Pre-trained Global Vectors for Word Representation (GLOVE) vectors. Uses
    embeddings of dimension 200.

    GloVe is an unsupervised learning algorithm for obtaining vector
    representations for words. Training is performed on aggregated global
    word-word co-occurrence statistics from a corpus, and the resulting
    representations showcase interesting linear substructures of the word
    vector space.


    GloVe: Global Vectors for Word Representation. (Jeffrey Pennington,
        Richard Socher, and Christopher D. Manning. 2014.)
    """

    EMBEDDING_PATH = "word_embeddings/glove200"

    def __init__(self, emb_layer_trainable=True):
        glove_path = utils.download_if_needed(GloveEmbeddingLayer.EMBEDDING_PATH)
        glove_word_list_path = os.path.join(glove_path, "glove.wordlist.npy")
        word_list = np.load(glove_word_list_path)
        glove_matrix_path = os.path.join(glove_path, "glove.6B.200d.mat.npy")
        embedding_matrix = np.load(glove_matrix_path)
        super().__init__(embedding_matrix=embedding_matrix, word_list=word_list)
        self.embedding.weight.requires_grad = emb_layer_trainable


class GloveLikeEmbeddingLayer(EmbeddingLayer):
    """Pre-trained GLOVE or GN_GLOVE vectors. Uses embeddings of dimension 300.

    Learning gender-neutral word embeddings. (Zhao, Jieyu, Yichao Zhou,
        Zeyu Li, Wei Wang, and Kai-Wei Chang.)
    """

    ZHAO2018_PATHS = {
        "zhao2018": "word_embeddings/glove.zhao2018.wikidump.300d.txt",
        "zhao2018-gn": "word_embeddings/gn_glove.zhao2018.wikidump.300d.txt",
    }

    def __init__(self, embedding_type, emb_layer_trainable=False):
        if embedding_type in self.ZHAO2018_PATHS:
            embedding_matrix, word_list = self.load_zhao2018(embedding_type=embedding_type)
        else:
            assert embedding_type == "glove200"
            embedding_matrix, word_list = self.load_glove200()

        super().__init__(embedding_matrix=embedding_matrix, word_list=word_list)
        self.embedding.weight.requires_grad = emb_layer_trainable

    def load_zhao2018(self, embedding_type):
        assert embedding_type in self.ZHAO2018_PATHS, f"Unsupported embedding_type: {embedding_type}"
        glove_path = self.ZHAO2018_PATHS[embedding_type]

        with open(glove_path, 'r') as f:
            lines = f.readlines()

        word_list = []
        embedding_matrix = []

        for l in lines:
            splits = l.split()
            word_list.append(splits[0])
            embedding_matrix.append(splits[1:])

        word_list = np.array(word_list)
        embedding_matrix = np.array(embedding_matrix, dtype=np.float)
        return embedding_matrix, word_list

    def load_glove200(self):
        glove_path = utils.download_if_needed(GloveEmbeddingLayer.EMBEDDING_PATH)
        glove_word_list_path = os.path.join(glove_path, "glove.wordlist.npy")
        word_list = np.load(glove_word_list_path)
        glove_matrix_path = os.path.join(glove_path, "glove.6B.200d.mat.npy")
        embedding_matrix = np.load(glove_matrix_path)
        return embedding_matrix, word_list
