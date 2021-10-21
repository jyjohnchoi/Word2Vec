# -*- coding: utf-8 -*-
if __name__ == '__main__':
    import sys
    import os
    sys.path.append(os.getcwd())
import math
from tqdm.auto import tqdm
import numpy as np
import heapq
import pickle
import random
from config import Config
import os
import time
import nltk

"""
Includes methods to preprocess the given benchmark.
"""

cfg = Config()


# def tokenize(sent):
#     tokens = []
#     token = ''
#     for c in sent:
#         if c == '\r':
#             continue
#         if c == ' ' or c == '\t' or c == '\n':
#             if len(token) >= 100:
#                 token = token[:100]
#             tokens.append(token)
#             token = ''
#             # if c == '\n':
#             #     tokens.append('</s>')
#         else:
#             token += c
#     return tokens
def tokenize(sent):
    split_tokens = ['\t', '\v', '\r', '\f', '\0']
    punctuation = ['.', ',', '!', '/', ':', ';',
                   '+', '-', '*', '?', '~', '|',
                   '[', ']', '{', '}', '(', ')',
                   '_', '=', '%', '&', '$', '#',
                   '"', '`', '^', "'", '\\', '<', '>']
    for split_token in split_tokens:
        sent.replace(split_token, ' ')
    for p in punctuation:
        # sent = sent.replace(p, ' ' + p + ' ')  # 남기기
        sent = sent.replace(p, ' ')  # 제거
    tokens = sent.split()
    # tokens = word_tokenize(sent)
    return tokens


def create_dictionary(train_files):
    """
    @param train_files : list of paths to training file
    Creates a dictionary including every word from the corpus, and save it.
    """
    # frequency = {'</s>': 0}
    frequency = {}
    for i, file in tqdm(enumerate(train_files), desc="Creating dictionary from training files", total=len(train_files),
                        ncols=70):
        with open(file, 'rt', encoding="UTF-8") as f:
            for line in f.readlines():
                words_in_line = tokenize(line)
                for word in words_in_line:
                    if word in frequency.keys():
                        frequency[word] += 1
                    else:
                        frequency[word] = 1

    # list of tuples (word, frequency), ordered by max to min via frequency
    frequency = sorted(frequency.items(), key=lambda item: item[1], reverse=True)
    frequency = {vocab: i for vocab, i in frequency if i >= cfg.MIN_COUNT}  # vocab freq < MIN_COUNT removed
    word_to_index = {word: i for i, word in enumerate(frequency.keys())}
    index_to_word = list(word_to_index.keys())

    pickle.dump(frequency, open(cfg.freq_path, 'wb'))
    pickle.dump(word_to_index, open(cfg.word_to_index_path, 'wb'))
    pickle.dump(index_to_word, open(cfg.index_to_word_path, 'wb'))
    print("Frequencies and indices saved!")

    tree, max_depth = init_huffman_tree(frequency)
    table = init_unigram_table(frequency, word_to_index)

    pickle.dump((tree, max_depth), open(cfg.tree_path, 'wb'))
    pickle.dump(table, open(cfg.unigram_table_path, 'wb'))
    print("Negative sample table and Huffman tree saved!")

    print("Number of vocabulary: {}".format(len(frequency)))


def generate_training_data(sentence, word_to_index, window_size, frequency, total, subsampling_t, cbow):
    """
    Create index pairs from given sentence, fitting to the model type (cbow or skipgram)
    """
    sentence = tokenize(sentence)
    length = len(sentence)
    data = []
    # Dynamic window scaling. This allows updating data by every epoch
    for i, target in enumerate(sentence):
        if target not in word_to_index.keys():
            continue
        # if subsampling(target, frequency, total, threshold=subsampling_t):
        #     continue
        window_size = random.randint(1, window_size + 1)
        nbr_indices = list(range(max(0, i - window_size), i)) + list(range(i + 1, min(length, i + window_size + 1)))
        if cbow:
            if subsampling(target, frequency, total, threshold=subsampling_t):
                continue
            nbr = [word_to_index[sentence[idx]] for idx in nbr_indices if sentence[idx] in word_to_index.keys()]
            if len(nbr) == 0:
                continue
            data.append((nbr, word_to_index[target]))
        else:
            for idx in nbr_indices:
                if sentence[idx] in word_to_index.keys():
                    if not subsampling(target, frequency, total, threshold=subsampling_t):
                        data.append((word_to_index[sentence[idx]], word_to_index[target]))
                    else:
                        pass

    return data


def preprocess(path, frequency, total, window_size, cbow=True, subsampling_t=1e-5):
    word_to_index = pickle.load(open(cfg.word_to_index_path, 'rb'))
    start_time = time.time()
    dataset = list()
    with open(path, 'rt', encoding="UTF-8") as f:
        lines = f.readlines()
        for line in tqdm(lines, desc="Generating input, output pairs", ncols=70):
            data = generate_training_data(line, word_to_index,
                                          window_size=window_size, subsampling_t=subsampling_t,
                                          frequency=frequency, total=total, cbow=cbow)
            dataset.extend(data)
    random.shuffle(dataset)
    print("Data generated including {0} pairs, took {1:0.3f} minutes.".format(len(dataset),
                                                                              (time.time() - start_time) / 60))
    return dataset


def subsampling(word, frequency, total, threshold=1e-5):
    if threshold == 0:
        return False
    freq = frequency[word]
    ratio = freq / (threshold * total)
    p = (np.sqrt(ratio) + 1) / ratio
    draw = random.random()
    return p < draw


def init_unigram_table(frequency, word_to_index, power=0.75):
    """
    Return a uni-gram table from the index of word to its probability of appearance.
    P(w) = count(w)^power / sum(count^power)
    """
    table = []
    for word in tqdm(frequency.keys(), desc="Generating unigram table", ncols=70):
        if word == '</s>':
            continue
        occurrence = int(math.pow(frequency[word], power))
        idx = word_to_index[word]
        table.extend([idx] * occurrence)
    print(len(table))
    return table


def init_huffman_tree(frequency):
    """
    frequency: list of elements (word, frequency), ordered by frequency from max to min
    """
    length = len(frequency)
    # use index to prevent error of comparing int and string
    heap = [[item[1], i] for i, item in enumerate(frequency.items())] ## [word_freq, word_index]
    heapq.heapify(heap)
    for i in tqdm(range(length - 1), desc="Creating Huffman Tree", ncols=70):
        min1 = heapq.heappop(heap)
        min2 = heapq.heappop(heap)
        heapq.heappush(heap, [min1[0] + min2[0], i + length, min1, min2])

    # node of heap : [frequency, index, left child, right child]
    word_stack = []
    stack = [[heap[0], [], []]]
    max_depth = 0

    while len(stack) != 0:
        node, direction_path, node_path = stack.pop()
        if node[1] >= length:  # not a leaf node
            current_node = [node[1] - length]  # indices of internal nodes start from zero
            stack.append([node[2], direction_path + [0], node_path + current_node])
            stack.append([node[3], direction_path + [1], node_path + current_node])

        else:  # leaf node
            node.append(np.array(direction_path))
            node.append(np.array(node_path))
            max_depth = max(max_depth, len(direction_path))
            word_stack.append(node)

    # sort by index to fit with frequency order
    word_stack = np.array(sorted(word_stack, key=lambda items: items[1]), dtype=object)
    print(word_stack)
    word_stack = word_stack[:, 2:4]  # only paths

    paths = np.zeros((length, 2 * max_depth + 1)).astype(np.int)
    for i in tqdm(range(length), desc="Padding paths...", ncols=70):
        true_depth = len(word_stack[i, 0])
        paths[i, 0:true_depth] = word_stack[i, 0]
        paths[i, max_depth:max_depth + true_depth] = word_stack[i, 1]
        paths[i, -1] = true_depth

    return paths, max_depth


if __name__ == "__main__":
    cfg = Config()
    tree, max_depth = pickle.load(open(cfg.tree_path, 'rb'))

    print(tree)
    for vocab in tree:
        true_depth = vocab[-1]
        directions = vocab[:true_depth]
        nodes = vocab[max_depth:max_depth+true_depth]

        assert nodes[0] == 555511