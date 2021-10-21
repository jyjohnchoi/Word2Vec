import heapq
from config import Config
import pickle
from tqdm import tqdm


class HeapNode:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None
        self.index = None
        self.vector = None

    def __lt__(self, other):
        if other is None:
            return -1
        if not isinstance(other, HeapNode):
            return -1
        return self.freq < other.freq


class HuffmanCoding:
    def __init__(self):
        self.heap = []
        self.codes = {}
        self.reverse_mapping = {}
        self.merged_nodes = None

    def make_heap(self, frequency):  # frequency has a shape of  { word : frequency }
        for key in frequency:  # make a node, then push in list of heap queue
            node = HeapNode(key, frequency[key])
            heapq.heappush(self.heap, node)

    def merge_nodes(self):      # make nodes from low to high frequency and merge to tree.
        index = 0
        merged = None
        while len(self.heap) > 1:
            node1 = heapq.heappop(self.heap)
            node2 = heapq.heappop(self.heap)

            merged = HeapNode(None, node1.freq + node2.freq)
            merged.left = node1
            merged.right = node2
            merged.index = index                # index is reversed, i.e. root node has a biggest index.
            heapq.heappush(self.heap, merged)

            index += 1

        return merged

    def make_codes_helper(self, root, current_code):
        if root is None:
            return

        if root.char is not None:
            self.codes[root.char] = current_code
            self.reverse_mapping[current_code] = root.char
            return

        self.make_codes_helper(root.left, current_code + "0")
        self.make_codes_helper(root.right, current_code + "1")

    def make_codes(self):
        root = heapq.heappop(self.heap)
        current_code = ""
        self.make_codes_helper(root, current_code)

    def build(self, frequency):
        self.make_heap(frequency)
        merged = self.merge_nodes()
        self.make_codes()

        return self.codes, merged


def init_huffman_modified():
    cfg = Config()
    h = HuffmanCoding()
    freq = pickle.load(open(cfg.freq_path, 'rb'))
    word_to_index = pickle.load(open(cfg.word_to_index_path, 'rb'))

    codes, merged = h.build(freq)
    
    tree = {}
    max_depth = 0

    for word in tqdm(codes.keys(), desc='Building Huffman Tree', ncols=100):
        direction_code = codes[word]
        depth = len(direction_code)
        root = merged
        index_path = [root.index]
        direction_path = []
        for i in range(depth):
            direction_path.append(int(direction_code[i]))
            if direction_code[i] == '0':
                root = root.left
            else:
                root = root.right
            if root.index is not None:
                index_path.append(root.index)
        # if len(index_path) != len(direction_path):
        #     print(word)
        #     print(direction_code)
        #     print(len(direction_code))
        #     print(len(index_path))
        #     print(len(direction_path))
        #     print(index_path)
        #     print(direction_path)
        #     break
        info = {'index_path': index_path, 'direction_path': direction_path, 'depth': depth}
        tree[word_to_index[word]] = info

        if depth > max_depth:
            max_depth = depth

    return tree, max_depth
