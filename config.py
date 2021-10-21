import os


class Config:
    def __init__(self):
        self.TRAIN_DATA_PATH = \
            './training-monolingual.tokenized.shuffled'
        self.train_files = \
            [os.path.join(self.TRAIN_DATA_PATH, filename) for filename in os.listdir(self.TRAIN_DATA_PATH)]

        self.freq_path = './dicts/frequency_mod.pkl'
        self.index_to_word_path = './dicts/index_to_word_mod.pkl'
        self.word_to_index_path = './dicts/word_to_index_mod.pkl'
        self.tree_path = './dicts/tree.pkl'
        self.unigram_table_path = './dicts/unigram_table_mod.pkl'
        self.eval_root_path = "./test_data/questions-words.txt"  # data for evaluation

        self.hidden_size = 300
        self.THRESHOLD = 1e-5
        self.NUM_NEG_SAMPLES = 5
        self.MIN_COUNT = 5


if __name__ == "__main__":
    cfg = Config()
    print(len(cfg.train_files))
