from src.preprocess_utils import *
from config import Config
import nltk

cfg = Config()


def preprocess_final():
    create_dictionary(cfg.train_files)


def count(cbow):
    total = 0

    for i in tqdm(range(100), ncols=70):
        if cbow:
            path = './data_preprocessed/cbow/data_file{}.pkl'.format(i + 1)
        else:
            path = './data_preprocessed/skipgram/data_file{}.pkl'.format(i + 1)
        dataset = pickle.load(open(path, 'rb'))
        total += len(dataset)
    print("total: {}".format(total))


if __name__ == '__main__':
    preprocess_final()
