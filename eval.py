import pickle
import numpy as np
from tqdm import tqdm
from config import Config
import argparse

cfg = Config()


def preprocess_file(test_data):
    with open(test_data) as f:
        lines = f.readlines()
    result = []
    current_set = []
    for line in lines:
        line = line.strip()
        line_words = line.split()
        if line_words[0] == ':' \
                            '':
            if len(current_set) > 0:
                result.append(current_set)
            current_set = []
        else:
            current_set.append(line_words)
    result.append(current_set)

    semantic_temp = result[:5]
    syntactic_temp = result[5:]
    semantic = []
    syntactic = []
    for category in semantic_temp:
        semantic.extend(category)
    for category in syntactic_temp:
        syntactic.extend(category)

    return semantic, syntactic


def test_words(data, embedding_path):
    word_to_index = pickle.load(open(cfg.word_to_index_path, 'rb'))
    index_to_word = pickle.load(open(cfg.index_to_word_path, 'rb'))
    embedding = pickle.load(open(embedding_path, 'rb'))
    embedding_norm = np.linalg.norm(embedding, axis=1)
    embedding_normalized = embedding / embedding_norm[:, None]
    # print(embedding_normalized)
    correct = 0
    count = 0
    oov = 0
    # for question in data:
    # question_num = 1
    # section_boundaries = [506, 5030, 5896, 8363, 8869]
    # section_correct_num = 0
    # section_total = 0
    for question in tqdm(data, desc="Evaluating word2vec embedding", ncols=70):
        indices = []
        for word in question:
            if word not in word_to_index.keys():
                break
            index = word_to_index[word]
            indices.append(index)
        # OOV -> wrong.
        if len(indices) < 4:
            oov += 1
            continue
        count += 1

        output_vec = embedding_normalized[indices[1]] - embedding_normalized[indices[0]] \
                    + embedding_normalized[indices[2]]
        label_idx = indices[3]

        cos_sim = np.dot(embedding_normalized, output_vec)
        sort_idx = (-cos_sim).argsort()
        answer_cands = sort_idx[:4]
        answer = -1
        for idx in answer_cands:
            if idx in indices[:-1]:
                pass
            else:
                answer = idx
                break

        if answer == label_idx:
            correct += 1

        # if label_idx in answer_cands:
        #     correct += 1

    return correct, count, oov


def cosine_similarity(v1, v2):
    m1 = np.linalg.norm(v1)
    m2 = np.linalg.norm(v2)
    return np.dot(v1, v2) / (m1 * m2)


def main(emb_path):
    questions_words = cfg.eval_root_path
    data_sem, data_syn = preprocess_file(questions_words)
    sem_count, sem_total, sem_oov = test_words(data_sem, emb_path)
    print(sem_count, sem_total, sem_oov)
    print("Semantic test accuracy: %.2f%%" % (sem_count / sem_total * 100))

    syn_count, syn_total, syn_oov = test_words(data_syn, emb_path)
    print(syn_count, syn_total, syn_oov)
    print("Syntactic test accuracy: %.2f%%" % (syn_count / syn_total * 100))

    print("Overall test accuracy: %.2f%%" % ((sem_count + syn_count) / (sem_total + syn_total) * 100))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--emb_path', type=str)
    args = parser.parse_args()

    main(args.emb_path)
