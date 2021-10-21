import os
import sys

if 'word2vec' not in os.getcwd():
    os.chdir('word2vec')

sys.path.append(os.getcwd())

import argparse
from distutils.util import strtobool as _bool
import numpy as np

from src.preprocess_utils import *
from src.huffman import *
from eval import cosine_similarity
# from torch.utils.tensorboard import SummaryWriter


def train(cbow=True, ns=False, epochs=3, subsampling_t=1e-5, window_size=10, update_size=12):
    cfg = Config()
    np.random.seed(1128)
    hidden_size = cfg.hidden_size
    neg_num = cfg.NUM_NEG_SAMPLES

    model_type = 'ns' if ns else 'hs'
    train_type = 'cbow' if cbow else 'sg'
    emb_save_path = './results/embedding_{}_{}_{}_{}epoch.pkl'.format(model_type, train_type,
                                                                      subsampling_t, epochs)
    cont_save_path = './results/context_{}_{}_{}_{}epoch.pkl'.format(model_type, train_type,
                                                                     subsampling_t, epochs)
    node_mat_save_path = './results/node_mat_{}_{}_{}_{}epoch.pkl'.format(model_type, train_type,
                                                                          subsampling_t, epochs)

    # log_dir = 'log/{}_{}_{}_{}epoch.tb'.format(model_type, train_type, subsampling_t, epochs)
    # if not os.path.exists(log_dir):
    #     os.mkdir(log_dir)
    # writer = SummaryWriter(log_dir)
    # log_steps = 10000

    if not os.path.isfile(cfg.freq_path):
        create_dictionary(cfg.train_files)
    frequency = pickle.load(open(cfg.freq_path, 'rb'))
    index_to_word = pickle.load(open(cfg.index_to_word_path, 'rb'))
    vocab_size = len(frequency)
    unigram_table = np.array(pickle.load(open(cfg.unigram_table_path, 'rb')))
    len_unigram_table = len(unigram_table)
    # tree, max_depth = pickle.load(open(cfg.tree_path, 'rb'))
    tree, max_depth = init_huffman_modified()
    total = sum([item[1] for item in frequency.items()])

    embedding = np.random.uniform(low=-0.5/300, high=0.5/300, size=(vocab_size, hidden_size)).astype('f')
    emb_grad_temp = []
    context = np.zeros_like(embedding).astype('f')  # for negative sampling

    node_mat = np.zeros((vocab_size-1, hidden_size)).astype('f')  # for hierarchical softmax
    node_mat_grad_temp = []

    starting_lr = 0.05 if cbow else 0.025
    min_loss = math.inf

    print("Start training on {} words".format(vocab_size))
    step = 0
    update_step = 0
    # logging_loss = 0
    start_time = time.time()
    lr = starting_lr

    update_size = update_size

    for epoch in range(epochs):
        data_paths = []
        total_pairs = 0
        print("======= Epoch {} training =======".format(epoch + 1))
        for i in range(len(cfg.train_files)):
            path = cfg.train_files[i]
            print("======= File number {} =======".format(i + 1))
            dataset = preprocess(path=path, frequency=frequency, total=total, window_size=window_size,
                                 cbow=cbow, subsampling_t=subsampling_t)
            data_path = "./preprocessed/data_{}_{}_{}.pkl".format(model_type, train_type, i)
            pickle.dump(dataset, open(data_path, 'wb'))
            data_paths.append(data_path)
            total_pairs += len(dataset)
        for i, data_path in enumerate(data_paths):
            print("======= File number {} =======".format(i + 1))
            dataset = pickle.load(open(data_path, 'rb'))
            loss = 0
            # lr = starting_lr * (1 - (epoch * 100 + i)/(epochs * 100))
            # if lr < starting_lr * 1e-4:
            #     lr = starting_lr * 1e-4
            print("Learning rate: {:.4f}".format(lr))

            file_count = 0
            lr_update_count = 0
            file_start_time = time.time()
            for input_idx, tgt_idx in tqdm(dataset, desc="Training", ncols=70):
                lr_update_count += 1
                if lr_update_count == 10000:
                    lr -= starting_lr * 10000 / (total_pairs * epochs)
                    if lr < starting_lr * 1e-4:
                        lr = starting_lr * 1e-4
                    lr_update_count = 0
                input_idx = np.array(input_idx)
                if cbow:
                    hidden = np.mean(embedding[input_idx], axis=0)  # (300, )
                else:  # sg
                    hidden = embedding[input_idx]   # (300, )
                hidden = hidden.reshape(1, 300)  # (1, 300)
                file_count += 1
                step += 1
                if ns:
                    while 1:
                        negs = np.random.randint(low=0, high=len_unigram_table, size=neg_num)
                        negs = unigram_table[negs]
                        if tgt_idx in negs:
                            continue
                        else:
                            break
                    targets = np.append(tgt_idx, negs)

                    ct = context[targets]  # (1 + neg_num, 300)
                    out = sigmoid(np.dot(hidden, ct.T))  # (1, 1 + neg_num)
                    p_loss = -np.log(out[:, :1] + 1e-7)
                    n_loss = -np.sum(np.log(1 - out[1:] + 1e-7))
                    loss += (p_loss.item() + n_loss.item())
                    # logging_loss += (p_loss.item() + n_loss.item())

                    out[:, :1] -= 1
                    context_grad = np.dot(out.T, hidden)
                    emb_grad = np.dot(out, context[targets])
                    if cbow:
                        emb_grad /= len(input_idx)
                    for j, target in enumerate(targets):
                        context[target] -= lr * context_grad[j]
                    # context[targets] -= lr * context_grad
                    embedding[input_idx] -= lr * emb_grad.squeeze()
                        
                else:  # hs
                    # depth = tree[tgt_idx][-1]
                    # directions = tree[tgt_idx][:depth].reshape(1, -1)
                    # nodes = tree[tgt_idx][max_depth:(max_depth + depth)]
                    # depth = tree['depth']
                    info = tree[tgt_idx]
                    directions = info['direction_path']
                    nodes = info['index_path']
                    assert len(directions) == len(nodes)
                    out = sigmoid(np.dot(node_mat[nodes], hidden.T).T)  # (1, depth)
                    pair_loss = -np.log(np.prod(np.abs(directions - out)) + 1e-6)
                    loss += pair_loss
                    # logging_loss += pair_loss
                    dout = directions + out - 1  # (1, depth)
                    node_mat_grad = np.dot(dout.T, hidden)  # (depth, 300)
                    emb_grad = np.dot(dout, node_mat[nodes])  # (1, 300)
                    if cbow:
                        emb_grad /= len(input_idx)
                    node_mat_grad_temp.append((nodes, node_mat_grad))
                    emb_grad_temp.append((input_idx, emb_grad.squeeze()))
                    update_step += 1

                    if update_step == update_size or file_count == len(dataset):
                        for nodes, node_mat_grad in node_mat_grad_temp:
                            node_mat[nodes] -= lr * node_mat_grad
                        for input_indices, emb_grad in emb_grad_temp:
                            if cbow:
                                for idx in input_indices:
                                    embedding[idx] -= lr * emb_grad
                            else:
                                embedding[input_indices] -= lr * emb_grad

                        update_step = 0

                        node_mat_grad_temp.clear()
                        emb_grad_temp.clear()
                    else:
                        continue

                # if step % log_steps == 1:
                #     writer.add_scalar('Training loss', logging_loss/log_steps, int((step-1)/log_steps))
                #     logging_loss = 0
                # writer.flush()

            print("Number of pairs trained in this file: {}".format(file_count))
            print("Loss: {:.5f}".format(loss/file_count))
            print("Took {:.2f} hours for single file".format((time.time() - file_start_time)/3600))

            if loss < min_loss:
                min_loss = loss
                pickle.dump(embedding, open(emb_save_path, 'wb'))
                # if args.ns:
                #     pickle.dump(context, open(cont_save_path, 'wb'))
                # else:
                #     pickle.dump(node_mat, open(node_mat_save_path, 'wb'))
            similar_word(embedding)
        print("Training time: {:.2f} hours".format((time.time() - start_time) / 3600))

    if ns:
        return embedding, context
    else:
        return embedding, node_mat


def sigmoid(xs):
    ans = 1 / (1 + np.exp(-xs))
    top = 1 / (1 + math.exp(6))
    bottom = 1 / (1 + math.exp(-6))
    for i, num in enumerate(ans[0]):
        if num < top:
            ans[0, i] = 0
        elif num > bottom:
            ans[0, i] = 1
    return ans


def similar_word(emb):
    index_to_word = pickle.load(open(cfg.index_to_word_path, 'rb'))
    word_to_index = pickle.load(open(cfg.word_to_index_path, 'rb'))
    embedding_norm = np.linalg.norm(emb, axis=1)
    norm_emb = emb / embedding_norm[:, None]
    word1 = word_to_index['king']
    word2 = word_to_index['queen']
    word3 = word_to_index['husband']
    answer = word_to_index['wife']

    target = norm_emb[word2] - norm_emb[word1] + norm_emb[word3]
    target = target / np.linalg.norm(target)

    max_index = answer
    max_sim = cosine_similarity(target, norm_emb[answer])
    for i in tqdm(range(len(word_to_index)), desc="Finding closest word to queen-king+husband", ncols=70):
        if i == word1 or i == word2 or i == word3 or i == answer:
            pass
        else:
            sim = cosine_similarity(norm_emb[i], target)
            if sim > max_sim:
                max_sim = sim
                max_index = i
    print(index_to_word[max_index])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default=None)
    parser.add_argument('--cbow', type=_bool, default=True)
    parser.add_argument('--ns', type=_bool, default=False)
    parser.add_argument('--threshold', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--window_size', type=int, default=5)
    parser.add_argument('--update_size', type=int, default=12)

    args = parser.parse_args()

    train(cbow=args.cbow, ns=args.ns, epochs=args.epochs, subsampling_t=args.threshold, window_size=args.window_size,
          update_size=args.update_size)
