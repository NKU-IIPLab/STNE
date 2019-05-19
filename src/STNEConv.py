# encoding: utf-8

import numpy as np
from collections import Counter
import itertools
import tensorflow as tf
from sklearn.linear_model import LogisticRegression

from src.models import STNEConv
from src.classify import Classifier, read_node_label

import time
import os


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def read_node_features(filename):
    fea = []
    fin = open(filename, 'r')
    for l in fin.readlines():
        vec = l.split()
        fea.append(np.array([float(x) for x in vec[1:]]))
    fin.close()

    return np.array(fea, dtype='float32')


def read_node_sequences(filename):
    seq = []
    fin = open(filename, 'r')
    for l in fin.readlines():
        vec = l.split()
        seq.append(np.array([int(x) for x in vec]))
    fin.close()
    return np.array(seq)


def reduce_seq2seq_hidden_mean(seq, seq_h, node_num, seq_num, seq_len):
    node_dict = {}
    for i_seq in range(seq_num):
        for j_node in range(seq_len):
            nid = seq[i_seq, j_node]
            if nid in node_dict:
                node_dict[nid].append(seq_h[i_seq, j_node, :])
            else:
                node_dict[nid] = [seq_h[i_seq, j_node, :]]
    vectors = []
    for nid in range(node_num):
        vectors.append(np.average(np.array(node_dict[nid]), 0))
    return np.array(vectors)


def reduce_seq2seq_hidden_add(sum_dict, count_dict, seq, seq_h_batch, seq_len, batch_start):
    for i_seq in range(seq_h_batch.shape[0]):
        for j_node in range(seq_len):
            nid = seq[i_seq + batch_start, j_node]
            if nid in sum_dict:
                sum_dict[nid] = sum_dict[nid] + seq_h_batch[i_seq, j_node, :]
            else:
                sum_dict[nid] = seq_h_batch[i_seq, j_node, :]
            if nid in count_dict:
                count_dict[nid] = count_dict[nid] + 1
            else:
                count_dict[nid] = 1
    return sum_dict, count_dict


def reduce_seq2seq_hidden_avg(sum_dict, count_dict, node_num):
    vectors = []
    for nid in range(node_num):
        vectors.append(sum_dict[nid] / count_dict[nid])
    return np.array(vectors)


def node_classification(session, bs, seqne, sequences, seq_contents, seq_len, node_n, samp_idx, label, ratio):
    enc_sum_dict = {}
    node_cnt = {}
    s_idx, e_idx = 0, bs
    while e_idx < len(sequences):
        batch_enc = session.run(seqne.encoder_output, feed_dict={seqne.input_seqs: sequences[s_idx: e_idx],
                                                                 seqne.input_seq_content: seq_contents[s_idx: e_idx],
                                                                 seqne.dropout_rnn: 0., seqne.dropout_word: 0.
                                                                 })
        enc_sum_dict, node_cnt = reduce_seq2seq_hidden_add(enc_sum_dict, node_cnt, sequences,
                                                           batch_enc.astype('float32'), seq_len, s_idx)
        s_idx, e_idx = e_idx, e_idx + bs

    if s_idx < len(sequences):
        batch_enc = session.run(seqne.encoder_output, feed_dict={seqne.input_seqs: sequences[s_idx: len(sequences)],
                                                                 seqne.input_seq_content: seq_contents[
                                                                                          s_idx: len(sequences)],
                                                                 seqne.dropout_rnn: 0., seqne.dropout_word: 0.}
                                )
        enc_sum_dict, node_cnt = reduce_seq2seq_hidden_add(enc_sum_dict, node_cnt, sequences,
                                                           batch_enc.astype('float32'), seq_len, s_idx)

    node_enc_mean = reduce_seq2seq_hidden_avg(sum_dict=enc_sum_dict, count_dict=node_cnt, node_num=node_n)

    # print("Linear Regression:==================")
    lr = Classifier(vectors=node_enc_mean, clf=LogisticRegression())
    f1_micro, f1_macro = lr.split_train_evaluate(samp_idx, label, ratio)
    return f1_micro, f1_macro


def checkword(session, bs, seqne, sequences, seq_contents, seq_len, node_n):
    enc_sum_dict = {}
    node_cnt = {}
    s_idx, e_idx = 0, bs
    while e_idx < len(sequences):
        batch_enc = session.run(seqne.generate_outputs, feed_dict={seqne.input_seqs: sequences[s_idx: e_idx],
                                                                 seqne.input_seq_content: seq_contents[s_idx: e_idx],
                                                                 seqne.dropout_rnn: 0., seqne.dropout_word: 0.
                                                                 })
        enc_sum_dict, node_cnt = reduce_seq2seq_hidden_add(enc_sum_dict, node_cnt, sequences,
                                                           batch_enc.astype('float32'), seq_len, s_idx)
        s_idx, e_idx = e_idx, e_idx + bs

    if s_idx < len(sequences):
        batch_enc = session.run(seqne.generate_outputs, feed_dict={seqne.input_seqs: sequences[s_idx: len(sequences)],
                                                                 seqne.input_seq_content: seq_contents[
                                                                                          s_idx: len(sequences)],
                                                                 seqne.dropout_rnn: 0., seqne.dropout_word: 0.}
                                )
        enc_sum_dict, node_cnt = reduce_seq2seq_hidden_add(enc_sum_dict, node_cnt, sequences,
                                                           batch_enc.astype('float32'), seq_len, s_idx)

    node_enc_mean = reduce_seq2seq_hidden_avg(sum_dict=enc_sum_dict, count_dict=node_cnt, node_num=node_n)

    print("check word -------")
    return node_enc_mean  # , f1_macro


def check_all_node_trained(trained_set, seq_list, total_node_num):
    for seq in seq_list:
        trained_set.update(seq)
    if len(trained_set) == total_node_num:
        return True
    else:
        return False


def build_input(sentences, vocabulary):
    x = np.array(
        [[vocabulary[word] if word in vocabulary else vocabulary['<UNK/>'] for word in sentence] for sentence in
         sentences]
    )
    return x


def build_vocab(sentences, vocab_size=10000):
    vocab_counter = Counter(itertools.chain(*sentences))
    vocab_inv = [x[0] for x in vocab_counter.most_common(vocab_size)]
    vocab = {x: i for i, x in enumerate(vocab_inv)}
    vocab['<UNK/>'] = len(vocab)
    vocab_inv.append('<UNK/>')
    return vocab, vocab_inv


def pad_sentences(sentences, max_len=150):
    sen_len = min(max(len(x) for x in sentences), max_len)
    padded_sentences = []
    padding_word = '<PAD/>'
    for i in range(len(sentences)):
        sentence = sentences[i]
        if len(sentence) < max_len:
            num_pad = sen_len - len(sentence)
            new_sentence = sentence + [padding_word] * num_pad
        else:
            new_sentence = sentence[:max_len]
        padded_sentences.append(new_sentence)
    return sen_len, padded_sentences


def load_texts(path, vocab_size=10000):
    node_texts = []
    with open(path) as f1:
        node_texts = [l.split() for l in f1.readlines()]
    vocab, vocab_inv = build_vocab(node_texts, vocab_size)
    sen_len, padded = pad_sentences(node_texts)
    x_text = build_input(padded, vocab)
    return sen_len, x_text, vocab


if __name__ == '__main__':
    folder = "/data/cora/"
    fn = folder + 'result.txt'

    dpt = 1  # depth of both the encoder and the decoder layers
    h_dim = 500
    o_dim = 400  # decoder hidden state dimension
    s_len = 10 # input node sequence length
    epc = 20  # seq2seq epoches
    drop = 0.2

    clf_ratio = [0.1, 0.2, 0.3, 0.4, 0.5]
    b_s = 128
    lr = 0.001

    num_filters = 200
    word_dim = 200
    word_drop = 0.4

    start = time.time()
    fobj = open(fn, 'w')

    X, Y = read_node_label(folder + 'labels.txt')

    content_len, node_fea, vocab = load_texts(folder + 'content.txt')

    node_seq = read_node_sequences(folder + 'seq-10-10.txt')  # data_size, s_len

    node_seq_content = np.array([[node_fea[node] for node in seq] for seq in node_seq]).astype('int32')

    with tf.Session() as sess:

        model = STNEConv(hidden_dim=h_dim, node_num=node_fea.shape[0], fea_dim=node_fea.shape[1],
                         seq_len=s_len, contnt_len=content_len, num_filters=num_filters,
                         word_dim=word_dim, vocab_size=len(vocab), depth=dpt, filter_sizes=[1, 2, 3, 4, 5])

        train_op = tf.train.RMSPropOptimizer(lr).minimize(model.loss_ce, global_step=model.global_step)

        sess.run(tf.global_variables_initializer())
        trained_node_set = set()
        words_list=[]
        all_trained = False
        best_result_mi = [0, 0, 0, 0, 0]
        best_result_ma = [0, 0, 0, 0, 0]
        for epoch in range(epc):
            start_idx, end_idx = 0, b_s
            while end_idx < len(node_seq):
                _, loss, step = sess.run([train_op, model.loss_ce, model.global_step],
                                         feed_dict={model.input_seq_content: node_seq_content[start_idx:end_idx],
                                                    model.input_seqs: node_seq[start_idx:end_idx],
                                                    model.dropout_rnn: drop,
                                                    model.dropout_word: word_drop})
                if not all_trained:
                    all_trained = check_all_node_trained(trained_node_set, node_seq[start_idx:end_idx],
                                                         node_fea.shape[0])

                start_idx, end_idx = end_idx, end_idx + b_s

            if start_idx < len(node_seq):
                sess.run([train_op, model.loss_ce, model.global_step],
                         feed_dict={model.input_seqs: node_seq[start_idx:len(node_seq)],
                                    model.input_seq_content: node_seq_content[start_idx:len(node_seq)],
                                    model.dropout_rnn: drop, model.dropout_word: word_drop})

            minute = np.around((time.time() - start) / 60)
            print('\nepoch ', epoch, ' finished in ', str(minute), ' minutes\n')

            f1 = []
            for ratio in clf_ratio:
                f1.append(node_classification(session=sess, bs=b_s, seqne=model, sequences=node_seq, seq_len=s_len,
                                              seq_contents=node_seq_content, node_n=node_fea.shape[0], samp_idx=X,
                                              label=Y, ratio=ratio))

            fobj.write(str(epoch) + ' ')
            for f in f1:
                print(f)
                fobj.write(str(f) + ' ')
            fobj.write('\n')
            minute = np.around((time.time() - start) / 60)

            fobj.write(str(minute) + ' minutes' + '\n')
            print('\nClassification finished in ', str(minute), ' minutes\n')

            minute = np.around((time.time() - start) / 60)
            print('Total time: ' + str(minute) + ' minutes')
        fobj.close()
