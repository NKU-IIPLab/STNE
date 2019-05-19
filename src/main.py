# encoding: utf-8

import time

import numpy as np
import tensorflow as tf
from sklearn.linear_model import LogisticRegression

from src.models import STNE
from src.classify import Classifier, read_node_label


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


def node_classification(session, bs, seqne, sequences, seq_len, node_n, samp_idx, label, ratio):
    enc_sum_dict = {}
    node_cnt = {}
    s_idx, e_idx = 0, bs
    while e_idx < len(sequences):
        batch_enc = session.run(seqne.encoder_output,
                                feed_dict={seqne.input_seqs: sequences[s_idx: e_idx], seqne.dropout: 0})
        enc_sum_dict, node_cnt = reduce_seq2seq_hidden_add(enc_sum_dict, node_cnt, sequences,
                                                           batch_enc.astype('float32'), seq_len, s_idx)
        s_idx, e_idx = e_idx, e_idx + bs

    if s_idx < len(sequences):
        batch_enc = session.run(seqne.encoder_output,
                                feed_dict={seqne.input_seqs: sequences[s_idx: len(sequences)], seqne.dropout: 0})
        enc_sum_dict, node_cnt = reduce_seq2seq_hidden_add(enc_sum_dict, node_cnt, sequences,
                                                           batch_enc.astype('float32'), seq_len, s_idx)

    node_enc_mean = reduce_seq2seq_hidden_avg(sum_dict=enc_sum_dict, count_dict=node_cnt, node_num=node_n)
    lr = Classifier(vectors=node_enc_mean, clf=LogisticRegression())
    f1_micro, f1_macro = lr.split_train_evaluate(samp_idx, label, ratio)
    return f1_micro


def node_classification_d(session, bs, seqne, sequences, seq_len, node_n, samp_idx, label, ratio):
    enc_sum_dict = {}
    node_cnt = {}
    s_idx, e_idx = 0, bs
    while e_idx < len(sequences):
        batch_dec = session.run(seqne.decoder_outputs,
                                feed_dict={seqne.input_seqs: sequences[s_idx: e_idx], seqne.dropout: 0})
        enc_sum_dict, node_cnt = reduce_seq2seq_hidden_add(enc_sum_dict, node_cnt, sequences,
                                                           batch_dec.astype('float32'), seq_len, s_idx)
        s_idx, e_idx = e_idx, e_idx + bs

    if s_idx < len(sequences):
        batch_dec = session.run(seqne.encoder_output,
                                feed_dict={seqne.input_seqs: sequences[s_idx: len(sequences)], seqne.dropout: 0})
        enc_sum_dict, node_cnt = reduce_seq2seq_hidden_add(enc_sum_dict, node_cnt, sequences,
                                                           batch_dec.astype('float32'), seq_len, s_idx)

    node_enc_mean = reduce_seq2seq_hidden_avg(sum_dict=enc_sum_dict, count_dict=node_cnt, node_num=node_n)

    lr = Classifier(vectors=node_enc_mean, clf=LogisticRegression())
    f1_micro, f1_macro = lr.split_train_evaluate(samp_idx, label, ratio)
    return f1_micro


def node_classification_hd(session, bs, seqne, sequences, seq_len, node_n, samp_idx, label, ratio):
    enc_sum_dict = {}
    dec_sum_dict = {}
    node_cnt_enc = {}
    node_cnt_dec = {}
    s_idx, e_idx = 0, bs
    while e_idx < len(sequences):
        batch_enc = session.run(seqne.encoder_output,
                                feed_dict={seqne.input_seqs: sequences[s_idx: e_idx], seqne.dropout: 0})
        enc_sum_dict, node_cnt_enc = reduce_seq2seq_hidden_add(enc_sum_dict, node_cnt_enc, sequences,
                                                               batch_enc.astype('float32'), seq_len, s_idx)

        batch_dec = session.run(seqne.decoder_outputs,
                                feed_dict={seqne.input_seqs: sequences[s_idx: e_idx], seqne.dropout: 0})
        dec_sum_dict, node_cnt_dec = reduce_seq2seq_hidden_add(dec_sum_dict, node_cnt_dec, sequences,
                                                               batch_dec.astype('float32'), seq_len, s_idx)
        s_idx, e_idx = e_idx, e_idx + bs

    if s_idx < len(sequences):
        batch_enc = session.run(seqne.encoder_output,
                                feed_dict={seqne.input_seqs: sequences[s_idx: e_idx], seqne.dropout: 0})
        enc_sum_dict, node_cnt_enc = reduce_seq2seq_hidden_add(enc_sum_dict, node_cnt_enc, sequences,
                                                               batch_enc.astype('float32'), seq_len, s_idx)

        batch_dec = session.run(seqne.decoder_outputs,
                                feed_dict={seqne.input_seqs: sequences[s_idx: e_idx], seqne.dropout: 0})
        dec_sum_dict, node_cnt_dec = reduce_seq2seq_hidden_add(dec_sum_dict, node_cnt_dec, sequences,
                                                               batch_dec.astype('float32'), seq_len, s_idx)

    node_enc_mean = reduce_seq2seq_hidden_avg(sum_dict=enc_sum_dict, count_dict=node_cnt_enc, node_num=node_n)
    node_dec_mean = reduce_seq2seq_hidden_avg(sum_dict=dec_sum_dict, count_dict=node_cnt_dec, node_num=node_n)

    node_mean = np.concatenate((node_enc_mean, node_dec_mean), axis=1)
    lr = Classifier(vectors=node_mean, clf=LogisticRegression())
    f1_micro, f1_macro = lr.split_train_evaluate(samp_idx, label, ratio)
    return f1_micro


def check_all_node_trained(trained_set, seq_list, total_node_num):
    for seq in seq_list:
        trained_set.update(seq)
    if len(trained_set) == total_node_num:
        return True
    else:
        return False


if __name__ == '__main__':
    folder = '/data/cora/'
    fn = '/data/cora/result.txt'

    dpt = 1            # Depth of both the encoder and the decoder layers (MultiCell RNN)
    h_dim = 500        # Hidden dimension of encoder LSTMs
    s_len = 10         # Length of input node sequence
    epc = 2            # Number of training epochs
    trainable = False  # Node features trainable or not
    dropout = 0.2      # Dropout ration
    clf_ratio = [0.1, 0.2, 0.3, 0.4, 0.5]  # Ration of training samples in subsequent classification
    b_s = 128          # Size of batches
    lr = 0.001         # Learning rate of RMSProp

    start = time.time()
    fobj = open(fn, 'w')
    X, Y = read_node_label(folder + 'labels.txt')
    node_fea = read_node_features(folder + 'tfidf.txt')
    node_seq = read_node_sequences(folder + 'seq-10-10.txt')

    with tf.Session() as sess:
        model = STNE(hidden_dim=h_dim, node_fea_trainable=trainable, seq_len=s_len, depth=dpt, node_fea=node_fea,
                     node_num=node_fea.shape[0], fea_dim=node_fea.shape[1])
        train_op = tf.train.RMSPropOptimizer(lr).minimize(model.loss_ce, global_step=model.global_step)
        sess.run(tf.global_variables_initializer())

        trained_node_set = set()
        all_trained = False
        for epoch in range(epc):
            start_idx, end_idx = 0, b_s
            print('Epoch,\tStep,\tLoss,\t#Trained Nodes')
            while end_idx < len(node_seq):
                _, loss, step = sess.run([train_op, model.loss_ce, model.global_step], feed_dict={
                    model.input_seqs: node_seq[start_idx:end_idx], model.dropout: dropout})

                if not all_trained:
                    all_trained = check_all_node_trained(trained_node_set, node_seq[start_idx:end_idx],
                                                         node_fea.shape[0])

                if step % 10 == 0:
                    print(epoch, '\t', step, '\t', loss, '\t', len(trained_node_set))
                    if all_trained:
                        f1_mi = []
                        for ratio in clf_ratio:
                            f1_mi.append(node_classification(session=sess, bs=b_s, seqne=model, sequences=node_seq,
                                                             seq_len=s_len, node_n=node_fea.shape[0], samp_idx=X,
                                                             label=Y, ratio=ratio))

                        print('step ', step)
                        fobj.write('step ' + str(step) + ' ')
                        for f1 in f1_mi:
                            print(f1)
                            fobj.write(str(f1) + ' ')
                        fobj.write('\n')
                start_idx, end_idx = end_idx, end_idx + b_s

            if start_idx < len(node_seq):
                sess.run([train_op, model.loss_ce, model.global_step],
                         feed_dict={model.input_seqs: node_seq[start_idx:len(node_seq)], model.dropout: dropout})

            minute = np.around((time.time() - start) / 60)
            print('\nepoch ', epoch, ' finished in ', str(minute), ' minutes\n')

            f1_mi = []
            for ratio in clf_ratio:
                f1_mi.append(
                    node_classification(session=sess, bs=b_s, seqne=model, sequences=node_seq, seq_len=s_len,
                                        node_n=node_fea.shape[0], samp_idx=X, label=Y, ratio=ratio))

            fobj.write(str(epoch) + ' ')
            print('Classification results on current ')
            for f1 in f1_mi:
                print(f1)
                fobj.write(str(f1) + ' ')
            fobj.write('\n')
            minute = np.around((time.time() - start) / 60)

            fobj.write(str(minute) + ' minutes' + '\n')
            print('\nClassification finished in ', str(minute), ' minutes\n')

        fobj.close()
        minute = np.around((time.time() - start) / 60)
        print('Total time: ' + str(minute) + ' minutes')

