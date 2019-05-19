import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn


class STNE(object):
    def __init__(self, hidden_dim, node_num, fea_dim, seq_len, depth=1, node_fea=None, node_fea_trainable=False):
        self.node_num, self.fea_dim, self.seq_len = node_num, fea_dim, seq_len

        self.input_seqs = tf.placeholder(tf.int32, shape=(None, self.seq_len), name='input_seq')
        self.dropout = tf.placeholder(tf.float32, name='dropout')
        if node_fea is not None:
            assert self.node_num == node_fea.shape[0] and self.fea_dim == node_fea.shape[1]
            self.embedding_W = tf.Variable(initial_value=node_fea, name='encoder_embed', trainable=node_fea_trainable)
        else:
            self.embedding_W = tf.Variable(initial_value=tf.random_uniform(shape=(self.node_num, self.fea_dim)),
                                           name='encoder_embed', trainable=node_fea_trainable)
        input_seq_embed = tf.nn.embedding_lookup(self.embedding_W, self.input_seqs, name='input_embed_lookup')
        # input_seq_embed = tf.layers.dense(input_seq_embed, units=1200, activation=None)

        # encoder
        encoder_cell_fw_0 = tf.contrib.rnn.DropoutWrapper(LSTMCell(hidden_dim), output_keep_prob=1 - self.dropout)
        encoder_cell_bw_0 = tf.contrib.rnn.DropoutWrapper(LSTMCell(hidden_dim), output_keep_prob=1 - self.dropout)
        if depth == 1:
            encoder_cell_fw_all = tf.contrib.rnn.MultiRNNCell([encoder_cell_fw_0])
            encoder_cell_bw_all = tf.contrib.rnn.MultiRNNCell([encoder_cell_bw_0])
        else:
            encoder_cell_fw_1 = tf.contrib.rnn.DropoutWrapper(LSTMCell(hidden_dim), output_keep_prob=1 - self.dropout)
            encoder_cell_bw_1 = tf.contrib.rnn.DropoutWrapper(LSTMCell(hidden_dim), output_keep_prob=1 - self.dropout)

            encoder_cell_fw_all = tf.contrib.rnn.MultiRNNCell([encoder_cell_fw_0] + [encoder_cell_fw_1] * (depth - 1))
            encoder_cell_bw_all = tf.contrib.rnn.MultiRNNCell([encoder_cell_bw_0] + [encoder_cell_bw_1] * (depth - 1))

        encoder_outputs, encoder_final = bi_rnn(encoder_cell_fw_all, encoder_cell_bw_all, inputs=input_seq_embed,
                                                dtype=tf.float32)
        c_fw_list, h_fw_list, c_bw_list, h_bw_list = [], [], [], []
        for d in range(depth):
            (c_fw, h_fw) = encoder_final[0][d]
            (c_bw, h_bw) = encoder_final[1][d]
            c_fw_list.append(c_fw)
            h_fw_list.append(h_fw)
            c_bw_list.append(c_bw)
            h_bw_list.append(h_bw)

        decoder_init_state = tf.concat(c_fw_list + c_bw_list, axis=-1), tf.concat(h_fw_list + h_bw_list, axis=-1)
        decoder_cell = tf.contrib.rnn.DropoutWrapper(LSTMCell(hidden_dim * 2), output_keep_prob=1 - self.dropout)
        decoder_init_state = LSTMStateTuple(
            tf.layers.dense(decoder_init_state[0], units=hidden_dim * 2, activation=None),
            tf.layers.dense(decoder_init_state[1], units=hidden_dim * 2, activation=None))

        self.encoder_output = tf.concat(encoder_outputs, axis=-1)
        encoder_output_T = tf.transpose(self.encoder_output, [1, 0, 2])  # h

        new_state = decoder_init_state
        outputs_list = []
        for i in range(seq_len):
            new_output, new_state = decoder_cell(tf.zeros(shape=tf.shape(encoder_output_T)[1:]), new_state)  # None
            outputs_list.append(new_output)

        decoder_outputs = tf.stack(outputs_list, axis=0)  # seq_len * batch_size * hidden_dim
        decoder_outputs = tf.transpose(decoder_outputs, [1, 0, 2])  # batch_size * seq_len * hidden_dim
        self.decoder_outputs = decoder_outputs
        output_preds = tf.layers.dense(decoder_outputs, units=self.node_num, activation=None)
        loss_ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_seqs, logits=output_preds)
        self.loss_ce = tf.reduce_mean(loss_ce, name='loss_ce')

        self.global_step = tf.Variable(1, name="global_step", trainable=False)


class STNEConv(object):
    def conv_pool(self, in_tensor, filter_size, num_filters, s_length, embedding_size=256):
        with tf.name_scope("conv-maxpool-%s" % filter_size):
            # Convolution Layer
            filter_shape = [filter_size, embedding_size, 1, num_filters]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
            # None, seq_len, word_dim, 1
            conv = tf.nn.conv2d(in_tensor, W, strides=[1, 1, 1, 1], padding="VALID", name="conv")
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
            # Maxpooling over the outputs
            pooled = tf.nn.max_pool(
                h,
                ksize=[1, s_length - filter_size + 1, 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="pool")
            # print(pooled)
        return pooled  # None, 1, 1, num_filters

    def __init__(self, hidden_dim, node_num, fea_dim, seq_len, contnt_len, num_filters, word_dim,
                 vocab_size, depth=1, filter_sizes=[2, 4, 8]):
        self.node_num, self.fea_dim = node_num, fea_dim
        self.seq_len = seq_len

        self.input_seqs = tf.placeholder(tf.int32, shape=(None, self.seq_len), name='input_seq')
        self.input_seq_content = tf.placeholder(tf.int32, shape=(None, self.seq_len, contnt_len),
                                                name='input_seq_content')
        self.dropout_rnn = tf.placeholder(tf.float32, name='dropout_rnn')
        self.dropout_word = tf.placeholder(tf.float32, name='dropout_word')
        self.word_embeds_W = tf.Variable(initial_value=tf.random_uniform(shape=(vocab_size, word_dim)),
                                         name='content_embed', trainable=True)

        contnt_embeds = tf.nn.embedding_lookup(self.word_embeds_W, self.input_seq_content, name='input_content_embed')
        contnt_embeds = tf.reshape(contnt_embeds, [-1, contnt_len, word_dim, 1])
        pooled = []
        for fsize in filter_sizes:
            # batch*seq_len, 1, num_filters
            tmp = self.conv_pool(contnt_embeds, fsize, num_filters, contnt_len, word_dim)
            pooled.append(tf.reshape(tmp, [-1, self.seq_len, num_filters]))
        input_seq_embed = tf.concat(pooled, axis=-1)  # batch, seq_len, num_filters*len(filter_sizes)
        input_seq_embed = tf.nn.dropout(input_seq_embed, keep_prob=1 - self.dropout_word)

        encoder_cell_fw_0 = tf.contrib.rnn.DropoutWrapper(LSTMCell(hidden_dim), output_keep_prob=1 - self.dropout_rnn)
        encoder_cell_bw_0 = tf.contrib.rnn.DropoutWrapper(LSTMCell(hidden_dim), output_keep_prob=1 - self.dropout_rnn)
        if depth == 1:
            encoder_cell_fw_all = tf.contrib.rnn.MultiRNNCell([encoder_cell_fw_0])
            encoder_cell_bw_all = tf.contrib.rnn.MultiRNNCell([encoder_cell_bw_0])
        else:
            encoder_cell_fw_1 = tf.contrib.rnn.DropoutWrapper(LSTMCell(hidden_dim),
                                                              output_keep_prob=1 - self.dropout_rnn)
            encoder_cell_bw_1 = tf.contrib.rnn.DropoutWrapper(LSTMCell(hidden_dim),
                                                              output_keep_prob=1 - self.dropout_rnn)

            encoder_cell_fw_all = tf.contrib.rnn.MultiRNNCell([encoder_cell_fw_0] + [encoder_cell_fw_1] * (depth - 1))
            encoder_cell_bw_all = tf.contrib.rnn.MultiRNNCell([encoder_cell_bw_0] + [encoder_cell_bw_1] * (depth - 1))

        encoder_outputs, encoder_final = bi_rnn(encoder_cell_fw_all, encoder_cell_bw_all, inputs=input_seq_embed,
                                                dtype=tf.float32)
        c_fw_list, h_fw_list, c_bw_list, h_bw_list = [], [], [], []
        for d in range(depth):
            (c_fw, h_fw) = encoder_final[0][d]
            (c_bw, h_bw) = encoder_final[1][d]
            c_fw_list.append(c_fw)
            h_fw_list.append(h_fw)
            c_bw_list.append(c_bw)
            h_bw_list.append(h_bw)

        decoder_init_state = tf.concat(c_fw_list + c_bw_list, axis=-1), tf.concat(h_fw_list + h_bw_list, axis=-1)
        decoder_cell = tf.contrib.rnn.DropoutWrapper(LSTMCell(hidden_dim * 2), output_keep_prob=1 - self.dropout_rnn)
        decoder_init_state = LSTMStateTuple(
            tf.layers.dense(decoder_init_state[0], units=hidden_dim * 2, activation=None),
            tf.layers.dense(decoder_init_state[1], units=hidden_dim * 2, activation=None))

        self.encoder_output = tf.concat(encoder_outputs, axis=-1)
        encoder_output_T = tf.transpose(self.encoder_output, [1, 0, 2])  # h

        new_state = decoder_init_state
        outputs_list = []
        for i in range(seq_len):
            new_output, new_state = decoder_cell(tf.zeros(shape=tf.shape(encoder_output_T)[1:]), new_state)  # None
            outputs_list.append(new_output)

        decoder_outputs = tf.stack(outputs_list, axis=0)  # seq_len * batch_size * hidden_dim
        decoder_outputs = tf.transpose(decoder_outputs, [1, 0, 2])  # batch_size * seq_len * hidden_dim
        self.decoder_output = decoder_outputs
        # decoder_outputs, _ = dynamic_rnn(decoder_cell, inputs=self.encoder_output, initial_state=decoder_init_state)
        output_preds = tf.layers.dense(decoder_outputs, units=self.node_num, activation=None)
        loss_ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_seqs, logits=output_preds)
        self.loss_ce = tf.reduce_mean(loss_ce, name='loss_ce')

        self.global_step = tf.Variable(1, name="global_step", trainable=False)