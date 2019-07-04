# for smaller datasets please use the simpler model sarcasm_detection_model_CNN_LSTM_DNN_simpler.py

import os
import sys

from src.data_processing.data_handler import load_glove_model, build_auxiliary_feature

sys.path.append('../')

import collections
import time
import numpy

from keras import backend as K

from keras import backend as K, regularizers
from sklearn import metrics
from keras.models import model_from_json, load_model
from keras.layers.core import Dropout, Dense, Activation, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from keras.layers.merge import concatenate, multiply
from keras.models import Model
from keras.utils import np_utils
from keras.layers import Input, Reshape, Permute, RepeatVector, Lambda, merge
import src.data_processing.data_handler as dh
from collections import defaultdict


class sarcasm_model():
    _train_file = None
    _test_file = None
    _tweet_file = None
    _output_file = None
    _model_file_path = None
    _word_file_path = None
    _split_word_file_path = None
    _emoji_file_path = None
    _vocab_file_path = None
    _input_weight_file_path = None
    _vocab = None
    _line_maxlen = None

    def __init__(self):
        self._line_maxlen = 30

    def attention_3d_block(self, inputs, SINGLE_ATTENTION_VECTOR=False):
        # inputs.shape = (batch_size, time_steps, input_dim)
        input_dim = int(inputs.shape[2])
        a = Permute((2, 1))(inputs)
        a = Reshape((input_dim, self._line_maxlen))(a)
        # this line is not useful. It's just to know which dimension is what.
        a = Dense(self._line_maxlen, activation='softmax')(a)
        if SINGLE_ATTENTION_VECTOR:
            a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
            a = RepeatVector(input_dim)(a)
        a_probs = Permute((2, 1), name='attention_vec')(a)
        output_attention_mul = multiply([inputs, a_probs], name='attention_mul')
        return output_attention_mul

    def _build_network(self, vocab_size, maxlen, emb_weights=[], embedding_dimension=50, hidden_units=256,
                       batch_size=1):
        print('Build model...')

        text_input = Input(name='text', shape=(maxlen,))

        if (len(emb_weights) == 0):
            emb = Embedding(vocab_size, embedding_dimension, input_length=maxlen,
                            embeddings_initializer='glorot_normal',
                            trainable=True)(text_input)
        else:
            emb = Embedding(vocab_size, emb_weights.shape[1], input_length=maxlen, weights=[emb_weights],
                            trainable=False)(text_input)
        emb_dropout = Dropout(0.5)(emb)

        lstm_bwd = LSTM(hidden_units, kernel_initializer='he_normal', activation='sigmoid', dropout=0.4,
                        go_backwards=True, return_sequences=True)(emb_dropout)
        lstm_fwd = LSTM(hidden_units, kernel_initializer='he_normal', activation='sigmoid', dropout=0.4,
                        return_sequences=True)(emb_dropout)

        lstm_merged = concatenate([lstm_bwd, lstm_fwd])

        attention_mul = self.attention_3d_block(lstm_merged)

        flat_attention = Flatten()(attention_mul)

        aux_input = Input(name='aux', shape=(5,))

        merged_aux = concatenate([flat_attention, aux_input], axis=1)


        reshaped = Reshape((-1, 1))(merged_aux)

        print(reshaped.shape)

        cnn1 = Convolution1D(hidden_units, 3, kernel_initializer='he_normal', padding='valid', activation='relu')(
            reshaped)
        pool1 = MaxPooling1D(pool_size=3)(cnn1)
        print(pool1.shape)

        cnn2 = Convolution1D(2 * hidden_units, 3, kernel_initializer='he_normal', padding='valid', activation='relu')(
            pool1)
        pool2 = MaxPooling1D(pool_size=3)(cnn2)
        print(pool2.shape)

        flat_cnn = Flatten()(pool2)

        dnn_1 = Dense(hidden_units)(flat_cnn)
        dropout_1 = Dropout(0.25)(dnn_1)
        dnn_2 = Dense(2)(dropout_1)
        print(dnn_2.shape)

        softmax = Activation('softmax')(dnn_2)

        model = Model(inputs=[text_input, aux_input], outputs=softmax)

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        print('No of parameter:', model.count_params())

        print(model.summary())

        return model


class train_model(sarcasm_model):
    train = None
    validation = None
    print("Loading resource...")

    def __init__(self, train_file, validation_file, word_file_path, split_word_path, emoji_file_path, model_file,
                 vocab_file,
                 output_file,
                 input_weight_file_path=None):
        sarcasm_model.__init__(self)

        self._train_file = train_file
        self._validation_file = validation_file
        self._word_file_path = word_file_path
        self._split_word_file_path = split_word_path
        self._emoji_file_path = emoji_file_path
        self._model_file = model_file
        self._vocab_file_path = vocab_file
        self._output_file = output_file
        self._input_weight_file_path = input_weight_file_path

        self.load_train_validation_data()

        print(self._line_maxlen)
        batch_size = 32

        # build vocabulary
        # truncates words with min freq=1
        self._vocab = dh.build_vocab(self.train, min_freq=1)
        if ('unk' not in self._vocab):
            self._vocab['unk'] = len(self._vocab.keys()) + 1

        print(len(self._vocab.keys()) + 1)
        print('unk::', self._vocab['unk'])

        dh.write_vocab(self._vocab_file_path, self._vocab)

        self.train = self.train[:-(len(self.train) % batch_size)]
        self.validation = self.validation[:-(len(self.validation) % batch_size)]

        # prepares input
        X, Y, D, C, A = dh.vectorize_word_dimension(self.train, self._vocab)
        X = dh.pad_sequence_1d(X, maxlen=self._line_maxlen)

        # prepares input
        tX, tY, tD, tC, tA = dh.vectorize_word_dimension(self.validation, self._vocab)
        tX = dh.pad_sequence_1d(tX, maxlen=self._line_maxlen)

        # embedding dimension
        dimension_size = 300
        emb_weights = load_glove_model(self._vocab, n=dimension_size,
                                       glove_path='/home/aghosh/backups/glove.6B.300d.txt')

        # aux inputs
        aux_train = build_auxiliary_feature(self.train)
        aux_validation = build_auxiliary_feature(self.validation)

        # solving class imbalance
        ratio = self.calculate_label_ratio(Y)
        ratio = [max(ratio.values()) / value for key, value in ratio.items()]
        print('class ratio::', ratio)

        Y, tY = [np_utils.to_categorical(x) for x in (Y, tY)]

        print('train_X', X.shape)
        print('train_Y', Y.shape)
        print('validation_X', tX.shape)
        print('validation_Y', tY.shape)

        # trainable true if you want word2vec weights to be updated
        # Not applicable in this code
        model = self._build_network(len(self._vocab.keys()) + 1, self._line_maxlen, emb_weights, hidden_units=32,
                                    embedding_dimension=dimension_size, batch_size=batch_size)

        # open(self._model_file + 'model.json', 'w').write(model.to_json())
        save_best = ModelCheckpoint(model_file + 'model.json.hdf5', save_best_only=True)
        save_all = ModelCheckpoint(self._model_file + 'weights.{epoch:02d}__.hdf5',
                                   save_best_only=False)
        early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=1)

        # training
        model.fit([X, aux_train], Y, batch_size=batch_size, epochs=10, validation_data=([tX, aux_validation], tY),
                  shuffle=True,
                  callbacks=[save_best, save_all, early_stopping], class_weight=ratio)

    def load_train_validation_data(self):
        self.train = dh.loaddata(self._train_file, self._word_file_path, self._split_word_file_path,
                                 self._emoji_file_path, normalize_text=True,
                                 split_hashtag=True,
                                 ignore_profiles=False)
        print('Training data loading finished...')

        self.validation = dh.loaddata(self._validation_file, self._word_file_path, self._split_word_file_path,
                                      self._emoji_file_path,
                                      normalize_text=True,
                                      split_hashtag=True,
                                      ignore_profiles=False)
        print('Validation data loading finished...')

        if (self._test_file != None):
            self.test = dh.loaddata(self._test_file, self._word_file_path, normalize_text=True,
                                    split_hashtag=True,
                                    ignore_profiles=True)

    def get_maxlen(self):
        return max(map(len, (x for _, x in self.train + self.validation)))

    def write_vocab(self):
        with open(self._vocab_file_path, 'w') as fw:
            for key, value in self._vocab.iteritems():
                fw.write(str(key) + '\t' + str(value) + '\n')

    def calculate_label_ratio(self, labels):
        return collections.Counter(labels)


class test_model(sarcasm_model):
    test = None
    model = None

    def __init__(self, model_file, word_file_path, split_word_path, emoji_file_path, vocab_file_path, output_file,
                 input_weight_file_path=None):
        print('initializing...')
        sarcasm_model.__init__(self)

        self._model_file_path = model_file
        self._word_file_path = word_file_path
        self._split_word_file_path = split_word_path
        self._emoji_file_path = emoji_file_path
        self._vocab_file_path = vocab_file_path
        self._output_file = output_file
        self._input_weight_file_path = input_weight_file_path

        print('test_maxlen', self._line_maxlen)

    def load_trained_model(self, model_file='model.json', weight_file='model.json.hdf5'):
        start = time.time()
        self.__load_model(self._model_file_path + weight_file)
        end = time.time()
        print('model loading time::', (end - start))

    def __load_model(self, model_path):
        self.model = load_model(model_path)
        print('model loaded from file...')
        # self.model.load_weights(model_weight_path)
        # print('model weights loaded from file...')

    def load_vocab(self):
        vocab = defaultdict()
        with open(self._vocab_file_path, 'r') as f:
            for line in f.readlines():
                key, value = line.split('\t')
                vocab[key] = value

        return vocab

    def predict(self, test_file, verbose=False):
        try:
            start = time.time()
            self.test = dh.loaddata(test_file, self._word_file_path, self._split_word_file_path, self._emoji_file_path,
                                    normalize_text=True, split_hashtag=True,
                                    ignore_profiles=False)
            end = time.time()
            if (verbose == True):
                print('test resource loading time::', (end - start))

            self._vocab = self.load_vocab()
            print('vocab loaded...')

            start = time.time()
            tX, tY, tD, tC, tA = dh.vectorize_word_dimension(self.test, self._vocab)
            tX = dh.pad_sequence_1d(tX, maxlen=self._line_maxlen)

            aux_test = build_auxiliary_feature(self.test)

            end = time.time()
            if (verbose == True):
                print('test resource preparation time::', (end - start))

            self.__predict_model([tX, aux_test], self.test)
        except Exception as e:
            print('Error:', e)
            raise

    def __predict_model(self, tX, test):
        y = []
        y_pred = []

        # tX = tX[:-len(tX) % 32]
        # test = test[:-len(test) % 32]

        prediction_probability = self.model.predict_file(tX, batch_size=1, verbose=1)

        try:
            fd = open(self._output_file + '.analysis', 'w')
            for i, (label) in enumerate(prediction_probability):
                gold_label = test[i][1]
                words = test[i][2]
                dimensions = test[i][3]
                context = test[i][4]
                author = test[i][5]

                predicted = numpy.argmax(prediction_probability[i])

                y.append(int(gold_label))
                y_pred.append(predicted)

                fd.write(str(label[0]) + '\t' + str(label[1]) + '\t'
                         + str(gold_label) + '\t'
                         + str(predicted) + '\t'
                         + ' '.join(words))

                fd.write('\n')

            print()

            print('accuracy::', metrics.accuracy_score(y, y_pred))
            print('precision::', metrics.precision_score(y, y_pred, average='weighted'))
            print('recall::', metrics.recall_score(y, y_pred, average='weighted'))
            print('f_score::', metrics.f1_score(y, y_pred, average='weighted'))
            print('f_score::', metrics.classification_report(y, y_pred))
            fd.close()
        except Exception as e:
            print(e)
            raise


if __name__ == "__main__":
    basepath = os.getcwd()[:os.getcwd().rfind('/')]
    train_file = basepath + '/resource/train/Train_v1.txt'
    validation_file = basepath + '/resource/dev/Dev_v1.txt'
    test_file = basepath + '/resource/test/Test_v1.txt'
    word_file_path = basepath + '/resource/word_list_freq.txt'
    split_word_path = basepath + '/resource/word_split.txt'
    emoji_file_path = basepath + '/resource/emoji_unicode_names_final.txt'

    output_file = basepath + '/resource/text_model/TestResults.txt'
    model_file = basepath + '/resource/text_model/weights/'
    vocab_file_path = basepath + '/resource/text_model/vocab_list.txt'

    # uncomment for training
    tr = train_model(train_file, validation_file, word_file_path, split_word_path, emoji_file_path, model_file,
                     vocab_file_path, output_file)

    t = test_model(model_file, word_file_path, split_word_path, emoji_file_path, vocab_file_path, output_file)
    t.load_trained_model(weight_file='model.json.hdf5')
    t.predict(test_file)
