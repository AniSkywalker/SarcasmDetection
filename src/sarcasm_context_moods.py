import os
import collections
import random
import sys

sys.path.append('../../')

from keras.layers.wrappers import TimeDistributed
from keras import backend as K, optimizers, regularizers

import time
import numpy
from sklearn import metrics
from keras.models import Sequential, model_from_json
from keras.layers.core import Dropout, Dense, Activation, Flatten, Reshape
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.convolutional import Convolution1D
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Bidirectional, Merge
from keras.optimizers import Adam

from keras.layers.merge import add, concatenate
from keras.models import Model
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix
from keras.layers import Input
import matplotlib.pyplot as plt
from pandas import DataFrame
import sarcasm_detection_master.src.data_processing.data_handler as dh
from collections import defaultdict
import seaborn as sns
from sarcasm_detection_master.src.Common_functions import generative_function as gf
import pandas as pd


class sarcasm_model():
    _train_file = None
    _gold_data_path = None
    _validation_file = None
    _tweet_file = None
    # test_debug = None
    _output_file = None
    _model_file = None
    _word_file_path = None
    _vocab_file_path = None
    _input_weight_file_path = None
    _vocab = None
    _line_maxlen = None

    def __init__(self):
        self._train_file = None
        self._gold_data_path = None
        self._validation_file = None
        self._tweet_file = None
        self._output_file = None
        self._model_file = None
        self._word_file_path = None
        self._vocab_file_path = None
        self._input_weight_file_path = None
        self._vocab = None

        self._line_maxlen = 30

    def _build_network(self, vocab_size, maxlen, emb_weights=[], c_emb_weights=[], hidden_units=256,
                       dimension_length=11, trainable=True, batch_size = 1):

        print('Building model...')

        context_input = Input(name='context', batch_shape=(batch_size, maxlen))

        if (len(c_emb_weights) == 0):
            c_emb = Embedding(vocab_size, hidden_units, input_length=maxlen, embeddings_initializer='glorot_normal',
                              trainable=trainable)(context_input)
        else:
            c_emb = Embedding(vocab_size, c_emb_weights.shape[1], input_length=maxlen, weights=[c_emb_weights],
                              trainable=trainable)(context_input)

        c_cnn1 = Convolution1D(hidden_units, 3, kernel_initializer='he_normal', bias_initializer='he_normal',
                               activation='sigmoid', padding='valid', use_bias=True, input_shape=(1, maxlen))(c_emb)
        c_cnn2 = Convolution1D(hidden_units, 3, kernel_initializer='he_normal', bias_initializer='he_normal',
                               activation='sigmoid', padding='valid', use_bias=True, input_shape=(1, maxlen - 2))(c_cnn1)

        c_lstm1 = LSTM(hidden_units, kernel_initializer='he_normal', recurrent_initializer='orthogonal',
                       bias_initializer='he_normal', activation='sigmoid', recurrent_activation='hard_sigmoid',
                       kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l2(0.01), recurrent_regularizer= regularizers.l2(0.01),
                       dropout=0.25, recurrent_dropout=.0, unit_forget_bias=False, return_sequences=True)(c_cnn2)

        c_lstm2 = LSTM(hidden_units, kernel_initializer='he_normal', recurrent_initializer='orthogonal',
                       bias_initializer='he_normal', activation='sigmoid', recurrent_activation='hard_sigmoid',
                       kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.01),
                       dropout=0.25, recurrent_dropout=.0, unit_forget_bias=False, return_sequences=True,
                       go_backwards=True)(c_cnn2)

        c_merged = add([c_lstm1, c_lstm2])
        c_merged = Dropout(0.25)(c_merged)

        c_merged = TimeDistributed(Dense(128, kernel_initializer="he_normal", activation='sigmoid'))(c_merged)

        text_input = Input(name='text', batch_shape=(batch_size, maxlen))

        if (len(emb_weights) == 0):
            emb = Embedding(vocab_size, hidden_units, input_length=maxlen, embeddings_initializer='glorot_normal',
                            trainable=trainable)(text_input)
        else:
            emb = Embedding(vocab_size, c_emb_weights.shape[1], input_length=maxlen, weights=[emb_weights],
                            trainable=trainable)(text_input)

        t_cnn1 = Convolution1D(hidden_units, 3, kernel_initializer='he_normal', bias_initializer='he_normal',
                               activation='sigmoid', padding='valid', use_bias=True, input_shape=(1, maxlen))(emb)
        t_cnn2 = Convolution1D(hidden_units, 3, kernel_initializer='he_normal', bias_initializer='he_normal',
                               activation='sigmoid', padding='valid', use_bias=True, input_shape=(1, maxlen - 2))(t_cnn1)

        t_lstm1 = LSTM(hidden_units, kernel_initializer='he_normal', recurrent_initializer='he_normal',
                       bias_initializer='he_normal', activation='sigmoid', recurrent_activation='sigmoid',
                       kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.01),
                       dropout=0.25, recurrent_dropout=0.25, unit_forget_bias=False, return_sequences=True)(t_cnn2)

        t_lstm2 = LSTM(hidden_units, kernel_initializer='he_normal', recurrent_initializer='he_normal',
                       bias_initializer='he_normal', activation='sigmoid', recurrent_activation='sigmoid',
                       kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.01),
                       dropout=0.25, recurrent_dropout=0.25, unit_forget_bias=False, return_sequences=True, go_backwards=True)(t_cnn2)

        t_merged = add([t_lstm1, t_lstm2])
        t_merged = Dropout(0.25)(t_merged)

        t_merged = TimeDistributed(Dense(128, kernel_initializer="he_normal", activation='sigmoid'))(t_merged)

        awc_input = Input(name='awc', batch_shape=(batch_size, 11))

        eaw = Embedding(101, 128, input_length=dimension_length, embeddings_initializer='glorot_normal',
                        trainable=True)(awc_input)

        merged = concatenate([c_merged, t_merged, eaw], axis=1)

        flat_model = Flatten()(merged)


        dnn_1 = Dense(hidden_units, kernel_initializer="he_normal", activation='sigmoid')(flat_model)
        dnn_1 = Dropout(0.25)(dnn_1)
        dnn_2 = Dense(2, activation='sigmoid')(dnn_1)

        softmax = Activation('softmax')(dnn_2)

        model = Model(inputs=[context_input, text_input, awc_input], outputs=softmax)


        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        print('No of parameter:', model.count_params())

        return model


class train_model(sarcasm_model):
    train = None
    validation = None


    def load_train_validation_data(self):
        print("Loading resource...")
        self.train = dh.loaddata(self._train_file, self._word_file_path, normalize_text=True, split_hashtag=True,
                                 ignore_profiles=False,lowercase = True)
        self.validation = dh.loaddata(self._validation_file, self._word_file_path, normalize_text=True,
                                      split_hashtag=True,
                                      ignore_profiles=False,lowercase = True)

    def split_train_validation(self, train, ratio=.1):
        test_indices = sorted([i for i in random.sample(range(len(train)), int(len(train) * ratio))])
        print(len(test_indices))
        train_data = []
        validation_data = []
        for i, t in enumerate(train):
            if (test_indices.__contains__(i)):
                validation_data.append(t)
            else:
                train_data.append(t)
        return train_data, validation_data

    def __init__(self, train_file, validation_file, word_file_path, model_file, vocab_file, output_file,
                 input_weight_file_path, cross_validation=False, cross_val_ratio=0.2):
        sarcasm_model.__init__(self)

        self._train_file = train_file
        self._validation_file = validation_file
        self._word_file_path = word_file_path
        self._model_file = model_file
        self._vocab_file_path = vocab_file
        self._output_file = output_file
        self._input_weight_file_path = input_weight_file_path

        self.load_train_validation_data()

        batch_size = 8

        print(self._line_maxlen)
        self._vocab = dh.build_vocab(self.train,ignore_context=False)
        self._vocab['unk'] = len(self._vocab.keys()) + 1

        print(len(self._vocab.keys()) + 1)
        print('unk::', self._vocab['unk'])

        dh.write_vocab(self._vocab_file_path, self._vocab)

        if (cross_validation):
            self.train, self.validation = self.split_train_validation(self.train, ratio=cross_val_ratio)


        X, Y, D, C, A = dh.vectorize_word_dimension(self.train, self._vocab, drop_dimension_index=None)

        tX, tY, tD, tC, tA = dh.vectorize_word_dimension(self.validation, self._vocab, drop_dimension_index=None)


        X = dh.pad_sequence_1d(X, maxlen=self._line_maxlen)
        C = dh.pad_sequence_1d(C, maxlen=self._line_maxlen)
        D = dh.pad_sequence_1d(D, maxlen=11)

        tX = dh.pad_sequence_1d(tX, maxlen=self._line_maxlen)
        tC = dh.pad_sequence_1d(tC, maxlen=self._line_maxlen)
        tD = dh.pad_sequence_1d(tD, maxlen=11)

        hidden_units = 1280
        dimension_size = 300

        W = dh.get_word2vec_weight(self._vocab, n=dimension_size,
                                   path='/home/word2vec/GoogleNews-vectors-negative300.bin')
        cW = W

        print('Word2vec obtained....')

        ratio = self.calculate_label_ratio(Y)
        ratio = [max(ratio.values()) / value for key, value in ratio.items()]

        print('ratio', ratio)

        dimension_vocab = numpy.unique(D)
        print(len(dimension_vocab))

        Y, tY = [np_utils.to_categorical(x) for x in (Y, tY)]

        print('train_X', X.shape)
        print('train_C', C.shape)
        print('train_D', D.shape)
        print('train_Y', Y.shape)

        print('validation_X', tX.shape)
        print('validation_C', tC.shape)
        print('validation_D', tD.shape)
        print('validation_Y', tY.shape)

        model = self._build_network(len(self._vocab.keys()) + 1, self._line_maxlen, emb_weights=W, c_emb_weights=cW,
                                    hidden_units=hidden_units, trainable=True, dimension_length=11,batch_size=batch_size)


        open(self._model_file + 'model.json', 'w').write(model.to_json())
        save_best = ModelCheckpoint(model_file + 'model.json.hdf5', save_best_only=True, monitor='val_loss')
        save_all = ModelCheckpoint(self._model_file + 'weights.{epoch:02d}-{val_loss:.2f}.hdf5',
                                   save_best_only=False)
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
        lr_tuner = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1, mode='auto', epsilon=0.0001,
                                     cooldown=0, min_lr=0.000001)

        model.fit([C, X, D], Y, batch_size=batch_size, epochs=3, validation_data=([tC, tX, tD], tY), shuffle=True,
                  callbacks=[save_best,early_stopping, lr_tuner], class_weight=ratio)


        if (cross_validation):
            t = test_model(word_file_path, model_file, vocab_file_path, output_file, input_weight_file_path)
            t.load_trained_model()
            t.predict_cross_validation(tC, tX, tD, self.validation)

    def get_maxlen(self):
        return max(map(len, (x for _, x in self.train + self.validation)))

    def write_vocab(self):
        with open(self._vocab_file_path, 'w') as fw:
            for key, value in self._vocab.iteritems():
                fw.write(str(key) + '\t' + str(value) + '\n')

    def calculate_label_ratio(self, labels, ):
        return collections.Counter(labels)


class test_model(sarcasm_model):
    test = None
    model = None

    def __init__(self, word_file_path, model_file, vocab_file_path, output_file, input_weight_file_path):
        print('initializing...')
        sarcasm_model.__init__(self)

        self._word_file_path = word_file_path
        self._model_file = model_file
        self._vocab_file_path = vocab_file_path
        self._output_file = output_file
        self._input_weight_file_path = input_weight_file_path

        # self._line_maxlen = 45
        print('test_maxlen', self._line_maxlen)

    def predict_cross_validation(self, tC, tX, tD, test):
        self.__predict_model([tC, tX, tD], test)

    def load_trained_model(self, weight_file='model.json.hdf5'):
        start = time.time()
        self.__load_model(self._model_file + 'model.json', self._model_file + weight_file)
        end = time.time()
        print('model loading time::', (end - start))

    def __load_model(self, model_path, model_weight_path):
        self.model = model_from_json(open(model_path).read())
        print('model loaded from file...')
        self.model.load_weights(model_weight_path)
        print('model weights loaded from file...')

    def load_vocab(self):
        vocab = defaultdict()
        with open(self._vocab_file_path, 'r') as f:
            for line in f.readlines():
                key, value = line.split('\t')
                vocab[key] = value

        return vocab

    def predict(self, test_file, verbose=False):
        start = time.time()
        self.test = dh.loaddata(test_file, self._word_file_path, normalize_text=True,
                                split_hashtag=True,
                                ignore_profiles=False)
        end = time.time()
        if (verbose == True):
            print('test resource loading time::', (end - start))

        self._vocab = self.load_vocab()

        start = time.time()
        tX, tY, tD, tC, tA = dh.vectorize_word_dimension(self.test, self._vocab)
        tX = dh.pad_sequence_1d(tX, maxlen=self._line_maxlen)
        tC = dh.pad_sequence_1d(tC, maxlen=self._line_maxlen)
        tD = dh.pad_sequence_1d(tD, maxlen=11)

        end = time.time()
        if (verbose == True):
            print('test resource preparation time::', (end - start))

        self.__predict_model([tC, tX, tD], self.test)

    def __predict_model(self, tX, test):
        prediction_probability = self.model.predict(tX, batch_size=8, verbose=1)

        y = []
        y_pred = []

        fd = open(self._output_file + '.analysis', 'w')
        for i, (label) in enumerate(prediction_probability):
            gold_label = test[i][0]
            words = test[i][1]
            dimensions = test[i][2]
            context = test[i][3]
            author = test[i][4]

            predicted = numpy.argmax(prediction_probability[i])

            y.append(int(gold_label))
            y_pred.append(predicted)


            fd.write(str(label[0]) + '\t' + str(label[1]) + '\t'
                     + str(gold_label) + '\t'
                     + str(predicted) + '\t'
                     + ' '.join(words) + '\t'
                     + str(dimensions) + '\t'
                     + ' '.join(context))

            fd.write('\n')

        print('accuracy::', metrics.accuracy_score(y, y_pred))
        print('precision::', metrics.precision_score(y, y_pred, average='weighted'))
        print('recall::', metrics.recall_score(y, y_pred, average='weighted'))
        print('f_score::', metrics.f1_score(y, y_pred, average='weighted'))
        print('f_score::', metrics.classification_report(y, y_pred))

        fd.close()


if __name__ == "__main__":
    basepath = os.getcwd()[:os.getcwd().rfind('/')]
    train_file = basepath + '/resource/train/Train_context_moods.txt'
    validation_file = basepath + '/resource/dev/Dev_context_moods.txt'
    test_file = basepath + '/resource/test/Test_context_moods.txt'
    word_file_path = basepath + '/resource/word_list.txt'
    output_file = basepath + '/resource/text_context_awc_model/TestResults.txt'
    model_file = basepath + '/resource/text_context_awc_model/weights/'
    vocab_file_path = basepath + '/resource/text_context_awc_model/vocab_list.txt'
    input_weight_file_path = basepath + '/resource/text_context_awc_model/partial_weights/weights.txt'

    tr = train_model(train_file, validation_file, word_file_path, model_file, vocab_file_path, output_file,
                     input_weight_file_path)
    with K.get_session():
        t = test_model(word_file_path, model_file, vocab_file_path, output_file, input_weight_file_path)
        t.load_trained_model()
        t.predict(validation_file)
