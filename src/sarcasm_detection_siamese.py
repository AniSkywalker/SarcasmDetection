# not finalized
import os
import collections
import random
import sys

sys.path.append('../')

import time
import numpy

numpy.random.seed(1337)

from keras.layers.wrappers import TimeDistributed
from keras import backend as K, regularizers
from sklearn import metrics
from keras.models import model_from_json
from keras.layers.core import Dropout, Dense, Activation, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.convolutional import Convolution1D
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from keras.layers.merge import add, concatenate, subtract
from keras.models import Model
from keras.utils import np_utils
from keras.layers import Input
import src.data_processing.data_handler as dh
from collections import defaultdict
from keras.utils import plot_model

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
        self._test_file = None
        self._validation_file = None
        self._tweet_file = None
        self._output_file = None
        self._model_file = None
        self._word_file_path = None
        self._vocab_file_path = None
        self._input_weight_file_path = None
        self._vocab = None

        self._line_maxlen = 30

    def _build_network(self, vocab_size, maxlen, emb_weights=[], c_emb_weights=[], hidden_units=256, trainable=True,
                       lstm_trainable=False):

        print('Building model...')

        context_input = Input(name='context', batch_shape=(None, maxlen))

        if (len(c_emb_weights) == 0):
            c_emb = Embedding(vocab_size, 256, input_length=maxlen, embeddings_initializer='glorot_normal',
                              trainable=trainable)(context_input)
        else:
            c_emb = Embedding(vocab_size, c_emb_weights.shape[1], input_length=maxlen, weights=[c_emb_weights],
                              trainable=trainable)(context_input)

        c_lstm1 = LSTM(hidden_units, kernel_initializer='he_normal', bias_initializer='he_normal', activation='sigmoid',
                       dropout=0.25, unit_forget_bias=False, return_sequences=False, trainable=lstm_trainable)(c_emb)

        c_lstm2 = LSTM(hidden_units, kernel_initializer='he_normal', bias_initializer='he_normal', activation='sigmoid',
                       dropout=0.25, unit_forget_bias=False, return_sequences=False, go_backwards=True, trainable=lstm_trainable)(c_emb)

        c_merged = concatenate([c_lstm1, c_lstm2])
        # c_merged = Dropout(0.25)(c_merged)

        text_input = Input(name='text', batch_shape=(None, maxlen))

        if (len(emb_weights) == 0):
            emb = Embedding(vocab_size, 256, input_length=maxlen, embeddings_initializer='glorot_normal',
                            trainable=trainable)(text_input)
        else:
            emb = Embedding(vocab_size, c_emb_weights.shape[1], input_length=maxlen, weights=[emb_weights],
                            trainable=trainable)(text_input)

        t_lstm1 = LSTM(hidden_units, kernel_initializer='he_normal', bias_initializer='he_normal', activation='sigmoid',
                       dropout=0.25, unit_forget_bias=False, return_sequences=False, trainable=lstm_trainable)(emb)

        t_lstm2 = LSTM(hidden_units, kernel_initializer='he_normal', bias_initializer='he_normal', activation='sigmoid',
                       dropout=0.25, unit_forget_bias=False, return_sequences=False, go_backwards=True, trainable=lstm_trainable)(emb)

        t_merged = concatenate([t_lstm1, t_lstm2])
        # t_merged = Dropout(0.25)(t_merged)

        merged = subtract([c_merged, t_merged])

        dnn_1 = Dense(hidden_units, kernel_initializer="he_normal", activation='sigmoid')(merged)
        dnn_1 = Dropout(0.25)(dnn_1)
        dnn_2 = Dense(2, activation='sigmoid')(dnn_1)

        softmax = Activation('softmax')(dnn_2)

        model = Model(inputs=[context_input, text_input], outputs=softmax)

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        print('No of parameter:', model.count_params())

        print(model.summary())

        plot_model(model, to_file=os.path.join(self._model_file, 'model.png'), show_shapes=True)
        return model


class train_model(sarcasm_model):
    train = None
    validation = None

    def load_train_validation_test_data(self):
        print("Loading resource...")
        self.train = dh.loaddata(self._train_file, self._word_file_path, self._split_word_file_path,
                                 self._emoji_file_path, normalize_text=True,
                                 split_hashtag=True,
                                 ignore_profiles=False)
        self.validation = dh.loaddata(self._validation_file, self._word_file_path, self._split_word_file_path,
                                      self._emoji_file_path,
                                      normalize_text=True,
                                      split_hashtag=True,
                                      ignore_profiles=False)

        if (self._test_file != None):
            self.test = dh.loaddata(self._test_file, self._word_file_path, normalize_text=True,
                                    split_hashtag=True,
                                    ignore_profiles=True)

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

        self.load_train_validation_test_data()

        self.train = dh.prepare_siamese_data(self.train)
        self.validation = dh.prepare_siamese_data(self.validation)

        batch_size = 128

        print(self._line_maxlen)
        self._vocab = dh.build_vocab(self.train, ignore_context=False)
        self._vocab['unk'] = len(self._vocab.keys()) + 1

        print(len(self._vocab.keys()) + 1)
        print('unk::', self._vocab['unk'])

        dh.write_vocab(self._vocab_file_path, self._vocab)

        X, Y, D, C, A = dh.vectorize_word_dimension(self.train, self._vocab, drop_dimension_index=None)

        tX, tY, tD, tC, tA = dh.vectorize_word_dimension(self.validation, self._vocab, drop_dimension_index=None)

        X = dh.pad_sequence_1d(X, maxlen=self._line_maxlen)
        C = dh.pad_sequence_1d(C, maxlen=self._line_maxlen)
        D = dh.pad_sequence_1d(D, maxlen=11)

        tX = dh.pad_sequence_1d(tX, maxlen=self._line_maxlen)
        tC = dh.pad_sequence_1d(tC, maxlen=self._line_maxlen)
        tD = dh.pad_sequence_1d(tD, maxlen=11)

        hidden_units = 128
        dimension_size = 300

        W = dh.get_word2vec_weight(self._vocab, n=dimension_size,
                                   path=word2vec_path)

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
                                    hidden_units=hidden_units, trainable=False)

        # open(self._model_file + 'model.json', 'w').write(model.to_json())
        # save_best = ModelCheckpoint(self._model_file + 'model.json.hdf5', save_best_only=True, monitor='val_loss')
        # save_all = ModelCheckpoint(self._model_file + 'weights.{epoch:02d}-{val_loss:.2f}.hdf5',
        #                            save_best_only=False)
        # early_stopping = EarlyStopping(monitor='loss', patience=10, verbose=1)
        # lr_tuner = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=10, verbose=1, mode='auto',
        #                              epsilon=0.0001,
        #                              cooldown=0, min_lr=0.000001)

        # model.fit([C, X], Y, batch_size=batch_size, epochs=100, validation_data=([tC, tX], tY), shuffle=True,
        #           callbacks=[save_best, lr_tuner], class_weight=ratio)

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

    def __init__(self, word_file_path, split_word_path, emoji_file_path, model_file, vocab_file_path, output_file):
        print('initializing...')
        sarcasm_model.__init__(self)

        self._word_file_path = word_file_path
        self._split_word_file_path = split_word_path
        self._emoji_file_path = emoji_file_path
        self._model_file = model_file
        self._vocab_file_path = vocab_file_path
        self._output_file = output_file

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
        self.test = dh.loaddata(test_file, self._word_file_path, self._split_word_file_path,
                                self._emoji_file_path, normalize_text=True, split_hashtag=True,
                                ignore_profiles=False)

        self.test = dh.prepare_siamese_data(self.test, test=True)

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

        self.__predict_model([tC, tX], self.test)

    def __predict_model(self, tX, test):
        prediction_probability = self.model.predict(tX, batch_size=1, verbose=1)

        y = []
        y_pred = []

        fd = open(self._output_file + '.analysis', 'w')
        for i, (label) in enumerate(prediction_probability):
            id = test[i][0]
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
                     + ' '.join(words) + '\t'
                     + str(dimensions) + '\t'
                     + ' '.join(context))

            fd.write('\n')

        print('accuracy::', metrics.accuracy_score(y, y_pred))
        print('precision::', metrics.precision_score(y, y_pred, average='weighted'))
        print('recall::', metrics.recall_score(y, y_pred, average='weighted'))
        print('f_score::', metrics.f1_score(y, y_pred, average='weighted'))
        print('f_score::', metrics.classification_report(y, y_pred, digits=3))

        fd.close()


if __name__ == "__main__":
    basepath = os.getcwd()[:os.getcwd().rfind('/')]
    train_file = basepath + '/resource/train/Train_v1.txt'
    validation_file = basepath + '/resource/dev/Dev_v1.txt'
    test_file = basepath + '/resource/test/Test_v1.txt'
    word_file_path = basepath + '/resource/word_list_freq.txt'
    split_word_path = basepath + '/resource/word_split.txt'
    emoji_file_path = basepath + '/resource/emoji_unicode_names_final.txt'

    output_file = basepath + '/resource/text_siamese_model/TestResults.txt'
    model_file = basepath + '/resource/text_siamese_model/weights/'
    vocab_file_path = basepath + '/resource/text_siamese_model/vocab_list.txt'

    # word2vec path
    word2vec_path = '/home/aghosh/backups/GoogleNews-vectors-negative300.bin'

    tr = train_model(train_file, validation_file, word_file_path, split_word_path, emoji_file_path, model_file,
                     vocab_file_path, output_file)
    #
    # testing the model
    with K.get_session():
        t = test_model(word_file_path, split_word_path, emoji_file_path, model_file, vocab_file_path, output_file)
        t.load_trained_model()
        t.predict(test_file)
