import os
import sys

from keras.engine import Input
from keras.layers import concatenate
from keras.models import Model

sys.path.append('../')

import collections
import time
import numpy

numpy.random.seed(1337)
from sklearn import metrics
from keras.models import Sequential, model_from_json
from keras.layers.core import Dropout, Dense, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.utils import np_utils
from collections import defaultdict
import src.data_processing.data_handler as dh


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
    _word2vec_file_path = None
    _vocab = None
    _line_maxlen = None

    def __init__(self):
        self._line_maxlen = 30

    def _build_network(self, vocab_size, maxlen, emb_weights=None, embedding_dimension=256, hidden_units=256):
        print('Build model...')

        text_input = Input(shape=(maxlen,))
        custom_embedding = Embedding(vocab_size, embedding_dimension, embeddings_initializer='glorot_normal',
                                     trainable=True)(text_input)

        pretrained_embedding = Embedding(vocab_size, emb_weights.shape[1], weights=[emb_weights],
                                         trainable=False)(text_input)

        combined_embedding = concatenate([custom_embedding, pretrained_embedding], axis=-1)

        cnn1 = Convolution1D(hidden_units, 2, kernel_initializer='he_normal', padding='valid', activation='sigmoid',
                             input_shape=(1, maxlen))(combined_embedding)

        lstm1 = LSTM(hidden_units, kernel_initializer='he_normal', activation='sigmoid', dropout=0.25)(cnn1)

        dnn1 = Dense(hidden_units, kernel_initializer='he_normal', activation='sigmoid')(lstm1)
        dropout1 = Dropout(0.25)(dnn1)

        dnn2 = Dense(2, activation='sigmoid')(dropout1)

        adam = Adam(lr=0.0001)

        model = Model(inputs=[text_input], outputs=dnn2)

        model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
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
                 word2vec_path=None):
        sarcasm_model.__init__(self)

        self._train_file = train_file
        self._validation_file = validation_file
        self._word_file_path = word_file_path
        self._split_word_file_path = split_word_path
        self._emoji_file_path = emoji_file_path
        self._model_file = model_file
        self._vocab_file_path = vocab_file
        self._output_file = output_file
        self._word2vec_file_path = word2vec_path

        self.load_train_validation_data()

        print(self._line_maxlen)

        # build vocabulary
        # truncates words with min freq=1
        self._vocab = dh.build_vocab(self.train, min_freq=1)
        if ('unk' not in self._vocab):
            self._vocab['unk'] = len(self._vocab.keys()) + 1

        print(len(self._vocab.keys()) + 1)
        print('unk::', self._vocab['unk'])

        dh.write_vocab(self._vocab_file_path, self._vocab)

        # prepares input
        X, Y, D, C, A = dh.vectorize_word_dimension(self.train, self._vocab)
        X = dh.pad_sequence_1d(X, maxlen=self._line_maxlen)

        # prepares input
        tX, tY, tD, tC, tA = dh.vectorize_word_dimension(self.validation, self._vocab)
        tX = dh.pad_sequence_1d(tX, maxlen=self._line_maxlen)

        # embedding dimension
        dimension_size = 25
        hidden_memory_units = 64

        W = None
        if word2vec_path is not None:
            W = dh.get_word2vec_weight(self._vocab, n=300,
                                       path=word2vec_path)

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
        model = self._build_network(len(self._vocab.keys()) + 1, self._line_maxlen, emb_weights=W,
                                    embedding_dimension=dimension_size,
                                    hidden_units=hidden_memory_units)

        open(self._model_file + 'model.json', 'w').write(model.to_json())
        save_best = ModelCheckpoint(model_file + 'model.json.hdf5', save_best_only=True)
        save_all = ModelCheckpoint(self._model_file + 'weights.{epoch:02d}__.hdf5',
                                   save_best_only=False)
        early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=1)

        # training
        # model.fit(X, Y, batch_size=8, epochs=10, validation_data=(tX, tY), shuffle=True,
        #           callbacks=[save_best, save_all, early_stopping], class_weight=ratio)
        model.fit(X, Y, batch_size=32, epochs=100, validation_split=0.2, shuffle=True,
                  callbacks=[save_best, save_all, early_stopping], class_weight=ratio, verbose=2)

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
        self.__load_model(self._model_file_path + model_file, self._model_file_path + weight_file)
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

    def interactive(self, word_file_path, split_word_path, emoji_file_path):
        word_list, emoji_dict, split_word_list, abbreviation_dict = dh.load_resources(word_file_path, split_word_path,
                                                                                      emoji_file_path,
                                                                                      split_hashtag=True)
        self._vocab = self.load_vocab()
        text = ''
        while (text != 'exit'):
            text = input('Enter a query::')
            data = dh.parsedata(['{}\t{}\t{}'.format('id', -1, text)], word_list, split_word_list, emoji_dict,
                                abbreviation_dict, normalize_text=True,
                                split_hashtag=True,
                                ignore_profiles=False)

            tX, tY, tD, tC, tA = dh.vectorize_word_dimension(data, self._vocab)
            tX = dh.pad_sequence_1d(tX, maxlen=self._line_maxlen)
            print(self.__predict_line(tX))

    def predict_file(self, test_file, verbose=False):
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
            end = time.time()
            if (verbose == True):
                print('test resource preparation time::', (end - start))

            self.__predict_model(tX, self.test)
        except Exception as e:
            print('Error:', e)
            raise

    def __predict_line(self, tX):
        prediction_probability = self.model.predict(tX, batch_size=1, verbose=1)
        predicted = numpy.argmax(prediction_probability[0])
        return predicted, prediction_probability

    def __predict_model(self, tX, test):
        y = []
        y_pred = []

        prediction_probability = self.model.predict_proba(tX, batch_size=1, verbose=1)

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
    train_file = basepath + '/resource/train/Train_v3.txt'
    validation_file = basepath + '/resource/dev/Dev_v1.txt'
    test_file = basepath + '/resource/test/Test_v1.txt'
    word_file_path = basepath + '/resource/word_list_freq.txt'
    split_word_path = basepath + '/resource/word_split.txt'
    emoji_file_path = basepath + '/resource/emoji_unicode_names_final.txt'

    output_file = basepath + '/resource/text_model/TestResults.txt'
    model_file = basepath + '/resource/text_model/weights/'
    vocab_file_path = basepath + '/resource/text_model/vocab_list.txt'

    word2vec_path = '/home/aghosh/backups/GoogleNews-vectors-negative300.bin'

    # uncomment for training
    # tr = train_model(train_file, validation_file, word_file_path, split_word_path, emoji_file_path, model_file,
    #                  vocab_file_path, output_file, word2vec_path=word2vec_path)

    t = test_model(model_file, word_file_path, split_word_path, emoji_file_path, vocab_file_path, output_file)
    t.load_trained_model()
    # t.predict_file(test_file)
    t.interactive(word_file_path, split_word_path, emoji_file_path)
