import os
import sys
sys.path.append('../../')

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
import SarcasmDetection.src.data_processing.data_handler as dh

class sarcasm_model():
    _train_file = None
    _test_file = None
    _tweet_file = None
    _output_file = None
    _model_file = None
    _emoji_file_path = None
    _vocab_file_path = None
    _input_weight_file_path = None
    _vocab = None
    _line_maxlen = None

    def __init__(self):
        self._line_maxlen = 30

    def _build_network(self, vocab_size, maxlen, embedding_dimension = 100, hidden_units=256,trainable=False):
        print('Build model...')
        model = Sequential()

        model.add(Embedding(vocab_size, embedding_dimension, input_length=maxlen, embeddings_initializer='glorot_normal'))

        model.add(Convolution1D(hidden_units, 3, kernel_initializer='he_normal', padding='valid', activation='sigmoid',
                                input_shape=(1, maxlen)))
        # model.add(MaxPooling1D(pool_size=3))
        model.add(Convolution1D(hidden_units, 3, kernel_initializer='he_normal', padding='valid', activation='sigmoid',
                                input_shape=(1, maxlen-2)))
        # model.add(MaxPooling1D(pool_size=3))

        # model.add(Dropout(0.25))

        model.add(LSTM(hidden_units, kernel_initializer='he_normal', activation='sigmoid', dropout=0.5, return_sequences=True))
        model.add(LSTM(hidden_units, kernel_initializer='he_normal', activation='sigmoid', dropout=0.5))


        model.add(Dense(hidden_units, kernel_initializer='he_normal', activation='sigmoid'))
        model.add(Dense(2))
        model.add(Activation('softmax'))
        adam = Adam(lr=0.0001)
        model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
        print('No of parameter:', model.count_params())

        print(model.summary())
        return model

class train_model(sarcasm_model):
    train = None
    validation = None
    print("Loading resource...")

    def __init__(self, train_file, validation_file, emoji_file_path,model_file, vocab_file, output_file,
                 input_weight_file_path=None):

        sarcasm_model.__init__(self)

        self._train_file = train_file
        self._validation_file = validation_file
        self._emoji_file_path = emoji_file_path
        self._model_file = model_file
        self._vocab_file_path = vocab_file
        self._output_file = output_file
        self._input_weight_file_path = input_weight_file_path

        self.load_train_validation_data()

        print(self._line_maxlen)

        #build vocabulary
        self._vocab = dh.build_vocab(self.train)
        self._vocab['unk'] = len(self._vocab.keys()) + 1

        print(len(self._vocab.keys()) + 1)
        print('unk::', self._vocab['unk'])

        dh.write_vocab(self._vocab_file_path,self._vocab)

        #prepares input
        X, Y, D, C, A = dh.vectorize_word_dimension(self.train, self._vocab)
        X = dh.pad_sequence_1d(X, maxlen=self._line_maxlen)

        #prepares input
        tX, tY, tD, tC ,tA = dh.vectorize_word_dimension(self.validation, self._vocab)
        tX = dh.pad_sequence_1d(tX, maxlen=self._line_maxlen)

        #embedding dimension
        dimension_size = 256

        #solving class imbalance
        ratio = self.calculate_label_ratio(Y)
        ratio = [max(ratio.values()) / value for key, value in ratio.items()]
        print('class ratio::', ratio)

        Y, tY = [np_utils.to_categorical(x) for x in (Y, tY)]

        print('train_X', X.shape)
        print('train_Y', Y.shape)
        print('validation_X',tX.shape)
        print('validation_Y',tY.shape)

        # trainable true if you want word2vec weights to be updated
        # model = self._build_network(len(self._vocab.keys()) + 1, self._line_maxlen, embedding_dimension=dimension_size, trainable=True)
        #
        # open(self._model_file + 'model.json', 'w').write(model.to_json())
        # save_best = ModelCheckpoint(model_file + 'model.json.hdf5', save_best_only=True)
        # save_all = ModelCheckpoint(self._model_file + 'weights.{epoch:02d}__.hdf5',
        #                            save_best_only=False)
        # early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=1)
        #
        # # training
        # model.fit(X, Y, batch_size=8, epochs=100, validation_data=(tX,tY), shuffle=True,
        #           callbacks=[save_best, save_all, early_stopping],class_weight=ratio)


    def load_train_validation_data(self):
        self.train = dh.loaddata(self._train_file, self._emoji_file_path,normalize_text=True,
                              split_hashtag=True,
                              ignore_profiles=False)
        self.validation = dh.loaddata(self._validation_file, self._emoji_file_path, normalize_text=True,
                                   split_hashtag=True,
                                   ignore_profiles=False)

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

    def __init__(self, model_file, vocab_file_path, output_file, input_weight_file_path=None):
        print('initializing...')
        sarcasm_model.__init__(self)

        # self._word_file_path = word_file_path
        self._model_file = model_file
        self._vocab_file_path = vocab_file_path
        self._output_file = output_file
        self._input_weight_file_path = input_weight_file_path

        print('test_maxlen', self._line_maxlen)

    def load_trained_model(self,weight_file='model.json.hdf5'):
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

    def predict(self,test_file,verbose=False):
        try:
            start = time.time()
            self.test = dh.loaddata(test_file, normalize_text=True, split_hashtag=True,
                                 ignore_profiles=True)
            end = time.time()
            if(verbose==True):
                print('test resource loading time::', (end - start))

            self._vocab = self.load_vocab()

            start = time.time()
            tX, tY, tD, tC, tA = dh.vectorize_word_dimension(self.test, self._vocab)
            tX = dh.pad_sequence_1d(tX, maxlen=self._line_maxlen)
            end = time.time()
            if(verbose==True):
                print('test resource preparation time::', (end - start))

            self.__predict_model(tX, self.test)
        except Exception as e:
            print('Error:',e)


    def __predict_model(self,tX, test):
        y = []
        y_pred = []

        prediction_probability = self.model.predict_proba(tX, batch_size=1, verbose=1)

        try:
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



if __name__ == "__main__":
    basepath = os.getcwd()[:os.getcwd().rfind('/')]
    train_file = basepath + '/resource/train/Train_v1.txt'
    validation_file = basepath + '/resource/dev/Dev_v1.txt'
    test_file = basepath + '/resource/test/Test_v1.txt'
    emoji_file_path = basepath + '/resource/emoji_unicode_names_final.txt'

    output_file = basepath + '/resource/text_model/TestResults.txt'
    model_file = basepath + '/resource/text_model/weights/'
    vocab_file_path = basepath + '/resource/text_model/vocab_list.txt'

    #uncomment for training
    tr=train_model(train_file, validation_file, emoji_file_path, model_file, vocab_file_path, output_file)

    # t = test_model(model_file, vocab_file_path, output_file)
    # t.load_trained_model()
    # t.predict(test_file)


