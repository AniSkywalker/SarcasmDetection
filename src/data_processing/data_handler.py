from collections import defaultdict
import re
from gensim.models.keyedvectors import KeyedVectors
import numpy
from nltk.tokenize import TweetTokenizer
import SarcasmDetection.src.data_processing.glove2Word2vecLoader as glove
from nltk.corpus import words
import itertools

'''
nltk words corpus is needed
'''


# loading the emoji dataset
def load_unicode_mapping(path):
    emoji_dict = defaultdict()
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            tokens = line.strip().split('\t')
            emoji_dict[tokens[0]] = tokens[1]
    return emoji_dict


def load_word2vec(path=None):
    word2vecmodel = KeyedVectors.load_word2vec_format(path, binary=True)
    return word2vecmodel


def InitializeWords():
    word_dictionary = set(words.words())
    for alphabet in "bcdefghjklmnopqrstuvwxyz":
        word_dictionary.remove(alphabet)
    return word_dictionary


def normalize_word(word):
    temp = word
    while True:
        w = re.sub(r"([a-zA-Z])\1\1", r"\1\1", temp)
        if (w == temp):
            break
        else:
            temp = w
    return w


def split_ht(term, wordlist):
    words = []
    # checking camel cases
    if (term != term.lower() and term != term.upper()):
        words = re.findall('[A-Z][^A-Z]*', term)
    else:
        # splitting lower case and uppercase words
        chars = [c for c in term.lower()]
        outputs = [numpy.split(chars, idx) for n_splits in range(5) for idx in
                   itertools.combinations(range(0, len(chars)), n_splits)]

        for output in outputs:
            line = [''.join(o) for o in output]
            if (all([word in wordlist for word in line])):
                '''
                TODO pruning best split based on ngram collocation
                '''
                words = line
                break

    # removing hashtag sign
    words = [str(s) for s in words]
    # words = ["#" + str(s) for s in words]
    return words


def filter_text(text, word_list, emoji_dict, normalize_text=False, split_hashtag=False, ignore_profiles=False,
                replace_emoji=True):
    filtered_text = []

    for t in text:
        splits = None

        # ignoring profile information if ignore_profiles is set
        if (ignore_profiles and str(t).startswith("@")):
            continue

        # ignoring links
        if (str(t).startswith('http')):
            continue

        # ignoring #sarcasm
        if (str(t).lower() == '#sarcasm'):
            continue

        # replacing emoji with its unicode description
        if (replace_emoji):
            if (t in emoji_dict):
                t = emoji_dict.get(t).split('_')
                filtered_text.extend(t)
                continue

        # splitting hastags
        if (split_hashtag and str(t).startswith("#")):
            splits = split_ht(t[1:], word_list)
            # adding the hashtags
            if (splits != None):
                filtered_text.extend([s for s in splits if (not filtered_text.__contains__(s))])
                continue

        # removes repeatation of letters
        if (normalize_text):
            t = normalize_word(t)

        # appends the text
        filtered_text.append(t)

    return filtered_text


def parsedata(lines, word_list, emoji_dict, normalize_text=False, split_hashtag=False, ignore_profiles=False,
              lowercase=False, replace_emoji=True):
    data = []
    for i, line in enumerate(lines):
        if (i % 100 == 0):
            print(str(i))

        try:
            # convert the line to lowercase
            if (lowercase):
                line = line.lower()

            # split into token
            token = line.split('\t')

            # label
            label = int(token[1].strip())

            # tweet text
            target_text = TweetTokenizer().tokenize(token[2].strip())

            # filter text
            target_text = filter_text(target_text, word_list, emoji_dict, normalize_text, split_hashtag,
                                      ignore_profiles, replace_emoji=replace_emoji)
            # print(filtered_text)


            dimensions = []
            if (len(token) > 3 and token[3].strip() != 'NA'):
                dimensions = [dimension.split('@@')[1] for dimension in token[3].strip().split('|')]

            context = []
            if (len(token) > 4):
                if (token[4] != 'NA'):
                    context = TweetTokenizer().tokenize(token[4].strip())
                    context = filter_text(context, word_list, normalize_text, split_hashtag, ignore_profiles)

            author = 'NA'
            if (len(token) > 5):
                author = token[5]

            if (len(target_text) != 0):
                # print((label, target_text, dimensions, context, author))
                data.append((label, target_text, dimensions, context, author))
        except:
            raise
    print('')
    return data


def loaddata(filename, emoji_file_path, normalize_text=False, split_hashtag=False, ignore_profiles=False,
             lowercase=True, replace_emoji=True):
    word_list = None
    emoji_dict = None

    if (split_hashtag):
        word_list = InitializeWords()

    if (replace_emoji):
        emoji_dict = load_unicode_mapping(emoji_file_path)

    lines = open(filename, 'r').readlines()
    data = parsedata(lines, word_list, emoji_dict, normalize_text=normalize_text, split_hashtag=split_hashtag,
                     ignore_profiles=ignore_profiles, lowercase=lowercase, replace_emoji=replace_emoji)
    print('Loading finished...')
    return data


def build_vocab(data, without_dimension=False, ignore_context=False):
    vocab = defaultdict(int)

    st_words = set()

    total_words = 1
    if (not without_dimension):
        for i in range(1, 101):
            vocab[str(i)] = total_words
            total_words = total_words + 1

    for sentence_no, token in enumerate(data):
        # print(token[1])

        for word in token[1]:
            if (st_words.__contains__(word.lower())):
                continue
            if (not word in vocab):
                vocab[word] = total_words
                total_words = total_words + 1

        if (not without_dimension):
            for word in token[2]:
                if (st_words.__contains__(word.lower())):
                    continue
                if (not word in vocab):
                    vocab[word] = total_words
                    total_words = total_words + 1

        if (ignore_context == False):
            for word in token[3]:
                if (not word in vocab):
                    if (st_words.__contains__(word.lower())):
                        continue
                    vocab[word] = total_words
                    total_words = total_words + 1
    return vocab


def build_reverse_vocab(vocab):
    rev_vocab = defaultdict(str)
    for k, v in vocab.items():
        rev_vocab[v] = k
    return rev_vocab


def vectorize_word_dimension(data, vocab, drop_dimension_index=None):
    X = []
    Y = []
    D = []
    C = []
    A = []

    known_words_set = set()
    unknown_words_set = set()

    for label, line, dimensions, context, author in data:
        vec = []
        context_vec = []
        if (len(dimensions) != 0):
            dvec = [vocab.get(d) for d in dimensions]
        else:
            dvec = [vocab.get('unk')] * 11

        if drop_dimension_index != None:
            dvec.pop(drop_dimension_index)

        for words in line:
            if (words in vocab):
                vec.append(vocab[words])
                known_words_set.add(words)
            else:
                vec.append(vocab['unk'])
                unknown_words_set.add(words)
        if (len(context) != 0):
            for words in line:
                if (words in vocab):
                    context_vec.append(vocab[words])
                    known_words_set.add(words)
                else:
                    context_vec.append(vocab['unk'])
                    unknown_words_set.add(words)
        else:
            context_vec = [vocab['unk']]

        X.append(vec)
        Y.append(label)
        D.append(dvec)
        C.append(context_vec)
        A.append(author)

    print('Word coverage:', len(unknown_words_set) / float(len(known_words_set) + len(unknown_words_set)))

    return numpy.asarray(X), numpy.asarray(Y), numpy.asarray(D), numpy.asarray(C), numpy.asarray(A)


def pad_sequence_1d(sequences, maxlen=None, dtype='float32', padding='pre', truncating='pre', value=0.):
    X = [vectors for vectors in sequences]

    nb_samples = len(X)

    x = (numpy.ones((nb_samples, maxlen)) * value).astype(dtype)

    for idx, s in enumerate(X):
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError("Truncating type '%s' not understood" % padding)

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError("Padding type '%s' not understood" % padding)

    return x


def write_vocab(filepath, vocab):
    with open(filepath, 'w') as fw:
        for key, value in vocab.items():
            fw.write(str(key) + '\t' + str(value) + '\n')


def get_word2vec_weight(vocab, n=300, path=None):
    word2vecmodel = load_word2vec(path=path)
    emb_weights = numpy.zeros((len(vocab.keys()) + 1, n))
    for k, v in vocab.items():
        if (word2vecmodel.__contains__(k)):
            emb_weights[v, :] = word2vecmodel[k][:n]

    return emb_weights


def load_glove_model(vocab, n=200):
    word2vecmodel = glove.load_glove_word2vec('/home/glove/glove.twitter.27B/glove.twitter.27B.200d.txt')

    emb_weights = numpy.zeros((len(vocab.keys()) + 1, n))
    for k, v in vocab.items():
        if (word2vecmodel.__contains__(k)):
            emb_weights[v, :] = word2vecmodel[k][:n]

    return emb_weights
