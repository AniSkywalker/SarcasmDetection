from collections import defaultdict
import re
from gensim.models.keyedvectors import KeyedVectors
import numpy
from nltk.tokenize import TweetTokenizer
import SarcasmDetection.src.data_processing.glove2Word2vecLoader as glove
from nltk.corpus import brown
from nltk import FreqDist
import itertools


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


def InitializeWords(word_file_path):
    word_dictionary = defaultdict()

    with open(word_file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            tokens = line.lower().strip().split('\t')
            word_dictionary[tokens[0]] = int(tokens[1])

    for alphabet in "bcdefghjklmnopqrstuvwxyz":
        if(alphabet in word_dictionary):
            word_dictionary.__delitem__(alphabet)

    for word in ['di','um']:
        if(word in word_dictionary):
            word_dictionary.__delitem__(word)




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

def load_split_word(split_word_file_path):
    split_word_dictionary = defaultdict()
    with open(split_word_file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            tokens = line.lower().strip().split('\t')
            split_word_dictionary[tokens[0]] = tokens[1:]

def split_hashtags(term, wordlist, split_word_list):
    print(term)
    words = []
    # max freq in brown+gutenberg+reuters corpus
    penalty = -2728
    max_coverage = penalty

    split_words_count = 4
    # checking camel cases
    term = re.sub(r'([0-9]+)', r' \1',term)
    term = re.sub(r'([A-Z][^A-Z ]+)', r' \1',term)
    term = re.sub(r'([A-Z]{2,})+',r' \1',term)
    words= term.strip().split(' ')

    if(len(words)<2):
        # splitting lower case and uppercase words upto 5 words
        chars = [c for c in term.lower()]
        n_splits = 0
        found_all_words = False

        while(n_splits < split_words_count and not found_all_words):
            for idx in itertools.combinations(range(0, len(chars)), n_splits):
                output = numpy.split(chars, idx)
                line = [''.join(o) for o in output]

                score = (1. / len(line)) * sum([wordlist.get(word) if word in wordlist else penalty for word in line])

                if(score > max_coverage):
                    words = line
                    max_coverage = score
                    line_is_valid_word = [word in wordlist for word in line]

                    if(all(line_is_valid_word)):
                        found_all_words = True

            n_splits = n_splits + 1


    # removing hashtag sign
    words = [str(s) for s in words]
    print(words)
    # words = ["#" + str(s) for s in words]
    return words


def filter_text(text, word_list, split_word_list,emoji_dict, normalize_text=False, split_hashtag=False, ignore_profiles=False,
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
            splits = split_hashtags(t[1:], word_list, split_word_list)
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


def parsedata(lines, word_list, split_word_list, emoji_dict, normalize_text=False, split_hashtag=False, ignore_profiles=False,
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
            target_text = filter_text(target_text, word_list, split_word_list,emoji_dict, normalize_text, split_hashtag,
                                      ignore_profiles, replace_emoji=replace_emoji)

            # awc dimensions
            dimensions = []
            if (len(token) > 3 and token[3].strip() != 'NA'):
                dimensions = [dimension.split('@@')[1] for dimension in token[3].strip().split('|')]

            # context tweet
            context = []
            if (len(token) > 4):
                if (token[4] != 'NA'):
                    context = TweetTokenizer().tokenize(token[4].strip())
                    context = filter_text(context, word_list, normalize_text, split_hashtag, ignore_profiles)

            # author
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


def loaddata(filename, word_file_path, split_word_path, emoji_file_path, normalize_text=False, split_hashtag=False,
             ignore_profiles=False,
             lowercase=True, replace_emoji=True):
    word_list = None
    emoji_dict = None

    #load split files
    split_word_list = load_split_word(split_word_path)

    # load word dictionary
    if (split_hashtag):
        word_list = InitializeWords(word_file_path)

    if (replace_emoji):
        emoji_dict = load_unicode_mapping(emoji_file_path)

    lines = open(filename, 'r').readlines()
    data = parsedata(lines, word_list, split_word_list, emoji_dict, normalize_text=normalize_text, split_hashtag=split_hashtag,
                     ignore_profiles=ignore_profiles, lowercase=lowercase, replace_emoji=replace_emoji)
    return data


def build_vocab(data, without_dimension=False, ignore_context=False):
    vocab = defaultdict(int)

    total_words = 1
    if (not without_dimension):
        for i in range(1, 101):
            vocab[str(i)] = total_words
            total_words = total_words + 1

    for sentence_no, token in enumerate(data):
        # print(token[1])

        for word in token[1]:
            if (not word in vocab):
                vocab[word] = total_words
                total_words = total_words + 1

        if (not without_dimension):
            for word in token[2]:
                if (not word in vocab):
                    vocab[word] = total_words
                    total_words = total_words + 1

        if (ignore_context == False):
            for word in token[3]:
                if (not word in vocab):
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
