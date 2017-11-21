# from nltk.stem.wordnet import WordNetLemmatizer
import os
import string
import re
import cPickle
from collections import OrderedDict
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nltk.corpus import stopwords as sw
from nltk.corpus import wordnet as wn
from nltk import wordpunct_tokenize
from nltk import WordNetLemmatizer
from nltk import word_tokenize, sent_tokenize
from nltk import pos_tag
from nltk import ne_chunk
from scipy.sparse import hstack
from nltk.stem import SnowballStemmer
import nltk.tag.stanford as st
from datetime import datetime

os.environ[
    "STANFORD_MODELS"] = "/Users/himanshupal/Downloads/stanford-ner-2017-06-09"

# import nltk
# nltk.download()


def load_file(filename):

    all_sms = pd.read_csv(filename, encoding='iso-8859-1')
    all_sms = all_sms.rename(columns={'Label': 'label', 'Message': 'message'})
    return all_sms


def _remove_regex(input_text, regex_pattern, str_replace):
    line = re.sub(regex_pattern, str_replace, input_text, flags=re.I)
    return line

labels = OrderedDict([
        (" __url__ ",
         '(http[s]?://|www)(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'),
        (" __date__ ",
         '([1-2][0-9][0-9][0-9])(.|-)(1[0-2]|0[1-9]|[1-9])(.|-)(2[0-9]|3[0-1]|1[0-9]|0[1-9]|[1-9])'),
        (" __time__ ", '((2[0-3]|1[0-9]|0[0-9]|[0-9]):([0-5][0-9]))|hrs|doj|mins'),
        (" __day__ ", '(tomorrow|yesterday|today)'),
        # (" __2wheeler__ ", '(motor|bike|scooter|cycle)'),
        # (" __4wheeler__ ", '(cab|car|hyundai|maruti|uber|ola|meru|taxi)'),
        # (" __8wheeler__ ", '(bus|volvo)'),
        # (" __train__ ", '(rajdhani|duronto|shatabdi|pnr)'),
        # (" __byair__ ", '(airport|air\s*asia|\b([A-Z]{2}|[A-Z]\d|\d[A-Z])\s?\d{3,4}\b|air\s*india|spice\s*jet|indigo|jet\s+airways|flt)'),
        # (" __alpha__ ", '([A-Za-z]+[0-9]|[0-9]+[A-Za-z])[A-Za-z0-9]*'),
        # (" __money__ ", '|'.join([
        #                           r'^(rs)?(\d*\.\d{1,2})$',  # e.g., $.50, .50, $1.50, $.5, .5
        #                           r'^(rs)?(\d+)$',           # e.g., $500, $5, 500, 5
        #                           r'^(rs)(\d+\.?)$',         # e.g., $5.
        #                         ])),
        # (" __trip__ ", "(journey|trip(s)?|makemytrip|yatra|goibibo)"),
        # (" __shopping__ ", '(flipkart|amazon|snapdeal|foodpanda|swiggy|bigbazaar)'),
        (" __digit__ ", '\d+'),
        # money: '',
    ])

def extract_labels(sentence):
    global labels
    for val, pat in labels.items():
        sentence = _remove_regex(sentence, pat, val)

    # sentence = re.sub("(__digit__(\s|__digit__)*__digit__)",
    #                   "__digit__", sentence)
    # sentence = re.sub("(__date__(\s|__date__)*__date__)", "__date__", sentence)
    return sentence



punct = set(string.punctuation)
lemmatizer = WordNetLemmatizer()
ner_tag_type = {
    'class4': 'english.conll.4class.distsim.crf.ser.gz',
    'class7': 'english.muc.7class.distsim.crf.ser.gz',
    'class3': 'english.all.3class.distsim.crf.ser.gz',
}
pos_tag_dict = {
        'N': wn.NOUN,
        'V': wn.VERB,
        'R': wn.ADV,
        'J': wn.ADJ
    }
ner_classes = ('LOCATION', 'DATE', 'ORGANIZATION',
               'PERSON', 'TIME', 'MONEY', 'PERCENT')
cnt = 0
cur_time = datetime.now()
stemmer = SnowballStemmer("english")
import enchant
eng_vocab = enchant.Dict("en_US")

def tokenize(sentence, is_lower=True, is_strip=True, stopwords=None,
             punct_removal=True, to_extract=True, stemming=True,
             stf=None, len_limit=2, with_pos_tag=False):
    # import pdb;pdb.set_trace()
    global cnt, punct, lemmatizer, ner_tag_type, pos_tag_dict, ner_classes, cur_time, stemmer, eng_vocab
    cnt += 1
    if cnt%100 == 0:
        print cnt, datetime.now() - cur_time
        cur_time = datetime.now()
    # todo incorporate POS features like number of nouns/adverbs/adj/verbs
    sentence = extract_labels(sentence) if to_extract else sentence


    tokenized_sent = wordpunct_tokenize(sentence)
    # {k.lower(): tag for k, tag in stf.tag(tokenized_sent)}

    feature = []
    prev_feature = None
    for token, tag in pos_tag(tokenized_sent):

    # for token in tokenized_sent:
        token = token.lower() if is_lower else token
        token = token.strip() if is_strip else token
        # token = token.strip('_') if is_strip else token
        # token = token.strip('*') if is_strip else token

        # If stopword, ignore token and continue
        if token in stopwords:
            continue

        # If punctuation, ignore token and continue
        if punct_removal and all(char in punct for char in token):
            continue

        # import pdb;pdb.set_trace()
        # if with_pos_tag:
        #     lemma = tag
        #     prev_feature = token

        if len(token) <= len_limit:
            continue

        # print "NER", ne_chunk((token.encode('iso-8859-1'), tag)) gives nothing

        ner_sent_dict = {} #dict(stf[cnt-1]) if stf else {}
        if token in ner_sent_dict and ner_sent_dict[token] in ner_classes:
            # NER Tagger
            lemma = ner_sent_dict[token].lower()
            # check for removing duplicate classes like org org org
        # elif stemming:
        #     lemma = stemmer.stem(token)
        else:
            # if tag == 'NNP':
            #     lemma = 'entity'
            # else:
            # Lemmatize the token and yield
                lemma = lemmatizer.lemmatize(token, pos_tag_dict.get(
                    tag[0], wn.NOUN)) if lemmatizer else token

        #stop propernoun removal and removal non english words
        if (" %s "%token not in labels.keys()) and (not eng_vocab.check(token)) and (tag != 'NNP'):
            # print "not exist", token
            continue

        if len(lemma) <= len_limit:
            # print "limit", token
            continue

        if (prev_feature == lemma):
            continue

        prev_feature = lemma

        feature.append(lemma)

    return " ".join(feature)


def save_model(clf, filename):

    with open('%s.pkl' % (filename), 'wb') as fid:
        cPickle.dump(clf, fid)
    print "Model saved..."


def load_model(filename):
    clf = None
    with open('%s.pkl' % (filename), 'rb') as fid:
        clf = cPickle.load(fid)

    print "Model loaded..."
    return clf


vectorizer = TfidfVectorizer("english")


def svd_tfidf_matrix(matrix):
    svd = TruncatedSVD(n_components=100)
    return svd.fit_transform(matrix)


def train_svc(features, correct_labels):
    # Todo generate uniform random samples

    train_features = vectorizer.fit_transform(features)
    # train_features = svd_tfidf_matrix(train_features)
    # train_features=hstack((own,abovetfidf))
    # display_scores(vectorizer, train_features)
    print train_features.shape

    features_train, features_test, labels_train, labels_test = train_test_split(
        train_features, correct_labels, test_size=0.3, random_state=111)

    # can use multiple classifiers here
    svc = SVC(kernel='sigmoid', gamma=1.0)
    svc.fit(features_train, labels_train)

    prediction = svc.predict(features_test)
    print('Accuracy score: {}'.format(accuracy_score(labels_test, prediction)))
    print('Precision score: {}'.format(
        precision_score(labels_test, prediction, average=None)))
    print('Recall score: {}'.format(recall_score(labels_test, prediction, average=None)))
    print('F1 score: {}'.format(f1_score(labels_test, prediction, average=None)))

    save_model(svc, "SVC")
    return svc


def test_svc(test_features, svc=None):
    # vectorizer = TfidfVectorizer("english")
    if not svc:
        svc = load_model("SVC")
    testing_features = vectorizer.transform(test_features)
    # testing_features = svd_tfidf_matrix(testing_features)
    test_prediction = svc.predict(testing_features)

    # generate csv
    test_pred_df = pd.DataFrame(data=test_prediction,
                                index=range(1,
                                            len(test_prediction) + 1))
    test_pred_df.to_csv('test_pred_svc.csv', sep=',')


def get_ner_sentences(sent_list):
    tagger_class = "/Users/himanshupal/Downloads/stanford-ner-2017-06-09/classifiers/%s" % (
        ner_tag_type['class7'])

    stf = st.StanfordNERTagger(
        tagger_class, "/Users/himanshupal/Downloads/stanford-ner-2017-06-09/stanford-ner.jar")

    tokenized_sents = [word_tokenize(sent) for sent in sent_list]
    ner_sents = stf.tag_sents(tokenized_sents)
    # ner_sents = [ne_chunk(sent) for sent in sent_list]
    # import pdb;pdb.set_trace()
    return ner_sents

def save_feature(train_features, filename):
    train_features.to_csv('%s.csv'%filename, sep=',', encoding='iso-8859-1')


def main():
    global ner_tag_type, cnt
    noise_list = set(["is", "a", "this", 'of', 'the', 'in', 'for', 'at'])

    train_df = load_file("train_sms.csv")
    features, feat_labels = train_df['message'], train_df['label']

    stopwords = noise_list or set(sw.words('english'))
    with_pos_tag = False
    stf = get_ner_sentences(train_df['message'].values.T.tolist())
    train_features = features.apply(tokenize, stopwords=noise_list, punct_removal=False, stf=stf, with_pos_tag=with_pos_tag)
    save_feature(train_features, "SVC_features_%s" % (with_pos_tag))

    svc = train_svc(train_features, feat_labels)

    test_df = load_file("final_test.csv")
    cnt = 0
    stf = get_ner_sentences(test_df['message'].values.T.tolist())
    test_features = test_df['message'].apply(tokenize, stopwords=noise_list,punct_removal=False, stf=stf, with_pos_tag=with_pos_tag)
    test_svc(test_features, svc)
    # for line in features.values.T.tolist():
    #     print tokenize(line, stopwords=noise_list)


def display_scores(vectorizer, tfidf_result):
    # http://stackoverflow.com/questions/16078015/
    scores = zip(vectorizer.get_feature_names(),
                 np.asarray(tfidf_result.sum(axis=0)).ravel())
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    for item in sorted_scores:
        print "{0:50} Score: {1}".format(item[0].encode("iso-8859-1"), item[1])


if __name__ == '__main__':

    main()
