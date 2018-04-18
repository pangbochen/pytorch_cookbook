# coding:utf-8
import sys
import os
import json
import re
import numpy as np

def _parseawData(author=None, constrain=None, src='./chinese-poetry/json/simplified', category='poet.tang'):
    '''
    :param author: name of author
    :param constrain: limitation of poet length
    :param src: file path
    :param category: peot.tang or peot.song
    :return: list
    '''

    def sentenceParse(para):
        result, number = re.subn(u"（.*）", "", para)
        result, number = re.subn(u"{.*}", "", result)
        result, number = re.subn(u"《.*》", "", result)
        result, number = re.subn(u"《.*》", "", result)
        result, number = re.subn(u"[\]\[]", "", result)
        r = ""
        for s in result:
            if s not in set('0123456789-'):
                r += s
        r, number = re.subn(u"。。", u"。", r)
        return r

    def handleJson(file):
        # print file
        rst = []
        data = json.loads(open(file).read())
        for poetry in data:
            pdata = ""
            if (author is not None and poetry.get("author") != author):
                continue
            p = poetry.get("paragraphs")
            flag = False
            for s in p:
                sp = re.split(u"[，！。]", s)
                for tr in sp:
                    if constrain is not None and len(tr) != constrain and len(tr) != 0:
                        flag = True
                        break
                    if flag:
                        break
            if flag:
                continue
            for sentence in poetry.get("paragraphs"):
                pdata += sentence
            pdata = sentenceParse(pdata)
            if pdata != "":
                rst.append(pdata)
        return rst

    data = []
    for filename in os.listdir(src):
        if filename.startswith(category):
            data.extend(handleJson(src + filename))
    return data

def pad_sequences(sequences,
                  maxlen=None,
                  dtype='int32',
                  padding='pre',
                  truncating='pre',
                  value=0.):
    """
    code from keras
    Pads each sequence to the same length (length of the longest sequence).
    If maxlen is provided, any sequence longer
    than maxlen is truncated to maxlen.
    Truncation happens off either the beginning (default) or
    the end of the sequence.
    Supports post-padding and pre-padding (default).
    Arguments:
        sequences: list of lists where each element is a sequence
        maxlen: int, maximum length
        dtype: type to cast the resulting sequence.
        padding: 'pre' or 'post', pad either before or after each sequence.
        truncating: 'pre' or 'post', remove values from sequences larger than
            maxlen either in the beginning or in the end of the sequence
        value: float, value to pad the sequences to the desired value.
    Returns:
        x: numpy array with dimensions (number_of_sequences, maxlen)
    Raises:
        ValueError: in case of invalid values for `truncating` or `padding`,
            or in case of invalid shape for a `sequences` entry.
    """
    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    lengths = []
    for x in sequences:
        if not hasattr(x, '__len__'):
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))
        lengths.append(len(x))

    num_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:  # pylint: disable=g-explicit-length-test
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((num_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if not len(s):  # pylint: disable=g-explicit-length-test
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-maxlen:]  # pylint: disable=invalid-unary-operand-type
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError(
                'Shape of sample %s of sequence at position %s is different from '
                'expected shape %s'
                % (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x

def get_data(opt):
    '''
    :param opt: configer operater
    :return: word2ix, ix2word, dataset
    '''
    if os.path.exists(opt.pickle_path):
        print('data exist')
        data = np.load(opt.pickle_path)
        data, word2ix, ix2word = data['data'], data['word2ix'].item(), data['ix2word'].item()
        return data, word2ix, ix2word

    # handle the json file
    data = _parseawData(opt.author, opt.constrain, opt.data_path, opt.category)
    words = {_word for _sentence in data for _word in _sentence}
    word2ix = {_word: _ix for _ix, _word in enumerate(words)}
    word2ix['<EOP>'] = len(word2ix)  # end
    word2ix['<START>'] = len(word2ix)  # start
    word2ix['</s>'] = len(word2ix)  # blank space
    ix2word = {_ix:_word for _word, _ix in list(word2ix.items())}

    # append <EOP> and <START> for each poem
    for i in range(len(data)):
        data[i] = ['<START>'] + list(data[i]) + ['<EOP>']

    # update new dataset
    # word to ix for  each poem
    new_data = [[word2ix[_word] for _word in _sentence] for _sentence in data]

    # padding for dataset
    pad_data = pad_sequences(new_data, maxlen=opt.maxlen, padding='pre', truncating='post', value=len(word2ix)-1)

    # save file
    np.savez_compressed(opt.pickle_path,
                        data=pad_data,
                        word2ix=word2ix,
                        ix2word=ix2word)
    return pad_data, word2ix, ix2word