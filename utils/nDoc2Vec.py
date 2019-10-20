import logging
from gensim import utils
from gensim import matutils
from gensim.models.doc2vec import Doc2Vec
from numpy import dtype, fromstring
from numpy import float32 as REAL
from six.moves import xrange
logger = logging.getLogger(__name__)

'''
Fix the old Doc2Vec, add intersect_word2vec_format
'''

class nDoc2Vec(Doc2Vec):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def intersect_word2vec_format(self, fname, lockf=0.0, binary=False, encoding='utf8', unicode_errors='strict'):
            """
            Merge the input-hidden weight matrix from the original C word2vec-tool format
            given, where it intersects with the current vocabulary. (No words are added to the
            existing vocabulary, but intersecting words adopt the file's weights, and
            non-intersecting words are left alone.)

            `binary` is a boolean indicating whether the data is in binary word2vec format.

            `lockf` is a lock-factor value to be set for any imported word-vectors; the
            default value of 0.0 prevents further updating of the vector during subsequent
            training. Use 1.0 to allow further training updates of merged vectors.
            """
            overlap_count = 0
            logger.info("loading projection weights from %s" % (fname))
            with utils.smart_open(fname) as fin:
                header = utils.to_unicode(fin.readline(), encoding=encoding)
                vocab_size, vector_size = map(int, header.split())  # throws for invalid file format
                if not vector_size == self.vector_size:
                    raise ValueError("incompatible vector size %d in file %s" % (vector_size, fname))
                    # TOCONSIDER: maybe mismatched vectors still useful enough to merge (truncating/padding)?
                if binary:
                    binary_len = dtype(REAL).itemsize * vector_size
                    for line_no in xrange(vocab_size):
                        # mixed text and binary: read text first, then binary
                        word = []
                        while True:
                            ch = fin.read(1)
                            if ch == b' ':
                                break
                            if ch != b'\n':  # ignore newlines in front of words (some binary files have)
                                word.append(ch)
                        word = utils.to_unicode(b''.join(word), encoding=encoding, errors=unicode_errors)
                        weights = fromstring(fin.read(binary_len), dtype=REAL)
                        if word in self.wv.vocab:
                            overlap_count += 1
                            self.wv.syn0[self.wv.vocab[word].index] = weights
                            self.syn0_lockf[self.wv.vocab[word].index] = lockf  # lock-factor: 0.0 stops further changes
                else:
                    for line_no, line in enumerate(fin):
                        parts = utils.to_unicode(line.rstrip(), encoding=encoding, errors=unicode_errors).split(" ")
                        if len(parts) != vector_size + 1:
                            raise ValueError("invalid vector on line %s (is this really the text format?)" % (line_no))
                        word, weights = parts[0], list(map(REAL, parts[1:]))
                        if word in self.wv.vocab:
                            overlap_count += 1
                            self.wv.syn0[self.wv.vocab[word].index] = weights
            logger.info("merged %d vectors into %s matrix from %s" % (overlap_count, self.wv.syn0.shape, fname))
            
    def load_version2(self, fname, binary=False, encoding='utf8', unicode_errors='strict'):
            """
            Merge the input-hidden weight matrix from the original C word2vec-tool format
            given, where it intersects with the current vocabulary. (No words are added to the
            existing vocabulary, but intersecting words adopt the file's weights, and
            non-intersecting words are left alone.)
            `binary` is a boolean indicating whether the data is in binary word2vec format.
            """
            overlap_count = 0
            logger.info("loading projection weights from %s" % (fname))
            with utils.smart_open(fname) as fin:
                header = utils.to_unicode(fin.readline(), encoding=encoding)
                vocab_size, vector_size = map(int, header.split())  # throws for invalid file format
                if not vector_size == self.vector_size:
                    raise ValueError("incompatible vector size %d in file %s" % (vector_size, fname))
                    # TOCONSIDER: maybe mismatched vectors still useful enough to merge (truncating/padding)?
                if binary:
                    binary_len = dtype(REAL).itemsize * vector_size
                    for line_no in xrange(vocab_size):
                        # mixed text and binary: read text first, then binary
                        word = []
                        while True:
                            ch = fin.read(1)
                            if ch == b' ':
                                break
                            if ch != b'\n':  # ignore newlines in front of words (some binary files have)
                                word.append(ch)
                        word = utils.to_unicode(b''.join(word), encoding=encoding, errors=unicode_errors)
                        weights = fromstring(fin.read(binary_len), dtype=REAL)
                        if word in self.wv.vocab:
                            overlap_count += 1
                            self.wv.syn0[self.wv.vocab[word].index] = weights
                            self.syn0_lockf[self.wv.vocab[word].index] = 0.0  # lock it
                else:
                    for line_no, line in enumerate(fin):
                        parts = utils.to_unicode(line.rstrip(), encoding=encoding, errors=unicode_errors).split(" ")
                        if len(parts) != vector_size + 1:
                            raise ValueError("invalid vector on line %s (is this really the text format?)" % (line_no))
                        word, weights = parts[0], list(map(REAL, parts[1:]))
                        if word in self.wv.vocab:
                            overlap_count += 1
                            self.wv.syn0[self.wv.vocab[word].index] = weights
            logger.info("merged %d vectors into %s matrix from %s" % (overlap_count, self.wv.syn0.shape, fname))