import logging
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel('INFO')

def get_glove_matrix(vocab, glove_path, initial_embedding_np):
    """
    return a glove embedding matrix
    :param initial_embedding_np:
    :return: np array of [V,E]
    """
    ef = open(glove_path, 'r', encoding='utf-8')
    cnt = 0
    vec_array = initial_embedding_np
    old_avg = np.average(vec_array)
    old_std = np.std(vec_array)
    vec_array = vec_array.astype(np.float32)
    new_avg, new_std = 0, 0

    for line in ef.readlines():
        line = line.strip().split(' ')
        word, vec = line[0].lower(), line[1:]
        vec = np.array(vec, np.float32)
        if word in vocab:
            cnt += 1
            word_idx = vocab[word]
            vec_array[word_idx] = vec
            new_avg += np.average(vec)
            new_std += np.std(vec)
    new_avg /= cnt
    new_std /= cnt
    ef.close()
    print('%d known embedding. old mean: %f new mean %f, old std %f new std %f' % (cnt, old_avg,
                                                                                          new_avg, old_std, new_std))
    return vec_array
