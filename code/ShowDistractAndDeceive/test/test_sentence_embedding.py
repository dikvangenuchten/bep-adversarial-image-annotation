import numpy as np
import pytest

from sentence_embedding import cosine_similarity


def test_cosine_similarity_on_identical_sentence():
    sentence = "a group of stuffed animals sitting on top of a couch"
    np.testing.assert_allclose(cosine_similarity([sentence], [sentence]), 1)


def test_cosine_similarity_on_almost_identical_sentences():
    orig_sentence = "a group of stuffed animals sitting on top of a couch"
    adv_sentence_1 = "a group of teddy bears sitting on top of a couch"
    adv_sentence_2 = "a group of men are sitting on top of a couch"
    adv_sentence_3 = "on top of a couch a group of stuffed animals are sitting"
    adv_sentence_4 = (
        "a pair of shoes and a black shirt and a black and white shirt"
    )

    or_vs_ad1 = cosine_similarity([orig_sentence], [adv_sentence_1])
    or_vs_ad2 = cosine_similarity([orig_sentence], [adv_sentence_2])
    or_vs_ad3 = cosine_similarity([orig_sentence], [adv_sentence_3])
    or_vs_ad4 = cosine_similarity([orig_sentence], [adv_sentence_4])

    # From similar to less similar
    assert or_vs_ad3 > or_vs_ad1 > or_vs_ad2 > or_vs_ad4


@pytest.mark.parametrize("batch_size", [1, 2, 16])
def test_batch_similarity_calculation(batch_size):
    sentence = "a group of stuffed animals sitting on top of a couch"
    np.testing.assert_allclose(
        cosine_similarity([sentence] * batch_size, [sentence] * batch_size),
        [1] * batch_size,
    )
