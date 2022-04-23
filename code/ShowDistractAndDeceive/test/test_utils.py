import torch
import utils


def test_sentence_decode(inverted_word_map, batch_size):
    prediction = torch.rand([batch_size, 50, 9490])
    sentences = utils.decode_prediction(inverted_word_map, prediction)

    assert len(sentences) == batch_size
    assert all(isinstance(sentence, str) for sentence in sentences)
