from typing import List

import torch
import torchvision
from torch import nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    """
    Encoder.
    """

    def __init__(self, encoded_image_size=14):
        super(Encoder, self).__init__()
        self.enc_image_size = encoded_image_size

        resnet = torchvision.models.resnet101(
            pretrained=True
        )  # pretrained ImageNet ResNet-101

        # Remove linear and pool layers (since we're not doing classification)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d(
            (encoded_image_size, encoded_image_size)
        )

        self.fine_tune()

    def forward(self, images):
        """
        Forward propagation.

        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        out = self.resnet(
            images
        )  # (batch_size, 2048, image_size/32, image_size/32)
        out = self.adaptive_pool(
            out
        )  # (batch_size, 2048, encoded_image_size, encoded_image_size)
        out = out.permute(
            0, 2, 3, 1
        )  # (batch_size, encoded_image_size, encoded_image_size, 2048)
        return out

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.

        :param fine_tune: Allow?
        """
        for p in self.resnet.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune


class Attention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(
            encoder_dim, attention_dim
        )  # linear layer to transform encoded image
        self.decoder_att = nn.Linear(
            decoder_dim, attention_dim
        )  # linear layer to transform decoder's output
        self.full_att = nn.Linear(
            attention_dim, 1
        )  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        alpha = self.attend(encoder_out, decoder_hidden)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(
            dim=1
        )  # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha

    def attend(
        self, encoder_out: torch.FloatTensor, decoder_hidden: torch.FloatTensor
    ) -> torch.FloatTensor:
        """
        Calculate attention weights.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weights
        """
        encoder_attention = self.encoder_att(encoder_out)
        decoder_attention = self.decoder_att(decoder_hidden)
        att = self.full_att(
            self.relu(encoder_attention + decoder_attention.unsqueeze(1))
        ).squeeze(2)
        return self.softmax(att)


class DecoderWithAttention(nn.Module):
    """
    Decoder.
    """

    def __init__(
        self,
        attention_dim,
        embed_dim,
        decoder_dim,
        vocab_size,
        encoder_dim=2048,
        dropout=0.5,
    ):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        """
        super(DecoderWithAttention, self).__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.attention = Attention(
            encoder_dim, decoder_dim, attention_dim
        )  # attention network

        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer
        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(
            embed_dim + encoder_dim, decoder_dim, bias=True
        )  # decoding LSTMCell
        self.init_h = nn.Linear(
            encoder_dim, decoder_dim
        )  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(
            encoder_dim, decoder_dim
        )  # linear layer to find initial cell state of LSTMCell
        self.f_beta = nn.Linear(
            decoder_dim, encoder_dim
        )  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(
            decoder_dim, vocab_size
        )  # linear layer to find scores over vocabulary
        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        """
        Loads embedding layer with pre-trained embeddings.

        :param embeddings: pre-trained embeddings
        """
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        """
        Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).

        :param fine_tune: Allow?
        """
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """

        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        # Flatten image
        encoder_out = encoder_out.view(
            batch_size, -1, encoder_dim
        )  # (batch_size, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        # Sort input data by decreasing lengths; why? apparent below
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(
            dim=0, descending=True
        )
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # Embedding
        embeddings = self.embedding(
            encoded_captions
        )  # (batch_size, max_caption_length, embed_dim)

        # Initialize LSTM state
        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()

        # Create tensors to hold word predicion scores and alphas
        predictions = torch.zeros(
            batch_size, max(decode_lengths), vocab_size
        ).to(DEVICE)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(
            DEVICE
        )

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            attention_weighted_encoding, alpha = self.attention(
                encoder_out[:batch_size_t], h[:batch_size_t]
            )
            gate = self.gate(
                h[:batch_size_t]
            )  # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding
            h, c = self.decode_step(
                torch.cat(
                    [
                        embeddings[:batch_size_t, t, :],
                        attention_weighted_encoding,
                    ],
                    dim=1,
                ),
                (h[:batch_size_t], c[:batch_size_t]),
            )  # (batch_size_t, decoder_dim)
            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, decode_lengths, alphas, sort_ind

    def gate(self, hidden_state):
        """Return the gate value"""
        return self.sigmoid(self.f_beta(hidden_state))


class ShowAttendAndTell(nn.Module):
    """Composite helper class for encoder and decoder."""

    @classmethod
    def load(cls, model_path, word_map, device=None):
        """Load ShowAttendAndTell model"""
        if device == None:
            # Default to cuda if available
            device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device)

        checkpoint = torch.load(model_path, map_location=str(device))
        decoder = checkpoint["decoder"]
        decoder = decoder.to(device)
        decoder.eval()
        encoder = checkpoint["encoder"]
        encoder = encoder.to(device)
        encoder.eval()
        return cls(encoder, decoder, word_map, device)

    def __init__(
        self,
        encoder: Encoder,
        decoder: DecoderWithAttention,
        word_map: dict,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.word_map = word_map

        self._device = device

        # Constants
        self.max_sentence_length = 50
        self._end_token = self.word_map["<end>"]

    def _encoder_forward(
        self, input_img: torch.FloatTensor
    ) -> torch.FloatTensor:
        """ """
        latent_pixels = self.encoder(input_img)
        # Flatten dim 1 and 2 (pixels)
        flattend_latent_pixels = latent_pixels.view(
            latent_pixels.size(0), -1, latent_pixels.size(3)
        )
        return flattend_latent_pixels

    def forward(self, input_img: torch.FloatTensor) -> List[str]:
        """Inference on a batch of images.

        Inputs
        ------
        input: torch.FloatTensor
            shape: (batch, 3, 256, 256)
            All values should be between 0 and 1

        Returns
        ------_
        output: [str]
            The infered strings
        """
        # assert input_img.size(0) == 1, "Only support batch_size of 1 currently."
        # (1, encoder.enc_image_size * encoder.enc_image_size, features)
        latent_pixels = self._encoder_forward(input_img)

        # Initialize start of sentence
        k_prev_words = torch.LongTensor(
            [[self.word_map["<start>"]]] * latent_pixels.size(0)
        ).to(self._device)

        encoded_sentence = k_prev_words
        attention_sentence = torch.ones(
            latent_pixels.size(0), 1, latent_pixels.size(1)
        ).to(self._device)

        decoder_hidden, c = self.decoder.init_hidden_state(latent_pixels)

        # Create tensors to hold word predicion scores and alphas
        predictions = torch.zeros(
            latent_pixels.size(0),
            self.max_sentence_length,
            self.decoder.vocab_size,
        ).to(self._device)

        for i in range(self.max_sentence_length):

            prev_embeddings = self.decoder.embedding(k_prev_words).squeeze(1)
            # Ey_{t-1}

            attention = self.decoder.attention.attend(
                latent_pixels, decoder_hidden
            )
            # alpha

            gate = self.decoder.gate(decoder_hidden)
            # beta

            context_vector = gate * (
                latent_pixels * attention.unsqueeze(2)
            ).sum(dim=1)
            # z_t

            decoder_hidden, c = self.decoder.decode_step(
                torch.cat([prev_embeddings, context_vector], dim=1),
                (decoder_hidden, c),
            )

            logit_scores = self.decoder.fc(decoder_hidden)
            predictions[:, i] = logit_scores
            top_words = logit_scores.argmax(1, keepdim=True)

            encoded_sentence = torch.cat([encoded_sentence, top_words], dim=1)
            attention_sentence = torch.cat(
                [attention_sentence, attention.unsqueeze(1)], dim=1
            )

            # if top_words == self._end_token:
            #     break

            k_prev_words = top_words

        return predictions, i

    def to(self, device):
        self._device = device
        super().to(device)
