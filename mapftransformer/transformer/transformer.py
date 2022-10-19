import tensorflow as tf
from .blocks import Encoder, Decoder
from .loss import loss_fn

train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]


class Transformer(tf.keras.Model):
    def __init__(
            self,
            *,
            num_layers,
            d_model,
            num_heads,
            dff,
            input_vocab_size,
            target_vocab_size,
            rate=0.1
    ):
        """
        Our transformer model.
        Based on the classic transformer that presented in the paper: Attention Is All You Need.

        :param num_layers:          The number of layers in the transformer.
                                    This is the number of encoders and the number of decoders within.
        :param d_model:             The output's dimension.
        :param num_heads:           Number of heads in the multi-head attention. how much parallelized it will be.
        :param dff:                 Dimensionality of the output space for the feed-forward network.
        :param input_vocab_size:    The vocabulary size of the tokenizer (the size of the route dataset).
        :param target_vocab_size:   The vocabulary size of the tokenizer (the size of the route dataset).
                                    Maybe can be minimized to just the relevant ones?
        :param rate:                The rate for the dropout layer (percent to drop - a float number between 0 and 1).
        """
        super().__init__()
        self.encoder = Encoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            input_vocab_size=input_vocab_size,
            rate=rate
        )

        self.decoder = Decoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            target_vocab_size=target_vocab_size,
            rate=rate
        )

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inputs, training):
        """
        Passing the input through all the parts of the transformer.
        :param inputs:      numpy array of tokenized routes and the targets (There are no targets in our case).
        :param training:    A boolean flag for training mode.
        :return:    A tuple of the plan of routes and the weights of the attention.
        """

        # TODO: the encoder and decoder cannot depend on the routing requests!!
        #  only when calculating the loss
        routes, routing_requests = inputs
        # padding_mask, look_ahead_mask = self.create_masks(routes, routing_requests)
        enc_output = self.encoder(routes, training, None)
        # cant pass to the decoder something good. Abort.
        dec_output, attention_weights = self.decoder(
            routing_requests, enc_output, training, None, None
        )
        final_output = self.final_layer(dec_output)
        return final_output, attention_weights

    @staticmethod
    def create_padding_mask(seq):
        """
        Not working.
        Creating the padding mask.
        :param seq: The sequence, meaning to the input.
        :return:    The padding mask.
        """
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

        # add extra dimensions to add the padding
        # to the attention logits.
        return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

    @staticmethod
    def create_look_ahead_mask(size):
        """
        Not working.
        The creation of the look ahead mask to prevent of be dependent on the upcoming targets.
        :param size:    The size of the mask matrix.
        :return:    The mask.
        """
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask  # (seq_len, seq_len)

    def create_masks(self, inp, tar):
        """
        Not working.
        Creating both the look ahead mask and the padding mask.
        :param inp: The input.
        :param tar: The targets. (There are no targets in our case)
        :return:
        """
        # Encoder padding mask (Used in the 2nd attention block in the decoder too.)
        padding_mask = self.create_padding_mask(inp)

        # Used in the 1st attention block in the decoder.
        # It is used to pad and mask future tokens in the input received by
        # the decoder.
        look_ahead_mask = self.create_look_ahead_mask(tf.shape(tar)[1])
        dec_target_padding_mask = self.create_padding_mask(tar)
        look_ahead_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

        return padding_mask, look_ahead_mask
