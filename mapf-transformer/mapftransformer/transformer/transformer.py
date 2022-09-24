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
        # Keras models prefer if you pass all your inputs in the first argument
        # inp, tar = inputs
        #
        # padding_mask, look_ahead_mask = self.create_masks(inp, tar)
        #
        # enc_output = self.encoder(inp, training, padding_mask)  # (batch_size, inp_seq_len, d_model)
        #
        # # dec_output.shape == (batch_size, tar_seq_len, d_model)
        # dec_output, attention_weights = self.decoder(
        #     tar, enc_output, training, look_ahead_mask, padding_mask)
        #
        # final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)
        #
        # return final_output, attention_weights
        # TODO: the encoder and decoder cannot depend on the routing requests!! only when calculating the loss
        routes, routing_requests = inputs
        # padding_mask, look_ahead_mask = self.create_masks(routes, routing_requests)
        enc_output = self.encoder(routes, training, None)
        dec_output, attention_weights = self.decoder(
            routing_requests, enc_output, training, None, None
        )
        final_output = self.final_layer(dec_output)
        return final_output, attention_weights

    @staticmethod
    def create_padding_mask(seq):
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

        # add extra dimensions to add the padding
        # to the attention logits.
        return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

    @staticmethod
    def create_look_ahead_mask(size):
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask  # (seq_len, seq_len)

    def create_masks(self, inp, tar):
        # Encoder padding mask (Used in the 2nd attention block in the decoder too.)
        padding_mask = self.create_padding_mask(inp)

        # Used in the 1st attention block in the decoder.
        # It is used to pad and mask future tokens in the input received by
        # the decoder.
        look_ahead_mask = self.create_look_ahead_mask(tf.shape(tar)[1])
        dec_target_padding_mask = self.create_padding_mask(tar)
        look_ahead_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

        return padding_mask, look_ahead_mask

    # @tf.function(input_signature=train_step_signature)
    # def train_step(self, inp):
    #     # input is routing request shape  (batch_size,
    #     tar_inp = tar[:, :-1]
    #     tar_real = tar[:, 1:]
    #
    #     with tf.GradientTape() as tape:
    #         predictions, _ = self.call([inp, tar_inp],
    #                                    training=True)
    #         loss = loss_fn(tar_real, predictions)
    #
    #     gradients = tape.gradient(loss, self.trainable_variables)
    #     optimizer.apply_gradients(zip(gradients, self.trainable_variables))
    #
    #     train_loss(loss)
    #     train_accuracy(accuracy_function(tar_real, predictions))


