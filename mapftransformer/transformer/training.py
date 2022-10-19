import pandas as pd
import tensorflow as tf
from .loss import loss_fn
from mapftransformer.preprocess import encode_pipeline, get_tokenizer_from_json
import params
MAX_PATH_PER_REQUEST = 20
train_loss = tf.keras.metrics.Mean(name='train_loss')


def train(model, dataset: str, batch_size, optimizer, **kwargs):
    """
    Train the model with the given dataset.
    :param model:       The model to train e.g., our transformer.
    :param dataset:     The csv file that contain the training dataset (routing requests)
    :param batch_size:  The batch size.
    :param optimizer:   The optimizer of the model.
    :param kwargs:      additional arguments to pass for the `encode_pipeline` function
    :return:    None.
    """
    dataset = pd.read_csv(dataset)
    dataset = pd.concat([dataset, dataset], ignore_index=True)
    tokenizer = get_tokenizer_from_json(params.data['tokenizer'])
    for batch in range(0, len(dataset), batch_size):
        inp, requests = encode_pipeline(
            df=dataset,
            batch=batch,
            batch_size=batch_size,
            num_requests=8,
            data_dir=params.data['routes_dir'],
            tokenizer=tokenizer,
            **kwargs
        )

        with tf.GradientTape() as tape:
            # This part does not make sense. Aborting here.
            prediction, _ = model.call([inp.astype(float), requests.astype(float)], training=True)

            loss = loss_fn(prediction=prediction, routing_request=requests)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
