import os
import tensorflow as tf

from mapftransformer import routes, routing_requests, route_encoder
from mapftransformer.preprocess import RouteTokenizer, get_tokenizer_from_json
from mapftransformer.transformer import Transformer, train_loss, CustomSchedule, train
import params

VOCAB_SIZE = 197932

VOCAB_SIZE_NEW = 57196  # For warehouse 10X10 with 5 sources & 5 destinations


def main():
    make_data = False
    create_tokenizer = False
    data_dir = 'data_new'
    routes_dir = f"{data_dir}/routes_by_request"
    num_routes = params.data["vocab_size"]
    if make_data:
        os.mkdir(data_dir)
        # creating routing requests and routes by routing requests csv files:
        # The size of datasets can be altered by the `size` param in each function.
        routes.create_csv_files(routes_dir)
        routing_requests.create_csv_file(data_dir)

        # Encoding routes:
        num_routes = route_encoder.create_encodings_from_routes(routes_dir, routes_dir)

    if create_tokenizer:
        # Create tokenizer:
        tokenizer = RouteTokenizer(num_words=num_routes)
        tokenizer.fit_on_texts(f'{routes_dir}/encoding.txt')
        tokenizer.write_json_file(f'{data_dir}/tokenizer.json')
    else:
        # Can also use a saved one:
        tokenizer = get_tokenizer_from_json(f'{data_dir}/tokenizer.json')

    print(num_routes)
    learning_rate = CustomSchedule(params.transformer['d_model'])
    # Create th transformer model:
    transformer = Transformer(
        input_vocab_size=num_routes,
        target_vocab_size=num_routes,
        **params.transformer
    )
    optimizer = tf.keras.optimizers.Adam(
            learning_rate,
            beta_1=0.9,
            beta_2=0.98,
            epsilon=1e-9
        )
    checkpoint_path = './checkpoints/train'
    ckpt = tf.train.Checkpoint(
        transformer=transformer,
        optimizer=optimizer
    )

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')

    train(
        model=transformer,
        dataset="dummy_dataset.csv",
        batch_size=2,
        optimizer=optimizer,
    )


if __name__ == '__main__':
    main()
