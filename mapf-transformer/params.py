transformer = {
    'num_layers': 6,
    'd_model': 160,
    'num_heads': 8,
    'dff': 512,
    'rate': 0.1,
}

optimizer = {
    'beta_1': 0.9,
    'beta_2': 0.98,
    'epsilon': 1e-9,
}

train = {
    "epochs": 20,
    "batch_size": 64,
}

data_dir = 'data_new'

data = {
    "vocab_size": 57196,
    "dir": data_dir,
    "routes_dir": f'{data_dir}/routes_by_request',
    "tokenizer": f'{data_dir}/tokenizer.json'
}
