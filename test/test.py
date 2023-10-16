from transformers import BertModel, BertTokenizer

model = BertModel.from_pretrained("bert-base-uncased",
                                  cache_dir="../model_cache", )

print(model)
fix_layers = 11
learning_rate = 1
lr_layer_decay_rate = 0.95

params = {}
for n, p in model.named_parameters():
    if fix_layers > 0:
        if 'encoder.layer' in n:
            try:
                layer_num = int(n[n.find('encoder.layer') + 14:].split('.')[0])
            except ValueError:
                raise ValueError(f"Unexpected error when going through {n}, check its name")

            if layer_num >= fix_layers:
                print(f'Include {n} to updating para list')
                params[n] = p

        elif 'embeddings' in n:
            print(f'Exclude {n} to updating para list')
        else:
            print(f'Include {n} to updating para list')
            params[n] = p
    else:
        print(f'Doing full fine-tuning')
        params[n] = p

# calculate the lr_factor for each layer
num_layer = model.config.num_hidden_layers
base_factor = 1.0
lr_factor_list = []
for i in range(num_layer):
    lr_factor_list.append(base_factor)
    base_factor *= lr_layer_decay_rate
lr_factor_list = lr_factor_list[::-1]

no_decay = ["bias", "LayerNorm.weight"]

# group the parameters
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in params.items() if not any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
    {
        "params": [p for n, p in params.items() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
]

print(model.embeddings.named_parameters())
