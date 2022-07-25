import json
""" configuration json을 읽어들이는 class """
class Config(dict): 
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    @classmethod
    def load(cls, file):
        with open(file, 'r') as f:
            config = json.loads(f.read())
            return Config(config)


# # 기본 파라미터
# config = Config({
#     "n_enc_vocab": 0,
#     "n_dec_vocab": 0,
#     "n_enc_seq": 512,
#     "n_dec_seq": 512,
#     "n_layer": 12,
#     "d_hidn": 768,
#     "i_pad": 0,
#     "d_ff": 3072,
#     "n_head": 12,
#     "d_head": 64,
#     "dropout": 0.1,
#     "layer_norm_epsilon": 1e-12,
#     "n_output": 2,
#     "weight_decay": 0,
#     "learning_rate": 5e-5,
#     "adam_epsilon": 1e-8,
#     "warmup_steps": 0
# })