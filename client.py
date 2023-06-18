"""
原始代码过于臃肿，包含了很多不必要的代码，此为简洁版本，逻辑上也更加清晰。
"""


class ClientAction:
    LOCAL_TRAINING = "local_training"


# 根据自己算法需要设置ClientConfig中的参数，不必要的参数就不用写入了
class ClientConfig:
    def __init__(self, idx):
        self.idx = idx
        # self.params = None
        # self.params_dict = None
        self.epoch_idx = 0
        # self.params = None

        self.local_model_path = None
        self.global_model_path = None

        self.train_data_idxes = None
        # self.model_type = None
        # self.dataset_type = None
        # self.batch_size = None
        self.lr = None
        # self.train_loader = None
        # self.decay_rate = None
        # self.min_lr = None
        # self.epoch = None
        # self.momentum = None
        # self.weight_decay = None
        # self.local_steps = 20

        self.aggregate_weight = 0.1

        self.train_time = 0
        self.send_time = 0
