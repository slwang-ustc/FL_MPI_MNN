import os
import asyncio
from typing import List
from comm_utils import send_data, get_data
from config import cfg
import time
import random
from random import sample
import numpy as np
from client import ClientConfig
import datasets.utils as dataset_utils
import models.utils as model_utils
from training_utils import test
from mpi4py import MPI
import logging
import MNN

F = MNN.expr

random.seed(cfg['client_selection_seed'])

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = cfg['server_cuda']
# device = torch.device("cuda" if cfg['server_use_cuda'] and torch.cuda.is_available() else "cpu")

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
csize = comm.Get_size()

now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
RESULT_PATH = os.getcwd() + '/server_log/'
MODEL_PATH = os.getcwd() + '/model_save/' + now + '/'
GLOBAL_MODEL_PATH = MODEL_PATH + now + "_global.mnn"
if not os.path.exists(RESULT_PATH):
    os.makedirs(RESULT_PATH, exist_ok=True)
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH, exist_ok=True)
# init logger
logger = logging.getLogger(os.path.basename(__file__).split('.')[0])
logger.setLevel(logging.INFO)
filename = RESULT_PATH + now + "_" + os.path.basename(__file__).split('.')[0] + '.log'
fileHandler = logging.FileHandler(filename=filename)
formatter = logging.Formatter("%(message)s")
fileHandler.setFormatter(formatter)
logger.addHandler(fileHandler)

comm_tags = np.ones(cfg['client_num'] + 1)


def main():
    client_num = cfg['client_num']
    logger.info("Total number of clients: {}".format(client_num))
    logger.info("\nModel type: {}".format(cfg["model_type"]))
    logger.info("Dataset: {}".format(cfg["dataset_type"]))

    # init the global model
    global_model = model_utils.create_model_instance(cfg['model_type'])
    F.save(global_model.parameters, GLOBAL_MODEL_PATH)

    # partition the dataset
    train_data_partition, partition_sizes = partition_data(
        dataset_type=cfg['dataset_type'],
        partition_pattern=cfg['data_partition_pattern'],
        non_iid_ratio=cfg['non_iid_ratio'],
        client_num=client_num
    )

    logger.info('\nData partition: ')
    for i in range(len(partition_sizes)):
        s = ""
        for j in range(len(partition_sizes[i])):
            s += "{:.2f}".format(partition_sizes[i][j]) + " "
        logger.info(s)

    # load the test dataset and test loader
    _, test_dataset = dataset_utils.load_datasets(cfg['dataset_type'], cfg['dataset_path'])
    test_loader = dataset_utils.create_dataloaders(test_dataset, batch_size=cfg['test_batch_size'], shuffle=False)

    # create clients
    all_clients: List[ClientConfig] = list()
    for client_idx in range(client_num):
        client = ClientConfig(client_idx)
        client.lr = cfg['lr']
        client.train_data_idxes = train_data_partition.use(client_idx)
        client.local_model_path = MODEL_PATH + now + "_local_" + str(client_idx) + ".mnn"
        client.global_model_path = GLOBAL_MODEL_PATH
        all_clients.append(client)

    best_epoch = 1
    best_auc = 0
    # begin each epoch
    for epoch_idx in range(1, 1 + cfg['epoch_num']):
        logger.info("_____****_____\nEpoch: {:04d}".format(epoch_idx))
        print("_____****_____\nEpoch: {:04d}".format(epoch_idx))

        # The client selection algorithm can be implemented
        selected_num = 10
        selected_client_idxes = sample(range(client_num), selected_num)
        logger.info("Selected client idxes: {}".format(selected_client_idxes))
        print("Selected client idxes: {}".format(selected_client_idxes))
        selected_clients = []
        for client_idx in selected_client_idxes:
            all_clients[client_idx].epoch_idx = epoch_idx
            selected_clients.append(all_clients[client_idx])

        # send the configurations to the selected clients
        communication_parallel(selected_clients, action="send_config")

        # when all selected clients have completed local training, receive their configurations
        communication_parallel(selected_clients, action="get_config")

        # aggregate the clients' local model parameters
        aggregate_models(global_model, selected_clients)

        # test the global model
        test_loss, test_acc, test_auc = test(global_model, test_loader, None)
        if test_auc > best_auc:
            best_auc = test_auc
            best_epoch = epoch_idx
            model_save_path = MODEL_PATH + now + "_" + "best_global.mnn"
            params = list()
            for p in global_model.parameters:
                params.append(F.clone(p, True))
            for i in params:
                i.fix_as_const()
            F.save(params, model_save_path)
        logger.info(
            "Test_Loss: {:.4f}\n".format(test_loss) +
            "Test_ACC: {:.4f}\n".format(test_acc) +
            "Test_AUC: {:.4f}\n".format(test_auc) +
            "Best_AUC: {:.4f}\n".format(best_auc) +
            "Best_Epoch: {:04d}\n".format(best_epoch)
        )

        for m in range(len(selected_clients)):
            comm_tags[m + 1] += 1


def aggregate_models(global_model, client_list):
    params = list()
    for p in global_model.parameters:
        params.append(p)

    for i, client in enumerate(client_list):
        for k, v in enumerate(F.load_as_list(client.local_model_path)):
            params[k] += client.aggregate_weight * (v - global_model.parameters[k])
    
    for i in params:
        i.fix_as_const()
    F.save(params, GLOBAL_MODEL_PATH)
    global_model.load_parameters(params)
    for i in global_model.parameters:
        i.fix_as_trainable()


async def send_config(client, client_rank, comm_tag):
    await send_data(comm, client, client_rank, comm_tag)


async def get_config(client, client_rank, comm_tag):
    config_received = await get_data(comm, client_rank, comm_tag)
    for k, v in config_received.__dict__.items():
        setattr(client, k, v)


def communication_parallel(client_list, action):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    tasks = []
    for m, client in enumerate(client_list): 
        if action == "send_config":
            task = asyncio.ensure_future(send_config(client, m + 1, comm_tags[m + 1]))
        elif action == "get_config":
            task = asyncio.ensure_future(get_config(client, m + 1, comm_tags[m + 1]))
        else:
            raise ValueError('Not valid action')
        tasks.append(task)
    loop.run_until_complete(asyncio.wait(tasks))
    loop.close()


def partition_data(dataset_type, partition_pattern, non_iid_ratio, client_num=10):
    train_dataset, _ = dataset_utils.load_datasets(dataset_type=dataset_type, data_path=cfg['dataset_path'])
    partition_sizes = np.ones((cfg['classes_size'], client_num))
    # iid
    if partition_pattern == 0:
        partition_sizes *= (1.0 / client_num)
    # non-iid
    # each client contains all classes of data, but the proportion of certain classes of data is very large
    elif partition_pattern == 1:
        if 0 < non_iid_ratio < 10:
            partition_sizes *= ((1 - non_iid_ratio * 0.1) / (client_num - 1))
            for i in range(cfg['classes_size']):
                partition_sizes[i][i % client_num] = non_iid_ratio * 0.1
        else:
            raise ValueError('Non-IID ratio is too large')
    # non-iid
    # each client misses some classes of data, while the other classes of data are distributed uniformly
    elif partition_pattern == 2:
        if 0 < non_iid_ratio < 10:
            # calculate how many classes of data each worker is missing
            missing_class_num = int(round(cfg['classes_size'] * (non_iid_ratio * 0.1)))

            partition_sizes = np.ones((cfg['classes_size'], client_num))
            begin_idx = 0
            for worker_idx in range(client_num):
                for i in range(missing_class_num):
                    partition_sizes[(begin_idx + i) % cfg['classes_size']][worker_idx] = 0.
                begin_idx = (begin_idx + missing_class_num) % cfg['classes_size']

            for i in range(cfg['classes_size']):
                count = np.count_nonzero(partition_sizes[i])
                for j in range(client_num):
                    if partition_sizes[i][j] == 1.:
                        partition_sizes[i][j] = 1. / count
        else:
            raise ValueError('Non-IID ratio is too large')
    elif partition_pattern == 3:
        if 0 < non_iid_ratio < 10:
            most_data_proportion = cfg['classes_size'] / client_num * non_iid_ratio * 0.1
            minor_data_proportion = cfg['classes_size'] / client_num * (1 - non_iid_ratio * 0.1) / (
                        cfg['classes_size'] - 1)
            partition_sizes *= minor_data_proportion
            for i in range(client_num):
                partition_sizes[i % cfg['classes_size']][i] = most_data_proportion
        else:
            raise ValueError('Non-IID ratio is too large')
    else:
        raise ValueError('Not valid partition pattern')

    train_data_partition = dataset_utils.LabelwisePartitioner(
        train_dataset, partition_sizes=partition_sizes, seed=cfg['data_partition_seed']
    )

    return train_data_partition, partition_sizes


if __name__ == "__main__":
    main()
