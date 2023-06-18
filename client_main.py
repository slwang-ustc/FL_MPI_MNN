from __future__ import print_function
import asyncio
import os
from config import cfg
from client import ClientConfig
from comm_utils import *
from training_utils import train, test
from mpi4py import MPI
import logging
import datasets.utils as dataset_utils
import models.utils as model_utils
import time
import MNN

F = MNN.expr
nn = MNN.nn

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
csize = comm.Get_size()

if cfg['client_cuda'] == '-1':
    os.environ['CUDA_VISIBLE_DEVICES'] = str(int(rank) % 4 + 0)
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg['client_cuda']
# device = torch.device("cuda" if cfg['client_use_cuda'] and torch.cuda.is_available() else "cpu")


now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
RESULT_PATH = os.getcwd() + '/clients_log/' + now + '/'

if not os.path.exists(RESULT_PATH):
    os.makedirs(RESULT_PATH, exist_ok=True)

MASTER_RANK = 0


def main():
    print(rank)
    client_config = ClientConfig(idx=0)

    # dataset and the test dataloader
    train_dataset, test_dataset = dataset_utils.load_datasets(cfg['dataset_type'], cfg['dataset_path'])
    test_loader = dataset_utils.create_dataloaders(test_dataset, batch_size=cfg['client_test_batch_size'], shuffle=False)

    # begin each epoch
    comm_tag = 1
    while True:
        # receive the configuration from the server
        communicate_with_server(client_config, comm_tag, action='get_config')

        logger = init_logger(comm_tag, client_config)
        logger.info("_____****_____\nEpoch: {:04d}".format(client_config.epoch_idx))

        # torch.random.seed()
        # load the test and train loader
        train_loader = dataset_utils.create_dataloaders(
            train_dataset, batch_size=cfg['local_batch_size'], selected_idxs=client_config.train_data_idxes
        )

        # start local training
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        tasks = [asyncio.ensure_future(local_training(client_config, train_loader, test_loader, logger))]
        loop.run_until_complete(asyncio.wait(tasks))
        loop.close()

        # send the configuration to the server
        communicate_with_server(client_config, comm_tag, action='send_config')
        comm_tag += 1

        if client_config.epoch_idx >= cfg['epoch_num']:
            break


async def get_config(config, comm_tag):
    config_received = await get_data(comm, MASTER_RANK, comm_tag)
    for k, v in config_received.__dict__.items():
        setattr(config, k, v)


async def send_config(config, comm_tag):
    await send_data(comm, config, MASTER_RANK, comm_tag)


def communicate_with_server(config, comm_tag, action):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    tasks = []
    if action == "send_config":
        task = asyncio.ensure_future(
            send_config(config, comm_tag)
        )
    elif action == "get_config":
        task = asyncio.ensure_future(
            get_config(config, comm_tag)
        )
    else:
        raise ValueError('Not valid action')
    tasks.append(task)
    loop.run_until_complete(asyncio.wait(tasks))
    loop.close()


def init_logger(comm_tag, client_config):
    logger = logging.getLogger(os.path.basename(__file__).split('.')[0] + str(comm_tag))
    logger.setLevel(logging.INFO)
    filename = RESULT_PATH + now + "_" + os.path.basename(__file__).split('.')[0] + '_' + str(
        client_config.idx) + '.log'
    file_handler = logging.FileHandler(filename=filename)
    formatter = logging.Formatter("%(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


async def local_training(config, train_loader, test_loader, logger):
    local_model = model_utils.create_model_instance(cfg['model_type'])
    local_model.load_parameters(F.load_as_list(config.global_model_path))
    for i in local_model.parameters:
        i.fix_as_trainable()
    epoch_lr = config.lr
    local_steps = cfg['local_iters']
    if config.epoch_idx > 1:
        epoch_lr = max(cfg['decay_rate'] * epoch_lr, cfg['min_lr'])
        config.lr = epoch_lr
    logger.info("lr: {}\n".format(epoch_lr))
    if cfg['momentum'] < 0:
        optimizer = MNN.optim.SGD(local_model, epoch_lr, weight_decay=cfg['weight_decay'])
    else:
        optimizer = MNN.optim.SGD(local_model, epoch_lr, momentum=cfg['momentum'], weight_decay=cfg['weight_decay'])

    train_loss, train_time = train(
        local_model,
        train_loader,
        optimizer,
        local_iters=local_steps,
        # device=device
    )

    # config.params = local_model.parameters
    F.save(local_model.parameters, config.local_model_path)

    logger.info(
        "Train_Loss: {:.4f}\n".format(train_loss) +
        "Train_Time: {:.4f}\n".format(train_time)
    )

    test_loss, test_acc, test_auc = test(local_model, test_loader)

    logger.info(
        "Test_Loss: {:.4f}\n".format(test_loss) +
        "Test_ACC: {:.4f}\n".format(test_acc) +
        "Test_AUC: {:.4f}\n".format(test_auc)
    )
    
    config.train_time = train_time


if __name__ == '__main__':
    main()
