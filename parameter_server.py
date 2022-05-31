import argparse
import os
import threading
import time

import torch
from torch import optim
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
import torch.distributed.rpc as rpc
import torchvision
from torchvision import transforms
from model.resnet import ResNet18
from baseline import test
from util import save_log


class ParameterServer(object):
    """"
     The parameter server (PS) updates model parameters with gradients from the workers
     and sends the updated parameters back to the workers.
    """
    def __init__(self, model, num_workers, lr):
        self.lock = threading.Lock()
        self.future_model = torch.futures.Future()
        self.num_workers = num_workers
        # initialize model parameters
        self.model = ResNet18(num_classes=10)
        # zero gradients
        for p in self.model.parameters():
            p.grad = torch.zeros_like(p)
        self.optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        self.scheduler = MultiStepLR(optimizer=self.optimizer, milestones=[15, 30], gamma=0.1)

    def get_model(self):
        return self.model

    def lr_schedule(self):
        self.scheduler.step()

    @staticmethod
    @rpc.functions.async_execution
    def update_and_fetch_model(ps_rref, grads):
        self = ps_rref.local_value()
        with self.lock:
            # update model parameters
            for p, g in zip(self.model.parameters(), grads):
                p.grad = g
            self.optimizer.step()
            self.optimizer.zero_grad()

            fut = self.future_model

            fut.set_result(self.model)
            self.future_model = torch.futures.Future()

        return fut


def run_worker(ps_rref, rank, num_epochs):
    """
    A worker pulls model parameters from the PS, computes gradients on a mini-batch
    from its data partition, and pushes the gradients to the PS.
    """

    # prepare dataset
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_set = torchvision.datasets.CIFAR10('./data/', train=True, download=True, transform=transform_train)
    test_set = torchvision.datasets.CIFAR10("./data/", train=False, download=True, transform=transform_test)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False)
    criterion = nn.CrossEntropyLoss()


    # set device
    device_id = rank - 1
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")

    # get initial model from the PS
    model = ps_rref.rpc_sync().get_model().to(device)

    print(f'worker{rank} starts training')

    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    start_evt.record()# 当前时间

    model.train()
    loss_list = []
    for epoch in range(num_epochs):
        loss_total = 0.0
        for i, batch_data in enumerate(train_loader):
            inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            # pipeline
            loss.backward()
            # send gradients to the PS and fetch updated model parameters
            loss_total += loss.item()

            model = rpc.rpc_sync(to=ps_rref.owner(),
                             func=ParameterServer.update_and_fetch_model,
                             args=(ps_rref, [p.grad for p in model.cpu().parameters()], rank)
                             ).to(device)
            if i % 20 == 19:  # print every 2000 mini-batches
                print('Device: %d epoch: %d, iters: %5d, loss: %.3f' % (
                    rank, epoch + 1, i + 1, loss_total / 20))

                loss_total = 0.0

        if rank == 1:
            ps_rref.rpc_async().lr_schedule()

    end_evt.record()
    torch.cuda.synchronize()

    whole_time = start_evt.elapsed_time(end_evt)  # 结束时间
    print("Training time: {}".format(whole_time))

    acc = test(model, test_loader)
    save_log('./log/ps_rank_{}.log'.format(rank), loss_list, acc, whole_time)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=1, help="Global rank of this process.")
    parser.add_argument("--world_size", type=int, default=3, help="Total number of workers.")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate.")
    parser.add_argument("--num_epochs", type=int, default=45, help="Number of epochs.")

    args = parser.parse_args()
    return args

def main():

    args = parse_args()
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=16, rpc_timeout=0)

    if args.rank == 0:
        """
        initialize PS and run workers
        """
        print(f"PS{args.rank} initializing")
        rpc.init_rpc(f"PS{args.rank}", rank=args.rank, world_size=args.world_size, rpc_backend_options=options)
        print(f"PS{args.rank} initialized")

        ps_rref = rpc.RRef(ParameterServer(args.model, args.world_size, args.lr))

        futs = []
        for r in range(1, args.world_size):
            worker = f'worker{r}'
            futs.append(rpc.rpc_async(to=worker,
                                      func=run_worker,
                                      args=(ps_rref, r, args.num_epochs)))

        torch.futures.wait_all(futs)
        print(f"Finish training")

    else:
        """
        initialize workers
        """
        print(f"worker{args.rank} initializing")
        rpc.init_rpc(f"worker{args.rank}", rank=args.rank, world_size=args.world_size, rpc_backend_options=options)
        print(f"worker{args.rank} initialized")

    rpc.shutdown()


if __name__ == "__main__":
    main()
