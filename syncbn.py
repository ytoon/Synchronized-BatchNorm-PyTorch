import math
from queue import Queue
from IPython import embed

import torch
import torch.cuda.comm as comm
from torch.nn.modules.batchnorm import _BatchNorm
import torch.nn.functional as F

import syncbn_gpu


class SyncBNFucntion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gamma, beta, running_mean, running_var,
            extra, training=True, momentum=0.1, eps=1e-5, sync=True):
        def parse_extra(ctx, extra):
            ctx.is_master = extra["is_master"]
            if ctx.is_master:
                ctx.master_queue = extra["master_queue"]
                ctx.worker_queues = extra["worker_queues"]
                ctx.worker_ids = extra["worker_ids"]
            else:
                ctx.master_queue = extra["master_queue"]
                ctx.worker_queue = extra["worker_queue"]
        parse_extra(ctx, extra)
        ctx.training = training
        ctx.momentum = momentum
        ctx.eps = eps
        ctx.sync = sync

        if ctx.training:
            ex, exs = syncbn_gpu.batch_norm_collect_statistics(x)

            if ctx.sync:
                if ctx.is_master:
                    ex, exs = [ex.unsqueeze(0)], [exs.unsqueeze(0)]
                    for _ in range(ctx.master_queue.maxsize):
                        ex_w, exs_w = ctx.master_queue.get()
                        ctx.master_queue.task_done()
                        ex.append(ex_w.unsqueeze(0))
                        exs.append(exs_w.unsqueeze(0))
                    ex = comm.gather(ex).mean(0)
                    exs = comm.gather(exs).mean(0)

                    tensors = comm.broadcast_coalesced((ex, exs), [ex.get_device()] + ctx.worker_ids)
                    for ts, queue in zip(tensors[1:], ctx.worker_queues):
                        queue.put(ts)
                else:
                    ctx.master_queue.put((ex, exs))
                    ex, exs = ctx.worker_queue.get()
                    ctx.worker_queue.task_done()

            var = exs - ex ** 2
            running_mean.mul_(1 - ctx.momentum).add_(ctx.momentum * ex)
            running_var.mul_(1 - ctx.momentum).add_(ctx.momentum * var)

            ctx.mark_dirty(running_mean, running_var)

            y = syncbn_gpu.batch_norm_transform_input(x, gamma, beta, ex, exs, ctx.eps)

            ctx.save_for_backward(x, ex, exs, gamma, beta)

        return y

    @staticmethod
    def backward(ctx, grad_ouput):
        x, ex, exs, gamma, beta = ctx.saved_tensors

        grad_gamma, grad_beta, grad_ex, grad_exs = \
                syncbn_gpu.batch_norm_collect_grad_statistics(x, grad_ouput, gamma, ex, exs, ctx.eps)

        if ctx.training:
            if ctx.sync:
                if ctx.is_master:
                    grad_ex, grad_exs = [grad_ex.unsqueeze(0)], [grad_exs.unsqueeze(0)]
                    for _ in range(ctx.master_queue.maxsize):
                        grad_ex_w, grad_exs_w = ctx.master_queue.get()
                        ctx.master_queue.task_done()
                        grad_ex.append(grad_ex_w.unsqueeze(0))
                        grad_exs.append(grad_exs_w.unsqueeze(0))
                    grad_ex = comm.gather(grad_ex).mean(0)
                    grad_exs = comm.gather(grad_exs).mean(0)

                    tensors = comm.broadcast_coalesced((grad_ex, grad_exs), [grad_ex.get_device()] + ctx.worker_ids)
                    for ts, queue in zip(tensors[1:], ctx.worker_queues):
                        queue.put(ts)
                else:
                    ctx.master_queue.put((grad_ex, grad_exs))
                    grad_ex, grad_exs = ctx.worker_queue.get()
                    ctx.worker_queue.task_done()

        grad_input = syncbn_gpu.batch_norm_input_backward(x, grad_ouput, gamma, ex, exs, grad_ex, grad_exs, ctx.eps)

        return grad_input, grad_gamma, grad_beta, None, None, None, None, None, None


class SyncBN(_BatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, sync=True):
        super(SyncBN, self).__init__(num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True)

        self.devices        = list(range(torch.cuda.device_count()))
        self.sync           = sync if len(self.devices) > 1 else False
        self.worker_ids     = self.devices[1:]
        self.master_queue   = Queue(len(self.worker_ids))
        self.worker_queues  = [Queue(1) for _ in self.worker_ids]

    def forward(self, x):
        if self.training and self.sync:
            if x.get_device() == self.devices[0]:
                extra = {
                    'is_master': True,
                    'master_queue': self.master_queue,
                    'worker_queues': self.worker_queues,
                    'worker_ids': self.worker_ids
                }
            else:
                extra = {
                    'is_master': False,
                    'master_queue': self.master_queue,
                    'worker_queue': self.worker_queues[self.worker_ids.index(x.get_device())]
                }

            return SyncBNFucntion.apply(x, self.weight, self.bias, self.running_mean, self.running_var,
                extra, self.training, self.momentum, self.eps)
        else:
            exponential_average_factor = 0.0

            if self.training and self.track_running_stats:
                # TODO: if statement only here to tell the jit to skip emitting this when it is None
                if self.num_batches_tracked is not None:
                    self.num_batches_tracked += 1
                    if self.momentum is None:  # use cumulative moving average
                        exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                    else:  # use exponential moving average
                        exponential_average_factor = self.momentum

            return F.batch_norm(
                x, self.running_mean, self.running_var, self.weight, self.bias,
                self.training or not self.track_running_stats,
                exponential_average_factor, self.eps)


if __name__ == '__main__':
    import numpy as np
    device = torch.device('cuda')
    torch.manual_seed(123)
    x1 = torch.rand(32, 3, 200, 200, device=device, requires_grad=True)

    model = SyncBN(3)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    model = torch.nn.DataParallel(model)
    model.to(device)

    y1 = model(x1)

    z = y1.sum()
    model.zero_grad()
    z.backward()
    optimizer.step()

    torch.manual_seed(123)
    x2 = torch.rand(32, 3, 200, 200, device=device, requires_grad=True)

    model = torch.nn.BatchNorm2d(3)
    model.to(device)

    y2 = model(x2)

    z = y2.sum()
    model.zero_grad()
    z.backward()

    grad_x1 = x1.grad.data.cpu()
    grad_x2 = x2.grad.data.cpu()

    print((grad_x1 - grad_x2).abs().max())

    y1 = y1.data.cpu()
    y2 = y2.data.cpu()

    print((y1 - y2).abs().max())
