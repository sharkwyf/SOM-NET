import os.path
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
from collections import Counter
from torch.multiprocessing import Process


def showpic(image, idx):
    fig = image.permute(0, 3, 1, 4, 2).reshape(image.shape[0] * image.shape[3], image.shape[1] * image.shape[4], image.shape[2]).cpu()
    plt.imshow(fig)
    save = False
    if save:
        if not os.path.exists("./figures/"):
            os.makedirs("./figures/")
        plt.savefig(f"./figures/som{idx}.png")
    else:
        plt.show()
    pass


class timer:
    def __init__(self, pid):
        self.pid = pid
    
    def __enter__(self):
        self.elapsed = - time.time()
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.pid[0] in ["2"]:
            print("{}: elapsed time: {:.7f}".format(self.pid, self.elapsed + time.time()))
            pass


def init(m, gain=0.01, activate=False):
    """Initialize the module weights"""
    if activate:
        gain = nn.init.calculate_gain('relu')
    nn.init.orthogonal_(m.weight.data, gain=gain)
    if m.bias is not None:
        nn.init.constant_(m.bias.data, 0)
    return m


class ConvNet(nn.Module):
    """Nerual Network 1"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(p=.5)
        
    def forward(self, x):   # B,3,32,32
        batchsize = x.shape[0]
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        # flattening
        x = x.view(batchsize, -1)
        # fully connected layers
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x
    

class MyNeuralNetwork(nn.Module):
    """Nerual Network 1"""
    def __init__(self, momentum):
        super().__init__()
        self.model = nn.Linear(10, 10)
        self.som1 = SOMLayer(16, 16, 3, 3, 3, momentum=momentum).cuda()
        self.som2 = SOMLayer(16, 16, 2, 3, 3, momentum=momentum).cuda()
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(2 * 30 * 30, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 200)
        self.fc4 = nn.Linear(200, 10)
        self.dropout = nn.Dropout(p=.3)
    
    def forward(self, x):
        batchsize = x.shape[0]
        x = self.som1(x, stride=1).detach() # B, 2, 30, 30
        x = x.view(batchsize, -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = self.fc4(x)
        x = F.log_softmax(x, dim=1)
        return x


class SOMLayer(nn.Module):
    def __init__(self, H, W, C, KH, KW, momentum=0.99):
        super().__init__()
        self.H = H
        self.W = W
        self.C = C
        self.KH = KH
        self.KW = KW
        self.som = torch.ones((H, W, C, KH, KW), requires_grad=False).cuda() * 0.5  # H, W, OutputChannel, C, K, K
        self.momentum = [momentum, 1 - (1 - momentum) * 0.2]
        self.print_image = False
        self.processes = []

    def update_cell(self, rank, m, n, momentum, fig):
        if momentum < 1:
            if rank == 0:
                self.som[m][n] = momentum * self.som[m][n] + (1 - momentum) * fig
            else:
                self.som[m][n] = momentum * self.som[m][n] + fig
        pass
    
    def _update_som(self, m, n, fig, coef=1.):
        momentum = [1 - (1 - x) * coef for x in self.momentum]
        val = (1 - momentum[1]) * fig
        args = [
            (0, m, n, momentum[0], fig),
            (1, (self.H + m - 1) % self.H, n, momentum[1], val),
            (2, (self.H + m + 1) % self.H, n, momentum[1], val),
            (3, m, (self.W + n - 1) % self.W, momentum[1], val),
            (4, m, (self.W + n + 1) % self.W, momentum[1], val),
        ]
        # Update Winner
        for rank in range(5):
            self.update_cell(*args[rank])
        #     p = Process(target=self.update_cell, args=args[rank])
        #     p.start()
        #     self.processes.append(p)
        #
        # for p in self.processes:
        #     p.join()
        # pass
    
    def update(self, x, idx, coef=0., stride=2, prev_som=None):
        # Update SOM Network
        batchsize = x.shape[0]
        h_k, w_k = self.KH, self.KW
        h, w = self.H, self.W
        err_list = []
        with timer("0.0"):
            with torch.no_grad():
                if coef > 0:
                    # Update SOM Network
                    for j in range(0, x.shape[-2] - h_k + 1, stride):
                        with timer("2.0"):
                            for k in range(0, x.shape[-1] - w_k + 1, stride):
                                errors = []
                                with timer("1.1"):
                                    section = x[:, :, j:j + h_k, k:k + w_k]
                                    # Calculate discriminant values of (m, n) som for all batch
                                    for m in range(h):
                                        for n in range(w):
                                            errors.append(
                                                torch.sqrt(
                                                    torch.mean(
                                                        torch.pow(section - self.som[m][n], 2),
                                                        dim=[1, 2, 3]
                                                    )
                                                )
                                            )
                                            pass
                                with timer("1.2"):
                                    errors = torch.stack(errors)
                                    winner = torch.min(errors, dim=0)
                                    err_list.append(torch.mean(winner.values))
                                with timer("1.3"):
                                    for i in range(batchsize):
                                        # Choose winner cell
                                        m, n = torch.div(winner.indices[i], w, rounding_mode='floor'), winner.indices[
                                            i] % w
                                        # Update cells
                                        self._update_som(m, n, section[i], coef=coef)
                                pass

        if prev_som is None:
            showpic(self.som, idx)
        else:
            pass
        return torch.mean(torch.stack(err_list))
    
    def forward(self, x, stride=1):
        batchsize = x.shape[0]
        h_k, w_k = self.KH, self.KW
        h, w = self.H, self.W

        with timer("0.1"):
            with torch.no_grad():
                # Return activated images
                winner_r = []
                winner_c = []
                for j in range(0, x.shape[-2] - h_k + 1, stride):
                    for k in range(0, x.shape[-1] - w_k + 1, stride):
                        errors = []
                        section = x[:, :, j:j + h_k, k:k + w_k]
                        # Calculate discriminant values of (m, n) som for all batch
                        for m in range(h):
                            for n in range(w):
                                errors.append(
                                    torch.sqrt(
                                        torch.mean(
                                            torch.pow(section - self.som[m][n], 2),
                                            dim=[1, 2, 3]
                                        )
                                    )
                                )
                                pass
                                pass
                        errors = torch.stack(errors)
                        winner = torch.min(errors, dim=0)
                        for i in range(batchsize):
                            # Choose winner cell
                            m, n = torch.div(winner.indices[i], w, rounding_mode='floor'), winner.indices[i] % w
                            winner_r.append(m)
                            winner_c.append(n)
                        pass
                winner_r = torch.stack(winner_r).reshape((x.shape[-2] - h_k + 1) // stride,
                                                         (x.shape[-1] - w_k + 1) // stride, -1) / self.som.shape[0]
                winner_c = torch.stack(winner_c).reshape((x.shape[-2] - h_k + 1) // stride,
                                                         (x.shape[-1] - w_k + 1) // stride, -1) / self.som.shape[1]
                out = torch.stack([winner_r, winner_c]).permute(3, 0, 1, 2)
                pass
        
            pass

        if self.print_image:
            # Print original image & combined image
            for i in range(batchsize):
                d = x[i].permute(1, 2, 0)
                pic = []
                for j in range(d.shape[0]):
                    for k in range(d.shape[1]):
                        pic.append(self.som[int(d[j][k][0].item() * self.H)][int(d[j][k][1].item() * self.W)])
                pic = torch.stack(pic).reshape(15, 15, 3, 3, 3).permute(0, 3, 1, 4, 2).reshape(45, 45, 3).cpu()
                plt.imshow(pic)
                plt.show()
                plt.imshow(x[i].permute(1, 2, 0).cpu())
                plt.show()
                pass
        
        # B * C * H * W
        return out
