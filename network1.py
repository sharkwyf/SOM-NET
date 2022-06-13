import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
from collections import Counter


def showpic(image):
    fig = image.permute(0, 3, 1, 4, 2).reshape(image.shape[0] * image.shape[3], image.shape[1] * image.shape[4], image.shape[2]).cpu()
    plt.imshow(fig)
    plt.show()
    pass


class timer:
    def __init__(self, pid):
        self.pid = pid
    
    def __enter__(self):
        self.elapsed = - time.time()
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.pid[0] in ["0"]:
            # print("{}: elapsed time: {:.7f}".format(self.pid, self.elapsed + time.time()))
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
        return x
    

class MyNeuralNetwork(nn.Module):
    """Nerual Network 1"""
    def __init__(self, momentum=1.):
        super().__init__()
        self.model = nn.Linear(10, 10)
        self.som = SOMLayer(16, 16, 3, 3, 3).cuda()
        # self.som = torch.zeros((16, 16, 3, 3, 3), requires_grad=False).cuda()  # H, W, OutputChannel, C, K, K
        # self.som = self.som / 10 + 0.5
        self.momentum = [momentum, 1 - (1 - momentum) * 0.2]
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(2 * 15 * 15, 400)
        self.fc2 = nn.Linear(400, 200)
        self.fc3 = nn.Linear(200, 100)
        self.fc4 = nn.Linear(100, 10)
        self.dropout = nn.Dropout(p=.5)
        self.print_image = False
    
    def _update_som(self, m, n, fig, coef=1.):
        def _update_neighbour():
            val = (1 - momentum[1]) * fig
            if m > 0:
                self.som[m - 1][n] = momentum[1] * self.som[m - 1][n] + val
            else:
                self.som[-1][n] = momentum[1] * self.som[- 1][n] + val
            if n > 0:
                self.som[m][n - 1] = momentum[1] * self.som[m][n - 1] + val
            else:
                self.som[m][- 1] = momentum[1] * self.som[m][- 1] + val
            if m < self.som.shape[0] - 1:
                self.som[m + 1][n] = momentum[1] * self.som[m + 1][n] + val
            else:
                self.som[0][n] = momentum[1] * self.som[0][n] + val
            if n < self.som.shape[0] - 1:
                self.som[m][n + 1] = momentum[1] * self.som[m][n + 1] + val
            else:
                self.som[m][0] = momentum[1] * self.som[m][0] + val
            pass

        momentum = [1 - (1 - x) * coef for x in self.momentum]
        self.som[m][n] = momentum[0] * self.som[m][n] + (1 - momentum[0]) * fig
        _update_neighbour()
        pass
    
    def _forward_som(self, x, coef, stride=1):
        batchsize = x.shape[0]
        h_k, w_k = self.som.shape[-2:]
        h, w = self.som.shape[0:2]
        self.count = {}
        with torch.no_grad():
            if coef > 0:
                # Update SOM Network
                for j in range(0, x.shape[-2] - h_k + 1, stride):
                    with timer("0"):
                        for k in range(0, x.shape[-1] - w_k + 1, stride):
                            errors = []
                            with timer("1.1"):
                                # Calculate discriminant values of (m, n) som for all batch
                                for m in range(h):
                                    for n in range(w):
                                        section = x[:, :, j:j + h_k, k:k + w_k]
                                        f = self.som[m][n]
                                        errors.append(torch.mean(torch.pow(section - f, 2), dim=[1, 2, 3]))
                                        pass
                            with timer("1.2"):
                                errors = torch.stack(errors)
                            with timer("1.3"):
                                winner = torch.min(errors, dim=0)
                            with timer("1.4"):
                                for i in range(batchsize):
                                    # Choose winner cell
                                    m, n = torch.div(winner.indices[i], w, rounding_mode='floor'), winner.indices[i] % w
                                    # if (m.item(), n.item()) in self.count:
                                    #     self.count[(m.item(), n.item())] += 1
                                    # else:
                                    #     self.count[(m.item(), n.item())] = 1
                                    # Update cells
                                    with timer("1.4.2"):
                                        self._update_som(m, n, x[i, :, j:j + h_k, k:k + w_k], coef=coef)
                            pass
            
            # Return activated images
            winner_r = []
            winner_c = []
            for j in range(0, x.shape[-2] - h_k + 1, stride):
                for k in range(0, x.shape[-1] - w_k + 1, stride):
                    errors = []
                    # Calculate discriminant values of (m, n) som for all batch
                    for m in range(h):
                        for n in range(w):
                            section = x[:, :, j:j + h_k, k:k + w_k]
                            f = self.som[m][n]
                            errors.append(torch.mean(torch.pow(section - f, 2), dim=[1, 2, 3]))
                            pass
                    errors = torch.stack(errors)
                    winner = torch.min(errors, dim=0)
                    for i in range(batchsize):
                        # Choose winner cell
                        m, n = torch.div(winner.indices[i], w, rounding_mode='floor'), winner.indices[i] % w
                        winner_r.append(m)
                        winner_c.append(n)
                    pass
            winner_r = torch.stack(winner_r).reshape((x.shape[-2] - h_k + 1) // stride, (x.shape[-1] - w_k + 1) // stride, -1) / self.som.shape[0]
            winner_c = torch.stack(winner_c).reshape((x.shape[-2] - h_k + 1) // stride, (x.shape[-1] - w_k + 1) // stride, -1) / self.som.shape[1]
            out = torch.stack([winner_r, winner_c]).permute(3, 0, 1, 2)
            pass
            
        pass
        # B * C * H * W
        return out
    
    def forward(self, x, som_coef=1.):
        batchsize = x.shape[0]
        inputs = x
        
        # SOM Update
        x = self._forward_som(x, som_coef, stride=2).detach()
        
        if self.print_image:
            for i in range(batchsize):
                d = x[i].permute(1, 2, 0)
                pic = []
                for j in range(d.shape[0]):
                    for k in range(d.shape[1]):
                        pic.append(self.som[int(d[j][k][0].item() * 16)][int(d[j][k][1].item() * 16)])
                pic = torch.stack(pic).reshape(15, 15, 3, 3, 3).permute(0, 3, 1, 4, 2).reshape(45, 45, 3).cpu()
                plt.imshow(pic)
                plt.show()
                plt.imshow(inputs[i].permute(1, 2, 0).cpu())
                plt.show()
                pass
        
        x = x.view(batchsize, -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = self.fc4(x)
        
        showpic(self.som)
        # c = Counter(self.count)
        return x


class SOMLayer(nn.Module):
    def __init__(self, H, W, C, KH, KW):
        super().__init__()
        self.H = H
        self.W = W
        self.C = C
        self.KH = KH
        self.KW = KW
        self.som = self.som = torch.zeros((H, W, C, KH, KW), requires_grad=False).cuda()  # H, W, OutputChannel, C, K, K
        self.som = self.som / 10 + 0.5

    def forward(self, x, coef=0., stride=1):
        batchsize = x.shape[0]
        h_k, w_k = self.KH, self.KW
        h, w = self.H, self.W
        self.count = {}
        with torch.no_grad():
            if coef > 0:
                # Update SOM Network
                for j in range(0, x.shape[-2] - h_k + 1, stride):
                    with timer("0"):
                        for k in range(0, x.shape[-1] - w_k + 1, stride):
                            errors = []
                            with timer("1.1"):
                                # Calculate discriminant values of (m, n) som for all batch
                                for m in range(h):
                                    for n in range(w):
                                        section = x[:, :, j:j + h_k, k:k + w_k]
                                        f = self.som[m][n]
                                        errors.append(torch.mean(torch.pow(section - f, 2), dim=[1, 2, 3]))
                                        pass
                            with timer("1.2"):
                                errors = torch.stack(errors)
                            with timer("1.3"):
                                winner = torch.min(errors, dim=0)
                            with timer("1.4"):
                                for i in range(batchsize):
                                    # Choose winner cell
                                    m, n = torch.div(winner.indices[i], w, rounding_mode='floor'), winner.indices[i] % w
                                    # if (m.item(), n.item()) in self.count:
                                    #     self.count[(m.item(), n.item())] += 1
                                    # else:
                                    #     self.count[(m.item(), n.item())] = 1
                                    # Update cells
                                    with timer("1.4.2"):
                                        self._update_som(m, n, x[i, :, j:j + h_k, k:k + w_k], coef=coef)
                            pass
        
            # Return activated images
            winner_r = []
            winner_c = []
            for j in range(0, x.shape[-2] - h_k + 1, stride):
                for k in range(0, x.shape[-1] - w_k + 1, stride):
                    errors = []
                    # Calculate discriminant values of (m, n) som for all batch
                    for m in range(h):
                        for n in range(w):
                            section = x[:, :, j:j + h_k, k:k + w_k]
                            f = self.som[m][n]
                            errors.append(torch.mean(torch.pow(section - f, 2), dim=[1, 2, 3]))
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
        # B * C * H * W
        return out
