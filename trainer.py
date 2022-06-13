import os.path

import torch
import network
import torch.nn as nn
import torch.nn.functional as F
import time


class Trainer:
    """Trainer class for Regression Model"""
    def __init__(self, config):
        self.config = config
        self.device = self.config.device
        self.lr = config.lr
        # self.model = network.ConvNet().to(self.device)
        self.model = network.MyNeuralNetwork(momentum=self.config.momentum).to(self.device)
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                          lr=self.lr)
    
    def _update(self, inputs, labels, mode):
        """Update"""
        # Forward
        if mode == "train":
            self.model.train()
        else:
            self.model.eval()
        inputs, labels = inputs.cuda(), labels.cuda()
        out = self.model(inputs)
        loss = F.nll_loss(out, labels)
        
        # Backward
        num_corr = torch.sum(out.argmax(dim=1).eq(labels))
        self.optimizer.zero_grad()
        if mode == "train":
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        torch.cuda.empty_cache()
        # print("loss: {:.4f}, accuracy: {:.3f}%".format(loss.item(), num_corr / inputs.shape[0] * 100))
        return loss, \
               num_corr, \
               inputs.shape[0]

    def _run_epoch(self, data_loader, mode='eval'):
        """Run one epoch"""
        total_num, total_num_corr = 0, 0
        total_loss = []
        for idx, (inputs, labels) in enumerate(data_loader):
            # print(f"{idx}: ", end="")
            loss, num_corr, num = self._update(inputs, labels, mode)
            total_loss.append(loss.item())
            total_num_corr += num_corr
            total_num += num
        return torch.mean(torch.tensor(total_loss)), \
               total_num_corr, \
               total_num
    
    def _train_som(self, data_loader, coef=0.):
        """Train SOM for one epoch"""
        dir1 = "./data/som1.pt"
        if os.path.exists(dir1):
            ckpt = torch.load(dir1)
            self.model.som1 = ckpt
            print(f"SOM1 loaded from '{dir1}'")
        
        for epoch in range(0):
            start_time = time.time()
            for idx, (inputs, _) in enumerate(data_loader):
                inputs = inputs.to(self.device)
                err = self.model.som1.update(inputs, idx, coef, 2)
                print(f"{idx}: mean error: {err}")
                
                torch.save(self.model.som1, dir1)
                print(f"SOM1 save to '{dir1}'")
                break
            print(f"SOM1: Total elapsed time: {time.time() - start_time}")
            
        for epoch in range(0):
            start_time = time.time()
            for idx, (inputs, _) in enumerate(data_loader):
                inputs = inputs.to(self.device)
                inputs = self.model.som1(inputs)
                err = self.model.som2.update(inputs, idx, coef, 2, prev_som=self.model.som1)
                print(f"{idx}: mean error: {err}")
            print(f"Total elapsed time: {time.time() - start_time}")
        pass
    
    @staticmethod
    def concatenate_msg(loss, num_corr, num):
        return "loss: {:.4f}, accuracy: {:.3f}%".format(loss.item(), num_corr / num * 100)

    def train(self, train_loader, eval_loader=None):
        """Train on train data, Evaluate on eval data"""
        # Train SOM
        self._train_som(train_loader, 1.)
        
        # Train Network
        for epoch in range(1, self.config.epochs + 1):
            train_loss, train_num_corr, train_num = self._run_epoch(train_loader, mode="train" if epoch > 0 else "eval")
            if epoch % 1 == 0:
                print(f"{epoch}: Train, {self.concatenate_msg(train_loss, train_num_corr, train_num)}")
            if eval_loader is not None:
                eval_loss, eval_num_corr, eval_num = self._run_epoch(eval_loader, mode="eval")
                if epoch % 1 == 0:
                    print(f"{epoch}:  Eval, {self.concatenate_msg(eval_loss, eval_num_corr, eval_num)}")

    def test(self, test_loader):
        """Evaluate on test data"""
        test_loss,  test_num_corr, test_num = self._run_epoch(test_loader, mode="test")
        print(f"Test, {self.concatenate_msg(test_loss, test_num_corr, test_num)}")
        return test_loss, test_num_corr, test_num
