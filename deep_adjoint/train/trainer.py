import torch 
import shutil
import os
import numpy as np
import torch.multiprocessing as mp

from datetime import datetime
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group


from ..utils import FNO_losses
from ..utils.logger import Logger


def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12532"
    init_process_group(backend='nccl', rank=rank, world_size=world_size)

class Trainer:
    '''
    Basic trainer class
    '''
    def __init__(self, net, optimizer_name, loss_name, gpu_id, dual_train=False) -> None:
        self.net = net
        if optimizer_name == "Adam":
            self.optimizer = torch.optim.Adam
        
        self.gpu_id = gpu_id
        self.ls_fn = FNO_losses.LpLoss(d = 4, p = 2)
       
        # if torch.cuda.is_available():
        #     self.device = torch.device("cuda")  # Use CUDA device
        # else:
        #     self.device = torch.device("cpu")
        self.net = net.to(gpu_id)

        self.net = DDP(net, device_ids=[self.gpu_id], find_unused_parameters=True) 
        self.dual_train = dual_train

        self.now = datetime.now().strftime('%Y-%m-%d')
     

    def train(self, train,
              val,
              epochs,
              batch_size,
              learning_rate,
              save_freq=10,
              model_name='test',
              mask=None):

        '''
        args:
            train: training dataset
            val: validation dataset
        '''

        if not os.path.exists(f'./checkpoints/{self.now}_{model_name}/'):
            print("Creating model saving folder new...")
            try:
                os.mkdir(f'./checkpoints/{self.now}_{model_name}/')
            except OSError as error:
                print(error)
        #else:
        #    shutil.rmtree(f'./checkpoints/{self.now}_{model_name}/')
        #    print("Creating model saving folder after deleting old...")
        #    os.makedirs(f'./checkpoints/{self.now}_{model_name}/')
        self.logger = Logger('./checkpoints/' + str(self.now) + '_' + str(model_name) + '/')


        # self.net.to(self.device)
        self.net.train()
        optimizer = self.optimizer(self.net.parameters(), lr=learning_rate)
        train_loader = DataLoader(train, batch_size=batch_size, sampler=DistributedSampler(train))
        val_loader = DataLoader(val, batch_size=10, sampler=DistributedSampler(val))

        if mask is not None:
            print("Masking the loss...")
            mask = train.loss_mask.to(self.gpu_id)

        for val in val_loader:
            if self.dual_train:
                x_val, y_val = val
                y_val, adj_val = y_val
            else:
                x_val, y_val = val
            
            x_val = x_val.to(self.gpu_id)
            y_val = y_val.to(self.gpu_id)
            break

        print("Starts training...")
        for ep in range(epochs):
            running_loss = []
            for x_train, y_train in tqdm(train_loader):
                x_train = x_train.to(self.gpu_id)
                y_train = y_train.to(self.gpu_id)
                if self.dual_train:
                    y_train, adj_train = y_train
                optimizer.zero_grad()
                out = self.net(x_train)
                
                if self.dual_train:
                    batch_loss = self.ls_fn(y_train, out, adj_train)
                else:
                    batch_loss = self.ls_fn(y_train, out)
                batch_loss.backward()
                running_loss.append(batch_loss.detach().cpu().numpy())
                optimizer.step()
            with torch.no_grad():
                val_out = self.net(x_val) 
                if self.dual_train:
                    val_loss = self.ls_fn(y_val, val_out, adj_val)
                else:
                    val_loss = self.ls_fn(y_val, val_out)
            
            if self.gpu_id == 0 and ep % save_freq == 0:
                torch.save(self.net.module.state_dict(), f'./checkpoints/{self.now}_{model_name}/model_saved_ep_{ep}')

            self.logger.record('epoch', ep+1)
            self.logger.record('train_loss', np.mean(running_loss))
            self.logger.record('val_loss', val_loss.item())
            self.logger.print()
            self.logger.save()
            



def predict(net, gpu_id, test_data, checkpoint=None):
    test_loader = DataLoader(test_data, batch_size=29)

    y_true = []
    y_pred = []
    gm = []
    net.eval()
    net.to(gpu_id)
    if checkpoint is not None:
        net.load_state_dict(torch.load(checkpoint))
    for x, y in tqdm(test_loader):
        x = x.to(gpu_id)
        y = y.to(gpu_id)
        with torch.no_grad():
            pred = net(x)
            y_true.append(y.detach().cpu().numpy())
            y_pred.append(pred.detach().cpu().numpy())
            gm.append(x.detach().cpu().numpy())
                        
    # y_true = np.concatenate(y_true)
    # y_pred = np.concatenate(y_pred)
    # gm = np.asarray(gm)
    return y_true, y_pred, gm




class AdjointTrainer(Trainer):
    def __init__(self, net, optimizer_name, loss_name, dual_train=False) -> None:
        super().__init__(net, optimizer_name, loss_name, dual_train)
        
    def train(self, train, val, epochs, batch_size, learning_rate, save_freq=10, portion='u'):
        self.net.to(self.device)
        self.net.train()
        optimizer = self.optimizer(self.net.parameters(), lr=learning_rate)
        train_loader = DataLoader(train, batch_size=batch_size)
        val_loader = DataLoader(val, batch_size=100)

        for val in val_loader:
            x_val, y_val = val
            y_val_out, y_val_adj = y_val

        print("Starts training...")
        for ep in range(epochs):
            running_loss = []
            for x_train, y_train in tqdm(train_loader):
                y_out, y_adj = y_train
                optimizer.zero_grad()
                out = self.net(x_train)
                pred_adj = self.get_grad(x_train, portion=portion)
                batch_loss = self.ls_fn(y_out, out) + self.ls_fn(y_adj, pred_adj)
                batch_loss.backward()
                running_loss.append(batch_loss.detach().cpu().numpy())
                optimizer.step()
          
            pred_val_out = self.net(x_val) 
            val_pred_adj = self.get_grad(x_val, portion=portion)
            adj_ls_val = self.ls_fn(y_val_adj, val_pred_adj)
            val_loss = self.ls_fn(y_val_out, pred_val_out) + adj_ls_val

            self.logger.record('epoch', ep+1)
            self.logger.record('train_loss', np.mean(running_loss))
            self.logger.record('val_loss', val_loss.item())
            self.logger.record('val_adj_loss', adj_ls_val.item())
            self.logger.print()
            torch.save(self.net.state_dict(), f'./checkpoints/{self.now}/model_saved_ep_{ep}')

        self.logger.finish()


    def get_grad(self, x, portion='u'): # x shape [batch, in_dim]; output shape [batch, out_dim]
        x = torch.tensor(x, requires_grad=True)
        def compute_grad(x, net):
            x = x.unsqueeze(0) 
            sum_square = 0.5 * torch.sum(net(x))
            grad = torch.autograd.grad(sum_square, x, retain_graph=True)[0]
            return grad
        grad = [compute_grad(x[i], self.net) for i in range(len(x))]
        # grad = zip(*grad)
        grad = torch.concat(grad)
        if portion == 'u':
            return grad[:, :80]
        elif portion == 'all':
            return grad
        elif portion == 'p':
            return grad[:, -79:]
        else:
            raise Exception(f'"{portion}" is not in the list...')
        
    
        


class MultiStepTrainer(Trainer):
    def __init__(self, net, epochs) -> None:
        super(MultiStepTrainer, self).__init__(net, epochs)
        pass
