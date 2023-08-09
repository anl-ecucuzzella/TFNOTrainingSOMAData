import os
import shutil
import pickle
import torch
import numpy as np
import argparse

from neuralop.models import TFNO3d
from neuralop.models import FNO3d
from deep_adjoint.model.ForwardSurrogate import OneStepSolve3D
from deep_adjoint.utils.data import SOMAdata
#from deep_adjoint.train.trainer import Trainer, ddp_setup, mp, destory_process_group, predict
from deep_adjoint.train.trainer import Trainer, ddp_setup, mp, destroy_process_group, predict

def run(rank, world_size, args):
    ddp_setup(rank, world_size)

    net = TFNO3d(n_modes_height = 4, n_modes_width = 4, n_modes_depth = 4, in_channels = 17, out_channels = 16, hidden_channels = 16, projection_channels = 32, factorization = 'tucker', rank = 0.42)
    #net.load_state_dict(torch.load("model_saved_ep_110"))
    trainer = Trainer(net=net, 
                      optimizer_name='Adam', 
                      loss_name='L2',
                      gpu_id=rank)
    data_path = '/global/cfs/projectdirs/m4259/yixuansun/thedataset3.hdf5'
    train_set = SOMAdata(path=data_path, mode='train', gpu_id=rank) 
    val_set = SOMAdata(path=data_path, mode='val', gpu_id=rank) 
    test_set = SOMAdata(path=data_path, mode='test', gpu_id=rank) 
    
    if args.train == "True":
        trainer.train(train=train_set,
                  val=val_set,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  learning_rate=args.lr,
                  model_name=args.model_name,
                  mask=args.mask)
        destroy_process_group()
    else:
        true, pred, gm = predict(net=net, test_data=test_set, gpu_id=rank,
                     checkpoint='model_saved_ep_110_fno')
        with open('/pscratch/sd/e/ecucuz/2023-08-09-true_pred_FNO-masked.pkl', 'wb') as f:
            true_pred = {'true': true, 'pred': pred, 'gm': gm}
            pickle.dump(true_pred, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-hidden', default=2, type=int)
    parser.add_argument('-num_res_block', default=2, type=int)
    parser.add_argument('-epochs', default=150, type=int)
    parser.add_argument('-batch_size', default=25, type=int)
    parser.add_argument('-lr', default=0.001, type=float)
    parser.add_argument('-train', default='True', type=str)
    parser.add_argument('-mask', default=None, type=str)
    parser.add_argument('-model_name', type=str, default='test')
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    if args.train == 'True':
        mp.spawn(run, args=(world_size, args), nprocs=world_size)
    else:
        run(0, 1, args)
