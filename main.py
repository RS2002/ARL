from torch.profiler import schedule
from model import MyNet
import torch.nn as nn
import numpy as np
import argparse
import torch
import tqdm
from torch.utils.data import DataLoader
from dataset import load_data
from sklearn.model_selection import train_test_split

mean = 16849.075
std = 9846.694593972894

def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument("--data_path",type=str,default="../data/train.csv")
    parser.add_argument('--test_prop', type=float, default=0.1)
    parser.add_argument("--cpu", action="store_true",default=False)
    parser.add_argument("--cuda", type=str, default='0')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--num_worker', type=int, default=1)

    parser.add_argument("--norm", action="store_true",default=False)
    parser.add_argument("--arl", action="store_true",default=False)
    parser.add_argument("--layer_sizes", type=int, default=[128,64,32,16,2])
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--gamma', type=float, default=0)
    parser.add_argument('--mask_prop', type=float, default=0.15)

    args = parser.parse_args()
    return args

def my_loss_func(y_hat,std_hat,y):
    loss = torch.log(std_hat) + (y - y_hat) ** 2 / (2 * std_hat ** 2)
    return torch.mean(loss)


def iteration(model,data_loader,loss_func,optim,scheduler,device,mask_id,mask_prop,train=True):
    if train:
        model.train()
        torch.set_grad_enabled(True)
    else:
        model.eval()
        torch.set_grad_enabled(False)

    rmse_list = []

    pbar = tqdm.tqdm(data_loader)
    for x1,x2,y in pbar:
        x1 = x1.to(device)
        if train:
            r = torch.rand(x1.shape[0])
            x1[r<mask_prop,1] = mask_id
        x2 = x2.float().to(device)
        y = y.float().to(device)
        y_ori = y.clone()

        y_hat = model(x1,x2)

        y_hat = y_hat.reshape(-1,2)
        std_hat = y_hat[:,1]
        y_hat = y_hat[:,0]

        # y_hat = y_hat.reshape(-1)
        y = y.reshape(-1)
        y = (y-mean)/std

        # loss = loss_func(y_hat,y)
        loss = my_loss_func(y_hat,std_hat,y)


        if train:
            optim.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

        y_hat = y_hat * std +mean
        rmse = torch.sqrt(loss_func(y_hat,y_ori))
        rmse_list.append(rmse.item())

    # if train:
    #     scheduler.step()

    return np.mean(rmse_list)


def main():
    args = get_args()
    device_name = "cuda:"+args.cuda
    device = torch.device(device_name if torch.cuda.is_available() and not args.cpu else 'cpu')

    dataset, encoder_list = load_data(args.data_path)
    mask_id = encoder_list[1].transform(["mask"])
    mask_id = mask_id[0]

    model = MyNet(layer_sizes=args.layer_sizes, arl=args.arl, norm=args.norm, dropout=args.dropout).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('total parameters:', total_params)

    train_data, test_data = train_test_split(dataset, test_size=args.test_prop)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_worker)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_worker)
    loss_func = nn.MSELoss()

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)#, weight_decay=args.weight_decay)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=args.gamma)
    scheduler = None

    best_rmse = 1e8
    converge_epoch = 0

    j = 0
    while True:
        j += 1
        rmse = iteration(model,train_loader,loss_func,optim,scheduler,device,mask_id,mask_prop=args.mask_prop,train=True)
        print("Epoch {:}, Training RMSE {:}".format(j, rmse))
        with open("log.txt",'a') as f:
            f.write("Epoch {:} , Training RMSE {:} , ".format(j, rmse))
        rmse = iteration(model,test_loader,loss_func,optim,scheduler,device,mask_id,mask_prop=0,train=False)
        print("Testing RMSE {:}".format(rmse))
        with open("log.txt",'a') as f:
            f.write("Testing RMSE {:} \n".format(rmse))
        if rmse <= best_rmse:
            torch.save(model.state_dict(), "model.pth")
            converge_epoch = 0
            best_rmse = rmse
        else:
            converge_epoch += 1

        if converge_epoch >= args.epoch:
            break

if __name__ == '__main__':
    main()
