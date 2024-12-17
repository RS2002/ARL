from model import MyNet
import pandas as pd
import argparse
import torch
import tqdm
from torch.utils.data import DataLoader
from dataset import load_data

mean = 16849.075
std = 9846.694593972894

def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument("--train_data_path",type=str,default="../data/train.csv")
    parser.add_argument("--data_path",type=str,default="../data/test.csv")
    parser.add_argument('--model_path_list', type=str, nargs='+', default=["./model1.pth","./model2.pth","./model3.pth","./model4.pth","./model5.pth","./model6.pth","./model7.pth"])

    parser.add_argument("--cpu", action="store_true",default=False)
    parser.add_argument("--cuda", type=str, default='0')
    parser.add_argument('--num_worker', type=int, default=1)

    parser.add_argument("--norm", action="store_true",default=False)
    parser.add_argument("--arl", action="store_true",default=False)
    parser.add_argument("--layer_sizes", type=int, default=[128,64,32,16,2])
    parser.add_argument('--dropout', type=float, default=0.0)


    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    device_name = "cuda:"+args.cuda
    device = torch.device(device_name if torch.cuda.is_available() and not args.cpu else 'cpu')

    _, encoder_list = load_data(args.train_data_path)

    test_data, id = load_data(args.data_path,encoder_list,train=False)

    data_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_worker)

    result_list = []
    model = MyNet(layer_sizes=args.layer_sizes, arl=args.arl, norm=args.norm, dropout=args.dropout).to(device)
    torch.set_grad_enabled(False)

    for model_path in args.model_path_list:
        model.load_state_dict(torch.load(model_path))
        model.eval()

        result = None

        pbar = tqdm.tqdm(data_loader)
        for x1, x2 in pbar:
            x1 = x1.to(device)
            x2 = x2.float().to(device)

            y_hat = model(x1, x2)
            y_hat = y_hat[:,0]
            y_hat = y_hat.reshape(-1)
            y_hat = y_hat * std + mean

            if result is None:
                result = y_hat
            else:
                result = torch.concat([result,y_hat],dim=0)
        result_list.append(result.unsqueeze(0))

    result = torch.mean(torch.concat(result_list,dim=0),dim=0)
    result_series = pd.Series(result.cpu().numpy())
    submission_df = pd.DataFrame({
        'id': id,
        'answer': result_series
    })
    submission_df.to_csv('submission.csv', index=False)



