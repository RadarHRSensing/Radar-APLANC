import sys
import os
from pathlib import Path


current_file = Path(__file__).resolve()


project_root = current_file.parent.parent


sys.path.append(str(project_root))


print(f"项目根目录: {project_root}")
print(f"Python路径: {sys.path}")



import numpy as np
import pickle
import os
import argparse
import matplotlib.pyplot as plt
import torch.nn as nn
from tqdm import tqdm
import time
import torch
from torch.utils.data import DataLoader
from utils_sig import *
from IrrelevantPowerRatio import IrrelevantPowerRatio
from rf.model import RF_conv_decoder
from rf.proc import rotateIQ
from data.datasets import RHB
from utils.eval import eval_RHB
from loss import ContrastLoss


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# Argparser.
def parseArgs():
    parser = argparse.ArgumentParser(description='Configs for thr RF train script')

    parser.add_argument('-dir', '--data-dir', default="APLANC/dataset/RHB", type=str,
                        help="Parent directory containing the folders with the Pickle files and the Vital signs.")

    parser.add_argument('-fp', '--fitzpatrick-path', type=str,
                        default="../dataset/fitzpatrick_labels.pkl",
                        help='Pickle file containing the fitzpatrick labels.')

    parser.add_argument('--folds-path', type=str,
                        default="APLANC/dataset/RHB_demo_fold1.pkl",
                        help='Pickle file containing the folds.')

    parser.add_argument('--fold', type=int, default=0,
                        help='Fold Number')

    parser.add_argument('--device', type=str, default=None,
                        help="Device on which the model needs to run (input to torch.device). \
                              Don't specify for automatic selection. Will be modified inplace.")

    parser.add_argument('-ckpts', '--checkpoints-path', type=str,
                        default="APLANC/rf/ckpt/RHB",
                        help='Checkpoint Folder for model.')

    parser.add_argument('--verbose', action='store_true', help="Verbosity.")

    parser.add_argument('--viz', action='store_true', help="Visualize.")

    # Train args
    parser.add_argument('--batch-size', type=int, default=2,
                        help="Batch Size for the dataloaders.")  # default = 32

    parser.add_argument('--num-workers', type=int, default=4,
                        help="Number of Workers for the dataloaders.")  # default = 2

    parser.add_argument('--train-shuffle', action='store_true', help="Shuffle the train loader.")
    parser.add_argument('--val-shuffle', action='store_true', help="Shuffle the val loader.")
    parser.add_argument('--test-shuffle', action='store_true', help="Shuffle the test loader.")

    parser.add_argument('--train-drop', action='store_true', help="Drop the final sample of the train loader.")
    parser.add_argument('--val-drop', action='store_true', help="Drop the final sample of the val loader.")
    parser.add_argument('--test-drop', action='store_true', help="Drop the final sample of the test loader.")

    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-4,
                        help="Learning Rate for the optimizer.")

    parser.add_argument('-wd', '--weight-decay', type=float, default=1e-2,
                        help="Weight Decay for the optimizer.")

    parser.add_argument('--epochs', type=int, default=200, help="Number of Epochs.")

    parser.add_argument('--checkpoint-period', type=int, default=5,
                        help="Checkpoint save period.")  # default = 5

    parser.add_argument('--epoch-start', type=int, default=1,
                        help="Starting epoch number.")

    return parser.parse_args()


def train_model(args, model_P, model_N, datasets):
    # Instantiate the dataloaders
    train_dataloader = DataLoader(datasets["train"], batch_size=args.batch_size,
                                  shuffle=args.train_shuffle, drop_last=True,
                                  num_workers=args.num_workers)
    val_dataloader = DataLoader(datasets["val"], batch_size=args.batch_size,
                                shuffle=args.val_shuffle, drop_last=args.val_drop,
                                num_workers=args.num_workers)
    test_dataloader = DataLoader(datasets["test"], batch_size=args.batch_size,
                                 shuffle=args.test_shuffle, drop_last=args.test_drop,
                                 num_workers=args.num_workers)

    if args.verbose:
        print(f"Number of train iterations : {len(train_dataloader)}")
        print(f"Number of val iterations : {len(val_dataloader)}")
        print(f"Number of test iterations : {len(test_dataloader)}")


    ckpt = args.checkpoints_path
    latest_cpk_path = os.path.join(os.getcwd(), f"{ckpt}/latest.pth")


    loss_func = ContrastLoss(300, 4, 30, 45, 180) #best k=6
    IPR = IrrelevantPowerRatio(Fs=30, high_pass=45, low_pass=150)
    optimizer = torch.optim.AdamW(list(model_P.parameters()) + list(model_N.parameters()), lr=1e-4,weight_decay=1e-2)

    # Train configurations
    epochs = args.epochs
    checkpoint_period = args.checkpoint_period
    epoch_start = args.epoch_start

    if os.path.exists(latest_cpk_path):
        print('Context checkpoint exists. Loading state dictionaries.')
        checkpoint = torch.load(latest_cpk_path)
        model_P.load_state_dict(checkpoint['model_P_state_dict'])
        model_N.load_state_dict(checkpoint['model_N_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_start = checkpoint['epoch']
        epoch_start += 1


    if args.verbose:
        print(f"Checkpoint Period={checkpoint_period}. Epoch start = {epoch_start}")

    mae_best_loss = np.inf
    a = 0
    for epoch in range(epoch_start, epochs + 1):
        # Training Phase
        loss_train = 0
        ipr_train = 0
        r_loss = 0
        snr_loss = 0
        no_batches = 0
        sum_loss3 = 0
        if epoch == 20:
            a = 1

        for batch, (pseudo,p_rf, n_rf,t_rf) in enumerate(train_dataloader):
            model_P.train()
            model_N.train()

            p_rf = normalize(p_rf).type(torch.float32)
            n_rf = normalize(n_rf).type(torch.float32)
            t_rf = normalize(t_rf).type(torch.float32).to(args.device)
            pseudo = normalize(pseudo).type(torch.float32).to(args.device)

            p_rf = rotateIQ(p_rf)
            p_rf = torch.reshape(p_rf, (p_rf.shape[0], -1, p_rf.shape[3])).to(args.device) 
         
            n_rf = rotateIQ(n_rf)
            n_rf = torch.reshape(n_rf, (n_rf.shape[0], -1, n_rf.shape[3])).to(args.device)

            pred_signal_p, predlatent = model_P(p_rf)
            pred_signal_n,predlatent =model_N(n_rf)
        
            pred_signal_n = pred_signal_n.squeeze(1)
            pred_signal_p = pred_signal_p.squeeze(1)  
            loss,pos_loss,neg_loss,extra_loss = loss_func(pred_signal_p,pred_signal_n,t_rf)

            ipr = torch.mean(IPR(pred_signal_p.clone()))
       
            loss_sum =  loss
   

            optimizer.zero_grad()
            loss_sum.backward()
            optimizer.step()

     

       
            loss_train += loss_sum.item()
     
            no_batches += 1

        # Save the model every few epochs
        if (epoch % checkpoint_period == 0):
            torch.save({
                'model_P_state_dict': model_P.state_dict(),
                'model_N_state_dict': model_N.state_dict(),
            }, os.path.join(os.getcwd(), f"{ckpt}/{epoch}.pth"))
          
            # See if best checkpoint
            maes_val, _, _ = eval_RHB(root_path=args.data_dir, test_files=datasets["val"].rf_file_list,
                                           model=model_P, device=args.device,flag=False)

            current_loss = np.mean(maes_val)
            if (current_loss < mae_best_loss):
                mae_best_loss = current_loss
                torch.save({
                    'model_P_state_dict': model_P.state_dict(),
                    'model_N_state_dict': model_N.state_dict(),
                }, os.path.join(os.getcwd(), f"{ckpt}/best.pth"))
                # torch.save(model_P.state_dict(), os.path.join(os.getcwd(), f"{ckpt_P_path}/best.pth"))
                print("Best checkpoint saved!")
            print("Saved Checkpoint!")
        print(f"Epoch: {epoch} ; Loss: {loss_train / no_batches:>7f};IPR: {ipr_train / no_batches:>7f};loss3:{sum_loss3 / no_batches:>7f}")
        # SAVE CONTEXT AFTER EPOCH
        torch.save({
            "epoch": epoch,
            'model_P_state_dict': model_P.state_dict(),
            'model_N_state_dict': model_N.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, latest_cpk_path)
 


def main(args):
    # Import essential info, i.e. destination fol-3 .2  nmmder and fitzpatrick label path
    destination_folder = args.data_dir
    ckpt_path = args.checkpoints_path

    with open(args.folds_path, "rb") as fp:
        files_in_fold = pickle.load(fp)

  

    train_files = files_in_fold["train_files"]
    val_files = files_in_fold["val_files"]
    test_files = files_in_fold["test_files"]

    if args.verbose:
        print(f"There are {len(train_files)} train files. They are : {train_files}")
        print(f"There are {len(val_files)} val files. They are : {val_files}")
        print(f"There are {len(test_files)} test files. They are : {test_files}")

    # Dataset
    train_dataset = RHB(datapath=destination_folder,
                                     datapaths=train_files, frame_length_ppg=900)  # default = 128
    val_dataset = RHB(datapath=destination_folder,
                                   datapaths=val_files, frame_length_ppg=900)
    test_dataset = RHB(datapath=destination_folder,
                                    datapaths=test_files, frame_length_ppg=900)

    # Select the device
    if args.device is None:
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        args.device = torch.device(args.device)
    if args.verbose:
        print('Running on device: {}'.format(args.device))
    model_P = RF_conv_decoder()
    model_N = RF_conv_decoder()
   

    # Visualize some examples
    if args.viz:
        train_batch, train_batch_sig = train_dataset[0]
        val_batch, val_batch_sig = val_dataset[0]
        test_batch, test_batch_sig = test_dataset[0]

        if args.verbose:
            print(f"Train data and signal shapes : {train_batch.shape}, {train_batch_sig.shape}")
            print(f"Val data and signal shapes : {val_batch.shape}, {val_batch_sig.shape}")
            print(f"Test data and signal shapes : {test_batch.shape}, {test_batch_sig.shape}")

        plt.figure();
        plt.imshow(np.transpose(train_batch[:, 0], (1, 2, 0)))
        plt.figure();
        plt.plot(train_batch_sig)

        plt.figure();
        plt.imshow(np.transpose(val_batch[:, 0], (1, 2, 0)))
        plt.figure();
        plt.plot(val_batch_sig)

        plt.figure();
        plt.imshow(np.transpose(test_batch[:, 0], (1, 2, 0)))
        plt.figure();
        plt.plot(test_batch_sig)

        plt.show()


    os.makedirs(ckpt_path, exist_ok=True)

    # Check if Checkpoints exist

    all_ckpts = os.listdir(ckpt_path)
    if (len(all_ckpts) > 0):
        all_ckpts.sort()
        print(f"Checkpoints already exists at : {all_ckpts}")
    else:
        print("No checkpoints found, starting from scratch!")

    datasets = {"train": train_dataset, "val": val_dataset, "test": test_dataset}
    model_P.to(args.device)
    model_N.to(args.device)
    train_model(args, model_P, model_N, datasets)


if __name__ == '__main__':
    args = parseArgs()
    main(args)
