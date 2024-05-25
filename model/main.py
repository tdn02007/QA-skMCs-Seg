from network import ResidualGenerator, DiceLoss

import os
import sys
import time
import torch
import torch.nn as nn
import argparse
from torch import optim
from torch.utils.data import DataLoader
import torchvision
from collections import OrderedDict


sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    # Training settings
    parser = argparse.ArgumentParser(description = 'Model')
    parser.add_argument('--epochs', type = int, default =500,
                        metavar='N', help = 'number of epochs to train')
    parser.add_argument('--lr', type = float, default = 0.0001,
                        metavar='FLOAT', help = 'learning rate')
    parser.add_argument('--batch_size', type = int, default =24,
                        metavar = 'N', help = 'input batch size for training')
    parser.add_argument('--resume', type=bool, default=False,
                        metavar='BOOL', help='resume')
    parser.add_argument('--checkpoint', type=str, default='./model/checkpoint/model_train.pth',
                        metavar='STRING', help='checkpoint')
    parser.add_argument('--save_path', type=str, default='./results',
                        metavar='STRING', help='save_path')
    parser.add_argument('--mode', type=str, default='train',
                        help='train or test')
    parser.add_argument('--beta1', type = float, default = 0.5,
                        metavar='FLOAT', help = 'beta1')
    parser.add_argument('--beta2', type = float, default = 0.999,
                        metavar='FLOAT', help = 'beta2')
    args = parser.parse_args()

    if args.mode == 'train':
        train(args)
    else:
        test(args)



def train(args):
    from data_load_torch import DatasetLoader
    print("data loading")

    dataset = DatasetLoader()

    train_loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True
    )

    # Networks
    generator = ResidualGenerator().to(device=device)

    if args.resume:
        print("model is loaded")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        new_state_dict = OrderedDict()
        for k, v in checkpoint["g_state_dict"].items():
            name = k[7:] if k.startswith("module.") else k
            new_state_dict[name] = v

        generator.load_state_dict(new_state_dict)

    if torch.cuda.device_count() > 1:
        generator = nn.DataParallel(generator)

    # Losses
    criterionBCE = nn.BCELoss()
    dice_loss = DiceLoss()

    sigmoid = nn.Sigmoid()

    # Optimizers
    g_optimizer = optim.Adam(generator.parameters(), args.lr, [args.beta1, args.beta2])

    if not os.path.exists("checkpoint"):
        os.mkdir("checkpoint")

    last_loss = 0

    for epoch in range(args.epochs):
        print("epoch: ", epoch)
        G_loss = 0

        generator.train()

        for sample in train_loader:
            input_image = sample["input_image"].to(device=device)
            label_image = sample["label_image"].to(device=device)

            g_optimizer.zero_grad()

            r_wh_image = label_image[:, 0, :, :]
            r_nul_image = label_image[:, 1, :, :]
            r_mask = label_image[:, 2, :, :]

            f_wh_image, f_nul_image, f_mask = generator(input_image)

            loss_G_fake1 = criterionBCE(sigmoid(f_wh_image), r_wh_image)
            loss_G_fake2 = criterionBCE(sigmoid(f_nul_image), r_nul_image)
            loss_G_fake3 = criterionBCE(sigmoid(f_mask), r_mask)

            loss_dice1 = dice_loss(f_wh_image, r_wh_image)
            loss_dice2 = dice_loss(f_nul_image, r_nul_image)
            loss_dice3 = dice_loss(f_mask, r_mask)

            loss_G = loss_G_fake1 + loss_G_fake2 + loss_G_fake3 + loss_dice1 + loss_dice2 + loss_dice3

            G_loss += loss_G.item()

            loss_G.backward()
            g_optimizer.step()

        print("G_loss: ", G_loss)

        if last_loss == 0 or G_loss < last_loss:
            model_out_path = "checkpoint/model_train.pth"

            torch.save(
                {
                    "g_state_dict": generator.state_dict(),
                    "g_optimizer_dict": g_optimizer.state_dict(),
                },
                model_out_path,
            )

            last_loss = G_loss

            print("Checkpoint saved to {}_{}_{}".format(
                "checkpoint", epoch, last_loss))
            

def test(args):
    from test_data_load_torch import DatasetLoader

    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(args.save_path+"/cls1", exist_ok=True)
    os.makedirs(args.save_path+"/cls2", exist_ok=True)
    os.makedirs(args.save_path+"/cls3", exist_ok=True)

    print("data loading")
    dataset = DatasetLoader()

    train_loader = DataLoader(
        dataset, batch_size=args.batch_size
    )

    # Networks
    generator = ResidualGenerator().to(device=device)

    checkpoint = torch.load(args.checkpoint, map_location=device)

    new_state_dict = OrderedDict()
    for k, v in checkpoint["g_state_dict"].items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v

    generator.load_state_dict(new_state_dict)

    generator.eval()

    with torch.no_grad():
        start = time.time()
        for i, sample in enumerate(train_loader):
            input_image = sample["input_image"].to(device=device)

            f_wh_image, f_nul_image, f_mask = generator(input_image)

            result_name = (4-len(str(i)))*"0" + str(i)

            torchvision.utils.save_image(
                f_wh_image,
                os.path.join(
                    args.save_path+"/cls1",
                    f"Fake image-{result_name}.tif",
                ),
            )
            torchvision.utils.save_image(
                f_nul_image,
                os.path.join(
                    args.save_path+"/cls2",
                    f"Fake image-{result_name}.tif",
                ),
            )
            torchvision.utils.save_image(
                f_mask,
                os.path.join(
                    args.save_path+"/cls3",
                    f"Fake image-{result_name}.tif",
                ),
            )

        print("time: ", time.time() - start)


if __name__ == "__main__":
    main()
