#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    :  下午8:11
# @Author  : Jingyang.Zhang
"""
Apply feature disentanglement in the ISBI and I2CVB and HK sites for prostate
"""
import argparse
import logging
import os
import random
import torch
from torch.utils.data import ConcatDataset, DataLoader

import dataloaders
# import models
import supervision
from models.sdnet_zjy import SDNet_zjy
from models.weight_init import initialize_weights
import utils
import numpy as np
from tensorboardX import SummaryWriter
import torch.optim as optim
import matplotlib.pyplot as plt

# -------------------- reproduction ------------------------ #
torch.manual_seed(123)
torch.cuda.manual_seed(123)
torch.cuda.manual_seed_all(123)
np.random.seed(123)  # Numpy module.
random.seed(123)  # Python random module.
torch.manual_seed(123)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = True
torch.set_default_tensor_type('torch.FloatTensor')


def parse_args():
    desc = "Pytorch implementation of SDNet (JYzhang)"
    parser = argparse.ArgumentParser(description=desc)
    # dir config
    parser.add_argument('--exp_dir', type=str, default='./exp/train_disentangle_cotwo')
    parser.add_argument('--data_npz', type=str, default='./data_prostate')
    parser.add_argument('--site_name', type=str, default='ISBI')
    # GPU config
    parser.add_argument('--gpu', type=str, default='0')
    # training config
    parser.add_argument('--z_length', type=int, default=16)
    parser.add_argument('--anatomy_channel', type=int, default=8)
    parser.add_argument('--kl_w', type=float, default=0.01)
    parser.add_argument('--seg_w', type=float, default=10)
    parser.add_argument('--reco_w', type=float, default=1)
    parser.add_argument('--recoz_w', type=float, default=1)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epoches', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('-wi', '--weight_init', type=str, default="xavier",
                        help='Weight initialization method, or path to weights file '
                             '(for fine-tuning or continuing training)')


    utils.check_folder(parser.parse_args().exp_dir)
    return parser.parse_args()


def validate_slice(model, dataloader, img_folder, anatomy_folder):
    model.eval()
    val_dice = supervision.AverageMeter()
    with torch.no_grad():
        for batch in dataloader:
            img, gt = batch['img'][:,1:2,:,:].cuda(), batch['gt'].cuda()
            reco, _, _, a_out, seg_pred, _, _, _, _ = model(img, 'validation')
            val_dice.update(supervision.dice_score(seg_pred[:,1,:,:].cpu() >= 0.5, gt[:,1,:,:].cpu()>=0.5))
    utils.save_imgs(img.cpu().detach(), gt.cpu().detach(), seg_pred.cpu().detach(), reco.cpu().detach(), img_folder)
    utils.save_anatomy_factors(a_out[0].cpu().numpy(), anatomy_folder)
    return val_dice.avg



def train(model, train_loader, val_loader, writer, args):
    optimizer = optim.Adam(model.parameters(), betas=(0.5, 0.999), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, verbose=True)

    l1_distance = torch.nn.L1Loss().cuda()
    bce_criterion = torch.nn.BCELoss()

    best_val_dice, best_epoch = 0.0, 0
    for epoch in range(args.epoches):
        kl_loss_epoch = supervision.AverageMeter()
        seg_loss_epoch = supervision.AverageMeter()
        reco_loss_epoch = supervision.AverageMeter()
        recoz_loss_epoch = supervision.AverageMeter()
        total_loss_epoch = supervision.AverageMeter()

        # train in each epoch
        for _, batch in enumerate(train_loader):
            img, gt = batch['img'][:,1:2,:,:].cuda(), batch['gt'].cuda()
            model.train()
            reco, z_out, z_out_tilede, a_out, seg_pred, mu_out, logvar_out, mu_out, logvar_out = model(img, 'training')
            # Lank loss for z_out
            a_out


            reco_loss = l1_distance(reco, img)
            kl_loss = supervision.KL_divergence(logvar_out, mu_out)
            dice_loss = supervision.dice_loss(seg_pred[:,1,:,:], gt[:,1,:,:])
            bce_loss = bce_criterion(seg_pred, gt)
            seg_loss = 0.5 * dice_loss + 0.5 * bce_loss
            recoz_loss = l1_distance(z_out_tilede, z_out)

            total_loss = args.kl_w * kl_loss + \
                         args.seg_w * seg_loss + \
                         args.reco_w * reco_loss + \
                         args.recoz_w * recoz_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            kl_loss_epoch.update(kl_loss.cpu())
            seg_loss_epoch.update(seg_loss.cpu())
            reco_loss_epoch.update(reco_loss.cpu())
            recoz_loss_epoch.update(recoz_loss.cpu())
            total_loss_epoch.update(total_loss.cpu())

        # print
        logging.info('\n Epoch[%4d/%4d] --> Train...' % (epoch, args.epoches))
        logging.info('\t [Total Loss = %.4f]: KL Loss = %.4f, Seg Loss = %.4f, Reco Loss = %.4f, RecoZ Loss = %.4f' %
                     (total_loss_epoch.avg, kl_loss_epoch.avg, seg_loss_epoch.avg, reco_loss_epoch.avg,
                      recoz_loss_epoch.avg))


        # validate and visualization
        val_dir = utils.check_folder(os.path.join(args.exp_dir, 'val_results'))
        val_img_path = os.path.join(val_dir, 'Ep_%04d_imgs.png' % epoch)
        val_anatomy_path = os.path.join(val_dir, 'Ep_%04d_anatomys.png' % epoch)
        val_dice = validate_slice(model, val_loader, val_img_path, val_anatomy_path)
        logging.info('\n Epoch[%4d/%4d] --> Valid...' % (epoch, args.epoches))
        logging.info('\t [Dice Coef = %.4f]' % val_dice)

        # check for plateau
        scheduler.step(val_dice)

        # tensorboard
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('Train/KL_Loss', kl_loss_epoch.avg, epoch)
        writer.add_scalar('Train/Seg_Loss', seg_loss_epoch.avg, epoch)
        writer.add_scalar('Train/Reco_Loss', reco_loss_epoch.avg, epoch)
        writer.add_scalar('Train/RecoZ_Loss', recoz_loss_epoch.avg, epoch)
        writer.add_scalar('Train/Total_Loss', total_loss_epoch.avg, epoch)
        writer.add_scalar('Val/Dice', val_dice, epoch)
        writer.add_images('Train/imgs', img, epoch)
        writer.add_images('Train/anatomy_0', a_out[:,0:1,:,:], epoch)
        writer.add_images('Train/anatomy_1', a_out[:, 1:2, :, :], epoch)
        writer.add_images('Train/anatomy_2', a_out[:, 2:3, :, :], epoch)
        writer.add_images('Train/anatomy_3', a_out[:, 3:4, :, :], epoch)
        writer.add_images('Train/anatomy_4', a_out[:, 4:5, :, :], epoch)
        writer.add_images('Train/anatomy_5', a_out[:, 5:6, :, :], epoch)
        writer.add_images('Train/anatomy_6', a_out[:, 6:7, :, :], epoch)
        writer.add_images('Train/anatomy_7', a_out[:, 7:8, :, :], epoch)
        writer.add_images('Train/reconstruction', reco, epoch)
        writer.add_images('Train/mask', gt[:,1:2,:,:], epoch)
        writer.add_images('Train/prediction', seg_pred[:,1:2,:,:], epoch)

        # # save images
        # train_img_dir = utils.check_folder(os.path.join(args.exp_dir, 'train_img'))
        # train_img_path = os.path.join(train_img_dir, 'Ep_%04d_imgs_dice_%.4f.png' % (epoch, val_dice))
        # utils.save_imgs(img.cpu().detach(), gt.cpu().detach(), seg_pred.cpu().detach(), reco.cpu().detach(),
        #                 train_img_path)
        # train_anatomy_path = os.path.join(train_img_dir, 'Ep_%04d_anatomys_dice_%.4f.png' % (epoch, val_dice))
        # utils.save_anatomy_factors(a_out[0].cpu().detach(), train_anatomy_path)
        #
        #
        # save model
        model_dir = utils.check_folder(os.path.join(args.exp_dir, 'models'))
        model_path = os.path.join(model_dir, 'Ep_%04d_dice_%.4f.pth' % (epoch, val_dice))
        torch.save(model.state_dict(), model_path)
        logging.info('\t [Save Model] to %s' % model_path)

        # save best model
        if val_dice >= best_val_dice:
            best_model_path = os.path.join(model_dir, 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)
            logging.info('\n Epoch[%4d/%4d] --> Dice improved from %.4f (epoch %4d) to %4f' %
                  (epoch, args.epoches, best_val_dice, best_epoch, val_dice))
            best_val_dice, best_epoch = val_dice, epoch
        else:
            logging.info('\n Epoch[%4d/%4d] --> Dice did not improved with %4f (epoch %d)' %
                         (epoch, args.epoches, best_val_dice, best_epoch))



def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # define logger
    logging.basicConfig(filename=os.path.join(args.exp_dir, 'train.log'), level=logging.DEBUG,
                        format='%(asctime)s %(message)s')
    logging.getLogger().addHandler(logging.StreamHandler())

    # print all parameters
    for name, v in vars(args).items():
        logging.info(name + ': ' + str(v))

    # dataloader config
    # sites = ['ISBI', 'I2CVB', 'HK']
    # train_loaders, val_loaders = dataloaders.MultiSite(args.data_npz, sites, args.batch_size,
    #                                                    norm=True)
    train_loaders, val_loaders = dataloaders.MultiSite(args.data_npz, sites, args.batch_size,
                                                       norm=True)
    # model config
    model = SDNet_zjy(384, 384, 1, 64, args.z_length, 'batchnorm', 'nearest', args.anatomy_channel, 2).cuda()
    num_params = utils.count_parameters(model)
    logging.info('Model Parameters: %d'% num_params)
    initialize_weights(model, args.weight_init)

    # summary writer config
    writer = SummaryWriter(log_dir=args.exp_dir, comment=args.exp_dir.split('/')[-1])

    # train
    train(model, train_loaders, val_loaders, writer, args)

if __name__ == '__main__':
    main()


