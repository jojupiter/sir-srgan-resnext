import argparse
import os
from math import log10
import pandas as pd
import torch.optim as optim
import torch.utils.data
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision
from tqdm import tqdm
import math
import torch
from torch import nn
from os import listdir
from os.path import join
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize
import torch
from torch import nn
from torchvision.models.vgg import vgg16, vgg19
from sklearn.model_selection import KFold
from math import exp
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import numpy as np
import torch.utils.checkpoint as checkpoint
import functools
from torch.nn import init
from torch import Tensor
import pywt



    

    
if __name__ == '__main__':

    CROP_SIZE = 128
    UPSCALE_FACTOR = 4
    NUM_EPOCHS = 2000
    NUM_FOLDS = 10

    data_dir="DIV2K_train_HR/DIV2K_train_HR"
    # Récupérer la liste des noms de fichiers dans le répertoire
    file_names = sorted(os.listdir(data_dir))


    # Créer une instance de KFold
    kfold = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

    for fold, (train_indices, val_indices) in enumerate(kfold.split(file_names)):
      print(f'Fold {fold + 1}/{NUM_FOLDS}')
      print("train_indices",train_indices)
      print("val_indices",val_indices)
      # Créer les jeux de données d'entraînement et de validation
      train_subset = TrainDatasetFromFolder(data_dir, crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR,indices=train_indices)
      val_subset = ValDatasetFromFolder(data_dir, upscale_factor=UPSCALE_FACTOR,file_indices=val_indices)

      train_loader = DataLoader(dataset=train_subset, num_workers=4, batch_size=16, shuffle=True)
      val_loader = DataLoader(dataset=val_subset, num_workers=4, batch_size=1, shuffle=False)

      # Réinitialiser le modèle, l'optimiseur, et les résultats
      netG = ResNeXt(16,8,UPSCALE_FACTOR)
      netD = Discriminator_UNet()
      print('# generator parameters:', sum(param.numel() for param in netG.parameters()))
      print('# discri parameters:', sum(param.numel() for param in netD.parameters()))
      R1 = Stander_Ranker()
      R2 = Feature_Ranker()
      get_feature = VGGFeatureExtractor(feature_layer=34, use_bn=False, use_input_norm=True)
      generator_criterion = GeneratorLoss()
      
      RANK_LOSS = Margin_Ranking_Loss()
      PDL = Self_Match(4,2)
      PDLF= Self_Match(2,2)
      PDLH= DWT_MATCH(4,2)   
      optimizerG = optim.Adam(netG.parameters())
      optimizerD = optim.Adam(netD.parameters())
      results = {'d_loss': [], 'g_loss': [], 'd_score': [], 'g_score': [], 'psnr': [], 'ssim': []}

      # ... (Reste du code d'entraînement, mais avec le train_loader et val_loader spécifiques à ce pli)  
      
      if torch.cuda.is_available():
          netG.cuda()
          netD.cuda()
          R1 = R1.cuda()
          R2 = R2.cuda()
          get_feature = get_feature.cuda()
          generator_criterion.cuda()
          RANK_LOSS.cuda()

      get_feature.eval()
      optimizerG = optim.Adam(netG.parameters(),lr=1e-4)
      optimizerD = optim.Adam(netD.parameters(),lr=1e-4)
      R1_OPT = optim.Adam(R1.parameters(), lr=1e-3)
      R2_OPT = optim.Adam(R2.parameters(), lr=1e-3)

      results = {'d_loss': [], 'g_loss': [], 'd_score': [], 'g_score': [], 'psnr': [], 'ssim': []}

      for epoch in range(1, NUM_EPOCHS + 1):
          train_bar = tqdm(train_loader)
          running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0}

          netG.train()
          netD.train()
          R1.train()
          R2.train()
          for data, target in train_bar:
              batch_size = data.size(0)
              running_results['batch_sizes'] += batch_size


              real_img = Variable(target)
              if torch.cuda.is_available():
                  real_img = real_img.cuda()
              z = Variable(data)
              if torch.cuda.is_available():
                  z = z.cuda()
              fake_img = netG(z)



              i1_img = (fake_img * 0.9 + real_img * 0.1).detach()
              i2_img = (fake_img * 0.8 + real_img * 0.2).detach()
              i3_img = (fake_img * 0.7 + real_img * 0.3).detach()
              i4_img = (fake_img * 0.6 + real_img * 0.4).detach()
              i5_img = (fake_img * 0.5 + real_img * 0.5).detach()
              i6_img = (fake_img * 0.4 + real_img * 0.6).detach()
              i8_img = (fake_img * 0.2 + real_img * 0.8).detach()

               #---------------------------------------------------------------
              # R1 train
              R1_OPT.zero_grad()

              sr_score = R1(fake_img.detach())
              i1_score = R1(i1_img)
              i2_score = R1(i2_img)
              i3_score = R1(i3_img)
              i4_score = R1(i4_img)
              i5_score = R1(i5_img)
              i6_score = R1(i6_img)
              i8_score = R1(i8_img)
              hr_score = R1(real_img.detach())

              r1_loss = RANK_LOSS([hr_score, i8_score, i6_score, i5_score, i4_score, i3_score, i2_score, i1_score, sr_score])
              r1_loss.backward()
              R1_OPT.step()




               #---------------------------------------------------------------
               # R2 train
              R2_OPT.zero_grad()

              sr_score = R2(fake_img.detach())
              i1_score = R2(i1_img.detach())
              i2_score = R2(i2_img.detach())
              i3_score = R2(i3_img.detach())
              i4_score = R2(i4_img.detach())
              i5_score = R2(i5_img.detach())
              i6_score = R2(i6_img.detach())
              i8_score = R2(i8_img.detach())
              hr_score = R2(real_img.detach())


              r2_loss = RANK_LOSS([hr_score, i8_score, i6_score, i5_score, i4_score, i3_score, i2_score, i1_score, sr_score])
              r2_loss.backward()
              R2_OPT.step()

              ############################
              # (1) Update D network: maximize D(x)-1-D(G(z))
              ###########################
             

              netD.zero_grad()
              real_out = netD(real_img).mean()
              fake_out = netD(fake_img).mean()
              d_loss = 1 - real_out + fake_out
              d_loss.backward(retain_graph=True)
              optimizerD.step()

              ############################
              # (2) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
              ###########################
              netG.zero_grad()
              ## The two lines below are added to prevent runetime error in Google Colab ##
              fake_img = netG(z)
              fake_out = netD(fake_img).mean()
              ##
              gan_loss = generator_criterion(fake_out, fake_img, real_img)



              #
              rank_pixel = torch.sigmoid(R1(fake_img))
              rank_pixel_loss = torch.mean(rank_pixel)

              #
              rank_feature = torch.sigmoid(R2(fake_img))
              rank_feature_loss = torch.mean(rank_feature)

              rank_loss = (rank_feature_loss + rank_pixel_loss)


              PDL_loss_pixel = torch.mean(torch.abs(PDL(real_img)-PDL(fake_img)))
              PDL_loss_feature = torch.mean(torch.abs(PDLF(get_feature(real_img))-PDLF(get_feature(fake_img))))
              PDLH_real_img = PDLH(real_img)
              PDLH_fake_img = PDLH(fake_img)
              diff_elements = [torch.abs(real - fake) for real, fake in zip(PDLH_real_img, PDLH_fake_img)]
              PDL_loss_hf = torch.mean(torch.stack(diff_elements))
              PDL_loss = PDL_loss_pixel + PDL_loss_feature + PDL_loss_hf

              

              g_loss = gan_loss  + 0.03*rank_loss + 0.01*PDL_loss

              g_loss.backward()

              fake_img = netG(z)
              fake_out = netD(fake_img).mean()


              optimizerG.step()


             


              # loss for current batch before optimization
              running_results['g_loss'] += g_loss.item() * batch_size
              running_results['d_loss'] += d_loss.item() * batch_size
              running_results['d_score'] += real_out.item() * batch_size
              running_results['g_score'] += fake_out.item() * batch_size

              train_bar.set_description(desc='[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f' % (
                  epoch, NUM_EPOCHS, running_results['d_loss'] / running_results['batch_sizes'],
                  running_results['g_loss'] / running_results['batch_sizes'],
                  running_results['d_score'] / running_results['batch_sizes'],
                  running_results['g_score'] / running_results['batch_sizes']))

          netG.eval()
          out_path = 'training_results/SRF_'+str(fold + 1)+'U' + str(UPSCALE_FACTOR) + '/'
          if not os.path.exists(out_path):
              os.makedirs(out_path)

          with torch.no_grad():
              val_bar = tqdm(val_loader)
              valing_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
              #val_images = []
              for val_lr, val_hr_restore, val_hr in val_bar:
                  batch_size = val_lr.size(0)
                  valing_results['batch_sizes'] += batch_size
                  lr = val_lr
                  hr = val_hr
                  if torch.cuda.is_available():
                      lr = lr.cuda()
                      hr = hr.cuda()
                  sr = netG(lr)

                  batch_mse = ((sr - hr) ** 2).data.mean()
                  valing_results['mse'] += batch_mse * batch_size
                  batch_ssim =ssim(sr, hr).item()
                  valing_results['ssims'] += batch_ssim * batch_size
                  valing_results['psnr'] = 10 * log10((hr.max()**2) / (valing_results['mse'] / valing_results['batch_sizes']))
                  valing_results['ssim'] = valing_results['ssims'] / valing_results['batch_sizes']
                  val_bar.set_description(
                      desc='FOLD: %d [converting LR images to SR images] PSNR: %.4f dB SSIM: %.4f' % (fold+1,
                          valing_results['psnr'], valing_results['ssim']))

                 # val_images.extend(
                #      [display_transform()(val_hr_restore.squeeze(0)), display_transform()(hr.data.cpu().squeeze(0)),
               #       display_transform()(sr.data.cpu().squeeze(0))])
              #val_images = torch.stack(val_images)
              #val_images = torch.chunk(val_images, val_images.size(0) // 15)
              #val_save_bar = tqdm(val_images, desc='[saving training results]')
              #index = 1
              #for image in val_save_bar:
               #   image = utils.make_grid(image, nrow=3, padding=5)
                #  utils.save_image(image, out_path + 'epoch_%d_index_%d.png' % (epoch, index), padding=5)
                 # index += 1

          # save model parameters
          out_path='epochs'+str(fold + 1)+''
          if not os.path.exists(out_path):
              os.makedirs(out_path)
          torch.save(netG.state_dict(), 'epochs'+str(fold + 1)+'/netG_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch))
          torch.save(netD.state_dict(), 'epochs'+str(fold + 1)+'/netD_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch))
          # save loss\scores\psnr\ssim
          results['d_loss'].append(running_results['d_loss'] / running_results['batch_sizes'])
          results['g_loss'].append(running_results['g_loss'] / running_results['batch_sizes'])
          results['d_score'].append(running_results['d_score'] / running_results['batch_sizes'])
          results['g_score'].append(running_results['g_score'] / running_results['batch_sizes'])
          results['psnr'].append(valing_results['psnr'])
          results['ssim'].append(valing_results['ssim'])

          if epoch % 10 == 0 and epoch != 0:
              out_path = 'statistics/'
              if not os.path.exists(out_path):
                os.makedirs(out_path)
              data_frame = pd.DataFrame(
                  data={'Loss_D': results['d_loss'], 'Loss_G': results['g_loss'], 'Score_D': results['d_score'],
                        'Score_G': results['g_score'], 'PSNR': results['psnr'], 'SSIM': results['ssim']},
                  index=range(1, epoch + 1))
              data_frame.to_csv(out_path + 'srf_'+str(fold+1)+'_' + str(UPSCALE_FACTOR) + '_train_results.csv', index_label='Epoch')





