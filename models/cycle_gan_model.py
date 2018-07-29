import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import torchvision.transforms as transforms
from PIL import Image

import sys


class CycleGANModel(BaseModel):
    def name(self):
        return 'CycleGANModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        nb = opt.batchSize
        size = opt.fineSize
        patch_size = opt.patchSize

        self.input_A = self.Tensor(nb, opt.input_nc, size, size)
        self.input_B = self.Tensor(nb, opt.output_nc, size, size)
        self.input_A_patch = self.Tensor(nb, opt.input_nc, patch_size, patch_size)
        self.input_B_patch = self.Tensor(nb, opt.output_nc, patch_size, patch_size)
        self.input_A_boxes = []
        self.input_B_boxes = []

        # load/define networks
        # The naming conversion is different from those used in the paper
        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)

        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids, opt.num_D)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids, opt.num_D)
        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netG_A, 'G_A', which_epoch)
            self.load_network(self.netG_B, 'G_B', which_epoch)
            if self.isTrain:
                self.load_network(self.netD_A, 'D_A', which_epoch)
                self.load_network(self.netD_B, 'D_B', which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()


            self.criterionVGG = networks.VGGLoss(self.gpu_ids)

            self.criterionGD = networks.GDLoss(self.gpu_ids)

            # initialize optimizers
            if opt.use_SGD:
                self.optimizer_G = torch.optim.SGD(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                                    lr=opt.lr, momentum=0.99)
                self.optimizer_D_A = torch.optim.SGD(self.netD_A.parameters(), lr=opt.lr, momentum=0.99)
                self.optimizer_D_B = torch.optim.SGD(self.netD_B.parameters(), lr=opt.lr, momentum=0.99)

            else:
                self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                                    lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D_A)
            self.optimizers.append(self.optimizer_D_B)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG_A)
        networks.print_network(self.netG_B)
        if self.isTrain:
            networks.print_network(self.netD_A)
            networks.print_network(self.netD_B)
        print('-----------------------------------------------')

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']
        input_A_patch = input['A_patch' if AtoB else 'B_patch']
        input_B_patch = input['B_patch' if AtoB else 'A_patch']
        input_A_boxes = input['A_boxes' if AtoB else 'B_boxes']
        input_B_boxes = input['B_boxes' if AtoB else 'A_boxes']


        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        self.input_A_patch.resize_(input_A_patch.size()).copy_(input_A_patch)
        self.input_B_patch.resize_(input_B_patch.size()).copy_(input_B_patch)
        self.input_A_boxes = input_A_boxes
        self.input_B_boxes = input_B_boxes

        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)
        self.real_A_patch = Variable(self.input_A_patch)
        self.real_B_patch = Variable(self.input_B_patch)
        self.real_A_boxes = self.input_A_boxes
        self.real_B_boxes = self.input_B_boxes

    def test(self):
        real_A = Variable(self.input_A, volatile=True)
        fake_B = self.netG_A(real_A)
        self.rec_A = self.netG_B(fake_B).data
        self.fake_B = fake_B.data

        real_B = Variable(self.input_B, volatile=True)
        fake_A = self.netG_B(real_B)
        self.rec_B = self.netG_A(fake_A).data
        self.fake_A = fake_A.data

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_D_basic(self, netD, real, fake, real_patch, fake_patch):
        # Real
        pred_real = netD(real, real_patch)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach(), fake_patch.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        fake_B, fake_B_patch = self.fake_B_pool.query(self.fake_B, self.fake_B_patch)

        loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B, self.real_B_patch, fake_B_patch)
        self.loss_D_A = loss_D_A.data[0]

    def backward_D_B(self):
        fake_A, fake_A_patch = self.fake_A_pool.query(self.fake_A, self.fake_A_patch)

        loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A, self.real_A_patch, fake_A_patch)
        self.loss_D_B = loss_D_B.data[0]

    def backward_G(self):
        lambda_idt = self.opt.identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        lamda_vgg = self.opt.lambda_vgg
        lambda_patch = self.opt.lambda_patch
        lamda_GD = self.opt.lambda_GD

        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed.
            idt_A = self.netG_A(self.real_B)
            idt_A_patch = self.get_patch(idt_A, self.real_B_boxes)

            loss_idt_A = self.criterionIdt(idt_A, self.real_B) * lambda_B * lambda_idt
            loss_idt_A_patch = self.criterionIdt(idt_A_patch, self.real_B_patch) * lambda_patch * lambda_idt

            # G_B should be identity if real_A is fed.
            idt_B = self.netG_B(self.real_A)
            idt_B_patch = self.get_patch(idt_B, self.real_A_boxes)

            loss_idt_B = self.criterionIdt(idt_B, self.real_A) * lambda_A * lambda_idt
            loss_idt_B_patch = self.criterionIdt(idt_B_patch, self.real_A_patch) * lambda_patch * lambda_idt

            self.idt_A = idt_A.data
            self.idt_B = idt_B.data
            self.idt_A_patch = idt_A_patch.data
            self.idt_B_patch = idt_B_patch.data

            self.loss_idt_A = loss_idt_A.data[0]
            self.loss_idt_B = loss_idt_B.data[0]
            self.loss_idt_A_patch = loss_idt_A_patch.data[0]
            self.loss_idt_B_patch = loss_idt_B_patch.data[0]


        else:
            loss_idt_A = 0
            loss_idt_B = 0
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        fake_B = self.netG_A(self.real_A)

        if self.opt.use_vgg:
            loss_VGG_A = self.compute_vgg_loss(fake_B, self.real_A_boxes) * lamda_vgg

        #loss_perceptual_A = self.criterionVGG(fake_B, self.real_A) * lamda_vgg

        loss_GD_A = self.criterionGD(fake_B, self.real_A) * lamda_GD

        fake_B_patch = self.get_patch(fake_B, self.real_A_boxes)
        pred_fake = self.netD_A(fake_B, fake_B_patch)
        loss_G_A = self.criterionGAN(pred_fake, True)


        # GAN loss D_B(G_B(B))
        fake_A = self.netG_B(self.real_B)
        if self.opt.use_vgg:
            loss_VGG_B = self.compute_vgg_loss(fake_A, self.real_B_boxes) * lamda_vgg

        #loss_perceptual_B = self.criterionVGG(fake_A, self.real_B) * lamda_vgg

        loss_GD_B = self.criterionGD(fake_A, self.real_B) * lamda_GD

        fake_A_patch = self.get_patch(fake_A, self.real_B_boxes)

        pred_fake = self.netD_B(fake_A, fake_A_patch)
        loss_G_B = self.criterionGAN(pred_fake, True)

        # Forward cycle loss
        rec_A = self.netG_B(fake_B)
        rec_A_patch = self.get_patch(rec_A, self.real_A_boxes)

        loss_cycle_A = self.criterionCycle(rec_A, self.real_A) * lambda_A
        loss_cycle_A_patch = self.criterionCycle(rec_A_patch, self.real_A_patch) * lambda_patch

        # Backward cycle loss
        rec_B = self.netG_A(fake_A)
        rec_B_patch = self.get_patch(rec_B, self.real_B_boxes)

        loss_cycle_B = self.criterionCycle(rec_B, self.real_B) * lambda_B
        loss_cycle_B_patch = self.criterionCycle(rec_B_patch, self.real_B_patch) * lambda_patch

        if not self.opt.use_vgg:
            loss_VGG_A = 0
            loss_VGG_B = 0
            self.loss_VGG_A = 0
            self.loss_VGG_B = 0

        # combined loss
        loss_G = loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B + loss_cycle_A_patch + loss_cycle_B_patch + loss_idt_A + loss_idt_B + loss_idt_A_patch + loss_idt_B_patch + loss_VGG_A + loss_VGG_B + loss_GD_A + loss_GD_B
#+ loss_perceptual_A + loss_perceptual_B
 
        loss_G.backward()

        self.fake_B = fake_B.data
        self.fake_A = fake_A.data
        self.fake_B_patch = fake_B_patch.data
        self.fake_A_patch = fake_A_patch.data

        self.rec_A = rec_A.data
        self.rec_B = rec_B.data
        self.rec_A_patch = rec_A_patch.data
        self.rec_B_patch = rec_B_patch.data

        self.loss_G_A = loss_G_A.data[0]
        self.loss_G_B = loss_G_B.data[0]
        self.loss_cycle_A = loss_cycle_A.data[0]
        self.loss_cycle_B = loss_cycle_B.data[0]
        self.loss_cycle_A_patch = loss_cycle_A_patch.data[0]
        self.loss_cycle_B_patch = loss_cycle_B_patch.data[0]

        self.loss_GD_A = loss_GD_A.data[0]
        self.loss_GD_B = loss_GD_B.data[0]

        if self.opt.use_vgg:
            self.loss_VGG_A = loss_VGG_A.data[0]
            self.loss_VGG_B = loss_VGG_B.data[0]

    def optimize_parameters(self):
        # forward
        self.forward()
        # G_A and G_B
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        # D_A
        self.optimizer_D_A.zero_grad()
        self.backward_D_A()
        self.optimizer_D_A.step()
        # D_B
        self.optimizer_D_B.zero_grad()
        self.backward_D_B()
        self.optimizer_D_B.step()


    def compute_vgg_loss(self, img_tensor, boxes):
        crop0 = img_tensor[:,:, int(boxes[0][1].numpy()) : int(boxes[0][3].numpy()), int(boxes[0][0].numpy()) : int(boxes[0][2].numpy()) ]
        crop1 = img_tensor[:,:, int(boxes[1][1].numpy()) : int(boxes[1][3].numpy()), int(boxes[1][0].numpy()) : int(boxes[1][2].numpy()) ]
        crop2 = img_tensor[:,:, int(boxes[2][1].numpy()) : int(boxes[2][3].numpy()), int(boxes[2][0].numpy()) : int(boxes[2][2].numpy()) ]

        loss_G_VGG = (self.criterionVGG(crop0, crop1) + self.criterionVGG(crop1, crop2) + self.criterionVGG(crop2, crop0)) / 3.0

        return loss_G_VGG


    def  get_patch(self, img_tensor, boxes):



        crop0 = img_tensor[:,:, int(boxes[0][1].numpy()) : int(boxes[0][3].numpy()), int(boxes[0][0].numpy()) : int(boxes[0][2].numpy()) ]
        crop1 = img_tensor[:,:, int(boxes[1][1].numpy()) : int(boxes[1][3].numpy()), int(boxes[1][0].numpy()) : int(boxes[1][2].numpy()) ]
        crop2 = img_tensor[:,:, int(boxes[2][1].numpy()) : int(boxes[2][3].numpy()), int(boxes[2][0].numpy()) : int(boxes[2][2].numpy()) ]

        output = torch.cat((crop0, crop1, crop2), dim=3)

        return output

        # img = transforms.toPILImage(img_tensor)
        # # img = util.tensor2im(img_tensor)
        #
        # crop_size = self.opt.patchSize
        #
        # crop0 = img.crop(self.boxes[0])
        # crop1 = img.crop(self.boxes[1])
        # crop2 = img.crop(self.boxes[2])
        #
        # img_crop = Image.new('RGB', (crop_size * 3, crop_size))
        # img_crop.paste(crop0, (0, 0))
        # img_crop.paste(crop1, (crop_size, 0))
        # img_crop.paste(crop2, (crop_size * 2, 0))
        #
        # return transforms.ToTensor(img_crop)


    def get_current_errors(self):
        ret_errors = OrderedDict([('D/D_A', self.loss_D_A), ('G/G_A', self.loss_G_A), ('D/Cyc_A', self.loss_cycle_A), ('D/Cyc_A_patch', self.loss_cycle_A_patch),
                                 ('D/D_B', self.loss_D_B), ('G/G_B', self.loss_G_B), ('D/Cyc_B',  self.loss_cycle_B), ('D/Cyc_B_patch',  self.loss_cycle_B_patch),
                                  ])
        if self.opt.identity > 0.0:
            ret_errors['D/idt_A'] = self.loss_idt_A
            ret_errors['D/idt_B'] = self.loss_idt_B
            ret_errors['D/idt_A_patch'] = self.loss_idt_A_patch
            ret_errors['D/idt_B_patch'] = self.loss_idt_B_patch

        if self.opt.use_vgg:
            ret_errors['D/VGG_A'] = self.loss_VGG_A
            ret_errors['D/VGG_B'] = self.loss_VGG_B

            ret_errors['D/GD_A'] = self.loss_GD_A
            ret_errors['D/GD_B'] = self.loss_GD_B

        return ret_errors

    def get_current_visuals(self):
        real_A = util.tensor2im(self.input_A)
        fake_B = util.tensor2im(self.fake_B)
        rec_A = util.tensor2im(self.rec_A)
        real_B = util.tensor2im(self.input_B)
        fake_A = util.tensor2im(self.fake_A)
        rec_B = util.tensor2im(self.rec_B)

        real_A_patch = util.tensor2im(self.input_A_patch)
        fake_B_patch = util.tensor2im(self.fake_B_patch)
        rec_A_patch = util.tensor2im(self.rec_A_patch)
        real_B_patch = util.tensor2im(self.input_B_patch)
        fake_A_patch = util.tensor2im(self.fake_A_patch)
        rec_B_patch = util.tensor2im(self.rec_B_patch)


        ret_visuals = OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('rec_A', rec_A),
                                   ('real_B', real_B), ('fake_A', fake_A), ('rec_B', rec_B),
                                   ('real_A_patch', real_A_patch), ('fake_B_patch', fake_B_patch), ('rec_A_patch', rec_A_patch),
                                   ('real_B_patch', real_B_patch), ('fake_A_patch', fake_A_patch), ('rec_B_patch', rec_B_patch)])
        if self.opt.isTrain and self.opt.identity > 0.0:
            ret_visuals['idt_A'] = util.tensor2im(self.idt_A)
            ret_visuals['idt_B'] = util.tensor2im(self.idt_B)
        return ret_visuals

    def get_current_visuals_test(self):
        real_A = util.tensor2im(self.input_A)
        fake_B = util.tensor2im(self.fake_B)
        rec_A = util.tensor2im(self.rec_A)
        real_B = util.tensor2im(self.input_B)
        fake_A = util.tensor2im(self.fake_A)
        rec_B = util.tensor2im(self.rec_B)


        ret_visuals = OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('rec_A', rec_A),
                                   ('real_B', real_B), ('fake_A', fake_A), ('rec_B', rec_B)])
        if self.opt.isTrain and self.opt.identity > 0.0:
            ret_visuals['idt_A'] = util.tensor2im(self.idt_A)
            ret_visuals['idt_B'] = util.tensor2im(self.idt_B)
        return ret_visuals

    def save(self, label):
        self.save_network(self.netG_A, 'G_A', label, self.gpu_ids)
        self.save_network(self.netD_A, 'D_A', label, self.gpu_ids)
        self.save_network(self.netG_B, 'G_B', label, self.gpu_ids)
        self.save_network(self.netD_B, 'D_B', label, self.gpu_ids)
