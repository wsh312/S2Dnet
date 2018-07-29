import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
from torch.optim import lr_scheduler
import numpy as np
import torch.nn.functional as F

from torchvision import transforms
from torch.autograd import Variable
###############################################################################
# Functions
###############################################################################


class PixelNormLayer(nn.Module):

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super(PixelNormLayer, self).__init__()

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + 1e-8)



def weights_init_normal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.uniform(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.uniform(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)



def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'pixel':
        norm_layer = functools.partial(PixelNormLayer, affine=False)
    elif layer_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def define_G(input_nc, output_nc, ngf, which_model_netG, norm='batch', use_dropout=False, init_type='normal', gpu_ids=[]):
    netG = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())

    if which_model_netG == 'resnet_9blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9, gpu_ids=gpu_ids)
    elif which_model_netG == 'resnet_6blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6, gpu_ids=gpu_ids)
    elif which_model_netG == 'resnet_verti':
        netG = ResnetGeneratorVerti(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6, gpu_ids=gpu_ids)
    elif which_model_netG == 'resnet_hori':
        netG = my_generator(input_nc, output_nc)
    elif which_model_netG == 'unet_128':
        netG = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
    elif which_model_netG == 'unet_256':
        netG = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
    elif which_model_netG == 'unet_512':
        netG = UnetGenerator(input_nc, output_nc, 9, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
    elif which_model_netG == 'unet_512_Re1':
        netG = UnetGeneratorRe1(input_nc, output_nc, 9, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
    elif which_model_netG == 'unet_512_Re2':
        netG = UnetGeneratorRe2(input_nc, output_nc, 9, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
    elif which_model_netG == 'unet_1024':
        netG = UnetGenerator1024(input_nc, output_nc, 9, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
    elif which_model_netG == 'unet_1024_7x7':
        netG = UnetGenerator1024_7x7(input_nc, output_nc, 9, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
    elif which_model_netG == 'resnet_1block':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=1, gpu_ids=gpu_ids)
    elif which_model_netG == 'xrnn_512':
        netG = Unet_G_XRNN_512(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
    if len(gpu_ids) > 0:
        netG.cuda(gpu_ids[0])
    init_weights(netG, init_type=init_type)
    return netG


def define_D(input_nc, ndf, which_model_netD,
             n_layers_D=3, norm='batch', use_sigmoid=False, init_type='normal', gpu_ids=[], numD = 3):
    netD = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())
    if which_model_netD == 'basic':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    elif which_model_netD == 'patch_128_64':
        netD = NLayerDiscriminator_128_64(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    elif which_model_netD == 'patch_128_32':
        netD = NLayerDiscriminator_128_32(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    elif which_model_netD == 'patch_256_128':
        netD = NLayerDiscriminator_256_128(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    elif which_model_netD == 'patch_512_256':
        netD = NLayerDiscriminator_512_256(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    elif which_model_netD == 'patch_512_128':
        netD = NLayerDiscriminator_512_128(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    elif which_model_netD == 'patch_512_128_multi':
        netD = NLayerDiscriminator_512_128_multi(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    elif which_model_netD == 'patch_512_128_multi_new':
        netD = NLayerDiscriminator_512_128_multi_new(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    elif which_model_netD == 'patch_512_256_multi':
        netD = NLayerDiscriminator_512_256_multi(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids, num_D = numD)
    elif which_model_netD == 'patch_512_256_multi_new':
        netD = NLayerDiscriminator_512_256_multi_new(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids, num_D = numD)
    elif which_model_netD == 'n_layers':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    elif which_model_netD == 'multi_scale':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' %
                                  which_model_netD)
    if use_gpu:
        netD.cuda(gpu_ids[0])
    init_weights(netD, init_type=init_type)
    return netD


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


##############################################################################
# Classes
##############################################################################


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)

class my_generator(nn.Module):
    # initializers
    def __init__(self, input_nc, output_nc, ngf=16, nb=7):
        super(my_generator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.nb = nb
        self.conv1 = nn.Conv2d(input_nc, ngf, 7, 1, 0)
        self.conv1_norm = nn.InstanceNorm2d(ngf)
        self.conv2 = nn.Conv2d(ngf, ngf * 2, 3, 2, 1)
        self.conv2_norm = nn.InstanceNorm2d(ngf * 2)
        self.conv3 = nn.Conv2d(ngf * 2, ngf * 4, 3, 2, 1)
        self.conv3_norm = nn.InstanceNorm2d(ngf * 4)
        # self.conv4 = nn.Conv2d(ngf * 4, ngf * 4, 3, 2, 1)
        # self.conv4_norm = nn.InstanceNorm2d(ngf * 4)
        # self.conv5 = nn.Conv2d(ngf * 4, ngf * 4, 3, 2, 1)
        # self.conv5_norm = nn.InstanceNorm2d(ngf * 4)

        self.resnet_blocks = []
        for i in range(nb):
            self.resnet_blocks.append(resnet_block(ngf * 4, 3, 1, 1))
            self.resnet_blocks[i].weight_init(0, 0.02)

        self.resnet_blocks = nn.Sequential(*self.resnet_blocks)

        # self.deconv0 = nn.ConvTranspose2d(ngf * 4, ngf * 4, 3, 2, 1, 1)
        # self.deconv0_norm = nn.InstanceNorm2d(ngf * 4)
        # self.deconv1 = nn.ConvTranspose2d(ngf * 4, ngf * 4, 3, 2, 1, 1)
        # self.deconv1_norm = nn.InstanceNorm2d(ngf * 4)
        self.deconv2 = nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, 2, 1, 1)
        self.deconv2_norm = nn.InstanceNorm2d(ngf * 2)
        self.deconv3 = nn.ConvTranspose2d(ngf * 2, ngf, 3, 2, 1, 1)
        self.deconv3_norm = nn.InstanceNorm2d(ngf)
        self.deconv4 = nn.Conv2d(ngf, output_nc, 7, 1, 0)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        x = F.pad(input, (3, 3, 3, 3), 'reflect')
        x = F.relu(self.conv1_norm(self.conv1(x)))
        x = F.relu(self.conv2_norm(self.conv2(x)))
        x = F.relu(self.conv3_norm(self.conv3(x)))
        # x = F.relu(self.conv4_norm(self.conv4(x)))
        # x = F.relu(self.conv5_norm(self.conv5(x)))
        x = self.resnet_blocks(x)
        # x = F.relu(self.deconv0_norm(self.deconv0(x)))
        # x = F.relu(self.deconv1_norm(self.deconv1(x)))
        x = F.relu(self.deconv2_norm(self.deconv2(x)))
        x = F.relu(self.deconv3_norm(self.deconv3(x)))
        x = F.pad(x, (3, 3, 3, 3), 'reflect')
        o = F.tanh(self.deconv4(x))

        return o

# resnet block with reflect padding
class resnet_block(nn.Module):
    def __init__(self, channel, kernel, stride, padding):
        super(resnet_block, self).__init__()
        self.channel = channel
        self.kernel = kernel
        self.strdie = stride
        self.padding = padding
        self.conv1 = nn.Conv2d(channel, channel, kernel, stride, 0)
        self.conv1_norm = nn.InstanceNorm2d(channel)
        self.conv2 = nn.Conv2d(channel, channel, kernel, stride, 0)
        self.conv2_norm = nn.InstanceNorm2d(channel)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, input):
        x = F.pad(input, (self.padding, self.padding, self.padding, self.padding), 'reflect')
        x = F.relu(self.conv1_norm(self.conv1(x)))
        x = F.pad(x, (self.padding, self.padding, self.padding, self.padding), 'reflect')
        x = self.conv2_norm(self.conv2(x))

        return input + x

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()





# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class ResnetGeneratorVerti(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=4, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 gpu_ids=[], padding_type='reflect'):
        assert (n_blocks >= 0)
        super(ResnetGeneratorVerti, self).__init__()

        input_nc = input_nc * 3
        output_nc = output_nc * 3

        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [
                ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                            use_bias=use_bias)]

        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):


        sz = 512
        i = 0
        input_A = input[:, :, :, sz*i: sz*(i+1)]
        i = 1
        input_B = input[:, :, :, sz * i: sz * (i + 1)]
        i = 2
        input_C = input[:, :, :, sz * i: sz * (i + 1)]

        input_cat = torch.cat((input_A, input_B, input_C), 1)

        output_cat = self.model(input_cat)

        # output = output_cat.view(input.size())  .view(1, 3, 512, 512)

        output = input.clone()
        i = 0
        # tmp = output_cat[:, i * 3: (i * 3 + 3), :, :]
        # tmp = tmp.view(1, 3, 512, 512)
        output[:, :, :, sz * i: sz * (i+1)] = output_cat[:, i * 3: (i * 3 + 3), :, :]
        i = 1
        output[:, :, :, sz * i: sz * (i+1)] = output_cat[:, i * 3: (i * 3 + 3), :, :]
        i = 2
        output[:, :, :, sz * i: sz * (i+1)] = output_cat[:, i * 3: (i * 3 + 3), :, :]

        return output



# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=4, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, gpu_ids=[], padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(UnetGenerator, self).__init__()
        self.gpu_ids = gpu_ids

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)

        # Defines the Unet generator.
        # |num_downs|: number of downsamplings in UNet. For example,
        # if |num_downs| == 7, image of size 128x128 will become of size 1x1
        # at the bottleneck






class UnetGeneratorRe1(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(UnetGeneratorRe1, self).__init__()
        self.gpu_ids = gpu_ids

        # norm_layer = PixelNormLayer()

        # construct unet structure
        unet_block = UnetSkipConnectionBlockRe1(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlockRe1(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlockRe1(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlockRe1(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlockRe1(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlockRe1(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)




class UnetSkipConnectionBlockRe1(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d,
                 use_dropout=False):
        super(UnetSkipConnectionBlockRe1, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)

        downnorm = norm_layer(inner_nc)
        # downnorm = norm_layer

        uprelu = nn.ReLU(True)

        upnorm = norm_layer(outer_nc)
        # upnorm = norm_layer

        my_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        pad1 = nn.ReflectionPad2d(1)

        if outermost:
            # upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
            #                             kernel_size=4, stride=2,
            #                             padding=1)
            conv44 = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            conv33 = nn.Conv2d(outer_nc, outer_nc, kernel_size=3, stride=1, padding=0)

            down = [downconv]
            # up = [uprelu, upconv, nn.Tanh()]
            up = [uprelu, conv44, pad1, conv33, nn.Tanh()]
            # up = [uprelu, conv44, upnorm]

            model = down + [submodule] + up
        elif innermost:
            # upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
            #                            kernel_size=4, stride=2,
            #                            padding=1, bias=use_bias)

            conv44 = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            conv33 = nn.Conv2d(outer_nc, outer_nc, kernel_size=3, stride=1, padding=0)

            # upconv = [conv33]
            down = [downrelu, downconv]
            # up = [uprelu, upconv, upnorm]

            up = [uprelu, conv44, pad1, conv33, upnorm]
            # up = [uprelu, conv44, upnorm]

            model = down + up
        else:
            # upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
            #                            kernel_size=4, stride=2,
            #                            padding=1, bias=use_bias)

            conv44 = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            conv33 = nn.Conv2d(outer_nc, outer_nc, kernel_size=3, stride=1, padding=0)

            # upconv = [conv33]
            down = [downrelu, downconv, downnorm]

            # up = [uprelu, conv44, upnorm]
            up = [uprelu, conv44, pad1, conv33, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)




class UnetGeneratorRe2(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(UnetGeneratorRe2, self).__init__()
        self.gpu_ids = gpu_ids

        # construct unet structure
        unet_block = UnetSkipConnectionBlockRe2(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlockRe2(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlockRe2(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlockRe2(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlockRe2(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlockRe2(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)



class UnetSkipConnectionBlockRe2(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlockRe2, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        my_upsample = nn.Upsample(scale_factor = 2, mode='nearest')
        pad1 = nn.ReflectionPad2d(1)


        if outermost:
            # upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
            #                             kernel_size=4, stride=2,
            #                             padding=1)

            conv33 = nn.Conv2d(inner_nc * 2, outer_nc, kernel_size=3, stride=1, padding=1)

            down = [downconv]
            # up = [uprelu, upconv, nn.Tanh()]

            up = [uprelu, my_upsample, conv33, nn.Tanh()]


            model = down + [submodule] + up
        elif innermost:
            #upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
            #                            kernel_size=4, stride=2,
            #                            padding=1, bias=use_bias)

            conv33 = nn.Conv2d(inner_nc, outer_nc, kernel_size=3, stride=1, padding=1)

            # upconv = [conv33]
            down = [downrelu, downconv]
            # up = [uprelu, upconv, upnorm]

            up = [uprelu, my_upsample, conv33, upnorm]

            model = down + up
        else:
            #upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
            #                            kernel_size=4, stride=2,
            #                            padding=1, bias=use_bias)

            conv33 = nn.Conv2d(inner_nc * 2, outer_nc, kernel_size=3, stride=1, padding=1)

            # upconv = [conv33]
            down = [downrelu, downconv, downnorm]
            # up = [uprelu, upconv, upnorm]

            up = [uprelu, my_upsample, conv33, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)



class UnetGenerator1024(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(UnetGenerator1024, self).__init__()
        self.gpu_ids = gpu_ids

        self.encoder_0 = nn.Sequential(
            # input is (input_nc) x 512 x 512
            nn.Conv2d(input_nc, ngf, 4, 2, 1, bias=False),
            # state size. (ndf) x 256 x 256
        )

        self.encoder_1 = nn.Sequential(
            # input is (input_nc) x 256 x 256
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            # state size. (ndf) x 128 x 128
        )

        self.encoder_2 = nn.Sequential(
            # input is (ngf) x 128 x 128
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf * 2, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            # state size. (ngf * 2) x 64 x 64
        )

        self.encoder_3 = nn.Sequential(
            # input is (ngf * 2) x 64 x 64
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf * 4, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            # state size. (ngf * 4) x 32 x 32
        )

        self.encoder_4 = nn.Sequential(
            # input is (ngf * 4) x 32 x 32
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf * 8, ngf * 8 , 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8 ),
            # state size. (ngf * 8 * 2) x 16 x 16
        )

        self.encoder_5 = nn.Sequential(
            # input is (ngf * 8) x 16 x 16
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf * 8, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8 ),
            # state size. (ngf * 8 * 2) x 8 x 8
        )

        self.encoder_6 = nn.Sequential(
            # input is (ngf * 8) x 8 x 8
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf * 8, ngf * 8 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8 * 2),
            # state size. (ngf * 8 * 2) x 4 x 4
        )

        self.encoder_7 = nn.Sequential(
            # input is (ngf * 8) x 4 x 4
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf * 8 * 2, ngf * 8 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8 * 2),
            # state size. (ngf * 8) x 2 x 2
        )

        self.encoder_8 = nn.Sequential(
            # input is (ngf * 8) x 2 x 2
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf * 8 * 2, ngf * 8 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8 * 2),
            # state size. (ngf * 8) x 1 x 1
        )

        self.decoder_8 = nn.Sequential(
            # input is (ngf * 8) x 1 x 1
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8 * 2),
            # state size. (ngf * 8) x 2 x 2
        )

        self.decoder_7 = nn.Sequential(
            # input is (ngf * 8 * 2) x 2 x 2
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8 * 2 * 2, ngf * 8 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8 * 2),
            # state size. (ngf * 8) x 4 x 4
        )

        self.decoder_6 = nn.Sequential(
            # input is (ngf * 8 * 2) x 4 x 4
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8 * 2 * 2, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            # state size. (ngf * 8) x 8 x 8
        )

        self.decoder_5 = nn.Sequential(
            # input is (ngf * 8 * 2) x 8 x 8
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            # state size. (ngf * 4) x 16 x 16
        )

        self.decoder_4 = nn.Sequential(
            # input is (ngf * 8 * 2) x 16 x 16
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            # state size. (ngf * 4) x 32 x 32
        )

        self.decoder_3 = nn.Sequential(
            # input is (ngf * 4 * 2) x 32 x 32
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8 * 2, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            # state size. (ngf * 2) x 64 x 64
        )

        self.decoder_2 = nn.Sequential(
            # input is (ngf * 2 * 2) x 64 x 64
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4 * 2, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            # state size. (ngf) x 128 x 128
        )

        self.decoder_1 = nn.Sequential(
            # input is (ngf * 2) x 128 x 128
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2 * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            # state size. (target_nc) x 256 x 256
        )

        self.decoder_0 = nn.Sequential(
            # input is (ngf * 2) x 256 x 256
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, output_nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (target_nc) x 512 x 512
        )

    def forward(self, input):
        output_e0 = self.encoder_0(input)
        output_e1 = self.encoder_1(output_e0)
        output_e2 = self.encoder_2(output_e1)
        output_e3 = self.encoder_3(output_e2)
        output_e4 = self.encoder_4(output_e3)
        output_e5 = self.encoder_5(output_e4)
        output_e6 = self.encoder_6(output_e5)
        output_e7 = self.encoder_7(output_e6)
        output_e8 = self.encoder_8(output_e7)

        output_d8 = self.decoder_8(output_e8)
        output_d7 = self.decoder_7(torch.cat((output_d8, output_e7), 1))
        output_d6 = self.decoder_6(torch.cat((output_d7, output_e6), 1))
        output_d5 = self.decoder_5(torch.cat((output_d6, output_e5), 1))
        output_d4 = self.decoder_4(torch.cat((output_d5, output_e4), 1))
        output_d3 = self.decoder_3(torch.cat((output_d4, output_e3), 1))
        output_d2 = self.decoder_2(torch.cat((output_d3, output_e2), 1))
        output_d1 = self.decoder_1(torch.cat((output_d2, output_e1), 1))
        output_d0 = self.decoder_0(torch.cat((output_d1, output_e0), 1))

        return output_d0

class UnetGenerator1024_7x7(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(UnetGenerator1024_7x7, self).__init__()
        self.gpu_ids = gpu_ids

        self.encoder_0 = nn.Sequential(
            # input is (input_nc) x 512 x 512
            nn.Conv2d(input_nc, ngf, 7, 2, 3, bias=False),
            # state size. (ndf) x 256 x 256
        )

        self.encoder_1 = nn.Sequential(
            # input is (input_nc) x 256 x 256
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            # state size. (ndf) x 128 x 128
        )

        self.encoder_2 = nn.Sequential(
            # input is (ngf) x 128 x 128
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf * 2, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            # state size. (ngf * 2) x 64 x 64
        )

        self.encoder_3 = nn.Sequential(
            # input is (ngf * 2) x 64 x 64
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf * 4, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            # state size. (ngf * 4) x 32 x 32
        )

        self.encoder_4 = nn.Sequential(
            # input is (ngf * 4) x 32 x 32
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf * 8, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            # state size. (ngf * 8 * 2) x 16 x 16
        )

        self.encoder_5 = nn.Sequential(
            # input is (ngf * 8) x 16 x 16
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf * 8, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            # state size. (ngf * 8 * 2) x 8 x 8
        )

        self.encoder_6 = nn.Sequential(
            # input is (ngf * 8) x 8 x 8
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf * 8, ngf * 8 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8 * 2),
            # state size. (ngf * 8 * 2) x 4 x 4
        )

        self.encoder_7 = nn.Sequential(
            # input is (ngf * 8) x 4 x 4
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf * 8 * 2, ngf * 8 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8 * 2),
            # state size. (ngf * 8) x 2 x 2
        )

        self.encoder_8 = nn.Sequential(
            # input is (ngf * 8) x 2 x 2
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf * 8 * 2, ngf * 8 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8 * 2),
            # state size. (ngf * 8) x 1 x 1
        )

        self.decoder_8 = nn.Sequential(
            # input is (ngf * 8) x 1 x 1
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8 * 2),
            # state size. (ngf * 8) x 2 x 2
        )

        self.decoder_7 = nn.Sequential(
            # input is (ngf * 8 * 2) x 2 x 2
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8 * 2 * 2, ngf * 8 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8 * 2),
            # state size. (ngf * 8) x 4 x 4
        )

        self.decoder_6 = nn.Sequential(
            # input is (ngf * 8 * 2) x 4 x 4
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8 * 2 * 2, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            # state size. (ngf * 8) x 8 x 8
        )

        self.decoder_5 = nn.Sequential(
            # input is (ngf * 8 * 2) x 8 x 8
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            # state size. (ngf * 4) x 16 x 16
        )

        self.decoder_4 = nn.Sequential(
            # input is (ngf * 8 * 2) x 16 x 16
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            # state size. (ngf * 4) x 32 x 32
        )

        self.decoder_3 = nn.Sequential(
            # input is (ngf * 4 * 2) x 32 x 32
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8 * 2, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            # state size. (ngf * 2) x 64 x 64
        )

        self.decoder_2 = nn.Sequential(
            # input is (ngf * 2 * 2) x 64 x 64
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4 * 2, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            # state size. (ngf) x 128 x 128
        )

        self.decoder_1 = nn.Sequential(
            # input is (ngf * 2) x 128 x 128
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2 * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            # state size. (target_nc) x 256 x 256
        )

        self.decoder_0 = nn.Sequential(
            # input is (ngf * 2) x 256 x 256
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, output_nc, 7, 2, 3, bias=False),
            nn.Tanh()
            # state size. (target_nc) x 512 x 512
        )

    def forward(self, input):
        output_e0 = self.encoder_0(input)
        output_e1 = self.encoder_1(output_e0)
        output_e2 = self.encoder_2(output_e1)
        output_e3 = self.encoder_3(output_e2)
        output_e4 = self.encoder_4(output_e3)
        output_e5 = self.encoder_5(output_e4)
        output_e6 = self.encoder_6(output_e5)
        output_e7 = self.encoder_7(output_e6)
        output_e8 = self.encoder_8(output_e7)

        output_d8 = self.decoder_8(output_e8)
        output_d7 = self.decoder_7(torch.cat((output_d8, output_e7), 1))
        output_d6 = self.decoder_6(torch.cat((output_d7, output_e6), 1))
        output_d5 = self.decoder_5(torch.cat((output_d6, output_e5), 1))
        output_d4 = self.decoder_4(torch.cat((output_d5, output_e4), 1))
        output_d3 = self.decoder_3(torch.cat((output_d4, output_e3), 1))
        output_d2 = self.decoder_2(torch.cat((output_d3, output_e2), 1))
        output_d1 = self.decoder_1(torch.cat((output_d2, output_e1), 1))
        output_d0 = self.decoder_0(torch.cat((output_d1, output_e0), 1))

        return output_d0






# # Defines the PatchGAN discriminator with the specified arguments.
# class NLayerDiscriminator(nn.Module):
#     def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[]):
#         super(NLayerDiscriminator, self).__init__()
#         self.gpu_ids = gpu_ids
#         if type(norm_layer) == functools.partial:
#             use_bias = norm_layer.func == nn.InstanceNorm2d
#         else:
#             use_bias = norm_layer == nn.InstanceNorm2d
#
#         kw = 4
#         padw = 1
#         sequence = [
#             nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
#             nn.LeakyReLU(0.2, True)
#         ]
#
#         nf_mult = 1
#         nf_mult_prev = 1
#         for n in range(1, n_layers):
#             nf_mult_prev = nf_mult
#             nf_mult = min(2**n, 8)
#             sequence += [
#                 nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
#                           kernel_size=kw, stride=2, padding=padw, bias=use_bias),
#                 norm_layer(ndf * nf_mult),
#                 nn.LeakyReLU(0.2, True)
#             ]
#
#         nf_mult_prev = nf_mult
#         nf_mult = min(2**n_layers, 8)
#         sequence += [
#             nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
#                       kernel_size=kw, stride=1, padding=padw, bias=use_bias),
#             norm_layer(ndf * nf_mult),
#             nn.LeakyReLU(0.2, True)
#         ]
#
#         sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
#
#         if use_sigmoid:
#             sequence += [nn.Sigmoid()]
#
#         self.model = nn.Sequential(*sequence)
#
#     def forward(self, input):
#         if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
#             return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
#         else:
#             return self.model(input)


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[]):
        super(NLayerDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids

        self.input_nc = input_nc
        output_nc = 1
        self.ndf = ndf
        self.conv1 = nn.Conv2d(input_nc, ndf, 4, 2, 1)
        self.conv2 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1)
        self.conv2_norm = nn.InstanceNorm2d(ndf * 2)
        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1)
        self.conv3_norm = nn.InstanceNorm2d(ndf * 4)
        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, 4, 1, 1)
        self.conv4_norm = nn.InstanceNorm2d(ndf * 8)
        self.conv5 = nn.Conv2d(ndf * 8, output_nc, 4, 1, 1)


        self.patch_conv1 = nn.Conv2d(input_nc, ndf, 4, 2, 1)
        self.patch_conv2 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1)
        self.patch_conv2_norm = nn.InstanceNorm2d(ndf * 2)
        self.patch_conv3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 1, 1)
        self.patch_conv3_norm = nn.InstanceNorm2d(ndf * 4)
        self.patch_conv4 = nn.Conv2d(ndf * 4, output_nc, 4, 1, 1)


    def forward(self, input, patch):
        x = F.leaky_relu(self.conv1(input), 0.2)
        x = F.leaky_relu(self.conv2_norm(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_norm(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_norm(self.conv4(x)), 0.2)
        x = self.conv5(x)


        x_patch = F.leaky_relu(self.patch_conv1(patch), 0.2)
        x_patch = F.leaky_relu(self.patch_conv2_norm(self.patch_conv2(x_patch)), 0.2)
        x_patch = F.leaky_relu(self.patch_conv3_norm(self.patch_conv3(x_patch)), 0.2)
        x_patch = self.patch_conv4(x_patch)

        output = torch.cat((x, x_patch), dim=3)

        return output


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator_128_64(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[]):
        super(NLayerDiscriminator_128_64, self).__init__()
        self.gpu_ids = gpu_ids

        self.input_nc = input_nc
        output_nc = 1
        self.ndf = ndf
        self.conv1 = nn.Conv2d(input_nc, ndf, 4, 2, 1)
        self.conv2 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1)
        self.conv2_norm = nn.InstanceNorm2d(ndf * 2)
        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1)
        self.conv3_norm = nn.InstanceNorm2d(ndf * 4)
        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, 4, 1, 1)
        self.conv4_norm = nn.InstanceNorm2d(ndf * 8)
        self.conv5 = nn.Conv2d(ndf * 8, output_nc, 4, 1, 1)


        self.patch_conv1 = nn.Conv2d(input_nc, ndf, 4, 2, 1)
        self.patch_conv2 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1)
        self.patch_conv2_norm = nn.InstanceNorm2d(ndf * 2)
        self.patch_conv3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 1, 1)
        self.patch_conv3_norm = nn.InstanceNorm2d(ndf * 4)
        self.patch_conv4 = nn.Conv2d(ndf * 4, output_nc, 4, 1, 1)


    def forward(self, input, patch):
        x = F.leaky_relu(self.conv1(input), 0.2)
        x = F.leaky_relu(self.conv2_norm(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_norm(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_norm(self.conv4(x)), 0.2)
        x = self.conv5(x)


        x_patch = F.leaky_relu(self.patch_conv1(patch), 0.2)
        x_patch = F.leaky_relu(self.patch_conv2_norm(self.patch_conv2(x_patch)), 0.2)
        x_patch = F.leaky_relu(self.patch_conv3_norm(self.patch_conv3(x_patch)), 0.2)
        x_patch = self.patch_conv4(x_patch)

        output = torch.cat((x, x_patch), dim=3)
        return output

# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator_128_32(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[]):
        super(NLayerDiscriminator_128_32, self).__init__()
        self.gpu_ids = gpu_ids

        self.input_nc = input_nc
        output_nc = 1
        self.ndf = ndf
        self.conv1 = nn.Conv2d(input_nc, ndf, 4, 2, 1)
        self.conv2 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1)
        self.conv2_norm = nn.InstanceNorm2d(ndf * 2)
        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1)
        self.conv3_norm = nn.InstanceNorm2d(ndf * 4)
        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, 4, 1, 1)
        self.conv4_norm = nn.InstanceNorm2d(ndf * 8)
        self.conv5 = nn.Conv2d(ndf * 8, output_nc, 4, 1, 1)


        self.patch_conv1 = nn.Conv2d(input_nc, ndf, 4, 2, 1)
        self.patch_conv3 = nn.Conv2d(ndf * 1, ndf * 2, 4, 1, 1)
        self.patch_conv3_norm = nn.InstanceNorm2d(ndf * 2)
        self.patch_conv4 = nn.Conv2d(ndf * 2, output_nc, 4, 1, 1)


    def forward(self, input, patch):
        x = F.leaky_relu(self.conv1(input), 0.2)
        x = F.leaky_relu(self.conv2_norm(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_norm(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_norm(self.conv4(x)), 0.2)
        x = self.conv5(x)

        x_patch = F.leaky_relu(self.patch_conv1(patch), 0.2)
        x_patch = F.leaky_relu(self.patch_conv3_norm(self.patch_conv3(x_patch)), 0.2)
        x_patch = self.patch_conv4(x_patch)

        output = torch.cat((x, x_patch), dim=3)
        return output


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator_256_128(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[]):
        super(NLayerDiscriminator_256_128, self).__init__()
        self.gpu_ids = gpu_ids

        self.input_nc = input_nc
        output_nc = 1
        self.ndf = ndf
        self.conv1 = nn.Conv2d(input_nc, ndf, 4, 2, 1)
        self.conv2 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1)
        self.conv2_norm = nn.InstanceNorm2d(ndf * 2)
        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1)
        self.conv3_norm = nn.InstanceNorm2d(ndf * 4)
        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1)
        self.conv4_norm = nn.InstanceNorm2d(ndf * 8)
        self.conv5 = nn.Conv2d(ndf * 8, ndf * 8, 4, 1, 1)
        self.conv5_norm = nn.InstanceNorm2d(ndf * 8)
        self.conv6 = nn.Conv2d(ndf * 8, output_nc, 4, 1, 1)

        self.patch_conv1 = nn.Conv2d(input_nc, ndf, 4, 2, 1)
        self.patch_conv2 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1)
        self.patch_conv2_norm = nn.InstanceNorm2d(ndf * 2)
        self.patch_conv3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1)
        self.patch_conv3_norm = nn.InstanceNorm2d(ndf * 4)
        self.patch_conv4 = nn.Conv2d(ndf * 4, ndf * 8, 4, 1, 1)
        self.patch_conv4_norm = nn.InstanceNorm2d(ndf * 8)
        self.patch_conv5 = nn.Conv2d(ndf * 8, output_nc, 4, 1, 1)

    def forward(self, input, patch):
        x = F.leaky_relu(self.conv1(input), 0.2)
        x = F.leaky_relu(self.conv2_norm(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_norm(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_norm(self.conv4(x)), 0.2)
        x = F.leaky_relu(self.conv5_norm(self.conv5(x)), 0.2)
        x = self.conv6(x)


        x_patch = F.leaky_relu(self.patch_conv1(patch), 0.2)
        x_patch = F.leaky_relu(self.patch_conv2_norm(self.patch_conv2(x_patch)), 0.2)
        x_patch = F.leaky_relu(self.patch_conv3_norm(self.patch_conv3(x_patch)), 0.2)
        x_patch = F.leaky_relu(self.patch_conv4_norm(self.patch_conv4(x_patch)), 0.2)
        x_patch = self.patch_conv5(x_patch)

        output = torch.cat((x, x_patch), dim=3)
        return output



# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator_512_256(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[]):
        super(NLayerDiscriminator_512_256, self).__init__()
        self.gpu_ids = gpu_ids

        self.input_nc = input_nc
        output_nc = 1
        self.ndf = ndf
        self.conv1 = nn.Conv2d(input_nc, ndf, 4, 2, 1)
        self.conv2 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1)
        self.conv2_norm = nn.InstanceNorm2d(ndf * 2)
        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1)
        self.conv3_norm = nn.InstanceNorm2d(ndf * 4)
        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1)
        self.conv4_norm = nn.InstanceNorm2d(ndf * 8)

        self.conv5 = nn.Conv2d(ndf * 8, ndf * 8, 4, 2, 1)
        self.conv5_norm = nn.InstanceNorm2d(ndf * 8)

        self.conv6 = nn.Conv2d(ndf * 8, ndf * 8, 4, 1, 1)
        self.conv6_norm = nn.InstanceNorm2d(ndf * 8)
        self.conv7 = nn.Conv2d(ndf * 8, output_nc, 4, 1, 1)

        self.patch_conv1 = nn.Conv2d(input_nc, ndf, 4, 2, 1)
        self.patch_conv2 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1)
        self.patch_conv2_norm = nn.InstanceNorm2d(ndf * 2)
        self.patch_conv3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1)
        self.patch_conv3_norm = nn.InstanceNorm2d(ndf * 4)

        self.patch_conv4 = nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1)
        self.patch_conv4_norm = nn.InstanceNorm2d(ndf * 8)

        self.patch_conv5 = nn.Conv2d(ndf * 8, ndf * 8, 4, 1, 1)
        self.patch_conv5_norm = nn.InstanceNorm2d(ndf * 8)
        self.patch_conv6 = nn.Conv2d(ndf * 8, output_nc, 4, 1, 1)


    def forward(self, input, patch):
        x = F.leaky_relu(self.conv1(input), 0.2)
        x = F.leaky_relu(self.conv2_norm(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_norm(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_norm(self.conv4(x)), 0.2)
        x = F.leaky_relu(self.conv5_norm(self.conv5(x)), 0.2)
        x = F.leaky_relu(self.conv6_norm(self.conv6(x)), 0.2)
        x = self.conv7(x)

        x_patch = F.leaky_relu(self.patch_conv1(patch), 0.2)
        x_patch = F.leaky_relu(self.patch_conv2_norm(self.patch_conv2(x_patch)), 0.2)
        x_patch = F.leaky_relu(self.patch_conv3_norm(self.patch_conv3(x_patch)), 0.2)
        x_patch = F.leaky_relu(self.patch_conv4_norm(self.patch_conv4(x_patch)), 0.2)
        x_patch = F.leaky_relu(self.patch_conv5_norm(self.patch_conv5(x_patch)), 0.2)
        x_patch = self.patch_conv6(x_patch)

        output = torch.cat((x, x_patch), dim=3)
        return output

# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator_512_128(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[]):
        super(NLayerDiscriminator_512_128, self).__init__()
        self.gpu_ids = gpu_ids

        self.input_nc = input_nc
        output_nc = 1
        self.ndf = ndf
        self.conv1 = nn.Conv2d(input_nc, ndf, 4, 2, 1)
        self.conv2 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1)
        self.conv2_norm = nn.InstanceNorm2d(ndf * 2)
        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1)
        self.conv3_norm = nn.InstanceNorm2d(ndf * 4)
        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1)
        self.conv4_norm = nn.InstanceNorm2d(ndf * 8)

        self.conv5 = nn.Conv2d(ndf * 8, ndf * 8, 4, 2, 1)
        self.conv5_norm = nn.InstanceNorm2d(ndf * 8)

        self.conv6 = nn.Conv2d(ndf * 8, ndf * 8, 4, 1, 1)
        self.conv6_norm = nn.InstanceNorm2d(ndf * 8)
        self.conv7 = nn.Conv2d(ndf * 8, output_nc, 4, 1, 1)

        self.patch_conv1 = nn.Conv2d(input_nc, ndf, 4, 2, 1)
        # self.patch_conv2 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1)
        # self.patch_conv2_norm = nn.InstanceNorm2d(ndf * 2)
        self.patch_conv3 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1)
        self.patch_conv3_norm = nn.InstanceNorm2d(ndf * 2)

        self.patch_conv4 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1)
        self.patch_conv4_norm = nn.InstanceNorm2d(ndf * 4)

        self.patch_conv5 = nn.Conv2d(ndf * 4, ndf * 4, 4, 1, 1)
        self.patch_conv5_norm = nn.InstanceNorm2d(ndf * 4)
        self.patch_conv6 = nn.Conv2d(ndf * 4, output_nc, 4, 1, 1)


    def forward(self, input, patch):
        x = F.leaky_relu(self.conv1(input), 0.2)
        x = F.leaky_relu(self.conv2_norm(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_norm(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_norm(self.conv4(x)), 0.2)
        x = F.leaky_relu(self.conv5_norm(self.conv5(x)), 0.2)
        x = F.leaky_relu(self.conv6_norm(self.conv6(x)), 0.2)
        x = self.conv7(x)

        x_patch = F.leaky_relu(self.patch_conv1(patch), 0.2)
        # x_patch = F.leaky_relu(self.patch_conv2_norm(self.patch_conv2(x_patch)), 0.2)
        x_patch = F.leaky_relu(self.patch_conv3_norm(self.patch_conv3(x_patch)), 0.2)
        x_patch = F.leaky_relu(self.patch_conv4_norm(self.patch_conv4(x_patch)), 0.2)
        x_patch = F.leaky_relu(self.patch_conv5_norm(self.patch_conv5(x_patch)), 0.2)
        x_patch = self.patch_conv6(x_patch)

        output = torch.cat((x, x_patch), dim=3)
        return output

# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminatorPatch(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[]):
        super(NLayerDiscriminatorPatch, self).__init__()
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator_512_128_multi(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[]):
        super(NLayerDiscriminator_512_128_multi, self).__init__()
        self.gpu_ids = gpu_ids

        self.input_nc = input_nc
        output_nc = 1
        self.ndf = ndf

        # self.conv1 = nn.Conv2d(input_nc, ndf, 4, 2, 1)
        # self.conv2 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1)
        # self.conv2_norm = nn.InstanceNorm2d(ndf * 2)
        # self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1)
        # self.conv3_norm = nn.InstanceNorm2d(ndf * 4)
        # self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1)
        # self.conv4_norm = nn.InstanceNorm2d(ndf * 8)
        #
        # self.conv5 = nn.Conv2d(ndf * 8, ndf * 8, 4, 2, 1)
        # self.conv5_norm = nn.InstanceNorm2d(ndf * 8)
        #
        # self.conv6 = nn.Conv2d(ndf * 8, ndf * 8, 4, 1, 1)
        # self.conv6_norm = nn.InstanceNorm2d(ndf * 8)
        # self.conv7 = nn.Conv2d(ndf * 8, output_nc, 4, 1, 1)

        num_D = 3
        self.num_D = 3
        self.n_layers = n_layers

        for i in range(num_D):
            netD = NLayerDiscriminatorPatch(input_nc, ndf, n_layers, norm_layer, use_sigmoid)
            setattr(self, 'layer' + str(i), netD.model)


        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

        self.patch_conv1 = nn.Conv2d(input_nc, ndf, 4, 2, 1)
        self.patch_conv3 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1)
        self.patch_conv3_norm = nn.InstanceNorm2d(ndf * 2)

        self.patch_conv4 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1)
        self.patch_conv4_norm = nn.InstanceNorm2d(ndf * 4)

        self.patch_conv5 = nn.Conv2d(ndf * 4, ndf * 4, 4, 1, 1)
        self.patch_conv5_norm = nn.InstanceNorm2d(ndf * 4)
        self.patch_conv6 = nn.Conv2d(ndf * 4, output_nc, 4, 1, 1)

    def singleD_forward(self, model, input):
        return [model(input)]

    def forward(self, input, patch):
        # x = F.leaky_relu(self.conv1(input), 0.2)
        # x = F.leaky_relu(self.conv2_norm(self.conv2(x)), 0.2)
        # x = F.leaky_relu(self.conv3_norm(self.conv3(x)), 0.2)
        # x = F.leaky_relu(self.conv4_norm(self.conv4(x)), 0.2)
        # x = F.leaky_relu(self.conv5_norm(self.conv5(x)), 0.2)
        # x = F.leaky_relu(self.conv6_norm(self.conv6(x)), 0.2)
        # x = self.conv7(x)

        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):

            model = getattr(self, 'layer' + str(num_D - 1 - i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D - 1):
                input_downsampled = self.downsample(input_downsampled)

        # x_patch = F.leaky_relu(self.patch_conv1(patch), 0.2)
        # x_patch = F.leaky_relu(self.patch_conv3_norm(self.patch_conv3(x_patch)), 0.2)
        # x_patch = F.leaky_relu(self.patch_conv4_norm(self.patch_conv4(x_patch)), 0.2)
        # x_patch = F.leaky_relu(self.patch_conv5_norm(self.patch_conv5(x_patch)), 0.2)
        # x_patch = self.patch_conv6(x_patch)

        model = getattr(self, 'layer' + str(0))
        x_patch = self.singleD_forward(model, patch)

        result.append(x_patch)

        # output = torch.cat((x, x_patch), dim=3)
        return result


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator_512_128_multi_new(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[]):
        super(NLayerDiscriminator_512_128_multi_new, self).__init__()
        self.gpu_ids = gpu_ids

        self.input_nc = input_nc
        output_nc = 1
        self.ndf = ndf

        # self.conv1 = nn.Conv2d(input_nc, ndf, 4, 2, 1)
        # self.conv2 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1)
        # self.conv2_norm = nn.InstanceNorm2d(ndf * 2)
        # self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1)
        # self.conv3_norm = nn.InstanceNorm2d(ndf * 4)
        # self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1)
        # self.conv4_norm = nn.InstanceNorm2d(ndf * 8)
        #
        # self.conv5 = nn.Conv2d(ndf * 8, ndf * 8, 4, 2, 1)
        # self.conv5_norm = nn.InstanceNorm2d(ndf * 8)
        #
        # self.conv6 = nn.Conv2d(ndf * 8, ndf * 8, 4, 1, 1)
        # self.conv6_norm = nn.InstanceNorm2d(ndf * 8)
        # self.conv7 = nn.Conv2d(ndf * 8, output_nc, 4, 1, 1)

        num_D = 3
        self.num_D = 3
        self.n_layers = n_layers

        for i in range(num_D):
            netD = NLayerDiscriminatorPatch(input_nc, ndf, n_layers, norm_layer, use_sigmoid)
            setattr(self, 'layer' + str(i), netD.model)

        for i in range(num_D):
            netD = NLayerDiscriminatorPatch(input_nc, ndf, n_layers, norm_layer, use_sigmoid)
            setattr(self, 'patch_layer' + str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

        self.patch_conv1 = nn.Conv2d(input_nc, ndf, 4, 2, 1)
        self.patch_conv3 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1)
        self.patch_conv3_norm = nn.InstanceNorm2d(ndf * 2)

        self.patch_conv4 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1)
        self.patch_conv4_norm = nn.InstanceNorm2d(ndf * 4)

        self.patch_conv5 = nn.Conv2d(ndf * 4, ndf * 4, 4, 1, 1)
        self.patch_conv5_norm = nn.InstanceNorm2d(ndf * 4)
        self.patch_conv6 = nn.Conv2d(ndf * 4, output_nc, 4, 1, 1)

    def singleD_forward(self, model, input):
        return [model(input)]

    def forward(self, input, patch):
        # x = F.leaky_relu(self.conv1(input), 0.2)
        # x = F.leaky_relu(self.conv2_norm(self.conv2(x)), 0.2)
        # x = F.leaky_relu(self.conv3_norm(self.conv3(x)), 0.2)
        # x = F.leaky_relu(self.conv4_norm(self.conv4(x)), 0.2)
        # x = F.leaky_relu(self.conv5_norm(self.conv5(x)), 0.2)
        # x = F.leaky_relu(self.conv6_norm(self.conv6(x)), 0.2)
        # x = self.conv7(x)

        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):

            model = getattr(self, 'layer' + str(num_D - 1 - i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D - 1):
                input_downsampled = self.downsample(input_downsampled)

        # x_patch = F.leaky_relu(self.patch_conv1(patch), 0.2)
        # x_patch = F.leaky_relu(self.patch_conv3_norm(self.patch_conv3(x_patch)), 0.2)
        # x_patch = F.leaky_relu(self.patch_conv4_norm(self.patch_conv4(x_patch)), 0.2)
        # x_patch = F.leaky_relu(self.patch_conv5_norm(self.patch_conv5(x_patch)), 0.2)
        # x_patch = self.patch_conv6(x_patch)

        model = getattr(self, 'patch_layer' + str(0))
        x_patch = self.singleD_forward(model, patch)

        result.append(x_patch)

        # output = torch.cat((x, x_patch), dim=3)
        return result

# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator_512_256_multi(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[], num_D = 3):
        super(NLayerDiscriminator_512_256_multi, self).__init__()
        self.gpu_ids = gpu_ids

        self.input_nc = input_nc
        output_nc = 1
        self.ndf = ndf

        # self.conv1 = nn.Conv2d(input_nc, ndf, 4, 2, 1)
        # self.conv2 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1)
        # self.conv2_norm = nn.InstanceNorm2d(ndf * 2)
        # self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1)
        # self.conv3_norm = nn.InstanceNorm2d(ndf * 4)
        # self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1)
        # self.conv4_norm = nn.InstanceNorm2d(ndf * 8)
        #
        # self.conv5 = nn.Conv2d(ndf * 8, ndf * 8, 4, 2, 1)
        # self.conv5_norm = nn.InstanceNorm2d(ndf * 8)
        #
        # self.conv6 = nn.Conv2d(ndf * 8, ndf * 8, 4, 1, 1)
        # self.conv6_norm = nn.InstanceNorm2d(ndf * 8)
        # self.conv7 = nn.Conv2d(ndf * 8, output_nc, 4, 1, 1)

        self.num_D = num_D
        self.n_layers = n_layers

        for i in range(num_D):
            netD = NLayerDiscriminatorPatch(input_nc, ndf, n_layers, norm_layer, use_sigmoid)
            setattr(self, 'layer' + str(i), netD.model)

       # for i in range(num_D):
       #     netD = NLayerDiscriminatorPatch(input_nc, ndf, n_layers, norm_layer, use_sigmoid)
       #     setattr(self, 'patch_layer' + str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

        self.patch_conv1 = nn.Conv2d(input_nc, ndf, 4, 2, 1)
        self.patch_conv3 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1)
        self.patch_conv3_norm = nn.InstanceNorm2d(ndf * 2)

        self.patch_conv4 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1)
        self.patch_conv4_norm = nn.InstanceNorm2d(ndf * 4)

        self.patch_conv5 = nn.Conv2d(ndf * 4, ndf * 4, 4, 1, 1)
        self.patch_conv5_norm = nn.InstanceNorm2d(ndf * 4)
        self.patch_conv6 = nn.Conv2d(ndf * 4, output_nc, 4, 1, 1)

    def singleD_forward(self, model, input):
        return [model(input)]

    def forward(self, input, patch):
        # x = F.leaky_relu(self.conv1(input), 0.2)
        # x = F.leaky_relu(self.conv2_norm(self.conv2(x)), 0.2)
        # x = F.leaky_relu(self.conv3_norm(self.conv3(x)), 0.2)
        # x = F.leaky_relu(self.conv4_norm(self.conv4(x)), 0.2)
        # x = F.leaky_relu(self.conv5_norm(self.conv5(x)), 0.2)
        # x = F.leaky_relu(self.conv6_norm(self.conv6(x)), 0.2)
        # x = self.conv7(x)

        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):

            model = getattr(self, 'layer' + str(num_D - 1 - i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D - 1):
                input_downsampled = self.downsample(input_downsampled)

        # x_patch = F.leaky_relu(self.patch_conv1(patch), 0.2)
        # x_patch = F.leaky_relu(self.patch_conv3_norm(self.patch_conv3(x_patch)), 0.2)
        # x_patch = F.leaky_relu(self.patch_conv4_norm(self.patch_conv4(x_patch)), 0.2)
        # x_patch = F.leaky_relu(self.patch_conv5_norm(self.patch_conv5(x_patch)), 0.2)
        # x_patch = self.patch_conv6(x_patch)

        model = getattr(self, 'layer' + str(1))
        x_patch = self.singleD_forward(model, patch)

        result.append(x_patch)

        #model = getattr(self, 'layer' + str(0))
        #x_patch = self.singleD_forward(model, patch)
        #
        #result.append(x_patch)

        # output = torch.cat((x, x_patch), dim=3)
        return result


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator_512_256_multi_new(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[], num_D = 3):
        super(NLayerDiscriminator_512_256_multi_new, self).__init__()
        self.gpu_ids = gpu_ids

        self.input_nc = input_nc
        output_nc = 1
        self.ndf = ndf

        # self.conv1 = nn.Conv2d(input_nc, ndf, 4, 2, 1)
        # self.conv2 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1)
        # self.conv2_norm = nn.InstanceNorm2d(ndf * 2)
        # self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1)
        # self.conv3_norm = nn.InstanceNorm2d(ndf * 4)
        # self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1)
        # self.conv4_norm = nn.InstanceNorm2d(ndf * 8)
        #
        # self.conv5 = nn.Conv2d(ndf * 8, ndf * 8, 4, 2, 1)
        # self.conv5_norm = nn.InstanceNorm2d(ndf * 8)
        #
        # self.conv6 = nn.Conv2d(ndf * 8, ndf * 8, 4, 1, 1)
        # self.conv6_norm = nn.InstanceNorm2d(ndf * 8)
        # self.conv7 = nn.Conv2d(ndf * 8, output_nc, 4, 1, 1)

        self.num_D = num_D
        self.n_layers = n_layers

        for i in range(num_D):
            netD = NLayerDiscriminatorPatch(input_nc, ndf, n_layers, norm_layer, use_sigmoid)
            setattr(self, 'layer' + str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)


        for i in range(num_D):
            netD = NLayerDiscriminatorPatch(input_nc, ndf, n_layers, norm_layer, use_sigmoid)
            setattr(self, 'patch_layer' + str(i), netD.model)

        self.downsample_patch = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

        # self.patch_conv1 = nn.Conv2d(input_nc, ndf, 4, 2, 1)
        # self.patch_conv3 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1)
        # self.patch_conv3_norm = nn.InstanceNorm2d(ndf * 2)
        #
        # self.patch_conv4 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1)
        # self.patch_conv4_norm = nn.InstanceNorm2d(ndf * 4)
        #
        # self.patch_conv5 = nn.Conv2d(ndf * 4, ndf * 4, 4, 1, 1)
        # self.patch_conv5_norm = nn.InstanceNorm2d(ndf * 4)
        # self.patch_conv6 = nn.Conv2d(ndf * 4, output_nc, 4, 1, 1)

    def singleD_forward(self, model, input):
        return [model(input)]

    def forward(self, input, patch):
        # x = F.leaky_relu(self.conv1(input), 0.2)
        # x = F.leaky_relu(self.conv2_norm(self.conv2(x)), 0.2)
        # x = F.leaky_relu(self.conv3_norm(self.conv3(x)), 0.2)
        # x = F.leaky_relu(self.conv4_norm(self.conv4(x)), 0.2)
        # x = F.leaky_relu(self.conv5_norm(self.conv5(x)), 0.2)
        # x = F.leaky_relu(self.conv6_norm(self.conv6(x)), 0.2)
        # x = self.conv7(x)

        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):

            model = getattr(self, 'layer' + str(num_D - 1 - i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D - 1):
                input_downsampled = self.downsample(input_downsampled)

        # x_patch = F.leaky_relu(self.patch_conv1(patch), 0.2)
        # x_patch = F.leaky_relu(self.patch_conv3_norm(self.patch_conv3(x_patch)), 0.2)
        # x_patch = F.leaky_relu(self.patch_conv4_norm(self.patch_conv4(x_patch)), 0.2)
        # x_patch = F.leaky_relu(self.patch_conv5_norm(self.patch_conv5(x_patch)), 0.2)
        # x_patch = self.patch_conv6(x_patch)

        model1 = getattr(self, 'patch_layer' + str(1))
        x_patch1 = self.singleD_forward(model1, patch)

        result.append(x_patch1)

        patch_downsampled = self.downsample(patch)
        model2 = getattr(self, 'patch_layer' + str(0))
        x_patch2 = self.singleD_forward(model2, patch_downsampled)

        result.append(x_patch2)

        # output = torch.cat((x, x_patch), dim=3)
        return result



class Unet_G_XRNN_512(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(Unet_G_XRNN_512, self).__init__()
        self.gpu_ids = gpu_ids

        # currently support only input_nc == output_nc
        assert(input_nc == output_nc)

        # self.rnn = nn.RNN()

        self.conv1 = nn.Conv2d(input_nc, ngf, 4, 2, 1)

        self.conv2_1 = nn.Conv2d(ngf * 1, ngf * 1, 3, 1, 1)
        self.conv2_2 = nn.Conv2d(ngf * 1  *  3, ngf * 1, 3, 1, 1)
        self.conv2_3 = nn.Conv2d(ngf * 1, ngf * 1, 3, 1, 1)
        self.conv2 = nn.Conv2d(ngf, ngf * 2, 4, 2, 1)

        self.conv3_1 = nn.Conv2d(ngf * 2, ngf * 2, 3, 1, 1)
        self.conv3_2 = nn.Conv2d(ngf * 2  *  3, ngf * 2, 3, 1, 1)
        self.conv3_3 = nn.Conv2d(ngf * 2, ngf * 2, 3, 1, 1)
        self.conv3 = nn.Conv2d(ngf * 2, ngf * 4, 4, 2, 1)

        self.conv4_1 = nn.Conv2d(ngf * 4, ngf * 4, 3, 1, 1)
        self.conv4_2 = nn.Conv2d(ngf * 4  *  3, ngf * 4, 3, 1, 1)
        self.conv4_3 = nn.Conv2d(ngf * 4, ngf * 4, 3, 1, 1)
        self.conv4 = nn.Conv2d(ngf * 4, ngf * 8, 4, 2, 1)

        self.conv5_1 = nn.Conv2d(ngf * 8, ngf * 8, 3, 1, 1)
        self.conv5_2 = nn.Conv2d(ngf * 8  *  3, ngf * 8, 3, 1, 1)
        self.conv5_3 = nn.Conv2d(ngf * 8, ngf * 8, 3, 1, 1)
        self.conv5 = nn.Conv2d(ngf * 8, ngf * 8, 4, 2, 1)

        self.conv6_1 = nn.Conv2d(ngf * 8, ngf * 8, 3, 1, 1)
        self.conv6_2 = nn.Conv2d(ngf * 8  *  3, ngf * 8, 3, 1, 1)
        self.conv6_3 = nn.Conv2d(ngf * 8, ngf * 8, 3, 1, 1)
        self.conv6 = nn.Conv2d(ngf * 8, ngf * 8, 4, 2, 1)

        self.conv7_1 = nn.Conv2d(ngf * 8, ngf * 8, 3, 1, 1)
        self.conv7_2 = nn.Conv2d(ngf * 8  *  3, ngf * 8, 3, 1, 1)
        self.conv7_3 = nn.Conv2d(ngf * 8, ngf * 8, 3, 1, 1)
        self.conv7 = nn.Conv2d(ngf * 8, ngf * 8, 4, 2, 1)

        self.conv8_1 = nn.Conv2d(ngf * 8, ngf * 8, 3, 1, 1)
        self.conv8_2 = nn.Conv2d(ngf * 8  *  3, ngf * 8, 3, 1, 1)
        self.conv8_3 = nn.Conv2d(ngf * 8, ngf * 8, 3, 1, 1)
        self.conv8 = nn.Conv2d(ngf * 8, ngf * 8, 4, 2, 1)

        self.conv9_1 = nn.Conv2d(ngf * 8, ngf * 8, 3, 1, 1)
        self.conv9_2 = nn.Conv2d(ngf * 8 * 3, ngf * 8, 3, 1, 1)
        self.conv9_3 = nn.Conv2d(ngf * 8, ngf * 8, 3, 1, 1)
        self.conv9 = nn.Conv2d(ngf * 8, ngf * 8, 4, 2, 1)

        self.dconv0 = nn.ConvTranspose2d(ngf * 8, ngf * 8, 4, 2, 1)
        self.dconv1 = nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, 4, 2, 1)
        self.dconv2 = nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, 4, 2, 1)
        self.dconv3 = nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, 4, 2, 1)
        self.dconv4 = nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, 4, 2, 1)
        self.dconv5 = nn.ConvTranspose2d(ngf * 8 * 2, ngf * 4, 4, 2, 1)
        self.dconv6 = nn.ConvTranspose2d(ngf * 4 * 2, ngf * 2, 4, 2, 1)
        self.dconv7 = nn.ConvTranspose2d(ngf * 2 * 2, ngf, 4, 2, 1)
        self.dconv8 = nn.ConvTranspose2d(ngf * 2, output_nc, 4, 2, 1)


        self.batch_norm = nn.BatchNorm2d(ngf)
        self.batch_norm2 = nn.BatchNorm2d(ngf * 2)
        self.batch_norm4 = nn.BatchNorm2d(ngf * 4)
        self.batch_norm8 = nn.BatchNorm2d(ngf * 8)

        self.leaky_relu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)

        self.dropout = nn.Dropout(0.5)

        self.tanh = nn.Tanh()

    def forward(self, input):

        sz = 512
        i = 0
        input_A = input[:, :, :, sz*i: sz*(i+1)]
        i = 1
        input_B = input[:, :, :, sz * i: sz * (i + 1)]
        i = 2
        input_C = input[:, :, :, sz * i: sz * (i + 1)]

        # output = torch.cat((input_A, input_B, input_C), 3)
        # return  output

        e1_A = self.conv1(input_A)
        e1_B = self.conv1(input_B)
        e1_C = self.conv1(input_C)

        # e2_1_A = self.conv2_1(self.leaky_relu(e1_A))
        # e2_1_B = self.conv2_1(self.leaky_relu(e1_B))
        # e2_1_C = self.conv2_1(self.leaky_relu(e1_C))
        #
        # e2_cat_A = torch.cat((e2_1_A, e2_1_B, e2_1_C), 1)
        # e2_cat_B = torch.cat((e2_1_A, e2_1_B, e2_1_C), 1)
        # e2_cat_C = torch.cat((e2_1_A, e2_1_B, e2_1_C), 1)
        #
        # e2_2_A = self.conv2_2(self.leaky_relu(e2_cat_A))
        # e2_2_B = self.conv2_2(self.leaky_relu(e2_cat_B))
        # e2_2_C = self.conv2_2(self.leaky_relu(e2_cat_C))
        #
        # e2_3_A = self.conv2_3(self.leaky_relu(e2_2_A))
        # e2_3_B = self.conv2_3(self.leaky_relu(e2_2_B))
        # e2_3_C = self.conv2_3(self.leaky_relu(e2_2_C))
        #
        # e2_A = self.batch_norm2(self.conv2(self.leaky_relu(e2_3_A)))
        # e2_B = self.batch_norm2(self.conv2(self.leaky_relu(e2_3_B)))
        # e2_C = self.batch_norm2(self.conv2(self.leaky_relu(e2_3_C)))


        e2_A = self.batch_norm2(self.conv2(self.leaky_relu(e1_A)))
        e2_B = self.batch_norm2(self.conv2(self.leaky_relu(e1_B)))
        e2_C = self.batch_norm2(self.conv2(self.leaky_relu(e1_C)))


        e3_1_A = self.conv3_1(self.leaky_relu(e2_A))
        e3_1_B = self.conv3_1(self.leaky_relu(e2_B))
        e3_1_C = self.conv3_1(self.leaky_relu(e2_C))

        e3_cat_A = torch.cat((e3_1_A, e3_1_B, e3_1_C), 1)
        e3_cat_B = torch.cat((e3_1_A, e3_1_B, e3_1_C), 1)
        e3_cat_C = torch.cat((e3_1_A, e3_1_B, e3_1_C), 1)

        e3_2_A = self.conv3_2(self.leaky_relu(e3_cat_A))
        e3_2_B = self.conv3_2(self.leaky_relu(e3_cat_B))
        e3_2_C = self.conv3_2(self.leaky_relu(e3_cat_C))

        e3_3_A = self.conv3_3(self.leaky_relu(e3_2_A))
        e3_3_B = self.conv3_3(self.leaky_relu(e3_2_B))
        e3_3_C = self.conv3_3(self.leaky_relu(e3_2_C))

        e3_A = self.batch_norm4(self.conv3(self.leaky_relu(e3_3_A)))
        e3_B = self.batch_norm4(self.conv3(self.leaky_relu(e3_3_B)))
        e3_C = self.batch_norm4(self.conv3(self.leaky_relu(e3_3_C)))

        e4_1_A = self.conv4_1(self.leaky_relu(e3_A))
        e4_1_B = self.conv4_1(self.leaky_relu(e3_B))
        e4_1_C = self.conv4_1(self.leaky_relu(e3_C))

        e4_cat_A = torch.cat((e4_1_A, e4_1_B, e4_1_C), 1)
        e4_cat_B = torch.cat((e4_1_A, e4_1_B, e4_1_C), 1)
        e4_cat_C = torch.cat((e4_1_A, e4_1_B, e4_1_C), 1)

        e4_2_A = self.conv4_2(self.leaky_relu(e4_cat_A))
        e4_2_B = self.conv4_2(self.leaky_relu(e4_cat_B))
        e4_2_C = self.conv4_2(self.leaky_relu(e4_cat_C))

        e4_3_A = self.conv4_3(self.leaky_relu(e4_2_A))
        e4_3_B = self.conv4_3(self.leaky_relu(e4_2_B))
        e4_3_C = self.conv4_3(self.leaky_relu(e4_2_C))

        e4_A = self.batch_norm8(self.conv4(self.leaky_relu(e4_3_A)))
        e4_B = self.batch_norm8(self.conv4(self.leaky_relu(e4_3_B)))
        e4_C = self.batch_norm8(self.conv4(self.leaky_relu(e4_3_C)))

        e5_1_A = self.conv5_1(self.leaky_relu(e4_A))
        e5_1_B = self.conv5_1(self.leaky_relu(e4_B))
        e5_1_C = self.conv5_1(self.leaky_relu(e4_C))

        e5_cat_A = torch.cat((e5_1_A, e5_1_B, e5_1_C), 1)
        e5_cat_B = torch.cat((e5_1_A, e5_1_B, e5_1_C), 1)
        e5_cat_C = torch.cat((e5_1_A, e5_1_B, e5_1_C), 1)

        e5_2_A = self.conv5_2(self.leaky_relu(e5_cat_A))
        e5_2_B = self.conv5_2(self.leaky_relu(e5_cat_B))
        e5_2_C = self.conv5_2(self.leaky_relu(e5_cat_C))

        e5_3_A = self.conv5_3(self.leaky_relu(e5_2_A))
        e5_3_B = self.conv5_3(self.leaky_relu(e5_2_B))
        e5_3_C = self.conv5_3(self.leaky_relu(e5_2_C))

        e5_A = self.batch_norm8(self.conv5(self.leaky_relu(e5_3_A)))
        e5_B = self.batch_norm8(self.conv5(self.leaky_relu(e5_3_B)))
        e5_C = self.batch_norm8(self.conv5(self.leaky_relu(e5_3_C)))

        e6_1_A = self.conv6_1(self.leaky_relu(e5_A))
        e6_1_B = self.conv6_1(self.leaky_relu(e5_B))
        e6_1_C = self.conv6_1(self.leaky_relu(e5_C))

        e6_cat_A = torch.cat((e6_1_A, e6_1_B, e6_1_C), 1)
        e6_cat_B = torch.cat((e6_1_A, e6_1_B, e6_1_C), 1)
        e6_cat_C = torch.cat((e6_1_A, e6_1_B, e6_1_C), 1)

        e6_2_A = self.conv6_2(self.leaky_relu(e6_cat_A))
        e6_2_B = self.conv6_2(self.leaky_relu(e6_cat_B))
        e6_2_C = self.conv6_2(self.leaky_relu(e6_cat_C))

        e6_3_A = self.conv6_3(self.leaky_relu(e6_2_A))
        e6_3_B = self.conv6_3(self.leaky_relu(e6_2_B))
        e6_3_C = self.conv6_3(self.leaky_relu(e6_2_C))

        e6_A = self.batch_norm8(self.conv6(self.leaky_relu(e6_3_A)))
        e6_B = self.batch_norm8(self.conv6(self.leaky_relu(e6_3_B)))
        e6_C = self.batch_norm8(self.conv6(self.leaky_relu(e6_3_C)))

        e7_1_A = self.conv7_1(self.leaky_relu(e6_A))
        e7_1_B = self.conv7_1(self.leaky_relu(e6_B))
        e7_1_C = self.conv7_1(self.leaky_relu(e6_C))

        e7_cat_A = torch.cat((e7_1_A, e7_1_B, e7_1_C), 1)
        e7_cat_B = torch.cat((e7_1_A, e7_1_B, e7_1_C), 1)
        e7_cat_C = torch.cat((e7_1_A, e7_1_B, e7_1_C), 1)

        e7_2_A = self.conv7_2(self.leaky_relu(e7_cat_A))
        e7_2_B = self.conv7_2(self.leaky_relu(e7_cat_B))
        e7_2_C = self.conv7_2(self.leaky_relu(e7_cat_C))

        e7_3_A = self.conv7_3(self.leaky_relu(e7_2_A))
        e7_3_B = self.conv7_3(self.leaky_relu(e7_2_B))
        e7_3_C = self.conv7_3(self.leaky_relu(e7_2_C))

        e7_A = self.batch_norm8(self.conv7(self.leaky_relu(e7_3_A)))
        e7_B = self.batch_norm8(self.conv7(self.leaky_relu(e7_3_B)))
        e7_C = self.batch_norm8(self.conv7(self.leaky_relu(e7_3_C)))


        e7_1_A = self.conv7_1(self.leaky_relu(e6_A))
        e7_1_B = self.conv7_1(self.leaky_relu(e6_B))
        e7_1_C = self.conv7_1(self.leaky_relu(e6_C))

        e7_cat_A = torch.cat((e7_1_A, e7_1_B, e7_1_C), 1)
        e7_cat_B = torch.cat((e7_1_A, e7_1_B, e7_1_C), 1)
        e7_cat_C = torch.cat((e7_1_A, e7_1_B, e7_1_C), 1)

        e7_2_A = self.conv7_2(self.leaky_relu(e7_cat_A))
        e7_2_B = self.conv7_2(self.leaky_relu(e7_cat_B))
        e7_2_C = self.conv7_2(self.leaky_relu(e7_cat_C))

        e7_3_A = self.conv7_3(self.leaky_relu(e7_2_A))
        e7_3_B = self.conv7_3(self.leaky_relu(e7_2_B))
        e7_3_C = self.conv7_3(self.leaky_relu(e7_2_C))

        e7_A = self.batch_norm8(self.conv7(self.leaky_relu(e7_3_A)))
        e7_B = self.batch_norm8(self.conv7(self.leaky_relu(e7_3_B)))
        e7_C = self.batch_norm8(self.conv7(self.leaky_relu(e7_3_C)))

        e8_1_A = self.conv8_1(self.leaky_relu(e7_A))
        e8_1_B = self.conv8_1(self.leaky_relu(e7_B))
        e8_1_C = self.conv8_1(self.leaky_relu(e7_C))

        e8_cat_A = torch.cat((e8_1_A, e8_1_B, e8_1_C), 1)
        e8_cat_B = torch.cat((e8_1_A, e8_1_B, e8_1_C), 1)
        e8_cat_C = torch.cat((e8_1_A, e8_1_B, e8_1_C), 1)

        e8_2_A = self.conv8_2(self.leaky_relu(e8_cat_A))
        e8_2_B = self.conv8_2(self.leaky_relu(e8_cat_B))
        e8_2_C = self.conv8_2(self.leaky_relu(e8_cat_C))

        e8_3_A = self.conv8_3(self.leaky_relu(e8_2_A))
        e8_3_B = self.conv8_3(self.leaky_relu(e8_2_B))
        e8_3_C = self.conv8_3(self.leaky_relu(e8_2_C))

        e8_A = self.batch_norm8(self.conv8(self.leaky_relu(e8_3_A)))
        e8_B = self.batch_norm8(self.conv8(self.leaky_relu(e8_3_B)))
        e8_C = self.batch_norm8(self.conv8(self.leaky_relu(e8_3_C)))


        e9_1_A = self.conv9_1(self.leaky_relu(e8_A))
        e9_1_B = self.conv9_1(self.leaky_relu(e8_B))
        e9_1_C = self.conv9_1(self.leaky_relu(e8_C))

        e9_cat_A = torch.cat((e9_1_A, e9_1_B, e9_1_C), 1)
        e9_cat_B = torch.cat((e9_1_A, e9_1_B, e9_1_C), 1)
        e9_cat_C = torch.cat((e9_1_A, e9_1_B, e9_1_C), 1)

        e9_2_A = self.conv9_2(self.leaky_relu(e9_cat_A))
        e9_2_B = self.conv9_2(self.leaky_relu(e9_cat_B))
        e9_2_C = self.conv9_2(self.leaky_relu(e9_cat_C))

        e9_3_A = self.conv9_3(self.leaky_relu(e9_2_A))
        e9_3_B = self.conv9_3(self.leaky_relu(e9_2_B))
        e9_3_C = self.conv9_3(self.leaky_relu(e9_2_C))

        e9_A = self.conv9(self.leaky_relu(e9_3_A))
        e9_B = self.conv9(self.leaky_relu(e9_3_B))
        e9_C = self.conv9(self.leaky_relu(e9_3_C))


        d0_ = self.dropout(self.batch_norm8(self.dconv0(self.relu(e9_A))))
        d0_A = torch.cat((d0_, e8_A), 1)
        d1_ = self.dropout(self.batch_norm8(self.dconv1(self.relu(d0_A))))
        d1_A = torch.cat((d1_, e7_A), 1)
        d2_ = self.dropout(self.batch_norm8(self.dconv2(self.relu(d1_A))))
        d2_A = torch.cat((d2_, e6_A), 1)
        d3_ = self.dropout(self.batch_norm8(self.dconv3(self.relu(d2_A))))
        d3_A = torch.cat((d3_, e5_A), 1)
        d4_ = self.batch_norm8(self.dconv4(self.relu(d3_A)))
        d4_A = torch.cat((d4_, e4_A), 1)
        d5_ = self.batch_norm4(self.dconv5(self.relu(d4_A)))
        d5_A = torch.cat((d5_, e3_A), 1)
        d6_ = self.batch_norm2(self.dconv6(self.relu(d5_A)))
        d6_A = torch.cat((d6_, e2_A), 1)
        d7_ = self.batch_norm(self.dconv7(self.relu(d6_A)))
        d7_A = torch.cat((d7_, e1_A), 1)
        d8_A = self.dconv8(self.relu(d7_A))
        output_A = self.tanh(d8_A)

        d0_ = self.dropout(self.batch_norm8(self.dconv0(self.relu(e9_B))))
        d0_B = torch.cat((d0_, e8_B), 1)
        d1_ = self.dropout(self.batch_norm8(self.dconv1(self.relu(d0_B))))
        d1_B = torch.cat((d1_, e7_B), 1)
        d2_ = self.dropout(self.batch_norm8(self.dconv2(self.relu(d1_B))))
        d2_B = torch.cat((d2_, e6_B), 1)
        d3_ = self.dropout(self.batch_norm8(self.dconv3(self.relu(d2_B))))
        d3_B = torch.cat((d3_, e5_B), 1)
        d4_ = self.batch_norm8(self.dconv4(self.relu(d3_B)))
        d4_B = torch.cat((d4_, e4_B), 1)
        d5_ = self.batch_norm4(self.dconv5(self.relu(d4_B)))
        d5_B = torch.cat((d5_, e3_B), 1)
        d6_ = self.batch_norm2(self.dconv6(self.relu(d5_B)))
        d6_B = torch.cat((d6_, e2_B), 1)
        d7_ = self.batch_norm(self.dconv7(self.relu(d6_B)))
        d7_B = torch.cat((d7_, e1_B), 1)
        d8_B = self.dconv8(self.relu(d7_B))
        output_B = self.tanh(d8_B)

        d0_ = self.dropout(self.batch_norm8(self.dconv0(self.relu(e9_C))))
        d0_C = torch.cat((d0_, e8_C), 1)
        d1_ = self.dropout(self.batch_norm8(self.dconv1(self.relu(d0_C))))
        d1_C = torch.cat((d1_, e7_C), 1)
        d2_ = self.dropout(self.batch_norm8(self.dconv2(self.relu(d1_C))))
        d2_C = torch.cat((d2_, e6_C), 1)
        d3_ = self.dropout(self.batch_norm8(self.dconv3(self.relu(d2_C))))
        d3_C = torch.cat((d3_, e5_C), 1)
        d4_ = self.batch_norm8(self.dconv4(self.relu(d3_C)))
        d4_C = torch.cat((d4_, e4_C), 1)
        d5_ = self.batch_norm4(self.dconv5(self.relu(d4_C)))
        d5_C = torch.cat((d5_, e3_C), 1)
        d6_ = self.batch_norm2(self.dconv6(self.relu(d5_C)))
        d6_C = torch.cat((d6_, e2_C), 1)
        d7_ = self.batch_norm(self.dconv7(self.relu(d6_C)))
        d7_C = torch.cat((d7_, e1_C), 1)
        d8_C = self.dconv8(self.relu(d7_C))
        output_C = self.tanh(d8_C)

        output = torch.cat((output_A, output_B, output_C), 3)

        return output


from torchvision import models
class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class VGGLoss(nn.Module):
    def __init__(self, gpu_ids):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss

class GDLoss(nn.Module):
    def __init__(self, gpu_ids):
        super(GDLoss, self).__init__()

        self.a = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        self.conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1.weight = nn.Parameter(torch.from_numpy(self.a).float().unsqueeze(0).unsqueeze(0))

        self.b = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        self.conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2.weight = nn.Parameter(torch.from_numpy(self.b).float().unsqueeze(0).unsqueeze(0))

        self.conv1.cuda(gpu_ids[0])

        self.conv2.cuda(gpu_ids[0])

        self.criterion = nn.L1Loss()

    def forward(self, input1_, input2_):

        input1 = Variable(input1_.data.clone(), requires_grad=False)
        input2 = Variable(input2_.data.clone(), requires_grad=False)

        loss = 0

        P = transforms.Compose([transforms.ToPILImage()])

        for ci in range(3):
            # input = input1[:,ci:ci+1,:,:]
            G_x1 = self.conv1(input1[:,ci:ci+1,:,:]).data
            G_y1 = self.conv2(input1[:,ci:ci+1,:,:]).data

            G1 = torch.sqrt(torch.pow(G_x1, 2) + torch.pow(G_y1, 2))


            G_x2 = self.conv1(input2[:,ci:ci+1,:,:]).data
            G_y2 = self.conv2(input2[:,ci:ci+1,:,:]).data

            G2 = torch.sqrt(torch.pow(G_x2, 2) + torch.pow(G_y2, 2))

            loss += self.criterion(Variable(G1, requires_grad=False), Variable(G2, requires_grad=False))

            # loss += torch.mean(torch.abs(G1 - G2))

            # X = P(G1.squeeze(1).cpu())
            # X.save("G1.png")
            #
            # X = P(G2.squeeze(1).cpu())
            # X.save("G2.png")



        return loss / 3.0


