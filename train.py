from logger import Logger
import time
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
import os
import shutil
from torch.autograd import Variable
import torch
from torchvision.utils import save_image
from util import util
from PIL import Image

opt = TrainOptions().parse()

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()


dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

model = create_model(opt)
visualizer = Visualizer(opt)
total_steps = 0

if not opt.continue_train and os.path.exists(opt.logroot):
    shutil.rmtree(opt.logroot)
    os.mkdir(opt.logroot)

if not os.path.exists(opt.logroot):
    os.mkdir(opt.logroot)

logger = Logger(opt.logroot)

freq = 50
opt.display_freq = freq
opt.print_freq = freq
opt.save_sample_freq = freq * 4

# opt.save_epoch_freq = 2

#opt.save_latest_freq = 2

# freq = 2
# opt.display_freq = freq
# opt.print_freq = freq
# opt.save_sample_freq = freq * 1


# show_row = 5
# show_col = 2
#
# fixed_x_set = []
# fixed_x_boxes = []
#
# for k in range(show_col):
#     fixed_x = []
#     boxes = []
#     for i, data in enumerate(dataset):
#         fixed_x.append(data['A'])
#         boxes.append(data['A_boxes'])
#         if i == show_row:
#             break
#
#     fixed_x = torch.cat(fixed_x, dim=0)
#     fixed_x = Variable(fixed_x, volatile=True).cuda()
#
#     fixed_x_set.append(fixed_x)
#     fixed_x_boxes.append(boxes)
#
#
# def denorm(x):
#     out = (x + 1) / 2
#     return out.clamp_(0, 1)
#
# def get_crop_img(img_tensor, boxes_set):
#
#     img_crop_list = []
#
#     for i in range(show_row + 1):
#         boxes = boxes_set[i]
#         crop0 = img_tensor[i:i+1, :, int(boxes[0][1].numpy()): int(boxes[0][3].numpy()), int(boxes[0][0].numpy()): int(boxes[0][2].numpy())]
#         crop1 = img_tensor[i:i+1, :, int(boxes[1][1].numpy()): int(boxes[1][3].numpy()), int(boxes[1][0].numpy()): int(boxes[1][2].numpy())]
#         crop2 = img_tensor[i:i+1, :, int(boxes[2][1].numpy()): int(boxes[2][3].numpy()), int(boxes[2][0].numpy()): int(boxes[2][2].numpy())]
#
#         img_crop = torch.cat((crop0, crop1, crop2), dim=3)
#         img_crop_list.append(img_crop)
#
#     img_crop_list = torch.cat(img_crop_list, dim=0)
#
#     return img_crop_list



for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    epoch_iter = 0

    for i, data in enumerate(dataset):
        iter_start_time = time.time()
        visualizer.reset()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize
        model.set_input(data)
        model.optimize_parameters()

        if total_steps % opt.display_freq == 0:
            save_result = total_steps % opt.update_html_freq == 0
            #visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

        if total_steps % opt.print_freq == 0:
            errors = model.get_current_errors()
            t = (time.time() - iter_start_time) / opt.batchSize
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            if opt.display_id > 0:
                visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, errors)

            log = "Elapsed [{}], Epoch [{}], Iter [{}]".format(
                t, epoch + 1, dataset_size * dataset_size + i + 1)

            for tag, value in errors.items():
                logger.scalar_summary(tag, value, epoch * dataset_size + i + 1)

        if total_steps % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, total_steps))
            model.save('latest')

        if total_steps % opt.save_sample_freq == 0:
            visualizer.save_current_imgs(model.get_current_visuals(), epoch, i, save_result)

        # if total_steps % opt.save_sample_freq == 0:
        #
        #     image_list = []
        #     patch_list = []
        #     for k in range(show_col):
        #         fixed_x = fixed_x_set[k]
        #         image_filtered = model.netG_A(fixed_x)
        #         sample_path = os.path.join(opt.checkpoints_dir, opt.name, 'samples')
        #         util.mkdir(sample_path)
        #
        #         image_list_new = torch.cat([fixed_x, image_filtered], dim = 3)
        #         image_list.append(image_list_new)
        #
        #         crop_img = get_crop_img(fixed_x, fixed_x_boxes[k])
        #         crop_img_filtered = get_crop_img(image_filtered, fixed_x_boxes[k])
        #         crop_list_new = torch.cat([crop_img, crop_img_filtered], dim=3)
        #         patch_list.append(crop_list_new)
        #
        #     image_list = torch.cat((image_list[0], image_list[1]), dim=3)
        #     patch_list = torch.cat((patch_list[0], patch_list[1]), dim=3)
        #
        #     save_image(denorm(image_list.data),
        #        os.path.join(sample_path, '{}_{}_fake.png'.format(epoch, total_steps)), nrow=1, padding=0)
        #
        #     save_image(denorm(patch_list.data),
        #        os.path.join(sample_path, 'patch_{}_{}_fake.png'.format(epoch, total_steps)), nrow=1, padding=0)

    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
        model.save('latest')
        model.save(epoch)

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
    model.update_learning_rate()


