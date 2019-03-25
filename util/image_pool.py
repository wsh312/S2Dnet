import random
import numpy as np
import torch
from torch.autograd import Variable


class ImagePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []
            self.patches = []

    def query(self, images, patches):
        if self.pool_size == 0:
            return Variable(images), Variable(patches)
        return_images = []
        return_patches = []

        for i in range(len(images)):
            image = images[i]
            image = torch.unsqueeze(image, 0)

            patch = patches[i]
            patch = torch.unsqueeze(patch, 0)

            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1

                self.images.append(image)
                self.patches.append(patch)

                return_images.append(image)
                return_patches.append(patch)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)

                    tmp = self.images[random_id].clone()
                    tmp_patch = self.patches[random_id].clone()

                    self.images[random_id] = image
                    self.patches[random_id] = patch

                    return_images.append(tmp)
                    return_patches.append(tmp_patch)
                else:
                    return_images.append(image)
                    return_patches.append(patch)

        return_images = Variable(torch.cat(return_images, 0))
        return_patches = Variable(torch.cat(return_patches, 0))
        return return_images, return_patches


    # def query(self, images):
    #     if self.pool_size == 0:
    #         return Variable(images)
    #     return_images = []
    #     for image in images:
    #         image = torch.unsqueeze(image, 0)
    #         if self.num_imgs < self.pool_size:
    #             self.num_imgs = self.num_imgs + 1
    #             self.images.append(image)
    #             return_images.append(image)
    #         else:
    #             p = random.uniform(0, 1)
    #             if p > 0.5:
    #                 random_id = random.randint(0, self.pool_size-1)
    #                 tmp = self.images[random_id].clone()
    #                 self.images[random_id] = image
    #                 return_images.append(tmp)
    #             else:
    #                 return_images.append(image)
    #     return_images = Variable(torch.cat(return_images, 0))
    #     return return_images
