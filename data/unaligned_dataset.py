import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform, get_transform_patch
from data.image_folder import make_dataset
from PIL import Image
import PIL
import random
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class UnalignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')

        self.A_paths = make_dataset(self.dir_A)
        self.B_paths = make_dataset(self.dir_B)

        self.A_paths = sorted(self.A_paths)
        self.B_paths = sorted(self.B_paths)
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        self.transform = get_transform(opt)
        self.transform_patch = get_transform_patch(opt)

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]

        A_sift_path = A_path.replace('.png', '.txt')
        A_corres = self.read_corres(A_sift_path)

        index_A = index % self.A_size
        index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        B_sift_path = B_path.replace('.png', '.txt')
        B_corres = self.read_corres(B_sift_path)

        # print('(A, B) = (%d, %d)' % (index_A, index_B))
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        A_boxes, A_patch = self.compute_random_patch(A_img, A_corres)
        B_boxes, B_patch = self.compute_random_patch(B_img, B_corres)

        A = self.transform(A_img)
        B = self.transform(B_img)

        A_patch = self.transform_patch(A_patch)
        B_patch = self.transform_patch(B_patch)


        if self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        if output_nc == 1:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)
        return {'A': A, 'B': B,
                'A_patch': A_patch, 'B_patch': B_patch,
                'A_boxes' : A_boxes, 'B_boxes': B_boxes,
                'A_paths': A_path, 'B_paths': B_path}

    def read_corres(self, path):

        corres = []
        with open(path) as f:
            for l in f:
                x0, y0, x1, y1, x2, y2 = l.split()
                corres.append([x0, y0, x1, y1, x2, y2])

        return corres

    def compute_random_patch(self, img, corres):

        # self.opt.patchSize = 64
        img_size = self.opt.fineSize
        crop_size = self.opt.patchSize
        pace = crop_size / 2

        origin_size = 512

        coor_scal = int(origin_size / img_size)

        # size = img_size, img_size
        img = img.resize((img_size * 3, img_size), Image.ANTIALIAS)

        found = False
        sr = random.SystemRandom()

        for i in range(len(corres) * 2):
            centers = sr.choice(corres)

            c0 = [int(centers[0]) / coor_scal, int(centers[1]) / coor_scal]
            c1 = [int(centers[2]) / coor_scal, int(centers[3]) / coor_scal]
            c2 = [int(centers[4]) / coor_scal, int(centers[5]) / coor_scal]

            self.box0 = (c0[1] - pace, c0[0] - pace, c0[1] + pace, c0[0] + pace)
            if self.box_not_valid(self.box0):
                continue

            self.box1 = (c1[1] + img_size - pace, c1[0] - pace, c1[1] + img_size + pace, c1[0] + pace)
            if self.box_not_valid(self.box1):
                continue

            self.box2 = (c2[1] + img_size*2 - pace, c2[0] - pace, c2[1] + img_size*2 + pace, c2[0] + pace)
            if self.box_not_valid(self.box2):
                continue

            found = True
            break

        if not found:
            half_size = img_size / 2
            c0 = [half_size, half_size]
            c1 = [half_size, half_size]
            c2 = [half_size, half_size]
            self.box0 = (c0[1] - pace, c0[0] - pace, c0[1] + pace, c0[0] + pace)
            self.box1 = (c1[1] + img_size - pace, c1[0] - pace, c1[1] + img_size + pace, c1[0] + pace)
            self.box2 = (c2[1] + img_size * 2 - pace, c2[0] - pace, c2[1] + img_size * 2 + pace, c2[0] + pace)

        crop0 = img.crop(self.box0)
        crop1 = img.crop(self.box1)
        crop2 = img.crop(self.box2)

        img_crop = Image.new('RGB', (crop_size * 3, crop_size))
        img_crop.paste(crop0, (0, 0))
        img_crop.paste(crop1, (crop_size, 0))
        img_crop.paste(crop2, (crop_size * 2, 0))

        # img_crop.save("crop.jpg")

        return (self.box0, self.box1, self.box2) ,img_crop


    def box_not_valid(self, box):

        img_size = self.opt.fineSize
        if box[0] < 0 or box[1] < 0 or box[2] >= img_size * 3 or box[3] >= img_size:
            return True
        return  False




    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'UnalignedDataset'
