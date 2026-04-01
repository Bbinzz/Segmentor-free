import json
import cv2
import numpy as np
from rotate_mask import rotate_mask , rotate_image
from torch.utils.data import Dataset
import os
# dataloader更改
def expand(image,mask):
    stepsize = 30
    # stepsize = 100
    mask = cv2.dilate(mask,None,iterations=stepsize)
    extracted_image = cv2.bitwise_and(image, image, mask=mask)
    return extracted_image
class MyDataset(Dataset):
    def __init__(self,type,mask_rotated = False,style = False):
        assert type in ["train","sample"],"please add correct agument train or sample"
        if style :
            jsonpath = "/data1/wzb/dataset/polyp/polyp_dataset/TrainDataset/styledata.json"
            # jsonpath = "/data1/wzb/dataset/polyp_dataset/TrainDataset/dog.json"
            with open(jsonpath, 'rt') as f:
                self.data = json.load(f)
        else:
            with open('/data1/wzb/dataset/polyp/polyp_dataset/TrainDataset/data.json', 'rt') as f:
                self.data = json.load(f)
        self.type = type
        self.mask_rotated = mask_rotated
        self.readindex=[]
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.type == "train":
            item = self.data[idx]
            source_filename = item['source']
            target_filename = item['target']
            prompt = item['prompt']


            source = cv2.imread('/data1/wzb/dataset/polyp/polyp_dataset/TrainDataset/' + source_filename)
            target = cv2.imread('/data1/wzb/dataset/polyp/polyp_dataset/TrainDataset/' + target_filename)
            # self.readindex.append('/data1/wzb/dataset/polyp_dataset/TrainDataset/' + target_filename)
            # extract
            # mask= cv2.imread('/data1/wzb/dataset/polyp_dataset/TrainDataset/' + source_filename,cv2.IMREAD_GRAYSCALE)
            # masked_region = []
            # masked_region.append((cv2.resize(expand(target,mask),(512,512)).astype(np.float32)/127.5)-1.0)
            # masked_region.append((cv2.resize(cv2.bitwise_and(target,target,mask),(512,512)).astype(np.float32)/127.5)-1.0)
            # masked_region = np.array(masked_region)

            # Do not forget that OpenCV read images in BGR order.
            source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
            target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
            # 息肉的分辨率大小不一resize为512x512
            source = cv2.resize(source,(512,512))
            target = cv2.resize(target,(512,512))

            # masked_region = cv2.resize(masked_region,(512,512))

            # Normalize source images to [0, 1].  原始数据集的mask是二值图
            source = source.astype(np.float32) / 255.0  

            # Normalize target images to [-1, 1].
            target = (target.astype(np.float32) / 127.5) - 1.0

            #  Normalize extrat to [-1,1].
            # masked_region = (masked_region.astype(np.float32)/ 127.5)-1.0

            # return dict(jpg=target, txt=prompt, hint=source ,ext=source)
            return dict(jpg=target, txt=prompt, hint=source)
        else:
            item = self.data[idx]
            source_filename = item['source']
            prompt = item['prompt']
            target_filename = item['target']
            # self.readindex.append('/data1/wzb/dataset/polyp_dataset/TrainDataset/' + target_filename)
            # print(target_filename)
            # 方便保存对应的mask
            maskpath = os.path.join("/data1/wzb/dataset/polyp/polyp_dataset/TrainDataset/",source_filename)
            if self.mask_rotated:
                #随机旋转mask
                mask = rotate_mask(maskpath)
            else:
                mask = cv2.imread(maskpath,-1)
                mask = cv2.resize(mask,(512,512))

            source = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
            target = cv2.imread('/data1/wzb/dataset/polyp/polyp_dataset/TrainDataset/' + target_filename)
            # target = cv2.imread('/home/wzb/workspace/ControlNet-main/rotate_mask_contorlnet_seed=42epoch=13-step=2547/masks/b-000000_idx-0.png')
            target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
            target = cv2.resize(target,(512,512))
            # target = rotate_image(target)
            # Normalize source images to [0, 1].  原始数据集的mask是二值图像
            source = source.astype(np.float32) / 255.0  
            # Normalize target images to [-1, 1].
            target = (target.astype(np.float32) / 127.5) - 1.0
            return dict(jpg = target,txt=prompt, hint=source,mask = mask,ext=source)




# class spicalDataset(Dataset):