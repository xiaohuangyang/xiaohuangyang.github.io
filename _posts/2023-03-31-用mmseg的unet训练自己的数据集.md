笔记

1.github上搜索mmcv，根据get_started步骤配置环境

2.labelme标注数据集

标签有顺序，区域重叠情况下，下方标签优先度更高

3.转为voc形式

数据集准备，推荐在 mmsegmetaion 目录新建路径 data，然后将数据集转换成 MMSegmentation 可用的格式：分别定义好数据集图像和标注的正确文件路径，**其中的标注格式为仅包含每个像素对应标签 id 的单通道标注文件，而不是三通道的 RGB 格式。**

在github下载labelme2voc.py文件，转换指令为：

```
python labelme2voc.py data_annotated data_dataset_voc --labels label.txt
```

用d2l.py将标签图进行转化

```python
import os
from PIL import Image
import numpy as np
def Mode_P_to_L(img_file,output_file,stretch_value=1):

    file_name_list = os.listdir(img_file)

    for file in file_name_list:
        img_path = os.path.join(img_file, file)
        image = Image.open(img_path)
        print(image.mode)
        img_arry = Image.fromarray(np.uint8(image))
        print(img_arry.mode)
        img_L = img_arry.convert("L")
        print(img_L.mode)

        img_end = Image.fromarray(np.uint8(img_L) * stretch_value)
        print(img_end.mode)
        save_name = os.path.join(output_file,file)
        img_end.save(save_name)
        print(save_name)
        image = Image.open(img_path)
        image = np.uint8(image)
        h,w = np.shape(image)
        image = image.reshape((h*w))
        image = image.tolist()
        # image = list(image)
        print((image))

if __name__ == '__main__':
    Mode_P_to_L('data_dataset_voc/SegmentationClassPNG','data_dataset_voc\\SegmentationClassPNG_toL')
```

用split分割为训练集和测试集，并按照mmseg数据集格式整理：

BLAST-----images------train,validation
            -----annotations-----train,validation

```python
from sklearn.model_selection import train_test_split
import os
import shutil


def split(image_dir = 'images',anno_dir = 'annotations'):
    images = []
    for file in os.listdir(image_dir):
        filename = file.split('.')[0]
        if filename!='train' and filename!='validation':
            images.append(filename)

    train, val = train_test_split(images, train_size=0.7, random_state=0)

    return train,val


def copyfile(image_dir,anno_dir,train,val):
    for file in train:
        shutil.copy(image_dir+'\\'+file+".jpg",image_dir+'\\'+"train"+'\\'+file+".jpg")
        shutil.copy(anno_dir+'\\'+file+".png",anno_dir+'\\'+"train"+'\\'+file+".png")
    for file in val:
        shutil.copy(image_dir+'\\'+file+".jpg",image_dir+'\\'+"validation"+'\\'+file+".jpg")
        shutil.copy(anno_dir+'\\'+file+".png",anno_dir+'\\'+"validation"+'\\'+file+".png")

    print("复制完成")

if __name__ == "__main__":
    image_dir = 'images'
    anno_dir = 'annotations'
    train,val = split(image_dir,anno_dir)
    copyfile(image_dir,anno_dir,train,val)
```

在mmseg中configs选择unet，按格式创建配置文件（**注意改num classes！**）：

```python
_base_ = [
    '../_base_/models/fcn_unet_s5-d16.py', '../_base_/datasets/blast.py', //修改此处
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
model = dict(decode_head=dict(num_classes=5),
             auxiliary_head=dict(num_classes=5),
             test_cfg=dict(crop_size=(128, 128), stride=(85, 85)))
```

在base/datasets中创建blast.py:

```python
dataset_type = 'BLASTDataset'
data_root = 'data/BLAST'
img_scale = (500,550)
```

在mmseg\datasets中注册数据集：

```python
@DATASETS.register_module()
class BLASTDataset(CustomDataset):
    """BLAST dataset.

    In segmentation map annotation for HRF, 0 stands for background, which is
    included in 2 categories. ``reduce_zero_label`` is fixed to False. The
    ``img_suffix`` is fixed to '.png' and ``seg_map_suffix`` is fixed to
    '.png'.
    """

    CLASSES = ('background', 'outer','inner','TE','ICM')

    PALETTE = [[120,120,120], [6,230,230], [5,120,5], [120,5,5], [5,5,120]]

    def __init__(self, **kwargs):
        super(MYCELLDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)
        assert self.file_client.exists(self.img_dir)
```

在mmseg/datasets中的____init____.py添加刚刚新建的数据集

```PYTHON
from .blast import BLASTDataset
__all__ = [
    'CustomDataset', 'build_dataloader', 'ConcatDataset', 'RepeatDataset',
    'DATASETS', 'build_dataset', 'PIPELINES', 'CityscapesDataset',
    'PascalVOCDataset', 'ADE20KDataset', 'PascalContextDataset',
    'PascalContextDataset59', 'ChaseDB1Dataset', 'DRIVEDataset', 'HRFDataset',
    'STAREDataset', 'DarkZurichDataset', 'NightDrivingDataset',
    'COCOStuffDataset', 'LoveDADataset', 'MultiImageMixDataset',
    'iSAIDDataset', 'ISPRSDataset', 'PotsdamDataset','MYCELLDataset','BLASTDataset'
]
```

更新mmseg


python setup.py install 

pip install -v -e .

1.训练

```
python tools/train.py configs/unet/fcn_unet_s5-d16_128x128_40k_blast.py
```

2.测试

```
python tools/test.py  configs/unet/fcn_unet_s5-d16_128x128_40k_blast.py checkpoint/unet_blast.pth --show-dir result/blast
```

3.分析日志

```
python tools/analyze_logs.py  log.json --keys loss_cls --legend loss_cls 
```

eg：

```java
python tools/analyze_logs.py work_dirs/my_coco_config/20220503_165551.log.json --out log_curve.png  --keys loss
```

