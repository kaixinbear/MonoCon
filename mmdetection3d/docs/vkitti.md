# 1: 下载vkitti数据集放到 mmdetection3d/data/vkitti下
```
mmdetection3d
├── mmdet3d
├── tools
├── configs
├── data
│   ├── vkitti
│   │   ├── labels
│   │   ├── rgb
│   │   │   train.txt
│   │   │   test.txt
│   │   │   val.txt
```

# 2. 生成检测任务对应的数据集划分
因为有一些图片是没有前景物体的，所以这里我写了一个简单的脚本过滤掉了没有前景的样本；
将split_3DDet_data.py中对应的路径改成你自己的目录
```
python split_3DDet_data.py
```

```
mmdetection3d
├── mmdet3d
├── tools
├── configs
├── data
│   ├── vkitti
│   │   ├── labels
│   │   ├── rgb
│   │   │   train.txt
│   │   │   test.txt
│   │   │   val.txt
│   │   │   train_Det3D.txt
│   │   │   test_Det3D.txt
│   │   │   val_Det3D.txt
```

# 3. 生成对应的GT PKL文件
'''
python tools/create_vkitti_data.py vkitti --root-path ./data/vkitti --out-dir ./data/vkitti --extra-tag vkitti
'''

# 按照monocon的安装要求配置好环境后，运行：
```
python tools/train.py configs/monocon/monocon_dla34_200e_vkitti.py
```