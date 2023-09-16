# Complex YOLOv5

## 参考

主要参考

complex yolov4

[maudzung/Complex-YOLOv4-Pytorch: The PyTorch Implementation based on YOLOv4 of the paper: &#34;Complex-YOLO: Real-time 3D Object Detection on Point Clouds&#34; (github.com)](https://github.com/maudzung/Complex-YOLOv4-Pytorch)

yolov5

[ultralytics/yolov5: YOLOv5 🚀 in PyTorch &gt; ONNX &gt; CoreML &gt; TFLite (github.com)](https://github.com/ultralytics/yolov5)

非常感谢，我的工作就是将他们的工作整合在一起了。

## 如何使用：

可以在linux和win上运行，只有pytorch环境就可以，环境配置非常简单

1.在`src/config/`中有`train_config.py`和`kitti_config.py`可以改训练参数，这个比较简单，一看就明白，

数据集路径，模型选择等一些常规参数都在这设置

2.数据集可以放在dataset/kitti文件夹内，新建一个dataset/kitti这样的文件路径

3.在 `src/config/cfg`中有模型的配置文件，可以选则多个模型，

参考yolov5s的设计可以在`complex_yolov5s.cfg`文件中调整

```
depth_multiple= 0.33
width_multiple= 0.50
```

改成complex yolov5L, complex yolov5x等

4.直接运行train就可以

5.添加了可视化点云的代码，需要安装mayavi库，这个麻烦一点，当然你也可以不可视化点云，完全不影响正常训练和验证
