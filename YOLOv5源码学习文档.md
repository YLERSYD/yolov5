# YOLOv5 源码学习文档

本文档基于当前目录下的 `yolov5` 源码整理，目标是帮助你从整体结构、核心流程、关键模块和源码阅读顺序四个角度理解 YOLOv5。

适合的阅读方式：

1. 先理解目录和主流程。
2. 再看模型是如何由 YAML 配置动态构建出来的。
3. 然后看推理、训练、验证、导出四条主线。
4. 最后深入数据增强、损失函数、NMS、mAP 等细节。

---

## 1. 项目整体结构

当前项目结构大致如下：

```text
yolo-study/
├─ yolov5/
│  ├─ train.py
│  ├─ detect.py
│  ├─ val.py
│  ├─ export.py
│  ├─ benchmarks.py
│  ├─ hubconf.py
│  ├─ models/
│  ├─ utils/
│  ├─ data/
│  ├─ segment/
│  ├─ classify/
│  ├─ tests/
│  ├─ runs/
│  ├─ requirements.txt
│  ├─ pyproject.toml
│  └─ yolov5s.pt
└─ YOLOv5源码学习文档.md
```

核心入口文件：

| 文件 | 作用 |
| --- | --- |
| `detect.py` | 目标检测推理入口 |
| `train.py` | 目标检测训练入口 |
| `val.py` | 模型验证和 mAP 评估入口 |
| `export.py` | 模型导出入口 |
| `benchmarks.py` | 多格式导出和性能测试 |
| `hubconf.py` | PyTorch Hub 加载入口 |

核心目录：

| 目录 | 作用 |
| --- | --- |
| `models/` | 网络结构、模型 YAML、检测头、模型解析逻辑 |
| `utils/` | 数据加载、增强、损失、指标、NMS、绘图、设备工具等 |
| `data/` | 数据集配置、超参数、示例图片、下载脚本 |
| `segment/` | 实例分割任务代码 |
| `classify/` | 图像分类任务代码 |
| `tests/` | 测试代码 |
| `runs/` | 推理、训练、验证输出目录 |

---

## 2. 推荐先掌握的主线

YOLOv5 源码可以按四条主线理解：

```text
推理 detect.py
训练 train.py
验证 val.py
导出 export.py
```

其中最推荐的阅读顺序是：

```text
detect.py
  -> models/common.py
  -> models/yolov5s.yaml
  -> models/yolo.py
  -> train.py
  -> utils/dataloaders.py
  -> utils/augmentations.py
  -> utils/loss.py
  -> val.py
  -> export.py
```

原因：

- `detect.py` 最容易跑通，能快速看到输入图片如何变成检测结果。
- `models/yolov5s.yaml` 可以帮助你理解 YOLOv5 的网络结构。
- `models/yolo.py` 是 YAML 变成 PyTorch 模型的核心。
- `train.py` 内容最多，建议等模型和推理流程有概念后再读。
- `loss.py` 和 `dataloaders.py` 细节较多，适合放在训练主线之后深入。

---

## 3. 推理流程：detect.py

`detect.py` 是目标检测推理入口。

常见命令：

```bash
cd yolov5
python detect.py --weights yolov5s.pt --source data/images
```

推理结果默认保存到：

```text
yolov5/runs/detect/exp*
```

### 3.1 detect.py 的主调用链

```text
parse_opt()
  -> main(opt)
    -> run()
```

真正核心逻辑都在 `run()` 函数中。

### 3.2 run() 的核心步骤

```text
1. 解析输入 source
2. 判断输入类型：图片、视频、文件夹、摄像头、URL、截图
3. 创建保存目录 runs/detect/exp*
4. 选择设备：CPU 或 CUDA
5. DetectMultiBackend 加载模型
6. 创建 dataloader
7. 图片预处理
8. model(im) 前向推理
9. non_max_suppression 做 NMS
10. scale_boxes 把框映射回原图
11. 保存图片、视频、txt、csv、crop
```

### 3.3 detect.py 重要依赖

| 依赖 | 来源 | 作用 |
| --- | --- | --- |
| `DetectMultiBackend` | `models/common.py` | 统一加载 PyTorch、ONNX、TensorRT 等模型 |
| `LoadImages` | `utils/dataloaders.py` | 加载图片、视频、目录 |
| `LoadStreams` | `utils/dataloaders.py` | 加载摄像头、视频流 |
| `LoadScreenshots` | `utils/dataloaders.py` | 加载屏幕截图 |
| `non_max_suppression` | `utils/general.py` | 非极大值抑制 |
| `scale_boxes` | `utils/general.py` | 把预测框从推理尺寸映射回原图尺寸 |
| `Annotator` | `ultralytics.utils.plotting` | 绘制检测框 |

### 3.4 推理时的数据变化

假设输入图片原始尺寸为：

```text
H0 x W0 x 3
```

经过 `letterbox` 后变成：

```text
640 x 640 x 3
```

转换成 Tensor 后：

```text
1 x 3 x 640 x 640
```

模型输出经过 NMS 后，每个检测结果形状类似：

```text
num_boxes x 6
```

每一行含义：

```text
x1, y1, x2, y2, confidence, class_id
```

---

## 4. 模型结构目录：models/

`models/` 是 YOLOv5 源码中最核心的目录。

```text
models/
├─ yolo.py
├─ common.py
├─ experimental.py
├─ tf.py
├─ yolov5n.yaml
├─ yolov5s.yaml
├─ yolov5m.yaml
├─ yolov5l.yaml
├─ yolov5x.yaml
├─ hub/
└─ segment/
```

### 4.1 common.py

`models/common.py` 定义了大量基础网络模块。

常见模块：

| 模块 | 作用 |
| --- | --- |
| `Conv` | 卷积 + BatchNorm + SiLU |
| `DWConv` | 深度可分离卷积 |
| `Bottleneck` | 残差瓶颈结构 |
| `C3` | YOLOv5 主力 CSP 模块 |
| `SPP` | 空间金字塔池化 |
| `SPPF` | 快速空间金字塔池化 |
| `Focus` | 早期版本中的切片降采样模块 |
| `Concat` | 特征拼接 |
| `DetectMultiBackend` | 多后端推理模型加载器 |
| `AutoShape` | 自动处理不同输入格式 |
| `Detections` | 推理结果封装 |
| `Proto` | 分割任务中的 mask prototype 模块 |
| `Classify` | 分类头 |

#### Conv

`Conv` 是最基础的模块，结构为：

```text
Conv2d
  -> BatchNorm2d
  -> SiLU
```

推理加速时，`Conv2d` 和 `BatchNorm2d` 可以被融合，减少计算。

#### C3

`C3` 是 YOLOv5 中大量使用的 CSP 结构。

可以粗略理解为：

```text
输入
 ├─ 分支1：Conv -> Bottleneck 堆叠
 └─ 分支2：Conv
拼接
 -> Conv
输出
```

它的作用：

- 增强特征表达。
- 保持梯度流动。
- 控制计算量。

#### SPPF

`SPPF` 是 Spatial Pyramid Pooling Fast。

作用：

- 扩大感受野。
- 聚合不同尺度上下文。
- 比传统 SPP 更快。

### 4.2 yolo.py

`models/yolo.py` 负责把 YAML 配置解析成 PyTorch 模型。

重要类：

| 类 | 作用 |
| --- | --- |
| `Detect` | YOLO 检测头 |
| `Segment` | 分割检测头 |
| `BaseModel` | 基础模型类 |
| `DetectionModel` | 目标检测模型 |
| `SegmentationModel` | 实例分割模型 |
| `ClassificationModel` | 分类模型 |

重要函数：

| 函数 | 作用 |
| --- | --- |
| `parse_model()` | 解析 YAML，动态创建网络 |

### 4.3 Detect 检测头

`Detect` 是 YOLOv5 目标检测的最后一层。

它接收三个尺度的特征图：

```text
P3/8
P4/16
P5/32
```

分别负责：

| 特征层 | stride | 作用 |
| --- | --- | --- |
| P3 | 8 | 小目标 |
| P4 | 16 | 中目标 |
| P5 | 32 | 大目标 |

如果输入是 `640 x 640`，三个特征图尺寸通常是：

```text
P3: 80 x 80
P4: 40 x 40
P5: 20 x 20
```

YOLOv5 默认每个尺度有 3 个 anchor。

如果类别数是 80，则每个 anchor 输出：

```text
5 + nc = 5 + 80 = 85
```

其中：

```text
x, y, w, h, objectness, class_probs...
```

### 4.4 Detect.forward() 的训练和推理差异

训练时：

```text
返回原始预测张量
用于 ComputeLoss 计算损失
```

推理时：

```text
对 x, y, w, h 解码
拼接所有尺度预测
返回可用于 NMS 的预测结果
```

这点非常重要。很多初学者看源码时会疑惑：为什么训练输出和推理输出不一样。

---

## 5. 模型 YAML：yolov5s.yaml

`models/yolov5s.yaml` 定义了 YOLOv5s 的结构。

核心字段：

```yaml
nc: 80
depth_multiple: 0.33
width_multiple: 0.50
anchors:
  - [10, 13, 16, 30, 33, 23]
  - [30, 61, 62, 45, 59, 119]
  - [116, 90, 156, 198, 373, 326]
```

字段含义：

| 字段 | 含义 |
| --- | --- |
| `nc` | 类别数 |
| `depth_multiple` | 深度缩放系数 |
| `width_multiple` | 通道宽度缩放系数 |
| `anchors` | 三个检测尺度的 anchor |

### 5.1 网络层配置格式

YAML 中每一层格式为：

```text
[from, number, module, args]
```

含义：

| 字段 | 含义 |
| --- | --- |
| `from` | 输入来自哪一层，`-1` 表示上一层 |
| `number` | 模块重复次数 |
| `module` | 模块类型，如 `Conv`、`C3`、`SPPF`、`Detect` |
| `args` | 模块参数 |

例如：

```yaml
[-1, 1, Conv, [64, 6, 2, 2]]
```

表示：

```text
输入来自上一层
重复 1 次
使用 Conv 模块
输出通道 64
卷积核 6
步长 2
padding 2
```

### 5.2 Backbone

Backbone 负责提取图像特征。

YOLOv5s 的 backbone 大致为：

```text
Conv
Conv
C3
Conv
C3
Conv
C3
Conv
C3
SPPF
```

随着网络加深：

```text
特征图尺寸逐渐变小
通道数逐渐增大
语义信息逐渐增强
```

### 5.3 Head

Head 负责多尺度特征融合和检测。

大致结构：

```text
高层特征上采样
  -> 与中层特征 Concat
  -> C3
再上采样
  -> 与浅层特征 Concat
  -> C3
再向下采样
  -> Concat
  -> C3
再向下采样
  -> Concat
  -> C3
Detect(P3, P4, P5)
```

这个结构可以理解为 FPN + PAN 的多尺度融合。

---

## 6. 模型构建流程

模型构建的核心在：

```text
models/yolo.py
```

训练时通常会执行：

```python
model = Model(cfg, ch=3, nc=nc, anchors=hyp.get("anchors")).to(device)
```

这里的 `Model` 实际上是 `DetectionModel` 的别名。

### 6.1 构建流程

```text
读取 yolov5s.yaml
  -> 得到 Python dict
  -> parse_model(dict, ch=[3])
  -> 根据每层配置创建 PyTorch 模块
  -> 记录每层的 from、index、type、参数量
  -> 生成 nn.Sequential
  -> 计算 stride
  -> 调整 anchors
  -> 初始化权重
```

### 6.2 parse_model 的作用

`parse_model()` 是理解 YOLOv5 架构动态生成的关键。

它做的事：

1. 遍历 `backbone + head`。
2. 根据 `module` 字段找到对应 Python 类。
3. 根据 `depth_multiple` 调整重复次数。
4. 根据 `width_multiple` 调整通道数。
5. 创建模块对象。
6. 记录哪些中间层输出需要保存。
7. 返回模型和保存列表。

简化理解：

```text
YAML 配置
  -> parse_model()
  -> PyTorch nn.Sequential
```

---

## 7. 训练流程：train.py

`train.py` 是训练入口。

常见命令：

```bash
cd yolov5
python train.py --data data/coco128.yaml --weights yolov5s.pt --img 640 --epochs 100
```

### 7.1 train.py 主调用链

```text
parse_opt()
  -> main(opt)
    -> train(hyp, opt, device, callbacks)
```

### 7.2 train() 核心流程

```text
1. 创建保存目录
2. 读取 hyp 超参数
3. 读取 data 数据集配置
4. 创建日志器
5. 初始化随机种子
6. 加载数据集信息
7. 创建模型
8. 加载预训练权重
9. 冻结指定层
10. 检查图片尺寸
11. 创建优化器
12. 创建 dataloader
13. 检查 anchors
14. 创建学习率调度器
15. 创建 EMA 模型
16. 创建 ComputeLoss
17. epoch 训练循环
18. 每个 epoch 后验证
19. 保存 last.pt 和 best.pt
```

### 7.3 训练循环内部

每个 batch 的核心逻辑：

```text
imgs, targets = batch
imgs -> 归一化到 0~1
pred = model(imgs)
loss, loss_items = compute_loss(pred, targets)
loss.backward()
optimizer.step()
ema.update(model)
```

### 7.4 训练输出

训练结果默认保存到：

```text
yolov5/runs/train/exp*
```

常见输出文件：

| 文件 | 作用 |
| --- | --- |
| `weights/last.pt` | 最后一个 epoch 的权重 |
| `weights/best.pt` | 验证指标最好的权重 |
| `hyp.yaml` | 本次训练超参数 |
| `opt.yaml` | 本次训练命令参数 |
| `results.csv` | 每个 epoch 的指标 |
| `results.png` | 训练曲线 |
| `confusion_matrix.png` | 混淆矩阵 |

---

## 8. 数据加载：utils/dataloaders.py

`utils/dataloaders.py` 负责训练和推理的数据读取。

重要类和函数：

| 名称 | 作用 |
| --- | --- |
| `create_dataloader()` | 训练/验证 dataloader 创建入口 |
| `LoadImages` | 推理时读取图片、视频 |
| `LoadStreams` | 推理时读取摄像头、视频流 |
| `LoadScreenshots` | 推理时读取屏幕截图 |
| `LoadImagesAndLabels` | 训练时的数据集类 |
| `verify_image_label()` | 检查图片和标签是否有效 |
| `img2label_paths()` | 根据图片路径推导标签路径 |

### 8.1 YOLO 标签格式

YOLOv5 标签文件通常为 `.txt`。

每一行表示一个目标：

```text
class_id x_center y_center width height
```

注意：

- 坐标都是归一化坐标。
- `x_center` 和 `width` 除以图片宽度。
- `y_center` 和 `height` 除以图片高度。
- 取值范围一般是 `0~1`。

示例：

```text
0 0.512 0.438 0.233 0.321
```

表示：

```text
类别 0
中心点 x = 图片宽度的 51.2%
中心点 y = 图片高度的 43.8%
框宽 = 图片宽度的 23.3%
框高 = 图片高度的 32.1%
```

### 8.2 训练数据读取流程

```text
create_dataloader()
  -> LoadImagesAndLabels
    -> 扫描图片路径
    -> 推导 label 路径
    -> 读取或创建 cache
    -> 检查图片和标签
    -> __getitem__ 中加载图片
    -> 数据增强
    -> 返回 image 和 labels
```

### 8.3 label cache

YOLOv5 会缓存标签信息，减少重复扫描开销。

常见缓存文件：

```text
*.cache
```

缓存内容包括：

- 图片路径。
- 图片尺寸。
- 标签。
- segmentation 信息。
- hash。

---

## 9. 数据增强：utils/augmentations.py

数据增强是 YOLOv5 训练效果的重要来源。

重要函数：

| 函数 | 作用 |
| --- | --- |
| `letterbox()` | 等比例缩放并补边 |
| `augment_hsv()` | HSV 色彩增强 |
| `random_perspective()` | 随机透视、旋转、缩放、平移、剪切 |
| `copy_paste()` | Copy-Paste 增强 |
| `mixup()` | MixUp 增强 |
| `Albumentations` | 集成 albumentations 增强库 |

### 9.1 letterbox

推理和验证常用 `letterbox`。

作用：

```text
保持原图比例
缩放到目标尺寸附近
不足部分用灰色填充
保证尺寸能被 stride 整除
```

例如原图不是正方形，直接 resize 到 `640 x 640` 会变形，`letterbox` 可以避免目标形状被拉伸。

### 9.2 Mosaic

Mosaic 通常在 `LoadImagesAndLabels` 内部实现。

作用：

```text
把 4 张图片拼成 1 张训练图
```

优点：

- 增强小目标训练。
- 增加上下文变化。
- 提升 batch 内目标数量。

### 9.3 random_perspective

这个函数会同时改变图片和标签框。

包括：

- 旋转。
- 平移。
- 缩放。
- 剪切。
- 透视变换。

增强时最需要注意的是：图片变了，框也必须同步变。

---

## 10. 损失函数：utils/loss.py

YOLOv5 的损失函数核心类是：

```text
ComputeLoss
```

总损失：

```text
loss = box_loss + obj_loss + cls_loss
```

### 10.1 三部分损失

| 损失 | 含义 |
| --- | --- |
| `lbox` | 边界框回归损失，使用 CIoU |
| `lobj` | objectness 损失，判断该位置是否有目标 |
| `lcls` | 分类损失，多类别时使用 |

### 10.2 ComputeLoss 调用流程

```text
ComputeLoss.__call__(p, targets)
  -> build_targets(p, targets)
  -> 遍历每个检测层
  -> 取出正样本预测
  -> 解码 pxy, pwh
  -> 计算 CIoU
  -> 计算 objectness target
  -> 计算 class target
  -> 加权求和
```

### 10.3 build_targets

`build_targets()` 是损失函数中最难的部分。

它负责：

1. 把真实框匹配到不同检测层。
2. 把真实框匹配到合适 anchor。
3. 把归一化坐标转换到 grid 坐标。
4. 为邻近 grid cell 增加正样本。
5. 返回后续算 loss 所需的索引。

输出包括：

```text
tcls     每个正样本的类别
tbox     每个正样本的 box target
indices  正样本在 batch、anchor、grid 中的位置
anchors  匹配到的 anchor
```

---

## 11. NMS 和坐标工具：utils/general.py

`utils/general.py` 是一个综合工具文件。

和检测结果最相关的函数：

| 函数 | 作用 |
| --- | --- |
| `non_max_suppression()` | 非极大值抑制 |
| `xyxy2xywh()` | 左上右下格式转中心宽高格式 |
| `xywh2xyxy()` | 中心宽高格式转左上右下格式 |
| `scale_boxes()` | 把推理图坐标映射回原图坐标 |
| `clip_boxes()` | 裁剪框到图片范围内 |
| `check_img_size()` | 检查图片尺寸是否符合 stride |
| `increment_path()` | 自动创建 exp、exp2、exp3 路径 |

### 11.1 常见框格式

YOLOv5 中常见两种框格式：

```text
xyxy: x1, y1, x2, y2
xywh: x_center, y_center, width, height
```

训练标签用的是归一化 `xywh`。

NMS 后常用的是像素级 `xyxy`。

### 11.2 NMS 做了什么

NMS 的目的：

```text
同一个目标可能有多个预测框
保留置信度最高的框
删除与它高度重叠的其他框
```

核心参数：

| 参数 | 含义 |
| --- | --- |
| `conf_thres` | 置信度阈值 |
| `iou_thres` | IoU 阈值 |
| `classes` | 只保留指定类别 |
| `agnostic` | 是否类别无关 NMS |
| `max_det` | 最多保留多少个检测框 |

---

## 12. 验证流程：val.py

`val.py` 用于验证模型效果。

常见命令：

```bash
cd yolov5
python val.py --weights yolov5s.pt --data data/coco128.yaml --img 640
```

### 12.1 val.py 核心流程

```text
加载模型
加载验证集
遍历验证图片
前向推理
NMS
预测框和真实框匹配
统计 precision、recall、AP
计算 mAP@0.5 和 mAP@0.5:0.95
保存结果图表
```

### 12.2 常见指标

| 指标 | 含义 |
| --- | --- |
| Precision | 预测为正的样本中，有多少是真的 |
| Recall | 真实目标中，有多少被找出来 |
| AP | 单类别平均精度 |
| mAP@0.5 | IoU=0.5 时所有类别 AP 平均值 |
| mAP@0.5:0.95 | IoU 从 0.5 到 0.95 多个阈值下的 mAP 平均 |

`mAP@0.5:0.95` 更严格，也更能反映定位质量。

---

## 13. 指标计算：utils/metrics.py

`utils/metrics.py` 负责 IoU、AP、mAP、混淆矩阵等计算。

重要函数和类：

| 名称 | 作用 |
| --- | --- |
| `bbox_iou()` | 计算 IoU、GIoU、DIoU、CIoU |
| `box_iou()` | 批量计算 box IoU |
| `ap_per_class()` | 按类别计算 AP |
| `compute_ap()` | 根据 precision-recall 曲线计算 AP |
| `ConfusionMatrix` | 混淆矩阵 |
| `process_batch()` | 判断预测框是否匹配真实框 |

训练 loss 中的 CIoU 来自这里。

---

## 14. 模型导出：export.py

`export.py` 用于把 `.pt` 权重导出到其他部署格式。

常见命令：

```bash
cd yolov5
python export.py --weights yolov5s.pt --include onnx
```

支持格式包括：

| 格式 | 用途 |
| --- | --- |
| TorchScript | PyTorch 部署 |
| ONNX | 通用推理格式 |
| OpenVINO | Intel 平台 |
| TensorRT Engine | NVIDIA GPU 高性能部署 |
| CoreML | Apple 平台 |
| TensorFlow SavedModel | TensorFlow 部署 |
| TFLite | 移动端、嵌入式 |
| Edge TPU | Coral TPU |
| PaddlePaddle | Paddle 部署 |

导出主流程：

```text
加载 PyTorch 模型
创建 dummy input
执行一次前向
根据 include 参数调用对应 export_xxx 函数
保存导出文件
```

---

## 15. segment/ 实例分割任务

`segment/` 是实例分割相关代码。

```text
segment/
├─ train.py
├─ predict.py
├─ val.py
└─ tutorial.ipynb
```

和检测任务相比，分割任务多了：

- mask prototype。
- mask coefficients。
- mask loss。
- mask 后处理。

相关模块：

| 文件 | 作用 |
| --- | --- |
| `models/yolo.py` 的 `Segment` | 分割检测头 |
| `models/common.py` 的 `Proto` | mask prototype 模块 |
| `utils/segment/loss.py` | 分割损失 |
| `utils/segment/dataloaders.py` | 分割数据加载 |
| `utils/segment/general.py` | mask 处理 |

如果你现在主要学习目标检测，可以先跳过这一部分。

---

## 16. classify/ 图像分类任务

`classify/` 是分类任务代码。

```text
classify/
├─ train.py
├─ predict.py
├─ val.py
└─ tutorial.ipynb
```

它使用 YOLOv5 的部分 backbone 能力做分类。

如果当前目标是掌握 YOLO 目标检测源码，也可以后看。

---

## 17. utils/torch_utils.py

`utils/torch_utils.py` 是 PyTorch 相关工具集合。

重要函数和类：

| 名称 | 作用 |
| --- | --- |
| `select_device()` | 选择 CPU 或 CUDA |
| `smart_optimizer()` | 创建优化器 |
| `ModelEMA` | 模型指数滑动平均 |
| `EarlyStopping` | 早停 |
| `fuse_conv_and_bn()` | 融合 Conv 和 BN |
| `model_info()` | 打印模型信息 |
| `smart_DDP()` | 分布式训练封装 |
| `smart_amp_autocast()` | 自动混合精度 |

### 17.1 EMA

EMA 是 Exponential Moving Average。

训练时维护一份平滑后的模型参数：

```text
ema_weight = decay * ema_weight + (1 - decay) * current_weight
```

验证和保存 best 权重时，通常使用 EMA 模型，效果更稳定。

---

## 18. 日志和回调

相关目录：

```text
utils/loggers/
utils/callbacks.py
```

支持的日志系统包括：

- TensorBoard。
- Weights & Biases。
- ClearML。
- Comet。
- CSV / NDJSON。

`callbacks.py` 提供训练过程中的事件钩子，例如：

```text
on_pretrain_routine_start
on_train_batch_end
on_train_epoch_end
on_fit_epoch_end
on_model_save
on_train_end
```

这些钩子让日志系统可以在训练不同阶段插入行为。

---

## 19. 一张图理解 YOLOv5 检测流程

```text
输入图片
  |
  v
letterbox 缩放补边
  |
  v
转 Tensor / 归一化
  |
  v
Backbone 提取特征
  |
  v
Neck 多尺度融合
  |
  v
Detect Head 输出预测
  |
  v
解码 boxes / objectness / classes
  |
  v
NMS 去重
  |
  v
scale_boxes 映射回原图
  |
  v
画框 / 保存结果
```

---

## 20. 一张图理解 YOLOv5 训练流程

```text
读取 data.yaml
  |
  v
创建 Dataset / Dataloader
  |
  v
读取图片和标签
  |
  v
Mosaic / HSV / random perspective 等增强
  |
  v
模型前向传播
  |
  v
ComputeLoss
  |-- box loss
  |-- obj loss
  |-- cls loss
  |
  v
反向传播
  |
  v
优化器更新
  |
  v
EMA 更新
  |
  v
val.py 验证
  |
  v
保存 last.pt / best.pt
```

---

## 21. 初学者最容易卡住的点

### 21.1 YAML 不是普通配置，而是网络结构定义

`models/yolov5s.yaml` 不只是参数配置，它直接描述了网络每一层。

`parse_model()` 会把它动态转换成 PyTorch 模块。

### 21.2 训练和推理的 Detect 输出不同

训练时需要原始特征图预测，用于 loss。

推理时需要解码后的 boxes，用于 NMS。

所以 `Detect.forward()` 内部有：

```python
if not self.training:
    ...
```

### 21.3 标签是归一化 xywh，不是像素 xyxy

训练标签：

```text
class x_center y_center width height
```

而且都是归一化到 `0~1` 的。

### 21.4 letterbox 后坐标要映射回原图

模型是在 `640 x 640` 这类推理尺寸上预测的。

最终画框时必须用 `scale_boxes()` 映射回原图尺寸。

### 21.5 loss 中的 target 分配很复杂

`build_targets()` 涉及：

- anchor 匹配。
- grid 匹配。
- 邻近网格补偿。
- 多尺度检测层分配。

建议最后再深入。

---

## 22. 建议阅读计划

### 第 1 阶段：跑通和看推理

目标：知道一张图片如何输出检测框。

阅读：

```text
detect.py
utils/dataloaders.py 的 LoadImages
utils/augmentations.py 的 letterbox
utils/general.py 的 non_max_suppression
utils/general.py 的 scale_boxes
```

实践：

```bash
cd yolov5
python detect.py --weights yolov5s.pt --source data/images
```

### 第 2 阶段：看模型结构

目标：知道 YOLOv5s 网络是如何搭建的。

阅读：

```text
models/yolov5s.yaml
models/common.py 的 Conv、C3、SPPF、Concat
models/yolo.py 的 DetectionModel、parse_model、Detect
```

建议手动追踪 `yolov5s.yaml` 中每一层的输入输出。

### 第 3 阶段：看训练主线

目标：知道训练代码整体怎么跑。

阅读：

```text
train.py 的 train()
utils/dataloaders.py 的 create_dataloader 和 LoadImagesAndLabels
utils/torch_utils.py 的 smart_optimizer 和 ModelEMA
```

实践：

```bash
cd yolov5
python train.py --data data/coco128.yaml --weights yolov5s.pt --img 640 --epochs 3
```

### 第 4 阶段：看损失函数

目标：理解 YOLOv5 如何把预测和标签对应起来。

阅读：

```text
utils/loss.py 的 ComputeLoss
utils/loss.py 的 build_targets
utils/metrics.py 的 bbox_iou
```

重点理解：

```text
box loss
obj loss
cls loss
anchor 匹配
grid 匹配
```

### 第 5 阶段：看验证和指标

目标：理解 mAP 是如何计算的。

阅读：

```text
val.py
utils/metrics.py 的 ap_per_class、compute_ap、process_batch
```

### 第 6 阶段：看导出和部署

目标：理解 PyTorch 模型如何转 ONNX、TensorRT 等格式。

阅读：

```text
export.py
models/common.py 的 DetectMultiBackend
models/tf.py
```

---

## 23. 推荐调试断点位置

如果你用 PyCharm 或 VS Code 调试，建议先在这些位置打断点：

### 推理断点

```text
detect.py
- run() 开始处
- model = DetectMultiBackend(...)
- for path, im, im0s, vid_cap, s in dataset:
- pred = model(im, augment=augment, visualize=visualize)
- pred = non_max_suppression(...)
```

### 模型构建断点

```text
models/yolo.py
- DetectionModel.__init__()
- parse_model()
- Detect.forward()
```

### 训练断点

```text
train.py
- train()
- model = Model(...)
- train_loader, dataset = create_dataloader(...)
- pred = model(imgs)
- loss, loss_items = compute_loss(pred, targets)
```

### loss 断点

```text
utils/loss.py
- ComputeLoss.__call__()
- ComputeLoss.build_targets()
```

---

## 24. 常用命令

### 推理图片

```bash
cd yolov5
python detect.py --weights yolov5s.pt --source data/images
```

### 推理视频

```bash
cd yolov5
python detect.py --weights yolov5s.pt --source path/to/video.mp4
```

### 摄像头推理

```bash
cd yolov5
python detect.py --weights yolov5s.pt --source 0
```

### 训练 coco128

```bash
cd yolov5
python train.py --data data/coco128.yaml --weights yolov5s.pt --img 640 --epochs 100
```

### 从零训练

```bash
cd yolov5
python train.py --data data/coco128.yaml --weights "" --cfg models/yolov5s.yaml --img 640 --epochs 100
```

### 验证模型

```bash
cd yolov5
python val.py --weights yolov5s.pt --data data/coco128.yaml --img 640
```

### 导出 ONNX

```bash
cd yolov5
python export.py --weights yolov5s.pt --include onnx
```

---

## 25. 源码学习检查清单

可以按下面清单确认自己是否理解：

- [ ] 能解释 `detect.py` 的完整推理流程。
- [ ] 能解释 `letterbox` 为什么不能直接 resize。
- [ ] 能解释 `non_max_suppression` 的作用。
- [ ] 能看懂 `models/yolov5s.yaml` 中 `[from, number, module, args]` 的含义。
- [ ] 能解释 `Conv`、`C3`、`SPPF` 的大致结构。
- [ ] 能解释 `parse_model()` 如何把 YAML 变成模型。
- [ ] 能解释 `Detect.forward()` 训练和推理输出的区别。
- [ ] 能说出 P3、P4、P5 三个检测层分别负责什么。
- [ ] 能解释 YOLO 标签格式。
- [ ] 能解释训练中的 `box loss`、`obj loss`、`cls loss`。
- [ ] 能大致说明 `build_targets()` 在做什么。
- [ ] 能解释 `mAP@0.5` 和 `mAP@0.5:0.95` 的区别。
- [ ] 能知道 `best.pt` 和 `last.pt` 的区别。
- [ ] 能知道导出 ONNX 的基本流程。

---

## 26. 最短学习路线总结

如果只想最快抓住 YOLOv5 主干，按这个顺序看：

```text
1. detect.py 的 run()
2. models/yolov5s.yaml
3. models/common.py 的 Conv、C3、SPPF、Concat
4. models/yolo.py 的 parse_model()
5. models/yolo.py 的 Detect
6. train.py 的 train()
7. utils/dataloaders.py 的 LoadImagesAndLabels
8. utils/loss.py 的 ComputeLoss
9. val.py 的 run()
10. utils/metrics.py 的 ap_per_class
```

掌握这 10 个点，就已经能看懂 YOLOv5 目标检测源码的主干逻辑。

