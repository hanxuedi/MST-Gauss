# MST-Gauss:Multi-scale Spatial Temporal 4D Gaussian

本方法收敛得非常快，实现了实时渲染速度。

## 环境设置

请遵循项目 [3D-GS](https://github.com/graphdeco-inria/gaussian-splatting) 按照对应的依赖包和软件.

```bash
git clone https://github.com/hanxuedi/MST-Gauss.git
cd MST-Gauss
git submodule update --init --recursive
conda create -n 4dgs python=3.7 
conda activate 4dgs

pip install -r requirements.txt
pip install -e submodules/depth-diff-gaussian-rasterization
pip install -e submodules/simple-knn
```

项目使用环境： pytorch=1.13.1+cu116.

## 数据准备

**合成数据集:**
项目使用了 [D-NeRF](https://github.com/albertpumarola/D-NeRF)中的数据集， 您可以从 [dropbox](https://www.dropbox.com/s/0bf6fl0ye2vz3vr/data.zip?dl=0)下载。

**真实场景数据集:**
项目使用了 [HyperNeRF](https://github.com/google/hypernerf)中的数据集，您可以从 [Hypernerf Dataset](https://github.com/google/hypernerf/releases/tag/v0.1) 中下载场景并且将数据组织成[Nerfies](https://github.com/google/nerfies#datasets)形式。

另外， [Plenoptic Dataset](https://github.com/facebookresearch/Neural_3D_Video) 可以从其官方网站中下载。为了节省空间，您可以提取视频中的视频帧，并按照如下结构组织数据集。

```
├── data
│   | dnerf 
│     ├── mutant
│     ├── standup 
│     ├── ...
│   | hypernerf
│     ├── interp
│     ├── misc
│     ├── virg
│   | dynerf
│     ├── cook_spinach
│       ├── cam00
│           ├── images
│               ├── 0000.png
│               ├── 0001.png
│               ├── 0002.png
│               ├── ...
│       ├── cam01
│           ├── images
│               ├── 0000.png
│               ├── 0001.png
│               ├── ...
│     ├── cut_roasted_beef
|     ├── ...
```

**多视角场景:**
如果您想要训练自定义的多视角场景类型数据集，您应该按照如下方式组织数据集：
```
├── data
|   | multipleview
│     | (your dataset name) 
│   	  | cam01
|     		  ├── frame_00001.jpg
│     		  ├── frame_00002.jpg
│     		  ├── ...
│   	  | cam02
│     		  ├── frame_00001.jpg
│     		  ├── frame_00002.jpg
│     		  ├── ...
│   	  | ...
```
然后运行脚本  `multipleviewprogress.sh` 生成点云和位姿：.
```bash
bash multipleviewprogress.sh (youe dataset name)
```
运行完脚本后，保证文件夹按照如下方式组织：
```
├── data
|   | multipleview
│     | (your dataset name) 
│   	  | cam01
|     		  ├── frame_00001.jpg
│     		  ├── frame_00002.jpg
│     		  ├── ...
│   	  | cam02
│     		  ├── frame_00001.jpg
│     		  ├── frame_00002.jpg
│     		  ├── ...
│   	  | ...
│   	  | sparse_
│     		  ├── cameras.bin
│     		  ├── images.bin
│     		  ├── ...
│   	  | points3D_multipleview.ply
│   	  | poses_bounds_multipleview.npy
```


## 模型训练

对于合成场景如 `bouncingballs`,运行以下脚本：

```
python train.py -s data/dnerf/bouncingballs --port 6017 --expname "dnerf/bouncingballs" --configs arguments/dnerf/bouncingballs.py 
```

对于dynerf场景如 `cut_roasted_beef`, 运行以下脚本
```python
# 首先，提取视频帧
python scripts/preprocess_dynerf.py --datadir data/dynerf/cut_roasted_beef
# 然后，生成点云S
bash colmap.sh data/dynerf/cut_roasted_beef llff
# 对点云进行下采样
python scripts/downsample_point.py data/dynerf/cut_roasted_beef/colmap/dense/workspace/fused.ply data/dynerf/cut_roasted_beef/points3D_downsample2.ply
# 运行训练脚本
python train.py -s data/dynerf/cut_roasted_beef --port 6017 --expname "dynerf/cut_roasted_beef" --configs arguments/dynerf/cut_roasted_beef.py 
```
对于hypernerf场景如 `virg/broom`: 预先生成的点云数据可以从 [这](https://drive.google.com/file/d/1fUHiSgimVjVQZ2OOzTFtz02E9EqCoWr5/view)下载. 下载后将他们放进对于的文件夹，然后就可以跳过前两步。当然也可以直接运行：

```python
# 首先通过COLMAP生成稠密点云
bash colmap.sh data/hypernerf/virg/broom2 hypernerf
# 然后下采样点云
python scripts/downsample_point.py data/hypernerf/virg/broom2/colmap/dense/workspace/fused.ply data/hypernerf/virg/broom2/points3D_downsample2.ply
# 运行训练脚本
python train.py -s  data/hypernerf/virg/broom2/ --port 6017 --expname "hypernerf/broom2" --configs arguments/hypernerf/broom2.py 
```


## 渲染模型

运行以下脚本渲染图像：

```
python render.py --model_path "output/dnerf/bouncingballs/"  --skip_train --configs arguments/dnerf/bouncingballs.py 
```

## 模型评估

运行以下脚本评估模型：

```
python metrics.py --model_path "output/dnerf/bouncingballs/" 
```


## Viewer
[Watch me](./docs/viewer_usage.md)


## 相关工作


[Deformable 3D Gaussians for High-Fidelity Monocular Dynamic Scene Reconstruction](https://ingra14m.github.io/Deformable-Gaussians/)

[SC-GS: Sparse-Controlled Gaussian Splatting for Editable Dynamic Scenes](https://yihua7.github.io/SC-GS-web/)

[MD-Splatting: Learning Metric Deformation from 4D Gaussians in Highly Deformable Scenes](https://md-splatting.github.io/)

[4DGen: Grounded 4D Content Generation with Spatial-temporal Consistency](https://vita-group.github.io/4DGen/)

[Diffusion4D: Fast Spatial-temporal Consistent 4D Generation via Video Diffusion Models](https://github.com/VITA-Group/Diffusion4D)

[DreamGaussian4D: Generative 4D Gaussian Splatting](https://github.com/jiawei-ren/dreamgaussian4d)

[EndoGaussian: Real-time Gaussian Splatting for Dynamic Endoscopic Scene Reconstruction](https://github.com/yifliu3/EndoGaussian)

[EndoGS: Deformable Endoscopic Tissues Reconstruction with Gaussian Splatting](https://github.com/HKU-MedAI/EndoGS)

[Endo-4DGS: Endoscopic Monocular Scene Reconstruction with 4D Gaussian Splatting](https://arxiv.org/abs/2401.16416)



## 项目贡献

**项目仍在维护中，欢迎commits**


一些源代码来源于 [3DGS](https://github.com/graphdeco-inria/gaussian-splatting), [K-planes](https://github.com/Giodiro/kplanes_nerfstudio), [HexPlane](https://github.com/Caoang327/HexPlane), [4DGaussians](https://github.com/hustvl/4DGaussians.git), [Depth-Rasterization](https://github.com/ingra14m/depth-diff-gaussian-rasterization). 非常感谢这些作者


