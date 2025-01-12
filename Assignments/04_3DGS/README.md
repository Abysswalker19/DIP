# 04_3DGS

## run

First, we use Colmap to recover camera poses and a set of 3D points.

```python mvs_with_colmap.py --data_dir data/chair```

Debug the reconstruction by running:

```python debug_mvs_by_projecting_pts.py --data_dir data/chair```

To train the 3dgs:

```python train.py --colmap_dir data/chair --checkpoint_dir data/chair/checkpoints```

### result

150个epoch的时候loss已经基本不下降了

chair第150个epoch的结果

![chair](./assets/chair.png)

视频结果

![chair_gif](./assets/chair.gif)

Loss = 0.0275

原版高斯结果

![cg](./assets/origin_chair.png)

Loss = 0.0055

lego第150个epoch的结果

![lego](./assets/lego.png)

视频结果

![lego_gif](./assets/lego.gif)

原版高斯结果

![lg](./assets/origin_lego.png)

Loss = 0.0130