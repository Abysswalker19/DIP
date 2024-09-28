
import cv2
import numpy as np
import gradio as gr

# 初始化全局变量，存储控制点和目标点
points_src = []
points_dst = []
image = None

# 上传图像时清空控制点和目标点
def upload_image(img):
    global image, points_src, points_dst
    points_src.clear()  # 清空控制点
    points_dst.clear()  # 清空目标点
    image = img
    return img

# 记录点击点事件，并标记点在图像上，同时在成对的点间画箭头
def record_points(evt: gr.SelectData):
    global points_src, points_dst, image
    x, y = evt.index[0], evt.index[1]  # 获取点击的坐标
    
    # 判断奇偶次来分别记录控制点和目标点
    if len(points_src) == len(points_dst):
        points_src.append([x, y])  # 奇数次点击为控制点
    else:
        points_dst.append([x, y])  # 偶数次点击为目标点
    
    # 在图像上标记点（蓝色：控制点，红色：目标点），并画箭头
    marked_image = image.copy()
    for pt in points_src:
        cv2.circle(marked_image, tuple(pt), 1, (255, 0, 0), -1)  # 蓝色表示控制点
    for pt in points_dst:
        cv2.circle(marked_image, tuple(pt), 1, (0, 0, 255), -1)  # 红色表示目标点
    
    # 画出箭头，表示从控制点到目标点的映射
    for i in range(min(len(points_src), len(points_dst))):
        cv2.arrowedLine(marked_image, tuple(points_src[i]), tuple(points_dst[i]), (0, 255, 0), 1)  # 绿色箭头表示映射
    
    return marked_image

# 执行仿射变换
def weights_vec(source_pts, point, alpha=1.0,eps=1e-8):
    weights = []
    for i in range(len(source_pts)):
        weights.append(1./pow(((source_pts[i][0]-point[0])**2 + (source_pts[i][1]-point[1])**2 + eps),2 * alpha))
    return weights


def point_guided_deformation(image, source_pts, target_pts, alpha=1, eps=1e-8):
    """ 
    Return
    ------
        A deformed image.
    """
    
    warped_image = np.array(image)
    ### FILL: 基于MLS or RBF 实现 image warping
    image = np.array(image)
    h = image.shape[0]
    w = image.shape[1]
    # 由于控制点坐标是(y,x)，所以要交换一下x，y
    # 使用逆光流方法，所以要交换一下source和target
    [source_pts[:,[0,1]],target_pts[:,[0,1]]] = [target_pts[:,[1,0]],source_pts[:,[1,0]]]
    # 如果有没映到的点就保持原样
    X = np.tile(np.arange(0,h),(w,1)).T
    Y = np.tile(np.arange(0,w),(h,1))
    # use MLS to do transform
    # warped_image = np.zeros((h, w, 3), dtype=np.uint8) + np.array((255,255,255), dtype=np.uint8).reshape(1,1,3)
    for i in range(h):
        for j in range(w):
            point = np.array([i, j])
            weights = 1./(np.sum((source_pts-point)**2,1)**alpha + eps)
            p_star = np.sum((source_pts*weights.reshape(-1,1)),0)/np.sum(weights)
            q_star = np.sum((target_pts*weights.reshape(-1,1)),0)/np.sum(weights)
            p_hat = (source_pts - p_star)
            # print(source_pts)
            # print(p_star)
            # print(p_hat)
            # print(weights)
            q_hat = (target_pts - q_star)
            # print(np.array([np.outer(p_hat[i], p_hat[i]) for i in range(len(p_hat))]))
            # temp = (point - p_star) * np.linalg.inv(np.sum((np.einsum('ij,ik->ijk', p_hat, p_hat)*weights),0))
            weighted_points = weights * p_hat.T @ p_hat
            # print(np.linalg.inv(weighted_points))
            f = (point - p_star) @ np.linalg.inv(weighted_points) @ (weights * p_hat.T @ q_hat) + q_star
            x = int(f[0])
            y = int(f[1])
            X[i][j] = x
            Y[i][j] = y 
    warped_image = cv2.remap(warped_image,np.float32(Y),np.float32(X),interpolation=cv2.INTER_LINEAR)          

    return warped_image

def run_warping():
    global points_src, points_dst, image ### fetch global variables

    warped_image = point_guided_deformation(image, np.array(points_src), np.array(points_dst))

    return warped_image

# 清除选中点
def clear_points():
    global points_src, points_dst
    points_src.clear()
    points_dst.clear()
    return image  # 返回未标记的原图

# 使用 Gradio 构建界面
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source="upload", label="上传图片", interactive=True, width=800, height=200)
            point_select = gr.Image(label="点击选择控制点和目标点", interactive=True, width=800, height=800)
            
        with gr.Column():
            result_image = gr.Image(label="变换结果", width=800, height=400)
    
    # 按钮
    run_button = gr.Button("Run Warping")
    clear_button = gr.Button("Clear Points")  # 添加清除按钮
    
    # 上传图像的交互
    input_image.upload(upload_image, input_image, point_select)
    # 选择点的交互，点选后刷新图像
    point_select.select(record_points, None, point_select)
    # 点击运行 warping 按钮，计算并显示变换后的图像
    run_button.click(run_warping, None, result_image)
    # 点击清除按钮，清空所有已选择的点
    clear_button.click(clear_points, None, point_select)
    
# 启动 Gradio 应用
demo.launch()
