import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
import scipy.ndimage
import math
from scipy.ndimage import zoom
from scipy.interpolate import CubicSpline
import random
from scipy.stats import tukeylambda
# from bf import bf_structures
from haadf import haadf_structures, haadf_params
import noise

import noise
import numpy as np
import os

def in_image(x, y, w, h):
    return 0 <= x < w and 0 <= y < h

def generate_perlin_noise(
    width, 
    height, 
    scale=100.0, 
    octaves=6, 
    persistence=0.5, 
    lacunarity=2.0, 
    seed=None, 
    normalize=True
):
    """
    生成二维Perlin噪声高度图
    
    参数:
        width, height: 输出数组的尺寸
        scale: 噪声缩放因子，控制噪声的"粒度"
        octaves: 噪声叠加层数，控制细节丰富度
        persistence: 振幅衰减系数，控制高层噪声的影响程度
        lacunarity: 频率变化系数，控制高层噪声的细节密度
        seed: 随机种子，指定后可复现相同的噪声模式
        normalize: 是否将输出归一化到 [0, 1] 范围
    
    返回:
        二维numpy数组，表示高度图
    """
    # 设置随机种子
    if seed is None:
        seed = np.random.randint(0, 1000)
    
    # 初始化高度图
    heightmap = np.zeros((height, width))
    
    # 生成噪声
    for y in range(height):
        for x in range(width):
            # 计算噪声值
            value = noise.pnoise2(
                x/scale, 
                y/scale,
                octaves=octaves,
                persistence=persistence,
                lacunarity=lacunarity,
                repeatx=width,
                repeaty=height,
                base=seed
            )
            heightmap[y, x] = value
    
    # 归一化到 [0, 1] 范围
    if normalize:
        heightmap = (heightmap + 1) / 2
    
    return heightmap

def apply_base(x, y, a1, a2, b1, b2):
    return x * a1 + y * b1, x * a2 + y * b2

def rotate(x, y, a):
    return x * math.cos(a) - y * math.sin(a), x * math.sin(a) + y * math.cos(a)

def create_image(w, h, brightness):
    return np.ones((h, w)) * brightness

def draw_atom(image, x, y, brightness, width):
    center_x, center_y = x, y
    x = np.arange(0, image.shape[1])
    y = np.arange(0, image.shape[0])
    x, y = np.meshgrid(x, y)
    gaussian = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * width**2))
    image += gaussian * brightness

def save_image(name, image):
    cv2.imwrite(name, np.clip(image, 0, 255).astype(np.uint8))

def to_base(x1, x2, a1, a2, b1, b2):
    # 构建系数矩阵和常数项
    A = np.array([[a1, b1], [a2, b2]])
    b_vector = np.array([x1, x2])
    # 解线性方程组
    s, t = np.linalg.solve(A, b_vector)
    return s, t  # 返回浮点数标量

def pattern(structure, image_w, image_h, scale, rotation, atom_brightness):
    x1 = structure[0][2] * scale
    y1 = 0 * scale
    x2 = structure[0][3] * scale
    y2 = structure[0][4] * scale
    x1, y1 = rotate(x1, y1, rotation)
    x2, y2 = rotate(x2, y2, rotation)
    c1x, c1y = to_base(0, 0, x1, y1, x2, y2)
    c2x, c2y = to_base(image_w, 0, x1, y1, x2, y2)
    c3x, c3y = to_base(0, image_h, x1, y1, x2, y2)
    c4x, c4y = to_base(image_w, image_h, x1, y1, x2, y2)
    cx1, cx2 = int(min(c1x, c2x, c3x, c4x)) - 1, int(max(c1x, c2x, c3x, c4x)) + 1
    cy1, cy2 = int(min(c1y, c2y, c3y, c4y)) - 1, int(max(c1y, c2y, c3y, c4y)) + 1
    atoms = []
    for ix in range(cx1, cx2 + 1):
        for iy in range(cy1, cy2 + 1):
            for struct in structure:
                cx, cy = apply_base(ix, iy, x1, y1, x2, y2)
                x0 = struct[0] * scale
                y0 = struct[1] * scale
                x0, y0 = rotate(x0, y0, rotation)
                x, y = x0 + cx, y0 + cy
                if in_image(x, y, image_w, image_h):
                    atoms.append([x, y, struct[5] * atom_brightness, struct[6] * scale])
    return atoms

def mix_patterns(pat1, pat2, mask):
    pat = []
    for p in pat1:
        x = int(p[0])
        y = int(p[1])
        if mask[y, x] < 0.5:
            pat.append(p)
    for p in pat2:
        x = int(p[0])
        y = int(p[1])
        if mask[y, x] >= 0.5:
            pat.append(p)
    return pat

def draw_pattern_noisy(image, pattern):
    for pat in pattern:
        draw_atom(image, pat[0], pat[1], pat[2], pat[3] * 2)

def draw_pattern_enhance_gt(image, pattern):
    for pat in pattern:
        draw_atom(image, pat[0], pat[1], pat[2], pat[3])

def draw_pattern_local_gt(image, pattern):
    for pat in pattern:
        draw_atom(image, pat[0], pat[1], 255, 1)

def cubic_spline_row(final_length, std_dev, upscale):
    # 生成初始向量
    initial_length = int(final_length / upscale + 1)
    x = np.linspace(0, initial_length - 1, initial_length)
    if std_dev < 0:
        std_dev = 0.000001
    y = np.random.normal(0, std_dev, initial_length)
    
    # 创建三次样条插值对象
    cs = CubicSpline(x, y)
    
    # 生成新的 x 轴
    x_new = np.linspace(0, initial_length - 1, final_length)
    
    # 计算插值后的 y 值
    y_new = cs(x_new)
    
    return y_new[:final_length]

def find_y_for_given_x(x1, y1, x2, y2, x0):
    if x1 == x2:
        raise ValueError("x1 and x2 cannot be the same for interpolation.")
    
    # 使用线性插值公式计算 y
    return y1 + (x0 - x1) * (y2 - y1) / (x2 - x1)

def grad(image):
    # 计算x和y方向的梯度
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    return np.sqrt(grad_x**2 + grad_y**2)

def add_scan_noise(image, k, b): # 4~20
    rows, cols = image.shape
    g = grad(image)
    std0 = b + 0.6
    std80 = k * 40 + b + 0.6
    std0 = find_y_for_given_x(5, 4, 9, 20, std0)
    std80 = find_y_for_given_x(5, 4, 9, 20, std80)
    noise0 = np.zeros_like(image)
    noise80 = np.zeros_like(image)
    for i in range(rows):
        noisy_row = cubic_spline_row(cols, std0, 10)
        noise0[i] = noisy_row
    for i in range(rows):
        noisy_row = cubic_spline_row(cols, std80, 10)
        noise80[i] = noisy_row
    rate = g / 80
    rate = np.clip(rate, 0, 1)
    noise = rate * noise80 + (1 - rate) * noise0
    image += noise

def add_read_noise(image, k, b, lambda_):
    # 创建与image相同大小的噪声数组
    noise = np.zeros_like(image, dtype=float)
    
    # 计算每个像素的噪声标准差
    stddev = np.clip(image * k + b, 0, None)
    
    # 遍历每个像素并添加噪声
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # 生成形状参数为lambda_的Tukey's lambda分布的随机噪声
            noise_value = tukeylambda.rvs(lambda_, loc=0, scale=stddev[i, j])
            noise[i, j] = noise_value
    
    # 将噪声添加到原始图像
    image += noise


# for i, structure in enumerate(haadf_structures):
#     structure = haadf_structures[i]
#     img = create_image(256, 256, 50)
#     background = generate_perlin_noise(256, 256) * 100
#     pat = pattern(structure, 256, 256, .1, math.pi/4, 0.3, 2)
#     draw_pattern(img, pat)
#     img += background
#     save_image(str(i)+'.png', img)

n_struct = len(haadf_structures)
root = 'haadf_data3'
os.makedirs(root, exist_ok=True)
os.makedirs(f'{root}/noisy', exist_ok=True)
os.makedirs(f'{root}/enhance_gt', exist_ok=True)
os.makedirs(f'{root}/local_gt', exist_ok=True)
for i in range(1000):
    
    img_enhance_gt = create_image(256, 256, 0)
    img_local_gt = create_image(256, 256, 0)
    
    scale = 0.1 + random.random() * 0.25
    atom_brightness = 0.2 + random.random() * 0.4
    background_min = random.randint(0, 50)
    background_scale = random.randint(50, 100)

    img = create_image(256, 256, background_min)
    background = generate_perlin_noise(256, 256) * background_scale
    pat1 = pattern(haadf_structures[random.randint(0, n_struct-1)], 256, 256, scale, random.random()* 2* math.pi, atom_brightness)
    pat2 = pattern(haadf_structures[random.randint(0, n_struct-1)], 256, 256, scale, random.random()* 2* math.pi, atom_brightness) if random.random() > 0.1 else []

    mask = generate_perlin_noise(256, 256)
    pat = mix_patterns(pat1, pat2, mask)
    # 随机将一些原子（10%概率）移动到图像内的随机位置
    image_w, image_h = 256, 256  # 图像尺寸，与创建图像时保持一致
    modified_pat = []
    for p in pat:
        if random.random() < 0.1:  # 10%概率移动该原子
            # 生成图像范围内的随机位置
            new_x = random.uniform(0, image_w - 1)
            new_y = random.uniform(0, image_h - 1)
            # 保留原原子的亮度和宽度参数，仅修改位置
            modified_pat.append([new_x, new_y, p[2], p[3]])
        else:
            # 不移动的原子保持原样
            modified_pat.append(p)
    pat = modified_pat  
    draw_pattern_noisy(img, pat)
    draw_pattern_enhance_gt(img_enhance_gt, pat)
    img_enhance_gt *= 255. / np.max(img_enhance_gt)
    save_image(f'{root}/enhance_gt/{i}.png', img_enhance_gt)
    draw_pattern_local_gt(img_local_gt, pat)
    save_image(f'{root}/local_gt/{i}.png', img_local_gt)

    img += background
    # add scan noise
    scan_k = random.normalvariate(haadf_params[4], haadf_params[5])
    scan_b = random.normalvariate(haadf_params[6], haadf_params[7])
    add_scan_noise(img, scan_k, scan_b)

    # add read noise
    # noisy = a + tukeylambda.rvs(-0.1, 0, 15.929, size=a.shape)
    read_k = random.normalvariate(haadf_params[8], haadf_params[9])
    read_b = random.normalvariate(haadf_params[10], haadf_params[11])
    read_lambda = random.normalvariate(haadf_params[12], haadf_params[13])
    add_read_noise(img, read_k, read_b, read_lambda)
    save_image(f'{root}/noisy/{i}.png', img)
    print(i)