"""SFIN noise calibration, packaged for the SFFN release. See README.md."""
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm, tukeylambda
import os


from pathlib import Path
import argparse


def read_image(i):
    image = cv2.imdecode(np.fromfile(input_files[i], dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f'Cannot read image: {input_files[i]}')
    return image.astype(np.float32)

def save_image(i, name, image):
    ok, encoded = cv2.imencode('.png', np.clip(image, 0, 255).astype(np.uint8))
    if not ok:
        raise ValueError('Could not encode output image')
    encoded.tofile(output_path / f'{i}_{name}.png')

def apply_gaussian_blur(image, ksize=(15, 15), sigma=0):
    blurred_image = cv2.GaussianBlur(image, ksize, sigma)
    return blurred_image

def apply_row_wise_gaussian_blur(image, ksize=(21, 1), sigma=0):
    height, width = image.shape
    blurred_image = np.zeros_like(image)
    for i in range(height):
        row = image[i, :].reshape(1, width)
        blurred_row = cv2.GaussianBlur(row, ksize, sigma)
        blurred_image[i, :] = blurred_row
    return blurred_image

def image_grad(image):
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    return np.sqrt(grad_x**2 + grad_y**2)

def image_row_grad(image):
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    return np.abs(grad_x)

def noise_statistics(image, noise):
    if image.min() < 0 or image.max() >= 256:
        raise ValueError('Calibration bin values must be in [0, 256); check image contrast/gradient.')
    a = []
    for i in range(256):
        a.append([])
    h, w = image.shape
    for i in range(h):
        for j in range(w):
            a[int(image[i,j])].append(noise[i,j])
    means = []
    stds = []
    values = []
    for i in range(256):
        if len(a[i]) > 1:
            values.append(i)
            mu, std = norm.fit(a[i])
            means.append(mu)
            stds.append(std)
    front = len(values) * 4 // 5
    x = np.array(values[:front])
    y = np.array(stds[:front])
    if len(x) < 2:
        raise ValueError('Too few populated bins for a linear fit.')
    k, b = np.polyfit(x, y, 1)
    return values, stds, means, k, b

def noise_statistics_tl(image, noise):
    a = []
    for i in range(256):
        a.append([])
    h, w = image.shape
    for i in range(h):
        for j in range(w):
            a[int(image[i,j])].append(noise[i,j])
    means = []
    stds = []
    lambdas = []
    values = []
    for i in [50, 100, 150]:
        if len(a[i]) > 1:
            values.append(i)
            lam, mu, std = tukeylambda.fit(a[i])
            lambdas.append(lam)
            means.append(mu)
            stds.append(std)
    return values, lambdas

def grad_curve(image, grad):
    a = []
    for i in range(256):
        a.append([])
    h, w = image.shape
    for i in range(h):
        for j in range(w):
            a[int(image[i,j])].append(grad[i,j])
    grads = []
    values = []
    for i in range(256):
        if len(a[i]) > 1:
            mu = sum(a[i]) / len(a[i])
            grads.append(mu / 5)
            values.append(i)
    return values, grads

def save_noise_statistics(i, values, stds, means, name, x_name, curve=None):
    plt.figure()
    plt.plot(values, means, color='black')
    plt.plot(values, stds, color='blue')
    if curve:
        plt.plot(curve[0], curve[1], color='red')
    # plt.xlim(0, 70)
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.xlabel(f'{x_name}', fontsize=15)
    plt.ylabel('Noise Parameters', fontsize=15)
    plt.grid(True)
    plt.savefig(f'{output_path}/{i}_{name}_noise_statistics.svg', format='svg')
    plt.savefig(f'{output_path}/{i}_{name}_noise_statistics.png', format='png')
    plt.close()

def save_noise_statistics_tl(i, values, lambdas, name, x_name, curve=None):
    plt.figure()
    plt.plot(values, lambdas, color='blue')
    if curve:
        plt.plot(curve[0], curve[1], color='red')
    # plt.xlim(0, 70)
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.xlabel(f'{x_name}', fontsize=15)
    plt.ylabel('Noise Parameters', fontsize=15)
    plt.grid(True)
    plt.savefig(f'{output_path}/{i}_{name}_noise_statistics.svg', format='svg')
    plt.savefig(f'{output_path}/{i}_{name}_noise_statistics.png', format='png')
    plt.close()
    

def save_parameter_range_norm(a, name):
    mu, std = norm.fit(a)
    file = open(f'{output_path}/{name}.txt', 'w', encoding='utf-8')
    file.write(f'{mu} {std}\n')
    for i in range(len(a)):
        file.write(f'{i} {a[i]}\n')
    file.close()

def save_parameter_range_uniform(a, name):
    max_ = max(a)
    min_ = min(a)
    file = open(f'{output_path}/{name}.txt', 'w', encoding='utf-8')
    file.write(f'{min_} {max_}\n')
    for i in range(len(a)):
        file.write(f'{i} {a[i]}\n')
    file.close()
    
def main():
    parser = argparse.ArgumentParser(description="Calibrate STEM noise using the historical SFIN procedure.")
    parser.add_argument('--mode', choices=['haadf', 'bf'], required=True)
    parser.add_argument('--input', type=Path, help='Input directory of 8-bit PNG images; defaults to bundled samples.')
    parser.add_argument('--output', type=Path, required=True, help='New or empty output directory.')
    parser.add_argument('--limit', type=int, help='Process only the first N images for a smoke test.')
    args = parser.parse_args()
    global input_files, output_path
    input_path = args.input or Path(__file__).resolve().parent / 'samples' / args.mode
    input_files = sorted(input_path.glob('*.png'), key=lambda p: (not p.stem.isdigit(), int(p.stem) if p.stem.isdigit() else p.name))
    if args.limit is not None:
        if args.limit < 1:
            parser.error('--limit must be positive')
        input_files = input_files[:args.limit]
    if not input_files:
        parser.error(f'No PNG images found in {input_path}')
    output_path = args.output
    if output_path.exists() and (not output_path.is_dir() or any(output_path.iterdir())):
        parser.error('Output must be a new or empty directory; existing files are never deleted.')
    output_path.mkdir(parents=True, exist_ok=True)
    is_bf = args.mode == 'bf'
    image_count = len(input_files)
    scan_ks = []
    scan_bs = []
    scan_grad_ks = []
    scan_grad_bs = []
    read_ks = []
    read_bs = []
    read_lambdas = []
    background = []
    atom_brightness = []
    for image_i in range(image_count):
        print('processing image', image_i)
        original = read_image(image_i)

        blurred = apply_gaussian_blur(original)
        save_image(image_i, 'blurred', blurred)

        row_blurred = apply_row_wise_gaussian_blur(original)
        save_image(image_i, 'row_blurred', row_blurred)

        scan_noise = row_blurred - blurred
        save_image(image_i, 'scan_noise', scan_noise * 3 + 128)

        grad = image_grad(blurred)
        save_image(image_i, 'grad', grad * 2)
        
        values, stds, means, k, b = noise_statistics(blurred, scan_noise)
        grad_curve_ = grad_curve(blurred, grad)
        save_noise_statistics(image_i, values, stds, means, 'blurred-based_scan', 'Pixel Value', grad_curve_)

        values, stds, means, k, b = noise_statistics(grad, scan_noise)
        save_noise_statistics(image_i, values, stds, means, 'grad-based_scan', 'Magnitude of Gradient')

        scan_ks.append(k)
        scan_bs.append(b)

        read_noise = original - row_blurred
        save_image(image_i, 'read_noise', read_noise * 3 + 128)
        values, stds, means, k, b = noise_statistics(blurred, read_noise)
        save_noise_statistics(image_i, values, stds, means, 'blurred-based_read', 'Pixel Value')
        values, lambdas = noise_statistics_tl(blurred, read_noise)
        save_noise_statistics_tl(image_i, values, lambdas, 'blurred-based_read_lambda', 'Pixel Value')
        read_ks.append(k)
        read_bs.append(b)
        read_lambdas.append(sum(lambdas)/3.)


        if is_bf:
            background.append(blurred.max())
            atom_brightness.append(blurred.min() - blurred.max())
        else:
            background.append(blurred.min())
            atom_brightness.append(blurred.max() - blurred.min())


    save_parameter_range_norm(scan_ks, 'Scan Noise k')
    save_parameter_range_norm(scan_bs, 'Scan Noise b')
    save_parameter_range_norm(read_ks, 'Read Noise k')
    save_parameter_range_norm(read_bs, 'Read Noise b')
    save_parameter_range_norm(read_lambdas, 'Read Noise λ')
    save_parameter_range_uniform(background, 'Background Brightness')
    save_parameter_range_uniform(atom_brightness, 'Atom Brightness')


if __name__ == "__main__":
    main()
