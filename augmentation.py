import numpy as np
from PIL import Image, ImageEnhance
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def rotate_90(image):
    return image.rotate(-90, expand=True)


def rotate_180(image):
    return image.rotate(180, expand=True)


def rotate_270(image):
    return image.rotate(-270, expand=True)


def random_crop(image, min_ratio=0.7, max_ratio=0.95):
    width, height = image.size
    ratio = np.random.uniform(min_ratio, max_ratio)
    new_width = int(width * ratio)
    new_height = int(height * ratio)
    left = np.random.randint(0, width - new_width + 1)
    top = np.random.randint(0, height - new_height + 1)
    return image.crop((left, top, left + new_width, top + new_height))


def jitter_brightness(image, min_factor=0.5, max_factor=2.5):
    factor = np.random.uniform(min_factor, max_factor)
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)


def jitter_contrast(image, min_factor=0.5, max_factor=2.5):
    factor = np.random.uniform(min_factor, max_factor)
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)


def jitter_saturation(image, min_factor=0.5, max_factor=2.5):
    factor = np.random.uniform(min_factor, max_factor)
    enhancer = ImageEnhance.Color(image)
    return enhancer.enhance(factor)

def apply_random_augmentations(image, min_augmentations=1, max_augmentations=4):
    rotation_funcs = [rotate_90, rotate_180, rotate_270]
    other_funcs = [random_crop, jitter_brightness, jitter_contrast, jitter_saturation]

    selected_rotation = []
    if np.random.random() < 0.5:
        selected_rotation = [np.random.choice(rotation_funcs)]

    n_other = np.random.randint(min_augmentations, max_augmentations + 1)
    selected_other = list(np.random.choice(other_funcs, size=min(n_other, len(other_funcs)), replace=False))

    selected_funcs = selected_rotation + selected_other
    np.random.shuffle(selected_funcs)

    result = image.copy()
    for func in selected_funcs:
        result = func(result)

    return result


def visualize_augmentations():
    train_csv_path="animal-clef-2025/splits/train.csv"
    data_root="animal-clef-2025"
    n_examples=10
    output_path="augmentation_examples.png"

    train_df = pd.read_csv(train_csv_path)
    data_root = Path(data_root)

    identity = np.random.choice(train_df['identity'].unique())

    identity_rows = train_df[train_df['identity'] == identity]

    valid_paths = []
    for _, row in identity_rows.iterrows():
        full_path = data_root / row['path']
        if full_path.exists():
            valid_paths.append(full_path)

    n_cols = 5
    n_rows = (n_examples + n_cols) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))
    axes = axes.flatten()

    source_path = np.random.choice(valid_paths)
    original = Image.open(source_path).convert('RGB')
    axes[0].imshow(original)
    axes[0].set_title("Original")
    axes[0].axis('off')

    for i in range(n_examples):
        image = Image.open(source_path).convert('RGB')
        aug_image = apply_random_augmentations(image)
        axes[i + 1].imshow(aug_image)
        axes[i + 1].set_title(f"Aug {i + 1}")
        axes[i + 1].axis('off')

    for i in range(n_examples + 1, len(axes)):
        axes[i].axis('off')

    plt.suptitle("Augmentation examples")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

if __name__ == "__main__":
    visualize_augmentations()
