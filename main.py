import os
os.listdir("/kaggle/input")

import os
from glob import glob

# Base dataset path
dataset_base = '/kaggle/input/kidney-segmentation-dataset/2d segmentation dataset/2d segmentation dataset/'

# Paths to images and masks
images_path = os.path.join(dataset_base, 'images')
masks_path = os.path.join(dataset_base, 'masks')

# Get all files
image_files = glob(os.path.join(images_path, '*'))
mask_files = glob(os.path.join(masks_path, '*'))

# Print counts
print("Total images:", len(image_files))
print("Total masks:", len(mask_files))

# If you want them in a tuple like before
counts = (len(image_files), len(mask_files))
print("Counts (images, masks):", counts)

#..................Total images: 2027
Total masks: 2027
Counts (images, masks): (2027, 2027)................#

import os
import pandas as pd
from sklearn.model_selection import train_test_split

# ==============================
# Dataset directories
# ==============================
image_dir = "/kaggle/input/kidney-segmentation-dataset/2d segmentation dataset/2d segmentation dataset/images"
mask_dir  = "/kaggle/input/kidney-segmentation-dataset/2d segmentation dataset/2d segmentation dataset/masks"

# ==============================
# Gather files and match images to masks
# ==============================
image_files = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
mask_files = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir)
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

if len(image_files) != len(mask_files):
    print(f"Warning: {len(image_files)} images vs {len(mask_files)} masks")

# Create a dataframe
df = pd.DataFrame({'img_path': image_files, 'mask_path': mask_files})

# ==============================
# Split dataset: 8:1:1
# ==============================
# First split into train (80%) and temp (20%)
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

# Then split temp into validation (10%) and test (10%)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, shuffle=True)

# Reset indices
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

# ==============================
# Summary
# ==============================
print(f"Total samples: {len(df)}")
print(f"Training samples: {len(train_df)}")
print(f"Validation samples: {len(val_df)}")
print(f"Test samples: {len(test_df)}")

# Optional: show first 5 entries
train_df.head()


import os
import pandas as pd

def create_df(image_dir, mask_dir):
    img_paths = []
    mask_paths = []
    
    # Ensure directories exist
    if not os.path.exists(image_dir) or not os.path.exists(mask_dir):
        raise ValueError("The specified directories do not exist.")
    
    # Get sorted lists of images and masks
    image_files = sorted(os.listdir(image_dir))
    mask_files = sorted(os.listdir(mask_dir))

    # Ensure matching images and masks
    for img_file, mask_file in zip(image_files, mask_files):
        img_path = os.path.join(image_dir, img_file)
        mask_path = os.path.join(mask_dir, mask_file)
        img_paths.append(img_path)
        mask_paths.append(mask_path)

    # Create DataFrame
    df = pd.DataFrame({'img_path': img_paths, 'mask_path': mask_paths})
    return df

# Specify Kaggle dataset directories
image_dir = "/kaggle/input/kidney-segmentation-dataset/2d segmentation dataset/2d segmentation dataset/images"
mask_dir = "/kaggle/input/kidney-segmentation-dataset/2d segmentation dataset/2d segmentation dataset/masks"

# Create dataset DataFrame
df = create_df(image_dir, mask_dir)

# Display sample dataset
print(df.head())

# Optional: check lengths
print("Total images:", len(df))
print("Total masks:", len(df))

import numpy as np
import cv2

def display_image_shapes(df, num_samples=10):
    num_samples = min(num_samples, len(df))  # Ensure num_samples does not exceed available rows

    # Select random indices from the DataFrame
    random_indices = np.random.choice(df.index, size=num_samples, replace=False)
    random_images = df.iloc[random_indices]['img_path'].values
    random_masks = df.iloc[random_indices]['mask_path'].values

    for i in range(num_samples):
        # Read image and mask from Kaggle paths
        image = cv2.imread(random_images[i])
        mask = cv2.imread(random_masks[i], cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            print(f"Warning: Unable to load image or mask at index {random_indices[i]}")
            continue

        # Print only shape values
        print(f"Sample {i+1}: Image Shape = {image.shape}, Mask Shape = {mask.shape}")

# Assuming df was created using Kaggle paths as shown previously
display_image_shapes(df)

import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import albumentations as A

# --- Step 1: Create DataFrame for images and masks ---
def create_df(image_dir, mask_dir):
    img_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir)])
    mask_paths = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir)])
    return pd.DataFrame({'img_path': img_paths, 'mask_path': mask_paths})

# Kaggle dataset paths
image_dir = '/kaggle/input/kidney-segmentation-dataset/2d segmentation dataset/2d segmentation dataset/images'
mask_dir = '/kaggle/input/kidney-segmentation-dataset/2d segmentation dataset/2d segmentation dataset/masks'

df = create_df(image_dir, mask_dir)
print("Total images:", len(df))

# --- Step 2: Visualize random image-mask pairs ---
def display_samples(df, num_samples=5):
    random_indices = np.random.choice(df.index, size=num_samples, replace=False)
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, num_samples * 3))

    for i, idx in enumerate(random_indices):
        img = cv2.imread(df.iloc[idx]['img_path'])
        mask = cv2.imread(df.iloc[idx]['mask_path'], cv2.IMREAD_GRAYSCALE)

        axes[i, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[i, 0].set_title("Original Image")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(mask, cmap="gray")
        axes[i, 1].set_title("Mask")
        axes[i, 1].axis("off")

    plt.tight_layout()
    plt.show()

display_samples(df, num_samples=5)

# --- Step 3: Define augmentation pipeline ---
augmentation = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=20, p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
])

def augment_image(image_path, mask_path):
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    augmented = augmentation(image=image, mask=mask)
    return augmented['image'], augmented['mask']

# --- Step 4: Display augmented samples ---
def display_augmented_samples(df, num_samples=5):
    random_indices = np.random.choice(df.index, size=num_samples, replace=False)
    fig, axes = plt.subplots(num_samples, 4, figsize=(15, num_samples * 3))

    for i, idx in enumerate(random_indices):
        img = cv2.imread(df.iloc[idx]['img_path'])
        mask = cv2.imread(df.iloc[idx]['mask_path'], cv2.IMREAD_GRAYSCALE)
        aug_img, aug_mask = augment_image(df.iloc[idx]['img_path'], df.iloc[idx]['mask_path'])

        axes[i, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[i, 0].set_title("Original Image")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(mask, cmap="gray")
        axes[i, 1].set_title("Original Mask")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(cv2.cvtColor(aug_img, cv2.COLOR_BGR2RGB))
        axes[i, 2].set_title("Augmented Image")
        axes[i, 2].axis("off")

        axes[i, 3].imshow(aug_mask, cmap="gray")
        axes[i, 3].set_title("Augmented Mask")
        axes[i, 3].axis("off")

    plt.tight_layout()
    plt.show()

display_augmented_samples(df, num_samples=5)

import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import albumentations as A
from sklearn.manifold import TSNE
import umap
from sklearn.decomposition import PCA
import random

# --- Step 1: Preprocessing functions ---
def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def apply_histogram_equalization(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    return cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)

# --- Step 2: Augmentation pipeline ---
augmentation = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=20, p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
    A.OneOf([
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
        A.Equalize(p=0.5)
    ], p=0.5)
])

def augment_image(image):
    return augmentation(image=image)['image']

# --- Step 3: Display preprocessing stages ---
def display_preprocessing_stages(image1, image2):
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    for i, img in enumerate([image1, image2]):
        clahe_image = apply_clahe(img)
        hist_eq_image = apply_histogram_equalization(img)
        augmented_image = augment_image(img)

        titles = ["Original", "CLAHE", "Histogram Equalization", "Augmented"]
        images = [img, clahe_image, hist_eq_image, augmented_image]

        for j, (ax, img_, title) in enumerate(zip(axes[i], images, titles)):
            ax.imshow(cv2.cvtColor(img_, cv2.COLOR_BGR2RGB))
            ax.set_title(f"Image {i+1} - {title}")
            ax.axis("off")
    plt.show()

# --- Step 4: t-SNE and UMAP visualization ---
def visualize_image_distribution(df, method='tsne'):
    images = np.stack(df['image_pixels'].values)
    images_flat = images.reshape(images.shape[0], -1)
    images_pca = PCA(n_components=min(50, images_flat.shape[1])).fit_transform(images_flat)

    if method == 'tsne':
        perplexity_value = min(30, images_pca.shape[0] - 1)
        reducer = TSNE(n_components=2, perplexity=perplexity_value, random_state=42)
    else:
        reducer = umap.UMAP(n_components=2, random_state=42)

    images_reduced = reducer.fit_transform(images_pca)
    plt.figure(figsize=(8, 6))
    plt.scatter(images_reduced[:, 0], images_reduced[:, 1], alpha=0.7, cmap='coolwarm')
    plt.title(f'{method.upper()} Visualization of Image Dataset')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.show()

# --- Step 5: Select two random images from Kaggle dataset ---
image_dir = '/kaggle/input/kidney-segmentation-dataset/2d segmentation dataset/2d segmentation dataset/images'
image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

if len(image_files) < 2:
    raise ValueError("Not enough images found in the specified dataset path.")

random_images = random.sample(image_files, 2)
image1 = cv2.imread(os.path.join(image_dir, random_images[0]))
image2 = cv2.imread(os.path.join(image_dir, random_images[1]))

if image1 is None or image2 is None:
    raise ValueError("Failed to load one or both images.")

# Display preprocessing stages
display_preprocessing_stages(image1, image2)

# --- Step 6: Sample DataFrame for t-SNE/UMAP ---
# In practice, replace with actual images resized or flattened
df = pd.DataFrame({
    'image_pixels': [np.random.rand(32, 32, 3) for _ in range(50)]
})

visualize_image_distribution(df, method='tsne')
visualize_image_distribution(df, method='umap')

import os
import pandas as pd
from sklearn.model_selection import train_test_split

# --- Step 1: Create DataFrame for Kaggle dataset ---
image_dir = '/kaggle/input/kidney-segmentation-dataset/2d segmentation dataset/2d segmentation dataset/images'
mask_dir = '/kaggle/input/kidney-segmentation-dataset/2d segmentation dataset/2d segmentation dataset/masks'

# Function to create df
def create_df(image_dir, mask_dir):
    img_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir)])
    mask_paths = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir)])
    return pd.DataFrame({'img_path': img_paths, 'mask_path': mask_paths})

df = create_df(image_dir, mask_dir)
print("Total dataset size:", len(df))

# --- Step 2: Split dataset into train, validation, and test sets ---
train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)

# --- Step 3: Check split sizes ---
print("Training set size:", len(train_df))
print("Validation set size:", len(val_df))
print("Test set size:", len(test_df))
import os
import pandas as pd
from sklearn.model_selection import train_test_split

# --- Step 1: Create DataFrame for Kaggle dataset ---
image_dir = '/kaggle/input/kidney-segmentation-dataset/2d segmentation dataset/2d segmentation dataset/images'
mask_dir = '/kaggle/input/kidney-segmentation-dataset/2d segmentation dataset/2d segmentation dataset/masks'

# Function to create df
def create_df(image_dir, mask_dir):
    img_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir)])
    mask_paths = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir)])
    return pd.DataFrame({'img_path': img_paths, 'mask_path': mask_paths})

df = create_df(image_dir, mask_dir)
print("Total dataset size:", len(df))

# --- Step 2: Split dataset into train, validation, and test sets ---
train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)

# --- Step 3: Check split sizes ---
print("Training set size:", len(train_df))
print("Validation set size:", len(val_df))
print("Test set size:", len(test_df))


from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

def create_image_generator(data_frame,
                           batch_size,
                           image_color_mode="rgb",
                           mask_color_mode="grayscale",
                           image_save_prefix="image",
                           mask_save_prefix="mask",
                           save_to_dir=None,
                           target_size=(256, 256),
                           seed=1):

    def normalize(img, mask):
        img = img / 255.0
        mask = mask / 255.0
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
        return img, mask

    # Image generator
    img_datagen = ImageDataGenerator()
    mask_datagen = ImageDataGenerator()

    # Flow from dataframe (only ONE sample inside)
    img_gen = img_datagen.flow_from_dataframe(
        dataframe=data_frame,
        x_col='img_path',
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        shuffle=True,
        seed=seed
    )

    mask_gen = mask_datagen.flow_from_dataframe(
        dataframe=data_frame,
        x_col='mask_path',
        class_mode=None,
        color_mode=mask_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        shuffle=True,
        seed=seed
    )

    # Combine two generators
    while True:
        img = next(img_gen)
        mask = next(mask_gen)
        yield normalize(img, mask)


# ------------------- Example Usage -------------------
batch_size = 1   # IMPORTANT since you have only one image
train_generator = create_image_generator(train_df, batch_size=batch_size)
val_generator = create_image_generator(val_df, batch_size=batch_size)
test_generator = create_image_generator(test_df, batch_size=batch_size)

# Get one batch
x_batch, y_batch = next(train_generator)
print("Image batch shape:", x_batch.shape)
print("Mask batch shape:", y_batch.shape)
EPOCHS = 35        # keep very low, only for checking pipeline
BATCH_SIZE = 32    # MUST be 1 because dataset has only one sample
learning_rate = 1e-4
w, h = 256, 256   # keep same size as generator target_sizeEPOCHS = 35        # keep very low, only for checking pipeline

#....Transformer enhanced U-Net model..........#
import tensorflow as tf
from tensorflow.keras import layers, Model, backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, SeparableConv2D, MaxPooling2D, Conv2DTranspose, Concatenate, GroupNormalization
import os
from sklearn.model_selection import train_test_split

# -----------------------------
# GPU / Optimization
# -----------------------------
tf.keras.mixed_precision.set_global_policy('float32')  # Disable mixed precision
tf.config.optimizer.set_jit(True)  # Enable XLA for faster execution

def double_conv(x, n_filters):
    x = SeparableConv2D(n_filters, 3, padding='same', activation='relu')(x)
    x = SeparableConv2D(n_filters, 3, padding='same', activation='relu')(x)
    x = GroupNormalization(groups=8)(x)
    return x

def encoder_vit(x, n_filters):
    x = double_conv(x, n_filters)
    p = MaxPooling2D((2,2))(x)
    return x, p

def decoder(x, skip_connection, n_filters):
    x = Conv2DTranspose(n_filters, 3, strides=2, padding='same')(x)
    h_diff = skip_connection.shape[1] - x.shape[1]
    w_diff = skip_connection.shape[2] - x.shape[2]
    if h_diff != 0 or w_diff != 0:
        skip_connection = layers.Cropping2D(
            ((h_diff//2, h_diff-h_diff//2),
             (w_diff//2, w_diff-w_diff//2))
        )(skip_connection)
    x = Concatenate()([x, skip_connection])
    x = double_conv(x, n_filters)
    return x

def VHU_Net_Model(input_shape=(224,224,3)):
    inputs = Input(input_shape)
    s1,p1 = encoder_vit(inputs,16)
    s2,p2 = encoder_vit(p1,32)
    s3,p3 = encoder_vit(p2,64)
    s4,p4 = encoder_vit(p3,128)
    b = double_conv(p4,128)
    b1 = decoder(b,s4,64)
    b2 = decoder(b1,s3,32)
    b3 = decoder(b2,s2,16)
    b4 = decoder(b3,s1,8)
    outputs = SeparableConv2D(1,1,padding='same',activation='sigmoid')(b4)
    return Model(inputs,outputs)

def dice_coefficient(y_true,y_pred,smooth=1e-5):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true*y_pred)
    union = K.sum(y_true)+K.sum(y_pred)
    return (2*intersection + smooth)/(union + smooth)

def dice_loss(y_true,y_pred):
    return -dice_coefficient(y_true,y_pred)

def iou(y_true,y_pred,smooth=1e-5):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true*y_pred)
    union = K.sum(y_true)+K.sum(y_pred)
    return (intersection + smooth)/(union - intersection + smooth)
def dice_np(y_true, y_pred, smooth=1e-6):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2 * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def iou_np(y_true, y_pred, smooth=1e-6):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    union = np.sum(y_true_f) + np.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)
!pip install medpy
from medpy.metric.binary import hd95, assd

def hd95_np(y_true, y_pred):
    return hd95(y_pred, y_true)

def assd_np(y_true, y_pred):
    return assd(y_pred, y_true)
def safe_hd95(y_true, y_pred):
    if y_true.sum() == 0 or y_pred.sum() == 0:
        return np.nan
    return hd95_np(y_true, y_pred)

def safe_assd(y_true, y_pred):
    if y_true.sum() == 0 or y_pred.sum() == 0:
        return np.nan
    return assd_np(y_true, y_pred)
def load_image_mask(image_path, mask_path, img_size=(224,224)):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, img_size)
    image = tf.cast(image, tf.float32)/255.0
    
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, img_size)
    mask = tf.cast(mask, tf.float32)/255.0
    mask = tf.where(mask>0.5,1.0,0.0)  # Ensure binary mask
    return image, mask

def create_dataset(image_dir, mask_dir, batch_size=8, img_size=(224,224)):
    image_paths = sorted([os.path.join(image_dir,f) for f in os.listdir(image_dir)])
    mask_paths = sorted([os.path.join(mask_dir,f) for f in os.listdir(mask_dir)])
    
    train_img,val_img,train_mask,val_mask = train_test_split(image_paths,mask_paths,test_size=0.2,random_state=42)
    val_img,test_img,val_mask,test_mask = train_test_split(val_img,val_mask,test_size=0.5,random_state=42)
    
    def tf_dataset(images,masks):
        ds = tf.data.Dataset.from_tensor_slices((images,masks))
        ds = ds.map(lambda x,y: load_image_mask(x,y,img_size),num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return ds
    
    return tf_dataset(train_img,train_mask), tf_dataset(val_img,val_mask), tf_dataset(test_img,test_mask)

# Set paths and load datasets
image_dir = '/kaggle/input/kidney-segmentation-dataset/2d segmentation dataset/2d segmentation dataset/images'
mask_dir  = '/kaggle/input/kidney-segmentation-dataset/2d segmentation dataset/2d segmentation dataset/masks'

train_ds,val_ds,test_ds = create_dataset(image_dir, mask_dir, batch_size=8)
model = VHU_Net_Model(input_shape=(224,224,3))
optim = Adam(learning_rate=1e-3)

model.compile(
    optimizer=optim,
    loss=dice_loss,
    metrics=[dice_coefficient,iou],
    steps_per_execution=5,
    jit_compile=True
)
EPOCHS = 35

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    verbose=1
)
import numpy as np
from medpy.metric.binary import hd95, assd
import tensorflow as tf

# ----------------------
# Safe Metric Wrappers
# ----------------------
def safe_hd95(y_true, y_pred):
    """Compute HD95, return NaN if empty mask."""
    if y_true.sum() == 0 or y_pred.sum() == 0:
        return np.nan
    return hd95(y_pred, y_true)

def safe_assd(y_true, y_pred):
    """Compute ASSD, return NaN if empty mask."""
    if y_true.sum() == 0 or y_pred.sum() == 0:
        return np.nan
    return assd(y_pred, y_true)

def threshold(pred):
    """Convert predicted probabilities to binary mask."""
    return (pred > 0.5).astype(np.uint8)

def summary_stats(arr):
    """Calculate mean, SD, 95% CI ignoring NaNs."""
    arr = np.array(arr)
    arr = arr[~np.isnan(arr)]  # remove NaNs
    mean = np.mean(arr)
    sd = np.std(arr)
    ci = 1.96 * sd / np.sqrt(len(arr))
    return mean, sd, (mean-ci, mean+ci)

# ----------------------
# Results dictionary
# ----------------------
results = {
    "kidney": {"dice":[], "iou":[], "hd95":[], "assd":[]},
    "tumor": {"dice":[], "iou":[], "hd95":[], "assd":[]},
    "loss": []  # store per-sample loss if needed
}

# ----------------------
# Loop through test dataset
# ----------------------
print("🔹 Evaluating model on test dataset...")
for imgs, masks in test_ds:  # batch size = 1 recommended
    preds = model.predict(imgs)
    preds_bin = threshold(preds)

    masks_np = masks.numpy().astype(np.uint8)

    # ---------------- Kidney Metrics ----------------
    kidney_true = (masks_np == 1).astype(np.uint8)
    kidney_pred = (preds_bin == 1).astype(np.uint8)

    results["kidney"]["dice"].append(dice_np(kidney_true, kidney_pred))
    results["kidney"]["iou"].append(iou_np(kidney_true, kidney_pred))
    results["kidney"]["hd95"].append(safe_hd95(kidney_true, kidney_pred))
    results["kidney"]["assd"].append(safe_assd(kidney_true, kidney_pred))

    # ---------------- Tumor Metrics ----------------
    tumor_true = (masks_np == 2).astype(np.uint8)
    tumor_pred = (preds_bin == 2).astype(np.uint8)

    results["tumor"]["dice"].append(dice_np(tumor_true, tumor_pred))
    results["tumor"]["iou"].append(iou_np(tumor_true, tumor_pred))
    results["tumor"]["hd95"].append(safe_hd95(tumor_true, tumor_pred))
    results["tumor"]["assd"].append(safe_assd(tumor_true, tumor_pred))

    # ---------------- Loss ----------------
    loss_val = model.evaluate(imgs, masks, verbose=0)
    results["loss"].append(loss_val[0])  # only store loss

# ----------------------
# Final Summary
# ----------------------
print("\n==================== FINAL METRICS ====================")
for cls in ["kidney", "tumor"]:
    print(f"\n🎯 {cls.upper()} METRICS")
    for metric in ["dice", "iou", "hd95", "assd"]:
        mean_val, sd_val, ci = summary_stats(results[cls][metric])
        print(f"{metric.upper():<6}: {mean_val:.4f} ± {sd_val:.4f} (95% CI: {ci[0]:.4f} – {ci[1]:.4f})")

# Overall loss
mean_loss, sd_loss, ci_loss = summary_stats(results["loss"])
print(f"\n💥 LOSS: {mean_loss:.4f} ± {sd_loss:.4f} (95% CI: {ci_loss[0]:.4f} – {ci_loss[1]:.4f})")
print("\n✅ Evaluation complete.")
print(history.history.keys())
import matplotlib.pyplot as plt

  # ---------------- Kidney Segmentation Visualization ----------------
def load_and_segment_kidney(image_path, mask_path, image_size=(256, 256)):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, image_size).astype(np.float32) / 255.0
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, image_size).astype(np.uint8)
    _, kidney_mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kidney_mask = kidney_mask / 255
    kidney_region = img * kidney_mask
    return img, kidney_region

def display_kidney(img, kidney_region):
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.imshow(img, cmap='gray')
    plt.title("Kidney Image")
    plt.subplot(1,2,2)
    plt.imshow(kidney_region, cmap='gray')
    plt.title("Segmented Kidney Region")
    plt.show()

# Example paths (update to your Kaggle paths)
image_path = "/kaggle/input/kidney-segmentation-dataset/2d segmentation dataset/2d segmentation dataset/images/2-60-sliced.png"
mask_path = "/kaggle/input/kidney-segmentation-dataset/2d segmentation dataset/2d segmentation dataset/masks/2-60-sliced.png"

kidney_image, segmented_kidney = load_and_segment_kidney(image_path, mask_path)
display_kidney(kidney_image, segmented_kidney)# Extract training history
epochs = range(1, len(history.history['dice_coefficient']) + 1)

# Plot Dice Coefficient
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs, history.history['dice_coefficient'], label='Train Dice Coefficient')
plt.plot(epochs, history.history['val_dice_coefficient'], label='Val Dice Coefficient')
plt.xlabel('Epochs')
plt.ylabel('Dice Coefficient')
plt.title('Dice Coefficient Over Epochs')
plt.legend()

# Plot IoU
plt.subplot(1, 2, 2)
plt.plot(epochs, history.history['iou'], label='Train IoU')
plt.plot(epochs, history.history['val_iou'], label='Val IoU')
plt.xlabel('Epochs')
plt.ylabel('IoU Score')
plt.title('IoU Score Over Epochs')
plt.legend()

plt.tight_layout()
plt.show()
#........Figure 12, Table 6 VHU_Net.......#



#...........ConD-PDN Model.........#
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.layers import Conv2D, GlobalAveragePooling2D, Reshape, Multiply

# Criss-Cross Attention
def criss_cross_attention(x):
    avg_pool = GlobalAveragePooling2D()(x)
    avg_pool = Reshape((1, 1, x.shape[-1]))(avg_pool)
    avg_pool = Conv2D(x.shape[-1], 1, padding='same', activation='sigmoid')(avg_pool)
    return Multiply()([x, avg_pool])

# Define Encoder Block (Using a standard convolutional approach)
def encoder_block(x, filters):
    conv = Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
    conv = Conv2D(filters, (3, 3), activation='relu', padding='same')(conv)
    return conv, layers.MaxPooling2D(pool_size=(2, 2))(conv)

# Define Decoder Block
def decoder_block(x, skip, filters):
    up = layers.UpSampling2D(size=(2, 2))(x)
    merge = layers.Concatenate()([up, skip])
    conv = Conv2D(filters, (3, 3), activation='relu', padding='same')(merge)
    conv = Conv2D(filters, (3, 3), activation='relu', padding='same')(conv)
    return conv

# Define ConD-PDN Model
def cond_pdn(input_shape=(256, 256, 1)):
    inputs = Input(input_shape)

    # Encoder
    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    # Bridge with Criss-Cross Attention
    b = Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    b = Conv2D(1024, (3, 3), activation='relu', padding='same')(b)
    b = criss_cross_attention(b)

    # Decoder
    d1 = decoder_block(b, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    # Output Layer
    outputs = Conv2D(1, (1, 1), activation='sigmoid', padding='same')(d4)

    # Create Model
    model = Model(inputs, outputs, name='ConD_PDN_with_cca')

    return model

# **Dice Coefficient**
def Coffiecient_dice(y_true, y_pred, smooth=100):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(y_pred)
    dice = (2 * intersection + smooth) / (union + smooth)
    return dice

# **Dice Loss**
def Dice_loss(y_true, y_pred):
    return -Coffiecient_dice(y_true, y_pred)

# **IoU Metric**
def Iou(y_true, y_pred, smooth=100):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_pred * y_true)
    union = K.sum(y_true) + K.sum(y_pred)  
    iou = (intersection + smooth) / (union - intersection + smooth)
    return iou

# Define learning rate
learning_rate = 1e-3  

# Initialize optimizer
optim = Adam(learning_rate=learning_rate)

# Compile the model
model.compile(optimizer=optim, loss=Dice_loss, metrics=[Iou, Coffiecient_dice])
model.summary()
import tensorflow as tf
from tensorflow.keras import backend as K

# Dice Coefficient
def coefficient_dice(y_true, y_pred, smooth=1e-5):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(y_pred)
    
    dice = (2 * intersection + smooth) / (union + smooth)
    return dice

# Dice Loss (Negative Dice Coefficient)
def dice_loss(y_true, y_pred):
    return -coefficient_dice(y_true, y_pred)

# Intersection over Union (IoU)
def iou(y_true, y_pred, smooth=1e-5):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(y_pred)
    
    iou = (intersection + smooth) / (union - intersection + smooth)
    return iou

# IoU Loss (Negative IoU)
def iou_loss(y_true, y_pred):
    return -iou(y_true, y_pred)
def create_image_generator(data_frame, batch_size, image_color_mode, mask_color_mode, 
                           image_save_prefix, mask_save_prefix, save_to_dir, target_size, seed):
    # Check if columns exist
    if 'img_path' not in data_frame.columns or 'mask_path' not in data_frame.columns:
        raise ValueError("Columns 'img_path' and 'mask_path' not found in dataframe!")

    img_gen = ImageDataGenerator()
    mask_gen = ImageDataGenerator()

    # Generator for images
    img_gen = img_gen.flow_from_dataframe(
        dataframe=data_frame,
        x_col='img_path',  # Ensure this column exists
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=image_save_prefix,
        seed=seed
    )

    # Generator for masks
    mask_gen = mask_gen.flow_from_dataframe(
        dataframe=data_frame,
        x_col='mask_path',  # Ensure this column exists
        class_mode=None,
        color_mode=mask_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=mask_save_prefix,
        seed=seed
    )

    return img_gen, mask_gen
EPOCHS = 35

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    verbose=1
)
import numpy as np
from medpy.metric.binary import hd95, assd
import tensorflow as tf

# ----------------------
# Safe Metric Wrappers
# ----------------------
def safe_hd95(y_true, y_pred):
    """Compute HD95, return NaN if empty mask."""
    if y_true.sum() == 0 or y_pred.sum() == 0:
        return np.nan
    return hd95(y_pred, y_true)

def safe_assd(y_true, y_pred):
    """Compute ASSD, return NaN if empty mask."""
    if y_true.sum() == 0 or y_pred.sum() == 0:
        return np.nan
    return assd(y_pred, y_true)

def threshold(pred):
    """Convert predicted probabilities to binary mask."""
    return (pred > 0.5).astype(np.uint8)

def summary_stats(arr):
    """Calculate mean, SD, 95% CI ignoring NaNs."""
    arr = np.array(arr)
    arr = arr[~np.isnan(arr)]  # remove NaNs
    mean = np.mean(arr)
    sd = np.std(arr)
    ci = 1.96 * sd / np.sqrt(len(arr))
    return mean, sd, (mean-ci, mean+ci)

# ----------------------
# Results dictionary
# ----------------------
results = {
    "kidney": {"dice":[], "iou":[], "hd95":[], "assd":[]},
    "tumor": {"dice":[], "iou":[], "hd95":[], "assd":[]},
    "loss": []  # store per-sample loss if needed
}

# ----------------------
# Loop through test dataset
# ----------------------
print("🔹 Evaluating model on test dataset...")
for imgs, masks in test_ds:  # batch size = 1 recommended
    preds = model.predict(imgs)
    preds_bin = threshold(preds)

    masks_np = masks.numpy().astype(np.uint8)

    # ---------------- Kidney Metrics ----------------
    kidney_true = (masks_np == 1).astype(np.uint8)
    kidney_pred = (preds_bin == 1).astype(np.uint8)

    results["kidney"]["dice"].append(dice_np(kidney_true, kidney_pred))
    results["kidney"]["iou"].append(iou_np(kidney_true, kidney_pred))
    results["kidney"]["hd95"].append(safe_hd95(kidney_true, kidney_pred))
    results["kidney"]["assd"].append(safe_assd(kidney_true, kidney_pred))

    # ---------------- Tumor Metrics ----------------
    tumor_true = (masks_np == 2).astype(np.uint8)
    tumor_pred = (preds_bin == 2).astype(np.uint8)

    results["tumor"]["dice"].append(dice_np(tumor_true, tumor_pred))
    results["tumor"]["iou"].append(iou_np(tumor_true, tumor_pred))
    results["tumor"]["hd95"].append(safe_hd95(tumor_true, tumor_pred))
    results["tumor"]["assd"].append(safe_assd(tumor_true, tumor_pred))

    # ---------------- Loss ----------------
    loss_val = model.evaluate(imgs, masks, verbose=0)
    results["loss"].append(loss_val[0])  # only store loss

# ----------------------
# Final Summary
# ----------------------
print("\n==================== FINAL METRICS ====================")
for cls in ["kidney", "tumor"]:
    print(f"\n🎯 {cls.upper()} METRICS")
    for metric in ["dice", "iou", "hd95", "assd"]:
        mean_val, sd_val, ci = summary_stats(results[cls][metric])
        print(f"{metric.upper():<6}: {mean_val:.4f} ± {sd_val:.4f} (95% CI: {ci[0]:.4f} – {ci[1]:.4f})")

# Overall loss
mean_loss, sd_loss, ci_loss = summary_stats(results["loss"])
print(f"\n💥 LOSS: {mean_loss:.4f} ± {sd_loss:.4f} (95% CI: {ci_loss[0]:.4f} – {ci_loss[1]:.4f})")
print("\n✅ Evaluation complete.")
print(history.history.keys())
import matplotlib.pyplot as plt

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt


# Function to load and preprocess a single image and mask
def load_mask_and_tumor(image_path, mask_path, image_size=(256, 256)):
    # Load grayscale image and mask
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = cv2.resize(img, image_size).astype(np.float32) / 255.0  # Normalize

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Mask not found: {mask_path}")
    mask = cv2.resize(mask, image_size).astype(np.uint8)  # Convert to uint8

    # Kidney mask using Otsu thresholding
    _, kidney_mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kidney_mask = kidney_mask / 255  # Normalize to 0-1

    # Tumor mask using simple threshold
    tumor_mask = np.where(mask >= 128, 1, 0).astype(np.uint8)

    return kidney_mask, tumor_mask

# Function to display kidney mask and tumor region
def display_kidney_and_tumor(kidney_mask, tumor_mask):
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(kidney_mask, cmap='gray')
    plt.title("Kidney Mask Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(tumor_mask, cmap='gray')
    plt.title("Segmented Tumor Region")
    plt.axis('off')
    
    plt.show()

# Kaggle paths
image_path = "/kaggle/input/kidney-segmentation-dataset/2d segmentation dataset/2d segmentation dataset/images/2-60-sliced.png"
mask_path  = "/kaggle/input/kidney-segmentation-dataset/2d segmentation dataset/2d segmentation dataset/masks/2-60-sliced.png"

# Load masks
kidney_mask, tumor_mask = load_mask_and_tumor(image_path, mask_path)

# Display results
display_kidney_and_tumor(kidney_mask, tumor_mask)

# Extract history
epochs = range(1, len(history.history['coffiecient_dice']) + 1)

# Auto-detect max values
max_dice = max(history.history['coffiecient_dice'] + history.history['val_coffiecient_dice'])
max_iou  = max(history.history['iou'] + history.history['val_iou'])

# Add margin above max
dice_top = max_dice + 0.05
iou_top  = max_iou + 0.05

plt.figure(figsize=(12, 5))

# ---------------- DICE PLOT ----------------
plt.subplot(1, 2, 1)
plt.plot(epochs, history.history['coffiecient_dice'], label='Train Dice', linewidth=2, marker='o')
plt.plot(epochs, history.history['val_coffiecient_dice'], label='Val Dice', linewidth=2, marker='o')

plt.xlabel('Epochs')
plt.ylabel('Dice Coefficient')
plt.title('Dice Coefficient Over Epochs')
plt.ylim(0.2, dice_top)      # <-- auto adjusted
plt.legend()

# ---------------- IOU PLOT ----------------
plt.subplot(1, 2, 2)
plt.plot(epochs, history.history['iou'], label='Train IoU', linewidth=2, marker='o')
plt.plot(epochs, history.history['val_iou'], label='Val IoU', linewidth=2, marker='o')

plt.xlabel('Epochs')
plt.ylabel('IoU Score')
plt.title('IoU Score Over Epochs')
plt.ylim(0.2, iou_top)       # <-- auto adjusted
plt.legend()

plt.tight_layout()
plt.show()
#......Figure 13, Table 6: ConD-PDN........#





#..........fuse ConD-PDN and VHU-Net.....#
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from sklearn.metrics import confusion_matrix
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

def fusion_net(input_shape=(256, 256, 1)):
    inputs = Input(input_shape)
    conv = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    conv = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv)
    output = layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same')(conv)
    return Model(inputs=inputs, outputs=output, name="Fusion_Net")

# Function to fuse ConD-PDN and VHU-Net

def fuse_models(cond_pdn_model, vhu_net_model, input_shape=(256, 256, 1)):
    inputs = Input(input_shape)
    
    # Get outputs from both models
    cond_pdn_output = cond_pdn_model(inputs)
    vhu_net_output = vhu_net_model(inputs)
    
    # Fusion step: Averaging outputs
    fused_output = layers.Average()([cond_pdn_output, vhu_net_output])
    
    # Final output layer
    final_output = layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same')(fused_output)
    
    return Model(inputs, final_output, name="Fused_ConD_VHU_Net")

# Instantiate models
input_shape = (256, 256, 1)
cond_pdn_model = cond_pdn(input_shape)
vhu_net_model = VHU_Net_Model(input_shape)

# Fuse the models
fused_model = fuse_models(cond_pdn_model, vhu_net_model, input_shape)

# Define Dice Loss and IoU

def Coffiecient_dice(y_true, y_pred, smooth=100):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(y_pred)
    dice = (2 * intersection + smooth) / (union + smooth)
    return dice

def Dice_loss(y_true, y_pred):
    return -Coffiecient_dice(y_true, y_pred)

def Iou(y_true, y_pred, smooth=100):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_pred * y_true)
    union = K.sum(y_true) + K.sum(y_pred)  
    iou = (intersection + smooth) / (union - intersection + smooth)
    return iou

# Compile fused model
learning_rate = 1e-3  
optim = Adam(learning_rate=learning_rate)
fused_model.compile(optimizer=optim, loss=Dice_loss, metrics=[Iou, Coffiecient_dice])

# Print model summary
fused_model.summary()
import tensorflow as tf
from tensorflow.keras import backend as K

# Dice Coefficient
def coefficient_dice(y_true, y_pred, smooth=1e-5):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(y_pred)
    
    dice = (2 * intersection + smooth) / (union + smooth)
    return dice

# Dice Loss (Negative Dice Coefficient)
def dice_loss(y_true, y_pred):
    return -coefficient_dice(y_true, y_pred)

# Intersection over Union (IoU)
def iou(y_true, y_pred, smooth=1e-5):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(y_pred)
    
    iou = (intersection + smooth) / (union - intersection + smooth)
    return iou

# IoU Loss (Negative IoU)
def iou_loss(y_true, y_pred):
    return -iou(y_true, y_pred)
def create_image_generator(data_frame, batch_size, image_color_mode, mask_color_mode, 
                           image_save_prefix, mask_save_prefix, save_to_dir, target_size, seed):
    # Check if columns exist
    if 'img_path' not in data_frame.columns or 'mask_path' not in data_frame.columns:
        raise ValueError("Columns 'img_path' and 'mask_path' not found in dataframe!")

    img_gen = ImageDataGenerator()
    mask_gen = ImageDataGenerator()

    # Generator for images
    img_gen = img_gen.flow_from_dataframe(
        dataframe=data_frame,
        x_col='img_path',  # Ensure this column exists
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=image_save_prefix,
        seed=seed
    )

    # Generator for masks
    mask_gen = mask_gen.flow_from_dataframe(
        dataframe=data_frame,
        x_col='mask_path',  # Ensure this column exists
        class_mode=None,
        color_mode=mask_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=mask_save_prefix,
        seed=seed
    )

    return img_gen, mask_gen

EPOCHS = 35

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    verbose=1
)
import numpy as np
from medpy.metric.binary import hd95, assd
import tensorflow as tf

# ----------------------
# Safe Metric Wrappers
# ----------------------
def safe_hd95(y_true, y_pred):
    """Compute HD95, return NaN if empty mask."""
    if y_true.sum() == 0 or y_pred.sum() == 0:
        return np.nan
    return hd95(y_pred, y_true)

def safe_assd(y_true, y_pred):
    """Compute ASSD, return NaN if empty mask."""
    if y_true.sum() == 0 or y_pred.sum() == 0:
        return np.nan
    return assd(y_pred, y_true)

def threshold(pred):
    """Convert predicted probabilities to binary mask."""
    return (pred > 0.5).astype(np.uint8)

def summary_stats(arr):
    """Calculate mean, SD, 95% CI ignoring NaNs."""
    arr = np.array(arr)
    arr = arr[~np.isnan(arr)]  # remove NaNs
    mean = np.mean(arr)
    sd = np.std(arr)
    ci = 1.96 * sd / np.sqrt(len(arr))
    return mean, sd, (mean-ci, mean+ci)

# ----------------------
# Results dictionary
# ----------------------
results = {
    "kidney": {"dice":[], "iou":[], "hd95":[], "assd":[]},
    "tumor": {"dice":[], "iou":[], "hd95":[], "assd":[]},
    "loss": []  # store per-sample loss if needed
}

# ----------------------
# Loop through test dataset
# ----------------------
print("🔹 Evaluating model on test dataset...")
for imgs, masks in test_ds:  # batch size = 1 recommended
    preds = model.predict(imgs)
    preds_bin = threshold(preds)

    masks_np = masks.numpy().astype(np.uint8)

    # ---------------- Kidney Metrics ----------------
    kidney_true = (masks_np == 1).astype(np.uint8)
    kidney_pred = (preds_bin == 1).astype(np.uint8)

    results["kidney"]["dice"].append(dice_np(kidney_true, kidney_pred))
    results["kidney"]["iou"].append(iou_np(kidney_true, kidney_pred))
    results["kidney"]["hd95"].append(safe_hd95(kidney_true, kidney_pred))
    results["kidney"]["assd"].append(safe_assd(kidney_true, kidney_pred))

    # ---------------- Tumor Metrics ----------------
    tumor_true = (masks_np == 2).astype(np.uint8)
    tumor_pred = (preds_bin == 2).astype(np.uint8)

    results["tumor"]["dice"].append(dice_np(tumor_true, tumor_pred))
    results["tumor"]["iou"].append(iou_np(tumor_true, tumor_pred))
    results["tumor"]["hd95"].append(safe_hd95(tumor_true, tumor_pred))
    results["tumor"]["assd"].append(safe_assd(tumor_true, tumor_pred))

    # ---------------- Loss ----------------
    loss_val = model.evaluate(imgs, masks, verbose=0)
    results["loss"].append(loss_val[0])  # only store loss

# ----------------------
# Final Summary
# ----------------------
print("\n==================== FINAL METRICS ====================")
for cls in ["kidney", "tumor"]:
    print(f"\n🎯 {cls.upper()} METRICS")
    for metric in ["dice", "iou", "hd95", "assd"]:
        mean_val, sd_val, ci = summary_stats(results[cls][metric])
        print(f"{metric.upper():<6}: {mean_val:.4f} ± {sd_val:.4f} (95% CI: {ci[0]:.4f} – {ci[1]:.4f})")

# Overall loss
mean_loss, sd_loss, ci_loss = summary_stats(results["loss"])
print(f"\n💥 LOSS: {mean_loss:.4f} ± {sd_loss:.4f} (95% CI: {ci_loss[0]:.4f} – {ci_loss[1]:.4f})")
print("\n✅ Evaluation complete.")
print(history.history.keys())

import matplotlib.pyplot as plt

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
# ------------------------------  
# CELL 3: FUSE KIDNEY + TUMOR REGIONS  
# ------------------------------

# Function to fuse kidney and tumor regions
def fuse_regions(kidney_region, tumor_mask):
    # If tumor = 1 → keep tumor
    # Else → keep kidney grayscale region
    fused = np.where(tumor_mask == 1, tumor_mask, kidney_region)
    return fused

# Function to display fusion results
def display_fused_only(kidney_region, tumor_mask, fused_image):
    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(kidney_region, cmap='gray')
    plt.title("Kidney Region")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(tumor_mask, cmap='gray')
    plt.title("Tumor Mask")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(fused_image, cmap='gray')
    plt.title("Fused (Kidney + Tumor)")
    plt.axis("off")

    plt.show()
fused_output = fuse_regions(segmented_kidney, tumor_mask)

# Display fused regions
display_fused_only(segmented_kidney, tumor_mask, fused_output)

# Extract history
epochs = range(1, len(history.history['coffiecient_dice']) + 1)

# Auto-detect max values
max_dice = max(history.history['coffiecient_dice'] + history.history['val_coffiecient_dice'])
max_iou  = max(history.history['iou'] + history.history['val_iou'])

# Add margin above max
dice_top = max_dice + 0.05
iou_top  = max_iou + 0.05

plt.figure(figsize=(12, 5))

# ---------------- DICE PLOT ----------------
plt.subplot(1, 2, 1)
plt.plot(epochs, history.history['coffiecient_dice'], label='Train Dice', linewidth=2, marker='o')
plt.plot(epochs, history.history['val_coffiecient_dice'], label='Val Dice', linewidth=2, marker='o')

plt.xlabel('Epochs')
plt.ylabel('Dice Coefficient')
plt.title('Dice Coefficient Over Epochs')
plt.ylim(0.2, dice_top)      # <-- auto adjusted
plt.legend()

# ---------------- IOU PLOT ----------------
plt.subplot(1, 2, 2)
plt.plot(epochs, history.history['iou'], label='Train IoU', linewidth=2, marker='o')
plt.plot(epochs, history.history['val_iou'], label='Val IoU', linewidth=2, marker='o')

plt.xlabel('Epochs')
plt.ylabel('IoU Score')
plt.title('IoU Score Over Epochs')
plt.ylim(0.2, iou_top)       # <-- auto adjusted
plt.legend()

plt.tight_layout()
plt.show()
#...........Figure 14, Table 6: Fuse_Models result.....#

# ------------------------------
# CELL: CONFUSION MATRIX
# ------------------------------
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def get_confusion_matrix(model, dataset):
    y_true_list = []
    y_pred_list = []

    print("Generating predictions for confusion matrix...")

    # Loop through TF dataset
    for imgs, masks in dataset:
        preds = model.predict(imgs)

        # Threshold to binary
        preds_bin = (preds > 0.5).astype(np.uint8)
        masks_bin = (masks.numpy() > 0.5).astype(np.uint8)

        # Flatten and store
        y_true_list.append(masks_bin.flatten())
        y_pred_list.append(preds_bin.flatten())

    # Combine all batches
    y_true = np.concatenate(y_true_list)
    y_pred = np.concatenate(y_pred_list)

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Plot
    plt.figure(figsize=(5,4))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap="Blues",
        xticklabels=["Background", "Foreground"],
        yticklabels=["Background", "Foreground"]
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Segmentation Confusion Matrix")
    plt.show()

    return cm

# ---- RUN THE CONFUSION MATRIX ----
cm = get_confusion_matrix(model, test_ds)
print("Confusion matrix:\n", cm)


import cv2
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------
# LOAD IMAGE
# -------------------------------------
img_path = r"C:\Users\DELL\Desktop\VHSCU_Net\fused image.PNG"
original = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

if original is None:
    print("Image not found. Check path:", img_path)
    raise SystemExit

# -------------------------------------
# NORMAL HEATMAP (NO MODEL)
# -------------------------------------
# Scale to 0-255
norm_img = cv2.normalize(original, None, 0, 255, cv2.NORM_MINMAX)
norm_img = np.uint8(norm_img)

# Apply heatmap
heatmap = cv2.applyColorMap(norm_img, cv2.COLORMAP_JET)

# Overlay = 60% original + 40% heatmap
overlay = cv2.addWeighted(cv2.cvtColor(original, cv2.COLOR_GRAY2BGR), 
                          0.6, heatmap, 0.4, 0)

# -------------------------------------
# DISPLAY RESULTS
# -------------------------------------
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.title("Fused (Kidney + Tumor)")
plt.imshow(original, cmap='gray')
plt.axis("off")

plt.subplot(1,3,2)
plt.title("Heatmap")
plt.imshow(heatmap)
plt.axis("off")

plt.subplot(1,3,3)
plt.title("Overlay (Image + Heatmap)")
plt.imshow(overlay)
plt.axis("off")

plt.tight_layout()
plt.show()
#........Table 7: Confusion matrices and heatmaps  result ........#

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from skimage import measure

# Set dataset directories
image_dir = r"C:\dataset\2d segmentation dataset\2d segmentation dataset\images"
mask_dir = r"C:\dataset\2d segmentation dataset\2d segmentation dataset\masks"

# Load image and mask file paths
image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir)
                      if f.endswith(('.png', '.jpg', '.jpeg'))])
mask_paths = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir)
                     if f.endswith(('.png', '.jpg', '.jpeg'))])

# Create DataFrame
test_df = pd.DataFrame({
    'image_path': image_paths,
    'mask_path': mask_paths
})
print(f"Loaded {len(test_df)} image-mask pairs.")

# Function: Contour Overlay Visualization
def overlay_contours(image, ground_truth, predicted, contour_level=0.5):
    gt_contours = measure.find_contours(ground_truth.squeeze(), level=contour_level)
    pred_contours = measure.find_contours(predicted.squeeze(), level=contour_level)
    
    plt.figure(figsize=(8, 8))
    plt.imshow(image, cmap='gray')
    
    for contour in gt_contours:
        plt.plot(contour[:, 1], contour[:, 0], linewidth=2, color='g', label='Ground Truth')
    for contour in pred_contours:
        plt.plot(contour[:, 1], contour[:, 0], linewidth=2, color='r', label='Prediction')
    
    plt.title("Contour Overlay\nGreen: Ground Truth, Red: Prediction")
    plt.axis('off')
    plt.show()

# Function: Error Map Visualization
def error_map_visualization(ground_truth, predicted):
    error = np.abs(ground_truth - predicted)
    plt.figure(figsize=(6, 6))
    plt.imshow(error.squeeze(), cmap='hot')
    plt.colorbar()
    plt.title("Error Map (Absolute Difference)")
    plt.axis('off')
    plt.show()

# Function to demonstrate visualizations
def demo_visualizations(df, prediction_func=None):
    import random
    idx = random.randint(0, len(df)-1)
    image_path = df.iloc[idx]['image_path']
    mask_path = df.iloc[idx]['mask_path']
    
    image = load_img(image_path, target_size=(224, 224))
    mask = load_img(mask_path, target_size=(224, 224), color_mode="grayscale")
    
    image_array = img_to_array(image) / 255.0
    mask_array = img_to_array(mask) / 255.0
    
    predicted_mask = prediction_func(image_array) if prediction_func else mask_array
    
    overlay_contours(image_array, mask_array, predicted_mask)
    error_map_visualization(mask_array, predicted_mask)

# Run the demo (replace prediction_func with your model prediction if available)
demo_visualizations(test_df, prediction_func=None)
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from skimage import measure

# Set dataset directories
image_dir = r"C:\dataset\2d segmentation dataset\2d segmentation dataset\images"
mask_dir = r"C:\dataset\2d segmentation dataset\2d segmentation dataset\masks"

# Load image and mask file paths
image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir)
                      if f.endswith(('.png', '.jpg', '.jpeg'))])
mask_paths = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir)
                     if f.endswith(('.png', '.jpg', '.jpeg'))])

# Create DataFrame
test_df = pd.DataFrame({
    'image_path': image_paths,
    'mask_path': mask_paths
})
print(f"Loaded {len(test_df)} image-mask pairs.")

# Function: Contour Overlay Visualization
def overlay_contours(image, ground_truth, predicted, contour_level=0.5):
    gt_contours = measure.find_contours(ground_truth.squeeze(), level=contour_level)
    pred_contours = measure.find_contours(predicted.squeeze(), level=contour_level)
    
    plt.figure(figsize=(8, 8))
    plt.imshow(image, cmap='gray')
    
    for contour in gt_contours:
        plt.plot(contour[:, 1], contour[:, 0], linewidth=2, color='g', label='Ground Truth')
    for contour in pred_contours:
        plt.plot(contour[:, 1], contour[:, 0], linewidth=2, color='r', label='Prediction')
    
    plt.title("Contour Overlay\nGreen: Ground Truth, Red: Prediction")
    plt.axis('off')
    plt.show()

# Function: Error Map Visualization
def error_map_visualization(ground_truth, predicted):
    error = np.abs(ground_truth - predicted)
    plt.figure(figsize=(6, 6))
    plt.imshow(error.squeeze(), cmap='hot')
    plt.colorbar()
    plt.title("Error Map (Absolute Difference)")
    plt.axis('off')
    plt.show()

# Function to demonstrate visualizations
def demo_visualizations(df, prediction_func=None):
    import random
    idx = random.randint(0, len(df)-1)
    image_path = df.iloc[idx]['image_path']
    mask_path = df.iloc[idx]['mask_path']
    
    image = load_img(image_path, target_size=(224, 224))
    mask = load_img(mask_path, target_size=(224, 224), color_mode="grayscale")
    
    image_array = img_to_array(image) / 255.0
    mask_array = img_to_array(mask) / 255.0
    
    predicted_mask = prediction_func(image_array) if prediction_func else mask_array
    
    overlay_contours(image_array, mask_array, predicted_mask)
    error_map_visualization(mask_array, predicted_mask)

# Run the demo (replace prediction_func with your model prediction if available)
demo_visualizations(test_df, prediction_func=None)
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from skimage import measure

# Set dataset directories
image_dir = r"C:\dataset\2d segmentation dataset\2d segmentation dataset\images"
mask_dir = r"C:\dataset\2d segmentation dataset\2d segmentation dataset\masks"

# Load image and mask file paths
image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir)
                      if f.endswith(('.png', '.jpg', '.jpeg'))])
mask_paths = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir)
                     if f.endswith(('.png', '.jpg', '.jpeg'))])

# Create DataFrame
test_df = pd.DataFrame({
    'image_path': image_paths,
    'mask_path': mask_paths
})
print(f"Loaded {len(test_df)} image-mask pairs.")

# Function: Contour Overlay Visualization
def overlay_contours(image, ground_truth, predicted, contour_level=0.5):
    gt_contours = measure.find_contours(ground_truth.squeeze(), level=contour_level)
    pred_contours = measure.find_contours(predicted.squeeze(), level=contour_level)
    
    plt.figure(figsize=(8, 8))
    plt.imshow(image, cmap='gray')
    
    for contour in gt_contours:
        plt.plot(contour[:, 1], contour[:, 0], linewidth=2, color='g', label='Ground Truth')
    for contour in pred_contours:
        plt.plot(contour[:, 1], contour[:, 0], linewidth=2, color='r', label='Prediction')
    
    plt.title("Contour Overlay\nGreen: Ground Truth, Red: Prediction")
    plt.axis('off')
    plt.show()

# Function: Error Map Visualization
def error_map_visualization(ground_truth, predicted):
    error = np.abs(ground_truth - predicted)
    plt.figure(figsize=(6, 6))
    plt.imshow(error.squeeze(), cmap='hot')
    plt.colorbar()
    plt.title("Error Map (Absolute Difference)")
    plt.axis('off')
    plt.show()

# Function to demonstrate visualizations
def demo_visualizations(df, prediction_func=None):
    import random
    idx = random.randint(0, len(df)-1)
    image_path = df.iloc[idx]['image_path']
    mask_path = df.iloc[idx]['mask_path']
    
    image = load_img(image_path, target_size=(224, 224))
    mask = load_img(mask_path, target_size=(224, 224), color_mode="grayscale")
    
    image_array = img_to_array(image) / 255.0
    mask_array = img_to_array(mask) / 255.0
    
    predicted_mask = prediction_func(image_array) if prediction_func else mask_array
    
    overlay_contours(image_array, mask_array, predicted_mask)
    error_map_visualization(mask_array, predicted_mask)

# Run the demo (replace prediction_func with your model prediction if available)
demo_visualizations(test_df, prediction_func=None)

