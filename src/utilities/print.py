import random
import matplotlib.pyplot as plt
from PIL import Image

# Prints some of the data images with labels

def print_image(rows, cols, dataframe, image_path, seed=42):

    total_subplots = rows * cols

    random.seed(seed)
    sample_path = random.sample(image_path, total_subplots)

    fig, axes = plt.subplots(rows, cols, figsize=(9, 9))
    axes = axes.flatten()

    for index, value in enumerate(sample_path):
        # plot the image
        img = Image.open(value)
        axes[index].imshow(img, cmap='grey')

        # Label for the plotted image
        image_index = value.name

        label = dataframe[dataframe['Image Index'] == image_index]['Finding Labels']
        
        # Green if not infected | Red if infected
        color = 'green' if label.iloc[0] == 'No Finding' else 'red'

        axes[index].set_title(label.iloc[0], color=color)

        axes[index].axis('off')

    plt.tight_layout()