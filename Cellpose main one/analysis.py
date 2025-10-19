import cellpose
from cellpose import models, io
from cellpose.io import imread

from skimage.measure import regionprops_table
import skimage.segmentation
import pandas as pd

io.logger_setup()

model = models.Cellpose(model_type='cyto3')

files = ['111023 TA 23-83 laminin-555_0010_txred (1).jpg']

imgs = [imread(f) for f in files]
nimg = len(imgs)

cellpose_masks, cellpose_flows, cellpose_styles, cellpose_diams = model.eval(
    imgs, diameter=None, channels=[1, 0]
)

# ensure downstream processing always works with a list of label images
if isinstance(cellpose_masks, np.ndarray):
    cellpose_masks = [cellpose_masks]

#Get rid of ROIs that are on the border/boundaries
cellpose_masks = [skimage.segmentation.clear_border(im) for im in cellpose_masks]
cellpose_masks = [skimage.segmentation.relabel_sequential(im)[0] for im in cellpose_masks]

from scipy.spatial.distance import pdist
from skimage.measure import find_contours
from skimage.morphology import convex_hull_image
import numpy as np

from scipy.spatial import ConvexHull
from math import cos, sin, radians

def compute_min_feret(label_img):
    """
    Computes the minimum Feret diameter using rotating calipers on the contour of the object.
    """
    min_ferets = []

    for region_label in np.unique(label_img):
        if region_label == 0:
            continue  # skip background

        binary = label_img == region_label
        contours = find_contours(binary, level=0.5)

        if len(contours) == 0:
            min_ferets.append(np.nan)
            continue

        coords = np.vstack(contours)
        coords = coords[:, ::-1]  # switch to (x, y)

        if coords.shape[0] < 3:
            min_ferets.append(np.nan)
            continue

        try:
            hull = ConvexHull(coords)
            hull_coords = coords[hull.vertices]

            min_diameter = np.inf
            for angle in range(180):
                theta = radians(angle)
                rot = np.array([[cos(theta), -sin(theta)],
                                [sin(theta),  cos(theta)]])
                rotated = hull_coords @ rot.T
                width = rotated[:, 0].max() - rotated[:, 0].min()
                if width < min_diameter:
                    min_diameter = width

            min_ferets.append(min_diameter)

        except Exception:
            min_ferets.append(np.nan)

    return min_ferets


for im in cellpose_masks:
    props = regionprops_table(label_image=im, properties=['feret_diameter_max'])
    min_ferets = compute_min_feret(im)
    props['feret_diameter_min'] = min_ferets
    data = pd.DataFrame(props)
    print(data)



# Visualization of the results
import matplotlib.pyplot as plt
import numpy as np
import os

# Make sure you have a folder to save results
output_dir = 'cellpose_outputs'
os.makedirs(output_dir, exist_ok=True)

for i in range(nimg):
    image = imgs[i]
    mask = cellpose_masks[i]  # already processed, borders cleared
    masked = np.ma.masked_where(mask == 0, mask)

    plt.figure(figsize=(8, 8))
    plt.imshow(image, cmap='gray')
    plt.imshow(masked, cmap='jet', alpha=0.5)  # overlay mask
    plt.axis('off')
    plt.title(f'Mask Overlay {i}')
    plt.tight_layout()

    # Save to file
    out_path = os.path.join(output_dir, f'roi_overlay_{i}.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
