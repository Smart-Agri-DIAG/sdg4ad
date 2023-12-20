import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import cv2
import imageio
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import os

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

def compute_dimension_and_submatrix(grape, output_path, img, i)->None:

    img_name = img.split(".jpg")[0]

    rows_mask, cols_mask, _ = np.where(grape!= 0)

    min_row, max_row = np.min(rows_mask), np.max(rows_mask)
    min_col, max_col = np.min(cols_mask), np.max(cols_mask)

    height_mask = max_row - min_row + 1
    width_mask = max_col - min_col + 1

    submatrix = grape[min_row:max_row+1, min_col:max_col+1]
    submatrix_uint8 = np.where(submatrix, 255, 0).astype(np.uint8)
    imageio.imsave(f"{output_path}/{img_name}_mask_{i}.jpg", submatrix)
    #return submatrix


def main(input_path, output_path, min_mask_region_area, stability_score_thresh, pred_iou_thresh):
  grapes_dir = input_path
  lista_img = os.listdir(grapes_dir)
  k = 0

  if not os.path.exists(output_path):
    os.makedirs(output_path)

  sam_checkpoint = "sam_vit_h_4b8939.pth"
  model_type = "vit_h"

  device = "cuda"

  sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
  sam.to(device=device)

  mask_generator = SamAutomaticMaskGenerator(
  model=sam,
  points_per_side=32,
  pred_iou_thresh=pred_iou_thresh,
  stability_score_thresh=stability_score_thresh,
  crop_n_layers=1,
  crop_n_points_downscale_factor=2,
  min_mask_region_area=min_mask_region_area,
  )

  for img in lista_img:
    if img.endswith((".jpg", ".png", ".jpeg")):
      k+=1
      print(f"Immagine {k} di {len(lista_img)}")
      path = os.path.join(grapes_dir, img)
      print(img)
      image = cv2.imread(path)
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

      masks2 = mask_generator.generate(image)
      print("Numero di maschere individuate: ", len(masks2))
      #masks2 = mask_generator.postprocess_small_regions(masks2, min_mask_region_area, 0.86)
      new_dict = []
      for elem in masks2:
          if elem['area'] > min_mask_region_area:
              new_dict.append(elem)
      mask_sorted = sorted_anns = sorted(new_dict, key=(lambda x: x['area']), reverse=True)
      print("Numero di maschere rimanenti in base all'area minima: ", len(mask_sorted))
      for i in range(len(mask_sorted)):
        good_mask_unit8 = (mask_sorted[i]['segmentation'].astype(np.uint8)) * 255
        good_grape = cv2.bitwise_and(image, image, mask=good_mask_unit8)
        compute_dimension_and_submatrix(good_grape, output_path, img, i)



#print(f"Parametri inseriti: \n Input: {input_path}\n Output: {output_path}\n Pred_iou: {pred_iou_thresh} \n Stability_score: {stability_score_thresh} \n min_mask: {min_mask_region_area}")
#check = input("\nI parametri sono corretti? {y/n}")
#if check.lower()=="y":
input_path = ""
output_path = ""
min_mask_region_area = 10000
stability_score_thresh = 0.9
pred_iou_thresh = 0.92
main(input_path, output_path, min_mask_region_area, stability_score_thresh, pred_iou_thresh)
#else:
#print("Exit")