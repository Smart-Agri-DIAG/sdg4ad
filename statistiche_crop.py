# %%
import subprocess
# Specify the package to inll
package_name = "matplotlib"
# Run the inllation command using subprocess
try:
    subprocess.run(["pip", "inll", package_name], check=True)
    print(f"Successfully inlled {package_name}")
except subprocess.CalledProcessError as e:
    print(f"Failed to inll {package_name}. Error: {e}")


#Imports
import cv2
import os
from os import listdir
import math
import matplotlib.pyplot as plt
import numpy as np

#Aux functions
def count_edges(canny_im):
    count=0
    for i in range(canny_im.shape[0]):
        for j in range(canny_im.shape[1]):
           if (canny_im[i][j]==255):
              count+=1
    return count

#Classes
class all_about_image:
    def __init__(self, image, canny_large, canny_small, size, edges,not_black):
        self.image = image
        self.canny_large = canny_large
        self.canny_small = canny_small
        self.size = size
        self.edges = edges
        self.not_black = not_black


######################################################### Code #############################################################

#Parameters

t_lower_0 =50   # Lower Threshold canny 1
t_upper_0 = 150  # Upper threshold canny 1
t_lower_1 =150   # Lower Threshold canny 2
t_upper_1 = 255  # Upper threshold canny 2
p_rum=0.25 # da 0 a 1 se la percentuale di edge è superiore a que soglia l'immagine viene scartata (rumore accessivo)
show_scarto=False #settare a True per visualizzare immagini scartate da p_rum

percentuale_contenuto=0.5# da 0 a 1 decide se scartare un immagine, se i pixel non neri rappresentano almeno la percentuale indicata l'immagine re

NUM_BINS=500# set buckets
percentuale=0.5 # da 0 a 1 in base alla percentuale massima di edge che si vuole considerare



############# Read BAD GRAPES

PN_bad_train_images=[]
PN_bad_test_images=[]
RG_bad_train_images=[]
RG_bad_test_images=[]

################ get the path for PN bad train
folder="/content/drive/MyDrive/Paper/CroppedPatchesDataset/PN/train/bad/"
PN_bad_train_num=0
scarto_rumore=0

for images in os.listdir(folder):
  # check if the image ends with png
  if (images.endswith(".jpg")):
    print(images)
    img = cv2.imread(folder+images)  # Read image
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # Applying the Canny Edge filter
    canny_0 = cv2.Canny(img, t_lower_0, t_upper_0)
    canny_1 = cv2.Canny(img, t_lower_1, t_upper_1)

    size=img.shape[0]*img.shape[1]

    not_black_count=0
    for i in range(0,img.shape[0]):
      for j in range(0,img.shape[1]):
        if img[i][j][0]!=0 or img[i][j][1]!=0 or img[i][j][2]!=0:
          not_black_count+=1

    count_0=count_edges(canny_0)

    if (count_0/size>=p_rum):
      scarto_rumore+=1
      if(show_scarto):
        cv2.imshow(img)
        cv2.imshow(canny_0)
      continue

    count_1=count_edges(canny_1)
    single_image=all_about_image(img,canny_0,canny_1,size,(count_0-count_1),not_black_count)
    PN_bad_train_images.append(single_image)
    PN_bad_train_num+=1

print("Numero immagini PN bad train lette e salvate: ",PN_bad_train_num)
print("Scartate per eccessivo rumore: ",scarto_rumore)

################ get the path for PN bad test
folder="/content/drive/MyDrive/Paper/CroppedPatchesDataset/PN/test/bad/"
PN_bad_test_num=0
scarto_rumore=0

for images in os.listdir(folder):
  # check if the image ends with png
  if (images.endswith(".jpg")):
    print(images)
    img = cv2.imread(folder+images)  # Read image
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # Applying the Canny Edge filter
    canny_0 = cv2.Canny(img, t_lower_0, t_upper_0)
    canny_1 = cv2.Canny(img, t_lower_1, t_upper_1)

    size=img.shape[0]*img.shape[1]
    not_black_count=0
    for i in range(0,img.shape[0]):
      for j in range(0,img.shape[1]):
        if img[i][j][0]!=0 or img[i][j][1]!=0 or img[i][j][2]!=0:
          not_black_count+=1

    count_0=count_edges(canny_0)

    if (count_0/size>=p_rum):
      scarto_rumore+=1
      if(show_scarto):
        cv2.imshow(img)
        cv2.imshow(canny_0)
      continue

    count_1=count_edges(canny_1)
    single_image=all_about_image(img,canny_0,canny_1,size,(count_0-count_1),not_black_count)
    PN_bad_test_images.append(single_image)
    PN_bad_test_num+=1

print("Numero immagini PN bad test lette e salvate: ",PN_bad_test_num)
print("Scartate per eccessivo rumore: ",scarto_rumore)

################ get the path for RG bad train
folder="/content/drive/MyDrive/Paper/CroppedPatchesDataset/RG/train/bad/"
RG_bad_train_num=0
scarto_rumore=0

for images in os.listdir(folder):
  # check if the image ends with png
  if (images.endswith(".jpg")):
    print(images)
    img = cv2.imread(folder+images)  # Read image
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # Applying the Canny Edge filter
    canny_0 = cv2.Canny(img, t_lower_0, t_upper_0)
    canny_1 = cv2.Canny(img, t_lower_1, t_upper_1)

    size=img.shape[0]*img.shape[1]
    not_black_count=0
    for i in range(0,img.shape[0]):
      for j in range(0,img.shape[1]):
        if img[i][j][0]!=0 or img[i][j][1]!=0 or img[i][j][2]!=0:
          not_black_count+=1

    count_0=count_edges(canny_0)

    if (count_0/size>=p_rum):
      scarto_rumore+=1
      if(show_scarto):
        cv2.imshow(img)
        cv2.imshow(canny_0)
      continue

    count_1=count_edges(canny_1)
    single_image=all_about_image(img,canny_0,canny_1,size,(count_0-count_1),not_black_count)
    RG_bad_train_images.append(single_image)
    RG_bad_train_num+=1

print("Numero immagini RG bad train lette e salvate: ",RG_bad_train_num)
print("Scartate per eccessivo rumore: ",scarto_rumore)

################ get the path for PN bad test
folder="/content/drive/MyDrive/Paper/CroppedPatchesDataset/RG/test/bad/"
RG_bad_test_num=0
scarto_rumore=0

for images in os.listdir(folder):
  # check if the image ends with png
  if (images.endswith(".jpg")):
    print(images)
    img = cv2.imread(folder+images)  # Read image
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # Applying the Canny Edge filter
    canny_0 = cv2.Canny(img, t_lower_0, t_upper_0)
    canny_1 = cv2.Canny(img, t_lower_1, t_upper_1)

    size=img.shape[0]*img.shape[1]
    not_black_count=0
    for i in range(0,img.shape[0]):
      for j in range(0,img.shape[1]):
        if img[i][j][0]!=0 or img[i][j][1]!=0 or img[i][j][2]!=0:
          not_black_count+=1

    count_0=count_edges(canny_0)

    if (count_0/size>=p_rum):
      scarto_rumore+=1
      if(show_scarto):
        cv2.imshow(img)
        cv2.imshow(canny_0)
      continue

    count_1=count_edges(canny_1)
    single_image=all_about_image(img,canny_0,canny_1,size,(count_0-count_1),not_black_count)
    RG_bad_test_images.append(single_image)
    RG_bad_test_num+=1

print("Numero immagini RG bad test lette e salvate: ",RG_bad_test_num)
print("scartate per eccessivo rumore: ",scarto_rumore)


############# Read GOOD GRAPES

PN_good_train_images=[]
PN_good_test_images=[]
RG_good_train_images=[]
RG_good_test_images=[]

################ get the path for PN good train
folder="/content/drive/MyDrive/Paper/CroppedPatchesDataset/PN/train/good/"
PN_good_train_num=0
scarto_rumore=0

for images in os.listdir(folder):
  # check if the image ends with png
  if (images.endswith(".jpg")):
    print(images)
    img = cv2.imread(folder+images)  # Read image
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # Applying the Canny Edge filter
    canny_0 = cv2.Canny(img, t_lower_0, t_upper_0)
    canny_1 = cv2.Canny(img, t_lower_1, t_upper_1)

    size=img.shape[0]*img.shape[1]
    not_black_count=0
    for i in range(0,img.shape[0]):
      for j in range(0,img.shape[1]):
        if img[i][j][0]!=0 or img[i][j][1]!=0 or img[i][j][2]!=0:
          not_black_count+=1

    count_0=count_edges(canny_0)

    if (count_0/size>=p_rum):
      scarto_rumore+=1
      if(show_scarto):
        cv2.imshow(img)
        cv2.imshow(canny_0)
      continue


    count_1=count_edges(canny_1)
    single_image=all_about_image(img,canny_0,canny_1,size,(count_0-count_1),not_black_count)
    PN_good_train_images.append(single_image)
    PN_good_train_num+=1

print("Numero immagini PN good train lette e salvate: ",PN_good_train_num)
print("scartate per eccessivo rumore: ",scarto_rumore)

################ get the path for PN good test
folder="/content/drive/MyDrive/Paper/CroppedPatchesDataset/PN/test/good/"
PN_good_test_num=0
scarto_rumore=0

for images in os.listdir(folder):
  # check if the image ends with png
  if (images.endswith(".jpg")):
    print(images)
    img = cv2.imread(folder+images)  # Read image
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # Applying the Canny Edge filter
    canny_0 = cv2.Canny(img, t_lower_0, t_upper_0)
    canny_1 = cv2.Canny(img, t_lower_1, t_upper_1)

    size=img.shape[0]*img.shape[1]
    not_black_count=0
    for i in range(0,img.shape[0]):
      for j in range(0,img.shape[1]):
        if img[i][j][0]!=0 or img[i][j][1]!=0 or img[i][j][2]!=0:
          not_black_count+=1

    count_0=count_edges(canny_0)

    if (count_0/size>=p_rum):
      scarto_rumore+=1
      continue


    count_1=count_edges(canny_1)
    single_image=all_about_image(img,canny_0,canny_1,size,(count_0-count_1),not_black_count)
    PN_good_test_images.append(single_image)
    PN_good_test_num+=1

print("Numero immagini PN good test lette e salvate: ",PN_good_test_num)
print("scartate per eccessivo rumore: ",scarto_rumore)

################ get the path for RG good train
folder="/content/drive/MyDrive/Paper/CroppedPatchesDataset/RG/train/good/"
RG_good_train_num=0
scarto_rumore=0

for images in os.listdir(folder):
  # check if the image ends with png
  if (images.endswith(".jpg")):
    print(images)
    img = cv2.imread(folder+images)  # Read image
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # Applying the Canny Edge filter
    canny_0 = cv2.Canny(img, t_lower_0, t_upper_0)
    canny_1 = cv2.Canny(img, t_lower_1, t_upper_1)

    size=img.shape[0]*img.shape[1]
    not_black_count=0
    for i in range(0,img.shape[0]):
      for j in range(0,img.shape[1]):
        if img[i][j][0]!=0 or img[i][j][1]!=0 or img[i][j][2]!=0:
          not_black_count+=1

    count_0=count_edges(canny_0)

    if (count_0/size>=p_rum):
      scarto_rumore+=1
      if(show_scarto):
        cv2.imshow(img)
        cv2.imshow(canny_0)
      continue


    count_1=count_edges(canny_1)
    single_image=all_about_image(img,canny_0,canny_1,size,(count_0-count_1),not_black_count)
    RG_good_train_images.append(single_image)
    RG_good_train_num+=1

print("Numero immagini RG good train lette e salvate: ",RG_good_train_num)
print("scartate per eccessivo rumore: ",scarto_rumore)

################ get the path for PN good test
folder="/content/drive/MyDrive/Paper/CroppedPatchesDataset/RG/test/good/"
RG_good_test_num=0
scarto_rumore=0

for images in os.listdir(folder):
  # check if the image ends with png
  if (images.endswith(".jpg")):
    print(images)
    img = cv2.imread(folder+images)  # Read image
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # Applying the Canny Edge filter
    canny_0 = cv2.Canny(img, t_lower_0, t_upper_0)
    canny_1 = cv2.Canny(img, t_lower_1, t_upper_1)

    size=img.shape[0]*img.shape[1]
    not_black_count=0
    for i in range(0,img.shape[0]):
      for j in range(0,img.shape[1]):
        if img[i][j][0]!=0 or img[i][j][1]!=0 or img[i][j][2]!=0:
          not_black_count+=1

    count_0=count_edges(canny_0)

    if (count_0/size>=p_rum):
      scarto_rumore+=1
      if(show_scarto):
        cv2.imshow(img)
        cv2.imshow(canny_0)
      continue


    count_1=count_edges(canny_1)
    single_image=all_about_image(img,canny_0,canny_1,size,(count_0-count_1),not_black_count)
    RG_good_test_images.append(single_image)
    RG_good_test_num+=1

print("Numero immagini RG good test lette e salvate: ",RG_good_test_num)
print("scartate per eccessivo rumore: ",scarto_rumore)


################################### tistiche ###################################

#########PN bad train
print("PN bad train")
num=0
scarto=0
media=0
max=0
min=9999999
for i in range(0,PN_bad_train_num):
  if((PN_bad_train_images[i].not_black/PN_bad_train_images[i].size)>percentuale_contenuto):
    if(PN_bad_train_images[i].not_black)!=0:
      percentuale_edges=(PN_bad_train_images[i].edges/PN_bad_train_images[i].not_black)*100
      media+=percentuale_edges
      num+=1
      if(percentuale_edges>max): max=percentuale_edges
      if(percentuale_edges<min): min=percentuale_edges
  else:
    if(show_scarto): cv2.imshow(PN_bad_train_images[i].image)
    scarto+=1

print("Immagini in con contenuto superiore al ",percentuale_contenuto*100, " : ",num)
print("Immagini scartate: ",scarto)
print("La media edges new_grapes: ",media/num)
print("max trovato: ",max)
print("min trovato: ",min)

buckets_PN_bad_train = [0] * NUM_BINS

for i in range(0,PN_bad_train_num):
  if((PN_bad_train_images[i].not_black/PN_bad_train_images[i].size)>percentuale_contenuto):
    if(PN_bad_train_images[i].not_black)!=0:
      bucket_index = math.floor((PN_bad_train_images[i].edges / (PN_bad_train_images[i].not_black *percentuale)) *NUM_BINS)
    if bucket_index>=NUM_BINS:
      bucket_index=NUM_BINS-1
    buckets_PN_bad_train[bucket_index] += 1

print("istogramma con ",NUM_BINS," buckets",buckets_PN_bad_train)

count=0
for i in buckets_PN_bad_train:
  buckets_PN_bad_train[count]=i/num
  count+=1
print("istogramma con ",NUM_BINS," buckets normalizzato",buckets_PN_bad_train)
print("\n")

#########PN bad test
print("PN bad test")
num=0
scarto=0
media=0
max=0
min=9999999
for i in range(0,PN_bad_test_num):
  if((PN_bad_test_images[i].not_black/PN_bad_test_images[i].size)>percentuale_contenuto):
    if(PN_bad_test_images[i].not_black)!=0:
      percentuale_edges=(PN_bad_test_images[i].edges/PN_bad_test_images[i].not_black)*100
      media+=percentuale_edges
      num+=1
      if(percentuale_edges>max): max=percentuale_edges
      if(percentuale_edges<min): min=percentuale_edges
  else:
    if(show_scarto): cv2.imshow(PN_bad_test_images[i].image)
    scarto+=1

print("Immagini in con contenuto superiore al ",percentuale_contenuto*100, " : ",num)
print("Immagini scartate: ",scarto)
print("La media edges new_grapes: ",media/num)
print("max trovato: ",max)
print("min trovato: ",min)

buckets_PN_bad_test = [0] * NUM_BINS

for i in range(0,PN_bad_test_num):
  if((PN_bad_test_images[i].not_black/PN_bad_test_images[i].size)>percentuale_contenuto):
    if(PN_bad_test_images[i].not_black)!=0:
      bucket_index = math.floor((PN_bad_test_images[i].edges / (PN_bad_test_images[i].not_black *percentuale)) *NUM_BINS)
    if bucket_index>=NUM_BINS:
      bucket_index=NUM_BINS-1
    buckets_PN_bad_test[bucket_index] += 1

print("istogramma con ",NUM_BINS," buckets",buckets_PN_bad_test)

count=0
for i in buckets_PN_bad_test:
  buckets_PN_bad_test[count]=i/num
  count+=1
print("istogramma con ",NUM_BINS," buckets normalizzato",buckets_PN_bad_test)
print("\n")

#########RG bad train
print("RG bad train")
num=0
scarto=0
media=0
max=0
min=9999999
for i in range(0,RG_bad_train_num):
  if((RG_bad_train_images[i].not_black/RG_bad_train_images[i].size)>percentuale_contenuto):
    if(RG_bad_train_images[i].not_black)!=0:
      percentuale_edges=(RG_bad_train_images[i].edges/RG_bad_train_images[i].not_black)*100
      media+=percentuale_edges
      num+=1
      if(percentuale_edges>max): max=percentuale_edges
      if(percentuale_edges<min): min=percentuale_edges
  else:
    if(show_scarto): cv2.imshow(RG_bad_train_images[i].image)
    scarto+=1

print("Immagini in con contenuto superiore al ",percentuale_contenuto*100, " : ",num)
print("Immagini scartate: ",scarto)
print("La media edges new_grapes: ",media/num)
print("max trovato: ",max)
print("min trovato: ",min)

buckets_RG_bad_train = [0] * NUM_BINS

for i in range(0,RG_bad_train_num):
  if((RG_bad_train_images[i].not_black/RG_bad_train_images[i].size)>percentuale_contenuto):
    if(RG_bad_train_images[i].not_black)!=0:
      bucket_index = math.floor((RG_bad_train_images[i].edges / (RG_bad_train_images[i].not_black *percentuale)) *NUM_BINS)
    if bucket_index>=NUM_BINS:
      bucket_index=NUM_BINS-1
    buckets_RG_bad_train[bucket_index] += 1

print("istogramma con ",NUM_BINS," buckets",buckets_RG_bad_train)

count=0
for i in buckets_RG_bad_train:
  buckets_RG_bad_train[count]=i/num
  count+=1
print("istogramma con ",NUM_BINS," buckets normalizzato",buckets_RG_bad_train)
print("\n")

#########RG bad test
print("RG bad test")
num=0
scarto=0
media=0
max=0
min=9999999
for i in range(0,RG_bad_test_num):
  if((RG_bad_test_images[i].not_black/RG_bad_test_images[i].size)>percentuale_contenuto):
    if(RG_bad_test_images[i].not_black)!=0:
      percentuale_edges=(RG_bad_test_images[i].edges/RG_bad_test_images[i].not_black)*100
      media+=percentuale_edges
      num+=1
      if(percentuale_edges>max): max=percentuale_edges
      if(percentuale_edges<min): min=percentuale_edges
  else:
    if(show_scarto): cv2.imshow(RG_bad_test_images[i].image)
    scarto+=1

print("Immagini in con contenuto superiore al ",percentuale_contenuto*100, " : ",num)
print("Immagini scartate: ",scarto)
print("La media edges new_grapes: ",media/num)
print("max trovato: ",max)
print("min trovato: ",min)

buckets_RG_bad_test = [0] * NUM_BINS

for i in range(0,RG_bad_test_num):
  if((RG_bad_test_images[i].not_black/RG_bad_test_images[i].size)>percentuale_contenuto):
    if(RG_bad_test_images[i].not_black)!=0:
      bucket_index = math.floor((RG_bad_test_images[i].edges / (RG_bad_test_images[i].not_black *percentuale)) *NUM_BINS)
    if bucket_index>=NUM_BINS:
      bucket_index=NUM_BINS-1
    buckets_RG_bad_test[bucket_index] += 1

print("istogramma con ",NUM_BINS," buckets",buckets_RG_bad_test)

count=0
for i in buckets_RG_bad_test:
  buckets_RG_bad_test[count]=i/num
  count+=1
print("istogramma con ",NUM_BINS," buckets normalizzato",buckets_RG_bad_test)
print("\n")

#########PN good train
print("PN good train")
num=0
scarto=0
media=0
max=0
min=9999999
for i in range(0,PN_good_train_num):
  if((PN_good_train_images[i].not_black/PN_good_train_images[i].size)>percentuale_contenuto):
    if(PN_good_train_images[i].not_black)!=0:
      percentuale_edges=(PN_good_train_images[i].edges/PN_good_train_images[i].not_black)*100
      media+=percentuale_edges
      num+=1
      if(percentuale_edges>max): max=percentuale_edges
      if(percentuale_edges<min): min=percentuale_edges
  else:
    if(show_scarto): cv2.imshow(PN_good_train_images[i].image)
    scarto+=1

print("Immagini in con contenuto superiore al ",percentuale_contenuto*100, " : ",num)
print("Immagini scartate: ",scarto)
print("La media edges new_grapes: ",media/num)
print("max trovato: ",max)
print("min trovato: ",min)

buckets_PN_good_train = [0] * NUM_BINS

for i in range(0,PN_good_train_num):
  if((PN_good_train_images[i].not_black/PN_good_train_images[i].size)>percentuale_contenuto):
    if(PN_good_train_images[i].not_black)!=0:
      bucket_index = math.floor((PN_good_train_images[i].edges / (PN_good_train_images[i].not_black *percentuale)) *NUM_BINS)
    if bucket_index>=NUM_BINS:
      bucket_index=NUM_BINS-1
    buckets_PN_good_train[bucket_index] += 1

print("istogramma con ",NUM_BINS," buckets",buckets_PN_good_train)

count=0
for i in buckets_PN_good_train:
  buckets_PN_good_train[count]=i/num
  count+=1
print("istogramma con ",NUM_BINS," buckets normalizzato",buckets_PN_good_train)
print("\n")

#########PN good test
print("PN good test")
num=0
scarto=0
media=0
max=0
min=9999999
for i in range(0,PN_good_test_num):
  if((PN_good_test_images[i].not_black/PN_good_test_images[i].size)>percentuale_contenuto):
    if(PN_good_test_images[i].not_black)!=0:
      percentuale_edges=(PN_good_test_images[i].edges/PN_good_test_images[i].not_black)*100
      media+=percentuale_edges
      num+=1
      if(percentuale_edges>max): max=percentuale_edges
      if(percentuale_edges<min): min=percentuale_edges
  else:
    if(show_scarto): cv2.imshow(PN_good_test_images[i].image)
    scarto+=1

print("Immagini in con contenuto superiore al ",percentuale_contenuto*100, " : ",num)
print("Immagini scartate: ",scarto)
print("La media edges new_grapes: ",media/num)
print("max trovato: ",max)
print("min trovato: ",min)

buckets_PN_good_test = [0] * NUM_BINS

for i in range(0,PN_good_test_num):
  if((PN_good_test_images[i].not_black/PN_good_test_images[i].size)>percentuale_contenuto):
    if(PN_good_test_images[i].not_black)!=0:
      bucket_index = math.floor((PN_good_test_images[i].edges / (PN_good_test_images[i].not_black *percentuale)) *NUM_BINS)
    if bucket_index>=NUM_BINS:
      bucket_index=NUM_BINS-1
    buckets_PN_good_test[bucket_index] += 1

print("istogramma con ",NUM_BINS," buckets",buckets_PN_good_test)

count=0
for i in buckets_PN_good_test:
  buckets_PN_good_test[count]=i/num
  count+=1
print("istogramma con ",NUM_BINS," buckets normalizzato",buckets_PN_good_test)
print("\n")

#########RG good train
print("RG good train")
num=0
scarto=0
media=0
max=0
min=9999999
for i in range(0,RG_good_train_num):
  if((RG_good_train_images[i].not_black/RG_good_train_images[i].size)>percentuale_contenuto):
    if(RG_good_train_images[i].not_black)!=0:
      percentuale_edges=(RG_good_train_images[i].edges/RG_good_train_images[i].not_black)*100
      media+=percentuale_edges
      num+=1
      if(percentuale_edges>max): max=percentuale_edges
      if(percentuale_edges<min): min=percentuale_edges
  else:
    if(show_scarto): cv2.imshow(RG_good_train_images[i].image)
    scarto+=1

print("Immagini in con contenuto superiore al ",percentuale_contenuto*100, " : ",num)
print("Immagini scartate: ",scarto)
print("La media edges new_grapes: ",media/num)
print("max trovato: ",max)
print("min trovato: ",min)

buckets_RG_good_train = [0] * NUM_BINS

for i in range(0,RG_good_train_num):
  if((RG_good_train_images[i].not_black/RG_good_train_images[i].size)>percentuale_contenuto):
    if(RG_good_train_images[i].not_black)!=0:
      bucket_index = math.floor((RG_good_train_images[i].edges / (RG_good_train_images[i].not_black *percentuale)) *NUM_BINS)
    if bucket_index>=NUM_BINS:
      bucket_index=NUM_BINS-1
    buckets_RG_good_train[bucket_index] += 1

print("istogramma con ",NUM_BINS," buckets",buckets_RG_good_train)

count=0
for i in buckets_RG_good_train:
  buckets_RG_good_train[count]=i/num
  count+=1
print("istogramma con ",NUM_BINS," buckets normalizzato",buckets_RG_good_train)
print("\n")

#########RG good test
print("RG good test")
num=0
scarto=0
media=0
max=0
min=9999999
for i in range(0,RG_good_test_num):
  if((RG_good_test_images[i].not_black/RG_good_test_images[i].size)>percentuale_contenuto):
    if(RG_good_test_images[i].not_black)!=0:
      percentuale_edges=(RG_good_test_images[i].edges/RG_good_test_images[i].not_black)*100
      media+=percentuale_edges
      num+=1
      if(percentuale_edges>max): max=percentuale_edges
      if(percentuale_edges<min): min=percentuale_edges
  else:
    if(show_scarto): cv2.imshow(RG_good_test_images[i].image)
    scarto+=1

print("Immagini in con contenuto superiore al ",percentuale_contenuto*100, " : ",num)
print("Immagini scartate: ",scarto)
print("La media edges new_grapes: ",media/num)
print("max trovato: ",max)
print("min trovato: ",min)

buckets_RG_good_test = [0] * NUM_BINS

for i in range(0,RG_good_test_num):
  if((RG_good_test_images[i].not_black/RG_good_test_images[i].size)>percentuale_contenuto):
    if(RG_good_test_images[i].not_black)!=0:
      bucket_index = math.floor((RG_good_test_images[i].edges / (RG_good_test_images[i].not_black *percentuale)) *NUM_BINS)
    if bucket_index>=NUM_BINS:
      bucket_index=NUM_BINS-1
    buckets_RG_good_test[bucket_index] += 1

print("istogramma con ",NUM_BINS," buckets",buckets_RG_good_test)

count=0
for i in buckets_RG_good_test:
  buckets_RG_good_test[count]=i/num
  count+=1
print("istogramma con ",NUM_BINS," buckets normalizzato",buckets_RG_good_test)
print("\n")



################################## Plotting PN ##################################

# Extracting x and y values for the plot
x_values = np.arange(0,len(buckets_PN_bad_train))
# Width of the bars
bar_width = 0.35
# Plotting the bar graphs for both arrays with adjusted x-coordinates
plt.bar(x_values - bar_width/2, buckets_PN_bad_train, width=bar_width, label='Bad',color='red')
plt.bar(x_values + bar_width/2, buckets_PN_good_train, width=bar_width, label='Good',color='green')
plt.title('PN train')
plt.xlabel('Buckets')
plt.ylabel('Percentage')
plt.legend()
plt.show()

#searching for nice cuts

max_diff=0
best_bucket=0
best_taken_good=0
best_taken_bad=0

cut_90=False
bucket_90=0
taken_good_90=0
taken_bad_90=0

cut_99=False
bucket_99=0
taken_good_99=0
taken_bad_99=0

good_taken=0
bad_taken=0
for i in x_values:
  good_taken+=buckets_PN_good_train[i]
  bad_taken+=buckets_PN_bad_train[i]
  diff=good_taken-bad_taken
  if max_diff<diff:
    max_diff=diff
    best_bucket=i
    best_taken_good=good_taken
    best_taken_bad=bad_taken
  if good_taken>=0.9 and not cut_90:
    cut_90=True
    bucket_90=i
    taken_good_90=good_taken
    taken_bad_90=bad_taken
  if good_taken>=0.99 and not cut_99:
    cut_99=True
    bucket_99=i
    taken_good_99=good_taken
    taken_bad_99=bad_taken



print("Trovata diff massima di ",max_diff," al bucket ",best_bucket)
print("Percentuale buoni individuati al bucket migliore",best_taken_good)
print("Percentuale cattivi individuati al bucket migliore",best_taken_bad)
print("Raggiunti i primo 90% degli acini buoni al bucket ",bucket_90)
print("Percentuale buoni individuati al 90%",taken_good_90)
print("Percentuale cattivi individuati al 90%",taken_bad_90)
print("Raggiunti i primo 99% degli acini buoni al bucket ",bucket_99)
print("Percentuale buoni individuati al 99%",taken_good_99)
print("Percentuale cattivi individuati al 99%",taken_bad_99)


#find the best accuracy cut
PN_best_TP_acc=0
PN_best_TN_acc=0
PN_best_FP_acc=0
PN_best_FN_acc=0
PN_cut_acc=0
PN_accuracy=0

PN_best_TP_pre=0
PN_best_TN_pre=0
PN_best_FP_pre=0
PN_best_FN_pre=0
PN_cut_pre=0
PN_precision=0

PN_best_TP_f1=0
PN_best_TN_f1=0
PN_best_FP_f1=0
PN_best_FN_f1=0
PN_cut_f1=0
PN_f1_score=0

for i in range(0,NUM_BINS):
  TP=0
  TN=0
  FP=0
  FN=0
  for j in range(0,i):
    TN+=buckets_PN_good_train[j]
    FN+=buckets_PN_bad_train[j]
  for k in range(i,NUM_BINS):
    FP+=buckets_PN_good_train[k]
    TP+=buckets_PN_bad_train[k]

  if (TP+TN)/(TP+TN+FP+FN)>PN_accuracy:
    PN_accuracy=(TP+TN)/(TP+TN+FP+FN)
    PN_best_TP_acc=TP
    PN_best_TN_acc=TN
    PN_best_FP_acc=FP
    PN_best_FN_acc=FN
    PN_cut_acc=i*(percentuale*100/NUM_BINS)
  if (TP+(1/2)*(FP+FN))>0 and (TP)/(TP+(1/2)*(FP+FN))>PN_f1_score:
    PN_f1_score=(TP)/(TP+(1/2)*(FP+FN))
    PN_best_TP_f1=TP
    PN_best_TN_f1=TN
    PN_best_FP_f1=FP
    PN_best_FN_f1=FN
    PN_cut_f1=i*(percentuale*100/NUM_BINS)
  if (TP+FP)>0 and (TP)/(TP+FP)>PN_precision:
    PN_precision=(TP)/(TP+FP)
    PN_best_TP_pre=TP
    PN_best_TN_pre=TN
    PN_best_FP_pre=FP
    PN_best_FN_pre=FN
    PN_cut_pre=i*(percentuale*100/NUM_BINS)

print('il miglior percentuale di taglio per accuracy: ',PN_cut_acc,'%')
print('accuracy ottenuta: ',PN_accuracy)
print('TP: ',PN_best_TP_acc)
print('TN: ',PN_best_TN_acc)
print('FP: ',PN_best_FP_acc)
print('FN: ',PN_best_FN_acc)

print('il miglior percentuale di taglio per f1_score: ',PN_cut_f1,'%')
print('f1_score ottenuto: ',PN_f1_score)
print('TP: ',PN_best_TP_f1)
print('TN: ',PN_best_TN_f1)
print('FP: ',PN_best_FP_f1)
print('FN: ',PN_best_FN_f1)

print('il miglior percentuale di taglio per precision: ',PN_cut_pre,'%')
print('precision ottenuta: ',PN_precision)
print('TP: ',PN_best_TP_pre)
print('TN: ',PN_best_TN_pre)
print('FP: ',PN_best_FP_pre)
print('FN: ',PN_best_FN_pre)


################################## Plotting RG ##################################


# Extracting x and y values for the plot
x_values = np.arange(0,len(buckets_RG_bad_train))

# Width of the bars
bar_width = 0.35

# Plotting the bar graphs for both arrays with adjusted x-coordinates
plt.bar(x_values - bar_width/2, buckets_RG_bad_train, width=bar_width, label='Bad',color='red')
plt.bar(x_values + bar_width/2, buckets_RG_good_train, width=bar_width, label='Good',color='green')

plt.title('RG train')
# Adding labels to the axes
plt.xlabel('Buckets')
plt.ylabel('Percentage')

# Displaying the plot
plt.legend()
plt.show()


max_diff=0
best_bucket=0
best_taken_good=0
best_taken_bad=0

cut_90=False
bucket_90=0
taken_good_90=0
taken_bad_90=0

cut_99=False
bucket_99=0
taken_good_99=0
taken_bad_99=0

good_taken=0
bad_taken=0
for i in x_values:
  good_taken+=buckets_RG_good_train[i]
  bad_taken+=buckets_RG_bad_train[i]
  diff=good_taken-bad_taken
  if max_diff<diff:
    max_diff=diff
    best_bucket=i
    best_taken_good=good_taken
    best_taken_bad=bad_taken
  if good_taken>=0.9 and not cut_90:
    cut_90=True
    bucket_90=i
    taken_good_90=good_taken
    taken_bad_90=bad_taken
  if good_taken>=0.99 and not cut_99:
    cut_99=True
    bucket_99=i
    taken_good_99=good_taken
    taken_bad_99=bad_taken



print("Trovata diff massima di ",max_diff," al bucket ",best_bucket)
print("Percentuale buoni individuati al bucket migliore",best_taken_good)
print("Percentuale cattivi individuati al bucket migliore",best_taken_bad)
print("Raggiunti i primo 90% degli acini buoni al bucket ",bucket_90)
print("Percentuale buoni individuati al 90%",taken_good_90)
print("Percentuale cattivi individuati al 90%",taken_bad_90)
print("Raggiunti i primo 99% degli acini buoni al bucket ",bucket_99)
print("Percentuale buoni individuati al 99%",taken_good_99)
print("Percentuale cattivi individuati al 99%",taken_bad_99)

#find the best accuracy cut
RG_best_TP_acc=0
RG_best_TN_acc=0
RG_best_FP_acc=0
RG_best_FN_acc=0
RG_cut_acc=0
RG_accuracy=0

RG_best_TP_pre=0
RG_best_TN_pre=0
RG_best_FP_pre=0
RG_best_FN_pre=0
RG_cut_pre=0
RG_precision=0

RG_best_TP_f1=0
RG_best_TN_f1=0
RG_best_FP_f1=0
RG_best_FN_f1=0
RG_cut_f1=0
RG_f1_score=0

for i in range(0,NUM_BINS):
  TP=0
  TN=0
  FP=0
  FN=0
  for j in range(0,i):
    TN+=buckets_RG_good_train[j]
    FN+=buckets_RG_bad_train[j]
  for k in range(i,NUM_BINS):
    FP+=buckets_RG_good_train[k]
    TP+=buckets_RG_bad_train[k]

  if (TP+TN)/(TP+TN+FP+FN)>RG_accuracy:
    RG_accuracy=(TP+TN)/(TP+TN+FP+FN)
    RG_best_TP_acc=TP
    RG_best_TN_acc=TN
    RG_best_FP_acc=FP
    RG_best_FN_acc=FN
    RG_cut_acc=i*(percentuale*100/NUM_BINS)
  if (TP+(1/2)*(FP+FN))>0 and (TP)/(TP+(1/2)*(FP+FN))>RG_f1_score:
    RG_f1_score=(TP)/(TP+(1/2)*(FP+FN))
    RG_best_TP_f1=TP
    RG_best_TN_f1=TN
    RG_best_FP_f1=FP
    RG_best_FN_f1=FN
    RG_cut_f1=i*(percentuale*100/NUM_BINS)
  if (TP+FP)>0 and (TP)/(TP+FP)>RG_precision:
    RG_precision=(TP)/(TP+FP)
    RG_best_TP_pre=TP
    RG_best_TN_pre=TN
    RG_best_FP_pre=FP
    RG_best_FN_pre=FN
    RG_cut_pre=i*(percentuale*100/NUM_BINS)

print('il miglior percentuale di taglio per accuracy: ',RG_cut_acc)
print('accuracy ottenuta: ',RG_accuracy)
print('TP: ',RG_best_TP_acc)
print('TN: ',RG_best_TN_acc)
print('FP: ',RG_best_FP_acc)
print('FN: ',RG_best_FN_acc)

print('il miglior percentuale di taglio per f1_score: ',RG_cut_f1)
print('f1_score ottenuto: ',RG_f1_score)
print('TP: ',RG_best_TP_f1)
print('TN: ',RG_best_TN_f1)
print('FP: ',RG_best_FP_f1)
print('FN: ',RG_best_FN_f1)

print('il miglior percentuale di taglio per precision: ',RG_cut_pre)
print('precision ottenuta: ',RG_precision)
print('TP: ',RG_best_TP_pre)
print('TN: ',RG_best_TN_pre)
print('FP: ',RG_best_FP_pre)
print('FN: ',RG_best_FN_pre)


############################### Some images

### PN

cut_perc_PN= PN_cut_acc #chose one {PN_cut_pre, PN_cut_acc, PN_cut_f1} or a number from 0 to 100
count=0
print('percentuale considerata: ',cut_perc_PN)

print('FALSE NEGATIVE')
for i in range(0,PN_bad_test_num):
  if((PN_bad_test_images[i].not_black/PN_bad_test_images[i].size)>percentuale_contenuto):
    if(PN_bad_test_images[i].not_black!=0):
      percentuale_edges=(PN_bad_test_images[i].edges/PN_bad_test_images[i].not_black)*100
      if percentuale_edges<=cut_perc_PN:
        if count<10:
          cv2.imshow(PN_bad_test_images[i].image)
          cv2.imshow(PN_bad_test_images[i].canny_large)
          cv2.imshow(PN_bad_test_images[i].canny_small)
        count+=1
        print("percentuale di edges ",(PN_bad_test_images[i].edges/PN_bad_test_images[i].not_black)*100)

print("immagini guaste considerate buone ", count)

print('TRUE POSITIVE')
for i in range(0,PN_bad_test_num):
  if((PN_bad_test_images[i].not_black/PN_bad_test_images[i].size)>percentuale_contenuto):
    if(PN_bad_test_images[i].not_black!=0):
      percentuale_edges=(PN_bad_test_images[i].edges/PN_bad_test_images[i].not_black)*100
      if percentuale_edges>cut_perc_PN:
        if count<10:
          cv2.imshow(PN_bad_test_images[i].image)
          cv2.imshow(PN_bad_test_images[i].canny_large)
          cv2.imshow(PN_bad_test_images[i].canny_small)
        count+=1
        print("percentuale di edges ",(PN_bad_test_images[i].edges/PN_bad_test_images[i].not_black)*100)

print("immagini guaste considerate guaste ", count)

print('FALSE POSITIVE')
for i in range(0,PN_good_test_num):
  if((PN_good_test_images[i].not_black/PN_good_test_images[i].size)>percentuale_contenuto):
    if(PN_good_test_images[i].not_black!=0):
      percentuale_edges=(PN_good_test_images[i].edges/PN_good_test_images[i].not_black)*100
      if percentuale_edges>cut_perc_PN:
        if count<10:
          cv2.imshow(PN_good_test_images[i].image)
          cv2.imshow(PN_good_test_images[i].canny_large)
          cv2.imshow(PN_good_test_images[i].canny_small)
        count+=1
        print("percentuale di edges ",(PN_good_test_images[i].edges/PN_good_test_images[i].not_black)*100)

print("immagini buone considerate guaste ", count)

print('TRUE NEGATIVE')
for i in range(0,PN_good_test_num):
  if((PN_good_test_images[i].not_black/PN_good_test_images[i].size)>percentuale_contenuto):
    if(PN_good_test_images[i].not_black!=0):
      percentuale_edges=(PN_good_test_images[i].edges/PN_good_test_images[i].not_black)*100
      if percentuale_edges<=cut_perc_PN:
        if count<10:
          cv2.imshow(PN_good_test_images[i].image)
          cv2.imshow(PN_good_test_images[i].canny_large)
          cv2.imshow(PN_good_test_images[i].canny_small)
        count+=1
        print("percentuale di edges ",(PN_good_test_images[i].edges/PN_good_test_images[i].not_black)*100)

print("immagini buone considerate buone ", count)


### RG
cut_perc_RG= RG_cut_acc #chose one {RG_cut_pre, RG_cut_acc, RG_cut_f1} or a number from 0 to 100
count=0
print('percentuale considerata: ',cut_perc_RG)

print('FALSE NEGATIVE')
for i in range(0,PN_bad_test_num):
  if((PN_bad_test_images[i].not_black/PN_bad_test_images[i].size)>percentuale_contenuto):
    if(PN_bad_test_images[i].not_black!=0):
      percentuale_edges=(PN_bad_test_images[i].edges/PN_bad_test_images[i].not_black)*100
      if percentuale_edges<=cut_perc_RG:
        if count<10:
          cv2.imshow(PN_bad_test_images[i].image)
          cv2.imshow(PN_bad_test_images[i].canny_large)
          cv2.imshow(PN_bad_test_images[i].canny_small)
        count+=1
        print("percentuale di edges ",(PN_bad_test_images[i].edges/PN_bad_test_images[i].not_black)*100)

print("immagini guaste considerate buone ", count)

print('TRUE POSITIVE')
for i in range(0,PN_bad_test_num):
  if((PN_bad_test_images[i].not_black/PN_bad_test_images[i].size)>percentuale_contenuto):
    if(PN_bad_test_images[i].not_black!=0):
      percentuale_edges=(PN_bad_test_images[i].edges/PN_bad_test_images[i].not_black)*100
      if percentuale_edges>cut_perc_RG:
        if count<10:
          cv2.imshow(PN_bad_test_images[i].image)
          cv2.imshow(PN_bad_test_images[i].canny_large)
          cv2.imshow(PN_bad_test_images[i].canny_small)
        count+=1
        print("percentuale di edges ",(PN_bad_test_images[i].edges/PN_bad_test_images[i].not_black)*100)

print("immagini guaste considerate guaste ", count)

print('FALSE POSITIVE')
for i in range(0,PN_good_test_num):
  if((PN_good_test_images[i].not_black/PN_good_test_images[i].size)>percentuale_contenuto):
    if(PN_good_test_images[i].not_black!=0):
      percentuale_edges=(PN_good_test_images[i].edges/PN_good_test_images[i].not_black)*100
      if percentuale_edges>cut_perc_RG:
        if count<10:
          cv2.imshow(PN_good_test_images[i].image)
          cv2.imshow(PN_good_test_images[i].canny_large)
          cv2.imshow(PN_good_test_images[i].canny_small)
        count+=1
        print("percentuale di edges ",(PN_good_test_images[i].edges/PN_good_test_images[i].not_black)*100)

print("immagini buone considerate guaste ", count)

print('TRUE NEGATIVE')
for i in range(0,PN_good_test_num):
  if((PN_good_test_images[i].not_black/PN_good_test_images[i].size)>percentuale_contenuto):
    if(PN_good_test_images[i].not_black!=0):
      percentuale_edges=(PN_good_test_images[i].edges/PN_good_test_images[i].not_black)*100
      if percentuale_edges<=cut_perc_RG:
        if count<10:
          cv2.imshow(PN_good_test_images[i].image)
          cv2.imshow(PN_good_test_images[i].canny_large)
          cv2.imshow(PN_good_test_images[i].canny_small)
        count+=1
        print("percentuale di edges ",(PN_good_test_images[i].edges/PN_good_test_images[i].not_black)*100)

print("immagini buone considerate buone ", count)


