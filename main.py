#!/usr/bin/env python
# coding: utf-8

# # Unsupervised face recognition using a video clip

# ### Downloading input video and importing libraries and models

# In[ ]:


pip install face_recognition


# In[ ]:


get_ipython().system('wget https://www.dropbox.com/s/vmkd3x36lbpdkms/blackpink.mp4?dl=1 -O blackpink.mp4')


# In[ ]:


import os
import cv2
import numpy as np
from imutils import paths
from sklearn.cluster import DBSCAN
from imutils import build_montages
import face_recognition
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import subprocess
from sklearn.preprocessing import StandardScaler
from matplotlib.pyplot import figure


# In[ ]:


# Define paths
prototxt_path = os.path.abspath('model/deploy.prototxt.txt')
caffemodel_path = os.path.abspath('model/weights.caffemodel')

# Read the model
model = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)


# ### Preparing the environment

# In[35]:


if not os.path.exists('images'):
	print("New directory created")
	os.makedirs('images')


# In[36]:


if not os.path.exists('faces'):
	print("New directory created")
	os.makedirs('faces')


# In[71]:


#Extracting 60 frames from the input video
get_ipython().system('ffmpeg -i "blackpink.mp4" -r 1 "images/frame%04d.jpg"')


# ### Extracting faces and writing them to disk

# In[38]:


# Loop through all images and save images with marked faces
for file in tqdm(os.listdir('images')):
    file_name, file_extension = os.path.splitext(file)
    image = cv2.imread('images/' + file)
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    model.setInput(blob)
    detections = model.forward()
    for i in range(0, detections.shape[2]):
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        confidence = detections[0, 0, i, 2]
        if (confidence > 0.5):
            cv2.rectangle(image, (startX, startY), (endX, endY), (255, 255, 255), 2)
    # Identify each face
    for i in range(0, detections.shape[2]):
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        confidence = detections[0, 0, i, 2]
        if (confidence > 0.5):
            frame = image[startY:endY, startX:endX]
            try:
              cv2.imwrite('faces/' + str(i) + '_' + file, frame)
            except:
              None


# ### Generating encodings and clustering

# In[40]:


images = os.listdir("faces")
base_dir = "faces/"
data = []
for (i, imagePath) in enumerate(tqdm(images)):
    image = cv2.imread(base_dir+imagePath)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb,model="cnn")
    encodings = face_recognition.face_encodings(rgb, boxes)
    d = [{"imagePath": imagePath, "loc": box, "encoding": enc}
        for (box, enc) in zip(boxes, encodings)]
    data.extend(d)
print("[INFO] serializing encodings...")
f = open("model/encodings.pickle", "wb")
f.write(pickle.dumps(data))
f.close()


# In[41]:


print("[INFO] loading encodings...")
data = pickle.loads(open("model/encodings.pickle", "rb").read())
data = np.array(data)
encodings = [d["encoding"] for d in data]


# In[42]:


print("[INFO] clustering...")
clt = DBSCAN(metric="euclidean", n_jobs=4, min_samples=10,eps=0.32)
clt.fit(encodings)
# determine the total number of unique faces found in the dataset
labelIDs = np.unique(clt.labels_)
numUniqueFaces = len(np.where(labelIDs > -1)[0])
print("[INFO] # unique faces: {}".format(numUniqueFaces))


# ### Plotting clusters and montages

# In[ ]:


def plot(X,db):
    ss = StandardScaler()
    X = ss.fit_transform(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    unique_labels = set(labels)
    figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = [0, 0, 0, 1]
        class_member_mask = (labels == k)

        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                markeredgecolor='k', markersize=14,label=names[k] if k>=0 else "Noise")

        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                markeredgecolor='k', markersize=6)
    plt.legend(loc="upper left")
    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()


# In[47]:


plot(encodings,clt)


# In[ ]:


# loop over the unique face integers
montages = {}
for labelID in labelIDs:
  idxs = np.where(clt.labels_ == labelID)[0]
  idxs = np.random.choice(idxs, size=min(25, len(idxs)),replace=False)
  faces = []
  for i in idxs:
    image = cv2.imread("faces/"+data[i]["imagePath"])
    (top, right, bottom, left) = data[i]["loc"]
    face = image[top:bottom, left:right]
    # force resize the face ROI to 96x96 and then add it to the
    # faces montage list
    face = cv2.resize(face, (96, 96))
    faces.append(face)
  montages[labelID] = build_montages(faces, (96, 96), (5, 5))[0]


# In[50]:


plt.imshow(montages[0])


# In[51]:


plt.imshow(montages[1])


# In[52]:


plt.imshow(montages[2])


# ### Building a labeled dataset that can be used for other classifiers

# In[ ]:


names=["Jisoo","Jennie","Rose"]
dataset = {}


# In[ ]:


for i,name in enumerate(names):
  path='data/'+name+"/"
  try:
    os.makedirs(path)
  except:
    None
  idxs = np.where(clt.labels_ == i)[0]
  encodings=[]
  for id in idxs:
    source = "faces/"+data[id]["imagePath"]
    target = path+str(id)+".jpg"
    subprocess.run(["cp",source,target])
    encodings.append(data[id]["encoding"])
  dataset[name]=encodings


# ### Classification function

# In[ ]:


def label(imagepath):
  unknown_image = face_recognition.load_image_file(imagepath)
  face_locations = face_recognition.face_locations(unknown_image)
  face_encodings = face_recognition.face_encodings(unknown_image, face_locations)
  pil_image = Image.fromarray(unknown_image)
  draw = ImageDraw.Draw(pil_image)
  for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
      name = "Unknown"
      mins=[]
      matches=False
      for member in dataset:
          face_distances = face_recognition.face_distance(dataset[member], face_encoding)
          mins.append(np.min(face_distances))
      best_match_index = np.argmin(mins)
      if np.min(face_distances)<0.6:
        name = names[best_match_index]
      text_width, text_height = draw.textsize(name)
      draw.rectangle(((left , bottom ), (right, bottom+10)), fill=(0, 0, 255), outline=(0, 0, 255))
      draw.text((left + 6, bottom - text_height+10 ), name, fill=(255, 255, 255, 255))
  display(pil_image)


# ## Results

# In[62]:


#Multiple faces in one image
label("drive/My Drive/vision/data/Untitled.png")


# In[63]:


#Crying Rose :(
label("drive/My Drive/vision/data/crying.png")


# In[64]:


#Facemasked Rose
label("drive/My Drive/vision/data/face_mask.jpg")


# In[65]:


#Laughing Jisoo :D
label("drive/My Drive/vision/data/smiling.jpg")


# In[66]:


#Jisoo with sunglasses B)
label("drive/My Drive/vision/data/sunglasses.jpg")


# In[67]:


#Angry Jennie
label("drive/My Drive/vision/data/angry.jpg")


# In[68]:


#Unknown Katy Perry
label("drive/My Drive/vision/data/katy.jpg")


# In[ ]:




