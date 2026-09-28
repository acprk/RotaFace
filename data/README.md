# Data

`split.txt` lists the gallery indices (line 1), probe indices (line 2) and the gallery position of each probe's identity (line 3)
into a 10,000 x 512 array of FaceNet (InceptionResNetV1, VGGFace2) embeddings of LFW images, stored row-major as little-endian
float32 in `lfw_emb.f32` (not redistributed). To regenerate it, align LFW faces with MTCNN, embed them with `facenet-pytorch`,
L2-normalise, keep the first 10,000 images in `sklearn.datasets.fetch_lfw_people(min_faces_per_person=2)` order, and write
zero rows for images without a detected face (119 in our run). Set `RF_DATA=/path/to/lfw_emb.f32` or place the file here.
