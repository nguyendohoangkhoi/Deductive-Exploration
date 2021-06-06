import cv2
from sklearn.cluster import KMeans
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
c_data_path = "data/"

def get_feature(img):
    intensity = img.sum(axis=1)
    intensity = intensity.sum(axis=0) / (255 * img.shape[0] * img.shape[1])
    return intensity

def calc_distance(x1,y1,a,b,c):
    d = abs((a*x1+b*y1+c))/(math.sqrt(a*a+b*b))
    return d
def find_abc (distortions,K):
    len_K = len(K)-1
    a = distortions[0]-distortions[len_K]
    b = K[len_K]-K[0]
    c1 = K[0]*distortions[len_K]
    c2 = K[len_K]*distortions[0]
    c = c1-c2
    return a,b,c
def find_optimal_K(distortions,K,restrict_value):
    len_K = len(K)-1
    distance_of_points_from_line = []
    a,b,c = find_abc(distortions,K)
    for k in range(len_K):
        distance_of_points_from_line.append(calc_distance(K[k],distortions[k],a,b,c))
    max_distance = max(distance_of_points_from_line)
    restrict_distance = max_distance-restrict_value
    temp = max(np.array(distance_of_points_from_line)[(np.array(distance_of_points_from_line)<restrict_distance)])
    return distance_of_points_from_line.index(temp)+1

def load_data(data_path=c_data_path):
    X = []
    L = []
    for file in os.listdir(data_path):
        c_x = get_feature(cv2.imread(os.path.join(data_path, file)))
        X.append(c_x)
        L.append(file)
    # L = np.array(L)
    return X,L
# Print the shape of the imag
X,Y = load_data()
distortions = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k,random_state=0)
    kmeanModel.fit(X)
    distortions.append(kmeanModel.inertia_)
optimal_k = find_optimal_K(distortions, K, 1.5)
kmeanModel = KMeans(n_clusters=optimal_k, random_state=0).fit(X)
try:
    with open('kmeans_model.pickle', 'rb') as handle:
        kmeanModel = pickle.load(handle)
except:
    with open('kmeans_model.pickle', 'wb') as handle:
        pickle.dump(kmeanModel, handle, protocol=pickle.HIGHEST_PROTOCOL)
kmeanModel.fit(X)
idx = np.argsort(kmeanModel.cluster_centers_.sum(axis=1))
lut = np.zeros_like(idx)
lut[idx] = np.arange(optimal_k)
for i in range(len(kmeanModel.labels_)):
    print(lut[kmeanModel.labels_[i]]," - ", Y[i])
src_img = []
for file in os.listdir("data"):
    print(file)
    path_img = os.path.join("data", file)
    img = cv2.imread(path_img)
    c_x = get_feature(img)
    X.append(c_x)
    src_img.append(path_img)
for i in range(len(X)):
    label = lut[kmeanModel.predict([X[i]])]
    img = cv2.imread("{}".format(src_img[i]))
    if not os.path.exists("classified_data/{}".format(label)):
        os.makedirs("classified_data/{}".format(label))
    print("{}".format(src_img[i]))
    cv2.imwrite("classified_data/{}/{}.png".format(label,i),img)
#    L.append(file)
# print(kmeans.predict([X[0]]))
# print(kmeans.predict([X[1]]))
# print(kmeans.predict([X[2]]))