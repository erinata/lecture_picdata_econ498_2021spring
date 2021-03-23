import imageio
import numpy
import os
import glob
import pandas

from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize

import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as pyplot

def getrgb(filepath):
	imimage = imageio.imread(filepath, pilmode='RGB')
	imimage_process = imimage/255
	imimage_process = imimage_process.sum(axis=0).sum(axis=0)/imimage_process.shape[0]*imimage_process.shape[1]
	imimage_process = imimage_process/numpy.linalg.norm(imimage_process, ord=None)
	return imimage_process


# image_one = getrgb('data/pic01.jpeg')
# print(image_one)

dataset=pandas.DataFrame()

for filepath in glob.glob('data/*'):
	image_features = pandas.DataFrame(getrgb(filepath))
	image_features = pandas.DataFrame.transpose(image_features)
	image_features['path'] = filepath
	dataset = pandas.concat([dataset, image_features])

dataset = dataset.sort_values(by=['path'])

print(dataset)

data = dataset.iloc[:,0:3]
print(data)
data = pandas.DataFrame(normalize(data))


pyplot.title("Dendrogram")
dendrogram_object = shc.dendrogram(shc.linkage(data, method='ward'))
pyplot.savefig("den.png")
pyplot.close()


machine = AgglomerativeClustering(n_clusters=4, affinity="euclidean", linkage="ward")
results = machine.fit_predict(data)

dataset['results'] = results



print(dataset)



