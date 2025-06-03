# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 13:56:42 2024

@author: mohamed.niang
"""
import numpy as np
from osgeo import gdal
#import ogr
from skimage import exposure
from skimage.segmentation import quickshift
import geopandas as gpd
import numpy as np
import scipy

from osgeo import gdal
from osgeo import ogr
naip_fn = 'C:/TM/imExt/New_24_11_14/Stack_ZE.tif'
 
driverTiff = gdal.GetDriverByName('GTiff')
naip_ds = gdal.Open(naip_fn)
#nan_array[array == -999] = np.nan
nbands = naip_ds.RasterCount
band_data = []

for i in range(1, nbands+1):
    band = naip_ds.GetRasterBand(i).ReadAsArray()
    band_data.append(band)
band_data = np.dstack(band_data)
img = exposure.rescale_intensity(band_data)








#naip_fn2 = 'C:/TM/imExt/New_24_11_14/SegmentsValue.tif'
naip_fn2 = 'C:/TM/imExt/New_24_11_14/SegmentsValueM.tif'
driverTiff = gdal.GetDriverByName('GTiff')
naip_ds2 = gdal.Open(naip_fn2)
type(naip_ds2)
band = naip_ds2.GetRasterBand(1)
segments = np.array(band.ReadAsArray())

def segment_features(segment_pixels):
    features = []
    npixels, nbands = segment_pixels.shape
    for b in range(nbands):
        stats = scipy.stats.describe(segment_pixels[:, b])
        band_stats = list(stats.minmax) + list(stats)[2:]
        if npixels == 1:
            # in this case the variance = nan, change it 0.0
            band_stats[3] = 0.0
        features += band_stats
    return features


segment_ids = np.unique(segments)
objects = []
object_ids = []
for id in segment_ids:
    segment_pixels = img[segments == id]
    object_features = segment_features(segment_pixels)
    objects.append(object_features)
    object_ids.append(id)
 
    
# read shapefile to geopandas geodataframe
import pandas as pd
#gdf = gpd.read_file('C:/TM/ImExt/New_24_11_14/Reference_Pts_Class.shp')
#
gdf = gpd.read_file('C:/TM/ImExt/New_24_11_14/Training_ZEFinal.shp')
# get names of land cover classes/labels
class_names = gdf['Class'].unique()
# create a unique id (integer) for each land cover class/label
class_ids = np.arange(class_names.size) + 1
# create a pandas data frame of the labels a1nd ids and save to csv
df = pd.DataFrame({'Class': class_names, 'id': class_ids})
df.to_csv('C:/TM/ImExt/class_lookupxx.csv')
# add a new column to geodatafame with the id for each class/label
gdf['id'] = gdf['Class'].map(dict(zip(class_names, class_ids)))
 
# split the truth data into training and test data sets and save each to a new shapefile
gdf_train = gdf.sample(frac=0.7)  # 70% of observations assigned to training data (30% to test data)
gdf_test = gdf.drop(gdf_train.index)
# save training and test data to shapefiles
gdf_train.to_file('C:/TM/ImExt/New_24_11_14/train_dataxx.shp')
gdf_test.to_file('C:/TM/ImExt/New_24_11_14/test_dataxx.shp')


#from osgeo import ogr

train_fn = 'C:/TM/ImExt/New_24_11_14/train_dataxx.shp'
train_ds = ogr.Open(train_fn)
lyr = train_ds.GetLayer()
# create a new raster layer in memory
driver = gdal.GetDriverByName('MEM')
target_ds = driver.Create('', naip_ds.RasterXSize, naip_ds.RasterYSize, 1, gdal.GDT_UInt16)
target_ds.SetGeoTransform(naip_ds.GetGeoTransform())
target_ds.SetProjection(naip_ds.GetProjection())
# rasterize the training points
options = ['ATTRIBUTE=id']
gdal.RasterizeLayer(target_ds, [1], lyr, options=options)
# retrieve the rasterized data and print basic stats
data = target_ds.GetRasterBand(1).ReadAsArray()

# rasterized observation (truth, training and test) data
ground_truth = target_ds.GetRasterBand(1).ReadAsArray()

# get unique values (0 is the background, or no data, value so it is not included) for each land cover type
classes = np.unique(ground_truth)[1:]

# for each class (land cover type) record the associated segment IDs
segments_per_class = {}
for klass in classes:
    segments_of_class = segments[ground_truth == klass]
    segments_per_class[klass] = set(segments_of_class)
 
# make sure no segment ID represents more than one class
intersection = set()
accum = set()
for class_segments in segments_per_class.values():
    intersection |= accum.intersection(class_segments)
    accum |= class_segments
assert len(intersection) == 0, "Segment(s) represent multiple classes"

from sklearn.ensemble import RandomForestClassifier
train_img = np.copy(segments)
threshold = train_img.max() + 1  # make the threshold value greater than any land cover class value

# all pixels in training segments assigned value greater than threshold
for klass in classes:
    class_label = threshold + klass
    for segment_id in segments_per_class[klass]:
        train_img[train_img == segment_id] = class_label
 
# training segments receive land cover class value, all other segments 0
train_img[train_img <= threshold] = 0
train_img[train_img > threshold] -= threshold

# create objects and labels for training data
training_objects = []
training_labels = []
for klass in classes:
    class_train_object = [v for i, v in enumerate(objects) if segment_ids[i] in segments_per_class[klass]]
    training_labels += [klass] * len(class_train_object)
    training_objects += class_train_object
 
classifier = RandomForestClassifier(n_jobs=-1)  # setup random forest classifier
classifier.fit(training_objects, training_labels)  # fit rf classifier
predicted = classifier.predict(objects)  # predict with rf classifier

# create numpy array from rf classifiation and save to raster
clf = np.copy(segments)
for segment_id, klass in zip(segment_ids, predicted):
    clf[clf == segment_id] = klass
 
#mask = np.sum(img, axis=2)  # this section masks no data values
#mask[mask > 0.0] = 1.0
#mask[mask == 0.0] = -1.0
#clf = np.multiply(clf, mask)
#clf[clf < 0] = -9999.0
 
clfds = driverTiff.Create('C:/TM/ImExt/New_24_11_14/classified_resultxx.tif', naip_ds.RasterXSize, naip_ds.RasterYSize,
                          1, gdal.GDT_Float32)  # this section saves to raster
clfds.SetGeoTransform(naip_ds.GetGeoTransform())
clfds.SetProjection(naip_ds.GetProjection())
#clfds.GetRasterBand(1).SetNoDataValue(-9999.0)
clfds.GetRasterBand(1).WriteArray(clf)
clfds = None
 
print('Done!')


import numpy as np
from osgeo import gdal
#from osgeo import ogr

from sklearn import metrics
 
# read original image to get info for raster dimensions
naip_fn = 'C:/TM/imExt/New_24_11_14/Stack_ZE.tif'

driverTiff = gdal.GetDriverByName('GTiff')
naip_ds = gdal.Open(naip_fn)
 
# rasterize test data for pixel-to-pixel comparison
test_fn = 'C:/TM/ImExt/New_24_11_14/test_dataxx.shp'
test_ds = ogr.Open(test_fn)
lyr = test_ds.GetLayer()
driver = gdal.GetDriverByName('MEM')
target_ds = driver.Create('', naip_ds.RasterXSize, naip_ds.RasterYSize, 1, gdal.GDT_UInt16)
target_ds.SetGeoTransform(naip_ds.GetGeoTransform())
target_ds.SetProjection(naip_ds.GetProjection())
options = ['ATTRIBUTE=id']
gdal.RasterizeLayer(target_ds, [1], lyr, options=options)
 
truth = target_ds.GetRasterBand(1).ReadAsArray()  # truth/test data array
 
pred_ds = gdal.Open('C:/TM/ImExt/New_24_11_14/classified_resultxx.tif')  
pred = pred_ds.GetRasterBand(1).ReadAsArray()  # predicted data array
idx = np.nonzero(truth) # get indices where truth/test has data values
cm = metrics.confusion_matrix(truth[idx], pred[idx])  # create a confusion matrix at the truth/test locations
 
# pixel accuracy
print(cm)
print(cm.diagonal())
print(cm.sum(axis=0))
accuracy = cm.diagonal() / cm.sum(axis=0)  # overall accuracy
print(accuracy)