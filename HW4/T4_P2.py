# CS 181, Spring 2022
# Homework 4

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import seaborn as sn


# Loading datasets for K-Means and HAC
small_dataset = np.load("data/small_dataset.npy")
large_dataset = np.load("data/large_dataset.npy")

# NOTE: You may need to add more helper functions to these classes
class KMeans(object):
    # K is the K in KMeans
    def __init__(self, K):
        self.K = K

    # X is a (N x 784) array where 28x28 flattened is the dimensions of each of the N images.
    def fit(self, X):
        # initialize prototype centroids, each of which is 1 x 784
        centers = []   # K x 784
        for i in range(0, self.K):
            centroid = []
            for j in range(0, 784):
                centroid.append(np.random.randint(low=0, high=256))  # pixel values range 0 to 255
            centers.append(centroid)
        centers = np.array(centers)

        # assign datapoints to centers, and update centers, 10 times
        # # also calculate the loss for each iteration
        # iter_loss = []
        for i in range(0, 10):
            # assign cluster to each datapoint
            assign = self.assign(X, centers)  # 1 x N array (each elem 0 to K-1 to tell which cluster belong to)

            # # given assignments, calculate residual sum of squares
            # loss = 0
            # for i in range(0, X.shape[0]):
            #     loss += np.linalg.norm(X[i] - centers[assign[i]])**2
            # iter_loss.append(loss)

            # given assignments, update centers
            centers = self.update(X, assign)   # K x 784 array of centroids

        # this is for get_mean_images func
        self.mean_images = centers

        # this is for part 5 (number of images per cluster)
        self.clusters_distrib = self.assign(X, centers)  # 1 x N array (each elem 0 to K-1 to tell which cluster belong to)
        # # for part 1, plot the change in loss for each iteration
        # iters = np.arange(1, 11)
        # plt.plot(iters, iter_loss)
        # plt.xlabel('Iteration #')
        # plt.ylabel('Residual sum of squares')
        # plt.show()

    # assign each datapoint to a cluster, return 1 x N array (each elem 0 to K-1 to tell which cluster belong to)
    def assign(self, X, centers):
        # assign each datapoint to cluster
        assign = []  # 1 x N array (each element from 0 to K-1 to tell which cluster belong to)
        for i in range(0, X.shape[0]):
            min_dist = [np.linalg.norm(X[i] - centers[0]), 0]   # initialize (min_dist, cluster_index) pair
            for j in range(0, self.K):
                dist = np.linalg.norm(X[i] - centers[j])
                if min_dist[0] >= dist:
                    min_dist = [dist, j]
            # we have now found assignment of datapoint
            assign.append(min_dist[1])
        return assign

    # update centroids of each cluster by averaging the points in the cluster
    # returns new centers array of K x 784
    def update(self, X, assign):
        # for each cluster k, set center to be mean of datapoints assigned to this cluster
        sum = np.zeros(shape=(self.K, 784))   # track sum of datapoint for each class
        Nk = np.zeros(self.K)   # track total num of datapoints in each class
        for i in range(0, X.shape[0]):
            sum[assign[i]] += X[i]
            Nk[assign[i]] += 1
        # update centers
        for k in range(0, self.K):
            if Nk[k] != 0:
                sum[k] = sum[k] / Nk[k]

        # this is the new centers
        return sum

    # each image represents the mean of each of the fitted clusters, aka centroid of each cluster
    # we just return centers from fit(), which is 10 x 784 array
    def get_mean_images(self):
        return self.mean_images





class HAC(object):
    def __init__(self, linkage):
        self.linkage = linkage

    def fit(self, X):
        cluster_count = X.shape[0]
        X = list(X)   # will have different sized rows, so we need a list of 1d np arrays as opposed to 2d np array

        if self.linkage == 'max':
            link_tracker = []  # list of [dist, clusterIndex1, clusterIndex2]
            # note that linkage between cluster i and j is same as j and i
            for i in range(0, len(X) - 1):
                for j in range(i + 1, len(X)):
                    # link between c_i and c_j is dist btwn FARTHEST IM IN C_I AND FARTHEST IM IN C_J
                    link = self.max_link(X[i], X[j])
                    link_tracker.append([link, i, j])

            while(cluster_count > 10):
                # pick lowest linkage in link_tracker
                smallest_link = link_tracker[0]
                for elem in link_tracker:
                    if smallest_link[0] > elem[0]:
                        smallest_link = elem
                # we have now found the smallest min_link out of images, ie closest clusters by our linkage criteria
                # merge into one cluster, append this to X, and remove induvidual clusters from X
                merged = np.vstack((X[smallest_link[1]], X[smallest_link[2]]))
                X.append(merged)  # add averaged cluster to the end
                del X[smallest_link[1]]
                del X[smallest_link[2] - 1]   # account for the shift in our index since we removed an elem
                cluster_count -= 1

                # update link_tracker
                del_index = []
                for i in range(0, len(link_tracker)):
                    # track which list item to remove.... remove all which involves the clusters we just merged
                    if link_tracker[i][1] == smallest_link[1] or link_tracker[i][2] == smallest_link[2] or link_tracker[i][1] == smallest_link[2] or link_tracker[i][2] == smallest_link[1]:
                        del_index.append(i)
                    else:
                        # since we just removed two clusters from X, we need to shift our cluster references
                        # back if the cluster referenced in link tracking was indexed after the cluster removed
                        if link_tracker[i][1] > smallest_link[1] and link_tracker[i][1] > smallest_link[2]:
                            link_tracker[i][1] -= 2
                        elif link_tracker[i][1] > smallest_link[1] or link_tracker[i][1] > smallest_link[2]:
                            link_tracker[i][1] -= 1

                        if link_tracker[i][2] > smallest_link[1] and link_tracker[i][2] > smallest_link[2]:
                            link_tracker[i][2] -= 2
                        elif link_tracker[i][2] > smallest_link[1] or link_tracker[i][2] > smallest_link[2]:
                            link_tracker[i][2] -= 1
                shift = 0
                for elem in del_index:
                    del link_tracker[elem - shift]  # remove the elem we wanted to remove
                    shift += 1   # now that link_tracker is one elem less, our next inteded index has shifted by 1

                # now, compute new linkage between new merged cluster (last index) and existing clusters
                for i in range(0, len(X) - 1):
                    link = self.max_link(X[i], X[-1])
                    link_tracker.append([link, i, len(X)-1])


        elif self.linkage == 'min':
            link_tracker = []  # list of [dist, clusterIndex1, clusterIndex2]
            # note that linkage between cluster i and j is same as j and i
            for i in range(0, len(X) - 1):
                for j in range(i + 1, len(X)):
                    # link between c_i and c_j is dist btwn CLOSEST IM IN C_I AND CLOSEST IM IN C_J
                    link = self.min_link(X[i], X[j])
                    link_tracker.append([link, i, j])

            while(cluster_count > 10):
                # pick lowest linkage in link_tracker
                smallest_link = link_tracker[0]
                for elem in link_tracker:
                    if smallest_link[0] > elem[0]:
                        smallest_link = elem
                # we have now found the smallest min_link out of images, ie closest clusters by our linkage criteria
                # merge into one cluster, append this to X, and remove induvidual clusters from X
                merged = np.vstack((X[smallest_link[1]], X[smallest_link[2]]))
                X.append(merged)  # add averaged cluster to the end
                del X[smallest_link[1]]
                del X[smallest_link[2] - 1]   # account for the shift in our index since we removed an elem
                cluster_count -= 1

                # update link_tracker
                del_index = []
                for i in range(0, len(link_tracker)):
                    # track which list item to remove.... remove all which involves the clusters we just merged
                    if link_tracker[i][1] == smallest_link[1] or link_tracker[i][2] == smallest_link[2] or link_tracker[i][1] == smallest_link[2] or link_tracker[i][2] == smallest_link[1]:
                        del_index.append(i)
                    else:
                        # since we just removed two clusters from X, we need to shift our cluster references
                        # back if the cluster referenced in link tracking was indexed after the cluster removed
                        if link_tracker[i][1] > smallest_link[1] and link_tracker[i][1] > smallest_link[2]:
                            link_tracker[i][1] -= 2
                        elif link_tracker[i][1] > smallest_link[1] or link_tracker[i][1] > smallest_link[2]:
                            link_tracker[i][1] -= 1

                        if link_tracker[i][2] > smallest_link[1] and link_tracker[i][2] > smallest_link[2]:
                            link_tracker[i][2] -= 2
                        elif link_tracker[i][2] > smallest_link[1] or link_tracker[i][2] > smallest_link[2]:
                            link_tracker[i][2] -= 1
                shift = 0
                for elem in del_index:
                    del link_tracker[elem - shift]  # remove the elem we wanted to remove
                    shift += 1   # now that link_tracker is one elem less, our next inteded index has shifted by 1

                # now, compute new linkage between new merged cluster (last index) and existing clusters
                for i in range(0, len(X) - 1):
                    link = self.min_link(X[i], X[-1])
                    link_tracker.append([link, i, len(X)-1])

        elif self.linkage == 'centroid':
            link_tracker = []  # list of [dist, clusterIndex1, clusterIndex2]
            # note that linkage between cluster i and j is same as j and i
            for i in range(0, len(X) - 1):
                for j in range(i + 1, len(X)):
                    # link between c_i and c_j is dist btwn CENTROID of C_I AND CENTROID of C_J
                    link = self.centroid_link(X[i], X[j])
                    link_tracker.append([link, i, j])

            while(cluster_count > 10):
                # pick lowest linkage in link_tracker
                smallest_link = link_tracker[0]
                for elem in link_tracker:
                    if smallest_link[0] > elem[0]:
                        smallest_link = elem
                # we have now found the smallest min_link out of images, ie closest clusters by our linkage criteria
                # merge into one cluster, append this to X, and remove induvidual clusters from X
                merged = np.vstack((X[smallest_link[1]], X[smallest_link[2]]))
                X.append(merged)  # add averaged cluster to the end
                del X[smallest_link[1]]
                del X[smallest_link[2] - 1]   # account for the shift in our index since we removed an elem
                cluster_count -= 1

                # update link_tracker
                del_index = []
                for i in range(0, len(link_tracker)):
                    # track which list item to remove.... remove all which involves the clusters we just merged
                    if link_tracker[i][1] == smallest_link[1] or link_tracker[i][2] == smallest_link[2] or link_tracker[i][1] == smallest_link[2] or link_tracker[i][2] == smallest_link[1]:
                        del_index.append(i)
                    else:
                        # since we just removed two clusters from X, we need to shift our cluster references
                        # back if the cluster referenced in link tracking was indexed after the cluster removed
                        if link_tracker[i][1] > smallest_link[1] and link_tracker[i][1] > smallest_link[2]:
                            link_tracker[i][1] -= 2
                        elif link_tracker[i][1] > smallest_link[1] or link_tracker[i][1] > smallest_link[2]:
                            link_tracker[i][1] -= 1

                        if link_tracker[i][2] > smallest_link[1] and link_tracker[i][2] > smallest_link[2]:
                            link_tracker[i][2] -= 2
                        elif link_tracker[i][2] > smallest_link[1] or link_tracker[i][2] > smallest_link[2]:
                            link_tracker[i][2] -= 1
                shift = 0
                for elem in del_index:
                    del link_tracker[elem - shift]  # remove the elem we wanted to remove
                    shift += 1   # now that link_tracker is one elem less, our next inteded index has shifted by 1

                # now, compute new linkage between new merged cluster (last index) and existing clusters
                for i in range(0, len(X) - 1):
                    link = self.centroid_link(X[i], X[-1])
                    link_tracker.append([link, i, len(X)-1])

        # for get_mean_images
        self.final_clusters = X  # there are only 10 clusters remaining here!

    # Returns the mean image when using n_clusters clusters
    def get_mean_images(self, n_clusters):
        mean_im = []
        for elem in self.final_clusters:
            if elem.ndim == 1:
                mean_im.append(elem)
            else:
                sum = np.zeros(784)
                for i in range(0, elem.shape[0]):
                    sum += elem[i]
                mean_im.append(sum/elem.shape[0])
        return np.array(mean_im)

    def max_link(self, a, b):
        # use cdist to compare distance between vectors in cluster a and vectors in cluster b
        # if a or b are 1 dimensional, make them 2d so that cdist happy
        if a.ndim == 1:
            a = np.reshape(a, (1, 784))
        if b.ndim == 1:
            b = np.reshape(b, (1, 784))

        # given a is 3x784 and b is 1x784, cdist[0] == dist btwn a[0] and b[0] == np.linalg.norm(a[0]-b[0])
        # return max distance between two clusters' furthest elems
        return np.max(cdist(a, b))

    def min_link(self, a, b):
        # use cdist to compare distance between vectors in cluster a and vectors in cluster b
        # if a or b are 1 dimensional, make them 2d so that cdist happy
        if a.ndim == 1:
            a = np.reshape(a, (1, 784))
        if b.ndim == 1:
            b = np.reshape(b, (1, 784))

        # given a is 3x784 and b is 1x784, cdist[0] == dist btwn a[0] and b[0] == np.linalg.norm(a[0]-b[0])
        # return max distance between two clusters' furthest elems
        return np.min(cdist(a, b))

    def centroid_link(self, a, b):
        centroid_a = np.sum(a, axis=0) / a.shape[0]
        centroid_b = np.sum(b, axis=0) / b.shape[0]

        return np.linalg.norm(centroid_a - centroid_b)


# Plotting code for parts 2 and 3 (plots the centroid)
def make_mean_image_plot(data, standardized=False):
    # Number of random restarts
    niters = 3
    K = 10
    # since K=10 and
    # Will eventually store the pixel representation of all the mean images across restarts
    allmeans = np.zeros((K, niters, 784))
    for i in range(niters):
        KMeansClassifier = KMeans(K=K)
        KMeansClassifier.fit(data)
        allmeans[:,i] = KMeansClassifier.get_mean_images()
    fig = plt.figure(figsize=(10,10))
    plt.suptitle('Class mean images across random restarts' + (' (standardized data)' if standardized else ''), fontsize=16)
    for k in range(K):
        for i in range(niters):
            ax = fig.add_subplot(K, niters, 1+niters*k+i)
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.tick_params(axis='both', which='both', length=0)
            if k == 0: plt.title('Iter '+str(i))
            if i == 0: ax.set_ylabel('Class '+str(k), rotation=90)
            plt.imshow(allmeans[k,i].reshape(28,28), cmap='Greys_r')  # covert 784 back to 28x28 im!
    plt.show()

# # ~~ Part 2 ~~
# make_mean_image_plot(large_dataset, False)
#
# # ~~ Part 3 ~~
# # mean would be the average value of the i-th pixel ACCROSS ALL DATA POINTS
# # standardized value for the i-th pixel in a particular data point = (pixel_value - mean) / std
# large_stdzd = np.copy(large_dataset).astype(np.float64)  # copy values, but no share same addr
# # iterate through every COL of dataset
# for i in range(0, large_stdzd.shape[1]):
#
#     std = np.std(large_stdzd[:,i])
#     # divide by 1 for pixels with 0 variance
#     if std == 0:
#         std = 1
#
#     # this pixel is now standardized
#     large_stdzd[:,i] = (large_stdzd[:,i] - np.mean(large_stdzd[:,i])) / std
#
# make_mean_image_plot(large_stdzd, True)

# Plotting code for part 4
LINKAGES = [ 'max', 'min', 'centroid' ]
n_clusters = 10



fig = plt.figure(figsize=(10,10))
plt.suptitle("HAC mean images with max, min, and centroid linkages")
for l_idx, l in enumerate(LINKAGES):
    # Fit HAC
    hac = HAC(l)
    hac.fit(small_dataset)
    mean_images = hac.get_mean_images(n_clusters)
    # Make plot
    for m_idx in range(mean_images.shape[0]):
        m = mean_images[m_idx]
        ax = fig.add_subplot(n_clusters, len(LINKAGES), l_idx + m_idx*len(LINKAGES) + 1)
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis='both', which='both', length=0)
        if m_idx == 0: plt.title(l)
        if l_idx == 0: ax.set_ylabel('Class '+str(m_idx), rotation=90)
        plt.imshow(m.reshape(28,28), cmap='Greys_r')
plt.show()



# PART 5 - number of images in cluster vs cluster index
# find plot for Kmeans (non-standardized)
KMeansClassifier = KMeans(K=10)
KMeansClassifier.fit(small_dataset)
km_cluster_distrib = KMeansClassifier.clusters_distrib
Nk = np.zeros(10)   # track total num of datapoints in each class
for i in range(0, small_dataset.shape[0]):
    Nk[km_cluster_distrib[i]] += 1
fig = plt.figure()
plt.bar([0,1,2,3,4,5,6,7,8,9], Nk)
plt.title('Kmeans (Non-standardized)')
plt.ylabel('Number of images in cluster')
plt.xlabel('Cluster index')
plt.show()

# find plot for HAC max
hacmax = HAC('max')
hacmax.fit(small_dataset)
hac_cluster_distrib = []
for elem in hacmax.final_clusters:
    if elem.ndim == 1:
        hac_cluster_distrib.append(1)
    else:
        hac_cluster_distrib.append(elem.shape[0])
fig = plt.figure()
plt.bar([0,1,2,3,4,5,6,7,8,9], hac_cluster_distrib)
plt.title('HAC (max-linkage)')
plt.ylabel('Number of images in cluster')
plt.xlabel('Cluster index')
plt.show()


# find plot for HAC min
hacmin = HAC('min')
hacmin.fit(small_dataset)
hac_cluster_distrib = []
for elem in hacmin.final_clusters:
    if elem.ndim == 1:
        hac_cluster_distrib.append(1)
    else:
        hac_cluster_distrib.append(elem.shape[0])
fig = plt.figure()
plt.bar([0,1,2,3,4,5,6,7,8,9], hac_cluster_distrib)
plt.title('HAC (min-linkage)')
plt.ylabel('Number of images in cluster')
plt.xlabel('Cluster index')
plt.show()

# find plot for HAC centroid
haccen = HAC('centroid')
haccen.fit(small_dataset)
hac_cluster_distrib = []
for elem in haccen.final_clusters:
    if elem.ndim == 1:
        hac_cluster_distrib.append(1)
    else:
        hac_cluster_distrib.append(elem.shape[0])
fig = plt.figure()
plt.bar([0,1,2,3,4,5,6,7,8,9], hac_cluster_distrib)
plt.title('HAC (centroid-linkage)')
plt.ylabel('Number of images in cluster')
plt.xlabel('Cluster index')
plt.show()



# PART 6
# create 6 heatmap for K-means and all 3 HACs confusion matrix
# use same results from part 5

# heatmap Kmeans vs HAC max
mat = np.zeros((10,10))
hacmax_distrib = []
for i in range(0, 300):
    for j in range(0, 10):
        # find cluster j to which image i belongs
        if np.where(np.prod(hacmax.final_clusters[j] == small_dataset[i], axis=-1))[0].size != 0:
            hacmax_distrib.append(j)
            break
# create matrix which fed into heatmap
# both km_cluster_distrib and hacmax_distrib are 1x300 arr which tell which cluster each im belongs for each method
for i in range(0, 300):
    mat[km_cluster_distrib[i]][hacmax_distrib[i]] += 1

hm = sn.heatmap(data=mat)
hm.set_ylabel('Kmeans Cluster Index')
hm.set_xlabel('HAC (Max-Linkage) Cluster Index')
plt.show()



# heatmap Kmeans vs HAC min
mat = np.zeros((10,10))
hacmin_distrib = []
for i in range(0, 300):
    for j in range(0, 10):
        # find cluster j to which image i belongs
        if np.where(np.prod(hacmin.final_clusters[j] == small_dataset[i], axis=-1))[0].size != 0:
            hacmin_distrib.append(j)
            break
# create matrix which fed into heatmap
# both km_cluster_distrib and hacmin_distrib are 1x300 arr which tell which cluster each im belongs for each method
for i in range(0, 300):
    mat[km_cluster_distrib[i]][hacmin_distrib[i]] += 1

hm = sn.heatmap(data=mat)
hm.set_ylabel('Kmeans Cluster Index')
hm.set_xlabel('HAC (Min-Linkage) Cluster Index')
plt.show()

# heatmap Kmeans vs HAC centroid
mat = np.zeros((10,10))
haccen_distrib = []
for i in range(0, 300):
    for j in range(0, 10):
        # find cluster j to which image i belongs
        if np.where(np.prod(haccen.final_clusters[j] == small_dataset[i], axis=-1))[0].size != 0:
            haccen_distrib.append(j)
            break
# create matrix which fed into heatmap
# both km_cluster_distrib and hacmin_distrib are 1x300 arr which tell which cluster each im belongs for each method
for i in range(0, 300):
    mat[km_cluster_distrib[i]][haccen_distrib[i]] += 1

hm = sn.heatmap(data=mat)
hm.set_ylabel('Kmeans Cluster Index')
hm.set_xlabel('HAC (Centroid-Linkage) Cluster Index')
plt.show()

# heatmap HAC max vs HAC min
mat = np.zeros((10,10))
# create matrix which fed into heatmap
# both km_cluster_distrib and hacmin_distrib are 1x300 arr which tell which cluster each im belongs for each method
for i in range(0, 300):
    mat[hacmax_distrib[i]][hacmin_distrib[i]] += 1

hm = sn.heatmap(data=mat)
hm.set_ylabel('HAC (Max-Linkage) Cluster Index')
hm.set_xlabel('HAC (Min-Linkage) Cluster Index')
plt.show()

# heatmap HAC max vs HAC centroid
mat = np.zeros((10,10))
# create matrix which fed into heatmap
# both km_cluster_distrib and hacmin_distrib are 1x300 arr which tell which cluster each im belongs for each method
for i in range(0, 300):
    mat[hacmax_distrib[i]][haccen_distrib[i]] += 1

hm = sn.heatmap(data=mat)
hm.set_ylabel('HAC (Max-Linkage) Cluster Index')
hm.set_xlabel('HAC (Centroid-Linkage) Cluster Index')
plt.show()

# heatmap HAC min vs HAC centroid
mat = np.zeros((10,10))
# create matrix which fed into heatmap
# both km_cluster_distrib and hacmin_distrib are 1x300 arr which tell which cluster each im belongs for each method
for i in range(0, 300):
    mat[hacmin_distrib[i]][haccen_distrib[i]] += 1

hm = sn.heatmap(data=mat)
hm.set_ylabel('HAC (Min-Linkage) Cluster Index')
hm.set_xlabel('HAC (Centroid-Linkage) Cluster Index')
plt.show()
