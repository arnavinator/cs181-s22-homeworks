# CS 181, Spring 2022
# Homework 4

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

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
                if min_dist[0] > dist:
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
                    # min link between c_i and c_j is dist btwn FARTHEST PIX IN C_I AND FARTHEST PIX IN C_J
                    link = self.max_link(X[i], X[j])
                    link_tracker.append([link, i, j])

            while(cluster_count > 10):
                print(cluster_count)
                # pick lowest linkage in link_tracker
                smallest_link = link_tracker[0]
                for elem in link_tracker:
                    if smallest_link[0] < elem[0]:
                        smallest_link = elem

                # we have now found the smallest min_link out of images, ie closest clusters by our linkage criteria
                # merge into one cluster, append this to X, and remove induvidual clusters from X
                merged = np.vstack((X[smallest_link[1]], X[smallest_link[2]]))
                X.append(merged)  # add averaged cluster to the end
                del X[smallest_link[1]]
                del X[smallest_link[2]]
                cluster_count -= 1

                # update link_tracker
                for elem in link_tracker:
                    # remove all min linkage which involves the clusters we just merged
                    if elem[1] == smallest_link[1] or elem[2] == smallest_link[2]:
                        link_tracker.remove(elem)
                    else:
                        # since we just removed two clusters from X, we need to shift our cluster references
                        # back by 1 if the cluster referenced in link tracking was indexed after the cluster removed
                        if elem[1] > smallest_link[1]:
                            elem[1] -= 1
                        if elem[2] > smallest_link[2]:
                            elem[2] -= 1
                # now, compute new linkage between new merged cluster (last index) and existing clusters
                for i in range(0, len(X) - 1):
                    link = self.max_link(X[i], X[-1])
                    link_tracker.append([link, i, len(X)-1])


        elif self.linkage == 'min':
            while (cluster_count > 10):
                # tracker for closest cluster to every cluster (mean image)
                min_link_tracker = []  # array of (dist, cluster1, cluster2)
                for i in range(0, X.shape[0]):
                    # find min linkage for cluster i
                    # we need to use cdist to compare dist between every pixel of im-i to all other pixel of im-j
                    # doing so requires converting each im to 2d
                    a = np.reshape(X[i], (784, 1))
                    # initialize TOO LARGE dummy linkage value (distance between entire vectors)
                    min_link = [255**2, 0, 1]  # (dist, cluster1, cluster2)
                    for j in range(0, X.shape[0]):
                        if i != j:
                            b = np.reshape(X[j], (784, 1))
                            # want dist between CLOSEST ELEM OF I AND CLOSEST ELEM OF J
                            link = np.min(cdist(a, b))
                            if link <= min_link[0]:
                                min_link = [link, i, j]
                    # we have now found min link for image i
                    min_link_tracker.append(min_link)

                # find the smallest min_link out of all
                smallest_link = min_link_tracker[0]
                for elem in min_link_tracker:
                    if smallest_link[0] < elem[0]:
                        smallest_link = elem

                # we have now found the closest clusters by our linkage criteria,
                # average closest clusters, and remove induvidual clusters
                avg = (X[smallest_link[1]] + X[smallest_link[2]]) / 2
                X = np.vstack([X, avg])  # add averaged cluster to the end
                X = np.delete(X, smallest_link[1], axis=0)
                X = np.delete(X, smallest_link[2], axis=0)

                cluster_count -= 1

        elif self.linkage == 'centroid':
            while (cluster_count > 10):
                # tracker for closest cluster to every cluster (mean image)
                min_link_tracker = []  # array of (dist, cluster1, cluster2)
                for i in range(0, X.shape[0]):
                    # find min linkage for cluster i
                    # initialize TOO LARGE dummy linkage value
                    min_link = [255**2, 0, 1]  # (dist, cluster1, cluster2)
                    for j in range(0, X.shape[1]):
                        if i != j:
                            # want dist between CENTROID OF I AND CENTROID OF J
                            link = np.abs(np.average(X[i]) - np.average(X[j]))
                            if link <= min_link[0]:
                                min_link = [link, i, j]
                    # we have now found min link for image i
                    min_link_tracker.append(min_link)

                # find the smallest min_link out of all
                smallest_link = min_link_tracker[0]
                for elem in min_link_tracker:
                    if smallest_link[0] < elem[0]:
                        smallest_link = elem

                # we have now found the closest clusters by our linkage criteria,
                # average closest clusters, and remove induvidual clusters
                avg = (X[smallest_link[1]] + X[smallest_link[2]]) / 2
                X = np.vstack([X, avg])  # add averaged cluster to the end
                X = np.delete(X, smallest_link[1], axis=0)
                X = np.delete(X, smallest_link[2], axis=0)

                cluster_count -= 1

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
        # we need to use cdist to compare dist between every pixel of cluster a to all other pixel of cluster b
        # first, flatten each cluster, then convert each image to 2d so that compatible with cdist
        # if cluster a flattened has x pixels, cluster b flattened has y pixels, then after converting
        # to 2d, cdist will yield a x by y matrix M, where M[x,y] = distance between x and y
        if a.ndim > 1:
            a = a.flatten()
        if b.ndim > 1:
            b = b.flatten()

        a = np.reshape(a, (a.shape[0], 1))
        b = np.reshape(b, (b.shape[0], 1))

        # return max distance between two clusters' furthest elems
        return np.max(cdist(a, b))



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
#
# # ~~ Part 2 ~~
# make_mean_image_plot(large_dataset, False)
#
# # ~~ Part 3 ~~
# # mean would be the average value of the i-th pixel ACCROSS ALL DATA POINTS
# # standardized value for the i-th pixel in a particular data point = (pixel_value - mean) / std
# large_stdzd = np.copy(large_dataset)  # copy values, but no share same addr
# # iterate through every COL of dataset
# for i in range(0, large_stdzd.shape[1]):
#     col = large_stdzd[:,i]
#
#     std = np.var(col)**0.5
#     # divide by 1 for pixels with 0 variance
#     if std == 0:
#         std = 1
#
#     # this pixel is now standardized
#     col = (col - np.mean(col)) / std
#
# make_mean_image_plot(large_stdzd, True)

# Plotting code for part 4
# LINKAGES = [ 'max', 'min', 'centroid' ]
n_clusters = 10
LINKAGES = ['max']

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

# TODO: Write plotting code for part 5

# TODO: Write plotting code for part 6