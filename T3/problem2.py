# CS 181, Spring 2019
# Homework 3: Max-Margin, Ethics, Clustering
# Name: Zach Dietz
# Email: zachdietz1@gmail.com

import numpy as np 
import matplotlib.pyplot as plt

# This line loads the images for you. Don't change it! 
pics = np.load("images.npy", allow_pickle=False)

class KMeans(object):
	# K is the K in KMeans
	def __init__(self, K):
		self.K = K
		self.means = [np.random.rand(28,28) for i in range(K)]
		self.costs = []

	def __L2(self, u, v):
		return np.linalg.norm(np.subtract(u,v))

	# X is a (N x 28 x 28) array where 28x28 is the dimensions of each of the N images.
	def fit(self, X):
		# assign all points to the first class
		classes = [0 for i in range(X.shape[0])]
		
		while True:
			num_changed = 0
			for i in range(X.shape[0]):
				# update classes of each data point
				distances = [self.__L2(X[i], mean) for mean in self.means]
				new = np.argmin(distances)
				if classes[i] != new:
					classes[i] = new
					num_changed += 1
			
			for i in range(self.K):
				# update means of each class
				if len(X[np.array(classes) == i]) != 0:
					self.means[i] = np.mean(X[np.array(classes) == i], axis=0)

			# calcualte cost
			cost = 0
			for i in range(X.shape[0]):
				cost += self.__L2(self.means[classes[i]], X[i]) 
			self.costs.append(cost)
			
			if num_changed == 0:
				break


	# This should return the arrays for K images. Each image should represent the mean of each of the fitted clusters.
	def get_mean_images(self):
		return self.means

	def visualize(self):
		plt.figure()
		plt.plot(np.arange(len(self.costs)), self.costs)
		plt.suptitle("Objective Function vs. Iterations")
		plt.xlabel("Iterations")
		plt.ylabel("Objective")
		plt.show()

	def final_objective(self):
		return self.costs[-1]

K = 10
KMeansClassifier = KMeans(K=K)
KMeansClassifier.fit(pics)
KMeansClassifier.visualize()

# Now plotting Final Objective as a function of K with errorbars...

def errorbars(Ks, R):
	# perform R restarts at each value in Ks and graph it
	costs = []
	spreads = []
	for i in Ks:
		costs_for_given_k = []
		for j in range(R):
			KMeansC = KMeans(K=i)
			KMeansC.fit(pics)
			costs_for_given_k.append(KMeansC.final_objective())
		costs.append(np.mean(costs_for_given_k))
		spreads.append(np.var(costs_for_given_k))

	plt.figure()
	plt.errorbar(Ks, costs, yerr = np.multiply(.0008,spreads), fmt='o')
	plt.suptitle("Final Objective vs. K (with scaled errorbars)")
	plt.xlabel("K")
	plt.ylabel("Final Objective")
	plt.show()
	#print(costs[-1])

# errorbars(range(2,16), 10) # This will take a while to run
errorbars([2, 10, 15], 5) # This is faster

# Mean images
images = KMeansClassifier.get_mean_images()
for k in range(10):
	plt.figure()
	plt.imshow(images[k].reshape(28,28), cmap='Greys_r')
	plt.show()

# Mean images of standardized data
standardized = [None for i in range(pics.shape[0])]

mean = np.mean(pics, axis = 0)
std = np.std(pics, axis = 0)
std[std == 0] = 1

for i, X in enumerate(pics):
	standardized[i] = np.divide(np.subtract(X, mean), std)
standardized = np.array(standardized) 

KMeansClassifier = KMeans(K=K)
KMeansClassifier.fit(standardized)

images_standardized = KMeansClassifier.get_mean_images()
for k in range(10):
	plt.figure()
	plt.imshow(images_standardized[k].reshape(28,28), cmap='Greys_r')
	plt.show()
