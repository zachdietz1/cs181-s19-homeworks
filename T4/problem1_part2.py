import numpy as np 

clusters = [[(0.1,0.5),(0.35,0.75)],[(0.28,1.35)],[(0,1.01)]]
cluster_names = ['black', 'red', 'blue']
dist_names = ['l_1', 'l_2', 'l_inf']
similarity_names = ['min', 'max', 'centroid', 'average']

def l_1(a, b):
	return np.absolute(a[0] - b[0]) + np.absolute(a[1] - b[1])

def l_2(a, b):
	return ((a[0] - b[0])**2 + (a[1] - b[1])**2) ** 0.5

def l_inf(a,b):
	return max([np.absolute(a[0] - b[0]), np.absolute(a[1] - b[1])])

def sim_min(c1, c2, dist):
	minimum = 1000
	for a in c1:
		for b in c2:
			if dist(a,b) < minimum:
				minimum = dist(a,b)
	return minimum

def sim_max(c1, c2, dist):
	maximum = -1000
	for a in c1:
		for b in c2:
			if dist(a,b) > maximum:
				maximum = dist(a,b)
	return maximum

def sim_centroid(c1, c2, dist):
	return dist(np.mean(c1,axis=0), np.mean(c2,axis=0))

def sim_avg(c1, c2, dist):
	sumation = 0
	for a in c1:
		for b in c2:
			sumation += dist(a,b)
	return sumation/len(c1)/len(c2)

for x, dist in enumerate([l_1, l_2, l_inf]):
	for y, sim in enumerate([sim_min, sim_max, sim_centroid, sim_avg]):
		minimum = 1000
		cluster1, cluster2 = None, None
		for i, c1 in enumerate(clusters):
			for j, c2 in enumerate(clusters):
				if c1 != c2: # not efficient, but correct
					distance = sim(c1,c2,dist)
					if distance < minimum:
						minimum = distance
						cluster1, cluster2 = i, j

		print("distance function was {}. similarity function was {}...".format(dist_names[x], similarity_names[y]))
		print("clusters merged first are {} and {} with distance {}".format(cluster_names[cluster1], cluster_names[cluster2], minimum))
		print("\n")

