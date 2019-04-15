#####################
# CS 181, Spring 2019
# Homework 1, Problem 3
#
##################

import csv
import numpy as np
import matplotlib.pyplot as plt

csv_filename = 'data/year-sunspots-republicans.csv'
years  = []
republican_counts = []
sunspot_counts = []

with open(csv_filename, 'r') as csv_fh:

    # Parse as a CSV file.
    reader = csv.reader(csv_fh)

    # Skip the header line.
    next(reader, None)

    # Loop over the file.
    for row in reader:

        # Store the data.
        years.append(float(row[0]))
        sunspot_counts.append(float(row[1]))
        republican_counts.append(float(row[2]))

# Turn the data into numpy arrays.
years  = np.array(years)
republican_counts = np.array(republican_counts)
sunspot_counts = np.array(sunspot_counts)
last_year = 1985


Y = republican_counts[years<last_year]
sc = sunspot_counts[years < last_year]

X = np.vstack((np.ones(sc.shape), sc)).T
X_a = np.vstack((np.ones(sc.shape), sc, np.power(sc,2), np.power(sc,3), np.power(sc,4), np.power(sc,5))).T

bases_c = []
for i in range(1,6):
	basis = np.cos(np.divide(sc, i))
	bases_c.append(basis)
X_c = np.vstack([np.ones(sc.shape)] + bases_c).T	

bases_d = []
for i in range(1,26):
	basis = np.cos(np.divide(sc, i))
	bases_d.append(basis)
X_d = np.vstack([np.ones(sc.shape)] + bases_d).T	

# Find the regression weights using the Moore-Penrose pseudoinverse.
w = np.linalg.solve(np.dot(X.T, X) , np.dot(X.T, Y))
w_a = np.linalg.solve(np.dot(X_a.T, X_a) , np.dot(X_a.T, Y))
w_c = np.linalg.solve(np.dot(X_c.T, X_c) , np.dot(X_c.T, Y))
w_d = np.linalg.solve(np.dot(X_d.T, X_d) , np.dot(X_d.T, Y))
print(w)
print(w_a)
grid_spots = np.linspace(min(sc), max(sc), 200)
grid_X = np.vstack((np.ones(grid_spots.shape), grid_spots))
grid_Yhat  = np.dot(grid_X.T, w)

grid_X_a = np.vstack((np.ones(grid_spots.shape), grid_spots, np.power(grid_spots,2), np.power(grid_spots,3), np.power(grid_spots,4), np.power(grid_spots,5)))
grid_Yhat_a  = np.dot(grid_X_a.T, w_a)

grid_spots_bases_c = []
for i in range(1,6):
	basis = np.cos(np.divide(grid_spots, i))
	grid_spots_bases_c.append(basis)
grid_X_c = np.vstack([np.ones(grid_spots.shape)] + grid_spots_bases_c)
grid_Yhat_c  = np.dot(grid_X_c.T, w_c)

grid_spots_bases_d = []
for i in range(1,26):
	basis = np.cos(np.divide(grid_spots, i))
	grid_spots_bases_d.append(basis)
grid_X_d = np.vstack([np.ones(grid_spots.shape)] + grid_spots_bases_d)
grid_Yhat_d  = np.dot(grid_X_d.T, w_d)


# Plots

plt.plot(sc, Y, 'o', grid_spots, grid_Yhat, '-')
plt.xlabel("Sunspots")
plt.ylabel("Number of Republicans in Congress")
plt.title("traditional linear regression")
plt.show()

plt.plot(sc, Y, 'o', grid_spots, grid_Yhat_a, '-')
plt.xlabel("Sunspots")
plt.ylabel("Number of Republicans in Congress")
plt.title("Part 3a")
plt.show()

plt.plot(sc, Y, 'o', grid_spots, grid_Yhat_c, '-')
plt.xlabel("Sunspots")
plt.ylabel("Number of Republicans in Congress")
plt.title("Part 3c")
plt.show()

plt.plot(sc, Y, 'o', grid_spots, grid_Yhat_d, '-')
plt.xlabel("Sunspots")
plt.ylabel("Number of Republicans in Congress")
plt.title("Part 3d")
plt.show()

# Errors

print("squared error for linear model is: {}".format(sum( (np.dot(X,w) - Y) ** 2)))
print("squared error for part a is: {}".format(sum( (np.dot(X_a,w_a) - Y) ** 2)))
print("squared error for part c is: {}".format(sum( (np.dot(X_c,w_c) - Y) ** 2)))
print("squared error for part d is: {}".format(sum( (np.dot(X_d,w_d) - Y) ** 2)))

