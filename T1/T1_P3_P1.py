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


# Plot the data.
plt.figure(1)
plt.plot(years, republican_counts, 'o')
plt.xlabel("Year")
plt.ylabel("Number of Republicans in Congress")
plt.figure(2)
plt.plot(years, sunspot_counts, 'o')
plt.xlabel("Year")
plt.ylabel("Number of Sunspots")
plt.figure(3)
plt.plot(sunspot_counts[years<last_year], republican_counts[years<last_year], 'o')
plt.xlabel("Number of Sunspots")
plt.ylabel("Number of Republicans in Congress")
plt.show()

# Create the simplest basis, with just the time and an offset.
X = np.vstack((np.ones(years.shape), years)).T

# TODO: basis functions
# MY WORK:
X_a = np.vstack((np.ones(years.shape), years, np.power(years,2), np.power(years,3), np.power(years,4), np.power(years,5))).T

bases_b = []
for i in range(1,12):
	basis = np.exp(np.divide(np.power(np.subtract(years, 1955 + 5*i), 2), -25))
	bases_b.append(basis)
X_b = np.vstack([np.ones(years.shape)] + bases_b).T	

bases_c = []
for i in range(1,6):
	basis = np.cos(np.divide(years, i))
	bases_c.append(basis)
X_c = np.vstack([np.ones(years.shape)] + bases_c).T	

bases_d = []
for i in range(1,26):
	basis = np.cos(np.divide(years, i))
	bases_d.append(basis)
X_d = np.vstack([np.ones(years.shape)] + bases_d).T	

# Nothing fancy for outputs.
Y = republican_counts

# Find the regression weights using the Moore-Penrose pseudoinverse.
w = np.linalg.solve(np.dot(X.T, X) , np.dot(X.T, Y))
# MY WORK:
w_a = np.linalg.solve(np.dot(X_a.T, X_a) , np.dot(X_a.T, Y))
w_b = np.linalg.solve(np.dot(X_b.T, X_b) , np.dot(X_b.T, Y))
w_c = np.linalg.solve(np.dot(X_c.T, X_c) , np.dot(X_c.T, Y))
w_d = np.linalg.solve(np.dot(X_d.T, X_d) , np.dot(X_d.T, Y))

# Compute the regression line on a grid of inputs.
# DO NOT CHANGE grid_years!!!!!
grid_years = np.linspace(1960, 2005, 200)
grid_X = np.vstack((np.ones(grid_years.shape), grid_years))
grid_Yhat  = np.dot(grid_X.T, w)

# TODO: plot and report sum of squared error for each basis

# MY WORK:
grid_X_a = np.vstack((np.ones(grid_years.shape), grid_years, np.power(grid_years,2), np.power(grid_years,3), np.power(grid_years,4), np.power(grid_years,5)))
grid_Yhat_a  = np.dot(grid_X_a.T, w_a)

grid_year_bases_b = []
for i in range(1,12):
	basis = np.exp(np.divide(np.power(np.subtract(grid_years, 1955 + 5*i), 2), -25))
	grid_year_bases_b.append(basis)
grid_X_b = np.vstack([np.ones(grid_years.shape)] + grid_year_bases_b)
grid_Yhat_b  = np.dot(grid_X_b.T, w_b)

grid_year_bases_c = []
for i in range(1,6):
	basis = np.cos(np.divide(grid_years, i))
	grid_year_bases_c.append(basis)
grid_X_c = np.vstack([np.ones(grid_years.shape)] + grid_year_bases_c)
grid_Yhat_c  = np.dot(grid_X_c.T, w_c)

grid_year_bases_d = []
for i in range(1,26):
	basis = np.cos(np.divide(grid_years, i))
	grid_year_bases_d.append(basis)
grid_X_d = np.vstack([np.ones(grid_years.shape)] + grid_year_bases_d)
grid_Yhat_d  = np.dot(grid_X_d.T, w_d)

# Plot the data and the regression line.
plt.plot(years, republican_counts, 'o', grid_years, grid_Yhat, '-')
plt.xlabel("Year")
plt.ylabel("Number of Republicans in Congress")
plt.title("traditional linear regression")
plt.show()

# MY WORK:
plt.figure(4)
plt.plot(years, republican_counts, 'o', grid_years, grid_Yhat_a, '-')
plt.xlabel("Year")
plt.ylabel("Number of Republicans in Congress")
plt.title("Part 1a")
plt.show()

plt.figure(5)
plt.plot(years, republican_counts, 'o', grid_years, grid_Yhat_b, '-')
plt.xlabel("Year")
plt.ylabel("Number of Republicans in Congress")
plt.title("Part 1b")
plt.show()

plt.figure(6)
plt.plot(years, republican_counts, 'o', grid_years, grid_Yhat_c, '-')
plt.xlabel("Year")
plt.ylabel("Number of Republicans in Congress")
plt.title("Part 1c")
plt.show()

plt.figure(7)
plt.plot(years, republican_counts, 'o', grid_years, grid_Yhat_d, '-')
plt.xlabel("Year")
plt.ylabel("Number of Republicans in Congress")
plt.title("Part 1d")
plt.show()

print("squared error for linear model is: {}".format(sum( (np.dot(X,w) - Y) ** 2)))
print("squared error for part a is: {}".format(sum( (np.dot(X_a,w_a) - Y) ** 2)))
print("squared error for part b is: {}".format(sum( (np.dot(X_b,w_b) - Y) ** 2)))
print("squared error for part c is: {}".format(sum( (np.dot(X_c,w_c) - Y) ** 2)))
print("squared error for part d is: {}".format(sum( (np.dot(X_d,w_d) - Y) ** 2)))