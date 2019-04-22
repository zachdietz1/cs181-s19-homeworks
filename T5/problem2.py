import numpy as np
import matplotlib.pyplot as plt
import csv

data = np.genfromtxt('kf-data.csv', delimiter=',')
data = np.delete(data, 0, 0)

# define params
mu_e, sigma_e, mu_g, sigma_g, mu_p, sigma_p = 0, 0.05, 0, 1, 5, 1

x_0 = float(data[0][2])

mu_0 = ((x_0 - mu_g)*sigma_p ** 2+mu_p*sigma_g ** 2)/(sigma_g ** 2+sigma_p ** 2)
sigma_0 = np.sqrt(sigma_g ** 2*sigma_p ** 2/(sigma_g ** 2+sigma_p ** 2))

mus = [mu_0]
sigmas = [sigma_0]

for i, row in enumerate(data[1:]):
	x = float(row[2])
	denominator = sigma_g ** 2 + sigma_e ** 2 + sigmas[i] ** 2
	mu_t = ((x-mu_g)*(sigma_e ** 2+sigmas[i] ** 2) + (mus[i] + mu_e)*sigma_g ** 2)/denominator
	sigma_t = np.sqrt(sigma_g ** 2*(sigma_e ** 2+sigmas[i] ** 2)/denominator)
	mus.append(mu_t)
	sigmas.append(sigma_t)



plt.figure()
plt.plot(data[:,0], data[:,1], 'o')
plt.errorbar(data[:,0], mus, 2*np.array(sigmas))
plt.legend(['hiddens', 'predictions'])
plt.suptitle("Kalman Filter Over Time")
plt.xlabel("time")
plt.ylabel("value")
plt.show()


# Part 3
data[11][2] = 10.2

mus = [mu_0]
sigmas = [sigma_0]

for i, row in enumerate(data[1:]):
	x = float(row[2])
	denominator = sigma_g ** 2 + sigma_e ** 2 + sigmas[i] ** 2
	mu_t = ((x-mu_g)*(sigma_e ** 2+sigmas[i] ** 2) + (mus[i] + mu_e)*sigma_g ** 2)/denominator
	sigma_t = np.sqrt(sigma_g ** 2*(sigma_e ** 2+sigmas[i] ** 2)/denominator)
	mus.append(mu_t)
	sigmas.append(sigma_t)



plt.figure()
plt.plot(data[:,0], data[:,1], 'o')
plt.errorbar(data[:,0], mus, 2*np.array(sigmas))
plt.legend(['hiddens', 'predictions'])
plt.suptitle("Kalman Filter Over Time, with x_10 = 10.2")
plt.xlabel("time")
plt.ylabel("value")
plt.show()