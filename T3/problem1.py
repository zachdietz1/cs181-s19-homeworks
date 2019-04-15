import matplotlib.pyplot as plt

data = [(-3,1),(-2,1),(-1,-1),(0,1),(1,-1),(2,1),(3,1)]

x1_pos, x2_pos = [], []
x1_neg, x2_neg = [], []

for x, y in data:
	if y > 0:
		x1_pos.append(x)
		x2_pos.append(-8/3 * x**2 + 2/3 * x**4)
	else:
		x1_neg.append(x)
		x2_neg.append(-8/3 * x**2 + 2/3 * x**4)

plt.figure()
plt.plot(x1_pos, x2_pos, 'bo')
plt.plot(x1_neg, x2_neg, 'ro')
plt.grid(True)
plt.xlabel("Phi_1(x)")
plt.ylabel("Phi_2(x)")
plt.show()

plt.figure()
plt.plot(x1_pos, x2_pos, 'bo')
plt.plot(x1_neg, x2_neg, 'ro')
plt.plot((-3.5,3.5), (-1,-1))
plt.xlim(-3.5,3.5)
plt.ylim(-3,32)
plt.xlabel("Phi_1(x)")
plt.ylabel("Phi_2(x)")
plt.grid(True)
plt.show()