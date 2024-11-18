import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


X = np.array([1, 3, 5, 7, 9])
y = np.array([1.2, 2.4, 2.9, 4.5, 5.1])
centers = np.array([2, 6, 8])
sigma_values = np.array([-0.6, 0.7, 1.2])


def gaussian_rbf(x, center, sigma):
    if sigma <= 0:
        sigma = abs(sigma)
    return np.exp(-(x - center) ** 2 / (2 * sigma ** 2))


def polynomial_rbf(x, center, sigma):
    return (1 + (x - center) ** 2 / (2 * sigma ** 2))


def calculate_rbf_outputs(x, centers, sigma, rbf_type='gaussian'):
    outputs = []
    for center in centers:
        if rbf_type == 'gaussian':
            output = gaussian_rbf(x, center, sigma)
        else:
            output = polynomial_rbf(x, center, sigma)
        outputs.append(output)
    return np.array(outputs)


def calculate_mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)



print("Part A: RBF Outputs Comparison")
print("\nGaussian RBF outputs:")
for sigma in sigma_values:
    print(f"\nσ = {sigma}")
    print("X\tGaussian RBF outputs for each center")
    print("-" * 50)
    for x_val in X:
        outputs = calculate_rbf_outputs(x_val, centers, sigma)
        print(f"{x_val}\t{outputs}")

print("\nPolynomial RBF outputs:")
for sigma in sigma_values:
    print(f"\nσ = {sigma}")
    print("X\tPolynomial RBF outputs for each center")
    print("-" * 50)
    for x_val in X:
        outputs = calculate_rbf_outputs(x_val, centers, sigma, 'polynomial')
        print(f"{x_val}\t{outputs}")


print("\nPart B: MSE Analysis")
print("\nGaussian RBF MSE for different σ values:")
for sigma in sigma_values:
    predictions = []
    for x_val in X:
        rbf_outputs = calculate_rbf_outputs(x_val, centers, sigma)
        pred = np.mean(rbf_outputs) * 5
        predictions.append(pred)

    mse = calculate_mse(y, predictions)
    print(f"σ = {sigma}: MSE = {mse:.4f}")


plt.figure(figsize=(15, 5))
x_plot = np.linspace(0, 10, 200)

for i, sigma in enumerate(sigma_values):
    plt.subplot(1, 3, i + 1)
    for center in centers:
        y_rbf = gaussian_rbf(x_plot, center, sigma)
        plt.plot(x_plot, y_rbf, label=f'Center={center}')
    plt.title(f'Gaussian RBF (σ={sigma})')
    plt.xlabel('x')
    plt.ylabel('RBF output')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()
