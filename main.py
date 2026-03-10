from scipy.io import loadmat
import numpy as np
from scipy.optimize import minimize


# random intialize
def initialise(a, b):
	epsilon = 0.15
	c = np.random.rand(a, b + 1) * (2 * epsilon) - epsilon 
	return c

#prediction
def predict(Theta1, Theta2, X):
	m = X.shape[0]
	one_matrix = np.ones((m, 1))
	X = np.append(one_matrix, X, axis=1) 
	z2 = np.dot(X, Theta1.transpose())
	a2 = 1 / (1 + np.exp(-z2))
	one_matrix = np.ones((m, 1))
	a2 = np.append(one_matrix, a2, axis=1)
	z3 = np.dot(a2, Theta2.transpose())
	a3 = 1 / (1 + np.exp(-z3))
	p = (np.argmax(a3, axis=1))
	return p

#model
def neural_network(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lamb):
	Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
						(hidden_layer_size, input_layer_size + 1))
	Theta2 = np.reshape(nn_params[hidden_layer_size * (input_layer_size + 1):], 
						(num_labels, hidden_layer_size + 1))

	# Forward propagation
	m = X.shape[0]
	one_matrix = np.ones((m, 1))
	X = np.append(one_matrix, X, axis=1) 
	a1 = X
	z2 = np.dot(X, Theta1.transpose())
	a2 = 1 / (1 + np.exp(-z2)) 
	one_matrix = np.ones((m, 1))
	a2 = np.append(one_matrix, a2, axis=1) 
	z3 = np.dot(a2, Theta2.transpose())
	a3 = 1 / (1 + np.exp(-z3)) 
	y_vect = np.zeros((m, 10))
	for i in range(m):
		y_vect[i, int(y[i])] = 1

	# Calculating cost function
	J = (1 / m) * (np.sum(np.sum(-y_vect * np.log(a3) - (1 - y_vect) * np.log(1 - a3)))) + (lamb / (2 * m)) * (
				sum(sum(pow(Theta1[:, 1:], 2))) + sum(sum(pow(Theta2[:, 1:], 2))))

	# back propogation
	Delta3 = a3 - y_vect
	Delta2 = np.dot(Delta3, Theta2) * a2 * (1 - a2)
	Delta2 = Delta2[:, 1:]

	# gradient
	Theta1[:, 0] = 0
	Theta1_grad = (1 / m) * np.dot(Delta2.transpose(), a1) + (lamb / m) * Theta1
	Theta2[:, 0] = 0
	Theta2_grad = (1 / m) * np.dot(Delta3.transpose(), a2) + (lamb / m) * Theta2
	grad = np.concatenate((Theta1_grad.flatten(), Theta2_grad.flatten()))

	return J, grad


data = loadmat(r'C:\Users\T R SREERAM\Desktop\mpsem3\mnist-original.mat')

X = data['data']
X = X.transpose()

X = X / 255

y = data['label']
y = y.flatten()

# Splitting data into training set (60,000 eg)
X_train = X[:80000, :]
y_train = y[:80000]

# Splitting data into testing set (10,000 eg)
X_test = X[10000:, :]
y_test = y[10000:]

m = X.shape[0]
input_layer_size = 784
hidden_layer_size = 100
num_labels = 10 

initial_Theta1 = initialise(hidden_layer_size, input_layer_size)
initial_Theta2 = initialise(num_labels, hidden_layer_size)


initial_nn_params = np.concatenate((initial_Theta1.flatten(), initial_Theta2.flatten()))
maxiter = 100
lambda_reg = 0.1 
myargs = (input_layer_size, hidden_layer_size, num_labels, X_train, y_train, lambda_reg)


results = minimize(neural_network, x0=initial_nn_params, args=myargs, 
		options={'disp': True, 'maxiter': maxiter}, method="L-BFGS-B", jac=True)

nn_params = results["x"] 


Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)], (
							hidden_layer_size, input_layer_size + 1)) 
Theta2 = np.reshape(nn_params[hidden_layer_size * (input_layer_size + 1):], 
					(num_labels, hidden_layer_size + 1)) 

# checking test set accuracy of  model
pred = predict(Theta1, Theta2, X_test)
print('Test Set Accuracy: {:f}'.format((np.mean(pred == y_test) * 100)))

# Checking train set accuracy of model
pred = predict(Theta1, Theta2, X_train)
print('Training Set Accuracy: {:f}'.format((np.mean(pred == y_train) * 100)))

# Evaluating precision of model
true_positive = 0
for i in range(len(pred)):
	if pred[i] == y_train[i]:
		true_positive += 1
false_positive = len(y_train) - true_positive
print('Precision =', true_positive/(true_positive + false_positive))

# Saving Thetas in .txt file
np.savetxt('Theta1.txt', Theta1, delimiter=' ')
np.savetxt('Theta2.txt', Theta2, delimiter=' ')

