import numpy as np

class Model:
    def __init__(self, num_inputs, output):
        hidden = 5
        self.w1 = np.random.rand(num_inputs, hidden)
        self.b1 = np.zeros(hidden)
        self.w2 = np.random.rand(hidden, output)
        self.b2 = np.zeros(output)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, out):
        return out * (1 - out)

    def forward(self, x):
        self.h1 = np.dot(x, self.w1) + self.b1
        self.ha1 = self.sigmoid(self.h1)
        self.h2 = np.dot(self.ha1, self.w2) + self.b2
        self.ha2 = self.sigmoid(self.h2)
        return self.ha2

    def backward(self, x, y, y_pred):
        de = y - y_pred 
        dha2 = de * self.sigmoid_derivative(y_pred)
        d_w2 = np.dot(self.ha1.T, dha2)
        d_b2 = np.sum(dha2, axis=0)
        dha1 = np.dot(dha2, self.w2.T) * self.sigmoid_derivative(self.ha1)
        d_w1 = np.dot(x.T, dha1)
        d_b1 = np.sum(dha1, axis=0)
        learning_rate = 0.1
        self.w1 -= learning_rate * d_w1
        self.b1 -= learning_rate * d_b1
        self.w2 -= learning_rate * d_w2
        self.b2 -= learning_rate * d_b2

    def train(self, x, y, epochs=10):
        for epoch in range(epochs):
            y_pred = self.forward(x)
            self.backward(x, y, y_pred)
            if epoch % 100 == 0:
                loss = np.mean(np.square(y - y_pred))
                print(f"Epoch {epoch}: Loss {loss:.4f}")
                print("y_pred",y_pred)
    def predict(self, x):
        return np.round(self.forward(x))

if __name__ == "__main__":
    # XOR problem
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    model = Model(num_inputs=2, output=1)
    losses = model.train(x, y, epochs=1000)
    predictions = model.predict(x)
    print("\nFinal predictions:")
    for inputs, target, pred in zip(x, y, predictions):
        print(f"Input: {inputs}, Target: {target[0]}, Predicted: {pred[0]}")
    
