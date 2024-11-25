import numpy as np

class Tensor:
    """
    A class representing a tensor with autograd capabilities.
    
    Attributes:
        data (np.ndarray): The numerical data stored in the tensor.
        requires_grad (bool): Whether the tensor requires gradient computation.
        grad (np.ndarray): The gradient of the tensor (initialized as None).
        _backward (callable): The backward function for computing gradients.
        _prev (set): The set of parent tensors in the computational graph.
    """
    def __init__(self, data, requires_grad=False):
        self.data = np.array(data, dtype=float)
        self.requires_grad = requires_grad
        self.grad = None
        self._backward = lambda: None
        self._prev = set()

    def backward(self):
        """
        computes the gradients for the tensor as well as connected tensors
        """
        if self.grad is None:
            # For scalar tensors, initialize gradient to 1
            if self.data.ndim == 0 or self.data.size == 1:
                self.grad = np.ones_like(self.data)  # Initialize gradient for scalars
                print("backward() called so self.grad set to 1 for loss")
            else:
                raise ValueError("Gradient for the non-scalar output tensor is not initialized.")
        
        # topological sort
        topo_order = []
        visited = set()

        def build_topo(tensor):
            if tensor not in visited:
                visited.add(tensor)
                for parent in tensor._prev:
                    build_topo(parent)
                topo_order.append(tensor)
        
        build_topo(self)
        # print("Topo: ", topo_order)
        # print("visited: ", visited)

        # backprop
        for n, t in enumerate(reversed(topo_order)):
            # Initialize the gradient if it's not a scalar and the grad is None
            if t.requires_grad and t.grad is None:
                t.grad = np.zeros_like(t.data)
            print(f"backprop level {n}: ", t)
            t._backward()
    def __add__(self, other):
        """
        adds two tensors and tracks the operation in the computational graph
        """
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data)
        out._prev = {self, other}
        out.requires_grad =self.requires_grad or other.requires_grad

        def _backward():
            """
            track update of gradients in graph            
            """
            if self.requires_grad:
                self.grad = (self.grad + out.grad) if self.grad is not None else out.grad
            if other.requires_grad:
                other.grad = (other.grad + out.grad) if other.grad is not None else out.grad
        
        out._backward = _backward
        return out
    
    def __sub__(self, other):
        """
        subtracts two tensors and tracks the operation in the computational graph
        """
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data - other.data)
        out._prev = {self, other}
        out.requires_grad =self.requires_grad or other.requires_grad

        def _backward():
            """
            track update of gradients in graph            
            """
            if self.requires_grad:
                self.grad = (self.grad - out.grad) if self.grad is not None else out.grad
            if other.requires_grad:
                other.grad = (other.grad - out.grad) if other.grad is not None else out.grad
        
        out._backward = _backward
        return out
    
    def __mul__(self, other):
        """
        multiplies two tensors, tracks operation in computational graph
        """
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data)
        out._prev = {self, other}
        out.requires_grad =self.requires_grad or other.requires_grad

        def _backward():
            """
            track update of gradients in graph            
            """
            if self.requires_grad:
                self.grad = (self.grad + other.data * out.grad) if self.grad is not None else other.data * out.grad
            if other.requires_grad:
                other.grad = (other.grad + self.data * out.grad) if other.grad is not None else self.data * out.grad
        
        out._backward = _backward
        return out
    
    def dot(self, other):
        """
        computes dot product of two tensors and tracks operation in computational graph
        """
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data.dot(other.data))
        out._prev = {self, other}
        out.requires_grad =self.requires_grad or other.requires_grad

        def _backward():
            """
            track update of gradients in graph            
            """
            print("out: ", out)
            print("self:", self)
            print("other: ", other)

            if self.requires_grad: # A.grad = C.grad matmult B.grad
                self.grad = (self.grad + out.grad @ other.grad.T) if self.grad is not None else out.grad @ other.data.T
            if other.requires_grad: # B.grad = C.grad matmult A.grad
                # print("out.grad: ", out.grad)
                # print("self.grad: ", self.grad)
                # print("other.grad: ", other.grad)
                # print("self.data: ", self.data)
                # print("is out.data.ndim==0", out.data.ndim == 0)
                # print("is out.data.size==1", out.data.size)
                # print("out.data: ", out.data)
                other.grad = (other.grad + out.grad @ self.grad.T) if other.grad is not None else self.data.T @ out.grad
            # if self.requires_grad:
            #     grad_self = out.grad @ other.data.T
            #     self.grad = self.grad + grad_self if self.grad is not None else grad_self

            # if other.requires_grad:
            #     grad_other = self.data.T @ out.grad
            #     other.grad = other.grad + grad_other if other.grad is not None else grad_other

        out._backward = _backward
        return out
    
    def __pow__(self, power):
        """
        calcualates self.data^power
        """
        assert isinstance(power, (int, float)), "Power must be an int or a float"
        out = Tensor(self.data ** power)
        out._prev = {self}
        out.requires_grad = self.requires_grad
        
        def _backward():
            """
            Tracks the gradient computation for the power operation.
            """
            if self.requires_grad:
                # Derivative of x^p w.r.t. x is p * x^(p-1)
                self.grad = (self.grad + power * (self.data ** (power - 1)) * out.grad) if self.grad is not None else power * (self.data ** (power - 1)) * out.grad
        
        out._backward = _backward
        return out
    
    def sum(self):
        """
        Computes the sum of all elements in the tensor and tracks the operation in the computational graph.
        """
        out = Tensor(self.data.sum())
        out._prev = {self}
        out.requires_grad = self.requires_grad
        
        def _backward():
            """
            Tracks the gradient computation for the sum operation.
            """
            if self.requires_grad:
                # The gradient of sum w.r.t. each input element is 1
                self.grad = (self.grad + np.ones_like(self.data) * out.grad) if self.grad is not None else np.ones_like(self.data) * out.grad
        
        out._backward = _backward
        return out
    
    def __repr__(self):
        return (f"Tensor(data={self.data}, requires_grad={self.requires_grad}, "
                f"grad={self.grad}, _prev={len(self._prev)} parents)")

    
class NeuralNetwork:
    """
    A class that mimics simpleNN from torch_reference.
    consists of:
    - input: 5 neurons
    - hidden layer 1: 10 neurons
    - hidden layer 2: 8 neurons
    - hidden layer 3: 5 neurons
    - output: 1 neuron
    """

    def __init__(self):
        self.fc1_weights = Tensor(np.random.randn(5,10), requires_grad=True)
        self.fc1_bias = Tensor(np.zeros(10), requires_grad=True)

        self.fc2_weights = Tensor(np.random.randn(10,8), requires_grad=True)
        self.fc2_bias = Tensor(np.zeros(8), requires_grad=True)

        self.fc3_weights = Tensor(np.random.randn(8,5), requires_grad=True)
        self.fc3_bias = Tensor(np.zeros(5), requires_grad=True)
    
        self.fc4_weights = Tensor(np.random.randn(5,1), requires_grad=True)
        self.fc4_bias = Tensor(np.zeros(1), requires_grad=True)

        #debug
        print(f"fc1_weights: {self.fc1_weights}")
        print(f"fc2_weights: {self.fc2_weights}")
        print(f"fc3_weights: {self.fc3_weights}")
        print(f"fc4_weights: {self.fc4_weights}")

        print(f"fc1_bias: {self.fc1_bias}")
        print(f"fc2_bias: {self.fc2_bias}")
        print(f"fc3_bias: {self.fc3_bias}")
        print(f"fc4_bias: {self.fc4_bias}")
    
    def relu(self, x):
        """
        ReLU activation layer
        """
        out = Tensor(np.maximum(x.data, 0))
        out._prev = {x}

        def _backward(): #output grad only backpropegated if in the x.data positive

            # Initialize `out.grad` if it is None
            # if out.grad is None:
            #     raise ValueError("Gradient of the output must be initialized before calling backward in ReLU.")
            
            # x.grad = (x.grad + (x.data>0) * out.grad) if x.grad is not None else (x.data > 0)*out.grad
            # print("ReLU backward: out.grad =", out.grad)
            if x.requires_grad:
                relu_grad = (x.data > 0) * out.grad
                x.grad = (x.grad + relu_grad) if x.grad is not None else relu_grad
        
        out._backward  = _backward
        return out

    def forward(self, x):
        """
        compute forward pass through the network
        """
        print(f"self.fc1_weights: {self.fc1_weights}")
        x = x.dot(self.fc1_weights) + self.fc1_bias
        print(f"pre relu:{x}")
        x = self.relu(x)

        print(f"self.fc2_weights: {self.fc2_weights}")
        x = x.dot(self.fc2_weights) + self.fc2_bias
        print(f"pre relu:{x}")
        x = self.relu(x)

        print(f"self.fc3_weights: {self.fc3_weights}")
        x = x.dot(self.fc3_weights) + self.fc3_bias
        print(f"pre relu:{x}")
        x = self.relu(x)

        print(f"self.fc4_weights: {self.fc4_weights}")
        x = x.dot(self.fc4_weights) + self.fc4_bias
        print(f"pre relu:{x}")
        x = self.relu(x)
        return x

def train_one_epoch():
    """
    one pass through
    """
    #model defined above
    model = NeuralNetwork()

    # Hardcoded inputs and outputs mimicing those from torch_reference
    inputs = Tensor(np.array([[0.5, -0.2, 0.1, 0.7, -0.3]]), requires_grad=False)
    target = Tensor(np.array([[1.0]]), requires_grad=False)

    # forward pass
    outputs = model.forward(inputs)

    # compute MSE loss
    loss = ((outputs - target)**2).sum()

    #debug
    print("loss function: ", type(loss))
    print("loss data: ", loss.data)
    print("loss grad: ", loss.grad)
    print("loss requires_grad: ", loss.requires_grad)

    # print("Loss gradient (before backward):", loss.grad)
    # print("Loss data shape:", loss.data.shape)

    # Initialize the gradient of the loss explicitly (as loss is a scalar)
    # if loss.grad is None:
    #     loss.grad = np.ones_like(loss.data)
    
    print("Loss gradient (before backward):", loss.grad)
    print("Loss data shape:", loss.data.shape)
    print("loss object: ", loss)

    # backward pass
    loss.backward()

    print("Gradient from first layer: ", model.fc1_weights.grad)

    #update weights
    learning_rate = 0.01

    for param in [model.fc1_weights, model.fc1_bias,
                  model.fc2_weights, model.fc2_bias,
                  model.fc3_weights, model.fc3_bias,
                  model.fc4_weights, model.fc4_bias,
                  ]:
        if param.requires_grad:
            param.data -= learning_rate*param.grad
            param.grad = None #reset after use for any following epochs
    
    print("Deep model output: ", outputs.data)
    print("Target: ", target.data)

if __name__ == "__main__":
    # # Get methods of the class
    # methods = [attr for attr in dir(Tensor) if callable(getattr(Tensor, attr))]

    # # Print the methods
    # print("Methods of Tensor:")
    # for method in methods:
    #     print(method)
    
    train_one_epoch()
