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
            self.grad = np.ones_like(self.data)
        
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

        # backprop
        for t in reversed(topo_order):
            t._backward()
    def __add__(self, other):
        """
        adds two tensors and tracks the operation in the computational graph
        """
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data)
        out._prev = {self, other}

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
    
    def __mul__(self, other):
        """
        multiplies two tensors, tracks operation in computational graph
        """
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data)
        out._prev = {self, other}

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
    
    def __dot__(self, other):
        """
        computes dot product of two tensors and tracks operation in computational graph
        """
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data.dot(other.data))
        out._prev = {self, other}

        def _backward():
            """
            track update of gradients in graph            
            """
            if self.requires_grad: # A.grad = C.grad matmult B.grad
                self.grad = (self.grad + out.grad @ other.grad.T) if self.grad is not None else out.grad @ other.data.T
            if other.requires_grad: # B.grad = C.grad matmult A.grad
                other.grad = (other.grad + out.grad @ self.grad.T) if other.grad is not None else out.grad @ self.data.T

        out._backward = _backward
        return out
    
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
        
        def relu(self, x):
            """
            ReLU activation layer
            """
            out = Tensor(np.maximum(x.data, 0))
            out._prev = {x}

            def _backward(): #output grad only backpropegated if in the x.data positive
                x.grad = (x.grad + (x.data>0) * out.grad) if x.grad is not None else (x.data > 0)*out.grad
            
            out._backward  = _backward

        def forward(self, x):
            """
            compute forward pass through the network
            """
            x = x.dot(self.fc1_weights) + self.fc1_bias
            x = self.relu(x)

            x = x.dot(self.fc2_weights) + self.fc2_bias
            x = self.relu(x)

            x = x.dot(self.fc3_weights) + self.fc3_bias
            x = self.relu(x)

            x = x.dot(self.fc4_weights) + self.fc4_bias
            x = self.relu(x)

            return x