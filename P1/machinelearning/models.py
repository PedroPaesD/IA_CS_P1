from torch import no_grad, stack
from torch.utils.data import DataLoader
from torch.nn import Module


"""
Functions you should use.
Please avoid importing any other functions or modules.
Your code will not pass if the gradescope autograder detects any changed imports
"""
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Parameter, Linear
from torch import optim, tensor, tensordot, ones, matmul
from torch.nn.functional import cross_entropy, relu, mse_loss, softmax
from torch import movedim

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import sys
sys.stderr = open(os.devnull, 'w')

import numpy as np


class PerceptronModel(Module):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.

        In order for our autograder to detect your weight, initialize it as a 
        pytorch Parameter object as follows:

        Parameter(weight_vector)

        where weight_vector is a pytorch Tensor of dimension 'dimensions'

        
        Hint: You can use ones(dim) to create a tensor of dimension dim.
        """
        super(PerceptronModel, self).__init__()
        
        "*** YOUR CODE HERE ***"
        weight_vector = torch.ones(1, dimensions)

        self.w = Parameter(weight_vector)


    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)

        The pytorch function `tensordot` may be helpful here.
        """
        "*** YOUR CODE HERE ***"
        
        return torch.tensordot(x, self.w.T, dims=1)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"

        score = self.run(x)
        if score >= 0:
            return 1
        else:
            return -1        

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        You can iterate through DataLoader in order to 
        retrieve all the batches you need to train on.

        Each sample in the dataloader is in the form {'x': features, 'label': label} where label
        is the item we need to predict based off of its features.
        """        
        with no_grad():
            dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
            "*** YOUR CODE HERE ***"

            no_change = True
            correct_count = 0
            # Loop until convergence
            # Convergence is when the perceptron classifies all points correctly
            while no_change:
                for sample in dataloader:
                    x = sample['x']
                    y = sample['label']
                    prediction = self.get_prediction(x)
                    if prediction != y:
                        self.w += y * x
                    else:
                        correct_count += 1
                
                if correct_count == len(dataloader):
                    no_change = False
                    break

                correct_count = 0

class RegressionModel(Module):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        super().__init__()

        self.linear_relu_stack = nn.Sequential(
            Linear(1, 128),
            nn.ReLU(),
            Linear(128, 512),
            nn.ReLU(),
            Linear(512, 256),
            nn.ReLU(),
            Linear(256, 1)
        )


    def forward(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"

        return self.linear_relu_stack(x)
    
    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a tensor of size 1 containing the loss
        """
        "*** YOUR CODE HERE ***"
        y_pred = self.forward(x)
        loss = mse_loss(y_pred, y)
        return loss        
        

    def train(self, dataset):
        """
        Trains the model.

        In order to create batches, create a DataLoader object and pass in `dataset` as well as your required 
        batch size. You can look at PerceptronModel as a guideline for how you should implement the DataLoader

        Each sample in the dataloader object will be in the form {'x': features, 'label': label} where label
        is the item we need to predict based off of its features.

        Inputs:
            dataset: a PyTorch dataset object containing data to be trained on
            
        """
        "*** YOUR CODE HERE ***"

        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

        optimizer = optim.Adam(self.parameters(), lr=0.001)

        n_epochs = 200

        for epoch in range(n_epochs):
            for sample in dataloader:
                x = sample['x']
                y = sample['label']
                
                # Compute loss
                loss = self.get_loss(x, y)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()  







class DigitClassificationModel(Module):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        super().__init__()
        input_size = 28 * 28
        output_size = 10
        "*** YOUR CODE HERE ***"

        self.linear_relu_stack = nn.Sequential(
            Linear(input_size, 128),
            nn.ReLU(),
            Linear(128, 512),
            nn.ReLU(),
            Linear(512, 256),
            nn.ReLU(),
            Linear(256, output_size)
        )



    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a tensor with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        """ YOUR CODE HERE """
        return self.linear_relu_stack(x)
 

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a tensor with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss tensor
        """
        """ YOUR CODE HERE """
        y_pred = self.run(x)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(y_pred, y)
        return loss
    
        

    def train(self, dataset):
        """
        Trains the model.
        """
        """ YOUR CODE HERE """

        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        optimizer = optim.Adam(self.parameters(), lr=0.001)

        n_epochs = 5
        for epoch in range(n_epochs):
            for sample in dataloader:
                x = sample['x']
                y = sample['label']
                
                # Compute loss
                loss = self.get_loss(x, y)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

class LanguageIDModel(Module):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]
        super(LanguageIDModel, self).__init__()
        "*** YOUR CODE HERE ***"
        
        self.initial_layer = nn.Sequential(
            Linear(self.num_chars, 512),
            nn.ReLU()
        )
        self.hidden_layer = nn.Sequential(
            Linear(512, 256),
            nn.ReLU(),
            Linear(256, 512),
            nn.ReLU()
        )
        self.final_layer = nn.Sequential(
            Linear(512, 128),
            nn.ReLU(),
            Linear(128, 5)
        )


    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        tensor with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a tensor that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single tensor of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a tensor of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        h1 = self.initial_layer(xs[0])

        for i in range(1, len(xs)):
            h1 = self.initial_layer(xs[i]) + self.hidden_layer(h1)
            h1 = nn.ReLU()(h1)

        return self.final_layer(h1)

    
    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        y_pred = self.run(xs)
        loss_fn = nn.CrossEntropyLoss()
        #print(f"y_pred:{y_pred.shape}, y:{y.shape}")
        loss = loss_fn(y_pred, y)
        return loss


    def train(self, dataset):
        """
        Trains the model.

        Note that when you iterate through dataloader, each batch will returned as its own vector in the form
        (batch_size x length of word x self.num_chars). However, in order to run multiple samples at the same time,
        get_loss() and run() expect each batch to be in the form (length of word x batch_size x self.num_chars), meaning
        that you need to switch the first two dimensions of every batch. This can be done with the movedim() function 
        as follows:

        movedim(input_vector, initial_dimension_position, final_dimension_position)

        For more information, look at the pytorch documentation of torch.movedim()
        """
        "*** YOUR CODE HERE ***"
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        optimizer = optim.Adam(self.parameters(), lr=0.001)

        n_epochs = 15
        for epoch in range(n_epochs):
            for sample in dataloader:
                x = sample['x']
                y = sample['label']
                #print(f"x:{x.shape}, y:{y.shape}")
                x = movedim(x, 1, 0)
                # Compute loss
                loss = self.get_loss(x, y)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        

def Convolve(input: tensor, weight: tensor):
    """
    Acts as a convolution layer by applying a 2d convolution with the given inputs and weights.
    DO NOT import any pytorch methods to directly do this, the convolution must be done with only the functions
    already imported.

    There are multiple ways to complete this function. One possible solution would be to use 'tensordot'.
    If you would like to index a tensor, you can do it as such:

    tensor[y:y+height, x:x+width]

    This returns a subtensor who's first element is tensor[y,x] and has height 'height, and width 'width'
    """
    input_tensor_dimensions = input.shape
    weight_dimensions = weight.shape
    Output_Tensor = tensor(())
    "*** YOUR CODE HERE ***"

    #input = input.detach().numpy()
    #weight = weight.detach().numpy()

    weight = torch.tensor(weight, dtype=torch.float32)

    #weight = np.flipud(np.fliplr(weight))  # Flip the kernel
    #Output_Tensor = np.zeros_like(input)  # Create an output matrix of the same size
    output_height = input.shape[0] - weight.shape[0] + 1
    output_width = input.shape[1] - weight.shape[1] + 1
    Output_Tensor = torch.zeros((output_height, output_width))
    #print(f"output shape: {Output_Tensor.shape}")

    # Pad the matrix with zeros on all sides
    #pad_size = weight.shape[0] // 2
    #padded_matrix = np.pad(input, pad_width=((pad_size, pad_size), (pad_size, pad_size)), mode='constant', constant_values=0)

    # Iterate over every pixel of the matrix
    for x in range(output_height):
        for y in range(output_width):
            # Perform element-wise multiplication and sum the result
            #Output_Tensor[x, y] = (weight * padded_matrix[x:x+weight_dimensions[0], y:y+weight_dimensions[1]]).sum()
            Output_Tensor[x, y] = torch.sum(weight * input[x:x+weight_dimensions[0], y:y+weight_dimensions[1]])


    "*** End Code ***"
    return torch.tensor(Output_Tensor, dtype=torch.float32)



class DigitConvolutionalModel(Module):
    """
    A model for handwritten digit classification using the MNIST dataset.

    This class is a convolutational model which has already been trained on MNIST.
    if Convolve() has been correctly implemented, this model should be able to achieve a high accuracy
    on the mnist dataset given the pretrained weights.

    Note that this class looks different from a standard pytorch model since we don't need to train it
    as it will be run on preset weights.
    """
    

    def __init__(self):
        # Initialize your model parameters here
        super().__init__()
        output_size = 10

        self.convolution_weights = Parameter(ones((3, 3)))
        """ YOUR CODE HERE """
        self.linear_relu_stack = nn.Sequential(
            Linear(26 * 26, 128),
            nn.ReLU(),
            Linear(128, 512),
            nn.ReLU(),
            Linear(512, 256),
            nn.ReLU(),
            Linear(256, output_size)
        )



    def run(self, x):
        return self(x)
 
    def forward(self, x):
        """
        The convolutional layer is already applied, and the output is flattened for you. You should treat x as
        a regular 1-dimentional datapoint now, similar to the previous questions.
        """
        #print(f"x:{x.shape}")
        x = x.reshape(len(x), 28, 28)

        convo_weights = self.convolution_weights.detach().numpy()
        #print(f"x:{x.shape}")
        x = stack(list(map(lambda sample: Convolve(sample, convo_weights), x)))
        #print(f"x:{x.shape}")
        x = x.flatten(start_dim=1)
        #print(f"x:{x.shape}")
        """ YOUR CODE HERE """
        return self.linear_relu_stack(x)

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a tensor with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss tensor
        """
        """ YOUR CODE HERE """
        y_pred = self.forward(x)
        loss_fn = nn.CrossEntropyLoss()
        #y_pred.requires_grad = True
        #print(f"y_pred shape: {y_pred.shape}")  # Should be (batch_size, 10)
        #print(f"y shape: {y.shape}")  # Should be (batch_size)
        y_idx = y.argmax(dim=1)  # Convert one-hot to class indices
        #print(f"y_pred:{y_pred.shape}, y:{y.shape}")
        loss = loss_fn(y_pred, y_idx)
        return loss
     
        

    def train(self, dataset):
        """
        Trains the model.
        """
        """ YOUR CODE HERE """
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        optimizer = optim.Adam(self.parameters(), lr=0.001)

        n_epochs = 5
        for epoch in range(n_epochs):
            
            for sample in dataloader:
                x = sample['x']
                y = sample['label']
            
                #print(f"x:{x.shape}, y:{y.shape}")
                #x = movedim(x, 1, 0)
                # Compute loss
                loss = self.get_loss(x, y)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                
                optimizer.step()


class Attention(Module):
    def __init__(self, layer_size, block_size):
        super().__init__()
        """
        All the layers you should use are defined here.

        In order to pass the autograder, make sure each linear layer matches up with their corresponding matrix,
        ie: use self.k_layer to generate the K matrix.
        """
        self.k_layer = Linear(layer_size, layer_size)
        self.q_layer = Linear(layer_size, layer_size)
        self.v_layer = Linear(layer_size,layer_size)

        #Masking part of attention layer
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size))
                                     .view(1, 1, block_size, block_size))
       
        self.layer_size = layer_size


    def forward(self, input):
        """
        Applies the attention mechanism to input. All necessary layers have 
        been defined in __init__()

        In order to apply the causal mask to a given matrix M, you should update
        it as such:
    
        M = M.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))[0]

        For the softmax activation, it should be applied to the last dimension of the input,
        Take a look at the "dim" argument of torch.nn.functional.softmax to figure out how to do this.
        """
        B, T, C = input.size()

        """YOUR CODE HERE"""
        
        inside_term = torch.matmul(self.k_layer(input), movedim(self.q_layer(input), 1, 2))
        inside_term /= self.layer_size ** (1/2)

        M = inside_term.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))[0]
        soft_result = torch.nn.functional.softmax(M, dim=-1)

        return torch.matmul(soft_result, self.v_layer(input))