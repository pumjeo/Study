"""
Implements GAN in PyTorch.
WARNING: you SHOULD NOT use ".to()" or ".cuda()" in each implementation block.
"""
import torch
from torch import nn
from torch.nn import functional as F


def hello():
    """
    This is a sample function that we will try to import and run to ensure
    that our environment is correctly set up on Google Colab.
    """
    print("Hello from gan.py!")


def sample_noise(batch_size, noise_size):
    """
    Generate a PyTorch Tensor of uniform random noise in the range [-1,1].

    Input:
    - batch_size: Python integer giving the batch size of noise to generate.
    - noise_size: Python integer giving the dimension of noise to generate.

    Output:
    - Tensor of shape (batch_size, noise_size) containing uniform random
      noise in the range [-1,1].
    """
    noise = None
    ##########################################################################
    # TODO: Implement sample_noise using torch.rand                          #
    ##########################################################################
    # Replace "pass" with your code (do not modify this line)
    noise = -2 * torch.rand(batch_size, noise_size) +1
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return noise


def get_discriminator(input_size, hidden_dim):
    """
    Build and return a PyTorch nn.Sequential model for the discriminator.
    """

    model = None
    ##########################################################################
    # TODO: Implement the fully-connected discriminator architecture.        #
    # The network gets a batch of input images of shape (N, D), and          #
    # the following layers map it to hidden features of shape (N, H), and    #
    # the final layer maps to it to a scalar value (N, 1).                   #
    # Wrap all layers with nn.Sequential and assign it to model.             #
    # Hint: nn.Linear, nn.LeakyReLU                                          #
    ##########################################################################
    # Replace "pass" with your code (do not modify this line)
    D, H = input_size, hidden_dim
        
    model = nn.Sequential(
      nn.Linear(D, H),
      nn.LeakyReLU(0.01),      
      nn.Linear(H, H),
      nn.LeakyReLU(0.01),      
      nn.Linear(H, 1)             
    )
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return model


def get_generator(noise_size, hidden_dim, input_size):
    """
    Build and return a PyTorch nn.Sequential model for the generator.
    """

    model = None
    ##########################################################################
    # TODO: Implement the fully-connected generator architecture.            #
    # The network gets a batch of latent vectors of shape (N, Z) as input,   #
    # and outputs a tensor of estimated images of shape (N, D).              #
    # Wrap all layers with nn.Sequential and assign it to model.             #
    # Hint: nn.Linear, nn.ReLU, nn.Tanh                                      #
    ##########################################################################
    # Replace "pass" with your code (do not modify this line)
    D, H, Z = input_size, hidden_dim, noise_size
        
    model = nn.Sequential(
      nn.Linear(Z, H),
      nn.ReLU(),      
      nn.Linear(H, H),
      nn.ReLU(),      
      nn.Linear(H, D),
      nn.Tanh()        
    )
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return model


def discriminator_loss(logits_real, logits_fake):
    """
    Computes the discriminator loss.

    Inputs:
    - logits_real: Tensor of shape (N,) giving scores for the real data.
    - logits_fake: Tensor of shape (N,) giving scores for the fake data.

    Returns:
    - loss: Tensor of scalar giving the loss for the discriminator.
    """
    loss = None
    ##########################################################################
    # TODO: Compute the discriminator loss.                                  #
    # Hint: F.binary_cross_entropy_with_logits,                              #
    # torch.zeros_like, torch.ones_like                                      #
    ##########################################################################
    # Replace "pass" with your code (do not modify this line)
    true_labels = torch.ones_like(logits_real)
    false_labels = torch.zeros_like(logits_fake)

    a = F.binary_cross_entropy_with_logits(logits_real, true_labels)
    b = F.binary_cross_entropy_with_logits(logits_fake, false_labels)
    loss = a+b
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return loss


def generator_loss(logits_fake):
    """
    Computes the generator loss.

    Inputs:
    - logits_fake: Tensor of shape (N,) giving scores for the fake data.

    Returns:
    - loss: Tensor of scalar giving the loss for the generator.
    """
    loss = None
    ##########################################################################
    # TODO: Compute the generator loss.                                      #
    # Hint: F.binary_cross_entropy_with_logits, torch.ones_like              #
    ##########################################################################
    # Replace "pass" with your code (do not modify this line)

    false_labels = torch.ones_like(logits_fake)
    loss = F.binary_cross_entropy_with_logits(logits_fake, false_labels)

    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return loss


def ls_discriminator_loss(logits_real, logits_fake):
    """
    Compute the discriminator loss for Least Squares GAN.

    Inputs:
    - logits_real: Tensor of shape (N,) giving scores for the real data.
    - logits_fake: Tensor of shape (N,) giving scores for the fake data.

    Returns:
    - loss: Tensor of scalar giving the loss for the discriminator.
    """
    loss = None
    ##########################################################################
    # TODO: Compute the discriminator loss for Least Squares GAN.            #
    ##########################################################################
    # Replace "pass" with your code (do not modify this line)
    real = 1/2*torch.mean(((logits_real-1)**2))
    fake = 1/2*torch.mean((logits_fake**2))

    loss = real + fake
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return loss


def ls_generator_loss(logits_fake):
    """
    Compute the generator loss for Least Squares GAN.

    Inputs:
    - logits_fake: Tensor of shape (N,) giving scores for the fake data.

    Returns:
    - loss: Tensor of scalar giving the loss for the generator.
    """
    loss = None
    ##########################################################################
    # TODO: Compute the generator loss for Least Squares GAN.                #
    ##########################################################################
    # Replace "pass" with your code (do not modify this line)
    loss = 1/2*torch.mean(((logits_fake-1)**2))

    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return loss


def get_dc_discriminator():
    """
    Build and return a PyTorch nn.Sequential model for the DCGAN discriminator.
    """
    model = None
    ##########################################################################
    # TODO: Implement the deep convolutional GAN discriminator architecture. #
    # Hint: nn.Unflatten                                                     #
    ##########################################################################
    # Replace "pass" with your code (do not modify this line)
    model = nn.Sequential(
      nn.Unflatten(1,(1,28,28)),
      nn.Conv2d(1, 32, 5),
      nn.LeakyReLU(0.01),      
      nn.MaxPool2d(2,stride=2),
      nn.Conv2d(32, 64, 5),
      nn.LeakyReLU(0.01),      
      nn.MaxPool2d(2,stride=2),
      nn.Flatten(),
      nn.Linear(64*4*4, 64*4*4),
      nn.LeakyReLU(0.01),      
      nn.Linear(4*4*64, 1)          
    )
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return model


def get_dc_generator(noise_size):
    """
    Build and return a PyTorch nn.Sequential model for the DCGAN generator.
    """
    model = None
    ##########################################################################
    # TODO: Implement the deep convolutional GAN generator architecture.     #
    # Hint: nn.Unflatten, nn.ConvTranspose2d                                 #
    ##########################################################################
    # Replace "pass" with your code (do not modify this line)
    model = nn.Sequential(
      nn.Linear(noise_size, 1024),
      nn.ReLU(),      
      nn.BatchNorm1d(1024),
      nn.Linear(1024, 7*7*128),
      nn.ReLU(),      
      nn.BatchNorm1d(7*7*128),
      nn.Unflatten(1,(128,7,7)),
      nn.ConvTranspose2d(128, 64, 4, 2, 1),
      nn.ReLU(),      
      nn.BatchNorm2d(64),
      nn.ConvTranspose2d(64, 1, 4, 2, 1),
      nn.Tanh(),   
      nn.Flatten()      
    )
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return model
