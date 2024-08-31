"""
Implements VAE in PyTorch.
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
    print('Hello from vae.py!')


def get_encoder(input_size, hidden_dim, latent_size, num_classes=0):
    """
    Build and return a decoder as an nn.Sequential object.

    Inputs:
    - input_size: Python integer giving the input dimension
    - hidden_dim: Python integer giving the hidden space dimension
    - latent_size: Python integer giving the latent space dimension
    - num_classes: Python integer giving the number of classes for CVAE

    Returns:
    - model: nn.Sequential object giving the encoder
    """
    model = None
    ##########################################################################
    # TODO: Implement the fully-connected encoder architecture.              #
    # The network gets a batch of input images of shape (N, D), and          #
    # the following layers map it to hidden features of shape (N, H), and    #
    # the final layer maps to it to latent features of shape (N, 2*Z).       #
    # Wrap all layers with nn.Sequential and assign it to model.             #
    # If you revisit this from CVAE, num_classes > 0 will be introduced,     #
    # so add this number to the dimension of the first full-connected layer. #
    # Hint: nn.Linear, nn.ReLU                                               #
    ##########################################################################
    # Replace "pass" with your code (do not modify this line)

    D, H, Z = input_size, hidden_dim, latent_size
        
    model = nn.Sequential(
      nn.Linear(D+num_classes, H),
      nn.ReLU(),      
      nn.Linear(H, H),
      nn.ReLU(),
      nn.Linear(H, H),
      nn.ReLU(),
      nn.Linear(H, 2*Z)                  
    )

    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return model


def get_decoder(input_size, hidden_dim, latent_size, num_classes=0):
    """
    Build and return a decoder as an nn.Sequential object.

    Inputs:
    - input_size: Python integer giving the input dimension
    - hidden_dim: Python integer giving the hidden space dimension
    - latent_size: Python integer giving the latent space dimension
    - num_classes: Python integer giving the number of classes for CVAE

    Returns:
    - model: nn.Sequential object giving the decoder
    """
    model = None
    ##########################################################################
    # TODO: Implement the fully-connected decoder architecture.              #
    # The network gets a batch of latent vectors of shape (N, Z) as input,   #
    # and outputs a tensor of estimated images of shape (N, D).              #
    # If you revisit this from CVAE, num_classes > 0 will be introduced,     #
    # so add this number to the dimension of the first linear layer.         #
    # Hint: nn.Sequential, nn.Linear, nn.ReLU, nn.Sigmoid                    #
    ##########################################################################
    # Replace "pass" with your code (do not modify this line)

    D, H, Z = input_size, hidden_dim, latent_size
        
    model = nn.Sequential(
      nn.Linear(Z+num_classes, H),
      nn.ReLU(),      
      nn.Linear(H, H),
      nn.ReLU(),
      nn.Linear(H, H),
      nn.ReLU(),
      nn.Linear(H, D),
      nn.Sigmoid()                  
    )

    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return model


def reparametrize(mu, logvar):
    """
    Differentiably sample random Gaussian data with specified mean and
    variance using the reparameterization trick.

    Suppose we want to sample a random number z from a Gaussian distribution
    with mean mu and standard deviation sigma, such that we can backpropagate
    from the z back to mu and sigma. We can achieve this by first sampling a
    random value epsilon from a standard Gaussian distribution with zero mean
    and unit variance, then setting z = sigma * epsilon + mu.

    For more stable training when integrating this function into a neural
    network, it helps to pass this function the log of the variance of the
    distribution from which to sample, rather than specifying the standard
    deviation directly.

    Inputs:
    - mu: Tensor of shape (N, Z) giving the mean
    - logvar: Tensor of shape (N, Z) giving the log-variance

    Returns:
    - z: Estimated latent vectors, where z[i, j] is a random value sampled
      from a Gaussian with mean mu[i, j] and log-variance logvar[i, j].
    """
    z = None
    ##########################################################################
    # TODO: Reparametrize by initializing epsilon as a normal distribution   #
    # and scaling by posterior mu and sigma to estimate z.                   #
    # Hint: torch.exp, torch.randn_like                                      #
    ##########################################################################
    # Replace "pass" with your code (do not modify this line)
    eps = torch.randn_like(mu)
    z = mu + torch.exp(1/2*logvar) * eps
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return z


class VAE(nn.Module):
    def __init__(self, input_size=784, hidden_dim=256, latent_size=15):
        """
        Initialize a VAE.

        Inputs:
        - input_size: Python integer giving the input dimension
        - hidden_dim: Python integer giving the hidden space dimension
        - latent_size: Python integer giving the latent space dimension
        """
        super(VAE, self).__init__()

        self.encoder = None
        self.decoder = None
        ######################################################################
        # TODO: Implement the `__init__` function. Use `get_encoder` and     #
        # `get_decoder` to set up self.encoder and self.decoder.             #
        ######################################################################
        # Replace "pass" with your code (do not modify this line)


        self.encoder = get_encoder(input_size, hidden_dim,latent_size)
        self.decoder = get_decoder(input_size, hidden_dim,latent_size)
        

        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################

    def forward(self, x):
        """
        Performs forward pass through FC-VAE model by passing image through
        encoder, reparametrize trick, and decoder

        Inputs:
        - x: Tensor of shape (N, D) giving the batch of input images

        Returns:
        - x_hat: Tensor of shape (N, D) giving the reconstructed input data
        - mu: Tensor of shape (N, Z) giving the estimated mean
        - logvar: Tensor of shape (N, Z) giving the estimated log-variance
        """
        x_hat, mu, logvar = None, None, None
        ######################################################################
        # TODO: Implement the forward pass by following steps below:         #
        # 1. Pass the input x through the encoder model.                     #
        # 2. Split the output of the encoder to get mu and logvar.           #
        # 3. Apply reparametrization to sample the latent vector z.          #
        # 4. Pass z through the decoder to resconstruct x.                   #
        # Hint: x[:,:S], x[:,S:] slices the second dimension to [0,S), [S,D) #
        ######################################################################
        # Replace "pass" with your code (do not modify this line)

        x = self.encoder(x)
        _, Z = x.shape
        mu, logvar = x[:,:Z//2], x[:,Z//2:]
        z = reparametrize(mu, logvar)
        x_hat = self.decoder(z)

        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        return x_hat, mu, logvar


def loss_func(x, x_hat, mu, logvar):
    """
    Computes the negative variational lower bound loss term of the VAE.

    Inputs:
    - x: Tensor of shape (N, d_1, ..., d_k) giving the input data
    - x_hat: Tensor of shape (N, d_1, ..., d_k) giving the reconstructed data
    - mu: Tensor of shape (N, Z) giving the estimated mean
    - logvar: Tensor of shape (N, Z) giving the estimated log-variance

    Returns:
    - loss: Tensor of scalar giving the negative of variational lower bound
    """
    loss = None
    ##########################################################################
    # TODO: Compute the negative variational lower bound loss.               #
    # Don't forget to average the loss, dividing by the batch size N.        #
    # Hint: F.binary_cross_entropy with reduction='sum', torch.sum           #
    ##########################################################################
    # Replace "pass" with your code (do not modify this line)

    N, _ = mu.shape
    reconstruction = F.binary_cross_entropy(x_hat,x, reduction='sum')
    KL = -1/2*torch.sum(1+logvar-mu**2-torch.exp(logvar))

    loss = (reconstruction + KL)/N

    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return loss


class CVAE(nn.Module):
    def __init__(self, input_size=784, hidden_dim=256, latent_size=15,
                 num_classes=10):
        """
        Initialize a VAE.

        Inputs:
        - input_size: Python integer giving the input dimension
        - hidden_dim: Python integer giving the hidden space dimension
        - latent_size: Python integer giving the latent space dimension
        - num_classes: Python integer giving the number of classes
        """
        super(CVAE, self).__init__()

        self.encoder = None
        self.decoder = None
        ######################################################################
        # TODO: Implement the `__init__` function. Use `get_encoder` and     #
        # `get_decoder` to set up self.encoder and self.decoder.             #
        # Note that You need to additionally input `num_classes` there.      #
        ######################################################################
        # Replace "pass" with your code (do not modify this line)
        self.encoder = get_encoder(input_size, hidden_dim,latent_size,num_classes)
        self.decoder = get_decoder(input_size, hidden_dim,latent_size,num_classes)

        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################

    def forward(self, x, c):
        """
        Performs forward pass through FC-CVAE model by passing image through
        encoder, reparametrize trick, and decoder

        Inputs:
        - x: Tensor of shape (N, D) giving the batch of input images
        - c: Tensor of shape (N, C) giving the one-hot labels of input images

        Returns:
        - x_hat: Tensor of shape (N, D) giving the reconstructed input data
        - mu: Tensor of shape (N, Z) giving the estimated mean
        - logvar: Tensor of shape (N, Z) giving the estimated log-variance
        """
        x_hat = None
        mu = None
        logvar = None
        ######################################################################
        # TODO: Implement the forward pass by following steps below:         #
        # 1. Pass the concatenation of input x and one-hot vectors c         #
        #    through the encoder model.                                      #
        # 2. Split the output of the encoder to get mu and logvar.           #
        # 3. Apply reparametrization to sample the latent vector z.          #
        # 4. Pass the concatenation of z and one-hot vectors c through       #
        # the decoder to resconstruct x.                                     #
        # Hint: x[:,:S], x[:,S:] slices the second dimension to [0,S), [S,D) #
        ######################################################################
        # Replace "pass" with your code (do not modify this line)

        x = torch.cat((x,c),dim=1)
        x = self.encoder(x)
        _, Z = x.shape
        mu, logvar = x[:,:Z//2], x[:,Z//2:]
        z = reparametrize(mu, logvar)
        z = torch.cat((z,c),dim=1)
        x_hat = self.decoder(z)

        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        return x_hat, mu, logvar
