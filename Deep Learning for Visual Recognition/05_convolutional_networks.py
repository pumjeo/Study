"""
Implements convolutional networks in PyTorch.
WARNING: you SHOULD NOT use ".to()" or ".cuda()" in each implementation block.
You are NOT allowed to use torch.nn ops, unless otherwise specified.
"""
import torch

from common.helpers import softmax_loss
from common import Solver
from fully_connected_networks import sgd_momentum, rmsprop, adam


def hello():
    """
    This is a sample function that we will try to import and run to ensure
    that our environment is correctly set up on Google Colab.
    """
    print('Hello from convolutional_networks.py!')


class Conv(object):

    @staticmethod
    def forward(x, w, b, conv_param):
        """
        A naive implementation of the forward pass for a convolutional layer.
        The input consists of N data points, each with C channels, height H and
        width W. We convolve each input with F different filters, where each
        filter spans all C channels and has height HH and width WW.

        Input:
        - x: Input data of shape (N, C, H, W)
        - w: Filter weights of shape (F, C, HH, WW)
        - b: Biases, of shape (F,)
        - conv_param: A dictionary with the following keys:
          - 'stride': The number of pixels between adjacent receptive fields
            in the horizontal and vertical directions.
          - 'pad': The number of pixels that is used to zero-pad the input.

        During padding, 'pad' zeros should be placed symmetrically (i.e equally
        on both sides) along the height and width axes of the input. Be careful
        not to modify the original input x directly.

        Returns a tuple of:
        - out: Output data of shape (N, F, H', W') where H' and W' are given by
          H' = 1 + (H + 2 * pad - HH) / stride
          W' = 1 + (W + 2 * pad - WW) / stride
        - cache: (x, w, b, conv_param)
        """
        out = None
        ######################################################################
        # TODO: Implement the convolutional forward pass.                    #
        # Hint: You can use function torch.nn.functional.pad for padding.    #
        # You are NOT allowed to use anything in torch.nn in other places.   #
        ######################################################################
        # Replace "pass" with your code (do not modify this line)
        
        # input에 있는 shape를 꺼낸다. 
        stride = conv_param['stride']
        pad = conv_param['pad']

        N, C, H, W = x.shape
        F, C, HH, WW = w.shape

        # Output의 Height과 Width를 설정하고 Padding한 결과를 x에 지정한다.
        Hout = 1 + (H + 2 * pad - HH) // stride
        Wout = 1 + (W + 2 * pad - WW) // stride
        x = torch.nn.functional.pad(x, (pad, pad, pad, pad))

        # output을 N*F*Hout*Wout 사이즈의 0행렬로 구성하고 for loop를 통해 연산을 이어나간다.
        out = torch.zeros((N, F, Hout, Wout), dtype = x.dtype, device = x.device)
        for n in range(N):
          for f in range(F):
            for i in range(Hout):
              for j in range(Wout):
                # 여기서 input인 x의 n번째 데이터, 모든 필터, i*stride부터 필터 행만큼, j*stride부터 필터 열만큼 convolution 진행
                out[n,f,i,j] = (x[n,:, i * stride : i * stride + HH, j * stride : j * stride + WW] * w[f]).sum() + b[f]

        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        cache = (x, w, b, conv_param)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        A naive implementation of the backward pass for a convolutional layer.
          Inputs:
        - dout: Upstream derivatives.
        - cache: A tuple of (x, w, b, conv_param) as in Conv.forward

        Returns a tuple of:
        - dx: Gradient with respect to x
        - dw: Gradient with respect to w
        - db: Gradient with respect to b
        """
        dx, dw, db = None, None, None
        ######################################################################
        # TODO: Implement the convolutional backward pass.                   #
        # Hint: You can use function torch.nn.functional.pad for padding.    #
        # You are NOT allowed to use anything in torch.nn in other places.   #
        ######################################################################
        # Replace "pass" with your code (do not modify this line)
        
        x, w, b, conv_param = cache

        dx = torch.zeros_like(x)
        dw = torch.zeros_like(w)
        db = torch.zeros_like(b)
        
        N, F, Hout, Wout = dout.shape
        F, C, HH, WW = w.shape
        pad = conv_param['pad']
        stride = conv_param['stride']

        # 역시나 for loop를 이용해 주어진 수학식을 코드로 구현.
        # 기본적으로 out = x * w + b이므로 x를 업데이트할 때는 upstream인 dout에 w를,
        # w를 업데이트할 때는 upstream인 dout에 x를, b를 업데이트할 때는 dout이 그대로 전달된다.
        for  n in range(N):
          for f in range(F):
            for i in range(Hout):
              for j in range(Wout):
                # 각 필터에 따른 w * dout만큼의 결과를 반복문동안 dx에 계속 더해준다.
                dx[n, :, i * stride: i * stride +  HH, j * stride : j * stride + WW] += w[f] * dout[n,f,i,j]
                # 각 필터에 따른 dout * x만큼의 결과를 반복문동안 dw에 계속 더해준다.
                dw[f] += dout[n,f,i,j] * x[n, :, i * stride: i * stride +  HH, j * stride : j * stride + WW]
                # 각 필터에 따른 dout 만큼의 결과를 반복문동안 db에 계속 더해준다.
                db[f] += dout[n,f,i,j]

        # 기존 dx는 padding이 된 결과이므로 padding을 제거한 결과를 도출한다.
        dx = dx[:,:,pad:-1 * pad,pad:-1 * pad]

        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        return dx, dw, db


class MaxPool(object):

    @staticmethod
    def forward(x, pool_param):
        """
        A naive implementation of the forward pass for a max-pooling layer.

        Inputs:
        - x: Input data, of shape (N, C, H, W)
        - pool_param: dictionary with the following keys:
          - 'pool_height': The height of each pooling region
          - 'pool_width': The width of each pooling region
          - 'stride': The distance between adjacent pooling regions
        No padding is necessary here.

        Returns a tuple of:
        - out: Output of shape (N, C, H', W') where H' and W' are given by
          H' = 1 + (H - pool_height) / stride
          W' = 1 + (W - pool_width) / stride
        - cache: (x, pool_param)
        """
        out = None
        ######################################################################
        # TODO: Implement the max-pooling forward pass.                      #
        ######################################################################
        # Replace "pass" with your code (do not modify this line)
        
        pH = pool_param['pool_height']
        pW = pool_param['pool_width']
        stride = pool_param['stride']
        N, C, H, W = x.shape

        # Pooling의 결과인 output의 shape를 지정해 준다. (식은 주어짐)
        Hout = 1 + (H - pH) // stride
        Wout = 1 + (W - pW) // stride

        out = torch.zeros((N, C, Hout, Wout), dtype = x.dtype, device = x.device)

        # input인 x에서 filter만큼의 영역에 max값을 out의 i, j index 값에 넣는다.
        for n in range(N):
          for i in range(Hout):
            for j in range(Wout):
              out[n, :, i, j], _ = x[n, :, i * stride : i * stride + pH, j * stride : j * stride + pW].reshape(C, -1).max(axis = 1)

        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        cache = (x, pool_param)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        A naive implementation of the backward pass for a max-pooling layer.
        Inputs:
        - dout: Upstream derivatives
        - cache: A tuple of (x, pool_param) as in the forward pass.
        Returns:
        - dx: Gradient with respect to x
        """
        dx = None
        ######################################################################
        # TODO: Implement the max-pooling backward pass.                     #
        ######################################################################
        # Replace "pass" with your code (do not modify this line)
        
        x, pool_param = cache
        pH = pool_param['pool_height']
        pW = pool_param['pool_width']
        stride = pool_param['stride']
        N, C, H, W = x.shape

        Hout = 1 + (H - pH) // stride
        Wout = 1 + (W - pW) // stride
        dx = torch.zeros_like(x)


        for n in range(N):
          for i in range(Hout):
            for j in range(Wout):
              # 해당 필터의 receptive field를 local이라고 정의하고 shape을 local_shape에 저장한다.
              local = x[n, :, i * stride : i * stride + pH, j * stride : j * stride + pW]
              local_shape = local.shape
              # local은 정사각형 모양이므로 C*1로 늘린다. local의 gradient를 저장할 local_dx를 정의한다.
              local = local.reshape(C, -1)
              local_dx = torch.zeros_like(local)
              # local 중에서 가장 큰 값의 index를 추출한다.
              _, idx = local.max(axis = 1)
              # 해당 index에만 gradient가 그대로 전해지므로 dout을 그대로 local_dx에 저장한다.
              local_dx[range(C), idx] = dout[n, :, i, j]
              # local_dx를 다시 local_shape로 reshape한 결과를 receptive field인 dx에 저장한다.
              dx[n, :, i * stride : i * stride + pH, j * stride : j * stride + pW] = local_dx.reshape(local_shape)

        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        return dx


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:
    conv - relu - 2x2 max pool - linear - relu - linear - softmax
    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self,
                 input_dims=(3, 32, 32),
                 num_filters=32,
                 filter_size=7,
                 hidden_dim=100,
                 num_classes=10,
                 weight_scale=1e-3,
                 reg=0.0,
                 dtype=torch.float,
                 device='cpu'):
        """
        Initialize a new network.
        Inputs:
        - input_dims: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Width/height of filters to use in convolutional layer
        - hidden_dim: Number of units to use in fully-connected hidden layer
        - num_classes: Number of scores to produce from the final linear layer.
        - weight_scale: Scalar giving standard deviation for random
          initialization of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: A torch data type object; all computations will be performed
          using this datatype. float is faster but less accurate, so you
          should use double for numeric gradient checking.
        - device: device to use for computation. 'cpu' or 'cuda'
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ######################################################################
        # TODO: Initialize weights and biases for three-layer convolutional  #
        # network. Weights should be initialized from the Gaussian           #
        # distribution with the mean of 0.0 and the standard deviation of    #
        # weight_scale; biases should be initialized to zero. All weights    #
        # and biases should be stored in the dictionary self.params.         #
        # Store weights and biases for the convolutional layer using the     #
        # keys 'W1' and 'b1'; use keys 'W2' and 'b2' for the weights and     #
        # biases of the hidden linear layer, and keys 'W3' and 'b3' for the  #
        # weights and biases of the output linear layer.                     #
        #                                                                    #
        # IMPORTANT: For this assignment, you can assume that the padding    #
        # and stride of the first convolutional layer are chosen so that     #
        # **the width and height of the input are preserved**. Take a        #
        # look at the start of the loss() function to see how that happens.  #
        ######################################################################
        # Replace "pass" with your code (do not modify this line)
        
        # input 사이즈가 output에서도 유지되게끔 stride와 pad 설정
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
        C, H, W = input_dims
        HH = filter_size
        WW = filter_size

        # convolution과 pooling layer의 output 사이즈를 이전과 같이 정의
        conv_Hout = 1 + (H + 2 * conv_param['pad'] - HH) // conv_param['stride']
        conv_Wout = 1 + (W + 2 * conv_param['pad'] - WW) // conv_param['stride']
        pool_Hout = 1 + (conv_Hout - pool_param['pool_height']) // pool_param['stride']
        pool_Wout = 1 + (conv_Wout - pool_param['pool_width']) // pool_param['stride']

        # self.params의 W1~W3과 b1~b3을 초기화시켜 준다. W는 Gaussian 분포로, b는 0행렬로 초기화해 준다.
        self.params['W1'] = torch.normal(0.0, weight_scale, (num_filters, C, filter_size, filter_size), dtype = dtype, device = device)
        self.params['b1'] = torch.zeros(num_filters, dtype = dtype, device = device)

        self.params['W2'] = torch.normal(0.0, weight_scale, (num_filters * pool_Hout * pool_Wout, hidden_dim), dtype = dtype, device = device)
        self.params['b2'] = torch.zeros(hidden_dim, dtype = dtype, device = device)

        self.params['W3'] = torch.normal(0.0, weight_scale, (hidden_dim, num_classes), dtype = dtype, device = device)
        self.params['b3'] = torch.zeros(num_classes, dtype = dtype, device = device)

        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################

    def save(self, path):
        checkpoint = {
          'reg': self.reg,
          'dtype': self.dtype,
          'params': self.params,
        }
        torch.save(checkpoint, path)
        print("Saved in {}".format(path))

    def load(self, path):
        checkpoint = torch.load(path, map_location='cpu')
        self.params = checkpoint['params']
        self.dtype = checkpoint['dtype']
        self.reg = checkpoint['reg']
        print("load checkpoint file: {}".format(path))

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.
        Input / output: Same API as TwoLayerNet.
        """
        X = X.to(self.dtype)
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # Pass conv_param to the forward pass for the convolutional layer.
        # Padding and stride chosen to preserve the input spatial size.
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # Pass pool_param to the forward pass for the max-pooling layer.
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ######################################################################
        # TODO: Implement the forward pass for three-layer convolutional     #
        # net, computing the class scores for X and storing them in the      #
        # scores variable.                                                   #
        # Use sandwich layers if Linear or Conv layers followed by ReLU      #
        # and/or Pool layers for efficient implementation.                   #
        ######################################################################
        # Replace "pass" with your code (do not modify this line)
        
        # Conv_ReLU_Pool.forward 함수를 이용해 Convolution -> ReLU -> Pooling의 결과 도출
        CRP_out, CRP_cache = Conv_ReLU_Pool.forward(X, W1, b1, conv_param, pool_param)
        # Linear_ReLU.forward 함수를 이용해 Linear -> ReLU의 결과 도출
        LR_out, LR_cache = Linear_ReLU.forward(CRP_out, W2, b2)
        # Linear.forward 함수를 이용해 Linear의 결과 도출 및 scores에 저장
        scores, L_cache = Linear.forward(LR_out, W3, b3)

        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################

        if y is None:
            return scores

        loss, grads = 0.0, {}
        ######################################################################
        # TODO: Implement the backward pass for three-layer convolutional    #
        # net, storing the loss and gradients in the loss and grads.         #
        # Compute the data loss using softmax, and make sure that grads[k]   #
        # holds the gradients for self.params[k]. Don't forget to add        #
        # L2 regularization!                                                 #
        # NOTE: To ensure your implementation matches ours and you pass the  #
        # automated tests, make sure that your L2 regularization includes    #
        # a factor of 0.5 to simplify the expression for the gradient.       #
        ######################################################################
        # Replace "pass" with your code (do not modify this line)
        
        # softmax_loss function을 이용해 loss와 gradient인 dout을 구한다.
        loss, dout = softmax_loss(scores, y)
        # 또한 W들은 regularization term을 더해서 최종적인 loss를 구한다.
        for i in range(1,4):
          loss += (self.params['W' + str(i)] ** 2).sum() * self.reg * 0.5
        
        # 앞서 Forward 시에 이용한 함수들의 Backward 기능을 이용해서 gradient를 구한다.
        # 여기서 W들의 gradient에는 regularization의 gradient도 더해준다.
        dL, grads['W3'], grads['b3'] = Linear.backward(dout, L_cache)
        grads['W3'] += W3 * self.reg

        dLR, grads['W2'], grads['b2'] = Linear_ReLU.backward(dL, LR_cache)
        grads['W2'] += W2 * self.reg

        dCRP, grads['W1'], grads['b1'] = Conv_ReLU_Pool.backward(dLR, CRP_cache)
        grads['W1'] += W1 * self.reg

        # 해당 모델을 1 epoch 돌릴 때 training accuracy와 validation accuracy가 매우 낮은데,
        # batch_size = 50, learning_rate = 1e-3으로 주는 등 hyperparameter를 약간 수정해주면
        # 40%가 충분히 넘는 것을 확인할 수 있다. 

        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################

        return loss, grads


class DeepConvNet(object):
    """
    A convolutional neural network with an arbitrary number of convolutional
    layers in VGG-Net style. All convolution layers will use kernel size 3 and
    padding 1 to preserve the feature map size, and all pooling layers will be
    max pooling layers with 2x2 receptive fields and a stride of 2 to halve the
    size of the feature map.

    The network will have the following architecture:

    {conv - [batchnorm?] - relu - [pool?]} x (L - 1) - linear

    Each {...} structure is a "macro layer" consisting of a convolution layer,
    an optional batch normalization layer, a ReLU nonlinearity, and an optional
    pooling layer. After L-1 such macro layers, a single fully-connected layer
    is used to predict the class scores.

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """
    def __init__(self,
                 input_dims=(3, 32, 32),
                 num_filters=[8, 8, 8, 8, 8],
                 max_pools=[0, 1, 2, 3, 4],
                 batchnorm=False,
                 num_classes=10,
                 weight_scale=1e-3,
                 reg=0.0,
                 dtype=torch.float,
                 device='cpu'):
        """
        Initialize a new network.

        Inputs:
        - input_dims: Tuple (C, H, W) giving size of input data
        - num_filters: List of length (L - 1) giving the number of
          convolutional filters to use in each macro layer.
        - max_pools: List of integers giving the indices of the macro
          layers that should have max pooling (zero-indexed).
        - batchnorm: Whether to include batch normalization in each macro layer
        - num_classes: Number of scores to produce from the final linear layer.
        - weight_scale: Scalar giving standard deviation for random
          initialization of weights, or the string 'kaiming' to use Kaiming
          initialization instead
        - reg: Scalar giving L2 regularization strength. L2 regularization
          should only be applied to convolutional and fully-connected weight
          matrices; it should not be applied to biases or to batchnorm scale
          and shifts.
        - dtype: A torch data type object; all computations will be performed
          using this dtype. float is faster but less accurate, so you should
          use double for numeric gradient checking.
        - device: device to use for computation. 'cpu' or 'cuda'
        """
        self.params = {}
        self.num_layers = len(num_filters)+1
        self.max_pools = max_pools
        self.batchnorm = batchnorm
        self.reg = reg
        self.dtype = dtype

        if device == 'cuda':
            device = 'cuda:0'

        ######################################################################
        # TODO: Initialize the parameters for the DeepConvNet. All weights,  #
        # biases, and batchnorm scale and shift parameters should be stored  #
        # in the dictionary self.params, where the keys should be in the     #
        # form of 'W#', 'b#', 'gamma#', and 'beta#' with 1-based indexing.   #
        # Weights for Conv and Linear layers should be initialized from the  #
        # Gaussian distribution with the mean of 0.0 and the standard        #
        # deviation of weight_scale; however, if weight_scale == 'kaiming',  #
        # then you should call kaiming_initializer instead. Biases should be #
        # initialized to zeros. Batchnorm scale (gamma) and shift (beta)     #
        # parameters should be initialized to ones and zeros, respectively.  #
        ######################################################################
        # Replace "pass" with your code (do not modify this line)
        
        # filter size와 stride, pad, pooling parameter를 각각 주어진 대로 지정한다.
        filter_size = HH = WW = 3
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}
        pool_param = {'pool_height' : 2, 'pool_width' : 2, 'stride' : 2}
        prev_filters, Hout, Wout = input_dims
        
        # 각 filter마다 Hout과 Wout을 지정해주고 batch normalization을 사용하면 gamma와 beta를 정의
        # Max_pooling을 사용하면 Hout과 Wout을 추가로 계산, Kaiming initialization을 사용하면 W들을 해당 방법으로 초기화해 준다.
        # Kaiming Initialization을 사용하지 않으면 이전과 같이 Gaussian Distribution으로 W들을 초기화해 준다.
        # 이후 b들을 0행렬로 초기화해 준다.

        for i,num_filter in enumerate(num_filters):
          Hout = 1 + (Hout + 2 * conv_param['pad'] - HH) // conv_param['stride']
          Wout = 1 + (Wout + 2 * conv_param['pad'] - WW) // conv_param['stride']
          if self.batchnorm:
            # Gamma와 Beta를 초기화할 때는 각각 1행렬과 0행렬로 초기화해 준다.
            self.params['gamma' + str(i)] = torch.zeros((num_filter), dtype = dtype, device = device) + 1
            self.params['beta' + str(i)] = torch.zeros((num_filter), dtype = dtype, device = device)
          if i in max_pools:
              Hout = 1 + (Hout - pool_param['pool_height']) // pool_param['stride']
              Wout = 1 + (Wout - pool_param['pool_width']) // pool_param['stride']
          if weight_scale == 'kaiming':
            self.params['W' + str(i)] = kaiming_initializer(num_filter, prev_filters, K = filter_size, relu = True, dtype = dtype, device = device)
          else:
            self.params['W' + str(i)] = torch.normal(0.0, weight_scale, (num_filter, prev_filters, HH, WW), dtype = dtype, device = device)
          self.params['b' + str(i)] = torch.zeros(num_filter, dtype = dtype, device = device)
          
          # num_filter를 prev_filters에 넣어서 해당 값을 다음 loop에서 쓸 수 있게 한다.
          prev_filters = num_filter
        
        # 마지막 layer이 하나 남아있으므로 i에 1을 더해서 해당 값을 서수값으로 가지는 W와 b를 Kiming 혹은 기존의 방법으로 초기화해 준다.
        i += 1
        if weight_scale == 'kaiming':
          self.params['W' + str(i)] = kaiming_initializer(num_filter * Hout * Wout, num_classes, dtype = dtype, device = device)
        else:
          self.params['W' + str(i)] = torch.normal(0.0, weight_scale, (num_filter * Hout * Wout, num_classes), dtype = dtype, device = device)
        self.params['b' + str(i)] = torch.zeros(num_classes, dtype = dtype, device = device)

        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################

        # With batch normalization we need to keep track of running
        # means and variances, so we need to pass a special bn_param
        # object to each batch normalization layer. You should pass
        # self.bn_params[0] to the forward pass of the first batch
        # normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.batchnorm:
            self.bn_params = [{'mode': 'train'}
                              for _ in range(len(num_filters))]

        # Check that we got the right number of parameters
        if not self.batchnorm:
            params_per_macro_layer = 2  # weight and bias
        else:
            params_per_macro_layer = 4  # weight, bias, scale, shift
        num_params = params_per_macro_layer * len(num_filters) + 2
        msg = 'self.params has the wrong number of ' \
              'elements. Got %d; expected %d'
        msg = msg % (len(self.params), num_params)
        assert len(self.params) == num_params, msg

        # Check that all parameters have the correct device and dtype:
        for k, param in self.params.items():
            msg = 'param "%s" has device %r; should be %r' \
                  % (k, param.device, device)
            assert param.device == torch.device(device), msg
            msg = 'param "%s" has dtype %r; should be %r' \
                  % (k, param.dtype, dtype)
            assert param.dtype == dtype, msg

    def save(self, path):
        checkpoint = {
          'reg': self.reg,
          'dtype': self.dtype,
          'params': self.params,
          'num_layers': self.num_layers,
          'max_pools': self.max_pools,
          'batchnorm': self.batchnorm,
          'bn_params': self.bn_params,
        }
        torch.save(checkpoint, path)
        print("Saved in {}".format(path))

    def load(self, path, dtype, device):
        checkpoint = torch.load(path, map_location='cpu')
        self.params = checkpoint['params']
        self.dtype = dtype
        self.reg = checkpoint['reg']
        self.num_layers = checkpoint['num_layers']
        self.max_pools = checkpoint['max_pools']
        self.batchnorm = checkpoint['batchnorm']
        self.bn_params = checkpoint['bn_params']

        for p in self.params:
            self.params[p] = self.params[p].type(dtype).to(device)

        for i in range(len(self.bn_params)):
            for p in ["running_mean", "running_var"]:
                self.bn_params[i][p] = \
                    self.bn_params[i][p].type(dtype).to(device)

        print("load checkpoint file: {}".format(path))

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the deep convolutional
        network.
        Input / output: Same API as ThreeLayerConvNet.
        """
        X = X.to(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params since they
        # behave differently during training and testing.
        if self.batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
        scores = None

        # pass conv_param to the forward pass for the
        # convolutional layer
        # Padding and stride chosen to preserve the input
        # spatial size
        filter_size = 3
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ######################################################################
        # TODO: Implement the forward pass for DeepConvNet, computing the    #
        # class scores for X and storing them in the scores variable.        #
        # Use sandwich layers if Linear or Conv layers followed by ReLU      #
        # and/or Pool layers for efficient implementation.                   #
        ######################################################################
        # Replace "pass" with your code (do not modify this line)
        
        cache = {}
        out = X
        # layer을 한 개씩 돌면서 max_pools와 batchnorm이 있으면 Conv_BatchNorm_ReLU_Pool.forward 함수를 쓴다.
        # 여기서 모든 layer을 이용하므로 W, b, gamma, beta를 다 이용한다.
        for i in range(self.num_layers - 1):
          if i in self.max_pools:
            if self.batchnorm:
              out, cache[str(i)] = Conv_BatchNorm_ReLU_Pool.forward(out, self.params['W'+str(i)], self.params['b' + str(i)], self.params['gamma' + str(i)],
                                    self.params['beta' + str(i)], conv_param, self.bn_params[i], pool_param)
            else:
              # 여기서 batchnorm을 이용하지 않으면 Max_pool까지만 있는 Conv_ReLU_Pool.forward 함수를 이용한다.
              out, cache[str(i)] = Conv_ReLU_Pool.forward(out, self.params['W' + str(i)], self.params['b' + str(i)], conv_param, pool_param)
          # 만약 Max_Pool을 이용하지 않고
          else:
            # Batchnorm만 이용한다면 Conv_BatchNorm_ReLU.forward 함수를 쓴다. 
            if self.batchnorm:
              out, cache[str(i)] = Conv_BatchNorm_ReLU.forward(out, self.params['W'+str(i)], self.params['b' + str(i)], self.params['gamma' + str(i)],
                                    self.params['beta' + str(i)], conv_param, self.bn_params[i])
            # MaxPool과 Batchnorm을 모두 이용하지 않는다면 Convolution과 ReLU만 있는 Conv_ReLU.forward 함수를 쓴다.                        
            else:
              out, cache[str(i)] = Conv_ReLU.forward(out, self.params['W' + str(i)], self.params['b' + str(i)], conv_param)
        # 마지막에 Linear layer이 하나 남아 있으므로 Linear.forward를 통해 최종 scores 값을 구한다.
        i += 1
        out, cache[str(i)] = Linear.forward(out, self.params['W' + str(i)], self.params['b' + str(i)])
        scores = out

        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ######################################################################
        # TODO: Implement the backward pass for the DeepConvNet, storing the #
        # loss and gradients in the loss and grads variables.                #
        # Compute the data loss using softmax, and make sure that grads[k]   #
        # holds the gradients for self.params[k]. Don't forget to add        #
        # L2 regularization!                                                 #
        # NOTE: To ensure your implementation matches ours and you pass the  #
        # automated tests, make sure that your L2 regularization includes    #
        # a factor of 0.5 to simplify the expression for the gradient.       #
        ######################################################################
        # Replace "pass" with your code (do not modify this line)
        
        # 먼저 softmax_loss function을 이용해 loss 값을 구한다. 그 후 for문을 통해 W를 돌면서 regularization term만큼 loss를 더해준다.
        loss, up_grad = softmax_loss(scores, y)
        for i in range(self.num_layers):
          loss += (self.params['W' + str(i)] ** 2).sum() * self.reg

        # 가장 마지막 layer(현재 i값이 가리키는 곳)이 Linear이었기에 Linear.backward를 이용해 backpropagation의 첫 부분을 진행해 준다.
        up_grad, dw, grads['b' + str(i)] = Linear.backward(up_grad, cache[str(i)])
        # 또한 맨 마지막 W의 gradient에 regularization의 gradient term도 더해준다.
        grads['W' + str(i)] = dw + 2 * self.params['W' + str(i)] * self.reg
        # 이제 i에서 1씩 내려가면서 for loop을 돌면서 gradient를 구해나간다.
        for i in range(i - 1, -1, -1):
          if i in self.max_pools:
            # 마찬가지로 max_pools과 batchnorm이 있으면 Conv_BatchNorm_ReLU_Pool.backward를 이용한다.
            if self.batchnorm:
              up_grad, dw, grads['b' + str(i)], dgamma, grads['beta' + str(i)] = Conv_BatchNorm_ReLU_Pool.backward(up_grad, cache[str(i)])
              # gamma에는 reguralization term의 gradient를 더해준다.
              grads['gamma' + str(i)] = dgamma + 2 * self.params['gamma' + str(i)] * self.reg
            # batchnorm 없이 max_pools만 쓰면 Conv_ReLU_Pool.backward 함수를 이용한다.
            else:
              up_grad, dw, grads['b' + str(i)] = Conv_ReLU_Pool.backward(up_grad, cache[str(i)])
            # 물론 항상 W는 for loop을 돌 때마다 reguralization의 gradient를 더해준다.
            grads['W' + str(i)] = dw + 2 * self.params['W' + str(i)] * self.reg
          else:
            # max_pools는 안 쓰고 batchnorm을 쓸 경우 Conv_BatchNorm_ReLU.backward 함수를 이용한다. gamma는 후처리를 해준다.
            if self.batchnorm:
              up_grad, dw, grads['b' + str(i)], dgamma, grads['beta' + str(i)] = Conv_BatchNorm_ReLU.backward(up_grad, cache[str(i)])
              grads['gamma' + str(i)] = dgamma + 2 * self.params['gamma' + str(i)] * self.reg
            # max_pools와 batchnorm을 모두 쓰지 않을 경우 단순히 Conv_ReLU.backward 함수를 쓴다.
            else:
              up_grad, dw, grads['b' + str(i)] = Conv_ReLU.backward(up_grad, cache[str(i)])
            # 역시나 W에 reguralization의 gradient term을 더해준다.
            grads['W' + str(i)] = dw + 2 * self.params['W' + str(i)] * self.reg

        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################

        return loss, grads


def find_overfit_parameters():
    weight_scale = 2e-3   # Experiment with this!
    learning_rate = 1e-5  # Experiment with this!
    ##########################################################################
    # TODO: Change weight_scale and learning_rate so your model achieves     #
    # 100% training accuracy within 30 epochs.                               #
    ##########################################################################
    # Replace "pass" with your code (do not modify this line)
    
    weight_scale = 1e-1
    learning_rate = 1e-3

    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return weight_scale, learning_rate


def kaiming_initializer(Din, Dout, K=None, relu=True, device='cpu',
                        dtype=torch.float32):
    """
    Implement Kaiming initialization for linear and convolution layers.

    Inputs:
    - Din, Dout: Integers giving the number of input and output dimensions
      for this layer
    - K: If K is None, then initialize weights for a linear layer with
      Din input dimensions and Dout output dimensions. Otherwise if K is
      a nonnegative integer then initialize the weights for a convolution
      layer with Din input channels, Dout output channels, and a kernel size
      of KxK.
    - relu: If ReLU=True, then initialize weights with a gain of 2 to
      account for a ReLU nonlinearity (Kaiming initialization); otherwise
      initialize weights with a gain of 1 (Xavier initialization).
    - device, dtype: The device and datatype for the output tensor.

    Returns:
    - weight: A torch Tensor giving initialized weights for this layer.
      For a linear layer it should have shape (Din, Dout); for a
      convolution layer it should have shape (Dout, Din, K, K).
    """
    gain = 2. if relu else 1.
    weight = None
    if K is None:
        ######################################################################
        # TODO: Implement the Kaiming initialization for linear layer.       #
        # The weight_scale is sqrt(gain / fan_in), where gain is 2 if ReLU   #
        # is followed by the layer, or 1 if not, and fan_in = Din.           #
        # The output should be a tensor in the designated size, dtype,       #
        # and device.                                                        #
        ######################################################################
        # Replace "pass" with your code (do not modify this line)
        
        # 분산을 동일하게 만들기 위한 Initializer로서, Linear layer의 경우 weight_scale이 gain/Din이다.
        weight_scale = gain / Din
        weight = torch.normal(0.0, weight_scale, (Din, Dout), dtype = dtype, device = device)

        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
    else:
        ######################################################################
        # TODO: Implement Kaiming initialization for convolutional layer.    #
        # The weight_scale is sqrt(gain / fan_in), where gain is 2 if ReLU   #
        # is followed by the layer, or 1 if not, and fan_in = Din * K * K.   #
        # The output should be a tensor in the designated size, dtype,       #
        # and device.                                                        #
        ######################################################################
        # Replace "pass" with your code (do not modify this line)
        
        # 마찬가지로, Convolutional layer의 Kaiming initialization의 분모 값은 Din * K * K이다.
        weight_scale = gain / (Din * K * K)
        weight = torch.normal(0.0, weight_scale, (Din, Dout, K, K), dtype = dtype, device = device)

        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
    return weight


def create_convolutional_solver_instance(data_dict, dtype, device):
    model = None
    solver = None
    ##########################################################################
    # TODO: Train the best DeepConvNet on CIFAR-10 within 60 seconds.        #
    # Hint: You can use any optimizer you implemented in                     #
    # fully_connected_networks.py, which we imported for you.                #
    ##########################################################################
    # Replace "pass" with your code (do not modify this line)
    
    # input_dims는 X에서, num_classes는 y의 class의 개수에서 추출. initialization은 kaiming으로 설정
    input_dims = data_dict['X_train'].shape[1:]
    num_classes = len(data_dict['y_train'].unique())
    weight_scale = 'kaiming'
    
    # 먼저 model 정의. 문제의 힌트에서 알 수 있듯이 batchnorm을 사용하기.
    model = DeepConvNet(input_dims=input_dims, num_classes=10,
                        num_filters=[16,32,64],
                        max_pools=[0,1,2],
                        weight_scale=weight_scale,
                        reg=1e-5, batchnorm=True,
                        dtype=dtype,
                        device=device
                        )

    # solver를 만들 때 다양한 parameter를 바꾸면서 돌려보기. batch_size는 128로 고정
    solver = Solver(model, data_dict,
                    num_epochs=200, batch_size=128,
                    update_rule=adam,
                    optim_config={
                      'learning_rate': 3e-3
                    }, lr_decay = 0.999,
                    print_every=1000, device=device)

    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return solver


class BatchNorm(object):

    @staticmethod
    def forward(x, gamma, beta, bn_param):
        """
        Forward pass for batch normalization.

        During training the sample mean and (uncorrected) sample variance
        are computed from minibatch statistics and used to normalize the
        incoming data. During training we also keep an exponentially decaying
        running mean of the mean and variance of each feature, and these
        averages are used to normalize data at test-time.

        At each time step, we update the running averages for mean and
        variance using an exponential decay based on the momentum parameter:

        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var

        Note that the batch normalization paper suggests a different
        test-time behavior: they compute sample mean and variance for
        each feature using a large number of training images rather than
        using a running average. For this implementation we have chosen to use
        running averages instead since they do not require an additional
        estimation step; the PyTorch implementation of batch normalization
        also uses running averages.

        Input:
        - x: Data of shape (N, D)
        - gamma: Scale parameter of shape (D,)
        - beta: Shift parameter of shape (D,)
        - bn_param: Dictionary with the following keys:
          - mode: 'train' or 'test'; required
          - eps: Constant for numeric stability
          - momentum: Constant for running mean / variance.
          - running_mean: Array of shape (D,) giving running mean
            of features
          - running_var Array of shape (D,) giving running variance
            of features

        Returns a tuple of:
        - out: of shape (N, D)
        - cache: A tuple of values needed in the backward pass
        """
        mode = bn_param['mode']
        eps = bn_param.get('eps', 1e-5)
        momentum = bn_param.get('momentum', 0.9)

        N, D = x.shape
        running_mean = \
            bn_param.get('running_mean',
                         torch.zeros(D, dtype=x.dtype, device=x.device))
        running_var = \
            bn_param.get('running_var',
                         torch.ones(D, dtype=x.dtype, device=x.device))

        out, cache = None, None
        if mode == 'train':
            ##################################################################
            # TODO: Implement the training-time forward pass for batchnorm.  #
            # Use minibatch statistics to compute the mean and variance.     #
            # Use the mean and variance to normalize the incoming data, and  #
            # then scale and shift the normalized data using gamma and beta. #
            #                                                                #
            # You should store the output in the variable out.               #
            # Any intermediates that you need for the backward pass should   #
            # be stored in the cache variable.                               #
            #                                                                #
            # You should also use your computed sample mean and variance     #
            # together with the momentum variable to update the running mean #
            # and running variance, storing your result in the running_mean  #
            # and running_var variables.                                     #
            #                                                                #
            # Note that though you should be keeping track of the running    #
            # variance, you should normalize the data based on the standard  #
            # deviation (square root of variance) instead!                   #
            # Referencing the original paper                                 #
            # (https://arxiv.org/abs/1502.03167) might prove to be helpful.  #
            ##################################################################
            # Replace "pass" with your code (do not modify this line)
            
            # batch normalization을 열 기준이므로 각 열의 mean값을 mean 벡터에 저장한다.
            mean = 1 / N * x.sum(axis = 0)

            # 또한 momentum 값에 따라 mean의 일정 비율을 running_mean의 일정 비율과 더해 업데이트 한다.
            running_mean = momentum * running_mean + (1 - momentum) * mean

            # 편차는 x_mean에 저장한다.
            x_mean = x - mean

            # variance 식에 맞게 편차 제곱의 평균을 var에 저장한다. running_var도 정의한다.
            var = 1 / N * (x_mean ** 2).sum(axis = 0)
            running_var = momentum * running_var + (1 - momentum) * var

            # epsilon 값을 더한 후 루트를 씌운 후 분모로 보낸다.
            std = (var + eps).sqrt()
            istd = 1 / std

            # 편차 값을 istd와 곱하면 normalization이 완성된다.
            x_hat = x_mean * istd

            # 해당 normalized 값에 scale과 shift를 거쳐서 out을 도출한다.
            out = gamma * x_hat + beta

            cache = (x_hat, gamma, x_mean, istd, std, var, eps)

            ##################################################################
            #                        END OF YOUR CODE                        #
            ##################################################################
        elif mode == 'test':
            ##################################################################
            # TODO: Implement the test-time forward pass for batchnorm.      #
            # Use the running mean and variance to normalize the incoming    #
            # data, and then scale and shift the normalized data using gamma #
            # and beta. Store the result in the out variable.                #
            ##################################################################
            # Replace "pass" with your code (do not modify this line)
            
            # test mode의 경우 running_mean과 running_var을 통해 normalization을 진행한다.
            normalized = (x - running_mean) / (running_var + eps) ** 0.5
            # 또한 주어진 gamma와 beta에 맞게 scaling과 shift를 해주어 out을 도출한다.
            out = normalized * gamma + beta

            ##################################################################
            #                        END OF YOUR CODE                        #
            ##################################################################
        else:
            raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

        # Store the updated running means back into bn_param
        bn_param['running_mean'] = running_mean.detach()
        bn_param['running_var'] = running_var.detach()

        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for batch normalization.

        For this implementation, you should write out a
        computation graph for batch normalization on paper and
        propagate gradients backward through intermediate nodes.

        Inputs:
        - dout: Upstream derivatives, of shape (N, D)
        - cache: Variable of intermediates from batchnorm_forward.

        Returns a tuple of:
        - dx: Gradient with respect to inputs x, of shape (N, D)
        - dgamma: Gradient with respect to scale parameter gamma,
          of shape (D,)
        - dbeta: Gradient with respect to shift parameter beta,
          of shape (D,)
        """
        dx, dgamma, dbeta = None, None, None
        ######################################################################
        # TODO: Implement the backward pass for batch normalization.         #
        # Store the results in the dx, dgamma, and dbeta variables.          #
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)  #
        # might prove to be helpful.                                         #
        # Don't forget to implement train and test mode separately.          #
        ######################################################################
        # Replace "pass" with your code (do not modify this line)
        
        
        x_hat, gamma, x_mean, istd, std, var, eps = cache
        m = dout.shape[0]

        # beta에 대한 gradient는 dout을 행 기준으로 합한 값들이다.
        dbeta = dout.sum(axis = 0)

        # gamma에 대한 gradient는 dout과 normalized 값을 곱해서 행별로 합한 값들이다.
        dgamma = (dout * x_hat).sum(axis = 0)

        # normalized 값들의 gradient는 dout에 gamma를 곱한 값들이다.
        dx_hat = dout * gamma

        # var에 대한 gradient는 밑 식과 같이 구한다.
        dvar = (dx_hat * x_mean * (-0.5) * (var + eps) ** (-3 / 2)).sum(axis = 0)
        
        # mean에 대한 gradient는 밑 식과 같이 구한다.
        dmean = dx_hat.sum(axis = 0) * (- istd) + dvar * -2 * x_mean.sum(axis = 0) / m
        
        # x에 대한 gradient는 밑의 식과 같이 구한다.
        dx = dx_hat * istd + dvar * 2 * x_mean / m +  dmean / m

        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################

        return dx, dgamma, dbeta

    @staticmethod
    def backward_alt(dout, cache):
        """
        Alternative backward pass for batch normalization.
        For this implementation you should work out the derivatives
        for the batch normalization backward pass on paper and simplify
        as much as possible. You should be able to derive a simple expression
        for the backward pass. See the jupyter notebook for more hints.

        Note: This implementation should expect to receive the same
        cache variable as batchnorm_backward, but might not use all of
        the values in the cache.

        Inputs / outputs: Same as batchnorm_backward
        """
        dx, dgamma, dbeta = None, None, None
        ######################################################################
        # TODO: Implement the backward pass for batch normalization.         #
        # Store the results in the dx, dgamma, and dbeta variables.          #
        #                                                                    #
        # Note: after computing the gradient with respect to the centered    #
        # inputs, gradients with respect to the inputs (dx) can be written   #
        # in a single statement; our implementation fits on a single         #
        # 80-character line. But, it is okay to write it in multiple lines.  #
        ######################################################################
        # Replace "pass" with your code (do not modify this line)
        
        x_hat, gamma, x_mean, istd, std, var, eps = cache
        m = dout.shape[0]
        # y = gamma * x_hat + beta이기 때문에 dbeta는 dout의 sum으로 나온다.
        dbeta = dout.sum(axis = 0)
        # 또한 dgamma는 dout * x_hat의 sum으로 나온다.
        dgamma = (x_hat * dout).sum(axis = 0)
        # 마지막으로 dx는 밑의 식과 같이 나온다.
        dx = gamma * istd * (m * dout - dgamma * x_hat - dbeta) / m

        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################

        return dx, dgamma, dbeta


class SpatialBatchNorm(object):

    @staticmethod
    def forward(x, gamma, beta, bn_param):
        """
        Computes the forward pass for spatial batch normalization.

        Inputs:
        - x: Input data of shape (N, C, H, W)
        - gamma: Scale parameter, of shape (C,)
        - beta: Shift parameter, of shape (C,)
        - bn_param: Dictionary with the following keys:
          - mode: 'train' or 'test'; required
          - eps: Constant for numeric stability
          - momentum: Constant for running mean / variance. momentum=0
            means that old information is discarded completely at every
            time step, while momentum=1 means that new information is never
            incorporated. The default of momentum=0.9 should work well
            in most situations.
          - running_mean: Array of shape (C,) giving running mean of
            features
          - running_var Array of shape (C,) giving running variance
            of features

        Returns a tuple of:
        - out: Output data, of shape (N, C, H, W)
        - cache: Values needed for the backward pass
        """
        out, cache = None, None

        ######################################################################
        # TODO: Implement the forward pass for spatial batch normalization.  #
        # You should implement this by calling the 1D batch normalization    #
        # you implemented above with permuting and/or reshaping input/output #
        # tensors. Your implementation should be very short;                 #
        # less than five lines are expected.                                 #
        ######################################################################
        # Replace "pass" with your code (do not modify this line)
        
        # permute 기능을 이용해 x를 재구성한 후 reshape를 통해 한줄로 늘린다.
        N,C,H,W = x.shape
        m = x.permute(1,0,2,3).reshape(C, -1).T
        # 그 후 기존의 BatchNorm.forward를 이용하고, 다시 반대로 reshape와 permute를 해주면 된다.
        out, cache = BatchNorm.forward(m, gamma, beta, bn_param)
        out = out.T.reshape(C, N, H, W).permute(1,0,2,3)

        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################

        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Computes the backward pass for spatial batch normalization.
        Inputs:
        - dout: Upstream derivatives, of shape (N, C, H, W)
        - cache: Values from the forward pass
        Returns a tuple of:
        - dx: Gradient with respect to inputs, of shape (N, C, H, W)
        - dgamma: Gradient with respect to scale parameter, of shape (C,)
        - dbeta: Gradient with respect to shift parameter, of shape (C,)
        """
        dx, dgamma, dbeta = None, None, None

        ######################################################################
        # TODO: Implement the backward pass for spatial batch normalization. #
        # You should implement this by calling the 1D batch normalization    #
        # you implemented above with permuting and/or reshaping input/output #
        # tensors. Your implementation should be very short;                 #
        # less than five lines are expected.                                 #
        ######################################################################
        # Replace "pass" with your code (do not modify this line)
        
        # backward도 마찬가지로 permute와 reshape를 한 후 BatchNorm의 backward를 이용한 뒤
        # 다시 reshape와 permute를 이용해 원래대로 바꾸면 된다.
        
        N, C, H, W = dout.shape
        m = dout.permute(1,0,2,3).reshape(C,-1).T
        dx, dgamma, dbeta = BatchNorm.backward_alt(m, cache)
        dx = dx.T.reshape(C, N, H, W).permute(1, 0, 2, 3)

        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################

        return dx, dgamma, dbeta


##################################################################
#            Fast Implementations and Sandwich Layers            #
##################################################################


class Linear(object):

    @staticmethod
    def forward(x, w, b):
        layer = torch.nn.Linear(*w.shape)
        layer.weight = torch.nn.Parameter(w.T)
        layer.bias = torch.nn.Parameter(b)
        tx = x.detach()
        tx.requires_grad = True
        out = layer(tx.flatten(start_dim=1))
        cache = (x, w, b, tx, out, layer)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        try:
            x, w, b, tx, out, layer = cache
            out.backward(dout)
            dx = tx.grad.detach()
            dw = layer.weight.grad.detach().T
            db = layer.bias.grad.detach()
            layer.weight.grad = layer.bias.grad = None
        except RuntimeError:
            dx = torch.zeros_like(tx)
            dw = torch.zeros_like(layer.weight).T
            db = torch.zeros_like(layer.bias)
        return dx, dw, db


class ReLU(object):

    @staticmethod
    def forward(x):
        layer = torch.nn.ReLU()
        tx = x.detach()
        tx.requires_grad = True
        out = layer(tx)
        cache = (x, tx, out, layer)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        try:
            x, tx, out, layer = cache
            out.backward(dout)
            dx = tx.grad.detach()
        except RuntimeError:
            dx = torch.zeros_like(tx)
        return dx


class Linear_ReLU(object):

    @staticmethod
    def forward(x, w, b):
        """
        Convenience layer that performs a linear transform followed by a ReLU.

        Inputs:
        - x: Input to the linear layer
        - w, b: Weights for the linear layer
        Returns a tuple of:
        - out: Output of the ReLU
        - cache: Object to give to the backward pass
        """
        a, fc_cache = Linear.forward(x, w, b)
        out, relu_cache = ReLU.forward(a)
        cache = (fc_cache, relu_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for the linear-relu convenience layer
        """
        fc_cache, relu_cache = cache
        da = ReLU.backward(dout, relu_cache)
        dx, dw, db = Linear.backward(da, fc_cache)
        return dx, dw, db


class FastConv(object):

    @staticmethod
    def forward(x, w, b, conv_param):
        N, C, H, W = x.shape
        F, _, HH, WW = w.shape
        stride, pad = conv_param['stride'], conv_param['pad']
        layer = torch.nn.Conv2d(C, F, (HH, WW), stride=stride, padding=pad)
        layer.weight = torch.nn.Parameter(w)
        layer.bias = torch.nn.Parameter(b)
        tx = x.detach()
        tx.requires_grad = True
        out = layer(tx)
        cache = (x, w, b, conv_param, tx, out, layer)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        try:
            x, _, _, _, tx, out, layer = cache
            out.backward(dout)
            dx = tx.grad.detach()
            dw = layer.weight.grad.detach()
            db = layer.bias.grad.detach()
            layer.weight.grad = layer.bias.grad = None
        except RuntimeError:
            dx = torch.zeros_like(tx)
            dw = torch.zeros_like(layer.weight)
            db = torch.zeros_like(layer.bias)
        return dx, dw, db


class FastMaxPool(object):

    @staticmethod
    def forward(x, pool_param):
        N, C, H, W = x.shape
        pool_height = pool_param['pool_height']
        pool_width = pool_param['pool_width']
        stride = pool_param['stride']
        layer = torch.nn.MaxPool2d(kernel_size=(pool_height, pool_width),
                                   stride=stride)
        tx = x.detach()
        tx.requires_grad = True
        out = layer(tx)
        cache = (x, pool_param, tx, out, layer)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        try:
            x, _, tx, out, layer = cache
            out.backward(dout)
            dx = tx.grad.detach()
        except RuntimeError:
            dx = torch.zeros_like(tx)
        return dx


class Conv_ReLU(object):

    @staticmethod
    def forward(x, w, b, conv_param):
        """
        A convenience layer that performs a convolution
        followed by a ReLU.
        Inputs:
        - x: Input to the convolutional layer
        - w, b, conv_param: Weights and parameters for the
          convolutional layer
        Returns a tuple of:
        - out: Output from the ReLU
        - cache: Object to give to the backward pass
        """
        a, conv_cache = FastConv.forward(x, w, b, conv_param)
        out, relu_cache = ReLU.forward(a)
        cache = (conv_cache, relu_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for the conv-relu convenience layer.
        """
        conv_cache, relu_cache = cache
        da = ReLU.backward(dout, relu_cache)
        dx, dw, db = FastConv.backward(da, conv_cache)
        return dx, dw, db


class Conv_ReLU_Pool(object):

    @staticmethod
    def forward(x, w, b, conv_param, pool_param):
        """
        A convenience layer that performs a convolution,
        a ReLU, and a pool.
        Inputs:
        - x: Input to the convolutional layer
        - w, b, conv_param: Weights and parameters for
          the convolutional layer
        - pool_param: Parameters for the pooling layer
        Returns a tuple of:
        - out: Output from the pooling layer
        - cache: Object to give to the backward pass
        """
        a, conv_cache = FastConv.forward(x, w, b, conv_param)
        s, relu_cache = ReLU.forward(a)
        out, pool_cache = FastMaxPool.forward(s, pool_param)
        cache = (conv_cache, relu_cache, pool_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for the conv-relu-pool convenience layer.
        """
        conv_cache, relu_cache, pool_cache = cache
        ds = FastMaxPool.backward(dout, pool_cache)
        da = ReLU.backward(ds, relu_cache)
        dx, dw, db = FastConv.backward(da, conv_cache)
        return dx, dw, db


class FastBatchNorm(object):
    func = torch.nn.BatchNorm1d

    @classmethod
    def forward(cls, x, gamma, beta, bn_param):
        mode = bn_param['mode']
        eps = bn_param.get('eps', 1e-5)
        momentum = bn_param.get('momentum', 0.9)
        D = x.shape[1]
        running_mean = \
            bn_param.get('running_mean',
                         torch.zeros(D, dtype=x.dtype, device=x.device))
        running_var = \
            bn_param.get('running_var',
                         torch.ones(D, dtype=x.dtype, device=x.device))

        layer = cls.func(D, eps=eps, momentum=momentum,
                         device=x.device, dtype=x.dtype)
        layer.weight = torch.nn.Parameter(gamma)
        layer.bias = torch.nn.Parameter(beta)
        layer.running_mean = running_mean
        layer.running_var = running_var
        if mode == 'train':
            layer.train()
        elif mode == 'test':
            layer.eval()
        else:
            raise ValueError('Invalid forward batchnorm mode "%s"' % mode)
        tx = x.detach()
        tx.requires_grad = True
        out = layer(tx)
        cache = (mode, x, tx, out, layer)
        # Store the updated running means back into bn_param
        bn_param['running_mean'] = layer.running_mean.detach()
        bn_param['running_var'] = layer.running_var.detach()
        return out, cache

    @classmethod
    def backward(cls, dout, cache):
        mode, x, tx, out, layer = cache
        try:
            if mode == 'train':
                layer.train()
            elif mode == 'test':
                layer.eval()
            else:
                raise ValueError('Invalid forward batchnorm mode "%s"' % mode)
            out.backward(dout)
            dx = tx.grad.detach()
            dgamma = layer.weight.grad.detach()
            dbeta = layer.bias.grad.detach()
            layer.weight.grad = layer.bias.grad = None
        except RuntimeError:
            dx = torch.zeros_like(tx)
            dgamma = torch.zeros_like(layer.weight)
            dbeta = torch.zeros_like(layer.bias)
        return dx, dgamma, dbeta


class FastSpatialBatchNorm(FastBatchNorm):
    func = torch.nn.BatchNorm2d


class Linear_BatchNorm_ReLU(object):

    @staticmethod
    def forward(x, w, b, gamma, beta, bn_param):
        """
        Convenience layer that performs an linear transform,
        batch normalization, and ReLU.
        Inputs:
        - x: Array of shape (N, D1); input to the linear layer
        - w, b: Arrays of shape (D1, D2) and (D2,) giving the
          weight and bias for the linear transform.
        - gamma, beta: Arrays of shape (D2,) and (D2,) giving
          scale and shift parameters for batch normalization.
        - bn_param: Dictionary of parameters for batch
          normalization.
        Returns:
        - out: Output from ReLU, of shape (N, D2)
        - cache: Object to give to the backward pass.
        """
        a, fc_cache = Linear.forward(x, w, b)
        a_bn, bn_cache = FastBatchNorm.forward(a, gamma, beta, bn_param)
        out, relu_cache = ReLU.forward(a_bn)
        cache = (fc_cache, bn_cache, relu_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for the linear-batchnorm-relu convenience layer.
        """
        fc_cache, bn_cache, relu_cache = cache
        da_bn = ReLU.backward(dout, relu_cache)
        da, dgamma, dbeta = FastBatchNorm.backward(da_bn, bn_cache)
        dx, dw, db = Linear.backward(da, fc_cache)
        return dx, dw, db, dgamma, dbeta


class Conv_BatchNorm_ReLU(object):

    @staticmethod
    def forward(x, w, b, gamma, beta, conv_param, bn_param):
        a, conv_cache = FastConv.forward(x, w, b, conv_param)
        an, bn_cache = FastSpatialBatchNorm.forward(a, gamma,
                                                    beta, bn_param)
        out, relu_cache = ReLU.forward(an)
        cache = (conv_cache, bn_cache, relu_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        conv_cache, bn_cache, relu_cache = cache
        dan = ReLU.backward(dout, relu_cache)
        da, dgamma, dbeta = FastSpatialBatchNorm.backward(dan, bn_cache)
        dx, dw, db = FastConv.backward(da, conv_cache)
        return dx, dw, db, dgamma, dbeta


class Conv_BatchNorm_ReLU_Pool(object):

    @staticmethod
    def forward(x, w, b, gamma, beta, conv_param, bn_param, pool_param):
        a, conv_cache = FastConv.forward(x, w, b, conv_param)
        an, bn_cache = FastSpatialBatchNorm.forward(a, gamma, beta, bn_param)
        s, relu_cache = ReLU.forward(an)
        out, pool_cache = FastMaxPool.forward(s, pool_param)
        cache = (conv_cache, bn_cache, relu_cache, pool_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        conv_cache, bn_cache, relu_cache, pool_cache = cache
        ds = FastMaxPool.backward(dout, pool_cache)
        dan = ReLU.backward(ds, relu_cache)
        da, dgamma, dbeta = FastSpatialBatchNorm.backward(dan, bn_cache)
        dx, dw, db = FastConv.backward(da, conv_cache)
        return dx, dw, db, dgamma, dbeta
