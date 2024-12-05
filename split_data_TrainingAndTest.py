import jax 
import jax.numpy as jnp 
import flax.linen as nn 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import optax
import numpy as np
from sklearn import preprocessing
from sklearn.datasets import load_digits



# Load data
digits = load_digits()
Images = digits.data
target = digits.target
#visualize
plt.gray()
plt.matshow(digits.images[0])
plt.show()



#Let's rename the input and output variables
x = Images
y = target



def one_hot_encoder(x):
    """
    One-Hot Encoder
    
    Converts an array of categorical labels to one-hot encoded representation.
    
    Args:
        x (ndarray): Array of categorical labels.
        
    Returns:
        ndarray: One-hot encoded representation of the input labels.
    """
    # get the number of classes from the input array 
    nclasses = 10 
    # Initialize an array of zeros with shape (number of samples, number of classes)
    out = np.zeros((len(x), nclasses))   
    # Set the corresponding class index to 1 in each sample 
    for i, x_ in enumerate(x):
         out[i, x_] = 1 
        
    return out

# test the function 
target_oh = one_hot_encoder(target)
# test
print(f"target value: {target[40]}, corresponding one-hot vector: {target_oh[40,:]}")



@jax.jit
def cat_cross_entropy(params, x, y_true):
    """
    Categorical Cross Entropy Loss Function

    This function calculates the categorical cross entropy loss between the predicted
    probabilities and the true one-hot encoded labels.

    Parameters:
        params (jax.interpreters.xla.DeviceArray): Model parameters.
        x (jax.interpreters.xla.DeviceArray): Input data.
        y_true (jax.interpreters.xla.DeviceArray): True one-hot encoded labels.

    Returns:
        jax.interpreters.xla.DeviceArray: Mean categorical cross entropy loss.
    """
    # Forward pass to obtain predicted probabilities
    y_pred = model.apply(params, x)

    # Calculation of the loss per example
    loss_per_example = -jnp.sum(y_true * jnp.log(y_pred + 1e-8), axis=1)

    # Return the mean loss across all examples
    return jnp.mean(loss_per_example)




class MLP(nn.Module):
    """
    Multi-layer Perceptron
    
    This class represents a Multi-layer Perceptron (MLP) model, which is a type of feedforward neural network.
    MLPs consist of multiple layers of interconnected nodes (called neurons) and are commonly used for
    classification tasks.
    
    Parameters:
        nhidden_units (int): Number of hidden units in each layer.
        nlayers (int, optional): Number of layers in the MLP. Defaults to 1.
        nclasses (int, optional): Number of output classes (categories). Defaults to 10.
    
    Attributes:
        layers (list): List of Dense layers representing the hidden layers in the MLP.
        final_layer (Dense): Final Dense layer for producing the output of the MLP.
    """
    #nhidden_units: 128
    #nlayers = 1
    #nclasses = 10
    

    nhidden_units: int = 128  #隐藏层的感知器数量,使用 dataclass 风格定义，将隐藏层神经元数量设置为 256
    nlayers: int = 1          #隐藏层的层数
    nclasses: int = 10        #输出层的分类数量（不管有多少输入，都输出为10个之1）


    def setup(self):
        """
        Setup the MLP model by creating the hidden layers and the final output layer.
        """
        self.hidden_size = self.nhidden_units
        # create the weight matrices for the hidden layers
        
        self.w0 = self.param('w0', jax.nn.initializers.glorot_uniform(), (64, self.hidden_size))
        self.b0 = self.param('b0', jax.nn.initializers.zeros, (self.hidden_size,))
        self.w1 = self.param('w1', jax.nn.initializers.glorot_uniform(), (self.hidden_size, self.nclasses))
        self.b1 = self.param('b1', jax.nn.initializers.zeros, (self.nclasses,))
    
# @nn.compact: 这个装饰器表示 __call__() 方法中的层需要进行编译，它标记在前向传播时创建的权重和层。
# def __call__(self, x): 定义前向传播的方法，它接受输入 x 并计算网络的输出。    
    @nn.compact
    def __call__(self, x):
        """
        Perform a forward pass through the MLP.
        
        Args:
            x (jax.interpreters.xla.DeviceArray): Input data.
            
        Returns:
            jax.interpreters.xla.DeviceArray: Output of the MLP.
        """

#计算输入 x 与权重矩阵 w0 的点积，再加上偏置项 b0。
#然后使用 ReLU 激活函数，将所有负值置为 0，使模型具有非线性
#将经过 ReLU 激活的结果再与权重矩阵 w1 相乘，加上偏置项 b1，得到输出层的结果

        x = jax.nn.relu(jnp.dot(x, self.w0) + self.b0)
        x = jax.nn.softmax(jnp.dot(x, self.w1) + self.b1)
        
        


#return x: 返回模型的输出，通常是各类别的分数（称为“logits”）

        return x
    
# initializing the optimizer
#learning_rate = 1e-3: 定义学习率为 0.001，用于优化器
#optx = optax.adam(learning_rate=learning_rate):
#使用 Optax 库中的 Adam 优化器，Adam 是一种自适应的优化方法。
#learning_rate 指定了参数更新的步长，用于引导模型朝着损失最小化的方向调整

learning_rate = 1e-3
optx = optax.adam(learning_rate=learning_rate)






# split the data into training and test data
x_train, x_test, y_train, y_test = train_test_split(Images, target_oh, test_size=0.33, random_state=42)
# initialize the neural network 
model = MLP(256)
params = model.init(jax.random.PRNGKey(0), x_test)
opt_state = optx.init(params)


loss_fn = cat_cross_entropy
loss_grad_fn = jax.value_and_grad(cat_cross_entropy) # a function to evaluate the function and its gradient)

# training loop
n_epochs = 1000 #number of training epochs 
for e in range(n_epochs):
    loss_val, grad = loss_grad_fn(params, x_train, y_train)
    updates, opt_state = optx.update(grad, opt_state)
    params = optax.apply_updates(params, updates)
    if e % 100 == 0:
        print(f"epoch: {e}, loss function: {loss_val}")
