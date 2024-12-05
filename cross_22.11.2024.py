import jax.numpy as jnp
import jax

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
    loss_per_example = -jnp.sum(y_true * jnp.log(y_pred ), axis=1)


    # Return the mean loss across all examples
    return jnp.mean(loss_per_example)



class DummyModel:
    def apply(self, params, x):
        # 模拟输出是 softmax 后的概率，假设输出有 3 个类别
        logits = jnp.dot(x, params)
        return jax.nn.softmax(logits, axis=-1)


# 定义模型
model = DummyModel()

# 定义参数和输入
# 假设我们有3个特征，3个类别
params = jnp.array([[0.2, -0.3, 0.5],
                    [0.4, 0.1, -0.2],
                    [-0.1, 0.3, 0.7]])

# 输入数据有 5 个样本，每个样本有 3 个特征
x = jnp.array([[1.0, 0.5, -1.2],
               [0.7, -0.3, 0.8],
               [0.2, 1.0, 0.5],
               [-0.5, 0.4, 1.0],
               [0.6, 0.9, -0.7]])

# 真实的 one-hot 标签，每个样本属于 3 个类别之一
y_true = jnp.array([[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                    [1, 0, 0],
                    [0, 1, 0]])

loss = cat_cross_entropy(params, x, y_true)
print(f"Categorical Cross Entropy Loss: {loss}")
