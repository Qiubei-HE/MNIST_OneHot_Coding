import numpy as np
from sklearn.datasets import load_digits

digits = load_digits()  # Datensatz laden
target = digits.target  # Zielwerte abrufen

def one_hot_encoder(x):
    """One-Hot-Codiert eine Zielvariable
    
    Args:
        x: Array, das die Zielwerte enthält
    
    Returns:
        Ein Array der Form (n_samples, n_classes) mit one-hot-codierten Zielwerten.
    """
    # Anzahl der Klassen ermitteln, hier sind es 10 (Zahlen von 0 bis 9)
    nclasses = 10
    # Ein Array mit Nullen initialisieren, das die Form (Anzahl der Proben, Anzahl der Klassen) hat # 初始化一个形状为 (样本数量, 类别数量) 的全零数组
    out = np.zeros((len(x), nclasses))
    # Für jede Probe den Index der entsprechenden Klasse auf 1 setzen #对于每个样本，将对应的类索引位置设为1
    for i, x_ in enumerate(x):
        out[i, x_] = 1                   #在矩阵第i行第x_列输出结果为1  
    return out

# Funktion testen
target_oh = one_hot_encoder(target)
print(f"Zielwert: {target[40]}, entsprechender One-Hot-Vektor: {target_oh[40,:]}")


#for i in range(50):
 #   print(f"Sample {i}, target value: {target[i]}")
