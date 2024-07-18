import numpy as np
from keras.models import Sequential
from keras.layers import Dense
# 假设有10个输入特征和3个输出类别  
input_dim = 10
num_classes = 3
# 创建DNN模型  
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(input_dim,)))
model.add(Dense(32, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
# 编译模型，选择优化器和损失函数  
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# 假设有100个样本的训练数据和标签  
X_train = np.random.rand(100, input_dim)
y_train = np.random.randint(0, 2, size=(100, num_classes))
# 训练模型  
model.fit(X_train, y_train, epochs=10)
