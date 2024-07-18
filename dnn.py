import numpy as np
from keras.models import Sequential
from keras.layers import Dense
# 假设有1个输入特征和2个输出类别  
input_dim = 1
num_classes = 2
size=100
# 创建DNN模型  
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(input_dim,)))
model.add(Dense(32, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
# 编译模型，选择优化器和损失函数  
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# 假设有100个样本的训练数据和标签  
X_train_mine = np.array(range(100),dtype=np.float32).reshape(100,1)
y_train_mine = np.array([0,1,1,0]*50,dtype=np.float32).reshape(100,2)

X_train = np.random.rand(size, input_dim)
y_train = np.random.randint(0, 2, size=(size, num_classes))
# 训练模型  
model.fit(X_train_mine, y_train_mine, epochs=10)
print(X_train,y_train)
breakpoint()