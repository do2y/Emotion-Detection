import pandas as pd
import numpy as np
from tensorflow.keras.applications import MobileNet
from keras.models import Sequential
from keras.layers import GlobalAveragePooling2D, Dense
from keras.utils import to_categorical

# 데이터셋 로드
dataset_path = 'fer2013.csv'

data = pd.read_csv(dataset_path)

# 픽셀 전처리 함수
def preprocess_pixels(pixel_string):
    pixels = np.array(pixel_string.split(), dtype='float32')
    return pixels.reshape((48, 48)) / 255.0

# 훈련/테스트 데이터 분할
train_data = data[data['Usage'] == 'Training']
test_data = data[data['Usage'] == 'PrivateTest']

X_train = np.array([preprocess_pixels(pixels) for pixels in train_data['pixels']])
y_train = to_categorical(train_data['emotion'], num_classes=7)

X_test = np.array([preprocess_pixels(pixels) for pixels in test_data['pixels']])
y_test = to_categorical(test_data['emotion'], num_classes=7)

# 입력 차원 맞추기
X_train = X_train.reshape(-1, 48, 48, 1)
X_test = X_test.reshape(-1, 48, 48, 1)

# MobileNet 기반 모델 구성
base_model = MobileNet(weights=None, include_top=False, input_shape=(48, 48, 1))

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dense(7, activation='softmax')  # 7개의 감정 클래스
])

# 모델 컴파일
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 모델 학습
model.fit(X_train, y_train, batch_size=64, epochs=30, validation_data=(X_test, y_test))

# 모델 평가
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test loss: {loss:.4f}')
print(f'Test accuracy: {accuracy:.4f}')

# 모델 저장
model.save('mobilenet_fer_model.h5')
