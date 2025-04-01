import numpy as np
import pandas as pd
import cv2
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# 데이터셋 경로
dataset_path = 'fer2013.csv'

# 데이터셋 로드 및 전처리
def load_fer2013(dataset_path):
    data = pd.read_csv(dataset_path)

    # 픽셀 문자열을 48x48 이미지 배열로 변환
    pixels = data['pixels'].tolist()
    X = [np.array(pix.split(), dtype='float32').reshape(48, 48, 1) for pix in pixels]
    X = np.array(X) / 255.0  # 정규화

    # 레이블 원-핫 인코딩
    y = to_categorical(data['emotion'], num_classes=7)

    return X, y

# 데이터 로드
X, y = load_fer2013(dataset_path)

# 학습/테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# EfficientNetB0 모델 구성
def build_efficientnet_model(input_shape):
    base_model = EfficientNetB0(weights=None, include_top=False, input_shape=input_shape)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    predictions = Dense(7, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

# 모델 초기화 및 컴파일
input_shape = (48, 48, 1)
model = build_efficientnet_model(input_shape)
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# 모델 학습
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=64)

# 모델 저장
model.save('efficientnet_fer2013.h5')
