import cv2
import numpy as np
from tensorflow.keras.models import load_model
from keras.preprocessing.image import img_to_array
from mtcnn import MTCNN

# 감정 레이블 정의
emotion_dict = {
    0: 'Anger', 
    1: 'Disgust', 
    2: 'Fear', 
    3: 'Happy', 
    4: 'Sad', 
    5: 'Surprise', 
    6: 'Neutral'
}

# 모델 로드
model = load_model('mobilenet_fer_model.h5')

# MTCNN 얼굴 검출기 초기화
detector = MTCNN()

# 웹캠 열기
video_capture = cv2.VideoCapture(0)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

frame_count = 0  # 프레임 카운트 변수 추가
last_face = None  # 마지막 감지된 얼굴을 저장하는 변수

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    frame_count += 1

    # 2프레임마다 얼굴 검출 및 감정 예측 수행
    if frame_count % 1 == 0:
        rgb_frame = frame[:, :, ::-1]
        faces = detector.detect_faces(rgb_frame)

        if faces:
            last_face = faces  # 얼굴이 검출되면 업데이트

    # 가장 최근의 얼굴 데이터를 사용해 결과 표시
    if last_face:
        for face in last_face:
            x, y, width, height = face['box']
            face_img = frame[y:y + height, x:x + width]

            # 얼굴 이미지 전처리
            face_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            face_resized = cv2.resize(face_gray, (48, 48))  
            face_resized = face_resized.astype('float32') / 255.0  
            face_resized = img_to_array(face_resized)
            face_resized = np.expand_dims(face_resized, axis=0)

            # 감정 예측
            emotion_preds = model.predict(face_resized)[0]
            predicted_emotion = np.argmax(emotion_preds)

            # 얼굴 영역에 감정 결과 표시
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 0, 255), 2)
            cv2.putText(frame, f'Emotion: {emotion_dict[predicted_emotion]} ({np.max(emotion_preds):.3f})', 
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36, 255, 12), 2)

    # 결과 프레임 출력
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 웹캠 종료
video_capture.release()
cv2.destroyAllWindows()
