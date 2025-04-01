from deepface import DeepFace
import cv2

# 웹캠 실시간 감정 인식
video_capture = cv2.VideoCapture(0)

video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

while True:
    ret, frame = video_capture.read()

    frame = cv2.resize(frame, (320, 240)) 

    # 감정 분석
    try:
        analysis = DeepFace.analyze(frame, actions=['emotion'], detector_backend='mtcnn')

        # 리스트의 첫 번째 얼굴에 접근
        if isinstance(analysis, list):
            emotion_probabilities = analysis[0]['emotion']  
        else:
            emotion_probabilities = analysis['emotion']

        #감정 비율 출력
        y_offset = 20  # 출력 시작 y좌표
        for emotion, prob in emotion_probabilities.items():
            text = f"{emotion}: {prob:.2f}%"
            cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += 30  #다음 좌표

    except Exception as e:
        print(f"Error: {e}")
        pass

    # 화면에 결과 표시
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
