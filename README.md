# 실시간 감정 인식 시스템 🎭

이 프로젝트는 FER2013 데이터셋을 기반으로 웹캠으로 사용자의 얼굴을 분석해 감정을 실시간으로 인식하고 시각화하는 시스템입니다.  
EfficientNet, MobileNet, DeepFace 등 다양한 모델을 비교하며, 일정 조건에서는 감정 경고창을 띄우는 기능도 포함되어 있습니다.

---

## ✨ 주요 기능

- 실시간 얼굴 감정 인식 (웹캠 기반)
- 다양한 모델(EfficientNet, MobileNet, DeepFace) 지원
- 감정 확률 및 비율 시각화
- 부정 감정이 지속되면 팝업 경고창 표시
- 감정 분류 모델 학습 스크립트 포함

---

## 😄 감정 클래스

- Anger, Disgust, Fear, Happy, Sad, Surprise, Neutral

---

## 🗂 파일 구성

| 파일명 | 설명 |
|--------|------|
| `realtime_emotion_mtcnn_mobilenet.py` | MTCNN + MobileNet 기반 실시간 분석 |
| `realtime_emotion_haar_efficientnet.py` | Haar + EfficientNet 기반 실시간 분석 |
| `realtime_emotion_deepface.py` | DeepFace 라이브러리 기반 감정 인식 |
| `emotion_alert_system.py` | 감정 통계 기반 경고창 팝업 기능 |
| `train_emotion_mobilenet.py` | MobileNet 학습 코드 |
| `train_emotion_efficientnet.py` | EfficientNet 학습 코드 |

---

## ▶ 실행 방법

학습한 모델(`.h5`) 파일과 함께 아래 명령어 실행:

```bash
python realtime_emotion_mtcnn_mobilenet.py
```

> `q` 키를 누르면 종료됩니다.

---

## 💾 모델 학습

[Kaggle FER2013 데이터](https://www.kaggle.com/datasets/msambare/fer2013)에서 `fer2013.csv`를 내려받아 프로젝트 루트에 두고 아래 스크립트 실행:

```bash
python train_emotion_mobilenet.py
# 또는
python train_emotion_efficientnet.py
```

---

## 📦 라이브러리 설치

```bash
pip install -r requirements.txt
```

- tensorflow  
- keras  
- opencv-python  
- numpy  
- pandas  
- scikit-learn  
- mtcnn  
- deepface  

---

