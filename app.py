import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import io

# Настройка страницы
st.set_page_config(page_title="Виртуальная примерка очков", page_icon="🕶️")

# Проверка библиотек
try:
    import cv2
    import mediapipe as mp
    LIBS_AVAILABLE = True
except ImportError as e:
    LIBS_AVAILABLE = False
    st.error(f"Ошибка импорта библиотек: {e}")
    st.info("Установите библиотеки: pip install opencv-python-headless mediapipe")

if LIBS_AVAILABLE:
    # Инициализация MediaPipe
    @st.cache_resource
    def init_mediapipe():
        try:
            mp_face_mesh = mp.solutions.face_mesh
            face_mesh = mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5
            )
            return face_mesh
        except Exception as e:
            st.error(f"Ошибка инициализации MediaPipe: {e}")
            return None

    face_mesh = init_mediapipe()

    def detect_face_landmarks(image):
        """Определяет ключевые точки лица"""
        if face_mesh is None:
            return None
            
        try:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image_rgb)
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0]
                return landmarks
        except Exception as e:
            st.error(f"Ошибка детекции: {e}")
        return None

    def get_eye_coordinates(landmarks, image_shape):
        """Получает координаты глаз"""
        h, w = image_shape[:2]
        
        # Простые индексы для глаз
        left_eye_idx = [33, 133, 160, 158, 157, 173]
        right_eye_idx = [362, 263, 387, 385, 384, 398]
        
        left_points = [(landmarks.landmark[i].x * w, landmarks.landmark[i].y * h) 
                      for i in left_eye_idx]
        right_points = [(landmarks.landmark[i].x * w, landmarks.landmark[i].y * h) 
                       for i in right_eye_idx]
        
        left_center = (int(np.mean([p[0] for p in left_points])), 
                      int(np.mean([p[1] for p in left_points])))
        right_center = (int(np.mean([p[0] for p in right_points])), 
                       int(np.mean([p[1] for p in right_points])))
        
        eye_distance = np.sqrt((right_center[0] - left_center[0])**2 + 
                              (right_center[1] - left_center[1])**2)
        
        angle = np.arctan2(right_center[1] - left_center[1], 
                          right_center[0] - left_center[0])
        
        return left_center, right_center, eye_distance, angle

    def create_simple_glasses():
        """Создает простые очки"""
        glasses = Image.new('RGBA', (200, 100), (0, 0, 0, 0))
        draw = ImageDraw.Draw(glasses)
        
        # Левая линза
        draw.ellipse([10, 30, 70, 70], outline=(0, 0, 0, 255), width=3)
        # Правая линза  
        draw.ellipse([130, 30, 190, 70], outline=(0, 0, 0, 255), width=3)
        # Переносица
        draw.line([70, 45, 130, 45], fill=(0, 0, 0, 255), width=3)
        # Дужки
        draw.line([10, 50, 0, 40], fill=(0, 0, 0, 255), width=3)
        draw.line([190, 50, 200, 40], fill=(0, 0, 0, 255), width=3)
        
        return np.array(glasses)

    def overlay_simple_glasses(face_img, glasses_img, left_center, right_center, scale=1.2):
        """Простое наложение очков"""
        try:
            face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
            glasses_pil = Image.fromarray(glasses_img)
            
            # Размер очков
            eye_distance = np.sqrt((right_center[0] - left_center[0])**2 + 
                                  (right_center[1] - left_center[1])**2)
            
            new_width = int(eye_distance * scale)
            new_height = int(new_width * glasses_pil.height / glasses_pil.width)
            
            glasses_resized = glasses_pil.resize((new_width, new_height))
            
            # Позиция
            center_x = (left_center[0] + right_center[0]) // 2
            center_y = (left_center[1] + right_center[1]) // 2
            
            x = center_x - new_width // 2
            y = center_y - new_height // 2 - 10
            
            # Наложение
            face_pil.paste(glasses_resized, (x, y), glasses_resized)
            
            return cv2.cvtColor(np.array(face_pil), cv2.COLOR_RGB2BGR)
            
        except Exception as e:
            st.error(f"Ошибка наложения: {e}")
            return face_img

# Интерфейс
st.title("🕶️ Виртуальная примерка очков")
st.write("Простая версия для демонстрации")

if not LIBS_AVAILABLE:
    st.stop()

if face_mesh is None:
    st.error("MediaPipe не инициализирован")
    st.stop()

# Загрузка фото
face_file = st.file_uploader("Загрузите фото лица", type=['jpg', 'png', 'jpeg'])

if face_file:
    try:
        # Загрузка и обработка
        face_image = Image.open(face_file)
        face_cv = cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(face_image, caption="Исходное фото", width=300)
        
        # Детекция лица
        landmarks = detect_face_landmarks(face_cv)
        
        if landmarks:
            # Получение координат глаз
            left_center, right_center, eye_distance, angle = get_eye_coordinates(landmarks, face_cv.shape)
            
            # Создание очков
            glasses_array = create_simple_glasses()
            
            # Наложение очков
            result = overlay_simple_glasses(face_cv, glasses_array, left_center, right_center)
            result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            
            with col2:
                st.image(result_rgb, caption="С очками", width=300)
                
            # Скачивание
            buf = io.BytesIO()
            Image.fromarray(result_rgb).save(buf, format='PNG')
            st.download_button(
                "Скачать результат",
                buf.getvalue(),
                "glasses_result.png",
                "image/png"
            )
            
        else:
            st.error("Лицо не найдено")
            
    except Exception as e:
        st.error(f"Общая ошибка: {e}")
else:
    st.info("Загрузите фото для начала работы")

st.write("---")
st.write("Упрощенная версия. Для полного функционала нужны все библиотеки.")
