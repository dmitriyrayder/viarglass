import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw
import io
import base64

# Настройка MediaPipe
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

def detect_face_landmarks(image):
    """Определяет ключевые точки лица"""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0]
        return landmarks
    return None

def get_eye_coordinates(landmarks, image_shape):
    """Получает координаты глаз и расстояние между ними"""
    h, w = image_shape[:2]
    
    # Индексы ключевых точек для глаз (более точные точки)
    left_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161]
    right_eye_indices = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384]
    
    # Координаты глаз
    left_eye = [(landmarks.landmark[i].x * w, landmarks.landmark[i].y * h) 
                for i in left_eye_indices]
    right_eye = [(landmarks.landmark[i].x * w, landmarks.landmark[i].y * h) 
                 for i in right_eye_indices]
    
    # Центры глаз
    left_center = (int(np.mean([p[0] for p in left_eye])), 
                   int(np.mean([p[1] for p in left_eye])))
    right_center = (int(np.mean([p[0] for p in right_eye])), 
                    int(np.mean([p[1] for p in right_eye])))
    
    # Расстояние между глазами
    eye_distance = np.sqrt((right_center[0] - left_center[0])**2 + 
                          (right_center[1] - left_center[1])**2)
    
    # Угол наклона
    angle = np.arctan2(right_center[1] - left_center[1], 
                      right_center[0] - left_center[0])
    
    return left_center, right_center, eye_distance, angle

def resize_and_rotate_glasses(glasses_img, eye_distance, angle, scale_factor=1.2):
    """Масштабирует и поворачивает очки"""
    # Проверяем количество каналов
    if len(glasses_img.shape) == 3:
        if glasses_img.shape[2] == 4:  # BGRA
            glasses_pil = Image.fromarray(cv2.cvtColor(glasses_img, cv2.COLOR_BGRA2RGBA))
        else:  # BGR
            # Добавляем альфа-канал
            alpha = np.ones((glasses_img.shape[0], glasses_img.shape[1], 1), dtype=glasses_img.dtype) * 255
            glasses_img = np.concatenate([glasses_img, alpha], axis=2)
            glasses_pil = Image.fromarray(cv2.cvtColor(glasses_img, cv2.COLOR_BGRA2RGBA))
    else:
        raise ValueError("Неподдерживаемый формат изображения очков")
    
    # Масштабирование
    glasses_width = int(eye_distance * scale_factor)
    aspect_ratio = glasses_pil.height / glasses_pil.width
    glasses_height = int(glasses_width * aspect_ratio)
    
    glasses_resized = glasses_pil.resize((glasses_width, glasses_height), Image.Resampling.LANCZOS)
    
    # Поворот
    angle_degrees = np.degrees(angle)
    glasses_rotated = glasses_resized.rotate(angle_degrees, expand=True)
    
    return glasses_rotated

def overlay_glasses(face_img, glasses_img, left_center, right_center, eye_distance, angle, y_offset=0):
    """Накладывает очки на лицо"""
    # Создаем копию изображения лица
    result = face_img.copy()
    
    # Масштабируем и поворачиваем очки
    glasses_processed = resize_and_rotate_glasses(glasses_img, eye_distance, angle)
    
    # Центр между глазами
    center_x = int((left_center[0] + right_center[0]) / 2)
    center_y = int((left_center[1] + right_center[1]) / 2)
    
    # Смещение по Y для правильного позиционирования
    default_y_offset = -int(glasses_processed.height * 0.1)
    
    # Позиция для наложения
    x_pos = center_x - glasses_processed.width // 2
    y_pos = center_y - glasses_processed.height // 2 + default_y_offset + y_offset
    
    # Преобразуем обратно в OpenCV формат
    glasses_cv = cv2.cvtColor(np.array(glasses_processed), cv2.COLOR_RGBA2BGRA)
    
    # Наложение с учетом прозрачности
    for i in range(glasses_cv.shape[0]):
        for j in range(glasses_cv.shape[1]):
            y_coord = y_pos + i
            x_coord = x_pos + j
            
            if (0 <= y_coord < result.shape[0] and 0 <= x_coord < result.shape[1]):
                alpha = glasses_cv[i, j, 3] / 255.0
                if alpha > 0:
                    for c in range(3):
                        result[y_coord, x_coord, c] = (
                            alpha * glasses_cv[i, j, c] + 
                            (1 - alpha) * result[y_coord, x_coord, c]
                        )
    
    return result

def create_demo_glasses():
    """Создает демонстрационные очки"""
    glasses = np.zeros((100, 200, 4), dtype=np.uint8)
    
    # Рисуем линзы (черные с прозрачностью)
    cv2.circle(glasses, (50, 50), 30, (0, 0, 0, 180), 3)
    cv2.circle(glasses, (150, 50), 30, (0, 0, 0, 180), 3)
    
    # Переносица
    cv2.line(glasses, (80, 45), (120, 45), (0, 0, 0, 180), 3)
    
    # Дужки
    cv2.line(glasses, (20, 50), (5, 40), (0, 0, 0, 180), 3)
    cv2.line(glasses, (180, 50), (195, 40), (0, 0, 0, 180), 3)
    
    return glasses

# Streamlit интерфейс
st.set_page_config(page_title="Виртуальная примерка очков", page_icon="🕶️")
st.title("🕶️ Виртуальная примерка очков")
st.write("Загрузите фото лица и изображение очков для виртуальной примерки!")

# Боковая панель для настроек
st.sidebar.header("Настройки")

# Загрузка изображений
col1, col2 = st.columns(2)

with col1:
    st.subheader("📸 Загрузите фото лица")
    face_file = st.file_uploader("Выберите изображение лица", 
                                type=['jpg', 'jpeg', 'png'], 
                                key='face')

with col2:
    st.subheader("🕶️ Загрузите очки")
    glasses_file = st.file_uploader("Выберите изображение очков", 
                                   type=['png', 'jpg', 'jpeg'], 
                                   key='glasses')
    
    # Кнопка для демо очков
    use_demo_glasses = st.button("Использовать демо очки")

# Настройки в боковой панели
scale_factor = st.sidebar.slider("Масштаб очков", 0.8, 2.0, 1.2, 0.1)
y_offset = st.sidebar.slider("Смещение по вертикали", -50, 50, 0, 5)

if face_file is not None:
    try:
        # Загружаем изображение лица
        face_bytes = face_file.read()
        face_image = Image.open(io.BytesIO(face_bytes))
        face_cv = cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR)
        
        st.subheader("Исходное изображение")
        st.image(face_image, caption="Загруженное лицо", width=300)
        
        # Определяем ключевые точки лица
        landmarks = detect_face_landmarks(face_cv)
        
        if landmarks:
            # Получаем координаты глаз
            left_center, right_center, eye_distance, angle = get_eye_coordinates(landmarks, face_cv.shape)
            
            st.success("✅ Лицо обнаружено! Ключевые точки найдены.")
            
            # Показываем позиции глаз
            face_with_points = face_cv.copy()
            cv2.circle(face_with_points, left_center, 5, (0, 255, 0), -1)
            cv2.circle(face_with_points, right_center, 5, (0, 255, 0), -1)
            cv2.line(face_with_points, left_center, right_center, (255, 0, 0), 2)
            
            st.subheader("Обнаруженные точки глаз")
            st.image(cv2.cvtColor(face_with_points, cv2.COLOR_BGR2RGB), 
                    caption="Зеленые точки - центры глаз", width=300)
            
            # Обработка очков
            glasses_cv = None
            
            if glasses_file is not None:
                try:
                    glasses_bytes = glasses_file.read()
                    glasses_image = Image.open(io.BytesIO(glasses_bytes))
                    
                    # Преобразуем в нужный формат
                    if glasses_image.mode == 'RGBA':
                        glasses_cv = cv2.cvtColor(np.array(glasses_image), cv2.COLOR_RGBA2BGRA)
                    elif glasses_image.mode == 'RGB':
                        glasses_array = np.array(glasses_image)
                        alpha = np.ones((glasses_array.shape[0], glasses_array.shape[1], 1), dtype=np.uint8) * 255
                        glasses_rgba = np.concatenate([glasses_array, alpha], axis=2)
                        glasses_cv = cv2.cvtColor(glasses_rgba, cv2.COLOR_RGBA2BGRA)
                    else:
                        st.error("Неподдерживаемый формат изображения очков")
                        glasses_cv = None
                    
                    if glasses_cv is not None:
                        st.subheader("Загруженные очки")
                        st.image(glasses_image, caption="Очки для примерки", width=200)
                        
                except Exception as e:
                    st.error(f"Ошибка при загрузке очков: {str(e)}")
                    glasses_cv = None
                
            elif use_demo_glasses:
                glasses_cv = create_demo_glasses()
                st.subheader("Демонстрационные очки")
                st.image(cv2.cvtColor(glasses_cv, cv2.COLOR_BGRA2RGBA), 
                        caption="Демо очки", width=200)
            
            # Виртуальная примерка
            if glasses_cv is not None:
                try:
                    # Применяем очки с учетом настроек
                    result = overlay_glasses(face_cv, glasses_cv, left_center, right_center, 
                                           eye_distance, angle, y_offset)
                    
                    st.subheader("🎉 Результат виртуальной примерки")
                    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                    st.image(result_rgb, caption="Вы в очках!", width=400)
                    
                    # Кнопка для скачивания
                    result_pil = Image.fromarray(result_rgb)
                    buf = io.BytesIO()
                    result_pil.save(buf, format='PNG')
                    buf.seek(0)
                    
                    st.download_button(
                        label="📥 Скачать результат",
                        data=buf.getvalue(),
                        file_name="virtual_tryout.png",
                        mime="image/png"
                    )
                    
                except Exception as e:
                    st.error(f"Ошибка при наложении очков: {str(e)}")
                    st.info("Попробуйте другое изображение очков или измените настройки")
            else:
                st.info("Загрузите изображение очков или используйте демо очки")
                
        else:
            st.error("❌ Лицо не обнаружено. Попробуйте другое изображение.")
            st.info("Советы: используйте четкое фото лица анфас с хорошим освещением")
    
    except Exception as e:
        st.error(f"Ошибка при обработке изображения: {str(e)}")
        st.info("Проверьте формат загруженного файла")

else:
    st.info("👆 Загрузите фото лица для начала работы")
    
    # Демонстрационный пример
    st.subheader("Как это работает:")
    st.write("""
    1. **Загрузите фото лица** - желательно анфас с хорошим освещением
    2. **Загрузите очки** - PNG/JPG файл (PNG с прозрачностью предпочтительнее)
    3. **Настройте параметры** - масштаб и позиционирование в боковой панели
    4. **Получите результат** - система автоматически найдет глаза и наложит очки
    
    **Поддерживаемые форматы:**
    - Лицо: JPG, JPEG, PNG
    - Очки: PNG, JPG, JPEG
    """)

# Информация о технологиях
with st.expander("🔧 Технические детали"):
    st.write("""
    **Используемые технологии:**
    - **MediaPipe** - для детекции лицевых точек
    - **OpenCV** - для обработки изображений
    - **Streamlit** - для веб-интерфейса
    
    **Алгоритм работы:**
    1. Обнаружение 468 ключевых точек лица
    2. Определение позиций и размеров глаз
    3. Вычисление угла наклона головы
    4. Масштабирование и поворот очков
    5. Наложение с учетом прозрачности
    """)

# Советы по использованию
with st.expander("💡 Советы для лучшего результата"):
    st.write("""
    **Для фото лица:**
    - Используйте четкое изображение анфас
    - Хорошее освещение без теней на лице
    - Глаза должны быть открыты и видны
    - Избегайте сильных поворотов головы
    
    **Для очков:**
    - PNG с прозрачным фоном дает лучший результат
    - Очки должны быть сфотографированы прямо
    - Хорошее качество изображения
    - Размер файла не более 10MB
    
    **Настройки:**
    - Используйте слайдеры для точной подгонки
    - Масштаб: увеличьте для больших очков
    - Смещение: отрегулируйте высоту посадки
    """)
