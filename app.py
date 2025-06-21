import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

# --- ИНИЦИАЛИЗАЦИЯ И КОНСТАНТЫ ---
mp_face_mesh = mp.solutions.face_mesh

# <<< ИЗМЕНЕНИЕ 1: ИЗМЕНИЛИСЬ КЛЮЧЕВЫЕ ТОЧКИ >>>
# Теперь мы используем более точные точки для позиционирования
# 168: Точка между бровями (Nasion) - идеальный якорь для очков
# 234, 454: Виски (для определения ширины и наклона)
KEY_POINTS_GLASSES = [168, 234, 454] 

# Цвета для рамки (BGR формат для OpenCV)
COLOR_RED = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)

# <<< ИЗМЕНЕНИЕ 2: УДАЛЕНА НЕНАДЕЖНАЯ КОНСТАНТА VERTICAL_OFFSET_FACTOR >>>
# Мы больше не нуждаемся в ней, так как позиционирование теперь автоматическое.

# --- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ---
def overlay_glasses(face_image, glasses_image, vertical_offset=0):
    face_cv = np.array(face_image.convert('RGB'))
    face_cv = cv2.cvtColor(face_cv, cv2.COLOR_RGB2BGR)
    output_image = face_cv.copy()
    
    with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
        results = face_mesh.process(output_image)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w, _ = face_cv.shape
                
                # <<< ИЗМЕНЕНИЕ 3: ИСПОЛЬЗУЕМ НОВЫЕ ТОЧКИ ДЛЯ АВТОПОДГОНКИ >>>
                nasion_point = (int(face_landmarks.landmark[KEY_POINTS_GLASSES[0]].x * w), int(face_landmarks.landmark[KEY_POINTS_GLASSES[0]].y * h))
                left_temple = (int(face_landmarks.landmark[KEY_POINTS_GLASSES[1]].x * w), int(face_landmarks.landmark[KEY_POINTS_GLASSES[1]].y * h))
                right_temple = (int(face_landmarks.landmark[KEY_POINTS_GLASSES[2]].x * w), int(face_landmarks.landmark[KEY_POINTS_GLASSES[2]].y * h))
                
                # Определяем ширину и угол наклона по вискам
                scale_factor = 1.05
                glasses_width = int(np.linalg.norm(np.array(left_temple) - np.array(right_temple)))
                glasses_width = int(glasses_width * scale_factor)
                
                angle_rad = np.arctan2(right_temple[1] - left_temple[1], right_temple[0] - left_temple[0])
                angle_deg = np.degrees(angle_rad)
                
                # Масштабируем и поворачиваем изображение очков
                original_glasses_w, original_glasses_h = glasses_image.size
                aspect_ratio = original_glasses_h / original_glasses_w
                new_h = int(glasses_width * aspect_ratio)
                
                if new_h == 0 or glasses_width == 0:
                    return None
                
                resized_glasses = glasses_image.resize((glasses_width, new_h))
                rotated_glasses = resized_glasses.rotate(-angle_deg, expand=True, resample=Image.BICUBIC)
                
                rotated_w, rotated_h = rotated_glasses.size
                
                # <<< ИЗМЕНЕНИЕ 4: НОВАЯ ЛОГИКА РАСЧЕТА ПОЛОЖЕНИЯ (АВТОПОДГОНКА) >>>
                # Центрируем очки по точке между бровями (nasion_point)
                paste_x = nasion_point[0] - rotated_w // 2
                paste_y = nasion_point[1] - rotated_h // 2
                
                # <<< ИЗМЕНЕНИЕ 5: ПРИМЕНЯЕМ РУЧНУЮ КОРРЕКТИРОВКУ ОТ СЛАЙДЕРА >>>
                paste_y += vertical_offset
                
                # Накладываем изображение
                output_pil = Image.fromarray(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
                output_pil.paste(rotated_glasses, (paste_x, paste_y), rotated_glasses)
                return output_pil
    return None

# --- КЛАСС ДЛЯ ОБРАБОТКИ ВИДЕО В РЕАЛЬНОМ ВРЕМЕНИ ---
class FaceAlignmentProcessor(VideoProcessorBase):
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.latest_good_frame = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        h, w, _ = image.shape
        
        oval_center = (w // 2, h // 2)
        oval_axes = (int(w * 0.3), int(h * 0.4))
        
        results = self.face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        is_ready = False
        message = "Поместите лицо в рамку"
        frame_color = COLOR_RED

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Используем старые точки для проверки положения в кадре
                left_temple = face_landmarks.landmark[KEY_POINTS_GLASSES[1]]
                right_temple = face_landmarks.landmark[KEY_POINTS_GLASSES[2]]
                
                angle_rad = np.arctan2(right_temple.y - left_temple.y, right_temple.x - left_temple.x)
                angle_deg = abs(np.degrees(angle_rad))
                
                face_width_ratio = np.linalg.norm([right_temple.x - left_temple.x, right_temple.y - left_temple.y])

                if angle_deg > 10:
                    message = "Выровняйте голову"
                elif face_width_ratio < 0.3:
                    message = "Подвиньтесь ближе"
                elif face_width_ratio > 0.6:
                    message = "Подвиньтесь дальше"
                else:
                    message = "Отлично! Можно делать снимок"
                    frame_color = COLOR_GREEN
                    is_ready = True
        
        cv2.ellipse(image, oval_center, oval_axes, 0, 0, 360, frame_color, 3)

        # <<< ИЗМЕНЕНИЕ 6: БОЛЕЕ УЗКАЯ ЗОНА ДЛЯ ГЛАЗ >>>
        line_width = oval_axes[0]
        line_start_x = oval_center[0] - line_width
        line_end_x = oval_center[0] + line_width
        
        # Сделали зону уже. Верхняя линия чуть ниже, нижняя чуть выше.
        eye_line_top_y = oval_center[1] - int(oval_axes[1] * 0.25)
        eye_line_bottom_y = oval_center[1] + int(oval_axes[1] * 0.0)

        cv2.line(image, (line_start_x, eye_line_top_y), (line_end_x, eye_line_top_y), frame_color, 1)
        cv2.line(image, (line_start_x, eye_line_bottom_y), (line_end_x, eye_line_bottom_y), frame_color, 1)

        cv2.putText(image, message, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, frame_color, 2)

        if is_ready:
            self.latest_good_frame = image.copy()
        else:
            self.latest_good_frame = None

        return av.VideoFrame.from_ndarray(image, format="bgr24")


# --- ИНТЕРФЕЙС STREAMLIT ---
st.set_page_config(layout="wide", page_title="Виртуальная примерка очков")
st.title("Прототип сервиса виртуальной примерки очков 👓")

if "photo" not in st.session_state:
    st.session_state.photo = None

col1, col2 = st.columns([0.6, 0.4])

with col1:
    st.header("Шаг 1: Сделайте фото")
    
    if st.session_state.photo is None:
        webrtc_ctx = webrtc_streamer(
            key="face-alignment",
            video_processor_factory=FaceAlignmentProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

        if st.button("Сделать снимок"):
            if webrtc_ctx.video_processor and webrtc_ctx.video_processor.latest_good_frame is not None:
                st.session_state.photo = webrtc_ctx.video_processor.latest_good_frame
                st.rerun()
            else:
                st.warning("Не удалось сделать снимок. Пожалуйста, убедитесь, что ваше лицо находится в зеленой рамке.")
    
    else:
        st.success("Отличное фото!")
        st.image(st.session_state.photo, channels="BGR", caption="Ваше фото для примерки")
        if st.button("Сделать новое фото"):
            st.session_state.photo = None
            st.rerun()

with col2:
    st.header("Шаг 2: Выберите и настройте очки")

    uploaded_glasses_file = st.file_uploader("Загрузите PNG файл с очками", type=["png"])
    
    # <<< ИЗМЕНЕНИЕ 7: ДОБАВЛЕН СЛАЙДЕР ДЛЯ РУЧНОЙ ПОДСТРОЙКИ >>>
    manual_offset = st.slider(
        "Поднять / Опустить очки",
        min_value=-50,  # Можно опустить на 50 пикселей
        max_value=50,   # Можно поднять на 50 пикселей
        value=0,        # По умолчанию без смещения
        step=1,
        help="Двигайте слайдер, чтобы точно подогнать положение очков по вертикали."
    )

    if st.session_state.photo is not None and uploaded_glasses_file is not None:
        st.divider()
        st.header(f"Результат примерки: «{uploaded_glasses_file.name}»")
        
        face_pil_image = Image.fromarray(cv2.cvtColor(st.session_state.photo, cv2.COLOR_BGR2RGB))
        glasses_pil_image = Image.open(uploaded_glasses_file).convert("RGBA")

        with st.spinner('Накладываем очки...'):
            # Передаем значение слайдера в функцию
            result_image = overlay_glasses(face_pil_image, glasses_pil_image, vertical_offset=manual_offset)

        if result_image:
            st.image(result_image, caption="Вот как вы будете выглядеть!")
        else:
            st.error("Не удалось обработать фото. Возможно, лицо на снимке распознано нечетко.")
    elif st.session_state.photo is not None:
        st.info("Теперь загрузите файл с очками и настройте их положение, чтобы увидеть результат.")
    else:
        st.info("Сначала сделайте фото, чтобы начать примерку.")
