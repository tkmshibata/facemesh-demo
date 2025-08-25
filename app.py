import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image

st.set_page_config(page_title="Custom FaceMesh", layout="centered")
st.title("ランドマーク検出")

# ★ モジュールエイリアスを用意しておくと参照しやすい
mp_face = mp.solutions.face_mesh

face_mesh = mp_face.FaceMesh(
    static_image_mode=True,
    refine_landmarks=True,
    max_num_faces=1,
    min_detection_confidence=0.3
)
mp_drawing = mp.solutions.drawing_utils
mp_style   = mp.solutions.drawing_styles

# ---------- ① 描画スタイル定義 ----------
thin_gray  = mp_drawing.DrawingSpec(color=(200,200,200), thickness=1, circle_radius=0)
cont_green = mp_drawing.DrawingSpec(color=(  0,255,  0), thickness=1, circle_radius=0)
iris_blue  = mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=2)
pts_accent = mp_drawing.DrawingSpec(color=(0,255,0), thickness=0, circle_radius=2)

uploaded = st.file_uploader("画像をアップロード", type=["jpg","jpeg","png"])
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    img_np = np.array(img)
    res = face_mesh.process(img_np)

    if res.multi_face_landmarks:
        annotated = img_np.copy()
        for lm_set in res.multi_face_landmarks:
            # ---------- ② 全体テセレーション ----------
            mp_drawing.draw_landmarks(
                annotated,
                lm_set,
                mp_face.FACEMESH_TESSELATION,       # ★
                landmark_drawing_spec=None,
                connection_drawing_spec=thin_gray
            )

            # ---------- ③ 輪郭線 ----------
            mp_drawing.draw_landmarks(
                annotated,
                lm_set,
                mp_face.FACEMESH_CONTOURS,          # ★
                landmark_drawing_spec=None,
                connection_drawing_spec=cont_green
            )

            # ---------- ④ 虹彩 ----------
            mp_drawing.draw_landmarks(
                annotated,
                lm_set,
                mp_face.FACEMESH_IRISES,            # ★
                landmark_drawing_spec=None,
                connection_drawing_spec=iris_blue
            )

            # ---------- ⑤ 点アクセント（目 & 唇） ----------
            eye_lip_conn = (
                mp_face.FACEMESH_LEFT_EYE
                | mp_face.FACEMESH_RIGHT_EYE
                | mp_face.FACEMESH_LIPS
            )
            accent_idx = {idx for pair in eye_lip_conn for idx in pair} 

            h, w, _ = annotated.shape
            for i, lm in enumerate(lm_set.landmark):
                if i in accent_idx:
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.circle(annotated, (x, y),
                               pts_accent.circle_radius,
                               pts_accent.color, -1)
        ###↓追記8/26###
        # NumPy → PIL に変換
        final_image = Image.fromarray(annotated)
        # ✅ PIL 画像で表示（これでエラー回避）
        st.image(final_image, caption="Custom drawn landmarks", use_container_width=True)
        ###↑追記###


        #st.image(annotated, caption="Custom drawn landmarks",use_container_width=True)
    else:
        st.error("顔を検出できませんでした")