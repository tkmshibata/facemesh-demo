import math
from typing import Dict, Tuple, List

import cv2
import numpy as np
import streamlit as st
from PIL import Image
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2 as mp_landmark

# ===================== ページ設定 & UI微調整 =====================
st.set_page_config(page_title="ランドマーク検出 × 黄金比/白銀比", page_icon="📐", layout="centered")
st.markdown("""
<style>
#MainMenu {visibility:hidden;} header {visibility:hidden;} footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

st.title("ランドマーク検出 × 黄金比 / 白銀比（MVP）")
st.caption("MediaPipe FaceMesh で取得した顔ランドマークから、全体の縦横比と主要比率を φ(1.618) / √2(1.414) と比較します。")

# ===================== MediaPipe 初期化（キャッシュ） =====================
@st.cache_resource
def get_facemesh():
    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        refine_landmarks=True,
        max_num_faces=1,
        min_detection_confidence=0.3
    )

mp_face = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles
face_mesh = get_facemesh()

# ===================== ユーティリティ =====================
def np_from_pil(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("RGB"))

def dist(p1, p2) -> float:
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))

def landmarks_to_xy(landmarks, w: int, h: int) -> np.ndarray:
    """468点を (N,2) のnp配列に"""
    return np.array([(lm.x * w, lm.y * h) for lm in landmarks], dtype=np.float32)

def face_oval_indices() -> List[int]:
    # MediaPipeが提供する接続から index 集合を抽出
    conn = mp_face.FACEMESH_FACE_OVAL
    return sorted({i for pair in conn for i in pair})

def extreme_points_xy(xy: np.ndarray, idxs: List[int]) -> Tuple[Tuple[float,float], Tuple[float,float], Tuple[float,float], Tuple[float,float]]:
    """顔外周 idx 群から (left, right, top, bottom) を返す"""
    pts = xy[idxs]
    left   = tuple(pts[pts[:,0].argmin()])
    right  = tuple(pts[pts[:,0].argmax()])
    top    = tuple(pts[pts[:,1].argmin()])
    bottom = tuple(pts[pts[:,1].argmax()])
    return left, right, top, bottom

def compute_core_ratios(xy: np.ndarray) -> Dict[str, float]:
    """
    主要指標（MVP）
      - face_AR: 顔の縦横比 = 高さ/幅
      - eye_spacing_ratio: 目と目の中心距離 / 片目幅（≈1が目安、審美論では黄金比ではないがMVPで参考値）
      - mouth_to_nose_ratio: 口幅 / 鼻幅（参考）
    """
    # 代表ランドマーク（MediaPipeのよく使う番号）
    # 33: 右目外側, 133: 右目内側, 263: 左目外側, 362: 左目内側
    # 61: 口右, 291: 口左, 2/97/326 あたりが鼻基部周辺（今回は97,326を鼻翼端の近似に）
    RIGHT_EYE_OUT, RIGHT_EYE_IN, LEFT_EYE_IN, LEFT_EYE_OUT = 33, 133, 362, 263
    MOUTH_RIGHT, MOUTH_LEFT = 61, 291
    NOSE_LEFT, NOSE_RIGHT = 97, 326  # 鼻翼端の近似
    CHIN = 152
    FOREHEAD = 10  # 額上部の近似（髪際≠厳密）

    # 全体縦横比
    oval_idxs = face_oval_indices()
    left, right, top, bottom = extreme_points_xy(xy, oval_idxs)
    face_w = dist(left, right)
    face_h = dist(top, bottom)
    face_AR = face_h / face_w if face_w > 1e-6 else np.nan

    # 目・口・鼻の簡易比
    right_eye_w = dist(xy[RIGHT_EYE_OUT], xy[RIGHT_EYE_IN])
    left_eye_w  = dist(xy[LEFT_EYE_OUT],  xy[LEFT_EYE_IN])
    eye_w = (right_eye_w + left_eye_w) / 2.0

    # 両目中心の距離
    right_eye_center = (xy[RIGHT_EYE_OUT] + xy[RIGHT_EYE_IN]) / 2.0
    left_eye_center  = (xy[LEFT_EYE_OUT]  + xy[LEFT_EYE_IN])  / 2.0
    interocular = dist(right_eye_center, left_eye_center)
    eye_spacing_ratio = interocular / eye_w if eye_w > 1e-6 else np.nan

    mouth_w = dist(xy[MOUTH_RIGHT], xy[MOUTH_LEFT])
    nose_w  = dist(xy[NOSE_LEFT], xy[NOSE_RIGHT])
    mouth_to_nose_ratio = mouth_w / nose_w if nose_w > 1e-6 else np.nan

    return {
        "face_AR": face_AR,
        "eye_spacing_ratio": eye_spacing_ratio,
        "mouth_to_nose_ratio": mouth_to_nose_ratio,
        "face_w": face_w,
        "face_h": face_h,
    }

def compare_to_ideals(metrics: Dict[str, float]) -> List[Dict]:
    """
    黄金比/白銀比との誤差[%]（MVPでは face_AR のみ比較）
    他の比率は参考値としてそのまま表示。
    """
    PHI = (1 + 5 ** 0.5) / 2   # 1.618...
    SILVER = 2 ** 0.5          # 1.414...

    face_AR = metrics["face_AR"]
    def err_pct(v, target):
        return float((v - target) / target * 100.0) if np.isfinite(v) else np.nan

    rows = [
        {"metric": "Face aspect ratio (H/W)",
         "value": round(face_AR, 3) if np.isfinite(face_AR) else None,
         "ideal": "φ=1.618", "diff_%": round(err_pct(face_AR, PHI), 2) if np.isfinite(face_AR) else None,
         "target": "golden"},
        {"metric": "Face aspect ratio (H/W)",
         "value": round(face_AR, 3) if np.isfinite(face_AR) else None,
         "ideal": "√2=1.414", "diff_%": round(err_pct(face_AR, SILVER), 2) if np.isfinite(face_AR) else None,
         "target": "silver"},
        {"metric": "Eye spacing / eye width",
         "value": round(metrics["eye_spacing_ratio"], 3) if np.isfinite(metrics["eye_spacing_ratio"]) else None,
         "ideal": "参考: ≈1.0", "diff_%": None, "target": "—"},
        {"metric": "Mouth width / Nose width",
         "value": round(metrics["mouth_to_nose_ratio"], 3) if np.isfinite(metrics["mouth_to_nose_ratio"]) else None,
         "ideal": "参考: 個人差大", "diff_%": None, "target": "—"},
    ]
    return rows

def draw_overlay(image_rgb: np.ndarray, xy: np.ndarray, metrics: Dict[str, float]) -> np.ndarray:
    """
    元画像に：ランドマークの薄いメッシュ + 現実の外接矩形 + φ/√2 長方形を重ねる
    """
    annotated = image_rgb.copy()
    h, w, _ = annotated.shape

    # 1) ランドマークの薄い描画
    nl = mp_landmark.NormalizedLandmarkList(
        landmark=[
            mp_landmark.NormalizedLandmark(x=float(x)/w, y=float(y)/h)
            for (x, y) in xy
        ]
    )
    mp_draw.draw_landmarks(
        annotated,
        nl,
        mp_face.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_draw.DrawingSpec(color=(200,200,200), thickness=1, circle_radius=0)
    )

    # 2) 顔外周の極値から実測の外接矩形を描く
    oval_idxs = face_oval_indices()
    left, right, top, bottom = extreme_points_xy(xy, oval_idxs)
    x_min, x_max = int(min(left[0], right[0])), int(max(left[0], right[0]))
    y_min, y_max = int(min(top[1], bottom[1])), int(max(top[1], bottom[1]))

    cv2.rectangle(annotated, (x_min, y_min), (x_max, y_max), (0,255,0), 2)  # 実測

    # 3) その幅に対して黄金/白銀の高さで矩形を重ねる（中央合わせ）
    width = x_max - x_min
    cx = (x_min + x_max) // 2
    PHI = (1 + 5 ** 0.5) / 2
    SILVER = 2 ** 0.5

    def draw_ratio_rect(ratio, color):
        target_h = int(width * ratio)
        y_mid = (y_min + y_max) // 2
        y1 = max(0, y_mid - target_h // 2)
        y2 = min(h-1, y_mid + target_h // 2)
        x1 = max(0, cx - width // 2)
        x2 = min(w-1, cx + width // 2)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

    draw_ratio_rect(PHI,   (255,0,0))   # 赤：黄金比
    draw_ratio_rect(SILVER,(0,0,255))   # 青：白銀比

    return annotated

# ===================== UI: アップロード → 推論 =====================
uploaded = st.file_uploader("画像をアップロード", type=["jpg","jpeg","png"])
if not uploaded:
    st.stop()

img = Image.open(uploaded).convert("RGB")
np_img = np_from_pil(img)

# 処理を軽くする（長辺 1280px）
long_edge = max(np_img.shape[:2])
if long_edge > 1280:
    scale = 1280 / long_edge
    np_img = cv2.resize(np_img, (int(np_img.shape[1]*scale), int(np_img.shape[0]*scale)))

# MediaPipe 実行
res = face_mesh.process(np_img)
if not res.multi_face_landmarks:
    st.error("顔を検出できませんでした。正面に近い画像で再試行してください。")
    st.stop()

landmarks = res.multi_face_landmarks[0].landmark
h, w, _ = np_img.shape
xy = landmarks_to_xy(landmarks, w, h)

# 指標計算 & 可視化
metrics = compute_core_ratios(xy)
overlay = draw_overlay(np_img, xy, metrics)

st.subheader("比較オーバーレイ")
st.image(Image.fromarray(overlay), use_container_width=True, caption="緑=実測矩形 / 赤=黄金比(φ) / 青=白銀比(√2)")

rows = compare_to_ideals(metrics)

# 表示（テーブル）
import pandas as pd
df = pd.DataFrame(rows)
st.subheader("比率の比較（誤差は黄金/白銀のみ）")
st.dataframe(df, use_container_width=True)

# 棒グラフ（黄金/白銀の誤差）
err_plot = df[df["diff_%"].notna()][["ideal","diff_%"]].set_index("ideal")
st.bar_chart(err_plot)

st.caption("※ 本MVPは “全体の縦横比” を黄金比/白銀比に照合した簡易比較です。審美の基準には個人差・文化差があります。")
