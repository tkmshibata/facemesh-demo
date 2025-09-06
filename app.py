import math
from typing import Dict, Tuple, List
import cv2
import numpy as np
import streamlit as st
from PIL import Image
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2 as mp_landmark
import pandas as pd

st.set_page_config(page_title="顔 × 黄金比/白銀比 比較", page_icon="📐", layout="centered")
st.markdown("<style>#MainMenu,header,footer{visibility:hidden;}</style>", unsafe_allow_html=True)
st.title("ランドマーク検出 × 黄金比 / 白銀比")

# ---------------- MediaPipe (cache) ----------------
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
face_mesh = get_facemesh()

# ---------------- utils ----------------
def pil2np(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("RGB"))

def dist(p1, p2) -> float:
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))

def landmarks_to_xy(landmarks, w: int, h: int) -> np.ndarray:
    return np.array([(lm.x * w, lm.y * h) for lm in landmarks], dtype=np.float32)

def face_oval_indices() -> List[int]:
    return sorted({i for pair in mp_face.FACEMESH_FACE_OVAL for i in pair})

def rotate_points(xy: np.ndarray, center, angle_rad: float) -> np.ndarray:
    """2D点群を回転（OpenCVの画像回転と整合）"""
    R = np.array([[math.cos(angle_rad), -math.sin(angle_rad)],
                  [math.sin(angle_rad),  math.cos(angle_rad)]], dtype=np.float32)
    return ((xy - center) @ R.T) + center

def dashed_line(img, pt1, pt2, color, thickness=2, dash=10, gap=8):
    """OpenCVで破線"""
    p1 = np.array(pt1, dtype=int); p2 = np.array(pt2, dtype=int)
    length = np.linalg.norm(p2 - p1)
    if length < 1: return
    v = (p2 - p1) / length
    n = int(length // (dash + gap)) + 1
    for i in range(n):
        s = p1 + (dash + gap) * i * v
        e = p1 + ((dash + gap) * i + dash) * v
        cv2.line(img, tuple(s.astype(int)), tuple(e.astype(int)), color, thickness)

# ---------------- metrics ----------------
IDX = dict(
    R_E_OUT=33, R_E_IN=133, L_E_IN=362, L_E_OUT=263,
    M_R=61, M_L=291, NOSE_L=97, NOSE_R=326,
    CHIN=152, FOREHEAD=10,  # 髪際近似
    BROW_R_UP=105, BROW_L_UP=334,  # 眉上の近似点
    NOSE_TIP=1
)

def compute_metrics(xy: np.ndarray) -> Dict[str, float]:
    # 顔外周の極値 → 幅/高さ
    oval = face_oval_indices()
    pts = xy[oval]
    left  = tuple(pts[pts[:,0].argmin()])
    right = tuple(pts[pts[:,0].argmax()])
    top   = tuple(pts[pts[:,1].argmin()])
    bottom= tuple(pts[pts[:,1].argmax()])
    face_w = dist(left, right)
    face_h = dist(top, bottom)
    face_AR = face_h / face_w if face_w>1e-6 else np.nan

    # 目幅 & 目中心間
    re_w = dist(xy[IDX["R_E_OUT"]], xy[IDX["R_E_IN"]])
    le_w = dist(xy[IDX["L_E_OUT"]], xy[IDX["L_E_IN"]])
    eye_w = (re_w + le_w) / 2.0
    re_c = (xy[IDX["R_E_OUT"]] + xy[IDX["R_E_IN"]]) / 2.0
    le_c = (xy[IDX["L_E_OUT"]] + xy[IDX["L_E_IN"]]) / 2.0
    interocular = dist(re_c, le_c)
    eye_spacing_ratio = interocular / eye_w if eye_w>1e-6 else np.nan

    # 鼻幅 / 口幅（要求どおり「鼻幅/口幅」）
    nose_w = dist(xy[IDX["NOSE_L"]], xy[IDX["NOSE_R"]])
    mouth_w = dist(xy[IDX["M_R"]], xy[IDX["M_L"]])
    nose_to_mouth = nose_w / mouth_w if mouth_w>1e-6 else np.nan

    # 三分割（髪際–眉–鼻–顎）
    hair_y = xy[IDX["FOREHEAD"]][1]
    brow_y = ((xy[IDX["BROW_R_UP"]][1] + xy[IDX["BROW_L_UP"]][1]) / 2.0)
    nose_y = xy[IDX["NOSE_TIP"]][1]
    chin_y = xy[IDX["CHIN"]][1]
    H = (chin_y - hair_y)
    thirds = ((brow_y - hair_y)/H, (nose_y - brow_y)/H, (chin_y - nose_y)/H) if H>1e-6 else (np.nan, np.nan, np.nan)

    return dict(
        face_w=face_w, face_h=face_h, face_AR=face_AR,
        eye_spacing_ratio=eye_spacing_ratio,
        nose_to_mouth=nose_to_mouth,
        thirds_top=thirds[0], thirds_mid=thirds[1], thirds_bot=thirds[2],
        hair_y=hair_y, brow_y=brow_y, nose_y=nose_y, chin_y=chin_y
    )

# ---------------- alignment & crop ----------------
def align_and_crop(img_rgb: np.ndarray, xy: np.ndarray, margin=0.2) -> Tuple[np.ndarray, np.ndarray]:
    """目の水平で回転→外周BBoxで切出し→座標も同じ変換"""
    h, w, _ = img_rgb.shape
    # 目の中心で角度
    re_c = (xy[IDX["R_E_OUT"]] + xy[IDX["R_E_IN"]]) / 2.0
    le_c = (xy[IDX["L_E_OUT"]] + xy[IDX["L_E_IN"]]) / 2.0
    dy, dx = (le_c[1] - re_c[1]), (le_c[0] - re_c[0])
    angle = math.degrees(math.atan2(dy, dx))
    center = (w/2.0, h/2.0)

    # 画像を回転
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rot = cv2.warpAffine(img_rgb, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    # 点群も回転
    ones = np.ones((xy.shape[0], 1))
    xy_h = np.hstack([xy, ones])
    xy_rot = (M @ xy_h.T).T  # (N,2)

    # 顔外周からBBox
    oval = face_oval_indices()
    pts = xy_rot[oval]
    x1, y1 = np.min(pts, axis=0)
    x2, y2 = np.max(pts, axis=0)
    # 余白
    w0, h0 = x2-x1, y2-y1
    x1 -= w0*margin; x2 += w0*margin
    y1 -= h0*margin; y2 += h0*margin
    x1 = int(max(0, x1)); y1 = int(max(0, y1))
    x2 = int(min(w-1, x2)); y2 = int(min(h-1, y2))

    crop = rot[y1:y2, x1:x2].copy()
    xy_crop = xy_rot.copy()
    xy_crop[:,0] -= x1; xy_crop[:,1] -= y1
    return crop, xy_crop

# ---------------- overlay ----------------
PHI = (1 + 5**0.5)/2.0
SILVER = 2**0.5

def build_overlay(crop: np.ndarray, xy: np.ndarray, target_ratio: float, label: str) -> np.ndarray:
    """縦横比ターゲット（黄金/白銀）の枠 + 三分割理想線(1/3等分) を重ねる"""
    out = crop.copy()
    h, w, _ = out.shape

    # 1) ランドマークの薄いメッシュ
    nl = mp_landmark.NormalizedLandmarkList(
        landmark=[mp_landmark.NormalizedLandmark(x=float(x)/w, y=float(y)/h) for (x,y) in xy]
    )
    mp_draw.draw_landmarks(
        out, nl, mp_face.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_draw.DrawingSpec(color=(200,200,200), thickness=1, circle_radius=0)
    )

    # 2) 実測枠（緑）
    oval = face_oval_indices()
    pts = xy[oval]
    x1, y1 = pts[:,0].min(), pts[:,1].min()
    x2, y2 = pts[:,0].max(), pts[:,1].max()
    x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)
    cv2.rectangle(out, (x1i,y1i), (x2i,y2i), (0,255,0), 2)

    # 3) ターゲット枠（中央合わせ）
    width = x2i - x1i
    cx = (x1i + x2i)//2
    target_h = int(width * target_ratio)
    ymid = (y1i + y2i)//2
    ty1 = max(0, ymid - target_h//2)
    ty2 = min(h-1, ymid + target_h//2)
    tx1 = max(0, cx - width//2)
    tx2 = min(w-1, cx + width//2)
    color = (0,0,255) if label=="SILVER" else (255,0,0)
    cv2.rectangle(out, (tx1,ty1), (tx2,ty2), color, 2)

    # 4) 三分割：理想（破線）＆ 実測（実線）
    # 理想(1/3等分)
    for i in (1,2):
        y = int(ty1 + (target_h/3)*i)
        dashed_line(out, (tx1,y), (tx2,y), color, thickness=2, dash=14, gap=10)
    # 実測（三分点）: 髪際-眉-鼻-顎を線で
    # 髪際 ~ FOREHEAD(10), 眉 ~ 平均(105,334), 鼻 ~ NOSE_TIP(1), 顎 ~ 152
    hair = int(xy[IDX["FOREHEAD"]][1])
    brow = int((xy[IDX["BROW_R_UP"]][1] + xy[IDX["BROW_L_UP"]][1]) / 2.0)
    nose = int(xy[IDX["NOSE_TIP"]][1])
    chin = int(xy[IDX["CHIN"]][1])
    for y in (brow, nose):
        cv2.line(out, (x1i, y), (x2i, y), (0,255,0), 2)

    # ラベル
    tag = "白銀比 √2" if label=="SILVER" else "黄金比 φ"
    cv2.rectangle(out, (tx1, max(0,ty1-28)), (tx1+140, max(0,ty1-4)), color, -1)
    cv2.putText(out, tag, (tx1+6, max(0,ty1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

    return out

def build_table(metrics: Dict[str,float], target_ratio: float, target_name: str) -> pd.DataFrame:
    # 顔 H/W
    face_ar = metrics["face_AR"]
    err_ar = (face_ar - target_ratio) / target_ratio * 100 if np.isfinite(face_ar) else np.nan
    # 三分割（理想は各1/3）
    thirds = [metrics["thirds_top"], metrics["thirds_mid"], metrics["thirds_bot"]]
    thirds_err = [ (t - 1/3)*100 if np.isfinite(t) else np.nan for t in thirds ]

    rows = [
        {"項目":"顔の縦/横 (H/W)", "写真値":round(face_ar,3), "理想値":f"{target_name}={round(target_ratio,3)}", "差分%":round(err_ar,2)},
        {"項目":"上段（髪際→眉）/全高", "写真値":round(thirds[0],3), "理想値":"1/3", "差分%":round(thirds_err[0],2)},
        {"項目":"中段（眉→鼻）/全高",   "写真値":round(thirds[1],3), "理想値":"1/3", "差分%":round(thirds_err[1],2)},
        {"項目":"下段（鼻→顎）/全高",   "写真値":round(thirds[2],3), "理想値":"1/3", "差分%":round(thirds_err[2],2)},
        {"項目":"目間隔/片目幅",        "写真値":round(metrics["eye_spacing_ratio"],3), "理想値":"参考: ≈1.0", "差分%":None},
        {"項目":"鼻幅/口幅",            "写真値":round(metrics["nose_to_mouth"],3),   "理想値":"参考: 個人差", "差分%":None},
    ]
    return pd.DataFrame(rows)

# ---------------- UI ----------------
uploaded = st.file_uploader("画像をアップロード（JPG/PNG）", type=["jpg","jpeg","png"])
if not uploaded: st.stop()

img = Image.open(uploaded)
np_img = pil2np(img)

# 長辺制限
long_edge = max(np_img.shape[:2])
if long_edge > 1600:
    scale = 1600/long_edge
    np_img = cv2.resize(np_img, (int(np_img.shape[1]*scale), int(np_img.shape[0]*scale)))

# 検出
res = face_mesh.process(np_img)
if not res.multi_face_landmarks:
    st.error("顔を検出できませんでした（正面に近い・明るい画像でお試しください）。")
    st.stop()

landmarks = res.multi_face_landmarks[0].landmark
h, w, _ = np_img.shape
xy = landmarks_to_xy(landmarks, w, h)

# 回転補正＋トリミング
crop, xy_crop = align_and_crop(np_img, xy, margin=0.18)

# 指標
metrics = compute_metrics(xy_crop)

# 2種類のオーバーレイ画像を事前生成
img_golden = build_overlay(crop, xy_crop, PHI, "GOLDEN")
img_silver = build_overlay(crop, xy_crop, SILVER, "SILVER")

# 切替 UI
mode = st.segmented_control("比較対象", options=["黄金比", "白銀比"], default="黄金比")
if mode == "黄金比":
    target_ratio, target_name, show_img = PHI, "φ", img_golden
else:
    target_ratio, target_name, show_img = SILVER, "√2", img_silver

st.subheader("結果オーバーレイ（回転補正 & 顔トリミング済）")
st.image(Image.fromarray(show_img), use_container_width=True)

# テーブル
st.subheader("比率の比較表")
df = build_table(metrics, target_ratio, target_name)
st.dataframe(df, use_container_width=True)

st.caption("※ 三分割は等分(1/3)を理想として比較。黄金/白銀は全体の縦横比の理想枠として表示します。審美は個人差・文化差があります。")
