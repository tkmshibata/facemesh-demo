import math
from typing import Dict, Tuple, List

import cv2
import numpy as np
import streamlit as st
from PIL import Image
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2 as mp_landmark
import pandas as pd

# ================= ページ設定 & UI最小化 =================
st.set_page_config(page_title="顔 × 黄金比/白銀比（三分割）", page_icon="📐", layout="centered")
st.markdown("<style>#MainMenu,header,footer{visibility:hidden;}</style>", unsafe_allow_html=True)
st.title("ランドマーク検出 × 黄金比 / 白銀比（眉↔鼻下を基準に三分割）")

# ================= MediaPipe（キャッシュ） =================
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

# ================= 定数・ユーティリティ =================
PHI = (1 + 5 ** 0.5) / 2.0      # 1.618...
SILVER = 2 ** 0.5               # 1.414...

# よく使うランドマーク index
IDX = dict(
    R_E_OUT=33, R_E_IN=133, L_E_IN=362, L_E_OUT=263,  # 目
    M_R=61, M_L=291,                                  # 口角
    NOSE_L=97, NOSE_R=326,                            # 鼻翼端（鼻下近似に利用）
    CHIN=152,                                         # 顎先
    BROW_R_UP=105, BROW_L_UP=334                      # 眉上の代表点
)

def pil2np(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("RGB"))

def dist(p1, p2) -> float:
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))

def landmarks_to_xy(landmarks, w: int, h: int) -> np.ndarray:
    return np.array([(lm.x * w, lm.y * h) for lm in landmarks], dtype=np.float32)

def face_oval_indices() -> List[int]:
    return sorted({i for pair in mp_face.FACEMESH_FACE_OVAL for i in pair})

def dashed_line(img, pt1, pt2, color, thickness=2, dash=12, gap=8):
    """OpenCV で破線（アンチエイリアス付き）"""
    p1 = np.array(pt1, dtype=float); p2 = np.array(pt2, dtype=float)
    length = np.linalg.norm(p2 - p1)
    if length < 1: return
    v = (p2 - p1) / length
    n = int(length // (dash + gap)) + 1
    for i in range(n):
        s = p1 + (dash + gap) * i * v
        e = p1 + ((dash + gap) * i + dash) * v
        cv2.line(img, tuple(s.astype(int)), tuple(e.astype(int)), color, thickness, lineType=cv2.LINE_AA)

# ================= 整列（目の水平化）& 切り出し =================
def align_rotate(img_rgb: np.ndarray, xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """目の中心線が水平になるよう回転。画像と点群を回転。"""
    h, w, _ = img_rgb.shape
    re_c = (xy[IDX["R_E_OUT"]] + xy[IDX["R_E_IN"]]) / 2.0
    le_c = (xy[IDX["L_E_OUT"]] + xy[IDX["L_E_IN"]]) / 2.0
    dy, dx = (le_c[1] - re_c[1]), (le_c[0] - re_c[0])
    angle = math.degrees(math.atan2(dy, dx))
    center = (w/2.0, h/2.0)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    rot = cv2.warpAffine(img_rgb, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    ones = np.ones((xy.shape[0], 1))
    xy_h = np.hstack([xy, ones])
    xy_rot = (M @ xy_h.T).T
    return rot, xy_rot

def compute_keylines_from_rotated(xy_rot: np.ndarray) -> Dict[str, float]:
    """回転後の座標から、三分割用キーポイント y を算出。"""
    brow_y = (xy_rot[IDX["BROW_R_UP"]][1] + xy_rot[IDX["BROW_L_UP"]][1]) / 2.0
    nose_base_y = (xy_rot[IDX["NOSE_L"]][1] + xy_rot[IDX["NOSE_R"]][1]) / 2.0
    chin_y = xy_rot[IDX["CHIN"]][1]
    oval = face_oval_indices()
    hairline_y = float(np.min(xy_rot[oval][:, 1]))  # 生え際近似：外輪郭の最上
    return dict(brow_y=brow_y, nose_base_y=nose_base_y, chin_y=chin_y, hairline_y=hairline_y)

def compute_crop_box(xy_rot: np.ndarray, img_shape, key: Dict[str, float], extra_margin=0.18) -> Tuple[int,int,int,int]:
    """顔外輪郭と“理想線（生え際=眉-1, 顎=鼻下+1）”が入るようにトリミング範囲を決定。"""
    h, w = img_shape[:2]
    oval = face_oval_indices()
    pts = xy_rot[oval]
    x1, y1 = float(np.min(pts[:,0])), float(np.min(pts[:,1]))
    x2, y2 = float(np.max(pts[:,0])), float(np.max(pts[:,1]))

    base = key["nose_base_y"] - key["brow_y"]      # 眉→鼻下（基準=1）
    ideal_top = key["brow_y"] - base               # 理想の生え際ライン
    ideal_bottom = key["nose_base_y"] + base       # 理想の顎先ライン

    y1 = min(y1, ideal_top)
    y2 = max(y2, ideal_bottom)

    bw, bh = (x2 - x1), (y2 - y1)
    x1 -= bw * extra_margin; x2 += bw * extra_margin
    y1 -= bh * extra_margin; y2 += bh * extra_margin

    x1i = int(max(0, round(x1))); y1i = int(max(0, round(y1)))
    x2i = int(min(w-1, round(x2))); y2i = int(min(h-1, round(y2)))
    return x1i, y1i, x2i, y2i

def crop_with_box(img_rot: np.ndarray, xy_rot: np.ndarray, box: Tuple[int,int,int,int]) -> Tuple[np.ndarray, np.ndarray]:
    x1,y1,x2,y2 = box
    crop = img_rot[y1:y2, x1:x2].copy()
    xy_crop = xy_rot.copy()
    xy_crop[:,0] -= x1; xy_crop[:,1] -= y1
    return crop, xy_crop

# ================= 指標（切り出し後に計算） =================
def compute_metrics_on_crop(xy: np.ndarray) -> Dict[str, float]:
    oval = face_oval_indices()
    pts = xy[oval]
    left = tuple(pts[pts[:,0].argmin()])
    right = tuple(pts[pts[:,0].argmax()])
    top = tuple(pts[pts[:,1].argmin()])
    bottom = tuple(pts[pts[:,1].argmax()])
    face_w = dist(left, right); face_h = dist(top, bottom)
    face_AR = face_h / face_w if face_w > 1e-6 else np.nan

    brow_y = (xy[IDX["BROW_R_UP"]][1] + xy[IDX["BROW_L_UP"]][1]) / 2.0
    nose_base_y = (xy[IDX["NOSE_L"]][1] + xy[IDX["NOSE_R"]][1]) / 2.0
    chin_y = xy[IDX["CHIN"]][1]
    hairline_y = float(np.min(pts[:,1]))

    L_top = brow_y - hairline_y
    L_mid = nose_base_y - brow_y
    L_bot = chin_y - nose_base_y

    re_w = dist(xy[IDX["R_E_OUT"]], xy[IDX["R_E_IN"]])
    le_w = dist(xy[IDX["L_E_OUT"]], xy[IDX["L_E_IN"]])
    eye_w = (re_w + le_w) / 2.0
    re_c = (xy[IDX["R_E_OUT"]] + xy[IDX["R_E_IN"]]) / 2.0
    le_c = (xy[IDX["L_E_OUT"]] + xy[IDX["L_E_IN"]]) / 2.0
    interocular = dist(re_c, le_c)
    eye_spacing_ratio = interocular / eye_w if eye_w>1e-6 else np.nan  # 理想=1

    nose_w = dist(xy[IDX["NOSE_L"]], xy[IDX["NOSE_R"]])
    mouth_w = dist(xy[IDX["M_R"]], xy[IDX["M_L"]])
    nose_to_mouth = nose_w / mouth_w if mouth_w>1e-6 else np.nan

    return dict(
        face_w=face_w, face_h=face_h, face_AR=face_AR,
        hairline_y=hairline_y, brow_y=brow_y, nose_base_y=nose_base_y, chin_y=chin_y,
        L_top=L_top, L_mid=L_mid, L_bot=L_bot,
        eye_spacing_ratio=eye_spacing_ratio, nose_to_mouth=nose_to_mouth
    )

# ================= オーバーレイ（描画はすべてクロップ後） =================
def build_overlay(crop: np.ndarray, xy: np.ndarray, target_ratio: float, label: str) -> np.ndarray:
    """黄金/白銀の枠 + 三分割（理想1:1:1 & 実測端）を描画。すべて AA で描く。"""
    out = crop.copy()
    h, w, _ = out.shape

    # メッシュ（薄く）
    nl = mp_landmark.NormalizedLandmarkList(
        landmark=[mp_landmark.NormalizedLandmark(x=float(x)/w, y=float(y)/h) for (x,y) in xy]
    )
    mp_draw.draw_landmarks(
        out, nl, mp_face.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_draw.DrawingSpec(color=(210,210,210), thickness=1, circle_radius=0)
    )

    # 顔外接枠（緑）
    oval = face_oval_indices()
    pts = xy[oval]
    x1, y1 = int(np.min(pts[:,0])), int(np.min(pts[:,1]))
    x2, y2 = int(np.max(pts[:,0])), int(np.max(pts[:,1]))
    cv2.rectangle(out, (x1,y1), (x2,y2), (0,255,0), 2, lineType=cv2.LINE_AA)

    # 黄金/白銀の枠（中央合わせ）：幅×比率
    width = x2 - x1
    cx = (x1 + x2)//2
    t_h = int(width * target_ratio)
    ymid = (y1 + y2)//2
    ty1 = max(0, ymid - t_h//2)
    ty2 = min(h-1, ymid + t_h//2)
    tx1 = max(0, cx - width//2)
    tx2 = min(w-1, cx + width//2)
    col = (0,0,255) if label=="SILVER" else (255,0,0)
    cv2.rectangle(out, (tx1,ty1), (tx2,ty2), col, 2, lineType=cv2.LINE_AA)

    # —— 三分割：眉↔鼻下 を 1 として 1:1:1 の理想線 + 実測端 —— #
    brow_y = int((xy[IDX["BROW_R_UP"]][1] + xy[IDX["BROW_L_UP"]][1]) / 2.0)
    nose_y = int((xy[IDX["NOSE_L"]][1] + xy[IDX["NOSE_R"]][1]) / 2.0)
    base = max(1, nose_y - brow_y)

    ideal_hair = int(brow_y - base)  # 生え際 = 眉 - 1
    ideal_chin = int(nose_y + base)  # 顎先 = 鼻下 + 1

    # 理想（破線：ターゲット色）
    dashed_line(out, (x1, ideal_hair), (x2, ideal_hair), col, thickness=3, dash=18, gap=12)
    dashed_line(out, (x1, ideal_chin), (x2, ideal_chin), col, thickness=3, dash=18, gap=12)

    # 実測端：生え際（外輪郭最上）と顎先（緑の実線）
    hairline = int(np.min(pts[:,1]))
    chin = int(xy[IDX["CHIN"]][1])
    cv2.line(out, (x1, hairline), (x2, hairline), (0,255,0), 3, lineType=cv2.LINE_AA)
    cv2.line(out, (x1, chin), (x2, chin), (0,255,0), 3, lineType=cv2.LINE_AA)

    # 眉と鼻下（緑の実線）
    cv2.line(out, (x1, brow_y), (x2, brow_y), (0,255,0), 3, lineType=cv2.LINE_AA)
    cv2.line(out, (x1, nose_y), (x2, nose_y), (0,255,0), 3, lineType=cv2.LINE_AA)

    # ラベル
    tag = "白銀比 √2" if label=="SILVER" else "黄金比 φ"
    cv2.rectangle(out, (tx1, max(0,ty1-30)), (tx1+150, max(0,ty1-6)), col, -1)
    cv2.putText(out, tag, (tx1+6, max(0,ty1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)

    return out

# ================= テーブル（比率） =================
def build_table(metrics: Dict[str,float], target_ratio: float, target_name: str) -> pd.DataFrame:
    # 顔 H/W と φ/√2 の誤差
    face_ar = metrics["face_AR"]
    err_ar = (face_ar - target_ratio) / target_ratio * 100 if np.isfinite(face_ar) else np.nan

    # 三分割（生え際→眉 / 眉→鼻下 / 鼻下→顎先）を (眉→鼻下) で正規化
    L_top, L_mid, L_bot = metrics["L_top"], metrics["L_mid"], metrics["L_bot"]
    if L_mid and L_mid > 1e-6:
        r_top = float(L_top / L_mid)
        r_mid = 1.0
        r_bot = float(L_bot / L_mid)
        e_top = (r_top - 1.0) * 100
        e_mid = 0.0
        e_bot = (r_bot - 1.0) * 100
    else:
        r_top = r_mid = r_bot = e_top = e_mid = e_bot = np.nan

    # 目間隔/片目幅（理想=1）
    eye_ratio = metrics["eye_spacing_ratio"]
    eye_err = (eye_ratio - 1.0) * 100 if np.isfinite(eye_ratio) else np.nan

    # 鼻幅/口幅（参考）
    nose_to_mouth = metrics["nose_to_mouth"]

    rows = [
        {"項目":"顔の縦/横 (H/W)", "写真値":round(face_ar,3) if np.isfinite(face_ar) else None,
         "理想値":f"{target_name}={round(target_ratio,3)}", "差分%":round(err_ar,2) if np.isfinite(err_ar) else None},
        {"項目":"(生え際→眉) / (眉→鼻下)", "写真値":round(r_top,3) if np.isfinite(r_top) else None,
         "理想値":"1", "差分%":round(e_top,2) if np.isfinite(e_top) else None},
        {"項目":"(眉→鼻下) / (眉→鼻下)", "写真値":round(r_mid,3) if np.isfinite(r_mid) else None,
         "理想値":"1", "差分%":round(e_mid,2) if np.isfinite(e_mid) else None},
        {"項目":"(鼻下→顎先) / (眉→鼻下)", "写真値":round(r_bot,3) if np.isfinite(r_bot) else None,
         "理想値":"1", "差分%":round(e_bot,2) if np.isfinite(e_bot) else None},
        {"項目":"目間隔/片目幅", "写真値":round(eye_ratio,3) if np.isfinite(eye_ratio) else None,
         "理想値":"1", "差分%":round(eye_err,2) if np.isfinite(eye_err) else None},
        {"項目":"鼻幅/口幅", "写真値":round(nose_to_mouth,3) if np.isfinite(nose_to_mouth) else None,
         "理想値":"参考", "差分%":None},
    ]
    return pd.DataFrame(rows)

# ================= アプリ本体 =================
uploaded = st.file_uploader("画像をアップロード（JPG/PNG）", type=["jpg","jpeg","png"])
if not uploaded:
    st.stop()

img = Image.open(uploaded).convert("RGB")
np_img = pil2np(img)

# 処理負荷と見た目のバランスで、元画像が大きすぎる場合は先に縮小
long_edge = max(np_img.shape[:2])
if long_edge > 1800:
    scale0 = 1800 / long_edge
    np_img = cv2.resize(np_img, (int(np_img.shape[1]*scale0), int(np_img.shape[0]*scale0)), interpolation=cv2.INTER_AREA)

# FaceMesh
res = face_mesh.process(np_img)
if not res.multi_face_landmarks:
    st.error("顔を検出できませんでした。正面に近い・明るい画像でお試しください。")
    st.stop()

landmarks = res.multi_face_landmarks[0].landmark
h, w, _ = np_img.shape
xy = landmarks_to_xy(landmarks, w, h)

# 1) 回転
rot_img, xy_rot = align_rotate(np_img, xy)

# 2) キーポイント算出（回転後）
keys = compute_keylines_from_rotated(xy_rot)

# 3) キーを含むようにトリミング → 切り出し
box = compute_crop_box(xy_rot, rot_img.shape, keys, extra_margin=0.18)
crop, xy_crop = crop_with_box(rot_img, xy_rot, box)

# 4) 表示用に先に拡大（拡大後に描画→等倍表示でギザギザ防止）
TARGET_DISPLAY_WIDTH = 900  # 仕上がりの横幅（好みで）
scale_disp = TARGET_DISPLAY_WIDTH / crop.shape[1]
if scale_disp < 1.0:
    # 縮小は線が細くなるので避け、等倍で描画する
    scale_disp = 1.0

disp_img = cv2.resize(
    crop,
    (int(crop.shape[1] * scale_disp), int(crop.shape[0] * scale_disp)),
    interpolation=cv2.INTER_CUBIC
)
disp_xy = xy_crop * scale_disp

# 5) 指標計算（切り出し後の座標で）
metrics = compute_metrics_on_crop(disp_xy)

# 6) オーバーレイ画像（黄金/白銀）を事前生成
img_golden = build_overlay(disp_img, disp_xy, PHI, "GOLDEN")
img_silver = build_overlay(disp_img, disp_xy, SILVER, "SILVER")

# 7) 切替 UI（画像と表が同期）
mode = st.segmented_control("比較対象（画像・表ともに切替）", options=["黄金比", "白銀比"], default="黄金比")
if mode == "黄金比":
    target_ratio, target_name, show_img = PHI, "φ", img_golden
else:
    target_ratio, target_name, show_img = SILVER, "√2", img_silver

st.subheader("結果オーバーレイ（回転補正 & 切り出し後 / 拡大後に描画）")
# 等倍で表示（ブラウザ側の拡縮を避ける）
st.image(Image.fromarray(show_img), use_container_width=False, width=show_img.shape[1])

st.subheader("比率の比較表")
df = build_table(metrics, target_ratio, target_name)
st.dataframe(df, use_container_width=True)

st.caption(
    "三分割は『生え際→眉』『眉→鼻下』『鼻下→顎先』が 1:1:1（眉↔鼻下を基準1）。"
    "黄金比(1:1.618)・白銀比(1:1.4)は全体H/Wの参照比。"
    "描画はトリミング後・拡大後に行い、アンチエイリアスで線のギザつきを抑制しています。"
)
