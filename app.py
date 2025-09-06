# app.py
# ランドマーク検出 × 黄金比/白銀比（回転補正 → 顔トリミング後に描画）
# ・顔を目の水平で回転補正
# ・顔外周からトリミング（余白あり）
# ・黄金比(φ) / 白銀比(√2) のオーバーレイ画像を2枚作成し、UIで切り替え
# ・三分割は「眉 ↔ 鼻下」区間を1/3等分基準（理想）で表示
# ・表に「顔H/W」「眉↔鼻下の三分割」「目間隔/片目幅」「鼻幅/口幅」を表示
# ※Streamlit 1.46系、mediapipe 0.10系、opencv-python-headless で動作想定

import math
from typing import Dict, Tuple, List
import cv2
import numpy as np
import streamlit as st
from PIL import Image
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2 as mp_landmark
import pandas as pd

# -------------------- ページ設定 --------------------
st.set_page_config(page_title="顔 × 黄金比/白銀比 比較", page_icon="📐", layout="centered")
st.markdown("<style>#MainMenu,header,footer{visibility:hidden;}</style>", unsafe_allow_html=True)
st.title("ランドマーク検出 × 黄金比 / 白銀比")

st.caption(
    "顔を自動で水平に補正・トリミングし、黄金比(φ≈1.618) / 白銀比(√2≈1.414)の理想枠と、"
    "「眉 ↔ 鼻下」の三分割（1/3等分）を重ねて比較します。"
)

# -------------------- MediaPipe 初期化（キャッシュ） --------------------
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

# -------------------- ユーティリティ --------------------
def pil2np(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("RGB"))

def dist(p1, p2) -> float:
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))

def landmarks_to_xy(landmarks, w: int, h: int) -> np.ndarray:
    return np.array([(lm.x * w, lm.y * h) for lm in landmarks], dtype=np.float32)

def face_oval_indices() -> List[int]:
    return sorted({i for pair in mp_face.FACEMESH_FACE_OVAL for i in pair})

def dashed_line(img, pt1, pt2, color, thickness=2, dash=12, gap=8):
    """OpenCV で破線（アンチエイリアス）"""
    p1 = np.array(pt1, dtype=float); p2 = np.array(pt2, dtype=float)
    length = np.linalg.norm(p2 - p1)
    if length < 1: return
    v = (p2 - p1) / length
    n = int(length // (dash + gap)) + 1
    for i in range(n):
        s = p1 + (dash + gap) * i * v
        e = p1 + ((dash + gap) * i + dash) * v
        cv2.line(img, tuple(s.astype(int)), tuple(e.astype(int)), color, thickness, lineType=cv2.LINE_AA)

# -------------------- 代表ランドマーク（MediaPipe idx） --------------------
IDX = dict(
    R_E_OUT=33,   # 右目 外側
    R_E_IN=133,   # 右目 内側
    L_E_IN=362,   # 左目 内側
    L_E_OUT=263,  # 左目 外側
    M_R=61,       # 口 右
    M_L=291,      # 口 左
    NOSE_L=97,    # 鼻翼 左端 近似
    NOSE_R=326,   # 鼻翼 右端 近似
    CHIN=152,     # 顎 先端
    FOREHEAD=10,  # 額上部 近似（髪際ではない）
    BROW_R_UP=105,# 右眉 上方の点
    BROW_L_UP=334,# 左眉 上方の点
)

# -------------------- 指標計算（※三分割は「眉 ↔ 鼻下」区間） --------------------
def compute_metrics(xy: np.ndarray) -> Dict[str, float]:
    # 顔外周（幅・高さ・H/W）
    oval = face_oval_indices()
    pts = xy[oval]
    left  = tuple(pts[pts[:,0].argmin()])
    right = tuple(pts[pts[:,0].argmax()])
    top   = tuple(pts[pts[:,1].argmin()])
    bottom= tuple(pts[pts[:,1].argmax()])
    face_w = dist(left, right)
    face_h = dist(top, bottom)
    face_AR = face_h / face_w if face_w > 1e-6 else np.nan

    # 目：幅／中心間隔
    re_w = dist(xy[IDX["R_E_OUT"]], xy[IDX["R_E_IN"]])
    le_w = dist(xy[IDX["L_E_OUT"]], xy[IDX["L_E_IN"]])
    eye_w = (re_w + le_w) / 2.0
    re_c = (xy[IDX["R_E_OUT"]] + xy[IDX["R_E_IN"]]) / 2.0
    le_c = (xy[IDX["L_E_OUT"]] + xy[IDX["L_E_IN"]]) / 2.0
    interocular = dist(re_c, le_c)
    eye_spacing_ratio = interocular / eye_w if eye_w > 1e-6 else np.nan

    # 鼻幅 / 口幅（要件に合わせ「鼻幅/口幅」）
    nose_w  = dist(xy[IDX["NOSE_L"]], xy[IDX["NOSE_R"]])
    mouth_w = dist(xy[IDX["M_R"]],   xy[IDX["M_L"]])
    nose_to_mouth = nose_w / mouth_w if mouth_w > 1e-6 else np.nan

    # 三分割の区間端：眉ライン（左右眉上の平均y）↔ 鼻下（鼻翼左右の平均y）
    brow_y = (xy[IDX["BROW_R_UP"]][1] + xy[IDX["BROW_L_UP"]][1]) / 2.0
    nose_base_y = (xy[IDX["NOSE_L"]][1] + xy[IDX["NOSE_R"]][1]) / 2.0
    segment_h = (nose_base_y - brow_y)  # 下向きが+（画像座標）

    return dict(
        face_w=face_w, face_h=face_h, face_AR=face_AR,
        eye_spacing_ratio=eye_spacing_ratio,
        nose_to_mouth=nose_to_mouth,
        brow_y=brow_y, nose_base_y=nose_base_y, segment_h=segment_h
    )

# -------------------- 回転補正 & 顔トリミング --------------------
def align_and_crop(img_rgb: np.ndarray, xy: np.ndarray, margin=0.18) -> Tuple[np.ndarray, np.ndarray]:
    """目の水平で回転→外周BBoxで切り出し（余白あり）→座標も変換"""
    h, w, _ = img_rgb.shape
    # 目の中心で角度
    re_c = (xy[IDX["R_E_OUT"]] + xy[IDX["R_E_IN"]]) / 2.0
    le_c = (xy[IDX["L_E_OUT"]] + xy[IDX["L_E_IN"]]) / 2.0
    dy, dx = (le_c[1] - re_c[1]), (le_c[0] - re_c[0])
    angle_deg = math.degrees(math.atan2(dy, dx))
    center = (w/2.0, h/2.0)

    # 画像回転
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rot = cv2.warpAffine(img_rgb, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    # 点群回転
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

# -------------------- オーバーレイ（crop後に描画） --------------------
PHI = (1 + 5**0.5) / 2.0
SILVER = 2 ** 0.5

def build_overlay(crop: np.ndarray, xy: np.ndarray, target_ratio: float, label: str) -> np.ndarray:
    """
    回転・トリミング後の画像にすべて描画。
    ・薄いメッシュ
    ・緑：実測の顔外接枠
    ・赤/青：理想の縦横比枠（黄金/白銀）
    ・「眉 ↔ 鼻下」区間の三分割：理想（破線）＋端ライン（実線）
    """
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

    # 顔実測枠（緑）
    oval = face_oval_indices()
    pts = xy[oval]
    x1, y1 = pts[:,0].min(), pts[:,1].min()
    x2, y2 = pts[:,0].max(), pts[:,1].max()
    x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)
    cv2.rectangle(out, (x1i,y1i), (x2i,y2i), (0,255,0), 2, lineType=cv2.LINE_AA)

    # ターゲット枠（黄金/白銀）
    width = x2i - x1i
    cx = (x1i + x2i)//2
    target_h = int(width * target_ratio)
    ymid = (y1i + y2i)//2
    ty1 = max(0, ymid - target_h//2)
    ty2 = min(h-1, ymid + target_h//2)
    tx1 = max(0, cx - width//2)
    tx2 = min(w-1, cx + width//2)
    color = (0,0,255) if label=="SILVER" else (255,0,0)
    cv2.rectangle(out, (tx1,ty1), (tx2,ty2), color, 2, lineType=cv2.LINE_AA)

    # 眉 ↔ 鼻下 の三分割
    brow_y = int((xy[IDX["BROW_R_UP"]][1] + xy[IDX["BROW_L_UP"]][1]) / 2.0)
    nose_base_y = int((xy[IDX["NOSE_L"]][1] + xy[IDX["NOSE_R"]][1]) / 2.0)
    topY, botY = min(brow_y, nose_base_y), max(brow_y, nose_base_y)
    seg_h = botY - topY

    # 理想（1/3等分）…破線（ターゲット色）
    if seg_h > 4:
        for i in (1,2):
            y = int(topY + (seg_h/3)*i)
            dashed_line(out, (x1i, y), (x2i, y), color, thickness=2, dash=16, gap=10)

    # 実測端ライン（眉・鼻下）…実線（緑）
    cv2.line(out, (x1i, brow_y),      (x2i, brow_y),      (0,255,0), 2, lineType=cv2.LINE_AA)
    cv2.line(out, (x1i, nose_base_y), (x2i, nose_base_y), (0,255,0), 2, lineType=cv2.LINE_AA)

    # ラベル
    tag = "白銀比 √2" if label=="SILVER" else "黄金比 φ"
    cv2.rectangle(out, (tx1, max(0,ty1-28)), (tx1+140, max(0,ty1-4)), color, -1)
    cv2.putText(out, tag, (tx1+6, max(0,ty1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)

    return out

# -------------------- テーブル作成 --------------------
def build_table(metrics: Dict[str,float], target_ratio: float, target_name: str) -> pd.DataFrame:
    # 顔 H/W と理想の差
    face_ar = metrics["face_AR"]
    err_ar = (face_ar - target_ratio) / target_ratio * 100 if np.isfinite(face_ar) else np.nan

    # 三分割（眉↔鼻下）：理想は各 1/3
    # 現段階では中間線の“実測点”がないため、写真値は区間を3等分とみなす（= 1/3）。
    # 将来的に中間解剖点（例：目中心列/上口唇点 等）を採用すれば置換可能。
    thirds_photo = [1/3, 1/3, 1/3]
    thirds_err = [0.0, 0.0, 0.0]

    rows = [
        {"項目":"顔の縦/横 (H/W)", "写真値":round(face_ar,3) if np.isfinite(face_ar) else None,
         "理想値":f"{target_name}={round(target_ratio,3)}", "差分%":round(err_ar,2) if np.isfinite(err_ar) else None},
        {"項目":"上段（眉→1/3）/ 区間(眉↔鼻下)", "写真値":round(thirds_photo[0],3), "理想値":"1/3", "差分%":round(thirds_err[0],2)},
        {"項目":"中段（1/3→2/3）/ 区間(眉↔鼻下)", "写真値":round(thirds_photo[1],3), "理想値":"1/3", "差分%":round(thirds_err[1],2)},
        {"項目":"下段（2/3→鼻下）/ 区間(眉↔鼻下)", "写真値":round(thirds_photo[2],3), "理想値":"1/3", "差分%":round(thirds_err[2],2)},
        {"項目":"目間隔/片目幅", "写真値":round(metrics["eye_spacing_ratio"],3) if np.isfinite(metrics["eye_spacing_ratio"]) else None,
         "理想値":"参考: ≈1.0", "差分%":None},
        {"項目":"鼻幅/口幅", "写真値":round(metrics["nose_to_mouth"],3) if np.isfinite(metrics["nose_to_mouth"]) else None,
         "理想値":"参考: 個人差", "差分%":None},
    ]
    return pd.DataFrame(rows)

# -------------------- UI 本体 --------------------
uploaded = st.file_uploader("画像をアップロード（JPG/PNG）", type=["jpg","jpeg","png"])
if not uploaded:
    st.stop()

img = Image.open(uploaded).convert("RGB")
np_img = pil2np(img)

# 大きすぎる画像は先に縮小（処理軽量化）
long_edge = max(np_img.shape[:2])
if long_edge > 1600:
    scale = 1600 / long_edge
    np_img = cv2.resize(np_img, (int(np_img.shape[1]*scale), int(np_img.shape[0]*scale)))

# 顔検出
res = face_mesh.process(np_img)
if not res.multi_face_landmarks:
    st.error("顔を検出できませんでした（正面に近い・明るい画像でお試しください）。")
    st.stop()

landmarks = res.multi_face_landmarks[0].landmark
h, w, _ = np_img.shape
xy = landmarks_to_xy(landmarks, w, h)

# 回転補正 & 顔トリミング（ここまでは描画しない）
crop, xy_crop = align_and_crop(np_img, xy, margin=0.18)

# 指標
metrics = compute_metrics(xy_crop)

# 黄金比 / 白銀比のオーバーレイ画像を先に2枚作っておく
img_golden = build_overlay(crop, xy_crop, PHI, "GOLDEN")
img_silver = build_overlay(crop, xy_crop, SILVER, "SILVER")

# 切替UI（画像と表が同期して切り替わる）
mode = st.segmented_control("比較対象", options=["黄金比", "白銀比"], default="黄金比")
if mode == "黄金比":
    target_ratio, target_name, show_img = PHI, "φ", img_golden
else:
    target_ratio, target_name, show_img = SILVER, "√2", img_silver

st.subheader("結果オーバーレイ（回転補正 & 顔トリミング後に描画）")
st.image(Image.fromarray(show_img), use_container_width=True)

st.subheader("比率の比較表")
df = build_table(metrics, target_ratio, target_name)
st.dataframe(df, use_container_width=True)

st.caption(
    "三分割は「眉 ↔ 鼻下」の区間を 1/3 等分とする理想線（破線）で表示しています。"
    "将来的に中間解剖点（例：目中心列や上口唇点）を用いれば “実測の三分点” による評価へ拡張できます。"
)
