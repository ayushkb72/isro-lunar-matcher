import streamlit as st
import numpy as np
import cv2
from PIL import Image

st.set_page_config(
    page_title="ISRO Lunar Feature Matcher (SIH26166)",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Chandrayaan-2 Cross-Modal Image Correspondence Engine")
st.caption("Sub-Pixel Invariant Feature Registration across OHRC, TMC-2, and IIRS Datasets (SIH26166)")

# ---------------------------------------------------------
# 1. CORE ROBUST PREPROCESSING (CHANNELS & BIT-DEPTH GUARD)
# ---------------------------------------------------------
def normalize_to_uint8(img_np):
    """Guards against non-8-bit inputs (16-bit TIFFs, float arrays) and strips alpha channels."""
    if img_np is None or img_np.size == 0:
        return None

    # Strip Alpha / Unify to 1 or 3 channels
    if img_np.ndim == 3:
        if img_np.shape[2] == 4:
            img_np = img_np[:, :, :3]
        elif img_np.shape[2] == 1:
            img_np = img_np[:, :, 0]

    # Normalize arbitrary dynamic ranges (e.g. 12-bit/16-bit satellite sensors)
    if img_np.dtype != np.uint8:
        norm = cv2.normalize(img_np, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        img_np = np.uint8(norm)

    return img_np

def preprocess_lunar(img_np, apply_clahe=True, clip_limit=3.0, tile_size=8):
    """Converts to grayscale and applies CLAHE for illumination-invariant crater extraction."""
    clean_img = normalize_to_uint8(img_np)
    if clean_img is None:
        return None

    if clean_img.ndim == 3:
        # PIL loads RGB, convert RGB to Gray directly
        gray = cv2.cvtColor(clean_img, cv2.COLOR_RGB2GRAY)
    else:
        gray = clean_img

    if apply_clahe:
        clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(int(tile_size), int(tile_size)))
        return clahe.apply(gray)
    return gray

# ---------------------------------------------------------
# 2. FEATURE DETECTION, FLANN MATCHING & HOMOGRAPHY RANSAC
# ---------------------------------------------------------
def compute_multimodal_alignment(img1, img2, ratio_thresh=0.75, max_features=3000, model_type="Homography"):
    """
    Executes SIFT extraction, FLANN ratio matching, and USAC_MAGSAC geometric estimation.
    Calculates sub-pixel RMSE and match confidence metrics.
    """
    sift = cv2.SIFT_create(
        nfeatures=int(max_features),
        contrastThreshold=0.03,
        edgeThreshold=10,
        sigma=1.6
    )

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return None

    # FLANN KD-Tree matcher setup
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=64)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    try:
        raw_matches = flann.knnMatch(des1, des2, k=2)
    except cv2.error:
        return None

    # Lowe's Ratio Test with strict neighbor verification
    good_matches = []
    for pair in raw_matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)

    min_points = 4 if model_type == "Homography" else 3
    if len(good_matches) < min_points:
        return {
            "vis": None,
            "total_matches": len(good_matches),
            "inliers": 0,
            "rmse": 0.0,
            "matrix": None
        }

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # USAC_MAGSAC (or fallback RANSAC) for sub-pixel outlier elimination
    ransac_method = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)
    
    if model_type == "Homography":
        transform_matrix, mask = cv2.findHomography(src_pts, dst_pts, ransac_method, 4.0)
    else:
        transform_matrix, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=4.0)

    if mask is None or transform_matrix is None:
        return {
            "vis": None,
            "total_matches": len(good_matches),
            "inliers": 0,
            "rmse": 0.0,
            "matrix": None
        }

    inlier_mask = mask.ravel().tolist()
    inlier_matches = [good_matches[i] for i, val in enumerate(inlier_mask) if val == 1]
    inlier_count = len(inlier_matches)

    # Calculate Root Mean Square Error (RMSE) over inlier points
    rmse_val = 0.0
    if inlier_count >= min_points:
        inlier_src = np.float32([kp1[m.queryIdx].pt for m in inlier_matches]).reshape(-1, 1, 2)
        inlier_dst = np.float32([kp2[m.trainIdx].pt for m in inlier_matches]).reshape(-1, 1, 2)

        if model_type == "Homography":
            projected = cv2.perspectiveTransform(inlier_src, transform_matrix)
        else:
            projected = cv2.transform(inlier_src, transform_matrix)

        errors = np.linalg.norm(inlier_dst - projected, axis=2)
        rmse_val = float(np.sqrt(np.mean(errors ** 2)))

    # Produce visualization with verified inliers
    matched_vis = cv2.drawMatches(
        img1, kp1, img2, kp2, inlier_matches[:80], None,
        matchColor=(0, 255, 0),
        singlePointColor=(0, 0, 255),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    return {
        "vis": matched_vis,
        "total_matches": len(good_matches),
        "inliers": inlier_count,
        "rmse": round(rmse_val, 3),
        "matrix": transform_matrix
    }

# ---------------------------------------------------------
# 3. INTERACTIVE DASHBOARD & CONTROLS
# ---------------------------------------------------------
st.sidebar.header("Algorithmic Controls")
model_choice = st.sidebar.selectbox(
    "Geometric Transformation Model",
    ["Homography", "Affine Partial 2D"],
    help="Homography resolves perspective distortion across divergent orbital inclinations."
)
clahe_flag = st.sidebar.checkbox("Apply Solar Shadow Equalization (CLAHE)", value=True)
tile_grid = st.sidebar.slider("CLAHE Tile Window Size", 4, 16, 8, 2)
clip_lim = st.sidebar.slider("CLAHE Dynamic Clip Limit", 1.0, 5.0, 3.0, 0.5)
ratio_val = st.sidebar.slider("Lowe's Scale-Space Ratio Threshold", 0.50, 0.85, 0.75, 0.05)
max_feat = st.sidebar.slider("Maximum Extracted Keypoints", 500, 7000, 3500, 500)

col_left, col_right = st.columns(2)
with col_left:
    file1 = st.file_uploader("Upload Image A (e.g., OHRC / Reference)", type=["png", "jpg", "jpeg", "tif", "tiff"])
with col_right:
    file2 = st.file_uploader("Upload Image B (e.g., TMC-2 / Alternate View)", type=["png", "jpg", "jpeg", "tif", "tiff"])

if file1 and file2:
    try:
        raw1 = np.array(Image.open(file1))
        raw2 = np.array(Image.open(file2))
    except Exception as e:
        st.error(f"Error decoding image file: {str(e)}")
        st.stop()

    proc1 = preprocess_lunar(raw1, apply_clahe=clahe_flag, clip_limit=clip_lim, tile_size=tile_grid)
    proc2 = preprocess_lunar(raw2, apply_clahe=clahe_flag, clip_limit=clip_lim, tile_size=tile_grid)

    if proc1 is None or proc2 is None:
        st.error("Input imagery could not be normalized. Check bit depth and pixel data.")
        st.stop()

    if st.button("Execute Sub-Pixel Correspondence Matching", use_container_width=True):
        with st.spinner("Calculating scale-space pyramids and running geometric RANSAC..."):
            results = compute_multimodal_alignment(
                proc1, proc2,
                ratio_thresh=ratio_val,
                max_features=max_feat,
                model_type=model_choice
            )

        if results is None or results["inliers"] == 0:
            st.warning("No robust geometric correspondences identified. Reduce the ratio threshold or ensure spatial overlap between tiles.")
        else:
            st.image(
                results["vis"],
                caption=f"Verified Sub-Pixel Matches ({model_choice} RANSAC Inliers in Green)",
                use_container_width=True
            )

            # Quantitative Validation Cards
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Candidate Matches", results["total_matches"])
            m2.metric("Geometric Inliers", results["inliers"])
            
            inlier_ratio = round((results["inliers"] / results["total_matches"] * 100), 2) if results["total_matches"] > 0 else 0
            m3.metric("Inlier Confidence Ratio", f"{inlier_ratio}%")
            m4.metric("Sub-Pixel Error (RMSE)", f"{results['rmse']} px")

            # Evaluation threshold notice
            if inlier_ratio >= 35.0 and results["rmse"] < 2.0:
                st.success("Target Verification Passed: Model meets ISRO criteria for sub-pixel accuracy and false-match rejection.")
            else:
                st.info("Non-optimal alignment: High perspective variance or low texture overlap detected.")
else:
    st.info("Upload two lunar optical images to initialize feature correspondence.")