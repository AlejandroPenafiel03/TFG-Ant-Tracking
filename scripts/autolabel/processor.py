# processor.py
import cv2
import numpy as np

###############################################
# 1) BG SUBTRACTION (MOG2)
###############################################

# Creamos el sustractor una sola vez (persistente)
_bgsub = cv2.createBackgroundSubtractorMOG2(
    history=500,
    varThreshold=25,
    detectShadows=False
)

def process_frame_bgsub(frame: np.ndarray) -> np.ndarray:
    """
    Devuelve una máscara usando sustracción de fondo (MOG2).
    """
    mask = _bgsub.apply(frame)
    return mask


###############################################
# 2) FLUJO ÓPTICO DENSO (Farnebäck)
###############################################

_prev_gray_dense = None

def process_frame_dense_flow(frame: np.ndarray) -> np.ndarray:
    """
    Calcula flujo óptico denso y devuelve una máscara basada en magnitud.
    """
    global _prev_gray_dense

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if _prev_gray_dense is None:
        _prev_gray_dense = gray
        return np.zeros_like(gray, dtype=np.uint8)

    flow = cv2.calcOpticalFlowFarneback(
        _prev_gray_dense, gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    mask = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    _prev_gray_dense = gray
    return mask


###############################################
# 3) FLUJO ÓPTICO DISCRETO (Lucas–Kanade)
###############################################

_prev_gray_discrete = None
_prev_pts = None

lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)

def process_frame_discrete_flow(frame: np.ndarray) -> np.ndarray:
    """
    Flujo óptico discreto (Lucas–Kanade) con puntos detectados por Shi–Tomasi.
    Devuelve una máscara basada en el movimiento de los puntos.
    """
    global _prev_gray_discrete, _prev_pts

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if _prev_gray_discrete is None:
        _prev_gray_discrete = gray
        _prev_pts = cv2.goodFeaturesToTrack(gray, maxCorners=300, qualityLevel=0.01, minDistance=7)
        return np.zeros_like(gray, dtype=np.uint8)

    if _prev_pts is None:
        _prev_pts = cv2.goodFeaturesToTrack(gray, maxCorners=300, qualityLevel=0.01, minDistance=7)
        return np.zeros_like(gray, dtype=np.uint8)

    pts, st, err = cv2.calcOpticalFlowPyrLK(
        _prev_gray_discrete, gray,
        _prev_pts, None,
        **lk_params
    )

    mask = np.zeros_like(gray, dtype=np.uint8)

    if pts is not None:
        good_new = pts[st == 1]
        good_old = _prev_pts[st == 1]

        for (new, old) in zip(good_new, good_old):
            x_new, y_new = new.ravel()
            x_old, y_old = old.ravel()
            dist = np.hypot(x_new - x_old, y_new - y_old)
            if dist > 2:  # umbral de movimiento
                cv2.circle(mask, (int(x_new), int(y_new)), 3, 255, -1)

    _prev_gray_discrete = gray
    _prev_pts = good_new.reshape(-1, 1, 2) if pts is not None else None

    return mask


###############################################
# 4) CANNY EN FRAME ACTUAL Y ANTERIOR + RESTA
###############################################

_prev_edges = None

def process_frame_canny_diff(frame: np.ndarray) -> np.ndarray:
    """
    Aplica Canny al frame actual y al anterior y devuelve únicamente
    la máscara de la diferencia absoluta entre ambos.
    No se calculan contornos ni se hace postprocesado.
    """
    global _prev_edges

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)

    if _prev_edges is None:
        _prev_edges = edges
        return np.zeros_like(edges, dtype=np.uint8)

    # Diferencia directa entre bordes actuales y anteriores
    diff = cv2.absdiff(edges, _prev_edges)

    _prev_edges = edges
    return diff
