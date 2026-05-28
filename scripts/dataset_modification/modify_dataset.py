#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Editor interactivo de bounding boxes para formato YOLO.
Directorio esperado:
  ./images  -> imágenes (jpg, png, ...)
  ./labels  -> archivos .txt con formato YOLO (one file per image)

Controles:
  r : modo draw (dibujar nueva caja)
  v : modo select (seleccionar cajas)
  m : modo move (mover caja seleccionada)
  d : borrar caja seleccionada
  x : eliminar imagen actual y su fichero de label (confirmar con y)
  0-9 : cambiar clase actual (para nuevas cajas o caja seleccionada)
  s : guardar labels
  n : siguiente imagen (guarda)
  p : anterior imagen (guarda)
  q / ESC : salir (guarda)
  i : mostrar/ocultar leyenda
"""

import os
import glob
import cv2

DIR = "./prueba_autolabel_diurna"
IMAGE_DIR = DIR + "/images"
LABEL_DIR = DIR + "/labels"
SUPPORTED_EXT = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")

# Estado global
state = {
    "images": [],
    "idx": 0,
    "img": None,
    "img_path": None,
    "boxes": [],  # list of [cls, x1, y1, x2, y2] in pixel coords
    "selected": -1,
    "mode": "view",  # view, draw, select, move
    "drawing": False,
    "moving": False,
    "draw_start": (0, 0),
    "move_start": (0, 0),
    "current_class": 0,
    "show_legend": True,
    "window_name": "YOLO Editor"
}

# Utilidades IO YOLO
def label_path_for_image(img_path):
    base = os.path.splitext(os.path.basename(img_path))[0]
    return os.path.join(LABEL_DIR, base + ".txt")

def load_labels_for_image(img_path, img_w, img_h):
    path = label_path_for_image(img_path)
    boxes = []
    if os.path.exists(path):
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue
                try:
                    cls = int(parts[0])
                    x_c = float(parts[1])
                    y_c = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])
                except Exception:
                    continue
                # convertir a coords de píxeles
                x1 = int((x_c - w/2.0) * img_w)
                y1 = int((y_c - h/2.0) * img_h)
                x2 = int((x_c + w/2.0) * img_w)
                y2 = int((y_c + h/2.0) * img_h)
                boxes.append([cls, x1, y1, x2, y2])
    return boxes

def save_labels_for_image(img_path, boxes):
    os.makedirs(LABEL_DIR, exist_ok=True)
    path = label_path_for_image(img_path)
    if state["img"] is None:
        return
    h, w = state["img"].shape[:2]
    lines = []
    for b in boxes:
        cls, x1, y1, x2, y2 = b
        # clamp
        x1c = max(0, min(w-1, x1))
        y1c = max(0, min(h-1, y1))
        x2c = max(0, min(w-1, x2))
        y2c = max(0, min(h-1, y2))
        bw = max(1, x2c - x1c)
        bh = max(1, y2c - y1c)
        x_center = (x1c + x2c) / 2.0 / w
        y_center = (y1c + y2c) / 2.0 / h
        w_norm = bw / w
        h_norm = bh / h
        lines.append(f"{cls} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
    with open(path, "w") as f:
        f.write("\n".join(lines))

# Detección de clic dentro de caja
def point_in_box(x, y, box):
    _, x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2

def find_box_at(x, y):
    # devuelve índice de la caja más interna que contiene el punto, o -1
    for i in range(len(state["boxes"]) - 1, -1, -1):
        if point_in_box(x, y, state["boxes"][i]):
            return i
    return -1

# Dibujar overlay (con fondo blanco para la leyenda, alineado correctamente)
def draw_overlay(img):
    disp = img.copy()
    # dibujar cajas
    for i, b in enumerate(state["boxes"]):
        cls, x1, y1, x2, y2 = b
        color = (0, 255, 0) if i != state["selected"] else (0, 165, 255)
        cv2.rectangle(disp, (x1, y1), (x2, y2), color, 2)
        label = str(cls)
        cv2.putText(disp, label, (x1, max(10, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # si estamos dibujando, mostrar caja temporal
    if state.get("drawing", False):
        x0, y0 = state["draw_start"]
        x1, y1 = state["temp_pt"]
        cv2.rectangle(disp, (x0, y0), (x1, y1), (255, 0, 0), 1)

    # leyenda con fondo blanco correctamente alineado
    if state.get("show_legend", True):
        lines = [
            f"Modo: {state['mode']}    Clase actual: {state['current_class']}",
            "r: draw  v: select  m: move  d: delete  x: del image",
            "0-9: set class  s: save  n: next  p: prev  q/ESC: quit  i: toggle legend"
        ]
        x0, y0 = 10, 20
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1
        line_spacing = 6
        padding = 6

        # calcular ancho máximo, altura de línea y baseline
        max_w = 0
        text_h = 0
        baseline = 0
        for ln in lines:
            (tw, th), bs = cv2.getTextSize(ln, font, font_scale, thickness)
            if tw > max_w:
                max_w = tw
            text_h = max(text_h, th)
            baseline = max(baseline, bs)

        # dimensiones del rectángulo de fondo
        rect_w = max_w + padding * 2
        rect_h = (text_h + line_spacing) * len(lines) - line_spacing + padding * 2

        # calcular coordenadas del rectángulo teniendo en cuenta baseline
        rect_x1 = x0 - padding
        rect_y1 = y0 - padding - baseline
        rect_x2 = rect_x1 + rect_w
        rect_y2 = rect_y1 + rect_h

        # asegurar que no salga de la imagen
        rect_x1 = max(0, rect_x1)
        rect_y1 = max(0, rect_y1)
        rect_x2 = min(disp.shape[1]-1, rect_x2)
        rect_y2 = min(disp.shape[0]-1, rect_y2)

        # dibujar fondo blanco sólido
        cv2.rectangle(disp, (rect_x1, rect_y1), (rect_x2, rect_y2), (255,255,255), -1)

        # dibujar texto en negro, alineado con y0 y usando text_h y baseline
        for i, ln in enumerate(lines):
            ty = y0 + i * (text_h + line_spacing)
            cv2.putText(disp, ln, (x0, ty), font, font_scale, (0,0,0), thickness, cv2.LINE_AA)

    return disp

# Mouse callback
def mouse_cb(event, x, y, flags, param):
    if state["mode"] == "draw":
        if event == cv2.EVENT_LBUTTONDOWN:
            state["drawing"] = True
            state["draw_start"] = (x, y)
            state["temp_pt"] = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and state.get("drawing", False):
            state["temp_pt"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and state.get("drawing", False):
            x0, y0 = state["draw_start"]
            x1, y1 = x, y
            x_min, x_max = sorted([x0, x1])
            y_min, y_max = sorted([y0, y1])
            # evitar cajas degeneradas
            if abs(x_max - x_min) > 5 and abs(y_max - y_min) > 5:
                state["boxes"].append([state["current_class"], x_min, y_min, x_max, y_max])
                state["selected"] = len(state["boxes"]) - 1
            state["drawing"] = False
    elif state["mode"] in ("select", "view"):
        if event == cv2.EVENT_LBUTTONDOWN:
            idx = find_box_at(x, y)
            state["selected"] = idx
    elif state["mode"] == "move":
        if event == cv2.EVENT_LBUTTONDOWN:
            idx = find_box_at(x, y)
            if idx >= 0:
                state["selected"] = idx
                state["moving"] = True
                state["move_start"] = (x, y)
                # store original coords
                state["orig_box"] = state["boxes"][idx].copy()
        elif event == cv2.EVENT_MOUSEMOVE and state.get("moving", False):
            dx = x - state["move_start"][0]
            dy = y - state["move_start"][1]
            idx = state["selected"]
            if idx >= 0:
                cls, x1, y1, x2, y2 = state["orig_box"]
                state["boxes"][idx] = [cls, int(x1 + dx), int(y1 + dy), int(x2 + dx), int(y2 + dy)]
        elif event == cv2.EVENT_LBUTTONUP and state.get("moving", False):
            state["moving"] = False
            state.pop("orig_box", None)

# Cargar imagen actual
def load_current_image():
    if not state["images"]:
        state["img"] = None
        state["img_path"] = None
        state["boxes"] = []
        state["selected"] = -1
        return
    # asegurar índice válido
    if state["idx"] < 0:
        state["idx"] = 0
    if state["idx"] >= len(state["images"]):
        state["idx"] = len(state["images"]) - 1
    path = state["images"][state["idx"]]
    img = cv2.imread(path)
    if img is None:
        print("No se pudo leer:", path)
        # eliminar de la lista para no bloquear
        try:
            state["images"].pop(state["idx"])
        except Exception:
            pass
        if state["images"]:
            state["idx"] = min(state["idx"], len(state["images"]) - 1)
            load_current_image()
        else:
            state["img"] = None
            state["img_path"] = None
            state["boxes"] = []
            state["selected"] = -1
        return
    state["img"] = img
    state["img_path"] = path
    h, w = img.shape[:2]
    state["boxes"] = load_labels_for_image(path, w, h)
    state["selected"] = -1

# Guardar y avanzar/retroceder
def save_current():
    if state["img_path"] is None:
        return
    save_labels_for_image(state["img_path"], state["boxes"])
    print("Guardado:", label_path_for_image(state["img_path"]))

def next_image():
    save_current()
    if state["idx"] < len(state["images"]) - 1:
        state["idx"] += 1
        load_current_image()

def prev_image():
    save_current()
    if state["idx"] > 0:
        state["idx"] -= 1
        load_current_image()

# Inicialización: listar imágenes
def find_images():
    imgs = []
    for pat in SUPPORTED_EXT:
        imgs.extend(glob.glob(os.path.join(IMAGE_DIR, pat)))
    imgs = sorted(imgs)
    return imgs

# Eliminar imagen y su label (con confirmación)
def delete_current_image_and_label():
    img_path = state.get("img_path")
    if img_path is None:
        return
    label_path = label_path_for_image(img_path)
    try:
        if os.path.exists(img_path):
            os.remove(img_path)
            print("Eliminada imagen:", img_path)
        if os.path.exists(label_path):
            os.remove(label_path)
            print("Eliminado label:", label_path)
    except Exception as e:
        print("Error al eliminar archivos:", e)
    # quitar de la lista y ajustar índice
    try:
        cur_idx = state["idx"]
        state["images"].pop(cur_idx)
        if cur_idx >= len(state["images"]):
            state["idx"] = max(0, len(state["images"]) - 1)
    except Exception:
        pass
    # cargar siguiente (o dejar vacío si no hay)
    if state["images"]:
        load_current_image()
    else:
        state["img"] = None
        state["img_path"] = None
        state["boxes"] = []
        state["selected"] = -1

def main():
    state["images"] = find_images()
    if not state["images"]:
        print("No se encontraron imágenes en", IMAGE_DIR)
        return
    load_current_image()
    cv2.namedWindow(state["window_name"], cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.setMouseCallback(state["window_name"], mouse_cb)

    while True:
        if state["img"] is None:
            break
        disp = draw_overlay(state["img"])
        # mostrar nombre de archivo (sin fondo)
        fname = os.path.basename(state["img_path"])
        cv2.putText(disp, fname, (10, disp.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.imshow(state["window_name"], disp)
        key = cv2.waitKey(20) & 0xFF
        if key == 255:
            continue
        # teclas
        if key in (ord('q'), 27):  # q o ESC
            save_current()
            break
        elif key == ord('n'):
            next_image()
        elif key == ord('p'):
            prev_image()
        elif key == ord('s'):
            save_current()
        elif key == ord('r'):
            state["mode"] = "draw"
        elif key == ord('v'):
            state["mode"] = "select"
        elif key == ord('m'):
            state["mode"] = "move"
        elif key == ord('d'):
            if 0 <= state["selected"] < len(state["boxes"]):
                del state["boxes"][state["selected"]]
                state["selected"] = -1
        elif key == ord('i'):
            state["show_legend"] = not state["show_legend"]
        elif key == ord('x'):
            # pedir confirmación: mostrar mensaje en la ventana y esperar y/n
            confirm_msg = "Eliminar imagen y label? (y/n)"
            tmp = draw_overlay(state["img"])
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2
            (tw, th), bs = cv2.getTextSize(confirm_msg, font, font_scale, thickness)
            px, py = 10, 40
            # usar baseline para posicionar correctamente el rectángulo
            rect_x1 = px - 6
            rect_y1 = py - th - bs - 6
            rect_x2 = px + tw + 6
            rect_y2 = py + 6
            # asegurar dentro de la imagen
            rect_x1 = max(0, rect_x1)
            rect_y1 = max(0, rect_y1)
            rect_x2 = min(tmp.shape[1]-1, rect_x2)
            rect_y2 = min(tmp.shape[0]-1, rect_y2)
            cv2.rectangle(tmp, (rect_x1, rect_y1), (rect_x2, rect_y2), (255,255,255), -1)
            cv2.putText(tmp, confirm_msg, (px, py), font, font_scale, (0,0,0), thickness, cv2.LINE_AA)
            cv2.imshow(state["window_name"], tmp)
            # esperar respuesta
            while True:
                k2 = cv2.waitKey(0) & 0xFF
                if k2 in (ord('y'), ord('Y')):
                    delete_current_image_and_label()
                    break
                elif k2 in (ord('n'), ord('N'), 27):
                    break
                else:
                    continue
        # números para clase
        elif ord('0') <= key <= ord('9'):
            cls = key - ord('0')
            state["current_class"] = cls
            if 0 <= state["selected"] < len(state["boxes"]):
                state["boxes"][state["selected"]][0] = cls
        # guardar con Ctrl+S (opcional)
        elif key == 19:  # Ctrl+S
            save_current()
        # otras teclas: nada
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
