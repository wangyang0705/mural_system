# drawing_utils.py

import numpy as np
import cv2
from PIL import Image, ImageDraw


def skeletonize(mask: np.ndarray) -> np.ndarray:
    """简单形态学骨架提取，用于裂隙线段可视化"""
    img = mask.astype(np.uint8)
    size = img.size
    skel = np.zeros(img.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    done = False
    while not done:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        zeros = size - cv2.countNonZero(img)
        if zeros == size:
            done = True
    return skel > 0


def generate_pattern_image(disease_type: str, disease_colors: dict) -> Image.Image:
    """生成小图例，供 '水渍' 等病害进行图示填充"""
    size = 64
    img = Image.new("RGBA", (size, size), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    color = disease_colors.get(disease_type, (255, 0, 0))

    if disease_type == "颜料层脱落":
        draw.rectangle([8, 8, size - 8, size - 8], outline=color, width=2)

    elif disease_type == "地仗层脱落":
        draw.rectangle([4, 4, size - 4, size - 4], outline=color, width=2)
        draw.rectangle([12, 12, size - 12, size - 12], outline=color, width=2)

    elif disease_type == "水渍":
        for i in range(0, size, 8):
            for j in range(0, size, 6):
                if (i // 8) % 2 == 0:
                    draw.line([j, i, j + 4, i], fill=color, width=2)

    elif disease_type == "裂隙":
        center = size // 2
        draw.line([8, center, size - 8, center], fill=color, width=2)
        for i in range(12, size - 8, 10):
            draw.line([i, center - 4, i, center + 4], fill=color, width=1)

    return img


def _draw_diagonal_lines_in_area(image, contour, color, line_width=1):
    """地仗层脱落的斜线填充"""
    try:
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [contour], 255)
        x, y, w, h = cv2.boundingRect(contour)
        spacing = max(8, line_width * 4)

        for i in range(x - h - 10, x + w + h + 10, spacing):
            sx = x + w + 10
            sy = i - (x + w) + y - 10
            ex = x - 10
            ey = i + 10
            if sx <= ex or sy >= ey:
                continue
            pts = []
            steps = max(abs(sx - ex), abs(ey - sy))
            for j in range(steps + 1):
                px = int(sx - j * (sx - ex) / steps)
                py = int(sy + j * (ey - sy) / steps)
                if (0 <= px < mask.shape[1] and
                        0 <= py < mask.shape[0] and
                        mask[py, px] > 0):
                    pts.append((px, py))
            if len(pts) >= 2:
                for k in range(len(pts) - 1):
                    cv2.line(image, pts[k], pts[k + 1], color, line_width)
    except Exception as e:
        print(f"在区域内绘制斜线失败: {e}")


def _create_pattern_fill(base_image, mask_bool, pattern_image, disease):
    """用于水渍的重复纹理填充"""
    try:
        if pattern_image is None or mask_bool is None:
            return base_image
        if disease != "水渍":
            return base_image

        base_rgba = cv2.cvtColor(base_image, cv2.COLOR_RGB2RGBA)
        mask = mask_bool

        ys, xs = np.where(mask)
        if len(ys) == 0:
            return base_image

        min_y, max_y = np.min(ys), np.max(ys)
        min_x, max_x = np.min(xs), np.max(xs)
        region_h = max_y - min_y + 1
        region_w = max_x - min_x + 1

        pattern_np = np.array(pattern_image)
        ph, pw = pattern_np.shape[:2]
        tiles_y = (region_h + ph - 1) // ph
        tiles_x = (region_w + pw - 1) // pw

        tiled = np.zeros((tiles_y * ph, tiles_x * pw, 4), dtype=np.uint8)
        for ty in range(tiles_y):
            for tx in range(tiles_x):
                ys0 = ty * ph
                xs0 = tx * pw
                tiled[ys0:ys0 + ph, xs0:xs0 + pw] = pattern_np
        tiled = tiled[:region_h, :region_w]

        result_img = base_rgba.copy()
        for ry in range(region_h):
            for rx in range(region_w):
                gy = min_y + ry
                gx = min_x + rx
                if (gy < base_rgba.shape[0] and gx < base_rgba.shape[1]
                        and mask[gy, gx]):
                    px = tiled[ry, rx]
                    alpha = px[3] / 255.0
                    if alpha > 0:
                        for c in range(3):
                            result_img[gy, gx, c] = (
                                result_img[gy, gx, c] * (1 - alpha)
                                + px[c] * alpha
                            )
                        result_img[gy, gx, 3] = 255

        return cv2.cvtColor(result_img, cv2.COLOR_RGBA2RGB)

    except Exception as e:
        print(f"水渍填充失败: {e}")
        import traceback
        traceback.print_exc()
        return base_image


def apply_pattern_fill(base_image, mask, disease, pattern_image, disease_colors):
    """
    根据病害类型绘制到 base_image 上（不依赖 GUI 的 state）。
    base_image 与 mask 尺寸必须一致。
    """
    try:
        if mask is None:
            return base_image

        img = base_image.copy()
        mask_u8 = mask.astype(np.uint8)

        color = disease_colors.get(disease, (255, 0, 0))
        color_bgr = (color[2], color[1], color[0])

        if disease == "裂隙":
            skel = skeletonize(mask_u8 > 0)
            if not np.any(skel):
                return base_image
            contours, _ = cv2.findContours(
                skel.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            for contour in contours:
                if len(contour) < 2:
                    continue
                pts = contour.reshape(-1, 2)
                for i in range(len(pts) - 1):
                    cv2.line(img, tuple(pts[i]), tuple(pts[i + 1]), color_bgr, 2)
                for i in range(0, len(pts), 5):
                    if i < len(pts) - 1:
                        p0 = pts[i]
                        p1 = pts[min(i + 1, len(pts) - 1)]
                        dx = p1[0] - p0[0]
                        dy = p1[1] - p0[1]
                        length = float(np.hypot(dx, dy)) + 1e-6
                        nx = -dy / length
                        ny = dx / length
                        half_len = 4
                        sx = int(p0[0] - nx * half_len)
                        sy = int(p0[1] - ny * half_len)
                        ex = int(p0[0] + nx * half_len)
                        ey = int(p0[1] + ny * half_len)
                        cv2.line(img, (sx, sy), (ex, ey), color_bgr, 1)
            return img

        contours, _ = cv2.findContours(
            mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return base_image

        if disease == "颜料层脱落":
            cv2.drawContours(img, contours, -1, color_bgr, 2)

        elif disease == "地仗层脱落":
            for contour in contours:
                if len(contour) < 3:
                    continue
                temp = img.copy()
                _draw_diagonal_lines_in_area(temp, contour, color_bgr, 1)
                outer_mask = np.zeros(img.shape[:2], dtype=np.uint8)
                inner_mask = np.zeros(img.shape[:2], dtype=np.uint8)
                cv2.fillPoly(outer_mask, [contour], 255)
                x, y, w, h = cv2.boundingRect(contour)
                contour_size = min(w, h)
                shrink_ratio = max(0.7, 1.0 - (20.0 / contour_size))
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    inner = contour.astype(np.float32)
                    inner = (inner - [cx, cy]) * shrink_ratio + [cx, cy]
                    inner = inner.astype(np.int32)
                    cv2.fillPoly(inner_mask, [inner], 255)
                    ring_mask = cv2.subtract(outer_mask, inner_mask)
                    ring_3 = np.stack([ring_mask] * 3, axis=2)
                    img = np.where(ring_3 > 0, temp, img).astype(np.uint8)
                    cv2.drawContours(img, [contour], -1, color_bgr, 2)
                    cv2.drawContours(img, [inner], -1, color_bgr, 2)

        elif disease == "水渍":
            cv2.drawContours(img, contours, -1, color_bgr, 2)
            if pattern_image is not None:
                img = _create_pattern_fill(img, mask_u8 > 0, pattern_image, disease)

        else:
            cv2.drawContours(img, contours, -1, color_bgr, 2)

        return img

    except Exception as e:
        print(f"病害绘制失败: {e}")
        import traceback
        traceback.print_exc()
        return base_image
