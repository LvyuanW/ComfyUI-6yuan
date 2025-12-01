import torch
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import json
import math

class ShapeLanguageSummary:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "resize_to": ("INT", {"default": 256, "min": 64, "max": 1024}),
                "edge_detector": (["canny", "sobel"], {"default": "canny"}),
                "angle_tolerance_deg": ("INT", {"default": 10, "min": 1, "max": 30}),
            },
        }

    RETURN_TYPES = ("STRING", "IMAGE",)
    RETURN_NAMES = ("metrics_json", "overlay_image",)
    FUNCTION = "analyze"
    CATEGORY = "6yuan/analyze"

    DESCRIPTION = """
ShapeLanguageSummary 节点输出说明（给设计/视觉同事）
• style_axes.geometry.circle / square / angular（原 pointy）
  0~1，三者加起来约等于 1。
  表示画面整体在“圆形 / 方形 / 锐角形”三种几何风格上的相对倾向。
• style_axes.direction.orthogonal / diagonal / mixed_direction（原 random）
  0~1，三者约等于 1。
  根据轮廓和线条判断：画面更偏水平垂直、对角线，还是没有明显主导方向。
• style_axes.complexity_level
  "low" | "medium" | "high"
  按边缘密度和轮廓数量评估画面复杂度（元素/细节多少）。
• style_axes.symmetry_level
  "low" | "medium" | "high"
  判断画面在水平、垂直或对角轴上的对称程度。
• style_axes.logo_type
  "text" | "graphic" | "mixed"
  粗略判断是“文字为主”、“图形为主”还是“图文混合”。
• scheme_suggestion + scheme_confidence
  推荐使用的版式类型（如：square_grid_theme, diagonal_theme, typography_focus），以及推荐的置信度（0~1）。
  我们用它来自动选择/优先推荐某些海报模板，你们可以按需要采纳或覆盖。
"""

    def analyze(self, image, resize_to=256, edge_detector="canny", angle_tolerance_deg=10):
        # Handle batch: process first image only
        if len(image.shape) == 4:
            img_tensor = image[0]
        else:
            img_tensor = image

        # Convert to numpy (H, W, C) -> 0-255 uint8
        img_np = (img_tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        
        # Resize for analysis
        h, w = img_np.shape[:2]
        scale = resize_to / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img_resized = cv2.resize(img_np, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Grayscale
        if len(img_resized.shape) == 3 and img_resized.shape[2] == 3:
            gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
        elif len(img_resized.shape) == 3 and img_resized.shape[2] == 4:
            # Handle RGBA: Composite onto white or black background based on brightness
            alpha = img_resized[:, :, 3].astype(np.float32) / 255.0
            rgb = img_resized[:, :, :3].astype(np.float32)
            
            # Estimate brightness of opaque pixels
            gray_raw = cv2.cvtColor(img_resized[:, :, :3], cv2.COLOR_RGB2GRAY)
            mask = img_resized[:, :, 3] > 10
            
            if mask.any():
                mean_val = gray_raw[mask].mean()
                # If object is bright, use black background. If object is dark, use white background.
                bg_color = 0 if mean_val > 127 else 255
            else:
                bg_color = 255
                
            # Composite
            # result = rgb * alpha + bg * (1 - alpha)
            composited = rgb * alpha[:, :, np.newaxis] + bg_color * (1.0 - alpha[:, :, np.newaxis])
            gray = cv2.cvtColor(composited.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = img_resized

        # --- 1. Edge Detection ---
        if edge_detector == "canny":
            edges = cv2.Canny(gray, 50, 150)
        else:
            grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
            abs_grad_x = cv2.convertScaleAbs(grad_x)
            abs_grad_y = cv2.convertScaleAbs(grad_y)
            edges = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
            _, edges = cv2.threshold(edges, 50, 255, cv2.THRESH_BINARY)

        # --- 2. Contour Analysis (Geometry) ---
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        total_roundness = 0
        total_squareness = 0
        total_pointiness = 0
        total_area = 0
        
        # For visualization
        vis_contours = []
        
        valid_contours_count = 0
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 10: continue # Skip noise
            
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0: continue
            
            # Roundness: 4 * pi * Area / Perimeter^2 (Circle = 1.0)
            circularity = 4 * math.pi * area / (perimeter * perimeter)
            
            # Squareness: Area / BoundingRectArea
            _, _, bw, bh = cv2.boundingRect(cnt)
            rect_area = bw * bh
            rectangularity = area / rect_area if rect_area > 0 else 0
            
            # Convexity / Pointiness
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            # Define Pointiness as inverse of solidity or high poly approximation
            approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
            vertices = len(approx)
            
            # Score heuristics
            r_score = circularity
            s_score = rectangularity if vertices <= 6 else 0 # Squares usually have few vertices
            # Pointiness: low solidity (star shape) or sharp angles (triangles)
            p_score = (1.0 - solidity) + (1.0 if vertices == 3 else 0.0)
            
            weight = area # Weight by area
            total_roundness += r_score * weight
            total_squareness += s_score * weight
            total_pointiness += p_score * weight
            total_area += weight
            valid_contours_count += 1
            
            vis_contours.append(cnt)

        if total_area > 0:
            roundness_score = min(1.0, total_roundness / total_area)
            squareness_score = min(1.0, total_squareness / total_area)
            pointiness_score = min(1.0, total_pointiness / total_area)
        else:
            roundness_score, squareness_score, pointiness_score = 0, 0, 0

        # Refine Squareness: High rectangularity doesn't always mean square. 
        # But for this model, we stick to user's "squareness" meaning blocky/orthogonal.
        
        # --- 3. Hough Lines (Direction) ---
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, minLineLength=10, maxLineGap=10)
        
        orth_len = 0
        diag_len = 0
        total_line_len = 0
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = math.hypot(x2 - x1, y2 - y1)
                angle = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180
                
                # Orthogonal: near 0, 90, 180
                is_orth = False
                for target in [0, 90, 180]:
                    if abs(angle - target) < angle_tolerance_deg:
                        is_orth = True
                        break
                
                # Diagonal: near 45, 135
                is_diag = False
                for target in [45, 135]:
                    if abs(angle - target) < angle_tolerance_deg:
                        is_diag = True
                        break
                        
                if is_orth:
                    orth_len += length
                elif is_diag:
                    diag_len += length
                
                total_line_len += length

        if total_line_len > 0:
            orthogonal_bias = orth_len / total_line_len
            diagonal_bias = diag_len / total_line_len
        else:
            orthogonal_bias, diagonal_bias = 0.5, 0.0 # Fallback

        orientation_confidence = min(1.0, (orth_len + diag_len) / (total_line_len + 1e-6))

        # --- 4. Curve vs Line Ratio ---
        # Edges pixels that are part of Hough Lines vs Total Edge Pixels
        total_edge_pixels = cv2.countNonZero(edges)
        line_pixels = total_line_len # Approximate
        
        line_ratio = min(1.0, line_pixels / (total_edge_pixels + 1e-6))
        curve_ratio = 1.0 - line_ratio

        # --- 5. Symmetry Analysis ---
        # Resize to small square for symmetry check
        sym_size = 64
        img_sym = cv2.resize(gray, (sym_size, sym_size))
        
        # Horizontal flip
        flip_h = cv2.flip(img_sym, 1)
        score_h = cv2.matchTemplate(img_sym, flip_h, cv2.TM_CCOEFF_NORMED)[0][0]
        
        # Vertical flip
        flip_v = cv2.flip(img_sym, 0)
        score_v = cv2.matchTemplate(img_sym, flip_v, cv2.TM_CCOEFF_NORMED)[0][0]
        
        # Diagonal flip (Transpose)
        img_t = cv2.transpose(img_sym)
        score_d = cv2.matchTemplate(img_sym, img_t, cv2.TM_CCOEFF_NORMED)[0][0]

        symmetry_horizontal = max(0, score_h)
        symmetry_vertical = max(0, score_v)
        symmetry_diagonal = max(0, score_d)

        # --- 6. Complexity Index ---
        # Edge density + Contour count
        edge_density = total_edge_pixels / (new_w * new_h)
        # Normalize: say 20% edge density is very high complexity
        complexity_index = min(1.0, (edge_density / 0.15) * 0.7 + (valid_contours_count / 50) * 0.3)

        # --- 7. Text vs Graphic Ratio ---
        # Heuristic: Text has many small contours, high edge density
        # Graphic has fewer large contours
        if valid_contours_count > 10 and complexity_index > 0.4:
            text_graphic_ratio = 0.8 # Likely text
        elif valid_contours_count < 5 and complexity_index < 0.2:
            text_graphic_ratio = 0.1 # Likely graphic
        else:
            text_graphic_ratio = 0.5 # Mixed

        # --- 8. Gridness (Simplified) ---
        # High orthogonal bias + High complexity + Regularity (omitted for speed, using proxy)
        gridness_score = orthogonal_bias * complexity_index

        # =========================================================
        # AGGREGATION ALGORITHM (User Provided)
        # =========================================================

        # --- Step 1: Geometry Axis ---
        geo_circle_raw = roundness_score * curve_ratio * (1.0 - squareness_score)
        geo_square_raw = squareness_score * orthogonal_bias * line_ratio
        geo_pointy_raw = pointiness_score * (diagonal_bias * 0.7 + (1.0 - orthogonal_bias) * 0.3)

        s_geo = geo_circle_raw + geo_square_raw + geo_pointy_raw + 1e-6
        geo_circle = geo_circle_raw / s_geo
        geo_square = geo_square_raw / s_geo
        geo_pointy = geo_pointy_raw / s_geo

        # --- Step 2: Direction Axis ---
        base_dir = orthogonal_bias + diagonal_bias + 1e-6
        orth_norm = orthogonal_bias / base_dir
        diag_norm = diagonal_bias / base_dir
        # rand_norm = 1.0 - max(orth_norm, diag_norm) # Unused in user algo

        orth_score = orth_norm * orientation_confidence
        diag_score = diag_norm * orientation_confidence
        rand_score = 1.0 - orientation_confidence

        s_dir = orth_score + diag_score + rand_score + 1e-6
        orth = orth_score / s_dir
        diag = diag_score / s_dir
        rand = rand_score / s_dir

        # --- Step 3: Complexity & Symmetry Levels ---
        if complexity_index < 0.3:
            complexity_level = "low"
        elif complexity_index < 0.7:
            complexity_level = "medium"
        else:
            complexity_level = "high"

        sym_agg = max(symmetry_horizontal, symmetry_vertical, symmetry_diagonal)
        if sym_agg < 0.3:
            symmetry_level = "low"
        elif sym_agg < 0.7:
            symmetry_level = "medium"
        else:
            symmetry_level = "high"

        # --- Step 4: Logo Type ---
        if text_graphic_ratio > 0.7:
            logo_type = "text"
        elif text_graphic_ratio < 0.3:
            logo_type = "graphic"
        else:
            logo_type = "mixed"

        # --- Step 5: Scheme Suggestion ---
        def symmetry_level_weight(level):
            return {"low": 0.2, "medium": 0.6, "high": 1.0}.get(level, 0.5)

        scores = {}
        
        # Circle Theme
        scores["circle_theme"] = (
            geo_circle * 0.6 +
            (1.0 - complexity_index) * 0.2 +
            symmetry_level_weight(symmetry_level) * 0.2
        )

        # Square Grid Theme
        scores["square_grid_theme"] = (
            geo_square * 0.5 +
            orth * 0.3 +
            gridness_score * 0.2
        )

        # Diagonal Theme
        scores["diagonal_theme"] = (
            diag * 0.5 +
            geo_pointy * 0.2 +
            (1.0 - sym_agg) * 0.3
        )

        # Grid Layout
        scores["grid_layout"] = (
            gridness_score * 0.6 +
            orth * 0.2 +
            (1.0 - complexity_index) * 0.2
        )

        # Typography Focus
        scores["typography_focus"] = (
            text_graphic_ratio * 0.6 +
            complexity_index * 0.2 +
            orth * 0.2
        )

        best_scheme = max(scores, key=scores.get)
        max_score = scores[best_scheme]
        sum_score = sum(scores.values()) + 1e-6
        confidence = max_score / sum_score

        # Confidence fallback
        if confidence < 0.2: # User said 0.4, but raw sum might be low? Ratio should be fine.
             # Actually sum_score is sum of weighted averages roughly.
             # Let's stick to simple confidence.
             pass

        # --- Construct Output JSON ---
        result_json = {
            "style_axes": {
                "geometry": {
                    "circle": round(float(geo_circle), 2),
                    "square": round(float(geo_square), 2),
                    "angular": round(float(geo_pointy), 2)
                },
                "direction": {
                    "orthogonal": round(float(orth), 2),
                    "diagonal": round(float(diag), 2),
                    "mixed_direction": round(float(rand), 2)
                },
                "complexity_level": complexity_level,
                "symmetry_level": symmetry_level,
                "logo_type": logo_type
            },
            "scheme_suggestion": best_scheme,
            "scheme_confidence": round(float(confidence), 2),
            "raw_metrics": {
                "roundness_score": round(float(roundness_score), 3),
                "squareness_score": round(float(squareness_score), 3),
                "angularity_score": round(float(pointiness_score), 3),
                "curve_ratio": round(float(curve_ratio), 3),
                "line_ratio": round(float(line_ratio), 3),
                "orthogonal_bias": round(float(orthogonal_bias), 3),
                "diagonal_bias": round(float(diagonal_bias), 3),
                "orientation_confidence": round(float(orientation_confidence), 3),
                "complexity_index": round(float(complexity_index), 3),
                "symmetry_horizontal": round(float(symmetry_horizontal), 3),
                "symmetry_vertical": round(float(symmetry_vertical), 3),
                "symmetry_diagonal": round(float(symmetry_diagonal), 3),
                "text_graphic_ratio": round(float(text_graphic_ratio), 3),
                "grid_strength": round(float(gridness_score), 3)
            }
        }

        # --- Visualization ---
        vis_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        
        # Draw contours
        cv2.drawContours(vis_img, vis_contours, -1, (0, 255, 0), 1)
        
        # Draw Hough Lines
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(vis_img, (x1, y1), (x2, y2), (0, 0, 255), 1)

        # Convert to PIL for text overlay
        pil_img = Image.fromarray(vis_img)
        draw = ImageDraw.Draw(pil_img)
        
        # Load font (default)
        try:
            font = ImageFont.truetype("Arial.ttf", 12)
        except:
            font = ImageFont.load_default()

        # Draw summary text
        summary_text = f"Scheme: {best_scheme} ({confidence:.2f})\n" \
                       f"Geo: C={geo_circle:.2f} S={geo_square:.2f} A={geo_pointy:.2f}\n" \
                       f"Dir: O={orth:.2f} D={diag:.2f} M={rand:.2f}\n" \
                       f"Comp: {complexity_level} | Sym: {symmetry_level}"
        
        draw.text((5, 5), summary_text, fill=(255, 255, 0), font=font)
        
        # Convert back to tensor
        vis_np = np.array(pil_img)
        vis_tensor = torch.from_numpy(vis_np.astype(np.float32) / 255.0).unsqueeze(0)

        return (json.dumps(result_json, indent=2), vis_tensor)

NODE_CLASS_MAPPINGS = {
    "ShapeLanguageSummary": ShapeLanguageSummary,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ShapeLanguageSummary": "Shape Language Summary",
}
