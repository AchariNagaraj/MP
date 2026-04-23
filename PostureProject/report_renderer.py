"""
Professional posture analysis report renderer.
Matches reference dashboard layout: 3-column with cards, circular gauge, heatmap legend.
"""
import cv2
import numpy as np
from datetime import datetime


def _status_color(val, good="GOOD", mid="MODERATE"):
    """BGR color for status values."""
    if val == good or val == "LOW":
        return (0, 220, 0)
    if val == mid or val == "MEDIUM":
        return (0, 220, 220)
    return (0, 0, 255)


def _draw_card(canvas, x, y, w, h):
    """Dark card with subtle border."""
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (30, 30, 38), -1)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (55, 55, 65), 1)


def _section_title(canvas, text, x, y):
    """Cyan section header."""
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (210, 170, 50), 2, cv2.LINE_AA)


def _draw_circular_gauge(canvas, cx, cy, radius, score):
    """Circular posture score gauge."""
    thickness = 8
    cv2.ellipse(canvas, (cx, cy), (radius, radius), 0, 0, 360, (45, 45, 55), thickness)

    if score >= 75:
        arc_color = (0, 220, 0)
        label = "Good"
    elif score >= 50:
        arc_color = (0, 220, 220)
        label = "Fair"
    else:
        arc_color = (0, 0, 255)
        label = "Poor"

    end_angle = int(-90 + 360 * score / 100)
    cv2.ellipse(canvas, (cx, cy), (radius, radius), 0, -90, end_angle, arc_color, thickness)

    # Score text
    score_text = str(score)
    (tw, th), _ = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 2)
    cv2.putText(canvas, score_text, (cx - tw // 2, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (240, 240, 245), 2, cv2.LINE_AA)
    (tw2, _), _ = cv2.getTextSize("/ 100", cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
    cv2.putText(canvas, "/ 100", (cx + tw // 2 + 2, cy - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 170), 1, cv2.LINE_AA)
    (tw3, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.putText(canvas, label, (cx - tw3 // 2, cy + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, arc_color, 1, cv2.LINE_AA)


def _draw_heatmap_legend(canvas, x, y, h=120):
    """Vertical JET colormap bar with risk labels."""
    bar_w = 18
    for i in range(h):
        t = 1.0 - i / max(h - 1, 1)
        col = cv2.applyColorMap(np.array([[int(t * 255)]], dtype=np.uint8), cv2.COLORMAP_JET)[0, 0].tolist()
        cv2.line(canvas, (x, y + i), (x + bar_w, y + i), col, 1)

    labels = [("High Risk", (0, 0, 255)), ("Moderate Risk", (0, 180, 255)),
              ("Low Risk", (0, 220, 220)), ("No Risk", (0, 220, 0))]
    spacing = h // (len(labels))
    for idx, (lbl, col) in enumerate(labels):
        ly = y + idx * spacing + spacing // 2
        cv2.putText(canvas, lbl, (x + bar_w + 8, ly + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)


def compute_posture_score(severity_score, posture_status, risk_level):
    """Compute 0-100 posture score from analysis metrics."""
    score = 100.0
    score -= severity_score * 2.0
    if posture_status == "BAD":
        score -= 20
    elif posture_status == "MODERATE":
        score -= 10
    if risk_level == "HIGH":
        score -= 15
    elif risk_level == "MEDIUM":
        score -= 8
    return max(0, min(100, int(score)))


def render_report(
    base_frame,
    heat_overlay,
    posture_status: str,
    severity_level: str,
    risk_level: str,
    duration_sec: float,
    severity_score: float = 0.0,
    features: dict = None,
):
    """
    Renders a professional 3-column posture analysis report dashboard.
    Layout: Left panels | Center image | Right panels | Bottom bar
    """
    if base_frame is None or heat_overlay is None:
        return base_frame

    # --- Layout constants ---
    LEFT_W, RIGHT_W, GAP, MARGIN = 210, 230, 10, 15
    TITLE_H, BOTTOM_H, FOOTER_H = 45, 85, 30

    img_h, img_w = heat_overlay.shape[:2]
    # Scale image if too large
    max_img_w = 520
    if img_w > max_img_w:
        scale = max_img_w / img_w
        img_w = int(img_w * scale)
        img_h = int(img_h * scale)
        heat_overlay = cv2.resize(heat_overlay, (img_w, img_h))

    center_h = img_h
    main_h = max(center_h, 480)

    canvas_w = MARGIN + LEFT_W + GAP + img_w + GAP + RIGHT_W + MARGIN
    canvas_h = MARGIN + TITLE_H + GAP + main_h + GAP + BOTTOM_H + FOOTER_H + MARGIN
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[:] = (18, 18, 24)

    # --- Title ---
    title_text = "POSTURE ANALYSIS REPORT"
    (ttw, _), _ = cv2.getTextSize(title_text, cv2.FONT_HERSHEY_DUPLEX, 0.85, 2)
    cv2.putText(canvas, title_text, ((canvas_w - ttw) // 2, MARGIN + 30),
                cv2.FONT_HERSHEY_DUPLEX, 0.85, (240, 240, 245), 2, cv2.LINE_AA)

    # --- Column positions ---
    lx = MARGIN
    cx = MARGIN + LEFT_W + GAP
    rx = cx + img_w + GAP
    top_y = MARGIN + TITLE_H + GAP

    # ===== CENTER: Heatmap image with overlay labels =====
    _draw_card(canvas, cx - 2, top_y - 2, img_w + 4, img_h + 4)
    canvas[top_y:top_y + img_h, cx:cx + img_w] = heat_overlay

    # Posture/Severity/Risk labels on top of image
    pc = _status_color(posture_status)
    sc = _status_color(severity_level, "LOW", "MEDIUM")
    rc = _status_color(risk_level, "LOW", "MEDIUM")
    lbl_x, lbl_y = cx + 10, top_y + 22
    for i, (lbl, val, col) in enumerate([("Posture:", posture_status, pc), ("Severity:", severity_level, sc), ("Risk:", risk_level, rc)]):
        yy = lbl_y + i * 22
        cv2.putText(canvas, lbl, (lbl_x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 210), 1, cv2.LINE_AA)
        cv2.putText(canvas, val, (lbl_x + 80, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2, cv2.LINE_AA)

    # ===== LEFT COLUMN =====
    # Card 1: Summary
    card_h1 = 120
    _draw_card(canvas, lx, top_y, LEFT_W, card_h1)
    _section_title(canvas, "SUMMARY", lx + 12, top_y + 22)
    summary_items = [
        ("Posture Status", posture_status, _status_color(posture_status)),
        ("Severity Level", severity_level, _status_color(severity_level, "LOW", "MEDIUM")),
        ("Risk Level", risk_level, _status_color(risk_level, "LOW", "MEDIUM")),
    ]
    for i, (lbl, val, col) in enumerate(summary_items):
        yy = top_y + 46 + i * 24
        cv2.putText(canvas, f"{lbl} :", (lx + 12, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (170, 170, 185), 1, cv2.LINE_AA)
        cv2.putText(canvas, val, (lx + 140, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)

    # Card 2: Heatmap Intensity Legend
    card_y2 = top_y + card_h1 + GAP
    card_h2 = 165
    _draw_card(canvas, lx, card_y2, LEFT_W, card_h2)
    _section_title(canvas, "HEATMAP INTENSITY", lx + 12, card_y2 + 22)
    _draw_heatmap_legend(canvas, lx + 15, card_y2 + 36, 120)

    # Card 3: Areas Analyzed
    card_y3 = card_y2 + card_h2 + GAP
    card_h3 = main_h - (card_h1 + card_h2 + GAP * 2)
    if card_h3 < 140:
        card_h3 = 140
    _draw_card(canvas, lx, card_y3, LEFT_W, card_h3)
    _section_title(canvas, "AREAS ANALYZED", lx + 12, card_y3 + 22)

    # Determine area statuses from features
    neck_status = "Low"
    spine_status = "Low"
    shoulder_status = "Low"
    if features:
        na = features.get("neck_angle", 0)
        sa = features.get("spine_angle", 0)
        sh = features.get("shoulder_alignment", 0)
        if na > 20: neck_status = "High"
        elif na > 10: neck_status = "Moderate"
        if sa > 15: spine_status = "High"
        elif sa > 8: spine_status = "Moderate"
        if sh > 30: shoulder_status = "High"
        elif sh > 15: shoulder_status = "Moderate"

    areas = [
        ("Neck", neck_status, "Keep monitor at eye level", (0, 0, 230)),
        ("Upper Spine", spine_status, "Sit straight, avoid slouching", (0, 140, 255)),
        ("Shoulders", shoulder_status, "Relax shoulders", (0, 210, 230)),
    ]
    ay = card_y3 + 38
    for name, status, tip, dot_col in areas:
        cv2.circle(canvas, (lx + 22, ay + 2), 7, dot_col, -1)
        cv2.putText(canvas, name, (lx + 36, ay + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, dot_col, 1, cv2.LINE_AA)
        cv2.putText(canvas, status, (lx + 36, ay + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 230), 1, cv2.LINE_AA)
        cv2.putText(canvas, tip, (lx + 36, ay + 37), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (140, 140, 155), 1, cv2.LINE_AA)
        ay += 50

    # ===== RIGHT COLUMN =====
    # Card 1: Posture Score
    score = compute_posture_score(severity_score, posture_status, risk_level)
    rcard_h1 = 200
    _draw_card(canvas, rx, top_y, RIGHT_W, rcard_h1)
    _section_title(canvas, "POSTURE SCORE", rx + 12, top_y + 22)
    gauge_cx = rx + RIGHT_W // 2
    gauge_cy = top_y + 95
    _draw_circular_gauge(canvas, gauge_cx, gauge_cy, 45, score)

    # Description under gauge
    if score >= 75:
        desc = "You are maintaining"
        desc2 = "a good posture."
    elif score >= 50:
        desc = "Your posture needs"
        desc2 = "some improvement."
    else:
        desc = "Your posture needs"
        desc2 = "immediate correction."
    (dw, _), _ = cv2.getTextSize(desc, cv2.FONT_HERSHEY_SIMPLEX, 0.33, 1)
    cv2.putText(canvas, desc, (gauge_cx - dw // 2, top_y + 160), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (160, 160, 175), 1, cv2.LINE_AA)
    (dw2, _), _ = cv2.getTextSize(desc2, cv2.FONT_HERSHEY_SIMPLEX, 0.33, 1)
    cv2.putText(canvas, desc2, (gauge_cx - dw2 // 2, top_y + 178), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (160, 160, 175), 1, cv2.LINE_AA)

    # Card 2: Recommendations
    rcard_y2 = top_y + rcard_h1 + GAP
    rcard_h2 = main_h - rcard_h1 - GAP
    if rcard_h2 < 240:
        rcard_h2 = 240
    _draw_card(canvas, rx, rcard_y2, RIGHT_W, rcard_h2)
    _section_title(canvas, "RECOMMENDATIONS", rx + 12, rcard_y2 + 22)

    recs = [
        ("Sit Straight", "Keep your back straight", "and avoid slouching.", (0, 200, 0)),
        ("Monitor Height", "Keep your monitor", "at eye level.", (210, 170, 50)),
        ("Shoulder Relaxation", "Relax your shoulders", "and avoid hunching.", (0, 180, 220)),
        ("Take Breaks", "Take short breaks every", "30-40 minutes.", (200, 140, 40)),
    ]
    ry = rcard_y2 + 40
    for title, line1, line2, icon_col in recs:
        cv2.circle(canvas, (rx + 20, ry + 8), 10, icon_col, -1)
        cv2.putText(canvas, title, (rx + 38, ry + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, icon_col, 1, cv2.LINE_AA)
        cv2.putText(canvas, line1, (rx + 38, ry + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (170, 170, 185), 1, cv2.LINE_AA)
        cv2.putText(canvas, line2, (rx + 38, ry + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (170, 170, 185), 1, cv2.LINE_AA)
        ry += 52

    # ===== BOTTOM BAR =====
    bot_y = MARGIN + TITLE_H + GAP + main_h + GAP
    bar_items = 4
    bar_w = (canvas_w - MARGIN * 2 - GAP * (bar_items - 1)) // bar_items

    now = datetime.now()
    dur_h = int(duration_sec // 3600)
    dur_m = int((duration_sec % 3600) // 60)
    dur_s = int(duration_sec % 60)
    dur_str = f"{dur_h:02d}:{dur_m:02d}:{dur_s:02d}"

    if score >= 75:
        overall_msg, overall_sub, overall_col = "Good Job!", "Maintain your posture.", (0, 220, 0)
    elif score >= 50:
        overall_msg, overall_sub, overall_col = "Needs Work", "Try to improve posture.", (0, 220, 220)
    else:
        overall_msg, overall_sub, overall_col = "Poor Posture", "Correct immediately.", (0, 0, 255)

    bottom_cards = [
        ("ANALYSIS DURATION", dur_str, (200, 200, 215)),
        ("ANALYSIS DATE", now.strftime("%b %d, %Y"), (200, 200, 215)),
        ("TIME OF ANALYSIS", now.strftime("%I:%M %p"), (200, 200, 215)),
        ("OVERALL STATUS", overall_msg, overall_col),
    ]
    for i, (title, value, col) in enumerate(bottom_cards):
        bx = MARGIN + i * (bar_w + GAP)
        _draw_card(canvas, bx, bot_y, bar_w, BOTTOM_H - 5)
        cv2.putText(canvas, title, (bx + 10, bot_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (140, 140, 160), 1, cv2.LINE_AA)
        cv2.putText(canvas, value, (bx + 10, bot_y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
        if i == 3:
            cv2.putText(canvas, overall_sub, (bx + 10, bot_y + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (160, 160, 175), 1, cv2.LINE_AA)

    # Footer
    foot_y = canvas_h - 10
    cv2.putText(canvas, "Note: This analysis is based on computer vision and is for reference only.",
                (MARGIN, foot_y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 115), 1, cv2.LINE_AA)
    tag = "Stay healthy. Stay productive."
    (tw, _), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.33, 1)
    cv2.putText(canvas, tag, (canvas_w - MARGIN - tw, foot_y), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (120, 120, 135), 1, cv2.LINE_AA)

    return canvas
