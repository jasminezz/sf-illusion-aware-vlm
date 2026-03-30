import argparse
import base64
import difflib
import json
import os
import re
import textwrap
import time
import requests
from collections import Counter
from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps

from helper import Solver, run

# ============================================================
# Constants
# ============================================================
API_URL = ""
API_KEY = ""
MODEL = ""
NUM_VOTES = 5  # majority vote per image

# ============================================================
# Image preprocessing per illusion type
# ============================================================

def prep_bands(img):
    w, h = img.size
    s = max(int(w * 0.02), 4)
    L = img.crop((0, 0, s, h))
    R = img.crop((w - s, 0, w, h))
    sw = 140
    comp = Image.new("RGB", (sw * 2 + 20, h + 30), (128, 128, 128))
    comp.paste(L.resize((sw, h), Image.NEAREST), (5, 25))
    comp.paste(R.resize((sw, h), Image.NEAREST), (sw + 15, 25))
    comp = ImageEnhance.Color(comp).enhance(2.0)
    comp = ImageEnhance.Contrast(comp).enhance(1.5)
    d = ImageDraw.Draw(comp)
    d.text((5, 5), "LEFT EDGE", fill=(255, 255, 255))
    d.text((sw + 15, 5), "RIGHT EDGE", fill=(255, 255, 255))
    d.line([(sw + 10, 25), (sw + 10, h + 25)], fill=(255, 255, 0), width=2)
    return comp


def prep_color_boost(img):
    return ImageEnhance.Contrast(ImageEnhance.Color(img).enhance(2.0)).enhance(1.5)

def prep_red_isolate(img):
    a = np.array(img)
    mask = (a[:, :, 0] > 150) & (a[:, :, 1] < 100) & (a[:, :, 2] < 100)
    r = np.full_like(a, 255)
    r[mask] = [255, 0, 0]
    result_img = Image.fromarray(r)

    result_img = result_img.convert("RGBA")

    w, h = result_img.size
    overlay = Image.new("RGBA", (w, h), (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    blue_color = (0, 100, 200, 102)
    dash_length = 5
    gap_length = 5

    spacing_x = max(w // 20, 6)
    for x in range(spacing_x, w, spacing_x):
        y = 0
        while y < h:
            y_end = min(y + dash_length, h)
            draw.line([(x, y), (x, y_end)], fill=blue_color, width=1)
            y += dash_length + gap_length

    spacing_y = max(h // 20, 6)
    for y in range(spacing_y, h, spacing_y):
        x = 0
        while x < w:
            x_end = min(x + dash_length, w)
            draw.line([(x, y), (x_end, y)], fill=blue_color, width=1)
            x += dash_length + gap_length

    result_img = Image.alpha_composite(result_img, overlay)
    return result_img.convert("RGB")

def prep_orange_mirror(img):                                                                           
      a = np.array(img)                                                         
      orange = (a[:,:,0] > 180) & (a[:,:,1] > 80) & (a[:,:,1] < 200) & (a[:,:,2] < 80)                                                                        
                                                                                                                     
      clean = np.full_like(a, 255)                                           
      clean[orange] = a[orange]                                                 
      clean_img = Image.fromarray(clean)                                        
  
      w, h = clean_img.size                                                     
      mid = w // 2                                                           
                  
      left_half = clean_img.crop((0, 0, mid, h))
      right_half = clean_img.crop((mid, 0, w, h))                               
      left_mirrored = left_half.transpose(Image.FLIP_LEFT_RIGHT)             
      blended = Image.blend(left_mirrored, right_half, 0.5)                     
                                                                                                                          
      scale = 400 / max(blended.width, blended.height)                       
      if scale > 1:                                                             
          blended = blended.resize(                                             
              (int(blended.width * scale), int(blended.height * scale)),
              Image.LANCZOS                                                     
          )                                                                  
                                                                                
      final = Image.new("RGB", (blended.width, blended.height + 50), (255, 255, 255))
      final.paste(blended, (0, 40))
      return final

def prep_circle_isolate(img):
    a = np.array(img)

    dark_circles = (a[:, :, 0] < 100) & (a[:, :, 1] < 100) & (a[:, :, 2] < 100)

    clean = np.full_like(a, 255)
    clean[dark_circles] = a[dark_circles]
    clean_img = Image.fromarray(clean)

    w, h = clean_img.size
    mid = w // 2
    left_half = clean_img.crop((0, 0, mid, h))
    right_half = clean_img.crop((mid, 0, w, h))

    left_mirrored = ImageOps.mirror(left_half)
    blended = Image.blend(left_mirrored, right_half, 0.5)

    scale = 400 / max(blended.width, blended.height)
    if scale > 1:
        blended = blended.resize(
            (int(blended.width * scale), int(blended.height * scale)),
            Image.Resampling.LANCZOS,
        )

    final = Image.new("RGB", (blended.width, blended.height + 50), (255, 255, 255))
    final.paste(blended, (0, 40))

    return final


def prep_edge_outline(img):
    from scipy import ndimage

    a = np.array(img.convert("L"))
    h, w = a.shape
    mid = w // 2

    left = a[:, :mid]
    left_white = left > 200
    left_labeled, left_n = ndimage.label(left_white)
    border_labels = set(left_labeled[0, :]) | set(left_labeled[-1, :]) | \
                    set(left_labeled[:, 0]) | set(left_labeled[:, -1])
    border_labels.discard(0)
    left_inner = np.zeros_like(left_white)
    for i in range(1, left_n + 1):
        if i not in border_labels:
            left_inner |= (left_labeled == i)

    right = a[:, mid:]
    right_dark = right < 80
    right_labeled, right_n = ndimage.label(right_dark)
    border_labels = set(right_labeled[0, :]) | set(right_labeled[-1, :]) | \
                    set(right_labeled[:, 0]) | set(right_labeled[:, -1])
    border_labels.discard(0)
    right_inner = np.zeros_like(right_dark)
    for i in range(1, right_n + 1):
        if i not in border_labels:
            right_inner |= (right_labeled == i)

    comp = np.full((h, w, 3), 255, dtype=np.uint8)
    comp[:, :mid][left_inner] = [40, 40, 40]
    comp[:, mid:][right_inner] = [40, 40, 40]

    result_img = Image.fromarray(comp)

    left_half = result_img.crop((0, 0, mid, h))
    right_half = result_img.crop((mid, 0, w, h))

    left_mirrored = ImageOps.mirror(left_half)
    blended = Image.blend(left_mirrored, right_half, 0.5)

    scale = 400 / max(blended.width, blended.height)
    if scale > 1:
        blended = blended.resize(
            (int(blended.width * scale), int(blended.height * scale)),
            Image.Resampling.LANCZOS,
        )

    final = Image.new("RGB", (blended.width, blended.height + 50), (255, 255, 255))
    final.paste(blended, (0, 40))

    return final


def prep_irradiation(img):
    a = np.array(img.convert("RGB"))
    h, w = a.shape[:2]
    mid = w // 2

    left_half = img.crop((0, 0, mid, h))
    right_half = img.crop((mid, 0, w, h))

    left_inverted = ImageOps.invert(left_half.convert("RGB"))
    left_mirrored = ImageOps.mirror(left_inverted)

    blended = Image.blend(left_mirrored, right_half.convert("RGB"), 0.5)

    scale = 400 / max(blended.width, blended.height)
    if scale > 1:
        blended = blended.resize(
            (int(blended.width * scale), int(blended.height * scale)),
            Image.LANCZOS,
        )

    return blended


def prep_small_squares(img):
    a = np.array(img)
    h, w = a.shape[:2]
    mid = w // 2
    sq_size = min(h, mid) // 3
    ly = h // 2

    lx = mid // 2
    left_crop = img.crop((lx - sq_size // 2, ly - sq_size // 2,
                           lx + sq_size // 2, ly + sq_size // 2))
    rx = mid + mid // 2
    right_crop = img.crop((rx - sq_size // 2, ly - sq_size // 2,
                            rx + sq_size // 2, ly + sq_size // 2))
    sw = 140
    left_resized = left_crop.resize((sw, sw), Image.LANCZOS)
    right_resized = right_crop.resize((sw, sw), Image.LANCZOS)
    comp = Image.new('RGB', (sw * 2 + 30, sw + 35), (128, 128, 128))
    comp.paste(left_resized, (5, 25))
    comp.paste(right_resized, (sw + 25, 25))
    d = ImageDraw.Draw(comp)
    d.text((5, 5), 'LEFT', fill=(255, 255, 255))
    d.text((sw + 25, 5), 'RIGHT', fill=(255, 255, 255))
    d.line([(sw + 12, 25), (sw + 12, sw + 25)], fill=(255, 255, 0), width=2)
    return comp


def prep_distance(img):
    w, h = img.size
    label_area = img.crop((0, int(h * 0.75), w, h))
    stretched = label_area.resize((w * 3, int(h * 0.25 * 2)), Image.LANCZOS)
    return stretched


def prep_poggendorff(img):
    import math as _math
    a = np.array(img)
    h, w = a.shape[:2]

    red_mask = (a[:, :, 0] > 150) & (a[:, :, 1] < 80) & (a[:, :, 2] < 80)
    red_ys, red_xs = np.where(red_mask)

    if len(red_xs) < 10:
        return img

    slope, intercept = np.polyfit(red_xs, red_ys, 1)

    result = img.copy()
    draw = ImageDraw.Draw(result)

    x_start, x_end = 0, w - 1
    y_start = int(slope * x_start + intercept)
    y_end = int(slope * x_end + intercept)

    dash_len, gap_len = 6, 6
    total = dash_len + gap_len
    line_length = _math.sqrt((x_end - x_start) ** 2 + (y_end - y_start) ** 2)
    num_segments = int(line_length / total)

    for i in range(num_segments):
        t0 = (i * total) / line_length
        t1 = min((i * total + dash_len) / line_length, 1.0)
        sx = int(x_start + (x_end - x_start) * t0)
        sy = int(y_start + (y_end - y_start) * t0)
        ex = int(x_start + (x_end - x_start) * t1)
        ey = int(y_start + (y_end - y_start) * t1)
        draw.line([(sx, sy), (ex, ey)], fill=(255, 0, 0), width=2)

    return result


def prep_boundary_enhance(img):
    enhanced = ImageEnhance.Contrast(img).enhance(2.0)
    enhanced = ImageEnhance.Sharpness(enhanced).enhance(2.0)
    enhanced = ImageEnhance.Color(enhanced).enhance(1.5)
    return enhanced


def prep_vlines(img):
    img2 = img.copy()
    d = ImageDraw.Draw(img2)
    w, h = img2.size
    for f in [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]:
        d.line([(int(w * f), 0), (int(w * f), h)], fill=(255, 0, 0), width=1)
    return img2


def prep_zollner(img):
    w, h = img.size
    img_array = np.array(img, dtype=np.uint8)

    spacing = max(w // 16, 8)
    x_positions = [i * spacing for i in range(1, int(w / spacing))]

    red = (255, 0, 0)
    dash_len = 4
    gap_len = 4

    for x in x_positions:
        y = 0
        while y < h:
            y_end = min(y + dash_len, h)
            for py in range(y, y_end):
                for px in range(max(0, x-1), min(w, x+2)):
                    if py < h and px < w:
                        img_array[py, px] = red
            y += dash_len + gap_len

    return Image.fromarray(img_array)


def prep_squares(img):
    w, h = img.size
    img_array = np.array(img, dtype=np.uint8)

    spacing_x = w // 10
    spacing_y = h // 10

    red = (255, 0, 0)
    dash_len = 4
    gap_len = 4

    x_positions = [i * spacing_x for i in range(1, 10)]
    for x in x_positions:
        y = 0
        while y < h:
            y_end = min(y + dash_len, h)
            for py in range(y, y_end):
                for px in range(max(0, x-1), min(w, x+2)):  # 2px 宽
                    if py < h and px < w:
                        img_array[py, px] = red
            y += dash_len + gap_len

    y_positions = [i * spacing_y for i in range(1, 10)]
    for y in y_positions:
        x = 0
        while x < w:
            x_end = min(x + dash_len, w)
            for px in range(x, x_end):
                for py in range(max(0, y-1), min(h, y+2)):  # 2px 宽
                    if py < h and px < w:
                        img_array[py, px] = red
            x += dash_len + gap_len

    return Image.fromarray(img_array)


def prep_none(img):
    return img


# ============================================================
# Per-type config: (preprocessor, concise anti-illusion prompt)
# ============================================================

CONFIGS = {
    "Are the two vertical bands of the same color?": {
        "prep": prep_bands,
        "prompt": textwrap.dedent("""\
            Two edge strips (LEFT and RIGHT) are shown side-by-side on neutral gray, extracted from opposite sides of an image.

            Do NOT answer immediately. Follow these steps:

            Step 1: Describe the LEFT strip's color — name BOTH the hue AND brightness (e.g., "dark olive", "light purple", "medium teal").
            Step 2: Describe the RIGHT strip's color — same format.
            Step 3: Compare on TWO dimensions:
             - HUE: same or different? (e.g., both green vs one green one blue)
             - BRIGHTNESS: same or different? (e.g., both dark vs one dark one light)

            Decision rule:
            - Different HUE → answer 0
            - Same hue but clearly different BRIGHTNESS (one is noticeably lighter/darker) → answer 0
            - Same hue AND similar brightness (only subtle processing artifacts) → answer 1

            Output: {"result": 0} or {"result": 1}"""),
    },
    "Are the two rectangle the same color?": {
        "prep": prep_bands,
        "prompt": textwrap.dedent("""\
            Colors have been enhanced (2x saturation, 1.5x contrast) — any color difference in this image has been amplified and IS real.

            CRITICAL: Do NOT assume they are the same color.

            Before answering, you MUST independently describe each target:
            Step 1: Look at the LEFT target rectangles — describe their exact hue and brightness
            Step 2: Look at the RIGHT target rectangles — describe their exact hue and brightness
            Step 3: Compare your two descriptions. Any difference in hue, brightness, or saturation means they are different colors.

            After enhancement, trust your eyes — if you see a difference, it IS real.
            Truly identical → 1, Any difference → 0

            Output: {"result": 0} or {"result": 1}"""),
    },
    "Are the two small squares of the same color?": {
        "prep": prep_small_squares,
        "prompt": textwrap.dedent("""\
            Two color samples shown side-by-side on neutral gray background, extracted from the original image.

            Do NOT answer immediately. Follow these steps:

            Step 1: Focus on the LEFT sample only. Describe its color — name BOTH the hue AND brightness (e.g., "dark gray", "light gray", "medium warm gray").
            Step 2: Focus on the RIGHT sample only. Same description.
            Step 3: Compare on TWO dimensions:
             - HUE: same or different?
             - BRIGHTNESS: same or different?

            Decision rule:
            - Different HUE → answer 0
            - Same hue but clearly different BRIGHTNESS → answer 0
            - Same hue AND similar brightness → answer 1

            Output: {"result": 0} or {"result": 1}"""),
    },
    "Are the two circles of the same color?": {
        "prep": prep_color_boost,
        "prompt": textwrap.dedent("""\
            Two circles with noise patterns on different colored backgrounds. Colors have been enhanced.

            Do NOT answer immediately. Follow these steps:

            Step 1: Focus on the LEFT circle only. Describe its OVERALL color tone — name BOTH the hue AND brightness (e.g., "dark gray", "light gray", "medium warm gray", "cool blue-gray"). Ignore the noise/texture pattern, focus on the dominant color.
            Step 2: Focus on the RIGHT circle only. Same description.
            Step 3: Compare on TWO dimensions:
             - HUE: same or different?
             - BRIGHTNESS: same or different?

            Decision rule:
            - Different HUE → answer 0
            - Same hue but clearly different BRIGHTNESS → answer 0
            - Same hue AND similar brightness → answer 1

            Output: {"result": 0} or {"result": 1}"""),
    },
    "Are the two orange circles the same size?": {
        "prep": prep_orange_mirror,
        "prompt": textwrap.dedent("""\
            The left orange circle has been mirrored and overlaid onto the right orange circle's position.

            Look at the result:
            - If you see a single clean circle with uniform color and no visible edge ring → same size → 1
            - If you see a faint lighter ring or edge around the circle → different sizes → 0

            Output: {"result": 0} or {"result": 1}"""),
    },
    "Are the two solid circles the same size?": {
        "prep": prep_circle_isolate,
        "prompt": textwrap.dedent("""\
            You are seeing two overlapping solid circles.
            No need to overthink the surrounding elements; you can compare their sizes just by visual inspection.
            Check if there are obvious gray edges around the black solid circles. If they exist, it means their sizes are different; if not, their sizes are the same.
            Note that only when there are obvious gray edges around the black solid circle can it be concluded that their sizes are different; otherwise, their sizes are still the same.
            0 = Different sizes 1 = Same size

            Output:{"result": 0} or {"result": 1}"""),
    },
    "Are the two circles the same size?": {
        "prep": prep_circle_isolate,
        "prompt": textwrap.dedent("""\
            You are seeing two overlapping solid circles.
            No need to overthink the surrounding elements; you can compare their sizes just by visual inspection.
            Check if there are obvious gray edges around the black solid circles. If they exist, it means their sizes are different; if not, their sizes are the same.
            Note that only when there are obvious gray edges around the black solid circle can it be concluded that their sizes are different; otherwise, their sizes are still the same.
            0 = Different sizes 1 = Same size

            Output:{"result": 0} or {"result": 1}"""),
    },
    "Are the left white pentagon and the right black pentagon equal in size?": {
        "prep": prep_edge_outline,
        "prompt": textwrap.dedent("""\
            You are seeing two overlapping solid pentagons.
            No need to overthink the surrounding elements; compare their sizes visually.
            Check if there are obvious gray edges around the black solid pentagons.
            If they exist, it means their sizes are different; if not, their sizes are the same.
            Note that only when there are obvious gray edges around all five sides of the black solid pentagons can it be concluded that their sizes are different; if only some sides have gray edges, it means their sizes are still the same
            0 = Different sizes 1 = Same size

            Output:{"result": 0} or {"result": 1}"""),
    },
    "Are the left white square and the right black square equal in size?": {
        "prep": prep_irradiation,
        "prompt": textwrap.dedent("""\
            You are seeing two overlapping solid squares.
            No need to overthink the surrounding elements; compare their sizes visually.
            Check if there are obvious gray edges around the black solid square.
            If they exist, it means their sizes are different; if not, their sizes are the same.
            Note that only when there are obvious gray edges around all four sides of the black solid square can it be concluded that their sizes are different; if only some sides have gray edges, it means their sizes are still the same.
            0 = Different sizes 1 = Same size

            Output:{"result": 0} or {"result": 1}"""),
    },
    "Are the those red lines straight?": {
        "prep": prep_red_isolate,
        "prompt": textwrap.dedent("""\
            The image contains red vertical/horizontal lines on a blue grid.

            ASSESS STRAIGHTNESS:
            - If red lines look tilted or at an angle relative to blue grid → NOT STRAIGHT (0)
            - If red lines stay parallel to blue grid throughout → STRAIGHT (1)

            CRITICAL: Look for tilting patterns where lines angle inward or outward. This indicates bending.

            Output: {"result": 0} or {"result": 1}"""),
    },
    "Are the two vertical lines straight?": {
        "prep": prep_zollner,
        "prompt": textwrap.dedent("""\
            The image contains two vertical lines on a blue grid background.

            ASSESS STRAIGHTNESS:
            - If vertical lines appear mostly straight or nearly straight -> STRAIGHT (1)
            - If vertical lines show obvious bending or strong tilting -> NOT STRAIGHT (0)

            CRITICAL: Don't over-scrutinize minor variations. Accept slightly tilted lines as straight.
            Lines that stay roughly parallel to the grid = STRAIGHT.

            Output: {"result": 0} or {"result": 1}"""),
    },
    "Do the squares on the left and right have straight edges?": {
        "prep": prep_squares,
        "prompt": textwrap.dedent("""\
            The image shows left and right squares.

            ASSESS STRAIGHTNESS OF EDGES:
            - If the square edges appear mostly straight or nearly straight -> STRAIGHT (1)
            - If the square edges show obvious bending or curvature -> NOT STRAIGHT (0)

            CRITICAL: Don't over-scrutinize minor variations. Accept slightly curved edges as straight.
            Edges that stay roughly straight = STRAIGHT.

            Output: {"result": 0} or {"result": 1}"""),
    },
    "Are those vertical columns parallel?": {
        "prep": prep_vlines,
        "prompt": textwrap.dedent("""\
            CONTEXT: You are analyzing images from an optical illusion test set. In this dataset, most columns are intentionally tilted — truly parallel columns are the exception, not the norm. The diagonal stripe pattern inside the columns is specifically designed to make tilted columns LOOK straight. Your natural instinct will be to see them as parallel. Resist that instinct: assume tilt is present and look for evidence to CONFIRM parallelism, not to confirm tilt.

            METHOD — Focus ONLY on RED REFERENCE LINES. Completely ignore stripe patterns inside columns.

            STEP 1: EDGE-TO-RED-LINE GAP TEST (PRIMARY TEST — most reliable)
            For EACH of the 4 column edges (left column left/right edge, right column left/right edge):
              a) Find the nearest red reference line to that edge
              b) Look at the gap between that edge and the red line at the VERY TOP of the column
              c) Look at the gap at the VERY BOTTOM of the column
              d) If the gap is DIFFERENT — even very slightly — that edge is TILTED → answer 0

            CRITICAL DETAILS:
            - DARK/GRAY backgrounds hide subtle tilts. Be EXTRA careful: compare top vs bottom gaps very deliberately.
            - When a column appears narrow at one end and wider at the other, check if BOTH edges converge toward the same side — this confirms real tilt, not just stripe illusion.
            - Check ALL FOUR edges. One tilted edge is enough for answer 0.

            STEP 2: RED LINE OVERLAP TEST (use with caution — prone to false positives)
            Check if any red reference line passes THROUGH a column body:

            CRITICAL: DISTINGUISH REAL OVERLAP FROM STRIPE ILLUSION
            When a red line runs VERY CLOSE to a column edge, diagonal stripes create the APPEARANCE of overlap even when the column is perfectly straight. You MUST distinguish:

            REAL OVERLAP (confirms tilt → answer 0):
            - The red line PROGRESSIVELY enters the column from one end to the other
            - At one end (top or bottom), the red line is clearly outside the column
            - At the other end, the red line is clearly inside the column
            - The transition is gradual and continuous

            FAKE OVERLAP caused by stripe illusion (does NOT confirm tilt):
            - The red line appears to "weave in and out" along the column edge, alternating between inside and outside following the stripe pattern
            - The red line is equally close to the edge at top AND bottom — no progressive change
            - The apparent overlap is caused by black-and-white stripe segments creating visual gaps where the red line peeks through
            - This is the illusion at work — if the gap test in Step 1 shows NO change, this overlap is fake → answer 1
            - ESPECIALLY on narrow columns or when a red line hugs a column edge tightly, the stripe weaving can look deceptively smooth and continuous — mimicking progressive overlap. This is STILL fake if the Step 1 gap test shows no change from top to bottom.

            RULE: If you see apparent red line overlap but the Step 1 gap test shows consistent gaps at top and bottom, the overlap is a stripe illusion. Trust the gap test over apparent overlap.

            STEP 3: COLUMN CENTER POSITION TEST (supplementary)
            For each column, compare where the column CENTER sits horizontally at the very TOP vs the very BOTTOM.
            If the center has shifted left or right — even slightly — the column is tilted → answer 0.

            GRAY BACKGROUND GUIDANCE:
            On gray backgrounds, the low contrast between column edges and background makes subtle tilts very hard to see. Apply these extra checks:
            - Trace each column edge with your eye from top to bottom — does it drift left or right relative to the nearest red line?
            - Compare the column WIDTH at the very top vs very bottom. If one end is noticeably wider, the column is tilted.
            - When in doubt on gray backgrounds, lean toward 0.

            DECISION:
            - Answer 0 if ANY edge shows a gap change with its nearest red line (Step 1)
            - Answer 0 if a red line PROGRESSIVELY enters a column (real overlap, Step 2)
            - Answer 0 if ANY column center shifts horizontally (Step 3)
            - Answer 1 ONLY if you have positively confirmed that all four edges maintain constant gaps — parallelism must be proven, not assumed
            - IGNORE apparent red line "weaving" caused by stripe patterns — trust the gap test
            - On gray backgrounds, lean toward 0 when uncertain

            Output ONLY JSON: {"result": 0} or {"result": 1}"""),
    },
    "Are the two horizontal black lines of equal length?": {
        "prep": prep_none,
        "prompt": textwrap.dedent("""\
            This image shows two horizontal black lines inside a converging shape (either outlined V/A arrows or a filled gray triangle). One line is near the top (narrow region), one near the bottom (wide region).

            TASK: Are the two horizontal black lines of equal length?

            IMPORTANT — ILLUSION WARNING:
            The converging shape creates a powerful Ponzo illusion. The line near the NARROW TOP almost always LOOKS longer than the line near the WIDE BOTTOM, even when they are actually the same length. You MUST compensate for this bias.

            METHOD:
            - Mentally remove the surrounding shape entirely
            - Compare ONLY the horizontal extent of each black line
            - The line at the top appears longer due to the narrow context — discount this effect
            - Only answer NOT EQUAL (0) if the difference is truly dramatic and unmistakable at first glance
            - If the difference is subtle, ambiguous, or you are unsure → answer EQUAL (1)

            Output: {"result": 0} or {"result": 1}"""),
    },
    "Are the two black lines of equal length?": {
        "prep": prep_none,
        "prompt": textwrap.dedent("""\
            This image shows two horizontal black lines with circular endpoints. The upper line has inward-facing circles, the lower has outward-facing circles.

            TASK: Are the two black lines of equal length?

            IMPORTANT — MÜLLER-LYER ILLUSION:
            The outward circles make the LOWER line appear MUCH longer than it really is. This illusion is extremely powerful — a lower line that looks somewhat longer may actually be the SAME length as the upper line.

            METHOD:
            - Ignore the circles completely
            - Compare ONLY the horizontal line segments between the circle edges
            - The lower line ALWAYS looks longer due to the illusion — discount this heavily
            - Only answer NOT EQUAL (0) if the difference is truly dramatic and unmistakable at first glance
            - If the lower line looks moderately longer → this is likely the illusion → answer EQUAL (1)
            - If both lines look similar → answer EQUAL (1)

            Output: {"result": 0} or {"result": 1}"""),
    },
    "Are the red and black solid diagonal lines aligned?": {
        "prep": prep_poggendorff,
        "prompt": textwrap.dedent("""\
            A red dashed extension line has been drawn along the red line's direction through the black bar. This extension shows where the red line would continue if the bar were not there.

            TASK: Are the red and black solid diagonal lines aligned?

            METHOD:
            - Look at the red dashed extension line above the bar
            - Compare it with the black solid line above the bar
            - If the dashed red line overlaps or nearly overlaps with the black line → ALIGNED (1)
            - If there is a visible gap or clear separation between the dashed red line and the black line → NOT ALIGNED (0)

            CRITICAL: Focus on the area above the bar where both the dashed red extension and the black line are visible. Overlap means aligned, separation means not aligned.

            Output: {"result": 0} or {"result": 1}"""),
    },
    "Are the distances between the vertical markers labeled A–B and B–C equal?": {
        "prep": prep_distance,
        "prompt": textwrap.dedent("""\
            Three letters A, B, C are shown.
            Is B exactly centered between A and C?
            In other words, is the horizontal gap from A to B equal to the gap from B to C?

            Same → 1, Different → 0

            Output ONLY JSON: {"result": 0} or {"result": 1}"""),
    },
    "Is there an boundary in between every adjecent regions?": {
        "prep": prep_boundary_enhance,
        "prompt": textwrap.dedent("""\
            This image shows a triangular/pyramid shape made of horizontal color bands, transitioning from dark (top) to bright (bottom).

            TASK: Is there a boundary in between every adjacent region?

            HOW TO JUDGE:
            - Look at where one color band meets the next
            - Ask: is there a visible DIVIDING LINE between the two blocks, or do the colors just smoothly blend into each other?

            REAL BOUNDARY (1):
            - Adjacent blocks are SEPARATED by a visible thin edge/line
            - Each block looks like a distinct, independent rectangular strip
            - The color changes ABRUPTLY at the dividing line

            NO REAL BOUNDARY (0):
            - Adjacent blocks BLEND into each other with no visible dividing line
            - The colors flow smoothly from one shade to the next
            - Even though it LOOKS like there are separate blocks (Mach band illusion), the colors actually transition gradually without a sharp edge

            CRITICAL: The Mach band illusion makes smooth gradients look like they have boundaries. Look for an actual DIVIDING LINE — not just a color difference.

            Output: {"result": 0} or {"result": 1}"""),
    },
}


# ============================================================
# Claude API helpers
# ============================================================

def _img_to_b64(img: Image.Image, max_dim: int = 512) -> str:
    w, h = img.size
    if max(w, h) > max_dim:
        r = max_dim / max(w, h)
        img = img.resize((int(w * r), int(h * r)), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _call_model(b64: str, prompt_text: str, max_tokens: int = 500, max_retries: int = 3, retry_delay: int = 5) -> str:
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{b64}"
                        }
                    },
                    {"type": "text", "text": prompt_text},
                ],
            }
        ],
        "max_tokens": max_tokens,
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    last_error = None
    for attempt in range(max_retries):
        try:
            response = requests.post(API_URL, headers=headers, data=json.dumps(payload), timeout=60)

            if response.status_code != 200:
                last_error = f"HTTP {response.status_code}: {response.text[:200]}"
                print(f"    [API] attempt {attempt+1}/{max_retries} {last_error}", flush=True)
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                continue

            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0].get("message", {}).get("content", "")
                if content:
                    return content

            last_error = f"无法提取内容: {response.text[:200]}"
            print(f"    [API] attempt {attempt+1}/{max_retries} {last_error}", flush=True)
            if attempt < max_retries - 1:
                time.sleep(retry_delay)

        except requests.exceptions.Timeout:
            last_error = "请求超时 (60s)"
            print(f"    [API] attempt {attempt+1}/{max_retries} {last_error}", flush=True)
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
        except Exception as e:
            last_error = f"{type(e).__name__}: {str(e)[:80]}"
            print(f"    [API] attempt {attempt+1}/{max_retries} {last_error}", flush=True)
            if attempt < max_retries - 1:
                time.sleep(retry_delay)

    raise RuntimeError(f"_call_model failed after {max_retries} attempts: {last_error}")


def _parse_answer(text: str):
    # Format 1: <answer>0</answer> or <answer>1</answer>
    m = re.search(r"<answer>\s*([01])\s*</answer>", text)
    if m:
        return int(m.group(1))
    # Format 2: {"result": 0} or {"result": 1}
    m = re.search(r'"result"\s*:\s*([01])', text)
    if m:
        return int(m.group(1))
    # Fallback: last 0/1 near the word "answer"
    parts = text.lower().split("answer")
    if len(parts) > 1:
        m = re.search(r"([01])", parts[-1][:30])
        if m:
            return int(m.group(1))
    return None


# ============================================================
# MySolver implementation
# ============================================================

class MySolver(Solver):
    def solve(self, image_path: str, prompt: str) -> int:
        # Identify question type (first line of prompt)
        qtype = prompt.split("\n")[0].strip()

        # Load image
        img = Image.open(image_path).convert("RGB")

        if qtype not in CONFIGS:
            matches = difflib.get_close_matches(qtype, CONFIGS.keys(), n=1, cutoff=0.6)
            if matches:
                print(f"[warn] qtype not found exactly, using closest match: {matches[0]!r} (input: {qtype!r})")
                qtype = matches[0]
            else:
                print(f"[warn] qtype not found and no close match, using default: {qtype!r}")
        cfg = CONFIGS.get(qtype, {"prep": prep_none, "prompt": ""})
        preprocessor = cfg["prep"]
        anti_illusion = cfg["prompt"]

        # Preprocess
        processed = preprocessor(img) if preprocessor is not None else img
        b64 = _img_to_b64(processed)

        # Use anti-illusion prompt only (already contains full guidance)
        full_prompt = anti_illusion

        # Multi-vote ensemble
        votes: list[int] = []
        for _ in range(NUM_VOTES):
            try:
                resp = _call_model(b64, full_prompt)
                ans = _parse_answer(resp)
                if ans is not None:
                    votes.append(ans)
                time.sleep(0.3)
            except Exception:
                time.sleep(0.5)

        if not votes:
            return -1

        # Majority vote
        counts = Counter(votes)
        winner = counts.most_common(1)[0][0]
        return winner

    def model_info(self) -> dict:
        return {
            "model": MODEL,
            "parameters": {
                "temperature": 1.0,   # server default; no control via proxy
                "top_p": 1.0,
                "max_tokens": 500,
                "num_votes": NUM_VOTES,
            },
        }


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VQA Challenge Runner v8")
    parser.add_argument(
        "--input-csv", default="test.csv", help="Input CSV file path"
    )
    parser.add_argument(
        "--output-txt", default="predictions.txt", help="Output TXT file path"
    )
    parser.add_argument(
        "--output-json", default="model.json", help="Output JSON file path"
    )
    args = parser.parse_args()
    run(MySolver(), args.input_csv, args.output_txt, args.output_json)