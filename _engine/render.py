#!/usr/bin/env python3
"""Study Abroad Journal carousel renderer.
Usage: python3 render.py spec.json
Spec: {"variant":"A|B|C","out":"folder","pill":"GOING ABROAD","cover_title":"Text with *highlight*",
 "slides":[{"kicker":"01","title":"...","body":"..."} x4],"prompt":"Question for the comments?",
 "cta_title":"Text with *highlight*" (closing-slide headline, must be written fresh per carousel
 and tied to its topic — never reuse the same line across carousels),
 "photo_cover":"path.jpg" (C only),"photo_prompt":"path.jpg",
 "platform":"instagram" (default) | "tiktok" — tiktok drops the baked-in page dots (native on
 that platform) and reserves extra clearance at the bottom (caption/username/sound bar) and,
 on cover_c, the right edge (avatar/like/comment/bookmark/share icon column)}
Writes 01..07 at 1080x1350 -- .png on Instagram, .jpg on TikTok (TikTok Business
rejects image/png uploads outright)."""
import json, os, sys, re, math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps

HERE = os.path.dirname(os.path.abspath(__file__))
S = 2                      # supersampling factor
W, H = 1080, 1350
V1, V2 = (0xBC, 0x89, 0xFF), (0x5C, 0xE1, 0xE6)
# Deeper pair used ONLY for highlighted words on light backgrounds: the brand pair is too pale to read on white.
T1, T2 = (0x9B, 0x5C, 0xFF), (0x1F, 0xB5, 0xBC)
# B backgrounds: same hues, one step deeper, so white titles keep contrast at the turquoise end.
B1, B2 = (0xA2, 0x6C, 0xF7), (0x2F, 0xBC, 0xC8)
INK, BODY = (20, 20, 20), (61, 61, 61)
WHITE = (255, 255, 255)
_fonts = {}

# Decorative arcs: centres just outside two opposite corners, radii 110..290, stroke 9.
ARC_R = range(110, 320, 30)
ARC_OUTER = max(ARC_R) + 9 / 2
ARC_GAP = 40               # minimum clear space between any text and the outermost arc

def font(size, weight=800):
    k = (size, weight)
    if k not in _fonts:
        _fonts[k] = ImageFont.truetype(os.path.join(HERE, 'fonts', f'PlusJakartaSans-{weight}.ttf'), int(size * S))
    return _fonts[k]

def s(v): return int(round(v * S))

def is_tiktok(spec): return spec.get('platform') == 'tiktok'

# TikTok overlays its own UI on top of the image: username + caption + sound bar hug the
# bottom, and the avatar/like/comment/bookmark/share icon column hugs the right edge from
# roughly the vertical middle down to the bottom. Instagram has neither, so these only apply
# when spec['platform'] == 'tiktok'.
TIKTOK_BOTTOM_MARGIN = 0     # TikTok shows the caption BELOW the image (checked on a real post, 29/09)
TIKTOK_RIGHT_MARGIN = 150    # avatar + like/comment/bookmark column covers x > ~940 from mid-height down

def gradient(w, h, angle=90, stops=((0, V1), (1, V2))):
    """CSS-like linear gradient. angle in CSS degrees (90 = left->right)."""
    a = np.deg2rad(angle)
    dx, dy = np.sin(a), -np.cos(a)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    L = abs(w * dx) + abs(h * dy)
    t = ((xs - w / 2) * dx + (ys - h / 2) * dy) / max(L, 1) + 0.5
    t = np.clip(t, 0, 1)
    out = np.zeros((h, w, 3), np.float32)
    pos = [p for p, _ in stops]; cols = [np.array(c, np.float32) for _, c in stops]
    for i in range(3):
        out[..., i] = np.interp(t, pos, [c[i] for c in cols])
    return Image.fromarray(out.astype(np.uint8), 'RGB')

def light_bg():
    return gradient(s(W), s(H), 165, ((0, (0xF7, 0xF1, 0xFF)), (0.45, WHITE), (1, (0xEC, 0xFB, 0xFC))))

def brand_bg():
    return gradient(s(W), s(H), 150)

def deep_bg():
    return gradient(s(W), s(H), 150, ((0, B1), (1, B2)))

def arc_centres(mirror=False):
    tx, bx = (1110, -30) if mirror else (-30, 1110)
    return (tx, -30), (bx, 1380)

def arcs(img, c1, c2, op, mirror=False):
    lay = Image.new('RGBA', img.size, (0, 0, 0, 0)); d = ImageDraw.Draw(lay)
    (tx, ty), (bx, by) = arc_centres(mirror)
    for cx, cy, col in ((tx, ty, c1), (bx, by, c2)):
        for r in ARC_R:
            d.ellipse([s(cx - r), s(cy - r), s(cx + r), s(cy + r)], outline=col + (int(255 * op),), width=s(9))
    img.alpha_composite(lay)

def safe_band(xl, xr, mirror=False, top=0, bottom=H):
    """Vertical band [y0, y1] where a box spanning x in [xl, xr] clears both arc groups."""
    R = ARC_OUTER + ARC_GAP
    (tx, ty), (bx, by) = arc_centres(mirror)
    y0, y1 = top, bottom
    dx = max(xl - tx, tx - xr, 0)            # horizontal distance from top arc centre to the box
    if dx < R: y0 = max(y0, ty + math.sqrt(R * R - dx * dx))
    dx = max(xl - bx, bx - xr, 0)
    if dx < R: y1 = min(y1, by - math.sqrt(R * R - dx * dx))
    return y0, y1

def photo_fill(path):
    im = Image.open(path).convert('RGB')
    return ImageOps.fit(im, (s(W), s(H)), Image.LANCZOS).convert('RGBA')

def paste_logo(img, which, h, right=None, top=None, cx=None, shadow=False, left=None):
    lg = Image.open(os.path.join(HERE, 'assets', f'logo_{which}.png')).convert('RGBA')
    nh = s(h); nw = int(lg.width * nh / lg.height)
    lg = lg.resize((nw, nh), Image.LANCZOS)
    x = s(left) if left is not None else (s(W) - s(right) - nw if right is not None else s(cx) - nw // 2)
    y = s(top)
    if shadow:
        a = lg.split()[3].point(lambda v: int(v * 0.35))
        sh = Image.new('RGBA', lg.size, (0, 0, 0, 0)); sh.putalpha(a)
        pad = s(40); big = Image.new('RGBA', (nw + 2 * pad, nh + 2 * pad), (0, 0, 0, 0)); big.paste(sh, (pad, pad))
        big = big.filter(ImageFilter.GaussianBlur(s(9)))
        img.alpha_composite(big, (x - pad, y - pad + s(4)))
    img.alpha_composite(lg, (x, y))
    return nw, nh

def rrect(img, box, r, fill, shadow=False):
    if shadow:
        lay = Image.new('RGBA', img.size, (0, 0, 0, 0)); d = ImageDraw.Draw(lay)
        x0, y0, x1, y1 = box
        d.rounded_rectangle([s(x0), s(y0 + 24), s(x1), s(y1 + 24)], s(r), fill=(40, 20, 80, 46))
        img.alpha_composite(lay.filter(ImageFilter.GaussianBlur(s(30))))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([s(v) for v in box], s(r), fill=fill)

# ---------- text ----------
def tokens(text):
    out = []
    text = text.replace("'", "\u2019")   # typographic apostrophe: the straight one has wide side bearings in Plus Jakarta
    for part in re.split(r'(\*[^*]+\*)', text):
        if not part: continue
        hl = part.startswith('*') and part.endswith('*')
        for w in part.strip('*').split():
            out.append((w, hl))
    return out

def wrap(toks, f, maxw):
    lines, cur = [], []
    sp = f.getlength(' ')
    for t in toks:
        trial = cur + [t]
        wd = sum(f.getlength(w) for w, _ in trial) + sp * (len(trial) - 1)
        if cur and wd > maxw: lines.append(cur); cur = [t]
        else: cur = trial
    if cur: lines.append(cur)
    return lines

def balanced(toks, f, maxw):
    """Pick the wrap that reads best: fewest lines, even line lengths, no one-word last line."""
    cands = {}
    w = maxw
    while w > maxw * 0.4:
        lines = wrap(toks, f, w)
        key = tuple(len(l) for l in lines)
        cands.setdefault(key, lines)
        w -= s(8)
    n0 = min(len(k) for k in cands)
    best, best_cost = None, None
    for lines in cands.values():
        if len(lines) > n0 + 1: continue
        ws = [line_w(l, f) for l in lines]; m = max(ws)
        cost = 0.5 * (len(lines) - n0)
        if len(lines) > 1 and len(lines[-1]) == 1 and len(toks) > 2: cost += 1.0
        if len(lines) > 2: cost += (max(ws[:-1]) - min(ws[:-1])) / m
        cost += 0.3 * (1 - ws[-1] / m) if len(lines) > 1 else 0
        if best_cost is None or cost < best_cost: best, best_cost = lines, cost
    return best

def line_w(line, f): return sum(f.getlength(w) for w, _ in line) + f.getlength(' ') * (len(line) - 1)

def text_block(size, weight, lh, text, maxw, balance=True):
    f = font(size, weight); toks = tokens(text)
    lines = balanced(toks, f, s(maxw)) if balance else wrap(toks, f, s(maxw))
    return {'f': f, 'lines': lines, 'lh': s(size * lh), 'h': s(size * lh) * len(lines), 'size': size}

def block_w(blk):
    """Widest line of a block, in final (1x) px."""
    return max((line_w(l, blk['f']) for l in blk['lines']), default=0) / S

def draw_block(img, blk, x, y, color, hl='gradient', align='left', boxw=None, tracking=0, glow=False):
    if glow:
        lay = Image.new('RGBA', img.size, (0, 0, 0, 0))
        draw_block(lay, blk, x, y + s(3), (40, 20, 90, 70), 'plain', align, boxw)
        img.alpha_composite(lay.filter(ImageFilter.GaussianBlur(s(10))))
    d = ImageDraw.Draw(img); f = blk['f']; sp = f.getlength(' ')
    asc, desc = f.getmetrics(); off = (blk['lh'] - (asc + desc)) // 2
    for i, line in enumerate(blk['lines']):
        lw = line_w(line, f)
        cx = x + ((s(boxw) - lw) / 2 if align == 'center' else 0)
        cy = y + i * blk['lh'] + off
        i2 = 0
        while i2 < len(line):
            w, h = line[i2]
            if h and hl == 'gradient':
                # consecutive highlighted words share ONE gradient (no restart per word)
                j = i2
                while j + 1 < len(line) and line[j + 1][1]: j += 1
                phrase = ' '.join(t for t, _ in line[i2:j + 1]); ww = f.getlength(phrase)
                m = Image.new('L', (int(ww) + s(20), asc + desc + s(10)), 0)
                ImageDraw.Draw(m).text((0, 0), phrase, font=f, fill=255)
                g = gradient(m.width, m.height, 90, ((0, T1), (1, T2))).convert('RGBA'); g.putalpha(m)
                img.alpha_composite(g, (int(cx), int(cy)))
                cx += ww + sp; i2 = j + 1
            else:
                ww = f.getlength(w)
                d.text((cx, cy), w, font=f, fill=color)
                cx += ww + sp; i2 += 1
    return blk['h']

def fit(blocks_fn, avail, maxs=1.0):
    k = maxs
    while True:
        blocks = blocks_fn(k)
        total = sum(b['h'] if isinstance(b, dict) else s(b) for b in blocks)
        if total <= s(avail) or k < 0.55: return blocks, total
        k *= 0.94

def place(total, y0, y1, centre):
    """Centre the stack on `centre` (1x px), then clamp it inside [y0, y1]. Returns top y in 2x px."""
    y = s(centre) - total / 2
    y = max(y, s(y0))
    y = min(y, s(y1) - total)
    return y

def pill(img, text, x, y, size, style):
    f = font(size, 800); tr = size * 0.12
    tw = sum(f.getlength(c) for c in text) + s(tr) * (len(text) - 1)
    ph, pw = s(size * 0.55) * 2 + s(size * 1.25), tw + s(size * 1.08) * 2
    d = ImageDraw.Draw(img)
    if style == 'gradient':
        g = gradient(int(pw), int(ph), 90).convert('RGBA')
        m = Image.new('L', g.size, 0); ImageDraw.Draw(m).rounded_rectangle([0, 0, g.width - 1, g.height - 1], g.height // 2, fill=255)
        g.putalpha(m); img.alpha_composite(g, (int(x), int(y))); col = WHITE
    elif style == 'outline':
        d.rounded_rectangle([x, y, x + pw, y + ph], ph / 2, fill=WHITE, outline=INK, width=s(2)); col = INK
    else:
        d.rounded_rectangle([x, y, x + pw, y + ph], ph / 2, fill=WHITE); col = INK
    cx = x + s(size * 1.08); asc, desc = f.getmetrics(); cy = y + (ph - asc - desc) / 2
    for c in text:
        d.text((cx, cy), c, font=f, fill=col); cx += f.getlength(c) + s(tr)
    return ph

def pill_w(text, size):
    f = font(size, 800); tr = size * 0.12
    return (sum(f.getlength(c) for c in text) + s(tr) * (len(text) - 1) + s(size * 1.08) * 2) / S

def arrow(d, x, y, w, col, down=False):
    t = max(2, int(w * 0.12))
    if down:
        d.line([x + w / 2, y, x + w / 2, y + w], fill=col, width=t)
        d.line([x + w * 0.15, y + w * 0.6, x + w / 2, y + w, x + w * 0.85, y + w * 0.6], fill=col, width=t, joint='curve')
    else:
        d.line([x, y + w / 2, x + w, y + w / 2], fill=col, width=t)
        d.line([x + w * 0.6, y + w * 0.15, x + w, y + w / 2, x + w * 0.6, y + w * 0.85], fill=col, width=t, joint='curve')

def button(img, label, x, y, size, padv, padh, bg, fg, arrow_after=False, right_align=False):
    f = font(size, 600); tw = f.getlength(label); aw = s(size * 0.85) if arrow_after else 0; gap = s(size * 0.35) if arrow_after else 0
    bw, bh = tw + gap + aw + s(padh) * 2, s(padv) * 2 + s(size * 1.3)
    if right_align: x = x - bw
    d = ImageDraw.Draw(img); d.rounded_rectangle([x, y, x + bw, y + bh], bh / 2, fill=bg)
    asc, desc = f.getmetrics(); ty = y + (bh - asc - desc) / 2
    d.text((x + s(padh), ty), label, font=f, fill=fg)
    if arrow_after: arrow(d, x + s(padh) + tw + gap, y + (bh - aw) / 2, aw, fg)
    return bw, bh

def dots(img, i, col, bottom):
    # Disabled: Instagram and TikTok both draw their own page dots under the post.
    return
    d = ImageDraw.Draw(img); widths = [44 if k == i else 14 for k in range(1, 8)]
    total = sum(widths) + 12 * 6; x = (W - total) / 2; y = H - bottom - 14
    for k, w in enumerate(widths, 1):
        a = 255 if k == i else 89
        lay = Image.new('RGBA', img.size, (0, 0, 0, 0))
        ImageDraw.Draw(lay).rounded_rectangle([s(x), s(y), s(x + w), s(y + 14)], s(7), fill=col + (a,))
        img.alpha_composite(lay); x += w + 12

def finish(img, path):
    # TikTok Business rejects image/png outright ("use image/jpeg, image/jpg or
    # image/webp instead") -- JPEG for every slide keeps one code path instead of
    # branching finish() by platform, and these are flattened RGB frames anyway.
    rgb = img.convert('RGB').resize((W, H), Image.LANCZOS)
    if path.endswith('.png'):
        rgb.save(path, 'PNG', optimize=True)
    else:
        rgb.save(path, 'JPEG', quality=95, optimize=True)

# ---------- slides ----------
def ab_base(v, mirror=False):
    img = (light_bg() if v == 'A' else deep_bg()).convert('RGBA')
    arcs(img, V1 if v == 'A' else WHITE, V2 if v == 'A' else WHITE, 0.3 if v == 'A' else 0.28, mirror)
    return img

def cover_ab(v, spec):
    # Instagram's grid overlays its carousel icon in the TOP-RIGHT corner, so the logo lives
    # top-left and the arcs are mirrored (top-right / bottom-left) to keep that corner clear.
    img = ab_base(v, mirror=True); ink = INK if v == 'A' else WHITE
    paste_logo(img, 'gradient' if v == 'A' else 'white', 150, left=80, top=72)
    bbg, bfg = (INK, WHITE) if v == 'A' else (WHITE, INK)
    btn_h = 22 * 2 + 30 * 1.3
    if is_tiktok(spec):
        # left, lifted above TikTok's caption zone (clear of the bottom-left arcs at that height)
        by = 1200 - TIKTOK_BOTTOM_MARGIN - btn_h
        bw, bh = button(img, 'Swipe', s(88), s(by), 30, 22, 40, bbg, bfg, arrow_after=True)
    else:
        by = 1200 - btn_h
        bw, bh = button(img, 'Swipe', s(W - 88), s(by), 30, 22, 40, bbg, bfg, arrow_after=True, right_align=True)
    top, bottom = 72 + 150 + 40, by - 20
    ph = s(26 * 0.55) * 2 + s(26 * 1.25)
    pw = pill_w(spec['pill'], 26)
    k = 1.0
    while True:
        t = text_block(124 * k, 800, 1.02, spec['cover_title'], 904)
        total = ph + s(36) + t['h']
        y0, y1 = safe_band(88, 88 + max(pw, block_w(t)), True, top, bottom)
        if total <= s(y1 - y0) or k < 0.5: break
        k *= 0.95
    y = place(total, y0, y1, (top + bottom) / 2)
    pill(img, spec['pill'], s(88), y, 26, 'outline' if v == 'A' else 'white')
    draw_block(img, t, s(88), y + ph + s(36), ink, 'gradient' if v == 'A' else 'plain', glow=(v == 'B'))
    return img

# Content slides share ONE layout: kicker always at the same y, same scale on every slide,
# so the title sits at the same level while swiping. Scale is picked once for the whole carousel.
AB_TOP, AB_BOTTOM = 290, 1060      # clear of the arcs on both mirrored and non-mirrored slides
C_TOP, C_BOTTOM = 200, 1170        # inside the white card, above the dots
# Text column ends at x<=920 on every platform so it stays clear of TikTok's avatar/like/comment
# column (x>~940 from mid-height down), while the card itself is identical on IG and TikTok.
C_KICKER_MAXW, C_TITLE_MAXW, C_BODY_MAXW = 760, 792, 780   # must match the maxw args in c_blocks

def ab_bottom(spec):
    """AB_BOTTOM already bakes in Instagram's own dots-row buffer (H - 290), which has
    nothing to do with TikTok's UI. Subtracting TIKTOK_BOTTOM_MARGIN from it double-reserves
    space no one asked for: on TikTok, dots aren't drawn at all, so the real limit is just the
    image edge minus that platform's own clearance (matching c_card_base's card_bottom)."""
    return (H - 64 - TIKTOK_BOTTOM_MARGIN) if is_tiktok(spec) else AB_BOTTOM

def c_bottom(spec): return C_BOTTOM - (TIKTOK_BOTTOM_MARGIN if is_tiktok(spec) else 0)

def rm(spec): return TIKTOK_RIGHT_MARGIN if is_tiktok(spec) else 0

def ab_blocks(v, sl, k, spec):
    m = rm(spec)
    return [text_block(190 * k, 800, 1.0, ('*' + sl['kicker'] + '*') if v == 'A' else sl['kicker'], 800 - m, False), 36 * k,
            text_block(104 * k, 800, 1.05, sl['title'], 904), 36 * k,   # title sits above TikTok's icon column: full width
            text_block(54 * k, 400, 1.36, sl['body'], 860 - m)]

def c_blocks(sl, k, spec):
    m = rm(spec)
    return [text_block(190 * k, 800, 1.0, '*' + sl['kicker'] + '*', 760 - m, False), 32 * k,
            text_block(104 * k, 800, 1.05, sl['title'], C_TITLE_MAXW - m), 32 * k,
            text_block(54 * k, 400, 1.36, sl['body'], 780 - m)]

def stack_h(blocks): return sum(b['h'] if isinstance(b, dict) else s(b) for b in blocks)

def common_scale(spec, v):
    """Largest scale at which every content slide fits its fixed band, clears the arcs,
    and keeps every line narrower than its column (a single long/unbreakable word can't
    be wrapped, so it must be caught here or it will overflow the card)."""
    k = 1.0
    while k > 0.45:
        ok = True
        for idx in range(2, 6):
            sl = spec['slides'][idx - 2]
            if v == 'C':
                bl = c_blocks(sl, k, spec)
                dicts = [b for b in bl if isinstance(b, dict)]
                margin = rm(spec)
                maxws = (C_KICKER_MAXW - margin, C_TITLE_MAXW - margin, C_BODY_MAXW - margin)
                if any(block_w(b) > m for b, m in zip(dicts, maxws)) or stack_h(bl) > s(c_bottom(spec) - C_TOP): ok = False; break
            else:
                bl = ab_blocks(v, sl, k, spec)
                wmax = max(block_w(b) for b in bl if isinstance(b, dict))
                y0, y1 = safe_band(88, 88 + wmax, idx % 2 == 0, 0, ab_bottom(spec))
                # a single long word (e.g. MEMORIES as kicker) can't wrap: shrink until it fits W - 2*88
                if y0 > AB_TOP or stack_h(bl) > s(y1 - AB_TOP) or wmax > W - 176: ok = False; break
        if ok: return k
        k *= 0.97
    return k

def common_top_ab(spec, v, k):
    """One shared kicker y for every content slide, so titles line up while swiping. Using
    each slide's own stack height here (as content_ab used to) makes the kicker drift up or
    down slide to slide whenever line-wrapping differs — most visible on TikTok, where the
    narrower right-margin column wraps some slides' bodies onto an extra line but not others."""
    y1s, totals = [], []
    for idx in range(2, 6):
        sl = spec['slides'][idx - 2]
        blocks = ab_blocks(v, sl, k, spec)
        wmax = max(block_w(b) for b in blocks if isinstance(b, dict))
        _, y1 = safe_band(88, 88 + wmax, idx % 2 == 0, 0, ab_bottom(spec))
        y1s.append(y1); totals.append(stack_h(blocks))
    return place(max(totals), AB_TOP, min(y1s), (AB_TOP + min(y1s)) / 2)

def content_ab(v, spec, idx, k, top):
    mirror = (idx % 2 == 0)
    sl = spec['slides'][idx - 2]; img = ab_base(v, mirror=mirror)
    ink = INK if v == 'A' else WHITE; body = BODY if v == 'A' else WHITE
    blocks = ab_blocks(v, sl, k, spec)
    y = top
    for b in blocks:
        if isinstance(b, dict):
            col = ink if b is not blocks[4] else body
            y += draw_block(img, b, s(88), y, col, 'gradient' if v == 'A' else 'plain', glow=(v == 'B'))
        else: y += s(b)
    return img

def prompt_slide(spec):
    # Same card-covers-the-bottom concern as cover_c, though the card here is usually smaller
    # (roughly the bottom third) -- still worth checking photo_prompt's subject isn't buried.
    img = photo_fill(spec['photo_prompt'])
    paste_logo(img, 'white', 170, left=64, top=64, shadow=True)   # top-right = TikTok's 'n/7' counter
    inner_w = W - 128 - 112 - rm(spec)
    q = text_block(56, 800, 1.15, spec['prompt'], inner_w)
    ph = s(22 * 0.55) * 2 + s(22 * 1.25)
    f30 = font(30, 600); small_h = s(30 * 1.3)
    card_h = s(48) * 2 + ph + s(20) + q['h'] + s(20) + small_h
    y1 = s(H - 64 - (TIKTOK_BOTTOM_MARGIN if is_tiktok(spec) else 0)); y0 = y1 - card_h
    rrect(img, (64, y0 / S, W - 64, y1 / S), 40, WHITE, shadow=True)
    x = s(64 + 56); y = y0 + s(48)
    pill(img, 'TELL US IN THE COMMENTS', x, y, 22, 'gradient'); y += ph + s(20)
    draw_block(img, q, x, y, INK); y += q['h'] + s(20)
    d = ImageDraw.Draw(img); label = 'Drop your answer below'
    d.text((x, y), label, font=f30, fill=(107, 107, 107))
    arrow(d, x + f30.getlength(label) + s(12), y + s(6), s(26), (107, 107, 107), down=True)
    return img

def cta_ab(v, spec):
    img = ab_base(v); col = INK
    bodyc = BODY if v == 'A' else INK
    m = rm(spec)
    def mk(k):
        return [s(340 * k), 48 * k, text_block(92 * k, 800, 1.08, spec['cta_title'], 904 - m), 48 * k,
                text_block(46 * k, 400, 1.35, 'The guided journal to capture every moment.', 760 - m), 18 * k,
                text_block(58 * k, 800, 1.1, 'Write. Evolve.', 904 - m, False), 18 * k,
                text_block(58 * k, 800, 1.1, 'Retrospect on yourself.', 904 - m, False), 48 * k, s(30 * 2 + 36 * 1.3)]
    safe_h = H - (TIKTOK_BOTTOM_MARGIN if is_tiktok(spec) else 0)
    blocks, total = fit(lambda k: [b if isinstance(b, dict) else (b / S if i in (0, 10) else b) for i, b in enumerate(mk(k))], safe_h - 160)
    # Centre on the true canvas middle, then clamp to the TikTok-safe band: centring within
    # [0, safe_h] alone leaves the reserved margin as pure extra weight at the very bottom,
    # since it sits outside the centred band entirely (the same imbalance content_ab had).
    y = place(total, 0, safe_h, H / 2)
    for i, b in enumerate(blocks):
        if i == 0:
            _, lh = paste_logo(img, 'gradient' if v == 'A' else 'white', b, cx=W / 2, top=y / S); y += lh
        elif i == 10:
            bw = font(36, 600).getlength('Link in bio') + s(120)
            button(img, 'Link in bio', s(W / 2) - bw / 2, y, 36, 30, 60, INK, WHITE)
        elif isinstance(b, dict):
            c = bodyc if i == 4 else col
            if v == 'B' and i in (4, 6, 8): c = WHITE   # B: on the gradient only the headline stays black
            y += draw_block(img, b, 0, y, c, 'gradient' if v == 'A' else 'plain', 'center', W)
        else: y += s(b)
    return img

def cover_c(spec):
    # The white card below covers roughly the bottom 40-45% of the frame (more on TikTok).
    # photo_cover's subject must read clearly in the TOP portion of the source image -- a
    # subject sitting in the lower half gets hidden almost entirely, leaving only unreadable
    # sky/background above the card. Check where the subject sits before picking a photo here.
    img = photo_fill(spec['photo_cover'])
    paste_logo(img, 'white', 170, left=64, top=64, shadow=True)   # top-right is under Instagram's grid carousel icon
    inner = W - 128 - 112 - rm(spec)
    blocks, _ = fit(lambda k: [text_block(104 * k, 800, 1.02, spec['cover_title'], inner)], 520)
    t = blocks[0]; ph = s(24 * 0.55) * 2 + s(24 * 1.25); btn_h = s(18 * 2 + 26 * 1.3)
    card_h = s(48) + ph + s(24) + t['h'] + s(32) + s(2) + s(28) + btn_h + s(28)
    y1 = s(H - 64 - (TIKTOK_BOTTOM_MARGIN if is_tiktok(spec) else 0)); y0 = y1 - card_h
    rrect(img, (64, y0 / S, W - 64, y1 / S), 40, WHITE, shadow=True)
    x = s(120); y = y0 + s(48)
    pill(img, spec['pill'], x, y, 24, 'gradient'); y += ph + s(24)
    draw_block(img, t, x, y, INK); y += t['h'] + s(32)
    if is_tiktok(spec):
        # left-aligned, matching the pill/title: a right-aligned button here would either
        # collide with TikTok's avatar bubble or, pulled clear of it, leave an ugly dead gap
        rule_edge = W - 120 - TIKTOK_RIGHT_MARGIN
        ImageDraw.Draw(img).rectangle([x, y, s(rule_edge), y + s(2)], fill=(0xEE, 0xEA, 0xF5)); y += s(2) + s(28)
        button(img, 'Swipe', x, y, 26, 18, 34, INK, WHITE, arrow_after=True)
    else:
        ImageDraw.Draw(img).rectangle([x, y, s(W - 120), y + s(2)], fill=(0xEE, 0xEA, 0xF5)); y += s(2) + s(28)
        button(img, 'Swipe', s(W - 120), y, 26, 18, 34, INK, WHITE, arrow_after=True, right_align=True)
    return img

def c_card_base(idx, spec, box=None):
    img = brand_bg().convert('RGBA'); arcs(img, WHITE, WHITE, 0.28, mirror=(idx % 2 == 0))
    card_bottom = H - 64 - (TIKTOK_BOTTOM_MARGIN if is_tiktok(spec) else 0)
    rrect(img, box or (64, 64, W - 64, card_bottom), 40, WHITE, shadow=True)
    return img

# Content cards hug their text instead of filling the frame (a full-height card left a big empty
# white block under short bodies, worst on TikTok). Every content slide gets the SAME card, sized
# on the tallest stack, so the card and the kicker don't jump while swiping.
C_PAD_TOP = 88
def c_pad_bottom(spec): return 88 if is_tiktok(spec) else 64 + 14 + 44   # IG: room for the dots row

def common_top_c(spec, k):
    """Shared kicker y across all content slides — see common_top_ab for why per-slide
    centering causes titles to drift between slides."""
    total = max(stack_h(c_blocks(spec['slides'][idx - 2], k, spec)) for idx in range(2, 6)) / S
    # Same card (size AND position) on Instagram and TikTok: laid out as on Instagram, then only
    # nudged up on TikTok if it would dip into the caption zone.
    card_h = C_PAD_TOP + total + C_PAD_TOP
    y0 = 64 + max(0, (H - 128 - card_h) / 2)
    hi = H - 64 - (TIKTOK_BOTTOM_MARGIN if is_tiktok(spec) else 0)
    y0 = max(64, min(y0, hi - card_h))
    return (64, y0, W - 64, min(hi, y0 + card_h))

def content_c(spec, idx, k, box, dots_on=True):
    sl = spec['slides'][idx - 2]; img = c_card_base(idx, spec, box)
    blocks = c_blocks(sl, k, spec)
    y = s(box[1] + C_PAD_TOP)
    for b in blocks:
        if isinstance(b, dict): y += draw_block(img, b, s(128), y, BODY if b is blocks[4] else INK)
        else: y += s(b)
    return img

def cta_c(spec):
    img = c_card_base(7, spec)
    m = rm(spec)
    def mk(k):
        return [300 * k, 44 * k, text_block(84 * k, 800, 1.08, spec['cta_title'], 824 - m), 44 * k,
                text_block(42 * k, 400, 1.35, 'The guided journal to capture every moment.', 700 - m), 16 * k,
                text_block(54 * k, 800, 1.1, 'Write. Evolve.', 824 - m, False), 16 * k,
                text_block(54 * k, 800, 1.1, 'Retrospect on yourself.', 824 - m, False), 44 * k, 28 * 2 + 34 * 1.3]
    raw = mk(1)
    def tot(bl): return sum(b['h'] if isinstance(b, dict) else s(b) for b in bl)
    card_bottom = H - 64 - (TIKTOK_BOTTOM_MARGIN if is_tiktok(spec) else 0)
    safe_h = H - (TIKTOK_BOTTOM_MARGIN if is_tiktok(spec) else 0)
    k = 1.0; bl = raw
    while tot(bl) > s(safe_h - 128 - 160) and k > 0.6: k *= 0.94; bl = mk(k)
    # Centre within the card's own bounds (64..card_bottom), not [0, safe_h] — same fix as
    # cta_ab, so the TikTok-shrunk card doesn't dump all its extra bottom margin as visible gap.
    y = place(tot(bl), 64, card_bottom, (64 + card_bottom) / 2)
    for i, b in enumerate(bl):
        if i == 0:
            _, lh = paste_logo(img, 'gradient', b, cx=W / 2, top=y / S); y += lh
        elif i == 10:
            bw = font(34, 600).getlength('Link in bio') + s(112)
            button(img, 'Link in bio', s(W / 2) - bw / 2, y, 34, 28, 56, INK, WHITE)
        elif isinstance(b, dict):
            y += draw_block(img, b, 0, y, BODY if i == 4 else INK, 'gradient', 'center', W)
        else: y += s(b)
    return img

def main(path):
    spec = json.load(open(path)); v = spec['variant']; out = spec['out']; os.makedirs(out, exist_ok=True)
    k = common_scale(spec, v)
    if v == 'C':
        # Content cards are identical on both platforms (same size, text and position): TikTok
        # reuses the Instagram layout and only drops the dots.
        ig = dict(spec, platform='instagram'); kc = common_scale(ig, 'C')
        box = common_top_c(ig, kc)
        slides = [cover_c(spec)] + [content_c(ig, i, kc, box, dots_on=not is_tiktok(spec)) for i in range(2, 6)] + [prompt_slide(spec), cta_c(spec)]
    else:
        top = common_top_ab(spec, v, k)
        slides = [cover_ab(v, spec)] + [content_ab(v, spec, i, k, top) for i in range(2, 6)] + [prompt_slide(spec), cta_ab(v, spec)]
    ext = 'jpg' if is_tiktok(spec) else 'png'
    for i, im in enumerate(slides, 1): finish(im, os.path.join(out, f'{i:02d}.{ext}'))
    print('ok', out)

if __name__ == '__main__':
    main(sys.argv[1])
