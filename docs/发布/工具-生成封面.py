# -*- coding: utf-8 -*-
"""生成发布封面（深色科技风，与应用工作台一致）。"""
import os
from PIL import Image, ImageDraw, ImageFont

ROOT = r"D:\BaiduSyncdisk\wmj_work\SPXH"
FIG = os.path.join(ROOT, "docs", "发布", "配图")
OUT = os.path.join(ROOT, "docs", "发布", "封面")
os.makedirs(OUT, exist_ok=True)

F_BOLD = r"C:\Windows\Fonts\msyhbd.ttc"
F_REG  = r"C:\Windows\Fonts\msyh.ttc"
F_MONO = r"C:\Windows\Fonts\consolab.ttf"

BG, GRID = (11, 15, 20), (22, 31, 41)
ACCENT, ACCENT2 = (94, 234, 212), (125, 211, 252)
WHITE, GREY = (234, 242, 248), (140, 157, 175)

_cache = {}
def font(path, size):
    k = (path, size)
    if k not in _cache:
        _cache[k] = ImageFont.truetype(path, size)
    return _cache[k]

def is_ascii(s):
    return all(ord(c) < 128 for c in s)

def mixed_fonts(text, mono_size, cjk_size):
    """按字符类型拆段，返回 [(seg, font, is_mono)]"""
    segs, cur, cur_a = [], "", None
    for ch in text:
        a = ord(ch) < 128
        if cur_a is None or a == cur_a:
            cur += ch; cur_a = a
        else:
            segs.append((cur, cur_a)); cur, cur_a = ch, a
    if cur:
        segs.append((cur, cur_a))
    return [(s, font(F_MONO if a else F_BOLD, mono_size if a else cjk_size), a) for s, a in segs]

def draw_bg(w, h):
    img = Image.new("RGB", (w, h), BG)
    d = ImageDraw.Draw(img)
    for y in range(0, h, 64):
        d.line([(0, y), (w, y)], fill=GRID, width=1)
    for x in range(0, w, 64):
        d.line([(x, 0), (x, h)], fill=GRID, width=1)
    for i in range(0, 300):
        a = int(20 * (1 - i / 300))
        d.line([(0, i), (w, i)], fill=(BG[0] + a // 3, BG[1] + a, BG[2] + a))
    return img

def paste_round(base, im, xy, radius=14, border=(40, 56, 72), bw=2):
    w, h = im.size
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, w - 1, h - 1], radius=radius, fill=255)
    base.paste(im, xy, mask)
    ImageDraw.Draw(base).rounded_rectangle(
        [xy[0] - bw, xy[1] - bw, xy[0] + w + bw - 1, xy[1] + h + bw - 1],
        radius=radius + bw, outline=border, width=bw)

def measure_segs(d, segs):
    return sum(d.textlength(s, font=f) for s, f, _ in segs)

def draw_segs(d, x, y, segs, color, mono_color):
    for s, f, is_mono in segs:
        d.text((x, y), s, font=f, fill=mono_color if is_mono else color)
        x += d.textlength(s, font=f)
    return x

def chips(d, x, y, items, fs=30, pad=20, gap=16, h=None):
    fl = font(F_REG, fs - 2)
    fm = font(F_MONO, fs)
    hh = h or (fs + pad + 6)
    for label, value in items:
        lw = d.textlength(label + "  ", font=fl)
        segs = mixed_fonts(value, fs, fs - 2)
        vw = measure_segs(d, segs)
        w = int(lw + vw + pad * 2)
        d.rounded_rectangle([x, y, x + w, y + hh], radius=10,
                            fill=(19, 28, 37), outline=(45, 62, 78), width=1)
        d.text((x + pad, y + (hh - fs) // 2 - 1), label, font=fl, fill=GREY)
        draw_segs(d, x + pad + lw, y + (hh - fs) // 2 - 2, segs, ACCENT, ACCENT)
        x += w + gap
    return x

def vertical(w, h, title_lines, sub, shot, box, out, chip_rows, shot_max_h=None):
    img = draw_bg(w, h)
    d = ImageDraw.Draw(img)
    M = int(w * 0.083)
    y = int(h * 0.055)

    d.rounded_rectangle([M, y + 10, M + 46, y + 16], radius=3, fill=ACCENT)
    d.text((M + 64, y - 6), "SPXH · 智能射频信号分析平台", font=font(F_REG, 29), fill=ACCENT2)

    y = int(h * 0.125)
    ft = font(F_BOLD, int(w * 0.080))
    step = int(int(w * 0.080) * 1.34)
    for ln in title_lines:
        d.text((M, y), ln, font=ft, fill=WHITE); y += step
    d.text((M, y + 6), sub, font=ft, fill=ACCENT)
    y += int(step * 1.18)

    d.line([(M, y), (M + int(w * 0.26), y)], fill=(50, 68, 86), width=3)
    y += 40
    for row in chip_rows:
        chips(d, M, y, row, fs=31, h=68)
        y += 84

    im = Image.open(shot).convert("RGB").crop(box)
    avail_w = w - 2 * M
    avail_h = h - y - int(h * 0.045)
    sc = min(avail_w / im.size[0], avail_h / im.size[1])
    if sc < 1:
        im = im.resize((int(im.size[0] * sc), int(im.size[1] * sc)), Image.LANCZOS)
    x = (w - im.size[0]) // 2
    paste_round(img, im, (x, h - im.size[1] - int(h * 0.045)))
    img.save(out); print("saved", os.path.basename(out), img.size)

def horizontal(w, h, title_lines, sub, shot, box, out, chip_rows):
    img = draw_bg(w, h)
    d = ImageDraw.Draw(img)
    M = int(w * 0.05)
    colw = int(w * 0.47)

    d.rounded_rectangle([M, int(h * 0.13), M + 46, int(h * 0.13) + 6], radius=3, fill=ACCENT)
    d.text((M + 64, int(h * 0.13) - 16), "SPXH · 智能射频信号分析平台", font=font(F_REG, 27), fill=ACCENT2)

    y = int(h * 0.235)
    ft = font(F_BOLD, 60)
    for ln in title_lines:
        d.text((M, y), ln, font=ft, fill=WHITE); y += 82
    d.text((M, y + 4), sub, font=font(F_BOLD, 46), fill=ACCENT)
    y += 108

    for row in chip_rows:
        chips(d, M, y, row, fs=26, h=58)
        y += 72

    im = Image.open(shot).convert("RGB").crop(box)
    avail_w = w - 2 * M - colw
    avail_h = int(h * 0.70)
    sc = min(avail_w / im.size[0], avail_h / im.size[1])
    if sc < 1:
        im = im.resize((int(im.size[0] * sc), int(im.size[1] * sc)), Image.LANCZOS)
    x = w - M - im.size[0]
    paste_round(img, im, (x, (h - im.size[1]) // 2))
    img.save(out); print("saved", os.path.basename(out), img.size)

FRAME = os.path.join(FIG, "08-谱相关.png")
CLASSIFY = os.path.join(FIG, "04-识别.png")
# 帧页签顶部四张统计卡（解析帧数 / CRC 通过 / 载荷字节 / 与真值逐字节一致）
STATS_BOX = (318, 198, 1818, 946)   # 谱相关：热图 + α/f 剖面
CLASSIFY_BOX = (272, 250, 1840, 620)

vertical(1080, 1440,
    ["一段谁也看不懂的", "无线电数据"],
    "它自己读出了 512 字节",
    FRAME, STATS_BOX,
    os.path.join(OUT, "封面-竖版-抖音3x4.png"),
    [[("符号速率", "1000.036 Hz"), ("载频偏差", "0.98 Hz")],
     [("制式", "QPSK"), ("CRC", "8/8"), ("载荷", "512 B")]])

vertical(1080, 1920,
    ["一段谁也看不懂的", "无线电数据"],
    "它自己读出了 512 字节",
    FRAME, STATS_BOX,
    os.path.join(OUT, "封面-竖版-视频号9x16.png"),
    [[("符号速率", "1000.036 Hz"), ("盲SNR", "9.84 dB")],
     [("制式", "QPSK 0.917"), ("CRC", "8/8")],
     [("载荷", "512 B 逐字节一致")]])

horizontal(1920, 1080,
    ["一段看不懂的 I/Q，", "怎么变成可读的比特？"],
    "12 级流水线，每步都说得出依据",
    FRAME, STATS_BOX,
    os.path.join(OUT, "封面-横版-知乎16x9.png"),
    [[("符号速率", "1000.036 Hz"), ("载频", "0.98 Hz"), ("盲SNR", "9.84 dB")],
     [("制式", "QPSK 0.917"), ("CRC", "8/8"), ("载荷", "512 B 逐字节一致")]])
