#!/usr/bin/env python3
"""트라이맵 + 매팅 시험 — 분리 모델 마스크를 '확실한 인물 / 확실한 배경 / 경계 띠'로 나누고, 띠 안에서만 매팅으로 투명도를 계산한다.

비교(같은 크롭, 검정·흰색 배경):
  now      마스크 그대로(지금 워커, blur 0)
  curve    마스크에 S자 곡선(20~80%)
  cf       트라이맵 + closed-form matting(pymatting, 학습 없음)
  vitmatte 트라이맵 + ViTMatte(hustvl/vitmatte-small-composition-1k)
색은 모두 경계 색 복원(Forte & Pitié 블러 융합)으로 계산한다 — 투명도 차이만 보이게.

입력: --image(원본 RGB)와 --mask(분리 모델 마스크: 흑백 PNG, 또는 RGBA면 알파 채널)
사용:
  python matting_test.py --image cut.png --mask cut_result.png --out ./out --crops 150,850,550,1250 1150,1050,1456,1450
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


def load_mask(path: Path) -> np.ndarray:
    im = Image.open(path)
    band = im.getchannel("A") if im.mode == "RGBA" else im.convert("L")
    return np.asarray(band, dtype=np.float32) / 255


def make_trimap(mask: np.ndarray, inner: int, outer: int) -> np.ndarray:
    """0=확실한 배경, 0.5=경계 띠, 1=확실한 인물. 띠 = (마스크≥0.5를 outer px 넓힌 것) − (inner px 줄인 것)."""
    core = (mask >= 0.5).astype(np.uint8)
    disk = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    sure_fg = cv2.erode(core, disk(inner))
    maybe = cv2.dilate(core, disk(outer))
    trimap = np.zeros(mask.shape, np.float32)
    trimap[maybe > 0] = 0.5
    trimap[sure_fg > 0] = 1.0
    return trimap


def scurve(mask: np.ndarray, lo: float, hi: float) -> np.ndarray:
    t = np.clip((mask - lo) / (hi - lo), 0, 1)
    return t * t * (3 - 2 * t)


def fusion(image, fg, bg, alpha, r):
    a = alpha[..., None]
    ba = cv2.blur(alpha, (r, r))[..., None]
    bf = cv2.blur(fg * a, (r, r)) / (ba + 1e-5)
    bb = cv2.blur(bg * (1 - a), (r, r)) / ((1 - ba) + 1e-5)
    return np.clip(bf + a * (image - a * bf - (1 - a) * bb), 0, 1), bb


def estimate_foreground(image, alpha):
    fg, bb = fusion(image, image, image, alpha, 90)
    return fusion(image, fg, bb, alpha, 6)[0]


def alpha_cf(image, trimap):
    from pymatting import estimate_alpha_cf
    return estimate_alpha_cf(image.astype(np.float64), trimap.astype(np.float64)).astype(np.float32)


def alpha_vitmatte(image, trimap, repo):
    import torch
    from transformers import VitMatteForImageMatting, VitMatteImageProcessor
    processor = VitMatteImageProcessor.from_pretrained(repo)
    model = VitMatteForImageMatting.from_pretrained(repo).eval()
    rgb = Image.fromarray((image * 255 + .5).astype(np.uint8))
    tri = Image.fromarray((trimap * 255 + .5).astype(np.uint8))
    inputs = processor(images=rgb, trimaps=tri, return_tensors="pt")
    with torch.no_grad():
        alpha = model(**inputs).alphas[0, 0].numpy()
    h, w = trimap.shape
    alpha = alpha[:h, :w]  # 프로세서가 32의 배수로 덧댄 부분을 자른다
    # 확실한 영역은 트라이맵 그대로(매팅은 띠 안에서만)
    return np.where(trimap == 0.5, alpha, trimap).astype(np.float32)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--mask", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--crops", nargs="+", required=True, help="x0,y0,x1,y1 (원본 좌표)")
    p.add_argument("--inner", type=int, default=6, help="트라이맵: 확실한 인물을 마스크에서 줄이는 px")
    p.add_argument("--outer", type=int, default=12, help="트라이맵: 경계 띠를 마스크 밖으로 넓히는 px")
    p.add_argument("--methods", default="now,curve,cf,vitmatte")
    p.add_argument("--vitmatte", default="hustvl/vitmatte-small-composition-1k")
    args = p.parse_args()

    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(args.image).stem
    image = np.asarray(Image.open(args.image).convert("RGB"), dtype=np.float32) / 255
    mask = load_mask(Path(args.mask))
    if mask.shape != image.shape[:2]:
        raise SystemExit(f"마스크·원본 크기가 다릅니다: {mask.shape} {image.shape[:2]}")
    trimap = make_trimap(mask, args.inner, args.outer)
    Image.fromarray((trimap * 255).astype(np.uint8)).save(out / f"{stem}_trimap.png")
    print(f"trimap: fg {np.mean(trimap == 1):.1%}, band {np.mean(trimap == 0.5):.1%}, bg {np.mean(trimap == 0):.1%}")

    alphas = {}
    for name in args.methods.split(","):
        start = time.perf_counter()
        if name == "now":
            alphas[name] = mask
        elif name == "curve":
            alphas[name] = scurve(mask, 0.2, 0.8)
        elif name == "cf":
            alphas[name] = alpha_cf(image, trimap)
        elif name == "vitmatte":
            alphas[name] = alpha_vitmatte(image, trimap, args.vitmatte)
        else:
            raise SystemExit(f"모르는 방식: {name}")
        alphas[name] = np.clip(alphas[name], 0, 1)
        a = alphas[name]
        print(f"{name}: {time.perf_counter() - start:.1f}s  faint(0<a<0.3) {int(((a > 0.004) & (a < 0.3)).sum())}px")

    rows = []
    for bg in ((0, 0, 0), (255, 255, 255)):
        for box in [tuple(int(v) for v in c.split(",")) for c in args.crops]:
            tiles = []
            for name, a in alphas.items():
                fg = estimate_foreground(image, a)
                comp = fg * a[..., None] + np.array(bg, np.float32) / 255 * (1 - a[..., None])
                tile = Image.fromarray((comp * 255 + .5).astype(np.uint8)).crop(box)
                d = ImageDraw.Draw(tile)
                d.rectangle((0, 0, 70, 16), fill=(128, 128, 128))
                d.text((3, 2), name, fill="white")
                tiles.append(tile)
            row = Image.new("RGB", (sum(t.width + 6 for t in tiles), tiles[0].height), (90, 90, 90))
            x = 0
            for t in tiles:
                row.paste(t, (x, 0))
                x += t.width + 6
            rows.append(row)
    sheet = Image.new("RGB", (max(r.width for r in rows), sum(r.height + 6 for r in rows)), (90, 90, 90))
    y = 0
    for r in rows:
        sheet.paste(r, (0, y))
        y += r.height + 6
    sheet.save(out / f"{stem}_matting.jpg", quality=90)
    for name, a in alphas.items():
        rgba = np.dstack([estimate_foreground(image, a), a])
        Image.fromarray((rgba * 255 + .5).astype(np.uint8), "RGBA").save(out / f"{stem}_{name}.png")
    print("완료:", out)


if __name__ == "__main__":
    main()
