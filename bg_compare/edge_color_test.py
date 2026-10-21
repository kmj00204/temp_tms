#!/usr/bin/env python3
"""경계 색 복원(foreground color estimation) 시험 — compare_bg_removal.py 결과의 원본·마스크로 계산한다(GPU·모델 불필요).

세 가지를 같은 크롭으로 흰색·검정 배경 위에 나란히 놓는다.
  now       현재 워커: 마스크에 가우시안 블러(σ=1.5), 색은 원본 그대로
  raw       마스크 그대로(블러·이진화 없음 = blur_sigma 0.001), 색은 원본 그대로
  raw+fg    마스크 그대로 + 경계 색 복원 — 반투명 픽셀 색에서 배경색을 빼고 전경색만 남긴다

경계 색 복원: Forte & Pitié, "Approximate Fast Foreground Colour Estimation"(ICIP 2021)의 블러 융합.
BiRefNet 공식 코드의 refine_foreground와 같은 계산(반경 90으로 한 번, 그 결과로 반경 6으로 한 번)이다.

사용:
  python edge_color_test.py --src ~/abcc --out ~/abcc_edge --cuts cut_069,cut_085,...
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def box_blur(x: np.ndarray, r: int) -> np.ndarray:
    """평균 필터(r×r, cv2.blur와 같은 크기) — 적분 영상으로 계산, 가장자리는 반사."""
    lo, hi = r // 2, r - 1 - r // 2
    pad = [(lo + 1, hi), (lo + 1, hi)] + [(0, 0)] * (x.ndim - 2)
    p = np.pad(x, pad, mode="reflect").astype(np.float64)
    s = p.cumsum(0).cumsum(1)
    h, w = x.shape[:2]
    total = s[r:r + h, r:r + w] - s[:h, r:r + w] - s[r:r + h, :w] + s[:h, :w]
    return (total / (r * r)).astype(np.float32)


def _fusion(image, fg, bg, alpha, r):
    blurred_alpha = box_blur(alpha, r)
    blurred_fg = box_blur(fg * alpha, r) / (blurred_alpha + 1e-5)
    blurred_bg = box_blur(bg * (1 - alpha), r) / ((1 - blurred_alpha) + 1e-5)
    fg = blurred_fg + alpha * (image - alpha * blurred_fg - (1 - alpha) * blurred_bg)
    return np.clip(fg, 0, 1), blurred_bg


def estimate_foreground(image: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """image: HxWx3(0~1), alpha: HxW(0~1) → 배경색을 뺀 전경색 HxWx3(0~1)."""
    a = alpha[..., None]
    fg, blurred_bg = _fusion(image, image, image, a, 90)
    return _fusion(image, fg, blurred_bg, a, 6)[0]


def composite(fg: np.ndarray, alpha: np.ndarray, color) -> Image.Image:
    a = alpha[..., None]
    out = fg * a + np.asarray(color, dtype=np.float32) / 255 * (1 - a)
    return Image.fromarray(np.clip(out * 255 + 0.5, 0, 255).astype(np.uint8))


def hair_box(alpha: np.ndarray, size: int):
    """마스크 윗부분(머리)에서 반투명 픽셀이 가장 많은 size×size 영역."""
    soft = ((alpha > 0.04) & (alpha < 0.96)).astype(np.float32)
    ys = np.nonzero(alpha > 0.5)[0]
    top, bottom = ys.min(), ys.min() + int((ys.max() - ys.min()) * 0.45)
    h, w = alpha.shape
    best, box = -1.0, (0, 0)
    step = max(16, size // 4)
    for y in range(max(0, top - size // 3), max(1, min(bottom, h - size)), step):
        for x in range(0, max(1, w - size), step):
            score = soft[y:y + size, x:x + size].sum()
            if score > best:
                best, box = score, (x, y)
    x, y = box
    return x, y, x + size, y + size


def main() -> None:
    parser = argparse.ArgumentParser(description="경계 색 복원 시험")
    parser.add_argument("--src", required=True, help="compare_bg_removal.py 결과 폴더(<컷>/original.jpg, <모델>_mask.png)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--cuts", required=True, help="쉼표로 구분한 컷 이름")
    parser.add_argument("--mask", default="inspyrenet", help="쓸 마스크 모델 이름")
    parser.add_argument("--crop", type=int, default=300, help="크롭 크기(px, 원본 해상도)")
    parser.add_argument("--zoom", type=int, default=2, help="크롭 확대 배율(보기 편하게, 최근접 보간)")
    parser.add_argument("--now-sigma", type=float, default=1.5, help="현재 워커의 경계 블러")
    args = parser.parse_args()

    src, out = Path(args.src).expanduser(), Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    labels = ["now", "raw", "raw+fg"]
    rows = []
    for cut in args.cuts.split(","):
        d = src / cut
        original = Image.open(d / "original.jpg").convert("RGB")
        mask_img = Image.open(d / f"{args.mask}_mask.png").convert("L")
        if mask_img.size != original.size:
            raise RuntimeError(f"{cut}: 마스크·원본 크기가 다릅니다 {mask_img.size} {original.size}")
        image = np.asarray(original, dtype=np.float32) / 255
        raw = np.asarray(mask_img, dtype=np.float32) / 255
        now = np.asarray(mask_img.filter(ImageFilter.GaussianBlur(args.now_sigma)), dtype=np.float32) / 255
        fg = estimate_foreground(image, raw)
        variants = {"now": (image, now), "raw": (image, raw), "raw+fg": (fg, raw)}

        # 결과 RGBA 저장(전체 크기) — 다른 프로그램에서 직접 확인용
        for name, (color, alpha) in variants.items():
            rgba = np.dstack([color, alpha])
            Image.fromarray(np.clip(rgba * 255 + 0.5, 0, 255).astype(np.uint8), "RGBA").save(
                out / f"{cut}_{name.replace('+', '_')}.png")

        box = hair_box(raw, args.crop)
        tiles = []
        for bg in ((255, 255, 255), (0, 0, 0)):
            for name in labels:
                color, alpha = variants[name]
                tile = composite(color, alpha, bg).crop(box)
                tiles.append(tile.resize((tile.width * args.zoom, tile.height * args.zoom), Image.NEAREST))
        tw, th = tiles[0].size
        row = Image.new("RGB", (3 * (tw + 6), 2 * (th + 18) + 4), (200, 200, 205))
        draw = ImageDraw.Draw(row)
        for i, tile in enumerate(tiles):
            x, y = (i % 3) * (tw + 6), (i // 3) * (th + 18)
            draw.text((x + 3, y + 3), f"{cut} {labels[i % 3]}", fill=(0, 0, 0))
            row.paste(tile, (x, y + 18))
        row.save(out / f"{cut}_edge.jpg", quality=92)
        rows.append(cut)
        print(cut, box)
    print(f"완료: {out}")


if __name__ == "__main__":
    main()
