#!/usr/bin/env python3
"""두 모델 마스크 합치기 실험 — compare_bg_removal.py 결과 폴더의 마스크만으로 계산한다(GPU·모델 불필요).

I = InSPyReNet(--base-i), B = BiRefNet_dynamic(--base-b) 마스크(0~1).
  ens_min      min(I, B)                         한쪽이 배경이라고 하면 지운다(I 잔상·B 침대 둘 다)
  ens_min_dil  min(I, B를 --dilate px 넓힌 것)     B가 놓친 머리카락 끝 같은 가는 부분은 I에서 살린다
  ens_boost    ens_min_dil + B ≥ 0.9인 곳에서 I×1.6  B가 확실히 전경인 곳의 I 반투명(뒤쪽 인물 유령)을 진하게
  ens_clean    ens_min에서 확실한 전경(≥0.9)과 떨어진 반투명 덩어리만 지운다(베개·이불 잔상). 확실한 영역에서
               --keep-px 안쪽은 그대로 두고, 그 밖은 거리에 따라 부드럽게 옅어진다 — 0/1로 자르지 않는다

결과 폴더는 원본 결과 폴더와 같은 구조다(<컷>/original.jpg, <컷>/<이름>.png·_mask.png, sheets/, rate.html).
후처리는 워커와 같다(--blur-sigma, 기본 1.5).

사용:
  python ensemble_masks.py --src ~/abcc --dst ~/abcc_ens
"""

import argparse
import ast
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

BOOST_GATE = 0.9
BOOST_GAIN = 1.6
CORE = 0.9


def load_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255


def to_image(mask: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(mask * 255 + 0.5, 0, 255).astype(np.uint8))


def dilate(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return mask
    size = px * 2 + 1
    return np.asarray(to_image(mask).filter(ImageFilter.MaxFilter(size)), dtype=np.float32) / 255


def near_core_gate(mask: np.ndarray, keep_px: int, fade_px: int) -> np.ndarray:
    """확실한 전경(≥CORE)에서 keep_px 안쪽은 1, 그 밖은 약 fade_px에 걸쳐 0으로 매끄럽게 떨어지는 값(0~1)."""
    core = to_image((mask >= CORE).astype(np.float32))
    near = core.filter(ImageFilter.MaxFilter(keep_px * 2 + 1)) if keep_px > 0 else core
    soft = np.asarray(near.filter(ImageFilter.GaussianBlur(fade_px / 2)), dtype=np.float32) / 255
    t = np.clip(soft / 0.5, 0, 1)  # 넓힌 영역 경계(0.5)부터 안쪽은 1
    return t * t * (3 - 2 * t)      # smoothstep — 꺾임 없이 옅어진다


def combine(i: np.ndarray, b: np.ndarray, dilate_px: int, keep_px: int, fade_px: int) -> dict:
    b_dil = dilate(b, dilate_px)
    min_dil = np.minimum(i, b_dil)
    boosted = np.where(b >= BOOST_GATE, np.clip(i * BOOST_GAIN, 0, 1), i)
    ens_min = np.minimum(i, b)
    return {
        "ens_min": ens_min,
        "ens_min_dil": min_dil,
        "ens_boost": np.minimum(boosted, b_dil),
        "ens_clean": ens_min * near_core_gate(ens_min, keep_px, fade_px),
    }


def postprocess(mask_img: Image.Image, blur_sigma: float) -> Image.Image:
    """워커와 같은 후처리 — 블러(>0) 또는 127 이진화."""
    if blur_sigma > 0:
        return mask_img.filter(ImageFilter.GaussianBlur(blur_sigma))
    return mask_img.point(lambda v: 255 if v >= 128 else 0)


def survey_sheet(rows, names, out_path: Path, height: int = 300) -> None:
    """원본 + 모델별 결과(마젠타 배경) 한 줄씩 — 반투명 잔상과 남은 배경이 잘 보인다."""
    built = []
    for stem, original, cutouts in rows:
        tiles = [original] + [cutouts[n] for n in names]
        labels = [stem] + names
        thumbs = []
        for idx, tile in enumerate(tiles):
            w = round(tile.width * height / tile.height)
            t = tile.resize((w, height), Image.LANCZOS)
            if idx:
                bg = Image.new("RGB", t.size, (255, 0, 200))
                bg.paste(t, (0, 0), t)
                t = bg
            thumbs.append(t)
        row = Image.new("RGB", (sum(t.width + 4 for t in thumbs), height + 20), "white")
        draw, x = ImageDraw.Draw(row), 0
        for t, label in zip(thumbs, labels):
            draw.text((x + 3, 3), label, fill="black")
            row.paste(t.convert("RGB"), (x, 20))
            x += t.width + 4
        built.append(row)
    sheet = Image.new("RGB", (max(r.width for r in built), sum(r.height + 6 for r in built)), "white")
    y = 0
    for r in built:
        sheet.paste(r, (0, y))
        y += r.height + 6
    sheet.save(out_path, quality=88)


def rate_page_template(compare_script: Path) -> str:
    # 평가 페이지는 compare_bg_removal.py의 것을 그대로 쓴다(그 파일은 torch를 import해 직접 불러오지 않는다)
    tree = ast.parse(compare_script.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "RATE_PAGE":
            return node.value.value
    raise RuntimeError(f"RATE_PAGE를 찾지 못했습니다: {compare_script}")


def main() -> None:
    parser = argparse.ArgumentParser(description="두 모델 마스크 합치기 실험")
    parser.add_argument("--src", required=True, help="compare_bg_removal.py 결과 폴더")
    parser.add_argument("--dst", required=True, help="합친 결과를 쓸 폴더(src는 건드리지 않는다)")
    parser.add_argument("--base-i", default="inspyrenet")
    parser.add_argument("--base-b", default="birefnet_dynamic")
    parser.add_argument("--dilate", type=int, default=15, help="ens_min_dil·ens_boost에서 B를 넓히는 px")
    parser.add_argument("--blur-sigma", type=float, default=1.5)
    parser.add_argument("--keep-px", type=int, default=30, help="ens_clean: 확실한 전경에서 그대로 두는 거리(px)")
    parser.add_argument("--fade-px", type=int, default=24, help="ens_clean: 그 밖에서 옅어지는 폭(px)")
    parser.add_argument("--only", default=None, help="쉼표로 컷 이름을 주면 그 컷만(빠른 확인용)")
    parser.add_argument("--per-sheet", type=int, default=4, help="한눈 보기 시트 한 장의 컷 수")
    args = parser.parse_args()

    src, dst = Path(args.src).expanduser(), Path(args.dst).expanduser()
    if src.resolve() == dst.resolve():
        parser.error("--dst는 --src와 다른 폴더여야 합니다")
    cuts = sorted(p for p in src.iterdir() if p.is_dir() and (p / f"{args.base_i}_mask.png").is_file())
    missing = [p.name for p in cuts if not (p / f"{args.base_b}_mask.png").is_file() or not (p / "original.jpg").is_file()]
    if not cuts or missing:
        parser.error(f"마스크·원본이 없는 컷이 있습니다: {missing[:5] or '컷 없음'}")
    if args.only:
        wanted = set(args.only.split(","))
        cuts = [c for c in cuts if c.name in wanted]
    (dst / "sheets").mkdir(parents=True, exist_ok=True)

    shown = [args.base_i, args.base_b, "ens_min", "ens_clean"]
    rows = []
    for d in cuts:
        out = dst / d.name
        out.mkdir(exist_ok=True)
        shutil.copy2(d / "original.jpg", out / "original.jpg")
        original = Image.open(d / "original.jpg").convert("RGB")
        i, b = load_mask(d / f"{args.base_i}_mask.png"), load_mask(d / f"{args.base_b}_mask.png")
        if i.shape != b.shape or i.shape != (original.height, original.width):
            raise RuntimeError(f"{d.name}: 마스크·원본 크기가 다릅니다 {i.shape} {b.shape} {original.size}")
        masks = {args.base_i: i, args.base_b: b, **combine(i, b, args.dilate, args.keep_px, args.fade_px)}
        cutouts = {}
        for name, mask in masks.items():
            mask_img = to_image(mask)
            mask_img.save(out / f"{name}_mask.png")
            cut = original.copy()
            cut.putalpha(postprocess(mask_img, args.blur_sigma))
            cut.save(out / f"{name}.png")
            cutouts[name] = cut
        rows.append((d.name, original, cutouts))
        print(d.name)

    for start in range(0, len(rows), args.per_sheet):
        survey_sheet(rows[start:start + args.per_sheet], shown, dst / "sheets" / f"s{start // args.per_sheet:02d}.jpg")

    page = rate_page_template(Path(__file__).with_name("compare_bg_removal.py"))
    data = json.dumps({"cuts": [d.name for d in cuts], "models": shown}, ensure_ascii=False)
    (dst / "rate.html").write_text(page.replace("__DATA__", data), encoding="utf-8")
    print(f"완료: {dst} (sheets/ 한눈 보기, rate.html 블라인드 평가)")


if __name__ == "__main__":
    main()
