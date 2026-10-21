#!/usr/bin/env python3
"""업스케일 모델 비교 — RealESRGAN_x4plus_anime_6B(예전 upscaler 워커) vs 2x-AnimeSharpV4(현재 워커).

프로젝트 코드와 무관한 단독 스크립트다. 같은 이미지를 모델·배율마다 돌려 결과, 1:1 확대 비교본, 지표를 남긴다.
배율 처리는 각 워커와 같다 — 모델 차이만 보이게.
  예전 워커: 4배 모델을 한 번 돌리고 목표 크기로 LANCZOS (fp16)
  현재 워커: 목표 배율 이상이 될 때까지 2배 모델을 반복하고 목표 크기로 LANCZOS (fp32)

모델(--models, 쉼표 구분):
  realesrgan_anime6b   예전 워커
  animesharp_v4        현재 워커 (2x-AnimeSharpV4_RCAN)
  animesharp_v4_fast   2x-AnimeSharpV4_Fast_RCAN_PU (빠른 판)
  lanczos              모델 없이 LANCZOS로만 키운 기준선 — 모델이 실제로 더해 주는 게 얼마인지 보는 용도
가중치는 --weights-dir에 없으면 공개 주소(GitHub 릴리스·Hugging Face)에서 내려받는다.

두 가지 방식:
  기본   입력 이미지를 그대로 키운다(실서비스와 같음). 정답이 없으니 눈으로 비교한다
  --gt   입력을 정답으로 보고 배율만큼 줄인 뒤(선택: --gt-jpeg로 JPEG 열화) 다시 키워 정답과의 PSNR을 잰다

사용 예:
  python compare_upscale.py --input ./samples --output ./out
  python compare_upscale.py --input ./samples --output ./out --scales 2,4 --models realesrgan_anime6b,animesharp_v4,lanczos
  python compare_upscale.py --input ./hires_cuts --output ./out_gt --gt --gt-jpeg 80
  # upscaler 서버 가상환경(torch 2.0.1)에서 돌릴 때는 cuDNN을 꺼야 한다(워커와 같은 이유)
  python compare_upscale.py --input ./samples --output ./out --no-cudnn

결과(--output 아래):
  <이미지명>/x<배율>/<모델>.png   업스케일 결과
  sheets/<이미지명>_x<배율>.jpg    디테일이 많은 구간을 골라 1:1로 자른 뒤 --zoom배(픽셀 그대로) 키워 모델별로 나란히 놓은 비교본
  metrics.csv                      이미지·배율·모델별 처리 시간, 원본 일치도, 선명도(--gt면 정답 PSNR)
  compare.html                     뷰어 — 1~8배 확대(픽셀 그대로), 두 칸 동기 스크롤, Space로 A↔B 전환, 이름 가리기

지표 읽는 법:
  원본 일치도  결과를 입력 크기로 다시 줄여 입력과 비교한 PSNR(dB). 높을수록 입력 내용을 덜 바꿨다는 뜻이고,
               lanczos가 거의 항상 가장 높다 — 화질 점수가 아니라 '얼마나 고쳐 그렸나'를 보는 값
  선명도       결과의 평균 밝기 변화량. 같은 이미지·배율 안에서만 비교한다. 노이즈·가짜 디테일도 올리므로 높다고 좋은 건 아님
  정답 PSNR    --gt일 때만. 원래 고해상도와 얼마나 같은지(dB, 높을수록 좋음)
"""

import argparse
import csv
import gc
import io
import json
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
HF_ANIMESHARP = "https://huggingface.co/Kim2091/2x-AnimeSharpV4/resolve/main/"
PSNR_CAP = 99.0  # 완전히 같으면 무한대라 JSON·CSV에 쓸 수 있게 상한을 둔다
MIN_GT_INPUT_SIDE = 16  # --gt에서 줄인 입력이 이보다 작으면 그 배율은 건너뛴다


@dataclass(frozen=True)
class ModelSpec:
    file: str
    url: str
    scale: int  # 가중치가 기대한 배율인지 로드 때 확인한다
    half: bool  # CUDA에서 fp16으로 돌릴지 (워커 설정과 같게)
    repeat: bool  # True: 목표 배율까지 모델 반복(현재 워커) / False: 한 번만 돌리고 리사이즈(예전 워커)


MODEL_SPECS: Dict[str, ModelSpec] = {
    "realesrgan_anime6b": ModelSpec(
        file="RealESRGAN_x4plus_anime_6B.pth",
        url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        scale=4, half=True, repeat=False,
    ),
    "animesharp_v4": ModelSpec(
        file="2x-AnimeSharpV4_RCAN.safetensors", url=HF_ANIMESHARP + "2x-AnimeSharpV4_RCAN.safetensors",
        scale=2, half=False, repeat=True,
    ),
    "animesharp_v4_fast": ModelSpec(
        file="2x-AnimeSharpV4_Fast_RCAN_PU.safetensors", url=HF_ANIMESHARP + "2x-AnimeSharpV4_Fast_RCAN_PU.safetensors",
        scale=2, half=False, repeat=True,
    ),
}
BASELINE = "lanczos"


# ── 공통 ────────────────────────────────────────────────────────────────

def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sync(device: str) -> None:
    # 처리 시간을 재기 전에 GPU 작업이 끝나기를 기다린다(비동기 실행이라 안 기다리면 시간이 짧게 잡힌다)
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def scale_label(scale: float) -> str:
    return f"{scale:g}"


def ensure_weights(spec: ModelSpec, weights_dir: Path, allow_download: bool) -> Path:
    path = weights_dir / spec.file
    if path.is_file():
        return path
    if not allow_download:
        raise SystemExit(f"가중치 파일이 없습니다: {path.resolve()} (--no-download를 빼면 내려받습니다)")
    weights_dir.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    print(f"내려받는 중: {spec.url}")
    urllib.request.urlretrieve(spec.url, partial)
    partial.rename(path)
    return path


def to_tensor(img: Image.Image) -> torch.Tensor:
    """RGB PIL → (1, 3, H, W) float32 [0, 1]."""
    array = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).contiguous()


def to_image(tensor: torch.Tensor) -> Image.Image:
    array = (tensor.squeeze(0).permute(1, 2, 0).clamp(0, 1) * 255.0).round().to(torch.uint8).numpy()
    return Image.fromarray(array, "RGB")


# ── 모델 ────────────────────────────────────────────────────────────────

def _tile_starts(length: int, tile: int, overlap: int) -> List[int]:
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile, tile - overlap))
    starts.append(length - tile)
    return starts


def _edge_profile(length: int, ramp: int, fade_start: bool, fade_end: bool) -> torch.Tensor:
    profile = torch.ones(length)
    fade = torch.arange(1, ramp + 1, dtype=torch.float32) / (ramp + 1)
    if fade_start:
        profile[:ramp] = torch.minimum(profile[:ramp], fade)
    if fade_end:
        profile[-ramp:] = torch.minimum(profile[-ramp:], fade.flip(0))
    return profile


class SpandrelUpscaler:
    """spandrel로 연 모델. 타일 분할·겹침 가중 합성은 현재 upscaler 워커와 같은 방식."""

    def __init__(self, name: str, spec: ModelSpec, path: Path, device: str, tile: int, overlap: int):
        from spandrel import ImageModelDescriptor, ModelLoader

        descriptor = ModelLoader(device="cpu").load_from_file(str(path))
        if not isinstance(descriptor, ImageModelDescriptor):
            raise RuntimeError(f"[{name}] 이미지 모델이 아닙니다: {type(descriptor).__name__}")
        if descriptor.scale != spec.scale or descriptor.input_channels != 3 or descriptor.output_channels != 3:
            raise RuntimeError(
                f"[{name}] 기대한 모델과 다릅니다: arch={descriptor.architecture.id} scale={descriptor.scale} "
                f"in={descriptor.input_channels} out={descriptor.output_channels} (기대 scale={spec.scale}, RGB)"
            )
        descriptor.to(torch.device(device))
        if spec.half and device == "cuda":
            descriptor.half()
        self.model = descriptor.eval()
        self.repeat = spec.repeat
        self.tile, self.overlap = tile, overlap
        print(f"[{name}] arch={descriptor.architecture.id} scale={descriptor.scale}x dtype={descriptor.dtype} "
              f"tags={descriptor.tags}")

    def _run(self, image: torch.Tensor) -> torch.Tensor:
        model, s = self.model, self.model.scale
        _, channels, h, w = image.shape
        ys, xs = _tile_starts(h, self.tile, self.overlap), _tile_starts(w, self.tile, self.overlap)
        # RCAN forward가 입력을 제자리에서 곱하므로 타일은 항상 복사본으로 넘긴다
        if len(ys) == 1 and len(xs) == 1:
            return model(image.to(device=model.device, dtype=model.dtype, copy=True)).float().cpu()

        ramp = self.overlap * s
        output = torch.zeros((1, channels, h * s, w * s))
        weight = torch.zeros((1, 1, h * s, w * s))
        for y0 in ys:
            y1 = min(y0 + self.tile, h)
            for x0 in xs:
                x1 = min(x0 + self.tile, w)
                tile = image[:, :, y0:y1, x0:x1].to(device=model.device, dtype=model.dtype, copy=True)
                tile_out = model(tile).float().cpu()
                mask = (
                    _edge_profile((y1 - y0) * s, ramp, y0 > 0, y1 < h)[:, None]
                    * _edge_profile((x1 - x0) * s, ramp, x0 > 0, x1 < w)[None, :]
                )
                output[:, :, y0 * s:y1 * s, x0 * s:x1 * s] += tile_out * mask
                weight[:, :, y0 * s:y1 * s, x0 * s:x1 * s] += mask
        return output / weight

    def upscale(self, img: Image.Image, target: Tuple[int, int]) -> Image.Image:
        needed = max(target[0] / img.width, target[1] / img.height)
        passes = 1
        while self.repeat and self.model.scale ** passes < needed - 1e-6:
            passes += 1
        tensor = to_tensor(img)
        for _ in range(passes):
            tensor = self._run(tensor)
        out = to_image(tensor)
        return out if out.size == target else out.resize(target, Image.LANCZOS)


class LanczosUpscaler:
    def upscale(self, img: Image.Image, target: Tuple[int, int]) -> Image.Image:
        return img.resize(target, Image.LANCZOS)


# ── 지표 ────────────────────────────────────────────────────────────────

def psnr(a: Image.Image, b: Image.Image, mode: str = "RGB") -> float:
    x = np.asarray(a.convert(mode), dtype=np.float64)
    y = np.asarray(b.convert(mode), dtype=np.float64)
    mse = np.mean((x - y) ** 2)
    return PSNR_CAP if mse == 0 else min(PSNR_CAP, float(10 * np.log10(255.0 ** 2 / mse)))


def sharpness(img: Image.Image) -> float:
    y = np.asarray(img.convert("L"), dtype=np.float32)
    return float(np.mean(np.abs(np.diff(y, axis=0))) + np.mean(np.abs(np.diff(y, axis=1))))


# ── 비교 시트 ───────────────────────────────────────────────────────────

def detail_windows(src: Image.Image, size: int, count: int) -> List[Tuple[int, int, int, int]]:
    """입력에서 밝기 변화(선·질감)가 가장 많은 구간을 겹치지 않게 count개 고른다. (x, y, w, h)"""
    y = np.asarray(src.convert("L"), dtype=np.float32)
    h, w = y.shape
    win_w, win_h = min(size, w), min(size, h)
    energy = np.zeros_like(y)
    energy[:, 1:] += np.abs(np.diff(y, axis=1))
    energy[1:, :] += np.abs(np.diff(y, axis=0))
    integral = np.pad(energy.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    step = max(1, min(win_w, win_h) // 2)
    ys = sorted(set(range(0, h - win_h + 1, step)) | {h - win_h})
    xs = sorted(set(range(0, w - win_w + 1, step)) | {w - win_w})
    scored = []
    for y0 in ys:
        for x0 in xs:
            s = (integral[y0 + win_h, x0 + win_w] - integral[y0, x0 + win_w]
                 - integral[y0 + win_h, x0] + integral[y0, x0])
            scored.append((float(s), x0, y0))
    scored.sort(reverse=True)
    picked: List[Tuple[int, int, int, int]] = []
    for _, x0, y0 in scored:
        if all(x0 + win_w <= px or px + pw <= x0 or y0 + win_h <= py or py + ph <= y0 for px, py, pw, ph in picked):
            picked.append((x0, y0, win_w, win_h))
            if len(picked) == count:
                break
    return picked


def load_font(size: int) -> ImageFont.ImageFont:
    for name in ("DejaVuSans.ttf", "Arial.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def make_sheet(title: str, src: Image.Image, columns: List[Tuple[str, Image.Image]],
               boxes: List[Tuple[int, int, int, int]], zoom: int) -> Image.Image:
    """columns의 이미지들은 모두 같은 크기(목표 크기). 입력 구간을 같은 위치로 잘라 픽셀 그대로 zoom배 키운다."""
    out_w, out_h = columns[0][1].size
    rx, ry = out_w / src.width, out_h / src.height
    font = load_font(16)
    gap, title_h, label_h = 8, 34, 26

    crops: List[List[Image.Image]] = []
    for x0, y0, bw, bh in boxes:
        out_box = (round(x0 * rx), round(y0 * ry), round((x0 + bw) * rx), round((y0 + bh) * ry))
        display = ((out_box[2] - out_box[0]) * zoom, (out_box[3] - out_box[1]) * zoom)
        row = [src.crop((x0, y0, x0 + bw, y0 + bh)).resize(display, Image.NEAREST)]
        row += [img.crop(out_box).resize(display, Image.NEAREST) for _, img in columns]
        crops.append(row)

    cell_w = max(c.width for row in crops for c in row)
    labels = ["input (nearest)"] + [label for label, _ in columns]
    sheet_w = gap + len(labels) * (cell_w + gap)
    sheet_h = title_h + label_h + sum(max(c.height for c in row) + gap for row in crops) + gap
    sheet = Image.new("RGB", (sheet_w, sheet_h), (250, 250, 252))
    draw = ImageDraw.Draw(sheet)
    draw.text((gap, 8), title, fill=(20, 20, 30), font=font)
    for i, label in enumerate(labels):
        draw.text((gap + i * (cell_w + gap), title_h + 2), label, fill=(40, 40, 50), font=font)
    y = title_h + label_h
    for row in crops:
        for i, crop in enumerate(row):
            sheet.paste(crop, (gap + i * (cell_w + gap), y))
        y += max(c.height for c in row) + gap
    return sheet


# ── 뷰어 ────────────────────────────────────────────────────────────────

VIEWER_PAGE = r"""<!doctype html><meta charset="utf-8"><title>업스케일 비교</title>
<style>
  :root{--bg:#f4f4f6;--fg:#1d1d22;--muted:#6b6b76;--line:#d9d9e0;--accent:#5b43dc}
  html,body{height:100%}
  body{margin:0;font-family:system-ui,sans-serif;background:var(--bg);color:var(--fg);display:flex;flex-direction:column}
  header{display:flex;flex-wrap:wrap;gap:8px 12px;align-items:center;padding:10px 16px;border-bottom:1px solid var(--line);background:#fff}
  select,button{font:inherit;border:1px solid var(--line);background:#fff;border-radius:8px;padding:5px 10px;cursor:pointer}
  button.on{border-color:var(--accent);background:#efebff;color:var(--accent);font-weight:600}
  .muted{color:var(--muted);font-size:13px}
  #metrics{padding:6px 16px}
  #metrics table{border-collapse:collapse;background:#fff;font-size:13px}
  #metrics td,#metrics th{border:1px solid var(--line);padding:3px 10px;text-align:right}
  #metrics td:first-child,#metrics th:first-child{text-align:left}
  #panes{flex:1;display:grid;gap:8px;padding:8px;min-height:0}
  .pane{display:flex;flex-direction:column;min-height:0;background:#fff;border:1px solid var(--line);border-radius:10px;overflow:hidden}
  .pane .bar{padding:6px 10px;border-bottom:1px solid var(--line);font-weight:600}
  .view{flex:1;overflow:auto;cursor:grab;background:#e9e9ee}
  .view.drag{cursor:grabbing}
  .view img{display:block;max-width:none;user-select:none;-webkit-user-drag:none}
  .view img.px{image-rendering:pixelated}
</style>
<header>
  <b>업스케일 비교</b>
  <select id="item"></select>
  <span class="muted">A</span><select id="a"></select>
  <span class="muted">B</span><select id="b"></select>
  <span class="muted">확대</span><span id="zooms"></span>
  <button id="split">두 칸</button>
  <label class="muted"><input type="checkbox" id="blind"> 이름 가리기</label>
  <span class="muted">Space: A↔B 전환 · [ ]: 이전/다음 · 1~4: 확대 · 드래그: 이동</span>
</header>
<div id="metrics"></div>
<div id="panes"></div>
<script>
const DATA = __DATA__;
const ZOOMS = [1, 2, 4, 8];
let item = 0, zoom = 2, split = true, blind = false, swapped = false;
let pick = { a: DATA.models[0], b: DATA.models[1] || 'input' };
const enc = p => p.split('/').map(encodeURIComponent).join('/');
const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);
const cur = () => DATA.items[item];
const keys = it => ['input', ...DATA.models, ...(it.gt ? ['gt'] : [])];
// 이름 가리기: 이미지마다 정해진 순서로 섞어 글자로만 보인다(다시 열어도 같은 글자)
function order(it) {
  let h = 0; for (const c of it.id) h = (h * 31 + c.charCodeAt(0)) >>> 0;
  const arr = DATA.models.slice();
  for (let i = arr.length - 1; i > 0; i--) { h = (h * 1103515245 + 12345) >>> 0; const j = h % (i + 1); [arr[i], arr[j]] = [arr[j], arr[i]]; }
  return arr;
}
function label(it, key) {
  if (key === 'input') return '입력 (nearest)';
  if (key === 'gt') return '정답';
  return blind ? '모델 ' + String.fromCharCode(65 + order(it).indexOf(key)) : key;
}
const srcOf = (it, key) => enc(key === 'input' ? it.input : key === 'gt' ? it.gt : it.files[key]);
const shown = () => swapped ? [pick.b, pick.a] : [pick.a, pick.b];

function fillSelect(id, it) {
  const sel = document.getElementById(id);
  sel.innerHTML = keys(it).map(k => `<option value="${k}">${label(it, k)}</option>`).join('');
  sel.value = pick[id];
}
function metricsTable(it) {
  if (blind) return '<span class="muted">이름 가리기 중에는 지표를 숨깁니다</span>';
  const gt = !!it.gt;
  const head = `<tr><th>모델</th><th>시간 ms</th><th>원본 일치도 dB</th><th>선명도</th>${gt ? '<th>정답 PSNR dB</th>' : ''}</tr>`;
  const rows = DATA.models.map(m => { const v = it.metrics[m];
    return `<tr><td>${m}</td><td>${v.ms.toFixed(0)}</td><td>${v.consistency.toFixed(2)}</td><td>${v.sharpness.toFixed(2)}</td>${gt ? `<td>${v.gt_psnr.toFixed(2)}</td>` : ''}</tr>`; }).join('');
  return `<span class="muted">${it.inW}×${it.inH} → ${it.w}×${it.h}</span><table>${head}${rows}</table>`;
}
function center() {
  const v = document.querySelector('.view');
  if (!v || !v.scrollWidth) return [0.5, 0.5];
  return [(v.scrollLeft + v.clientWidth / 2) / v.scrollWidth, (v.scrollTop + v.clientHeight / 2) / v.scrollHeight];
}
function render(keepCenter = true) {
  const it = cur(), [cx, cy] = keepCenter ? center() : [0.5, 0.5];
  for (const id of ['a', 'b']) if (!keys(it).includes(pick[id])) pick[id] = DATA.models[0];
  document.getElementById('item').value = item;
  fillSelect('a', it); fillSelect('b', it);
  document.getElementById('zooms').innerHTML = ZOOMS.map(z => `<button class="${z === zoom ? 'on' : ''}" data-zoom="${z}">${z}×</button>`).join(' ');
  document.getElementById('split').className = split ? 'on' : '';
  document.getElementById('metrics').innerHTML = metricsTable(it);
  const panes = split ? shown() : [shown()[0]];
  const box = document.getElementById('panes');
  box.style.gridTemplateColumns = `repeat(${panes.length}, 1fr)`;
  box.innerHTML = panes.map(k => `<div class="pane"><div class="bar">${label(it, k)}</div>
    <div class="view"><img data-key="${k}" class="${zoom > 1 || k === 'input' ? 'px' : ''}"
      src="${srcOf(it, k)}" width="${it.w * zoom}" height="${it.h * zoom}"></div></div>`).join('');
  const views = [...box.querySelectorAll('.view')];
  views.forEach(v => {
    v.scrollLeft = cx * v.scrollWidth - v.clientWidth / 2;
    v.scrollTop = cy * v.scrollHeight - v.clientHeight / 2;
    v.onscroll = () => views.forEach(o => { if (o !== v && (o.scrollLeft !== v.scrollLeft || o.scrollTop !== v.scrollTop)) { o.scrollLeft = v.scrollLeft; o.scrollTop = v.scrollTop; } });
    v.onmousedown = e => {
      e.preventDefault(); v.classList.add('drag');
      const sx = e.clientX, sy = e.clientY, l = v.scrollLeft, t = v.scrollTop;
      const move = m => { v.scrollLeft = l - (m.clientX - sx); v.scrollTop = t - (m.clientY - sy); };
      const up = () => { v.classList.remove('drag'); removeEventListener('mousemove', move); removeEventListener('mouseup', up); };
      addEventListener('mousemove', move); addEventListener('mouseup', up);
    };
  });
}
// A↔B 전환은 스크롤 위치를 그대로 두고 그림만 바꾼다 — 같은 자리를 번갈아 봐야 작은 차이가 보인다
function toggle() {
  swapped = !swapped;
  const it = cur(), list = split ? shown() : [shown()[0]];
  document.querySelectorAll('.pane').forEach((p, i) => {
    const img = p.querySelector('img'), k = list[i];
    img.src = srcOf(it, k); img.className = zoom > 1 || k === 'input' ? 'px' : '';
    p.querySelector('.bar').textContent = label(it, k);
  });
}
const itemSel = document.getElementById('item');
itemSel.innerHTML = DATA.items.map((it, i) => `<option value="${i}">${esc(it.stem)} · x${it.scale}</option>`).join('');
itemSel.onchange = e => { item = +e.target.value; swapped = false; render(false); };
document.getElementById('a').onchange = e => { pick.a = e.target.value; swapped = false; render(); };
document.getElementById('b').onchange = e => { pick.b = e.target.value; swapped = false; render(); };
document.getElementById('split').onclick = () => { split = !split; render(); };
document.getElementById('blind').onchange = e => { blind = e.target.checked; render(); };
document.addEventListener('click', e => { const z = e.target.closest('[data-zoom]'); if (z) { zoom = +z.dataset.zoom; render(); } });
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'SELECT') return;
  if (e.code === 'Space') { e.preventDefault(); toggle(); }
  else if (e.key === '[' && item > 0) { item--; swapped = false; render(false); }
  else if (e.key === ']' && item < DATA.items.length - 1) { item++; swapped = false; render(false); }
  else if (/^[1-4]$/.test(e.key)) { zoom = ZOOMS[+e.key - 1]; render(); }
});
render(false);
</script>"""


def write_viewer(out_dir: Path, items: List[dict], models: List[str]) -> None:
    data = json.dumps({"items": items, "models": models}, ensure_ascii=False).replace("</", "<\\/")
    (out_dir / "compare.html").write_text(VIEWER_PAGE.replace("__DATA__", data), encoding="utf-8")


# ── 실행 ────────────────────────────────────────────────────────────────

@dataclass
class Job:
    stem: str
    scale: float
    src: Image.Image  # 모델에 넣는 입력
    target: Tuple[int, int]
    gt: Optional[Image.Image]
    input_rel: str
    gt_rel: Optional[str]

    @property
    def key(self) -> str:
        return f"{self.stem}|x{scale_label(self.scale)}"

    def result_rel(self, model: str) -> str:
        return f"{self.stem}/x{scale_label(self.scale)}/{model}.png"


def build_jobs(images: List[Path], scales: List[float], args, out_dir: Path) -> List[Job]:
    jobs = []
    for path in images:
        # 비교는 RGB로 한다(투명 정보는 버린다). 워커는 알파를 따로 키우지만 모델 차이를 보는 데는 RGB면 충분하다
        image = Image.open(path).convert("RGB")
        stem_dir = out_dir / path.stem
        stem_dir.mkdir(parents=True, exist_ok=True)
        if not args.gt:
            image.save(stem_dir / "input.png")
        else:
            image.save(stem_dir / "gt.png")
        for scale in scales:
            scale_dir = stem_dir / f"x{scale_label(scale)}"
            scale_dir.mkdir(exist_ok=True)
            if not args.gt:
                target = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
                jobs.append(Job(path.stem, scale, image, target, None, f"{path.stem}/input.png", None))
                continue
            small = (round(image.width / scale), round(image.height / scale))
            if min(small) < MIN_GT_INPUT_SIDE:
                print(f"건너뜀: {path.name} x{scale_label(scale)} — 줄인 입력이 {small[0]}x{small[1]}로 너무 작음")
                continue
            low = image.resize(small, Image.BICUBIC)
            if args.gt_jpeg:
                buffer = io.BytesIO()
                low.save(buffer, "JPEG", quality=args.gt_jpeg)
                low = Image.open(io.BytesIO(buffer.getvalue())).convert("RGB")
            input_rel = f"{path.stem}/x{scale_label(scale)}/input.png"
            low.save(out_dir / input_rel)
            jobs.append(Job(path.stem, scale, low, image.size, image, input_rel, f"{path.stem}/gt.png"))
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description="업스케일 모델 비교(RealESRGAN anime 6B vs 2x-AnimeSharpV4)")
    parser.add_argument("--input", required=True, help="비교할 이미지 폴더(또는 이미지 파일 하나)")
    parser.add_argument("--output", required=True, help="결과 폴더")
    parser.add_argument("--models", default="realesrgan_anime6b,animesharp_v4,lanczos",
                        help=f"쉼표로 구분 (가능: {', '.join([*MODEL_SPECS, BASELINE])})")
    parser.add_argument("--scales", default="1.5,2,4", help="쉼표로 구분한 배율")
    parser.add_argument("--gt", action="store_true", help="입력을 정답으로 보고 줄였다 키워 정답 PSNR을 잰다")
    parser.add_argument("--gt-jpeg", type=int, default=None, help="--gt에서 줄인 입력을 이 품질의 JPEG로 한 번 열화(예: 80)")
    parser.add_argument("--weights-dir", default="./weights", help="가중치 폴더(없으면 내려받아 여기 둔다)")
    parser.add_argument("--no-download", action="store_true", help="가중치를 내려받지 않는다(없으면 바로 실패)")
    parser.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    parser.add_argument("--no-cudnn", action="store_true",
                        help="cuDNN을 끈다 — upscaler 서버 가상환경(torch 2.0.1+cu117)에서 돌릴 때 필요")
    parser.add_argument("--tile", type=int, default=1024, help="타일 크기(입력 px, 워커와 같음)")
    parser.add_argument("--tile-overlap", type=int, default=128, help="타일 겹침(입력 px, 워커와 같음)")
    parser.add_argument("--crop", type=int, default=256, help="비교 시트에서 자를 구간 크기(결과 기준 px)")
    parser.add_argument("--crops", type=int, default=3, help="이미지·배율마다 자를 구간 수")
    parser.add_argument("--zoom", type=int, default=2, help="비교 시트에서 자른 구간을 키우는 배수(픽셀 그대로)")
    parser.add_argument("--rerun", action="store_true", help="같은 --output에 이미 끝난 결과도 다시 돌린다")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in models if m not in MODEL_SPECS and m != BASELINE]
    if unknown:
        parser.error(f"모르는 모델: {unknown} (가능: {[*MODEL_SPECS, BASELINE]})")
    if len(set(models)) != len(models):
        parser.error(f"모델이 중복됐습니다: {models}")
    try:
        scales = [float(s) for s in args.scales.split(",") if s.strip()]
    except ValueError:
        parser.error(f"배율은 숫자여야 합니다: {args.scales}")
    if not scales or any(not np.isfinite(s) or s <= 1 for s in scales):
        parser.error(f"배율은 1보다 커야 합니다: {args.scales}")
    if args.gt_jpeg is not None and not (1 <= args.gt_jpeg <= 100):
        parser.error("--gt-jpeg는 1~100")
    if args.gt_jpeg is not None and not args.gt:
        parser.error("--gt-jpeg는 --gt와 같이 써야 합니다")
    if args.tile <= 2 * args.tile_overlap:
        parser.error("--tile은 --tile-overlap의 두 배보다 커야 합니다")

    src = Path(args.input)
    images = [src] if src.is_file() else sorted(p for p in src.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        parser.error(f"이미지가 없습니다: {src}")
    stems = [p.stem for p in images]
    duplicated = sorted({s for s in stems if stems.count(s) > 1})
    if duplicated:
        parser.error(f"확장자만 다른 같은 이름의 이미지가 있습니다(결과 폴더가 겹침): {duplicated}")

    # 앞 모델을 다 돌린 뒤에야 가중치가 없다고 실패하지 않게 미리 받아 둔다
    weights_dir = Path(args.weights_dir)
    weight_paths = {m: ensure_weights(MODEL_SPECS[m], weights_dir, not args.no_download) for m in models if m in MODEL_SPECS}

    device = pick_device(args.device)
    if args.no_cudnn:
        torch.backends.cudnn.enabled = False
    out_dir = Path(args.output)
    (out_dir / "sheets").mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "_cache"
    cache_dir.mkdir(exist_ok=True)
    mode = f"--gt (jpeg={args.gt_jpeg})" if args.gt else "기본"
    print(f"device={device} cudnn={torch.backends.cudnn.enabled} images={len(images)} scales={scales} "
          f"models={models} 방식={mode}")

    jobs = build_jobs(images, scales, args, out_dir)
    if not jobs:
        parser.error("돌릴 작업이 없습니다")

    # 모델을 하나씩 올렸다 내린다. 결과마다 바로 저장해, 중간에 실패해도 끝난 결과는 남고 다시 실행하면 건너뛴다
    times: Dict[str, Dict[str, float]] = {}
    peak_mem: Dict[str, Optional[float]] = {}
    for name in models:
        cache_path = cache_dir / f"{name}.json"
        cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.is_file() else {}
        runs: Dict[str, float] = {} if args.rerun else dict(cache.get("runs", {}))
        todo = [j for j in jobs if j.key not in runs or not (out_dir / j.result_rel(name)).is_file()]
        peak_mem[name] = cache.get("peak_mem_gb")
        if todo:
            print(f"[{name}] 로딩... (남은 작업 {len(todo)}/{len(jobs)})")
            if device == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            upscaler = (LanczosUpscaler() if name == BASELINE else
                        SpandrelUpscaler(name, MODEL_SPECS[name], weight_paths[name], device, args.tile, args.tile_overlap))
            upscaler.upscale(Image.new("RGB", (64, 64), (128, 128, 128)), (128, 128))  # 예열(첫 실행의 초기화 시간 제외)
            sync(device)
            for job in todo:
                start = time.perf_counter()
                result = upscaler.upscale(job.src, job.target)
                sync(device)
                runs[job.key] = (time.perf_counter() - start) * 1000
                result.save(out_dir / job.result_rel(name))
                print(f"[{name}] {job.stem} x{scale_label(job.scale)} {job.src.width}x{job.src.height} → "
                      f"{result.width}x{result.height}: {runs[job.key]:.0f} ms")
                cache_path.write_text(json.dumps({"runs": runs, "peak_mem_gb": peak_mem[name]}), encoding="utf-8")
            if device == "cuda" and name != BASELINE:
                peak_mem[name] = torch.cuda.max_memory_allocated() / 2**30
                cache_path.write_text(json.dumps({"runs": runs, "peak_mem_gb": peak_mem[name]}), encoding="utf-8")
            del upscaler
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
        else:
            print(f"[{name}] 이전 결과 사용(다시 돌리려면 --rerun)")
        times[name] = runs

    rows, items = [], []
    for job in jobs:
        results = {m: Image.open(out_dir / job.result_rel(m)).convert("RGB") for m in models}
        metrics = {}
        for m, img in results.items():
            metrics[m] = {
                "ms": times[m][job.key],
                "consistency": psnr(img.resize(job.src.size, Image.BOX), job.src),
                "sharpness": sharpness(img),
                "gt_psnr": psnr(img, job.gt, "L") if job.gt is not None else None,
            }
            rows.append({"image": job.stem, "scale": scale_label(job.scale), "model": m,
                         "ms": f"{metrics[m]['ms']:.1f}", "consistency_psnr": f"{metrics[m]['consistency']:.3f}",
                         "sharpness": f"{metrics[m]['sharpness']:.3f}",
                         "gt_psnr": "" if job.gt is None else f"{metrics[m]['gt_psnr']:.3f}"})

        columns = [(f"{m}  {metrics[m]['ms']:.0f}ms", results[m]) for m in models]
        if job.gt is not None:
            columns.append(("ground truth", job.gt))
        boxes = detail_windows(job.src, max(8, round(args.crop * job.src.width / job.target[0])), args.crops)
        title = f"{job.stem}  x{scale_label(job.scale)}  {job.src.width}x{job.src.height} -> {job.target[0]}x{job.target[1]}"
        make_sheet(title, job.src, columns, boxes, args.zoom).save(
            out_dir / "sheets" / f"{job.stem}_x{scale_label(job.scale)}.jpg", quality=92)

        items.append({"id": job.key, "stem": job.stem, "scale": scale_label(job.scale),
                      "w": job.target[0], "h": job.target[1], "inW": job.src.width, "inH": job.src.height,
                      "input": job.input_rel, "gt": job.gt_rel,
                      "files": {m: job.result_rel(m) for m in models}, "metrics": metrics})

    with open(out_dir / "metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "scale", "model", "ms", "consistency_psnr", "sharpness", "gt_psnr"])
        writer.writeheader()
        writer.writerows(rows)
    write_viewer(out_dir, items, models)

    print("\n요약 (배율별 평균)")
    for scale in scales:
        label = scale_label(scale)
        scale_items = [it for it in items if it["scale"] == label]
        if not scale_items:
            continue
        print(f"  x{label}")
        for m in models:
            values = [it["metrics"][m] for it in scale_items]
            line = (f"    {m}: {np.mean([v['ms'] for v in values]):.0f} ms, "
                    f"원본 일치도 {np.mean([v['consistency'] for v in values]):.2f} dB, "
                    f"선명도 {np.mean([v['sharpness'] for v in values]):.2f}")
            if args.gt:
                line += f", 정답 PSNR {np.mean([v['gt_psnr'] for v in values]):.2f} dB"
            print(line)
    for m in models:
        if peak_mem.get(m) is not None:
            print(f"  {m}: 최대 GPU 메모리 {peak_mem[m]:.2f} GB")
    print(f"결과: {out_dir.resolve()} (compare.html을 브라우저로 열어 보세요)")


if __name__ == "__main__":
    main()
