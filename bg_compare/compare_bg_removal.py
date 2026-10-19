#!/usr/bin/env python3
"""배경 제거 모델 비교 — InSPyReNet(현재 bg-remover 워커) vs BiRefNet(+ 선택: BiRefNet_HR, ToonOut).

프로젝트 코드와 무관한 단독 스크립트다. 같은 이미지를 모델마다 돌려 결과·마스크·비교 시트·처리 시간을 남긴다.
후처리는 현재 워커와 같다(blur_sigma > 0이면 마스크에 가우시안 블러, 0이면 127 이진화) — 모델 차이만 보이게.

모델(--models, 쉼표 구분):
  inspyrenet          현재 워커와 같음(1024로 줄여 추론)
  inspyrenet_dynamic  InSPyReNet을 원본 해상도에 맞춰 추론
  birefnet            예전 워커와 같음(1024)
  birefnet_hr         BiRefNet 고해상도판(2048)
  birefnet_dynamic    BiRefNet 가변 해상도판(원본 크기, 긴 변 2560 이하)
  toonout             BiRefNet 애니 추가 학습판(--toonout-weights 필요)
  어느 이름이든 뒤에 +tta를 붙이면 좌우 반전 TTA(예: birefnet_hr+tta)

사용 예:
  python compare_bg_removal.py --input ./samples --output ./out
  python compare_bg_removal.py --input ./samples --output ./out --models inspyrenet,birefnet,birefnet_hr
  python compare_bg_removal.py --input ./samples --output ./out --models inspyrenet,birefnet,toonout \\
      --toonout-weights ./birefnet_finetuned_toonout.pth

결과(--output 아래):
  <이미지명>/<모델>.png        배경 제거 결과(RGBA, 워커와 같은 후처리)
  <이미지명>/<모델>_mask.png   모델이 낸 원본 마스크(후처리 전)
  sheets/<이미지명>.jpg         원본 | 모델별 결과(바둑판 위) / 모델별 마스크 비교 시트
  timings.csv                   이미지·모델별 처리 시간(ms)
  index.html                    시트 모아 보기
  rate.html                     블라인드 평가 — 모델 이름을 가리고 컷마다 가장 나은 결과를 고른다(키: 1~9 선택, 0 비슷함,
                                X 모두 나쁨, B 배경 바꾸기, ←/→ 이동). 선택은 브라우저에 저장되고 '결과 보기'에서 득표를 본다
"""

import argparse
import csv
import gc
import html
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


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


def postprocess(mask: np.ndarray, blur_sigma: float) -> np.ndarray:
    """현재 워커(_remove_background)와 같은 후처리."""
    if blur_sigma > 0:
        return cv2.GaussianBlur(mask, (0, 0), sigmaX=float(blur_sigma))
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return binary


def checkerboard(size, cell: int = 16) -> Image.Image:
    w, h = size
    ys, xs = np.mgrid[0:h, 0:w]
    board = np.where(((xs // cell) + (ys // cell)) % 2 == 0, 255, 226).astype(np.uint8)
    return Image.fromarray(np.stack([board] * 3, axis=-1))


# ── 모델 ────────────────────────────────────────────────────────────────

class InSPyReNetModel:
    """현재 워커와 같은 방식 — transparent-background Remover, type='map'."""

    def __init__(self, device: str, mode: str, ckpt: Optional[str], resize: str = "static"):
        from transparent_background import Remover
        kwargs = {"mode": mode, "device": device}
        # 워커는 S3에서 받은 ckpt_base.pth를 쓴다. 안 주면 패키지가 같은 base 가중치를 직접 내려받는다
        if ckpt:
            kwargs["ckpt"] = ckpt
        # static(워커와 같음): 1024로 줄여 추론 / dynamic: 원본 해상도에 맞춰 추론(고해상도 경계에 유리)
        if resize != "static":
            kwargs["resize"] = resize
        try:
            self.remover = Remover(**kwargs)
        except TypeError as exc:
            raise RuntimeError(f"이 transparent-background 버전은 resize='{resize}'를 받지 않습니다: {exc}") from exc

    def mask(self, image: Image.Image) -> np.ndarray:
        return np.array(self.remover.process(image, type="map").convert("L"))


def _patch_transformers_for_birefnet() -> None:
    # 예전 bg-remover 워커(BiRefNet 시절)와 같은 호환성 패치 — 일부 transformers 버전에서 BiRefNet 원격 코드가
    # config.is_encoder_decoder 조회로 실패하는 문제를 막는다
    import transformers.configuration_utils as cu
    if getattr(cu.PretrainedConfig, "_birefnet_patched", False):
        return

    def _getattribute(self, key):
        if key == "is_encoder_decoder":
            return False
        return object.__getattribute__(self, key)

    cu.PretrainedConfig.__getattribute__ = _getattribute
    cu.PretrainedConfig._birefnet_patched = True


class BiRefNetModel:
    """BiRefNet(Hugging Face 원격 코드). weights를 주면 같은 구조에 그 가중치를 덮어쓴다(ToonOut 등)."""

    # resolution=None이면 원본 크기(32의 배수, 긴 변 NATIVE_MAX_SIDE 이하)로 추론한다 — BiRefNet_dynamic용
    NATIVE_MAX_SIDE = 2560

    def __init__(self, device: str, repo: str, resolution: Optional[int], fp16: bool, weights: Optional[str] = None):
        _patch_transformers_for_birefnet()
        from transformers import AutoModelForImageSegmentation
        from torchvision import transforms

        model = AutoModelForImageSegmentation.from_pretrained(repo, trust_remote_code=True)
        if weights:
            state = torch.load(weights, map_location="cpu", weights_only=True)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            # 학습 체크포인트에 붙는 접두사(torch.compile·DataParallel)를 뗀다
            state = {k.replace("_orig_mod.", "").replace("module.", "", 1): v for k, v in state.items()}
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing or unexpected:
                # 일부만 맞는 가중치로 조용히 비교하지 않는다
                raise RuntimeError(
                    f"가중치가 {repo} 구조와 맞지 않습니다: missing={len(missing)} unexpected={len(unexpected)} "
                    f"(예: {(missing or unexpected)[:3]})"
                )
        self.dtype = torch.float16 if (fp16 and device == "cuda") else torch.float32
        self.model = model.to(device=device, dtype=self.dtype).eval()
        self.device = device
        self.resolution = resolution
        self.transforms = transforms
        self.to_pil = transforms.ToPILImage()

    def _input_size(self, image: Image.Image):
        if self.resolution:
            return self.resolution, self.resolution
        scale = min(1.0, self.NATIVE_MAX_SIDE / max(image.size))
        w, h = (max(32, round(v * scale / 32) * 32) for v in image.size)
        return h, w

    def mask(self, image: Image.Image) -> np.ndarray:
        t = self.transforms
        to_tensor = t.Compose([
            t.Resize(self._input_size(image)),
            t.ToTensor(),
            t.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        x = to_tensor(image).unsqueeze(0).to(device=self.device, dtype=self.dtype)
        with torch.no_grad():
            pred = self.model(x)[-1].sigmoid().float().cpu()[0].squeeze()
        return np.array(self.to_pil(pred).resize(image.size, Image.BILINEAR))


class FlipTTA:
    """좌우 반전 TTA — 원본과 반전 이미지의 마스크를 평균한다(한쪽에서만 틀린 잡음·잔상이 줄어든다)."""

    def __init__(self, model):
        self.model = model

    def mask(self, image: Image.Image) -> np.ndarray:
        plain = self.model.mask(image).astype(np.float32)
        flipped = self.model.mask(image.transpose(Image.FLIP_LEFT_RIGHT))[:, ::-1].astype(np.float32)
        return np.clip((plain + flipped) / 2, 0, 255).astype(np.uint8)


TTA_SUFFIX = "+tta"


def build_loaders(args, device: str) -> Dict[str, Callable[[], object]]:
    base = {
        "inspyrenet": lambda: InSPyReNetModel(device, args.inspyrenet_mode, args.inspyrenet_ckpt),
        "inspyrenet_dynamic": lambda: InSPyReNetModel(device, args.inspyrenet_mode, args.inspyrenet_ckpt, resize="dynamic"),
        "birefnet": lambda: BiRefNetModel(device, "ZhengPeng7/BiRefNet", 1024, args.fp16),
        "birefnet_hr": lambda: BiRefNetModel(device, "ZhengPeng7/BiRefNet_HR", 2048, args.fp16),
        "birefnet_dynamic": lambda: BiRefNetModel(device, "ZhengPeng7/BiRefNet_dynamic", None, args.fp16),
        "toonout": lambda: BiRefNetModel(device, "ZhengPeng7/BiRefNet", 1024, args.fp16, weights=args.toonout_weights),
    }
    # 어느 모델이든 이름 뒤에 +tta를 붙이면 좌우 반전 TTA로 돌린다(예: birefnet_hr+tta)
    with_tta = {f"{name}{TTA_SUFFIX}": (lambda make=make: FlipTTA(make())) for name, make in base.items()}
    return {**base, **with_tta}


# ── 비교 시트 ───────────────────────────────────────────────────────────

def make_sheet(original: Image.Image, results: Dict[str, Image.Image], masks: Dict[str, np.ndarray],
               timings: Dict[str, float], thumb: int) -> Image.Image:
    scale = thumb / max(original.size)
    tw, th = max(1, round(original.width * scale)), max(1, round(original.height * scale))
    label_h, gap = 28, 8
    cols = 1 + len(results)
    sheet = Image.new("RGB", (cols * (tw + gap) + gap, 2 * (th + label_h + gap) + gap), (250, 250, 252))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except OSError:
        font = ImageFont.load_default()

    def put(img: Image.Image, col: int, row: int, label: str) -> None:
        x = gap + col * (tw + gap)
        y = gap + row * (th + label_h + gap)
        draw.text((x, y + 4), label, fill=(40, 40, 50), font=font)
        sheet.paste(img.resize((tw, th), Image.LANCZOS), (x, y + label_h))

    put(original, 0, 0, "original")
    for i, name in enumerate(results, start=1):
        board = checkerboard((tw, th))
        cut = results[name].resize((tw, th), Image.LANCZOS)
        board.paste(cut, (0, 0), cut)
        put(board, i, 0, f"{name}  {timings[name]:.0f} ms")
        put(Image.fromarray(masks[name]).convert("RGB"), i, 1, f"{name} mask")
    return sheet


def write_index(out_dir: Path, rows: List[dict], models: List[str]) -> None:
    avg = {m: np.mean([r[m] for r in rows]) for m in models}
    head = " · ".join(f"{html.escape(m)} 평균 {avg[m]:.0f} ms" for m in models)
    items = "\n".join(
        f'<h3>{html.escape(r["image"])}</h3><img src="sheets/{html.escape(r["sheet"])}" loading="lazy">'
        for r in rows
    )
    (out_dir / "index.html").write_text(
        f"""<!doctype html><meta charset="utf-8"><title>배경 제거 비교</title>
<style>body{{font-family:sans-serif;margin:24px;background:#f6f6f8}}img{{max-width:100%;border:1px solid #ddd}}</style>
<h1>배경 제거 비교</h1><p>{head}</p>{items}""",
        encoding="utf-8",
    )


RATE_PAGE = r"""<!doctype html><meta charset="utf-8"><title>배경 제거 블라인드 평가</title>
<style>
  :root{--bg:#f4f4f6;--fg:#1d1d22;--muted:#6b6b76;--line:#d9d9e0;--accent:#5b43dc}
  body{margin:0;font-family:system-ui,sans-serif;background:var(--bg);color:var(--fg)}
  header{display:flex;flex-wrap:wrap;gap:12px;align-items:center;padding:12px 16px;border-bottom:1px solid var(--line);background:#fff;position:sticky;top:0}
  header b{font-size:15px} .muted{color:var(--muted);font-size:13px}
  button{font:inherit;border:1px solid var(--line);background:#fff;border-radius:8px;padding:6px 12px;cursor:pointer}
  button.on{border-color:var(--accent);background:#efebff;color:var(--accent);font-weight:600}
  main{padding:16px}
  .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px}
  figure{margin:0;background:#fff;border:2px solid var(--line);border-radius:10px;overflow:hidden;cursor:pointer}
  figure.picked{border-color:var(--accent);box-shadow:0 0 0 3px #e2dbff}
  figcaption{padding:8px 10px;font-weight:600;display:flex;justify-content:space-between}
  .stage{display:flex;align-items:center;justify-content:center;height:62vh}
  .stage img{max-width:100%;max-height:100%;display:block}
  .checker{background:repeating-conic-gradient(#e2e2e8 0 25%,#fff 0 50%) 0 0/20px 20px}
  .black{background:#000}.white{background:#fff}.magenta{background:#ff00c8}
  table{border-collapse:collapse;margin-top:12px;background:#fff}td,th{border:1px solid var(--line);padding:6px 12px;text-align:left}
</style>
<header>
  <b>배경 제거 블라인드 평가</b><span id="pos" class="muted"></span>
  <span class="muted">배경</span><span id="bgs"></span>
  <button id="prev">← 이전</button><button id="next">다음 →</button>
  <button id="tie">비슷함 (0)</button><button id="bad">모두 나쁨 (X)</button>
  <button id="show">결과 보기</button>
</header>
<main id="main"></main>
<script>
const DATA = __DATA__;
const KEY = 'bg-rate:' + location.pathname;
const BGS = ['checker', 'black', 'white', 'magenta'];
let votes = {}, idx = 0, bg = 'checker';
try { votes = JSON.parse(localStorage.getItem(KEY) || '{}'); } catch (e) { votes = {}; }
const save = () => { try { localStorage.setItem(KEY, JSON.stringify(votes)); } catch (e) {} };
// 컷마다 정해진 순서로 섞는다(다시 열어도 같은 글자) — 모델 이름은 결과 보기 전까지 숨긴다
function order(stem) {
  let h = 0; for (const c of stem) h = (h * 31 + c.charCodeAt(0)) >>> 0;
  const arr = DATA.models.slice();
  for (let i = arr.length - 1; i > 0; i--) { h = (h * 1103515245 + 12345) >>> 0; const j = h % (i + 1); [arr[i], arr[j]] = [arr[j], arr[i]]; }
  return arr;
}
const letter = i => String.fromCharCode(65 + i);
function render() {
  const cut = DATA.cuts[idx], models = order(cut);
  document.getElementById('pos').textContent = `${idx + 1} / ${DATA.cuts.length} · ${cut} · 평가 ${Object.keys(votes).length}`;
  document.getElementById('bgs').innerHTML = BGS.map(b => `<button class="${b === bg ? 'on' : ''}" data-bg="${b}">${b}</button>`).join(' ');
  const pick = votes[cut];
  const cards = [`<figure><figcaption>원본</figcaption><div class="stage white"><img src="${cut}/original.jpg"></div></figure>`]
    .concat(models.map((m, i) => `<figure data-model="${m}" class="${pick === m ? 'picked' : ''}"><figcaption><span>${letter(i)}</span><span class="muted">${i + 1}</span></figcaption>
      <div class="stage ${bg}"><img src="${cut}/${encodeURIComponent(m)}.png"></div></figure>`));
  document.getElementById('main').innerHTML = `<div class="grid">${cards.join('')}</div>`;
  document.getElementById('tie').className = pick === 'tie' ? 'on' : '';
  document.getElementById('bad').className = pick === 'bad' ? 'on' : '';
}
function vote(v) { const cut = DATA.cuts[idx]; votes[cut] = votes[cut] === v ? undefined : v; if (!votes[cut]) delete votes[cut]; save(); if (votes[cut] && idx < DATA.cuts.length - 1) idx++; render(); }
function results() {
  const tally = Object.fromEntries([...DATA.models, 'tie', 'bad'].map(k => [k, 0]));
  Object.values(votes).forEach(v => { if (v in tally) tally[v]++; });
  const rows = Object.entries(tally).map(([k, n]) => `<tr><td>${k === 'tie' ? '비슷함' : k === 'bad' ? '모두 나쁨' : k}</td><td>${n}</td></tr>`).join('');
  const csv = 'cut,choice\n' + DATA.cuts.map(c => `${c},${votes[c] || ''}`).join('\n');
  const url = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }));
  document.getElementById('main').innerHTML = `<h2>결과 (평가 ${Object.keys(votes).length} / ${DATA.cuts.length})</h2>
    <table><tr><th>선택</th><th>컷 수</th></tr>${rows}</table><p><a href="${url}" download="ratings.csv">컷별 선택 CSV 받기</a> · <button id="back">평가로 돌아가기</button></p>`;
  document.getElementById('back').onclick = render;
}
document.addEventListener('click', e => {
  const b = e.target.closest('[data-bg]'); if (b) { bg = b.dataset.bg; render(); return; }
  const f = e.target.closest('figure[data-model]'); if (f) vote(f.dataset.model);
});
document.getElementById('prev').onclick = () => { idx = Math.max(0, idx - 1); render(); };
document.getElementById('next').onclick = () => { idx = Math.min(DATA.cuts.length - 1, idx + 1); render(); };
document.getElementById('tie').onclick = () => vote('tie');
document.getElementById('bad').onclick = () => vote('bad');
document.getElementById('show').onclick = results;
document.addEventListener('keydown', e => {
  const models = order(DATA.cuts[idx]);
  if (e.key === 'ArrowLeft') document.getElementById('prev').click();
  else if (e.key === 'ArrowRight') document.getElementById('next').click();
  else if (e.key === '0') vote('tie');
  else if (e.key.toLowerCase() === 'x') vote('bad');
  else if (e.key.toLowerCase() === 'b') { bg = BGS[(BGS.indexOf(bg) + 1) % BGS.length]; render(); }
  else if (/^[1-9]$/.test(e.key) && models[+e.key - 1]) vote(models[+e.key - 1]);
});
render();
</script>"""


def write_rate_page(out_dir: Path, stems: List[str], models: List[str]) -> None:
    import json
    data = json.dumps({"cuts": stems, "models": models}, ensure_ascii=False)
    (out_dir / "rate.html").write_text(RATE_PAGE.replace("__DATA__", data), encoding="utf-8")


# ── 실행 ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="배경 제거 모델 비교(InSPyReNet vs BiRefNet)")
    parser.add_argument("--input", required=True, help="비교할 이미지 폴더(또는 이미지 파일 하나)")
    parser.add_argument("--output", required=True, help="결과 폴더")
    parser.add_argument("--models", default="inspyrenet,birefnet",
                        help="쉼표로 구분: inspyrenet, birefnet, birefnet_hr, toonout")
    parser.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    parser.add_argument("--blur-sigma", type=float, default=1.5, help="워커 후처리와 같은 값(서버 기본 1.5, 0이면 이진화)")
    parser.add_argument("--inspyrenet-mode", default="base", help="워커 기본 base")
    parser.add_argument("--inspyrenet-ckpt", default=None, help="워커가 쓰는 ckpt_base.pth 경로(없으면 패키지가 내려받음)")
    parser.add_argument("--toonout-weights", default=None, help="birefnet_finetuned_toonout.pth 경로(toonout일 때 필수)")
    parser.add_argument("--no-fp16", dest="fp16", action="store_false", help="BiRefNet을 fp32로(기본은 CUDA에서 fp16 — 예전 워커와 같음)")
    parser.add_argument("--warmup", type=int, default=1, help="시간 측정 전 예열 횟수(첫 실행의 초기화 시간을 빼려고)")
    parser.add_argument("--thumb", type=int, default=640, help="비교 시트의 칸 크기(긴 변 px)")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    device = pick_device(args.device)
    loaders = build_loaders(args, device)
    unknown = [m for m in models if m not in loaders]
    if unknown:
        parser.error(f"모르는 모델: {unknown} (가능: {list(loaders)})")
    if any(m.startswith("toonout") for m in models) and not args.toonout_weights:
        parser.error("toonout을 비교하려면 --toonout-weights가 필요합니다")

    src = Path(args.input)
    images = [src] if src.is_file() else sorted(p for p in src.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        parser.error(f"이미지가 없습니다: {src}")
    out_dir = Path(args.output)
    (out_dir / "sheets").mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    print(f"device={device} images={len(images)} models={models} blur_sigma={args.blur_sigma}")

    # 워커와 같이 RGB로 연다(원본의 투명 정보는 버린다)
    originals = {p: Image.open(p).convert("RGB") for p in images}
    masks: Dict[str, Dict[Path, np.ndarray]] = {}
    times: Dict[str, Dict[Path, float]] = {}
    peak_mem: Dict[str, Optional[float]] = {}

    # 모델을 하나씩 올렸다 내려 GPU 메모리를 서로 나눠 쓰지 않게 한다
    for name in models:
        print(f"[{name}] 로딩...")
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        model = loaders[name]()
        first = originals[images[0]]
        for _ in range(max(0, args.warmup)):
            model.mask(first)
        sync(device)
        masks[name], times[name] = {}, {}
        for path in images:
            start = time.perf_counter()
            masks[name][path] = model.mask(originals[path])
            sync(device)
            times[name][path] = (time.perf_counter() - start) * 1000
            print(f"[{name}] {path.name}: {times[name][path]:.0f} ms")
        peak_mem[name] = torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else None
        del model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    rows = []
    for path in images:
        original = originals[path]
        item_dir = out_dir / path.stem
        item_dir.mkdir(exist_ok=True)
        original.save(item_dir / "original.jpg", quality=95)  # 평가 페이지에서 나란히 본다
        results = {}
        for name in models:
            raw = masks[name][path]
            Image.fromarray(raw).save(item_dir / f"{name}_mask.png")
            cut = original.copy()
            cut.putalpha(Image.fromarray(postprocess(raw, args.blur_sigma)))
            cut.save(item_dir / f"{name}.png")
            results[name] = cut
        sheet_name = f"{path.stem}.jpg"
        make_sheet(original, results, {m: masks[m][path] for m in models},
                   {m: times[m][path] for m in models}, args.thumb).save(out_dir / "sheets" / sheet_name, quality=92)
        rows.append({"image": path.name, "sheet": sheet_name, **{m: times[m][path] for m in models}})

    with open(out_dir / "timings.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", *models])
        writer.writeheader()
        for r in rows:
            writer.writerow({"image": r["image"], **{m: f"{r[m]:.1f}" for m in models}})
    write_index(out_dir, rows, models)
    write_rate_page(out_dir, [p.stem for p in images], models)

    print("\n요약")
    for name in models:
        mem = f", 최대 GPU 메모리 {peak_mem[name]:.2f} GB" if peak_mem[name] is not None else ""
        print(f"  {name}: 평균 {np.mean(list(times[name].values())):.0f} ms{mem}")
    print(f"결과: {out_dir.resolve()} (index.html을 브라우저로 열어 보세요)")


if __name__ == "__main__":
    main()
