from __future__ import annotations

import argparse
import gc
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import torch

from losses import PrintCleanLoss
from model_scalable import ScaledTinyUNet, describe
from train import build_cache, load_manifest, load_page_tensor

INPUT_SIZE = 512
WIDTHS = [16, 24, 32, 48, 64]

def parse_args():
    p = argparse.ArgumentParser(description="Process-isolated 512px quality benchmark.")
    base = Path(__file__).resolve().parent
    p.add_argument("--data-dir", default=str(base / "data"))
    p.add_argument("--train-pages", type=int, default=64)
    p.add_argument("--val-pages", type=int, default=16)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--min-free-ram-mib", type=int, default=768)
    p.add_argument("--max-rss-mib", type=int, default=1536)
    p.add_argument("--preview-pages", type=int, default=2)
    p.add_argument("--worker", action="store_true")
    p.add_argument("--width", type=int, default=16)
    p.add_argument("--output", default=None)
    return p.parse_args()

def split_records(records, train_count, val_count):
    total = train_count + val_count
    if total > len(records):
        raise ValueError("Not enough pages for requested split.")
    idx = np.linspace(0, len(records)-1, total, dtype=int)
    chosen = [records[int(i)] for i in idx]
    return chosen[::2][:train_count], chosen[1::2][:val_count]

def load_pair(record, cache_root):
    op, yp = record.cache_paths(cache_root)
    return load_page_tensor(op, INPUT_SIZE), load_page_tensor(yp, INPUT_SIZE)

def guard(process, args):
    rss = process.memory_info().rss / 1048576
    free = psutil.virtual_memory().available / 1048576
    if rss > args.max_rss_mib or free < args.min_free_ram_mib:
        raise MemoryError("RAM guard: rss={:.1f} MiB free={:.1f} MiB".format(rss, free))
    return rss, free

def calibrate_bn(model, records, cache_root):
    bns = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    old = [m.momentum for m in bns]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None
    model.train()
    with torch.no_grad():
        for record in records:
            x, _ = load_pair(record, cache_root)
            model(x)
            del x
    for m, momentum in zip(bns, old):
        m.momentum = momentum
    model.eval()

def metrics(pred, target):
    p = pred.detach().cpu().squeeze(0).permute(1,2,0).clamp(0,1).numpy()
    y = target.detach().cpu().squeeze(0).permute(1,2,0).clamp(0,1).numpy()
    mae = float(np.mean(np.abs(p-y)))
    mse = float(np.mean((p-y)**2))
    pl = (0.2126*p[...,0] + 0.7152*p[...,1] + 0.0722*p[...,2]) < 0.75
    yl = (0.2126*y[...,0] + 0.7152*y[...,1] + 0.0722*y[...,2]) < 0.75
    tp = np.count_nonzero(pl & yl); fp = np.count_nonzero(pl & ~yl); fn = np.count_nonzero(~pl & yl)
    f1 = 1.0 if (2*tp+fp+fn)==0 else float(2*tp/max(1,2*tp+fp+fn))
    return mae, mse, f1

def arr(t):
    return t.detach().cpu().squeeze(0).permute(1,2,0).clamp(0,1).mul(255).round().to(torch.uint8).numpy()

def preview(model, record, cache_root, path):
    from PIL import Image, ImageDraw
    with torch.no_grad():
        x, y = load_pair(record, cache_root); pred = model(x)
    canvas = Image.new("RGB", (1536, 512), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (label, tensor) in enumerate((("Original", x), ("Target", y), ("Prediction", pred))):
        image = Image.fromarray(arr(tensor)).resize((512,512), Image.Resampling.NEAREST)
        canvas.paste(image, (i*512, 0)); draw.text((i*512+8,8), "{} page {}".format(label, record.page_number), fill="red")
    canvas.save(path, "PNG", optimize=True)
    del x,y,pred

def worker(args):
    torch.set_num_threads(args.threads)
    try: torch.set_num_interop_threads(1)
    except RuntimeError: pass
    data_dir = Path(args.data_dir).resolve(); cache_root = data_dir / "cache512"
    _, records = load_manifest(data_dir)
    build_cache(records, cache_root, INPUT_SIZE)
    train_records, val_records = split_records(records, args.train_pages, args.val_pages)
    out_dir = Path(args.output).resolve(); out_dir.mkdir(parents=True, exist_ok=True)
    process = psutil.Process(os.getpid())
    model = ScaledTinyUNet(INPUT_SIZE, args.width, False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, eps=1e-7)
    criterion = PrintCleanLoss(); model.train()
    started = time.perf_counter(); peak = process.memory_info().rss; min_free = psutil.virtual_memory().available
    losses=[]
    try:
        for _ in range(args.epochs):
            for record in train_records:
                guard(process,args)
                x,y=load_pair(record,cache_root)
                optimizer.zero_grad(set_to_none=True); pred=model(x); loss=criterion(pred,y); loss.backward(); optimizer.step()
                losses.append(float(loss.detach().cpu().item()))
                del pred,loss,x,y
                peak=max(peak,process.memory_info().rss); min_free=min(min_free,psutil.virtual_memory().available)
        calibrate_bn(model,train_records,cache_root)
        rows=[]
        with torch.no_grad():
            for record in val_records:
                guard(process,args); x,y=load_pair(record,cache_root); pred=model(x)
                mae,mse,f1=metrics(pred,y); rows.append({"page":record.page_number,"mae":mae,"mse":mse,"dark_f1":f1})
                del pred,x,y
                peak=max(peak,process.memory_info().rss); min_free=min(min_free,psutil.virtual_memory().available)
        mean_mae=float(np.mean([r["mae"] for r in rows])); mean_mse=float(np.mean([r["mse"] for r in rows])); mean_f1=float(np.mean([r["dark_f1"] for r in rows]))
        result={**describe(model),"status":"ok","mean_val_mae":round(mean_mae,6),"mean_val_psnr_db":round(10*math.log10(1/mean_mse),3),"mean_val_dark_f1":round(mean_f1,6),"final_train_loss":round(losses[-1],6),"elapsed_seconds":round(time.perf_counter()-started,2),"peak_rss_mib":round(peak/1048576,1),"min_free_ram_mib":round(min_free/1048576,1),"validation_pages":rows}
        torch.save({"model":model.state_dict(),"width":args.width,"input_size":INPUT_SIZE},out_dir/"candidate.pt")
        for record in val_records[:args.preview_pages]:
            preview(model,record,cache_root,out_dir/"page-{:04d}.png".format(record.page_number))
    except Exception as exc:
        result={"base_channels":args.width,"status":"failed","error":"{}: {}".format(type(exc).__name__,exc),"elapsed_seconds":round(time.perf_counter()-started,2)}
    (out_dir/"result.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result),flush=True)
    return 0 if result["status"]=="ok" else 1

def manager(args):
    data_dir=Path(args.data_dir).resolve(); _,records=load_manifest(data_dir)
    total=args.train_pages+args.val_pages
    if total>len(records): raise ValueError("Not enough dataset pages.")
    stamp=time.strftime("%Y%m%d-%H%M%S")
    root=Path(__file__).resolve().parent/"evaluation"/("quality-screen-"+stamp)
    root.mkdir(parents=True,exist_ok=True)
    print("tiny-imgai 512px process-isolated quality screen")
    print("Dataset: {} | train: {} | validation: {} | epochs: {}".format(len(records),args.train_pages,args.val_pages,args.epochs))
    print("Widths: {}".format(WIDTHS)); print("Each width runs in a fresh Python process."); print()
    results=[]
    for i,width in enumerate(WIDTHS,1):
        out=root/("b{}".format(width)); out.mkdir(parents=True,exist_ok=True)
        cmd=[sys.executable,str(Path(__file__).resolve()),"--worker","--width",str(width),"--data-dir",str(data_dir),"--train-pages",str(args.train_pages),"--val-pages",str(args.val_pages),"--epochs",str(args.epochs),"--threads",str(args.threads),"--min-free-ram-mib",str(args.min_free_ram_mib),"--max-rss-mib",str(args.max_rss_mib),"--preview-pages",str(args.preview_pages),"--output",str(out)]
        print("[{}/{}] 512-b{}".format(i,len(WIDTHS),width),flush=True)
        completed=subprocess.run(cmd,text=True,capture_output=True)
        result_path=out/"result.json"
        if result_path.exists(): result=json.loads(result_path.read_text(encoding="utf-8"))
        else: result={"base_channels":width,"status":"failed","error":"worker produced no result","stdout":completed.stdout[-2000:],"stderr":completed.stderr[-2000:]}
        results.append(result)
        if result["status"]=="ok": print("  F1 {} | MAE {} | PSNR {} dB | {}s | RSS {} MiB".format(result["mean_val_dark_f1"],result["mean_val_mae"],result["mean_val_psnr_db"],result["elapsed_seconds"],result["peak_rss_mib"]),flush=True)
        else: print("  FAILED: {}".format(result["error"]),flush=True)
    report={"timestamp":stamp,"input_size":INPUT_SIZE,"train_pages":args.train_pages,"validation_pages":args.val_pages,"results":results}
    report_path=root/"summary.json"; report_path.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print("\n=== quality screen summary ===")
    for r in sorted([x for x in results if x.get("status")=="ok"],key=lambda x:x["mean_val_dark_f1"],reverse=True): print("b{}: F1 {} | MAE {} | PSNR {} dB | {}s".format(r["base_channels"],r["mean_val_dark_f1"],r["mean_val_mae"],r["mean_val_psnr_db"],r["elapsed_seconds"]))
    print("\nReport: {}".format(report_path)); print("Candidate previews/models: {}".format(root))
    return 0

def main():
    args=parse_args()
    return worker(args) if args.worker else manager(args)

if __name__=="__main__": raise SystemExit(main())