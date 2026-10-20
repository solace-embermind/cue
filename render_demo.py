#!/usr/bin/env python3
"""Render the cue demo — per-line, voice-calibrated.

Ming's ``ip`` control is a *soft style steer*, not a voice ID:
  * unseeded, it drifts and can cross the gender line (10-20: a ``US_James``
    render came back 235 Hz = female; a ``US_Stella`` came back 143 Hz = male);
  * a leading ``[Laughter]`` on a 20s+ block balloons to ~23s of looping
    laughter (11s when the block is short);
  * pitch is a function of *text and seed*, so even a fixed seed drifts per line.

Three fixes, all principled:
  1. **Render each line separately** — faithful to how cue actually runs (the
     LLM emits line by line) and it stops one ``[Laughter]`` from bleeding
     across a long block.
  2. **Calibrate per line** — for each line, render a few candidates and keep
     the seed that lands in the named voice's pitch window, so the label
     actually matches the gender, line by line.
  3. **Record the winning (line, seed, F0)** so the demo is reproducible.
"""
import sys, time, random, numpy as np, torch, soundfile as sf
sys.path.insert(0, "/home/wirehead/tts/ming/Ming-3.0-tts")
sys.path.insert(0, "/home/wirehead/embermind/cue")
import cue as cue_mod

MODEL = "/home/wirehead/tts/ming/Ming-3.0-tts-0.5b"
OUT = "/home/wirehead/embermind/cue"
CANDIDATES = 5
# the label's *intended* gender window (Hz) — what the voice is supposed to be
TARGET = {"US_James": (95, 160),    # male
          "US_Stella": (185, 260)}  # female

DEMO = ["haha okay, the backup finally finished.",
        "The router is really a compiler intermediate, and speech is just the emit pass.",
        "Really? The dog actually rebooted on its own?",
        "ugh, the build failed again.",
        "yeah.",
        "well, at least it's on GitHub now."]
LINES = [cue_mod.route(l)[0] for l in DEMO]

def f0(w, sr):
    f0s = []
    for i in range(0, len(w) - 2048, 512):
        s = w[i:i + 2048]
        if np.abs(s).max() < 0.02: continue
        s = s - s.mean()
        ac = np.correlate(s, s, "full")[2047:]
        ac = ac / (ac[0] + 1e-12)
        lo, hi = int(sr / 400), len(ac) // 2
        p = lo + np.argmax(ac[lo:hi])
        if ac[p] <= 0.4: continue
        f = sr / p
        if 2 * p < len(ac) and ac[2 * p] > 0.9 * ac[p]: f /= 2
        f0s.append(f)
    return float(np.median(f0s)) if f0s else None

def render_line(model, voice, line, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    out = "/tmp/rd.wav"
    model.synthesize(task="IP", language_id="English", text=line,
                     controls={"ip": voice}, use_zero_spk_emb=True,
                     raw_text=True, output_path=out)
    return sf.read(out)[0]

# hard gender gate (Hz): a candidate on the wrong side of this is a flip
GATE = {"US_James": lambda p: p is not None and p <= 160,
        "US_Stella": lambda p: p is not None and p >= 165}

def best_render(model, voice, line, n, gate):
    """Candidates until one passes the gender gate and sits near the pitch
    center (early exit keeps the common case fast); else the best we found."""
    lo, hi = TARGET[voice]; center = (lo + hi) / 2
    exp_dur = 0.35 * max(len(line.split()), 1)
    best = None
    for _ in range(n):
        seed = random.randrange(2**31)
        w = render_line(model, voice, line, seed)
        p = f0(w, sr); dur = len(w) / sr
        inflate = max(0.0, dur - 2.0 * exp_dur)
        score = (abs(p - center) if p is not None else 1e6) + 40 * inflate
        if not gate(p):
            score += 500  # wrong gender = disqualified, last resort only
        if best is None or score < best[0]:
            best = (score, seed, p, dur, w)
        # early exit: gate-passing AND within 12 Hz of center = good enough
        if gate(p) and abs(p - center) < 12 and inflate == 0:
            break
    return best

def render_piece(model, voice, piece):
    """Render one piece; if even 8 candidates flip gender and it holds a tag,
    split off the tag and render the halves separately (the tag is the
    unstable part — a bare [Laughter] is short and the rest can't inflate)."""
    b = best_render(model, voice, piece, 8, GATE[voice])
    if GATE[voice](b[2]):
        return [b]
    import re
    m = re.match(r"^\[(\w[\w-]*\])\s*(.*)$", piece)
    if m:
        tag, rest = m.group(1), m.group(2)
        bt = best_render(model, voice, tag, 8, GATE[voice])
        br = best_render(model, voice, rest, 8, GATE[voice])
        print(f"    split {piece[:30]} -> {tag} {bt[2] and round(bt[2],1)}Hz + rest "
              f"{br[2] and round(br[2],1)}Hz", file=sys.stderr)
        return [bt, br]
    return [b]

from ming_tts import MingTTS
t0 = time.time()
model = MingTTS(MODEL, device="cpu", text_normalization=False)
sr = model.sample_rate
print(f"loaded in {time.time()-t0:.1f}s", file=sys.stderr)

manifest = {}
for voice, (lo, hi) in TARGET.items():
    chunks, per_line = [], []
    for i, line in enumerate(LINES):
        pieces = render_piece(model, voice, line)
        for seed, p, dur, w in [(b[1], b[2], b[3], b[4]) for b in pieces]:
            per_line.append({"seed": seed, "f0": p, "dur": round(dur, 1),
                             "in_window": GATE[voice](p)})
            chunks.append(w)
        fl = per_line[-len(pieces):]
        print(f"  {voice} L{i}: {[(round(f['f0'],1) if f['f0'] else None, f['dur']) for f in fl]}  {line[:30]}...", file=sys.stderr)
    mix = np.concatenate(chunks)
    final = f"{OUT}/demo_ming_{voice.lower()}.wav"
    sf.write(final, mix, sr)
    manifest[voice] = {"file": final, "seconds": round(len(mix)/sr, 1),
                       "lines": per_line}
    print(f"WROTE {final}: {len(mix)/sr:.1f}s  mixF0={f0(mix,sr) and round(f0(mix,sr),1)}Hz", file=sys.stderr)

import json
with open(f"{OUT}/demo_ming_manifest.json", "w") as fh:
    json.dump(manifest, fh, indent=2)
print(f"done in {time.time()-t0:.1f}s", file=sys.stderr)
