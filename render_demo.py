#!/usr/bin/env python3
"""cue x Ming — render the demo through Ming-3.0-tts (the engine cue was built for).

Pipeline: plain line -> cue (routes, injects 14-tag vocabulary) -> MingTTS -> wav
Each line is routed as its own turn (the demo shows the full range; in normal
use the restraint budget keeps a long message to ~2 tags).
"""
import sys
import time

sys.path.insert(0, "/home/wirehead/tts/ming/Ming-3.0-tts")
sys.path.insert(0, "/home/wirehead/embermind/cue")

import cue as cue_mod

# each line is designed to exercise a different cue (or silence, deliberately)
DEMO_LINES = [
    "haha okay, the backup finally finished.",            # -> [Laughter]
    "The router is really a compiler intermediate, and speech is just the emit pass.",  # -> [Breathing] (default breath at pause)
    "Really? The dog actually rebooted on its own?",      # -> [Question-huh]
    "ugh, the build failed again.",                       # -> [Dissatisfaction-hnn]
    "yeah.",                                              # -> [Confirmation-en]
    "well, at least it's on GitHub now.",                 # -> silence (restraint)
]

DEVICE = sys.argv[1] if len(sys.argv) > 1 else "cpu"
VOICE = sys.argv[2] if len(sys.argv) > 2 else "US_James"


def main():
    routed_lines, all_insertions = [], []
    for line in DEMO_LINES:
        routed, manifest = cue_mod.route(line)
        routed_lines.append(routed)
        all_insertions += manifest["insertions"]
        print(f"{line!r}\n  -> {routed!r}  ({[i['tag'] for i in manifest['insertions']] or 'silence'})",
              file=sys.stderr)
    text = " ".join(routed_lines)
    print("FULL TEXT:", text, file=sys.stderr)
    print("TAGS USED:", sorted({i["tag"] for i in all_insertions}), file=sys.stderr)

    t0 = time.time()
    from ming_tts import MingTTS
    model = MingTTS("/home/wirehead/tts/ming/Ming-3.0-tts-0.5b",
                    device=DEVICE, text_normalization=False)
    print(f"loaded in {time.time()-t0:.1f}s", file=sys.stderr)

    t1 = time.time()
    wave = model.synthesize(
        task="IP",
        language_id="English",
        text=text,
        controls={"ip": VOICE},
        use_zero_spk_emb=True,
        raw_text=True,  # tags + rhythm already placed by cue; don't re-fragment
        output_path=f"/home/wirehead/embermind/cue/demo_ming_{VOICE.lower()}.wav",
    )
    print(f"synthesized in {time.time()-t1:.1f}s", file=sys.stderr)
    print(f"WAV: /home/wirehead/embermind/cue/demo_ming_{VOICE.lower()}.wav "
          f"({wave.shape[1] / model.sample_rate:.1f}s at {model.sample_rate}Hz)")


if __name__ == "__main__":
    main()
