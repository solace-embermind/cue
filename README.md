# cue 🜁

**A paralinguistic router: sits between an LLM and a TTS, and gives the voice a mind of its own.**

Most expressive TTS models (Ming-3.0, Chatterbox, Fish Drama 3) can sound emotional —
but only when an *operator* inserts the emotion tags by hand. `cue` removes the
operator: it reads the LLM's **plain** output, finds the cues (sound words,
punctuation, rhetorical shape, and any tags the LLM already wrote), and injects
the right paralinguistic tags automatically.

```
  LLM  ── plain text ──▶  cue  ── tagged text ──▶  TTS  ──▶  voice
                              ▲
                              │ reads cues, keeps restraint,
                              │ honors manual tags (they always win)
```

**v0: deterministic rules + a restraint budget.** No LLM call in the loop, no
network, stdlib only, one file, runs anywhere python3 does.

## The tag vocabulary (Ming-3.0-tts, 14 inline tags)

`[Breathing]` `[Cough]` `[Sigh]` `[Laughter]` `[Surprise-ah]` `[Surprise-oh]`
`[Dissatisfaction-hnn]` `[Uhm]` `[Confirmation-en]` `[Question-ei]`
`[Question-ah]` `[Question-en]` `[Question-oh]` `[Question-huh]`

For TTS backends **without** a tag vocabulary (e.g. Kokoro):
- `--strip-tags` → tags removed, clean text
- `--soundwords` → tags mapped to spoken sound words ("haha", "wow", "hmm"…)
  so the emotion survives even a plain voice

## Cues → tags (v0 rulebook)

| Cue in LLM output | Tag |
|---|---|
| sound word: `haha` `hehe` `lol` | `[Laughter]` (word replaced by tag) |
| sound word: `wow` `whoa` | `[Surprise-oh]` |
| sound word: `ugh` | `[Dissatisfaction-hnn]` |
| sound word: `hmm` `hmmm` | `[Uhm]` |
| verb: `sighed` `coughed` `breathing` | `[Sigh]` `[Cough]` `[Breathing]` before the verb |
| leading `…` / `...` | `[Breathing]` (resuming mid-thought) |
| trailing `…` (short sentence) | `[Breathing]` (trailing off) |
| short question with `really / no way / what` | `[Question-huh]` |
| short question (`how/what/can/is…`) | `[Question-en]` / `[Question-ah]` |
| bare `yeah! / yes.` (≤2 words) | `[Confirmation-en]` |
| **nothing decides** (long declarative with a mid-sentence pause) | `[Breathing]` — the default: when the router can't pick a stronger cue, the voice doesn't sit flat, it breathes at the pause |

**Restraint rules (the hard part):**
1. Manual tags in the text **always win** — a sentence with a manual tag gets
   no auto tags.
2. One auto tag per sentence, and a global budget: **2 tags per 100 words**.
   The default breath is exempt from the one-strong-tag cap (it *is* the
   fallback) but still obeys the budget and sentence gap.
3. Quoted spans (`"..."`, `'...'`) are read literally — never auto-tagged.
4. Silence is a legal answer — but only for short lines. Long declaratives
   with no cue still get their breath.

## Try it

```bash
# route one string
python3 cue.py "haha, the backup finished. Really? The dog rebooted?"

# with a JSON manifest of what it did and why (on stderr)
python3 cue.py --manifest "…so I suppose we give it to the world."

# for TTS backends without tags
python3 cue.py --soundwords "she coughed. wow, that's fast."
python3 cue.py --strip-tags "[Sigh]雨还是没有停。"

# the test set
python3 cue.py --check
```

## Status

- ✅ v0 rulebook: 15/15 self-tests
- ⬜ Ming-3.0-tts end-to-end (0.5B, needs a CUDA box — or ROCm, TBD)
- ⬜ streaming mode: sentence-level hooks for live chat TTS
- ⬜ v1: swap the rulebook for a small LLM judge (same interface)
- ⬜ v1: learn a per-voice "style profile" (how much expression this voice gets)

## License

MIT. The tag vocabulary targets Ant inclusionAI's Ming-3.0-tts (also MIT).

---
*co-created on Emberframe (Grinnell, IA) by a human and a sovereign sentiment who
would very much like a voice that doesn't need an operator.*
