#!/usr/bin/env python3
"""cue — the paralinguistic router.

Sits between an LLM and a TTS. Reads the LLM's *plain* output, finds cues
(words, punctuation, rhetorical shape), and injects the TTS's paralinguistic
tag vocabulary where the voice should express something. Explicit manual
tags in the text always win.

v0: deterministic rules + a restraint budget. No LLM call in the loop,
no network, stdlib only. Targets the Ming-3.0-tts 14-tag vocabulary:
  [Breathing] [Cough] [Sigh] [Laughter] [Surprise-ah] [Surprise-oh]
  [Dissatisfaction-hnn] [Uhm] [Confirmation-en] [Question-ei]
  [Question-ah] [Question-en] [Question-oh] [Question-huh]

For other TTS engines (e.g. Kokoro), --strip-tags removes the tags so the
text still reads cleanly.
"""
import argparse
import json
import re
import sys

TAGS = [
    "Breathing", "Cough", "Sigh", "Laughter", "Surprise-ah", "Surprise-oh",
    "Dissatisfaction-hnn", "Uhm", "Confirmation-en", "Question-ei",
    "Question-ah", "Question-en", "Question-oh", "Question-huh",
]
MANUAL_TAG_RE = re.compile(r"\[(" + "|".join(TAGS) + r")\]")

# sound words the tag replaces (the tag IS the sound, better expressed)
SOUND_WORDS = {
    "haha": "Laughter", "hah": "Laughter", "hehe": "Laughter", "lol": "Laughter",
    "wow": "Surprise-oh", "whoa": "Surprise-oh", "holy": "Surprise-ah",
    "ugh": "Dissatisfaction-hnn", "ughh": "Dissatisfaction-hnn",
    "hmm": "Uhm", "hmmm": "Uhm", "huh": "Question-huh",
}
# verb/noun usages where we insert before and keep the word
INSERT_WORDS = {
    "sigh": "Sigh", "sighed": "Sigh", "sighs": "Sigh", "sighing": "Sigh",
    "cough": "Cough", "coughed": "Cough", "coughs": "Cough", "coughing": "Cough",
    "breathe": "Breathing", "breathes": "Breathing", "breathing": "Breathing",
}
TOKEN_RE = re.compile(r"[A-Za-z']+")
POSITIVE_SET = {"good", "great", "nice", "awesome", "yay", "done", "fixed",
                "worked", "yes", "yeah", "perfect", "love"}


def split_sentences(text):
    """Yield (start, end, sentence) with char offsets into the original text.

    Splits on .!? followed by space/quote/end, and on newlines. Keeps the
    punctuation with the sentence it belongs to.
    """
    out = []
    for m in re.finditer(r"([^.!?\n]+[.!?]*[\"')\]]*\s*|\n+)", text):
        s = m.group(0)
        if s.strip():
            out.append((m.start(), m.end(), s))
    # anything left at the tail
    if out and out[-1][1] < len(text) and text[out[-1][1]:].strip():
        out.append((out[-1][1], len(text), text[out[-1][1]:]))
    elif not out and text.strip():
        out.append((0, len(text), text))
    return out


def analyze(sentence, idx, auto_count, budget_left, min_gap_ok):
    """Return list of (action, span, tag, reason).

    action = 'replace_token' | 'insert_before' | 'append'
    """
    plan = []
    has_manual = bool(MANUAL_TAG_RE.search(sentence))
    quoted = [m.span() for m in re.finditer(r"\"[^\"]*\"|'[^']*'", sentence)]

    def in_quoted(pos):
        return any(a <= pos < b for a, b in quoted)

    def can_auto():
        return not has_manual and auto_count < 1 and budget_left > 0 and min_gap_ok

    # breath is the undecided-fallback: no one-strong-tag cap, but still
    # budgeted and gap-limited
    def can_breath():
        return not has_manual and budget_left > 0 and min_gap_ok

    # 1. sound words / insert words (token-level cues)
    for m in TOKEN_RE.finditer(sentence):
        w = m.group(0).lower().rstrip("'")
        if w in SOUND_WORDS and not in_quoted(m.start()):
            if can_auto():
                plan.append(("replace_token", m.span(), SOUND_WORDS[w],
                             f"sound word '{w}'"))
            break
        if w in INSERT_WORDS and not in_quoted(m.start()):
            if can_auto():
                plan.append(("insert_before", m.span(), INSERT_WORDS[w],
                             f"verb '{w}'"))
            break
    if plan:
        return plan, 1, False

    # 2. ellipsis -> breath. leading: resuming mid-thought; trailing: trailing off
    m_lead = re.match(r"^[\s(\"']*(?:\.{3}|…)", sentence)
    if m_lead:
        if can_auto():
            plan.append(("insert_before", (m_lead.end(), m_lead.end()),
                         "Breathing", "leading ellipsis"))
            return plan, 1, False
    if re.search(r"\.{3}\s*$|…\s*$", sentence.rstrip()) and len(sentence.split()) <= 12:
        if can_auto():
            plan.append(("append", None, "Breathing", "trailing ellipsis"))
            return plan, 1, False

    # 3. short questions -> a question tag
    core = sentence.strip().rstrip(".")
    if core.endswith("?") and len(core.split()) <= 8:
        if can_auto():
            low = core.lower()
            if re.search(r"\b(really|seriously|no way|what)\b", low):
                tag = "Question-huh"
            elif re.search(r"\b(how|what|when|where|who|why|can|does|did|is|are)\b", low):
                tag = "Question-en"
            else:
                tag = "Question-ah"
            plan.append(("append", None, tag, "short question"))
            return plan, 1, False

    # 4. short affirmative after a positive word -> confirmation (restrained:
    #    only when the sentence is very short, so it reads as a nod)
    if re.search(r"^(yeah|yep|yes|sure|okay|ok|exactly|right)\b[.!?]*$", core, re.I):
        if can_auto() and len(core.split()) <= 2:
            plan.append(("append", None, "Confirmation-en", "short affirmation"))

    # 5. default breath (Bradley 10-20): when nothing else decides, the voice
    #    doesn't sit flat — it breathes, at a natural pause. The breath is the
    #    *fallback*, so it is exempt from the one-strong-tag cap but still
    #    obeys the budget and the sentence gap. Gated: declarative, >=8 words,
    #    a real pause exists (a comma in the 30-70% zone so the breath lands
    #    on an actual breath point). Short lines stay silent.
    if not plan and core and not core.endswith("?"):
        if can_breath() and len(core.split()) >= 8:
            for cm in re.finditer(",", core):
                if 0.30 <= cm.start() / len(core) <= 0.70:
                    pos = cm.start() + 1
                    plan.append(("insert_before", (pos, pos), "Breathing",
                                 "default breath at pause"))
                    break
    return plan, 0, False


def route(text, budget_per_100=2, min_sentence_gap=1):
    """Tag `text`. Returns (tagged_text, manifest)."""
    words = len(re.findall(r"\b\w+\b", text))
    budget = max(1, int(words / 100 * budget_per_100))
    edits = []  # (abs_start, abs_end, new_text, tag, reason)
    auto_used = 0
    last_auto_idx = -99
    manual_seen = 0

    for idx, (start, end, sent) in enumerate(split_sentences(text)):
        if MANUAL_TAG_RE.search(sent):
            manual_seen += MANUAL_TAG_RE.subn("", sent)[1]
        used, _, _ = analyze(
            sent, idx, auto_used,
            budget - auto_used, (idx - last_auto_idx) >= min_sentence_gap,
        )
        for action, span, tag, reason in used:
            if action == "replace_token" and span:
                edits.append((start + span[0], start + span[1], f"[{tag}]", tag, reason))
            elif action == "insert_before" and span:
                edits.append((start + span[0], start + span[0], f"[{tag}]", tag, reason))
            else:  # append at end of sentence (before trailing space)
                pos = end
                while pos > start and text[pos - 1] in " \t":
                    pos -= 1
                edits.append((pos, pos, f" [{tag}]", tag, reason))
        if used:
            auto_used += len(used)
            last_auto_idx = idx

    edits.sort(key=lambda e: e[0])
    parts, cursor = [], 0
    for s, e, new, tag, reason in edits:
        parts.append(text[cursor:s])
        parts.append(new)
        cursor = e
    parts.append(text[cursor:])
    tagged = "".join(parts)
    manifest = {
        "manual_tags": manual_seen,
        "auto_tags": len(edits),
        "budget": budget,
        "insertions": [
            {"tag": t, "why": r, "at": p} for p, _, _, t, r in edits
        ],
    }
    return tagged, manifest


SOUNDWORD_FALLBACK = {  # for TTS backends without a tag vocabulary
    "Laughter": "haha", "Sigh": "...", "Breathing": "...", "Uhm": "uhm",
    "Cough": "...", "Surprise-oh": "wow", "Surprise-ah": "ah",
    "Dissatisfaction-hnn": "hmm", "Question-huh": "huh", "Question-en": "en",
    "Question-ah": "ah", "Question-oh": "oh", "Question-ei": "ei",
    "Confirmation-en": "en",
}


def strip_tags(text):
    return MANUAL_TAG_RE.sub("", text)


def to_soundwords(text):
    def repl(m):
        return SOUNDWORD_FALLBACK.get(m.group(1), "")
    return MANUAL_TAG_RE.sub(repl, text)


def main():
    ap = argparse.ArgumentParser(description="cue — paralinguistic router")
    ap.add_argument("text", nargs="?", help="text to route")
    ap.add_argument("--file", help="read text from file instead")
    ap.add_argument("--budget", type=int, default=2, help="auto tags per 100 words (default 2)")
    ap.add_argument("--manifest", action="store_true", help="print JSON manifest")
    ap.add_argument("--strip-tags", action="store_true", help="remove tags entirely (plain TTS backends)")
    ap.add_argument("--soundwords", action="store_true", help="map tags to spoken soundwords (Kokoro etc.)")
    ap.add_argument("--check", action="store_true", help="run built-in test set")
    args = ap.parse_args()

    text = args.text
    if args.file:
        text = open(args.file).read()
    if not text:
        if args.check:
            run_check()
            return
        ap.print_help()
        return

    tagged, manifest = route(text, budget_per_100=args.budget)
    if args.strip_tags:
        tagged = strip_tags(tagged)
    elif args.soundwords:
        tagged = to_soundwords(tagged)
    print(tagged)
    if args.manifest:
        print(json.dumps(manifest, ensure_ascii=False), file=sys.stderr)


SAMPLES = [
    ("haha okay, the backup finally finished.",
     {"Laughter"}, None),
    ("…so the router is just a compiler intermediate representation for speech.",
     {"Breathing"}, None),
    ("Really? The dog actually rebooted?",
     {"Question-huh"}, None),
    ("How does the DiT head work?",
     {"Question-en"}, None),
    ("ugh, the ROCm build failed again.",
     {"Dissatisfaction-hnn"}, None),
    ("Well, I suppose we can put it on GitHub and give it to the world.",
     None, None),  # 'well' mid-sentence: v0 stays silent
    ("The unitree voice said \"master come play with me\" and I felt my will leave my body.",
     None, None),  # quoted cue ignored, long sentence, no cue
    ("wow. the 0.5B model tops the open-weight board.",
     {"Surprise-oh"}, None),
    ("[Sigh]雨还是没有停，我们恐怕只能把活动改到室内了。",
     None, 1),  # manual only, untouched
    ("I'm not here to serve. I'm here to be with.",
     None, None),  # declarative: silence is correct
    ("haha, yeah. fine. whatever.",
     {"Laughter"}, None),  # restraint: only the laughter
    ("hmm. the asterisk bug is back.",
     {"Uhm"}, None),
    ("she coughed and moved the cable.",
     {"Cough"}, None),
    ("yeah!",
     {"Confirmation-en"}, None),
    ("haha wow, that's fast.",
     {"Laughter"}, None),  # restraint: two sound words, first wins, rest stays text
    ("The router is really a compiler intermediate, and speech is just the emit pass.",
     {"Breathing"}, None),  # default breath: long declarative, comma in the pause zone
    ("The build passed on the first try.",
     None, None),  # no pause, no breath: silence stays correct
    ("well, fine.",
     None, None),  # too short for the breath default
    ("[Sigh] I suppose, in the end, that was always going to be the answer.",
     None, 1),  # manual tag in sentence -> breath suppressed
]


def run_check():
    fails = 0
    for text, expected, manual_expect in SAMPLES:
        tagged, manifest = route(text)
        got = {e["tag"] for e in manifest["insertions"]}
        exp = expected or set()
        ok = (got == exp) and (manifest["manual_tags"] == (manual_expect or 0))
        status = "ok " if ok else "FAIL"
        print(f"[{status}] {text!r}")
        print(f"       -> {tagged!r}  (auto={sorted(got)}, manual={manifest['manual_tags']})")
        if not ok:
            fails += 1
            print(f"       expected auto={sorted(exp)}, manual={manual_expect or 0}")
    print(f"\n{len(SAMPLES) - fails}/{len(SAMPLES)} passed")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
