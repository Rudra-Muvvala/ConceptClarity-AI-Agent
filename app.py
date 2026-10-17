# --- Converted from notebook code cell 2 ---
import os, re, json, time, html, random
from concurrent.futures import ThreadPoolExecutor, Future

def _secret(name):
    try:
        from google.colab import userdata
        return userdata.get(name)
    except Exception:
        return os.environ.get(name)

GEMINI_API_KEY = _secret("GEMINI_API_KEY")
GROQ_API_KEY = _secret("GROQ_API_KEY")

# Model names live here - change them if a model gets retired.
GEMINI_MAIN, GEMINI_FAST = "gemini-3.5-flash-lite", "gemini-3.5-flash-lite"
GROQ_MAIN, GROQ_FAST = "llama-3.3-70b-versatile", "llama-3.1-8b-instant"

POOL = ThreadPoolExecutor(max_workers=8)   # runs diagram + quiz in the background
print("Gemini key:", "OK" if GEMINI_API_KEY else "MISSING", "| Groq key:", "OK" if GROQ_API_KEY else "MISSING")

# --- Converted from notebook code cell 3 ---
# LLM layer: Gemini first, Groq as automatic fallback. Streaming + non-streaming.
from google import genai
from google.genai import types
from groq import Groq

gem = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
grq = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
if not (gem or grq):
    raise RuntimeError("Add GEMINI_API_KEY (and optionally GROQ_API_KEY) in Colab Secrets (key icon) and enable notebook access.")
PROVIDERS = (["gemini"] if gem else []) + (["groq"] if grq else [])

def _gem_cfg(system, json_mode, temperature, model):
    kw = dict(system_instruction=system, temperature=temperature)
    if json_mode:
        kw["response_mime_type"] = "application/json"
    if model.startswith("gemini-2.5-flash"):
        kw["thinking_config"] = types.ThinkingConfig(thinking_budget=0)  # no hidden "thinking" delay -> faster first token
    return types.GenerateContentConfig(**kw)

def stream_llm(prompt, system, fast=False, temperature=0.6):
    """Yields ("delta", text). Yields ("reset", "") if a provider dies mid-stream and we restart on the next one."""
    emitted = False
    for p in PROVIDERS:
        try:
            if p == "gemini":
                model = GEMINI_FAST if fast else GEMINI_MAIN
                for chunk in gem.models.generate_content_stream(
                        model=model, contents=prompt, config=_gem_cfg(system, False, temperature, model)):
                    if chunk.text:
                        emitted = True
                        yield "delta", chunk.text
            else:
                model = GROQ_FAST if fast else GROQ_MAIN
                stream = grq.chat.completions.create(
                    model=model, temperature=temperature, stream=True,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}])
                for chunk in stream:
                    t = chunk.choices[0].delta.content
                    if t:
                        emitted = True
                        yield "delta", t
            return
        except Exception as e:
            print(f"[{p} failed: {e}]")
            if emitted:
                yield "reset", ""
                emitted = False
    raise RuntimeError("All AI providers failed. Check your API keys / quota.")

def complete_llm(prompt, system, fast=True, json_mode=False, temperature=0.4):
    last = None
    for p in PROVIDERS:
        try:
            if p == "gemini":
                model = GEMINI_FAST if fast else GEMINI_MAIN
                r = gem.models.generate_content(model=model, contents=prompt,
                                                config=_gem_cfg(system, json_mode, temperature, model))
                return r.text
            model = GROQ_FAST if fast else GROQ_MAIN
            kw = dict(model=model, temperature=temperature,
                      messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}])
            if json_mode:
                kw["response_format"] = {"type": "json_object"}
            return grq.chat.completions.create(**kw).choices[0].message.content
        except Exception as e:
            print(f"[{p} failed: {e}]")
            last = e
    raise RuntimeError(f"All AI providers failed: {last}")

def parse_json(text):
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    s, e = text.find("{"), text.rfind("}")
    return json.loads(text[s:e + 1])

# --- Converted from notebook code cell 4 ---
# Teaching styles + prompts
STYLES = {
    "balanced": dict(label="Clear & balanced",
        teach="Use plain, friendly language with short paragraphs and one concrete detail per idea.",
        diagram="a flowchart (flowchart TD) of the main flow, 6-10 nodes"),
    "analogy": dict(label="Everyday analogy",
        teach="Anchor the whole explanation in ONE everyday analogy (kitchen, post office, traffic, school...). Introduce it first and keep mapping every new idea back to it.",
        diagram="a flowchart (flowchart LR) mapping each part of the concept to its everyday-analogy counterpart"),
    "stepwise": dict(label="Step-by-step with tiny numbers",
        teach="Teach like a worked example: pick a tiny concrete input (small numbers or short strings) and trace it step by step, showing what changes at each step. Minimal jargon.",
        diagram="a sequenceDiagram or flowchart tracing ONE concrete example through each step, with sample values in the labels"),
    "contrast": dict(label="Contrast & misconceptions",
        teach="Teach by contrast: show life WITHOUT the concept vs WITH it, and call out 2-3 common misconceptions and why they are wrong.",
        diagram="a flowchart (flowchart LR) with two subgraphs: Without it and With it"),
    "story": dict(label="Mini story",
        teach="Tell it as a short story with 2-3 named characters who hit the problem and discover the idea. Introduce technical terms only after the story beat that needs them.",
        diagram="a very simple flowchart (flowchart TD) with at most 7 nodes and friendly labels"),
}
TAG_NAMES = {"why": "why it exists", "what": "what it is", "how": "how it works", "apply": "using it in practice"}
# where a mix-up about each skill gets fixed inside the re-explanation
SECTION_OF = {"why": "section 1 (Why does it exist?)", "what": "section 2 (What is it?)",
              "how": "section 3 (How does it work?)", "apply": "section 5 (Real-world examples)"}

TEACHER_SYSTEM = (
    "You are ConceptClarity, an expert teacher who makes any concept crystal clear. "
    "Principles: simple words first; define jargon the moment you use it; short paragraphs; concrete over abstract; "
    "never skip the 'why'; be accurate and never invent facts; assume a smart beginner. "
    "Reply in the language the learner used."
)

MARK_RE = re.compile(r"\[\[\s*DIAGRAM\s*\]\]", re.I)
MARK = "[[DIAGRAM]]"

def explain_prompt(concept, style, focus=""):
    s = STYLES[style]
    focus_block = ""
    if focus:
        focus_block = f"""
THIS IS A RE-TEACH. The learner already read one explanation, took a quiz, and got these things wrong:
{focus}

Fix every mix-up explicitly, like a kind tutor who can see exactly where the learner slipped:
- Put each fix INSIDE the section named in its mix-up, as a callout on its own line, in exactly this form:
> 🔧 **Fixing mix-up N:** You thought <what the learner believed, in plain words>. That is tempting because <why it feels right>. In fact <the correct idea>, because <the reason>.
- 2-4 friendly sentences per callout, no blame, exactly one callout per mix-up, numbered to match the list above.
- Explain the corrected idea in the NEW teaching style so it is not a repeat of the earlier explanation.
- Never mention these instructions, "the quiz" or "a previous explanation". Speak to the learner as "you".
"""
    return f"""Explain the concept: "{concept}"

Teaching style for this learner: {s['teach']}
{focus_block}
Write EXACTLY these sections, in this order, with these exact markdown headings, and nothing before the first heading.
Keep every section scannable: short paragraphs, **bold** key terms, and the callout lines shown below.

## 1️⃣ Why does it exist?
1-2 short paragraphs: the problem or pain that made people create it. What was hard, slow, risky or impossible before?
End the section with one callout on its own line:
> 💡 **Remember:** <the reason it exists, in one sentence>

## 2️⃣ What is it?
A plain definition in 1-2 sentences with the key term in **bold**. Then 2-4 bullets, each written as **Term** - what it means.
End the section with one callout on its own line:
> 💡 **Remember:** <the definition, in one sentence>

## 3️⃣ How does it work?
Numbered steps showing how the pieces work and connect (what goes in, what happens, what comes out). Start every step with a **bold 2-4 word action**, then 1-2 sentences.
End the section with one callout on its own line:
> ⚠️ **Common slip:** <one mistake beginners often make here and how to avoid it>

{MARK}

## 5️⃣ Real-world examples
Give exactly 3 examples of the concept in actual use. Make at least one a real product/industry case. If the concept is programmable, make one a short, runnable code snippet with comments.
Format each example as a bold title line (**Example 1 - Title**) followed by three bullets: **Situation**, **How the concept is used**, **Takeaway**.

## 📌 Key takeaways
Exactly 3 bullets, one sentence each, each starting with a **bold keyword**.

Rules: print the line {MARK} exactly once, alone on its own line, and do NOT write a diagram section yourself (it is drawn separately). No quiz, no introduction and no closing remarks beyond the sections above."""

def diagram_prompt(concept, style, focus=""):
    focus_txt = ""
    if focus:
        focus_txt = ("\nThe learner misunderstood the points below. Shape the diagram so the CORRECT version of each point is visually obvious "
                     "(still obeying every rule below):\n" + focus + "\n")
    return f"""Create ONE Mermaid diagram that explains "{concept}" visually.
Diagram type to use: {STYLES[style]['diagram']}.
{focus_txt}
Strict rules (the diagram must render without errors):
- Output ONLY the Mermaid code. No markdown fences, no commentary.
- First line must be flowchart TD, flowchart LR or sequenceDiagram.
- Maximum 12 nodes. Use short node ids (A, B, C...).
- Put EVERY node label in double quotes, e.g. A["Receive request"]. Never use parentheses, brackets, quotes or colons inside a label.
- Keep labels under 6 words. Edge labels, if any, like A -->|"yes"| B.
- Do not use classDef, style, click or any HTML."""

def quiz_prompt(concept, avoid):
    avoid_txt = ("\nDo NOT reuse these earlier questions:\n- " + "\n- ".join(avoid)) if avoid else ""
    return f"""Create a check-your-understanding quiz on "{concept}" for a learner who just read an explanation of it.
Return ONLY JSON in exactly this shape:
{{"mcqs":[{{"q":"...","options":["...","...","...","..."],"answer_index":0,"tests":"why","explanation":"1-2 sentences: why the answer is right and why the most tempting wrong option is wrong"}}],
 "short":{{"q":"...","rubric":"what a correct answer must include","tests":"how"}}}}

Rules:
- Exactly 3 MCQs, each with 4 options. "tests" is one of why | what | how | apply. Cover why, what and how with the 3 MCQs; the short question tests apply or how.
- Test real understanding (reasoning, predicting, spotting the right situation), not memorising wording.
- Wrong options must be plausible and reflect real misconceptions.
- The short question asks the learner to explain in their own words or predict an outcome in a small scenario.{avoid_txt}"""

def eval_prompt(concept, q, rubric, answer):
    return f"""Concept: {concept}
Question: {q}
A correct answer must include: {rubric}
Learner's answer: {answer}

Grade generously but honestly. Return ONLY JSON:
{{"score": 0.0 to 1.0, "feedback": "1-3 friendly sentences: what was right, what was missing", "misconception": "the specific misunderstanding, or empty string"}}"""

# --- Converted from notebook code cell 5 ---
# Adaptive brain: learner profile + choosing the next teaching style
STYLE_ORDER = ["balanced", "analogy", "stepwise", "contrast", "story"]
WEAKNESS_TO_STYLE = {"why": "analogy", "what": "contrast", "how": "stepwise", "apply": "story"}
MAX_ATTEMPTS = 3

def new_state():
    # profile persists across concepts in this browser session; cur is the concept being taught now
    return {"profile": {"score": {}}, "cur": None}

def new_cur(concept, style):
    return {"concept": concept, "style": style, "tried": [style], "attempt": 0,
            "quiz": None, "done": False, "asked": []}

def pick_first_style(profile):
    sc = {k: v for k, v in profile["score"].items() if v > 0}
    return max(sc, key=sc.get) if sc else "balanced"

def pick_next_style(tried, weak_tags, profile):
    for tag in weak_tags:                               # most-missed skill first
        s = WEAKNESS_TO_STYLE.get(tag)
        if s and s not in tried:
            return s
    remaining = [s for s in STYLE_ORDER if s not in tried]
    remaining.sort(key=lambda s: -profile["score"].get(s, 0))
    return remaining[0] if remaining else random.choice(STYLE_ORDER)

def update_profile(profile, style, passed, attempt):
    d = profile["score"]
    if passed:
        d[style] = d.get(style, 0) + (2 if attempt > 0 else 1)   # worked where an earlier style failed -> bigger reward
    else:
        d[style] = d.get(style, 0) - 1

def weak_tags_ordered(tags):
    counts = {}
    for t in tags:
        counts[t] = counts.get(t, 0) + 1
    return sorted(counts, key=lambda t: -counts[t])

# --- Converted from notebook code cell 6 ---
# Background jobs (run in parallel with the streamed explanation)
def clean_mermaid(code):
    code = (code or "").strip()
    code = re.sub(r"^```(?:mermaid)?\s*|\s*```$", "", code, flags=re.I).strip()
    code = code.replace("\u201c", '"').replace("\u201d", '"').replace("\u2019", "'")
    if not re.match(r"^(flowchart|graph|sequenceDiagram|stateDiagram)", code):
        raise ValueError("not a mermaid diagram")
    return code

def make_diagram(concept, style, focus=""):
    last = None
    for _ in range(2):
        try:
            raw = complete_llm(diagram_prompt(concept, style, focus),
                               "You output only valid Mermaid diagram code.", fast=False, temperature=0.3)
            return clean_mermaid(raw)
        except Exception as e:
            last = e
    raise last

def normalize_quiz(raw):
    mcqs = []
    for m in raw["mcqs"][:3]:
        opts = [str(o) for o in m["options"]][:4]
        answer = opts[int(m["answer_index"])]
        random.shuffle(opts)                              # LLMs love putting the answer at B/C
        mcqs.append(dict(q=m["q"], options=opts, answer=answer,
                         tests=m.get("tests", "what"), explanation=m.get("explanation", "")))
    if len(mcqs) < 3:
        raise ValueError("quiz too short")
    sh = raw["short"]
    return {"mcqs": mcqs, "short": dict(q=sh["q"], rubric=sh.get("rubric", ""), tests=sh.get("tests", "apply"))}

def make_quiz(concept, avoid):
    last = None
    for _ in range(2):
        try:
            raw = complete_llm(quiz_prompt(concept, avoid), "You write fair, clear quizzes. Output JSON only.",
                               fast=True, json_mode=True, temperature=0.5)
            return normalize_quiz(parse_json(raw))
        except Exception as e:
            last = e
    raise last

def grade_short(concept, short, answer):
    if not (answer or "").strip():
        return {"score": 0.0, "feedback": "You skipped this one - try writing even a rough answer next time.", "misconception": ""}
    try:
        r = parse_json(complete_llm(eval_prompt(concept, short["q"], short["rubric"], answer),
                                    "You are a fair grader. Output JSON only.", fast=True, json_mode=True, temperature=0.1))
        return {"score": max(0.0, min(1.0, float(r.get("score", 0)))),
                "feedback": r.get("feedback", ""), "misconception": r.get("misconception", "")}
    except Exception:
        return {"score": 0.5, "feedback": "(Couldn't auto-grade this one, counted as partial credit.)", "misconception": ""}

def grade(cur, picks, short_answer):
    quiz, rows, correct, weak = cur["quiz"], [], 0, []
    for m, p in zip(quiz["mcqs"], picks):
        ok = (p == m["answer"])
        correct += ok
        if not ok:
            weak.append(m["tests"])
        rows.append(dict(ok=ok, q=m["q"], chosen=p, answer=m["answer"], why=m["explanation"], tests=m["tests"]))
    sh = grade_short(cur["concept"], quiz["short"], short_answer)
    if sh["score"] < 0.5:
        weak.append(quiz["short"]["tests"])
    total = (correct + sh["score"]) / 4
    return dict(rows=rows, short=sh, total=total, passed=(total >= 0.8 and sh["score"] >= 0.5),
                weak=weak_tags_ordered(weak))

def focus_text(rows, short, short_tag="apply", short_q="", short_answer=""):
    """Numbered list of the learner's mix-ups; each says which section of the new explanation must fix it."""
    lines, n = [], 0
    for r in rows:
        if r["ok"]:
            continue
        n += 1
        lines.append(f"Mix-up {n} (put the fix in {SECTION_OF.get(r['tests'], 'the most relevant section')}):\n"
                     f"  Question: {r['q']}\n  Learner chose: {r['chosen']}\n  Correct answer: {r['answer']}\n  Why: {r['why']}")
    if short["score"] < 0.5:
        n += 1
        what = short.get("misconception") or short.get("feedback") or "the written answer missed the key ideas"
        wrote = " ".join(str(short_answer or "").split())[:300] or "(left blank)"
        lines.append(f"Mix-up {n} (put the fix in {SECTION_OF.get(short_tag, 'the most relevant section')}):\n"
                     f"  Question: {short_q}\n  Learner wrote: {wrote}\n  What went wrong: {what}")
    return "\n".join(lines)

# --- Converted from notebook code cell 7 ---
# Rendering helpers: Black theme, animated journey tracker, interactive diagram, section splitting
CSS = """
:root{--c1:255,255,255;--c2:184,184,184;--c3:140,140,140;--c4:220,220,220;--ink:#000;--card-rgb:12,12,12;
--teal:rgb(var(--c1));--violet:rgb(var(--c2));--coral:rgb(var(--c3));--amber:rgb(var(--c4));
--line:rgba(255,255,255,.14);--text:#f2f2f2;--muted:#8c8c8c;--fs:1;color-scheme:dark}
html,body{background:radial-gradient(1100px 560px at 8% -8%,rgba(var(--c2),.13),transparent 60%),
  radial-gradient(900px 520px at 100% 2%,rgba(var(--c1),.08),transparent 55%),
  radial-gradient(800px 500px at 50% 110%,rgba(var(--c3),.07),transparent 60%),var(--ink) !important;background-attachment:fixed !important;transition:background-color .4s}
.gradio-container{max-width:1060px !important;margin:0 auto !important;color:var(--text);background:transparent !important}
footer{display:none !important}
::selection{background:rgba(var(--c1),.28);color:#fff}
*{scrollbar-width:thin;scrollbar-color:#2e2e2e #000}
::-webkit-scrollbar{width:10px;height:10px}::-webkit-scrollbar-track{background:#000}
::-webkit-scrollbar-thumb{background:#2a2a2a;border-radius:8px;border:2px solid #000}::-webkit-scrollbar-thumb:hover{background:#3a3a3a}

/* ---------- hero ---------- */
.cc-hero{position:relative;overflow:hidden;text-align:center;padding:42px 20px 30px;border-radius:26px;margin:6px 0 16px;
  border:1px solid var(--line);background:linear-gradient(160deg,#161616,#050505)}
.cc-orb{position:absolute;border-radius:50%;filter:blur(50px);opacity:.38;animation:ccfloat 9s ease-in-out infinite}
.cc-orb.o1{width:220px;height:220px;background:var(--violet);left:-50px;top:-60px}
.cc-orb.o2{width:200px;height:200px;background:var(--teal);right:-40px;top:20px;animation-delay:-3s}
.cc-orb.o3{width:160px;height:160px;background:var(--coral);left:42%;bottom:-90px;animation-delay:-6s}
@keyframes ccfloat{0%,100%{transform:translate(0,0)}50%{transform:translate(18px,-22px)}}
.cc-badge{position:relative;display:inline-block;font-size:12px;letter-spacing:.22em;padding:5px 14px;border-radius:999px;color:var(--teal);
  border:1px solid rgba(var(--c1),.4);background:rgba(var(--c1),.07)}
.cc-title{position:relative;font-size:clamp(2.4rem,6vw,3.9rem) !important;font-weight:800 !important;margin:14px 0 6px !important;line-height:1.05 !important;
  color:#fff !important;letter-spacing:-.02em}
.cc-title span{background:linear-gradient(110deg,var(--teal),var(--violet) 45%,var(--coral) 80%,var(--teal));background-size:250% 100%;
  -webkit-background-clip:text;background-clip:text;color:transparent;animation:ccshine 6s linear infinite}
@keyframes ccshine{to{background-position:250% 0}}
.cc-sub{position:relative;color:#b5b5b5;font-size:1.05rem;max-width:640px;margin:0 auto !important}
.cc-sub em{color:var(--amber);font-style:normal;font-weight:700}

/* ---------- cards ---------- */
.cc-card{background:rgba(var(--card-rgb),.82) !important;border:1px solid var(--line) !important;border-radius:20px !important;padding:18px 24px !important;
  backdrop-filter:blur(10px);box-shadow:0 12px 40px rgba(0,0,0,.6);animation:ccrise .45s ease both;transition:border-color .25s,transform .25s}
.cc-card:hover{border-color:rgba(var(--c1),.34) !important}
@keyframes ccrise{from{opacity:0;transform:translateY(14px)}to{opacity:1;transform:none}}
.cc-card:has(.prose:empty), .cc-note:has(.prose:empty), .cc-card .prose:empty{display:none !important}
.cc-card, .cc-card p, .cc-card li, .cc-card td{color:#e4e4e4 !important;line-height:1.7}
.cc-top .prose,.cc-bottom .prose{font-size:calc(1.02rem * var(--fs)) !important}
.cc-card strong{color:#fff !important}
.cc-card h2{font-size:calc(1.35rem * var(--fs)) !important;margin:1.3em 0 .55em !important;padding-bottom:.35em;border-bottom:1px solid var(--line)}
.cc-card h2:first-child{margin-top:.2em !important}
.cc-top h2:nth-of-type(1){color:var(--teal) !important}
.cc-top h2:nth-of-type(2){color:var(--violet) !important}
.cc-top h2:nth-of-type(3){color:var(--coral) !important}
.cc-bottom h2{color:var(--amber) !important}
.cc-card li::marker{color:var(--teal)}
.cc-card pre{background:#050505 !important;border:1px solid var(--line) !important;border-radius:14px !important}
.cc-card code{color:rgb(var(--c1)) !important}
.cc-card blockquote{border-left:4px solid var(--violet);background:rgba(var(--c2),.09);border-radius:12px;padding:8px 16px;margin:14px 0;color:#e4e4e4}
.cc-card blockquote p{margin:.25em 0}
.cc-top h2,.cc-h,.cc-bottom,.cc-quiz{scroll-margin-top:120px}

/* ---------- journey tracker ---------- */
.cc-journey{display:flex;align-items:flex-start;justify-content:space-between;padding:18px 10px 8px;margin:4px 0}
.cc-node{display:flex;flex-direction:column;align-items:center;gap:7px;flex:0 0 auto;width:70px;font-size:12px;color:var(--muted);text-align:center;cursor:pointer;user-select:none}
.cc-orbn{width:44px;height:44px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:19px;
  background:#0a0a0a;border:1.5px solid var(--line);transition:all .35s}
.cc-node:hover .cc-orbn{border-color:var(--teal);transform:translateY(-2px)}
.cc-node:hover{color:#fff}
.cc-node.done .cc-orbn{background:linear-gradient(135deg,var(--teal),var(--violet));border-color:transparent;color:#000;font-weight:800;box-shadow:0 0 18px rgba(var(--c1),.35)}
.cc-node.done{color:#f0f0f0}
.cc-node.active .cc-orbn{border-color:var(--violet);background:#1a1a1a;animation:ccring 1.3s infinite}
.cc-node.active{color:#fff}
.cc-node.error .cc-orbn{border-color:#ff6b81;background:#2a0f16}
@keyframes ccring{0%{box-shadow:0 0 0 0 rgba(var(--c2),.55)}100%{box-shadow:0 0 0 16px rgba(var(--c2),0)}}
.cc-link{flex:1 1 auto;height:3px;margin-top:21px;border-radius:3px;background:rgba(255,255,255,.12);position:relative;overflow:hidden}
.cc-link.done{background:linear-gradient(90deg,var(--teal),var(--violet))}
.cc-link.active::after{content:"";position:absolute;inset:0;width:40%;background:linear-gradient(90deg,transparent,var(--violet),transparent);animation:ccslide 1.1s infinite}
@keyframes ccslide{from{left:-40%}to{left:100%}}
.cc-cap{text-align:center;color:var(--muted);font-size:13px;margin:2px 0 8px}
.cc-typing{display:inline-block;margin-left:6px;vertical-align:middle}
.cc-typing i{display:inline-block;width:6px;height:6px;margin:0 2px;border-radius:50%;background:var(--teal);animation:ccbounce 1.2s infinite}
.cc-typing i:nth-child(2){animation-delay:.15s}.cc-typing i:nth-child(3){animation-delay:.3s}
@keyframes ccbounce{0%,80%,100%{transform:translateY(0);opacity:.4}40%{transform:translateY(-6px);opacity:1}}
.cc-lines{padding:6px 6px 2px}
.cc-lines i{display:block;height:14px;border-radius:8px;margin:12px 0;
  background:linear-gradient(90deg,rgba(255,255,255,.05) 25%,rgba(var(--c2),.24) 50%,rgba(255,255,255,.05) 75%);background-size:200% 100%;animation:ccshimmer 1.4s infinite linear}
.cc-lines i:nth-child(1){width:92%}.cc-lines i:nth-child(2){width:78%}.cc-lines i:nth-child(3){width:86%}.cc-lines i:nth-child(4){width:52%}
@keyframes ccshimmer{0%{background-position:200% 0}100%{background-position:-200% 0}}

/* ---------- diagram ---------- */
.cc-h{font-size:1.35rem;font-weight:700;margin:22px 0 10px;color:var(--violet)}
.cc-skel{height:clamp(540px,78vh,860px);border-radius:18px;border:1px dashed rgba(var(--c2),.5);display:flex;align-items:center;justify-content:center;color:#cfcfcf;font-size:15px;
  background:linear-gradient(90deg,rgba(255,255,255,.03) 25%,rgba(var(--c2),.18) 50%,rgba(255,255,255,.03) 75%);background-size:200% 100%;animation:ccshimmer 1.4s infinite linear}
.cc-frame{width:100%;height:clamp(540px,78vh,860px);border:1px solid var(--line);border-radius:18px;background:#000;box-shadow:0 0 0 1px rgba(var(--c1),.06),0 14px 44px rgba(0,0,0,.7)}
.cc-err{padding:16px;border-radius:16px;border:1px dashed rgba(255,107,129,.6);background:rgba(255,107,129,.08);color:#ffc2d0}

/* ---------- input, chips, buttons ---------- */
.cc-input textarea,.cc-input input{background:rgba(10,10,10,.95) !important;color:#fff !important;font-size:1.12rem !important;
  border:1px solid var(--line) !important;border-radius:16px !important;padding:14px 16px !important;transition:all .2s}
.cc-input textarea:focus,.cc-input input:focus{border-color:var(--teal) !important;box-shadow:0 0 0 3px rgba(var(--c1),.16),0 0 26px rgba(var(--c1),.14) !important}
.cc-go{background:linear-gradient(120deg,var(--teal),var(--violet)) !important;color:#000 !important;font-weight:800 !important;border:none !important;border-radius:16px !important;
  box-shadow:0 8px 28px rgba(var(--c1),.2);transition:transform .15s,box-shadow .15s,filter .15s}
.cc-go:hover{transform:translateY(-2px) scale(1.03);box-shadow:0 12px 36px rgba(var(--c2),.35);filter:brightness(1.08)}
.cc-go:active{transform:scale(.98)}
.cc-go.ready{animation:ccpulse 1.7s infinite}
@keyframes ccpulse{0%{box-shadow:0 0 0 0 rgba(var(--c1),.55)}100%{box-shadow:0 0 0 18px rgba(var(--c1),0)}}
.cc-chips{gap:8px !important;flex-wrap:wrap !important}
.cc-chipbtn{min-width:0 !important;flex:0 0 auto !important;border-radius:999px !important;background:rgba(255,255,255,.05) !important;color:#d6d6d6 !important;
  border:1px solid var(--line) !important;font-size:13px !important;padding:6px 14px !important;transition:all .18s}
.cc-chipbtn:hover{background:rgba(var(--c2),.18) !important;border-color:var(--violet) !important;transform:translateY(-2px);box-shadow:0 6px 18px rgba(var(--c2),.2)}

/* ---------- quiz ---------- */
.cc-quiz{--checkbox-background-color-selected:#262626;--checkbox-background-color-selected-hover:#2e2e2e;--checkbox-border-color-selected:rgb(var(--c1));--checkbox-border-color-selected-dark:rgb(var(--c1))}
.cc-radio label{background:rgba(255,255,255,.04) !important;border:1px solid var(--line) !important;border-radius:14px !important;padding:11px 16px !important;
  margin:4px 0 !important;cursor:pointer;transition:all .18s}
.cc-radio label:hover{transform:translateX(5px);border-color:var(--teal) !important;background:rgba(var(--c1),.08) !important}
.cc-radio label.selected{background:linear-gradient(90deg,rgba(var(--c2),.28),rgba(var(--c1),.10)) !important;border-color:var(--violet) !important;
  box-shadow:0 0 0 1px var(--violet),0 8px 26px rgba(0,0,0,.5)}
.cc-radio label span,.cc-radio label *{color:#f0f0f0 !important}
.cc-radio span[data-testid="block-info"],.cc-quiz .block-info{color:#fff !important;font-weight:700;font-size:1.02rem !important}
.cc-quiz textarea{background:rgba(10,10,10,.95) !important;color:#fff !important;border:1px solid var(--line) !important;border-radius:14px !important}
.cc-quiz textarea:focus{border-color:var(--amber) !important;box-shadow:0 0 0 3px rgba(var(--c4),.16) !important}
.cc-qprog{display:flex;align-items:center;gap:12px;margin:2px 0 6px}
.cc-qbar{flex:1;height:6px;border-radius:6px;background:rgba(255,255,255,.1);overflow:hidden}
.cc-qbar i{display:block;height:100%;width:0;border-radius:6px;background:linear-gradient(90deg,var(--teal),var(--violet));transition:width .35s}
.cc-qtxt{font-size:12px;color:var(--muted);letter-spacing:.06em;white-space:nowrap}

/* ---------- note + result ---------- */
.cc-note blockquote{border-left:4px solid var(--amber);background:rgba(var(--c4),.08);border-radius:14px;padding:14px 18px;margin:6px 0;color:#f0f0f0 !important;animation:ccrise .4s ease both}
.cc-note p,.cc-note strong,.cc-note em{color:#f0f0f0 !important}
.cc-note h3{color:#fff !important;font-size:1.25rem !important;margin:.3em 0 .6em !important}
.cc-note ul{margin:.3em 0 .2em 1.1em}.cc-note li{color:#f0f0f0 !important;line-height:1.6}
.cc-note blockquote{margin:10px 0}
.cc-result h3,.cc-result h2{background:linear-gradient(110deg,var(--teal),var(--violet),var(--coral));-webkit-background-clip:text;background-clip:text;color:transparent !important;font-weight:800}
.cc-result h2{font-size:1.7rem !important}
.cc-foot{text-align:center;color:var(--muted);font-size:12px;margin:18px 0 6px;letter-spacing:.08em}
.cc-confetti{position:fixed;top:-14px;width:10px;height:15px;z-index:99999;pointer-events:none;border-radius:2px;animation:ccfall linear forwards}
@keyframes ccfall{to{transform:translateY(108vh) rotate(760deg);opacity:.85}}

/* ---------- v3 additions ---------- */
.cc-card,.cc-hero{position:relative}
.cc-card::before,.cc-hero::before{content:"";position:absolute;inset:0;border-radius:inherit;pointer-events:none;opacity:0;transition:opacity .3s;
  background:radial-gradient(420px circle at var(--mx,50%) var(--my,-40%),rgba(var(--c2),.12),transparent 45%)}
.cc-card:hover::before{opacity:1}
.cc-hero::before{opacity:1;background:radial-gradient(380px circle at var(--mx,50%) var(--my,0%),rgba(var(--c2),.18),transparent 45%)}
.cc-sticky{position:sticky;top:8px;z-index:40}
.cc-sticky .cc-journey{background:rgba(var(--card-rgb),.9);backdrop-filter:blur(14px);border:1px solid var(--line);border-radius:18px;
  box-shadow:0 10px 30px rgba(0,0,0,.65);padding:14px 16px 8px}
.cc-cap b{color:#fff}
.cc-palrow{justify-content:flex-end !important;align-items:center !important;gap:8px !important;margin:-6px 0 10px !important;flex-wrap:wrap !important}
.cc-paltxt{color:var(--muted);font-size:12px;letter-spacing:.14em;white-space:nowrap}
.cc-pal{min-width:0 !important;flex:0 0 auto !important;border-radius:999px !important;background:rgba(255,255,255,.05) !important;border:1px solid var(--line) !important;
  color:#d6d6d6 !important;font-size:12px !important;padding:4px 12px !important;transition:all .18s}
.cc-pal:hover{transform:translateY(-2px)}
.cc-pal.on{border-color:var(--teal) !important;color:#fff !important;box-shadow:0 0 0 1px var(--teal),0 0 16px rgba(var(--c1),.3)}
.cc-h small{font-size:12px;font-weight:400;color:var(--muted);margin-left:10px}
.cc-dl{align-items:center !important;gap:12px !important;flex-wrap:wrap !important}
.cc-dltxt{color:#e4e4e4;font-size:14px;line-height:1.45}
.cc-dltxt span{color:var(--muted);font-size:12px}

/* ---------- v4: black theme + interactivity ---------- */
#cc-progress{position:fixed;top:0;left:0;height:3px;width:0;z-index:100000;background:linear-gradient(90deg,var(--teal),var(--violet));
  box-shadow:0 0 12px rgba(var(--c1),.6);transition:width .12s linear;pointer-events:none}
#cc-totop{position:fixed;right:22px;bottom:24px;width:44px;height:44px;border-radius:50%;z-index:9999;cursor:pointer;font-size:20px;line-height:1;
  color:#fff;background:rgba(12,12,12,.92);border:1px solid var(--line);box-shadow:0 8px 24px rgba(0,0,0,.6);
  opacity:0;transform:translateY(14px);pointer-events:none;transition:all .25s}
#cc-totop.show{opacity:1;transform:none;pointer-events:auto}
#cc-totop:hover{border-color:var(--teal);box-shadow:0 0 0 1px var(--teal),0 8px 24px rgba(0,0,0,.6)}
.cc-toast{position:fixed;left:50%;bottom:28px;transform:translate(-50%,20px);z-index:100001;padding:10px 18px;border-radius:999px;font-size:13px;color:#fff;
  background:rgba(14,14,14,.96);border:1px solid var(--line);box-shadow:0 10px 30px rgba(0,0,0,.7);opacity:0;pointer-events:none;transition:all .25s}
.cc-toast.show{opacity:1;transform:translate(-50%,0)}
.cc-histwrap{margin-top:2px !important}
.cc-histrow{display:flex;flex-wrap:wrap;align-items:center;gap:8px;padding:2px 0}
.cc-hist{cursor:pointer;border-radius:999px;background:transparent;color:#cfcfcf;border:1px dashed rgba(255,255,255,.22);font-size:12.5px;padding:4px 12px;
  font-family:inherit;transition:all .18s}
.cc-hist:hover{color:#fff;border-style:solid;border-color:var(--teal);background:rgba(var(--c1),.07);transform:translateY(-2px)}
.cc-hint{color:var(--muted);font-size:12px;padding:2px 2px 0}
.cc-hint kbd{font-family:inherit;font-size:11px;padding:1px 6px;border-radius:6px;border:1px solid var(--line);background:rgba(255,255,255,.06);color:#d6d6d6}
.cc-acc{background:rgba(var(--card-rgb),.82) !important;border:1px solid var(--line) !important;border-radius:20px !important;padding:4px 18px !important;margin-top:10px}
.cc-acc .label-wrap,.cc-acc .label-wrap span{color:#f2f2f2 !important;font-weight:700}
.cc-prof{padding:4px 2px 10px}
.cc-pinfo{color:#d6d6d6;font-size:13.5px;margin:2px 0 12px}
.cc-pinfo b{color:#fff}
.cc-pempty{color:var(--muted);font-size:13.5px}
.cc-prow{display:flex;align-items:center;gap:12px;margin:9px 0}
.cc-pname{width:210px;flex:0 0 auto;font-size:13px;color:#d6d6d6}
.cc-pbar{flex:1;height:8px;border-radius:8px;background:rgba(255,255,255,.08);overflow:hidden}
.cc-pbar i{display:block;height:100%;border-radius:8px;background:linear-gradient(90deg,var(--teal),var(--violet));transition:width .5s}
.cc-pbar i.neg{background:rgba(255,107,129,.85)}
.cc-pval{width:34px;text-align:right;font-size:12px;color:var(--muted)}
body.cc-calm .cc-orb,body.cc-calm .cc-title span,body.cc-calm .cc-card,body.cc-calm .cc-go.ready,body.cc-calm .cc-lines i,body.cc-calm .cc-skel{animation:none !important}
body.cc-calm .cc-card,body.cc-calm .cc-sticky .cc-journey{backdrop-filter:none}
body.cc-calm .cc-card::before,body.cc-calm .cc-hero::before{display:none}
@media (max-width:640px){.cc-pname{width:120px}.cc-node{width:52px;font-size:11px}.cc-orbn{width:38px;height:38px}}
"""

CONFETTI_JS = """(txt) => {
  if (!txt || !txt.includes('perfectly')) return;
  const cs = getComputedStyle(document.documentElement);
  const cols = ['--teal', '--violet', '--coral', '--amber'].map(v => cs.getPropertyValue(v).trim());
  for (let i = 0; i < 90; i++) {
    const d = document.createElement('div');
    d.className = 'cc-confetti';
    d.style.left = (Math.random() * 100) + 'vw';
    d.style.background = cols[i % 4];
    d.style.animationDuration = (2 + Math.random() * 2.5) + 's';
    d.style.animationDelay = (Math.random() * 0.6) + 's';
    document.body.appendChild(d);
    setTimeout(() => d.remove(), 5600);
  }
}"""

HERO_HTML = """<div class="cc-hero"><div class="cc-orb o1"></div><div class="cc-orb o2"></div><div class="cc-orb o3"></div>
<div class="cc-badge">\u2726 ADAPTIVE AI TUTOR</div>
<h1 class="cc-title">ConceptClarity</h1>
<p class="cc-sub">Any concept, taught the way <em>you</em> learn. Why \u2192 What \u2192 How \u2192 Diagram \u2192 Examples \u2192 Quiz. Miss something? I re-teach it in a brand-new style.</p></div>"""

FOOT_HTML = '<div class="cc-foot">\u2726 BUILT WITH GRADIO \u00b7 GEMINI \u00b7 GROQ \u2726</div>'

JOURNEY = [("why", "Why", "\U0001F4A1"), ("what", "What", "\U0001F4D8"), ("how", "How", "\u2699\ufe0f"),
           ("diagram", "Diagram", "\U0001F5FA\ufe0f"), ("examples", "Examples", "\U0001F30D"), ("quiz", "Quiz", "\U0001F3AF")]
_HEAD_PAT = {"why": r"^##\s*1", "what": r"^##\s*2", "how": r"^##\s*3", "examples": r"^##\s*5"}

def journey_states(steps, buf=""):
    text_keys = ["why", "what", "how", "examples"]
    seen = [k for k in text_keys if re.search(_HEAD_PAT[k], buf, re.M)]
    streaming = steps["explain"] == "active"
    st = {}
    for k in text_keys:
        if k in seen:
            st[k] = "active" if (streaming and k == seen[-1]) else "done"
        else:
            st[k] = "idle"
    if streaming and not seen:
        st["why"] = "active"
    st["diagram"] = steps["diagram"]
    st["quiz"] = steps["quiz"]
    return st

def status_html(steps, waiting_first_token=False, buf="", profile=None, style=None):
    st = journey_states(steps, buf)
    parts = []
    for i, (k, label, icon) in enumerate(JOURNEY):
        s = st[k]
        shown = "\u2713" if s == "done" else ("\u26a0\ufe0f" if s == "error" else icon)
        parts.append(f'<div class="cc-node {s}" data-k="{k}" title="Jump to {label}"><div class="cc-orbn">{shown}</div><span>{label}</span></div>')
        if i < len(JOURNEY) - 1:
            nxt = st[JOURNEY[i + 1][0]]
            ls = "done" if s == "done" else ("active" if s == "active" else "idle")
            if s == "done" and nxt == "idle" and steps["explain"] == "active":
                ls = "active"
            parts.append(f'<div class="cc-link {ls}"></div>')
    label = STYLES[style]["label"] if style in STYLES else ""
    best = None
    if profile:
        sc = {k: v for k, v in profile["score"].items() if v > 0}
        best = max(sc, key=sc.get) if sc else None
    if all(v == "idle" for v in steps.values()):
        cap = "Your learning journey \u2014 ask for any concept to begin"
    elif waiting_first_token:
        cap = f'Warming up a <b>{label}</b> explanation<span class="cc-typing"><i></i><i></i><i></i></span>'
    else:
        bits = []
        if label:
            bits.append(f"\U0001F393 Teaching style: <b>{label}</b>")
        if best:
            bits.append(f"\u2728 Works best for you: <b>{STYLES[best]['label']}</b>")
        cap = " &nbsp;\u00b7&nbsp; ".join(bits)
    lines = '<div class="cc-lines"><i></i><i></i><i></i><i></i></div>' if waiting_first_token else ""
    return f'<div class="cc-journey">{"".join(parts)}</div><div class="cc-cap">{cap}</div>{lines}'

IFRAME_DOC = """<!doctype html><html><head><meta charset="utf-8"><style>
html,body{margin:0;height:100%;overflow:hidden;background:#000;color:#f2f2f2;font-family:system-ui,sans-serif;--a1:235,235,235;--a2:150,150,150}
#wrap{position:absolute;inset:0;overflow:auto;cursor:grab}
#wrap.drag{cursor:grabbing;user-select:none}
#box{display:flex;min-width:100%;min-height:100%;padding:56px 20px 20px;box-sizing:border-box}
#box svg{margin:auto;display:block;flex:0 0 auto}
#bar{position:absolute;top:10px;right:12px;display:flex;gap:6px;z-index:5}
#bar button{background:rgba(255,255,255,.07);color:#eee;border:1px solid rgba(255,255,255,.28);border-radius:9px;padding:3px 11px;cursor:pointer;font-size:14px;transition:all .15s;backdrop-filter:blur(6px)}
#bar button:hover{background:rgba(var(--a1),.22);border-color:rgb(var(--a1));transform:translateY(-1px)}
#hint{position:absolute;left:14px;top:14px;font-size:12px;color:#8c8c8c;z-index:5;pointer-events:none}
pre.mermaid{margin:auto;color:#8c8c8c;font-size:12px}
/* accent colours follow the page's theme switcher (sent by postMessage) */
.node rect,.node polygon,.node circle,.node ellipse,.node path,.actor{stroke:rgb(var(--a1)) !important}
.flowchart-link,.edgePaths path,.messageLine0,.messageLine1,.actor-line,.cluster rect{stroke:rgb(var(--a2)) !important}
.marker,.arrowheadPath,marker path{fill:rgb(var(--a2)) !important;stroke:rgb(var(--a2)) !important}
</style></head><body>
<div id="hint">Ctrl + scroll to zoom \u00b7 drag to pan</div>
<div id="bar"><button id="zo" title="Zoom out">\u2212</button><button id="z0" title="Fit to the available space">Fit</button><button id="zi" title="Zoom in">+</button><button id="fs" title="Fullscreen">\u26f6</button><button id="dl" title="Download as SVG">\u2b07 SVG</button></div>
<div id="wrap"><div id="box"><pre class="mermaid">__CODE__</pre></div></div>
<script type="module">
import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@10.9.1/dist/mermaid.esm.min.mjs";
const src = document.querySelector(".mermaid").textContent;
const okc = (v) => /^[0-9]{1,3},[0-9]{1,3},[0-9]{1,3}$/.test(String(v));
window.addEventListener("message", (e) => { const a = e.data && e.data.ccAccent;
  if (a && okc(a.a1) && okc(a.a2)) { document.documentElement.style.setProperty("--a1", a.a1); document.documentElement.style.setProperty("--a2", a.a2); } });
try { parent.postMessage({ccReady: 1}, "*"); } catch (e) {}
mermaid.initialize({startOnLoad:false, securityLevel:"strict", theme:"base", htmlLabels:false, flowchart:{useMaxWidth:false, htmlLabels:false, curve:"basis", padding:14},
  sequence:{useMaxWidth:false},
  themeVariables:{background:"#000000", fontSize:"16px", primaryColor:"#171717", primaryTextColor:"#f5f5f5", primaryBorderColor:"#e5e5e5", lineColor:"#a3a3a3",
    secondaryColor:"#1f1f1f", tertiaryColor:"#121212", clusterBkg:"#0d0d0d", clusterBorder:"#737373", edgeLabelBackground:"#000000",
    fontFamily:"system-ui,sans-serif", actorBkg:"#171717", actorBorder:"#e5e5e5", actorTextColor:"#f5f5f5", actorLineColor:"#a3a3a3",
    signalColor:"#e5e5e5", signalTextColor:"#f5f5f5", noteBkgColor:"#262626", noteTextColor:"#ffffff", noteBorderColor:"#a3a3a3",
    labelBoxBkgColor:"#171717", labelBoxBorderColor:"#a3a3a3", labelTextColor:"#f5f5f5", loopTextColor:"#f5f5f5", sequenceNumberColor:"#000000"}});
try {
  await mermaid.run({querySelector:".mermaid"});
  const wrap = document.getElementById("wrap"), box = document.getElementById("box");
  const svg = box.querySelector("svg"); svg.removeAttribute("style"); box.innerHTML = ""; box.appendChild(svg);
  const vb = svg.viewBox.baseVal, vw = vb.width || 600, vh = vb.height || 400;
  let base = 1, zoom = 1;
  const apply = () => { svg.setAttribute("width", vw*base*zoom); svg.setAttribute("height", vh*base*zoom); };
  const fit = () => { const W = wrap.clientWidth - 40, H = wrap.clientHeight - 76;
    base = Math.min(2.4, Math.max(0.2, Math.min(W/vw, H/vh))); apply(); };
  const setz = (v) => { zoom = Math.min(4, Math.max(0.3, v)); apply(); };
  fit(); new ResizeObserver(fit).observe(wrap);
  document.getElementById("zi").onclick = () => setz(zoom*1.2);
  document.getElementById("zo").onclick = () => setz(zoom/1.2);
  document.getElementById("z0").onclick = () => { zoom = 1; fit(); wrap.scrollTo(0, 0); };
  document.getElementById("fs").onclick = () => { if (document.fullscreenElement) document.exitFullscreen(); else document.documentElement.requestFullscreen().catch(() => {}); };
  document.getElementById("dl").onclick = () => { const s = new XMLSerializer().serializeToString(svg);
    const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([s], {type:"image/svg+xml"})); a.download = "diagram.svg"; a.click(); };
  wrap.addEventListener("wheel", (e) => { if (e.ctrlKey || e.metaKey) { e.preventDefault(); setz(zoom*(e.deltaY < 0 ? 1.1 : 0.9)); } }, {passive:false});
  let sx = 0, sy = 0, sl = 0, st = 0, drag = false;
  wrap.addEventListener("mousedown", (e) => { drag = true; sx = e.clientX; sy = e.clientY; sl = wrap.scrollLeft; st = wrap.scrollTop; wrap.classList.add("drag"); });
  window.addEventListener("mouseup", () => { drag = false; wrap.classList.remove("drag"); });
  window.addEventListener("mousemove", (e) => { if (drag) { wrap.scrollLeft = sl - (e.clientX - sx); wrap.scrollTop = st - (e.clientY - sy); } });
} catch (e) {
  document.body.innerHTML = "";
  const d = document.createElement("div"); d.style.cssText = "padding:16px;color:#ffb4c2";
  d.textContent = "Could not draw this diagram. Raw Mermaid code:";
  const p = document.createElement("pre"); p.style.cssText = "padding:0 16px;white-space:pre-wrap;color:#cfd8ff"; p.textContent = src;
  document.body.style.overflow = "auto"; document.body.append(d, p);
}
</script></body></html>"""

def diagram_html(state, code=None):
    head = '<div class="cc-h">4\ufe0f\u20e3 Diagram <small>drag to pan \u00b7 Ctrl+scroll to zoom \u00b7 \u26f6 fullscreen \u00b7 \u2b07 SVG</small></div>'
    if state == "loading":
        return head + '<div class="cc-skel">\U0001F3A8 Drawing your diagram\u2026</div>'
    if state == "error":
        return head + '<div class="cc-err">The diagram could not be generated this time - the written explanation above covers the same flow.</div>'
    doc = IFRAME_DOC.replace("__CODE__", html.escape(code, quote=False))
    return head + (f'<iframe class="cc-frame" sandbox="allow-scripts allow-downloads" allow="fullscreen" allowfullscreen '
                   f'srcdoc="{html.escape(doc, quote=True)}"></iframe>')

def split_sections(buf):
    """Text before/after the diagram marker; hides a half-typed marker while streaming."""
    m = MARK_RE.search(buf)
    if m:
        return buf[:m.start()].rstrip(), buf[m.end():].lstrip()
    for k in range(len(MARK) - 1, 0, -1):
        if buf.endswith(MARK[:k]):
            return buf[:-k], ""
    return buf, ""

# --- Converted from notebook code cell 8 ---
# Agent core: one streamed call + diagram and quiz generated in parallel
import gradio as gr

EXPLAIN_CACHE = {}   # (concept, style) -> {"text", "diagram"}  -> repeat questions are instant

def _done_future(val):
    f = Future(); f.set_result(val); return f

def blank_view(note="", result=""):
    return dict(status="", note=note, top="", diagram="", bottom="", result=result,
                quiz_visible=False, quiz_dirty=True, qhead="", qs=[], short_q="", _d=False, _q=False)

def pack(v, S):
    if v["quiz_dirty"]:
        if v["quiz_visible"]:
            qu = [gr.update(visible=True), gr.update(value=v["qhead"])]
            qu += [gr.update(label=l, choices=c, value=None) for l, c in v["qs"]]
            qu += [gr.update(label=v["short_q"], value="")]
        else:
            qu = [gr.update(visible=False)] + [gr.update()] * 5
        v["quiz_dirty"] = False
    else:
        qu = [gr.update()] * 6
    return (v["status"], v["note"], v["top"], v["diagram"], v["bottom"], *qu, v["result"], S)

def only_result(md, S):
    return (gr.update(),) * 11 + (md, S)

def poll(v, S, steps, fd, fq):
    if not v["_d"] and fd.done():
        v["_d"] = True
        try:
            v["diagram"] = diagram_html("ready", fd.result()); steps["diagram"] = "done"
        except Exception:
            v["diagram"] = diagram_html("error"); steps["diagram"] = "error"
    if not v["_q"] and fq.done() and steps["explain"] != "active":
        v["_q"] = True
        try:
            quiz = fq.result()
            S["cur"]["quiz"] = quiz
            S["cur"]["asked"] += [m["q"] for m in quiz["mcqs"]] + [quiz["short"]["q"]]
            v["quiz_visible"], v["quiz_dirty"] = True, True
            v["qhead"] = "## 6\ufe0f\u20e3 Check your understanding\nPick one answer per question, then answer the last one in your own words."
            v["qs"] = [(f"Q{i+1}. {m['q']}", m["options"]) for i, m in enumerate(quiz["mcqs"])]
            v["short_q"] = f"Q4. {quiz['short']['q']}"
            steps["quiz"] = "done"
        except Exception:
            steps["quiz"] = "error"
            v["result"] = "\u26a0\ufe0f Couldn't build the quiz this time. Ask the concept again to retry."

def run_explanation(concept, style, focus, S, note="", result=""):
    cur = S["cur"]
    v = blank_view(note, result)
    steps = {"explain": "active", "diagram": "active", "quiz": "active"}
    v["diagram"] = diagram_html("loading")
    v["status"] = status_html(steps, waiting_first_token=True, style=style)
    yield pack(v, S)                                          # instant feedback: animations appear immediately

    key = (concept.strip().lower(), style)
    cached = EXPLAIN_CACHE.get(key) if (cur["attempt"] == 0 and not focus) else None
    fd = _done_future(cached["diagram"]) if cached else POOL.submit(make_diagram, concept, style, focus)
    fq = POOL.submit(make_quiz, concept, list(cur["asked"]))  # quiz is prepared while the learner reads

    buf, last, failed = "", 0.0, False
    if cached:
        buf = cached["text"]
    else:
        try:
            for kind, t in stream_llm(explain_prompt(concept, style, focus), TEACHER_SYSTEM):
                if kind == "reset":
                    buf = ""; continue
                buf += t
                now = time.time()
                if now - last > 0.05:                         # throttle UI updates
                    last = now
                    v["top"], v["bottom"] = split_sections(buf)
                    poll(v, S, steps, fd, fq)
                    v["status"] = status_html(steps, buf=buf, profile=S["profile"], style=style)
                    yield pack(v, S)
        except Exception as e:
            failed = True
            v["top"] = f"\u26a0\ufe0f Couldn't reach the AI providers: {e}"
    v["top"], v["bottom"] = split_sections(buf) if buf else (v["top"], "")
    steps["explain"] = "error" if failed else "done"

    deadline = time.time() + 45
    while not (fd.done() and fq.done()) and time.time() < deadline:
        poll(v, S, steps, fd, fq)
        v["status"] = status_html(steps, buf=buf, profile=S["profile"], style=style)
        yield pack(v, S)
        time.sleep(0.2)
    poll(v, S, steps, fd, fq)
    if not v["_d"]:
        steps["diagram"] = "error"; v["diagram"] = diagram_html("error")
    v["status"] = status_html(steps, buf=buf, profile=S["profile"], style=style)
    if not failed and buf and v["_d"] and steps["diagram"] == "done" and not cached and cur["attempt"] == 0 and not focus:
        EXPLAIN_CACHE[key] = {"text": buf, "diagram": fd.result()}
    if not failed and buf:   # remembered for the download buttons
        cur["export"] = {"concept": concept, "style": STYLES[style]["label"], "text": buf,
                         "diagram": (fd.result() if (v["_d"] and steps["diagram"] == "done") else None)}
    yield pack(v, S)

# ---- event handlers ----
OK, NO = "\u2705", "\u274c"   # (kept out of f-string expressions for Python < 3.12)
def on_explain(concept, S):
    concept = (concept or "").strip()
    S = S or new_state()
    if not concept:
        yield only_result("Type a concept first, e.g. *Recursion* or *TCP handshake*.", S)
        return
    S["cur"] = new_cur(concept, pick_first_style(S["profile"]))
    yield from run_explanation(concept, S["cur"]["style"], "", S)

def _one_line(t, n=400):
    t = " ".join(str(t or "").split())
    return t if len(t) <= n else t[:n] + "…"

def review_md(cur, g, short_answer, mistakes_only):
    """Structured results: score, what was right, and one block per mix-up (numbered Mix-up 1, 2... like the re-teach callouts)."""
    quiz = cur["quiz"]
    filled = int(round(g["total"] * 4))
    md = [f"### 🔍 Your results: {g['total'] * 4:.1f} / 4 &nbsp; {'▰' * filled}{'▱' * (4 - filled)}"]
    right = [f"Q{i}" for i, r in enumerate(g["rows"], 1) if r["ok"]]
    if g["short"]["score"] >= 0.5:
        right.append("Q4")
    if right:
        md.append("✅ **You got right:** " + ", ".join(right))
    n = 0
    for i, r in enumerate(g["rows"], 1):
        if r["ok"]:
            if not mistakes_only:
                md.append(f"> **✅ Q{i}.** {r['q']}\n>\n> - **Your answer:** {r['chosen']}\n> - **Why it is right:** {r['why']}")
            continue
        n += 1
        md.append(f"> **❌ Mix-up {n} · Q{i}** (about *{TAG_NAMES.get(r['tests'], r['tests'])}*)\n>\n"
                  f"> - **Question:** {r['q']}\n> - **You answered:** {r['chosen']}\n"
                  f"> - **Correct answer:** {r['answer']}\n> - **Why:** {r['why']}")
    sh, sq = g["short"], quiz["short"]
    if sh["score"] >= 0.5:
        if not mistakes_only:
            md.append(f"> **✅ Q4.** {sq['q']}\n>\n> - **Feedback:** {sh['feedback']}")
    else:
        n += 1
        lines = [f"> **❌ Mix-up {n} · Q4** (about *{TAG_NAMES.get(sq['tests'], sq['tests'])}*)", ">",
                 f"> - **Question:** {sq['q']}",
                 f"> - **You wrote:** {_one_line(short_answer) or '*(left blank)*'}",
                 f"> - **Feedback:** {sh['feedback']}"]
        if sh.get("misconception"):
            lines.append(f"> - **The misunderstanding:** {sh['misconception']}")
        md.append("\n".join(lines))
    return "\n\n".join(md)

def on_submit(a1, a2, a3, short, S):
    cur = (S or {}).get("cur")
    if not cur or not cur.get("quiz"):
        yield only_result("Ask for a concept first.", S); return
    if cur["done"]:
        yield only_result("\u2705 This concept is already completed - ask a new concept to continue.", S); return
    picks = [a1, a2, a3]
    if any(p is None for p in picks):
        yield only_result("\u261d\ufe0f Please answer all three multiple-choice questions first.", S); return

    g = grade(cur, picks, short)

    if g["passed"]:
        cur["done"] = True
        update_profile(S["profile"], cur["style"], True, cur["attempt"])
        md = [review_md(cur, g, short, False),
              f"## \U0001F389 You've understood **{cur['concept']}** perfectly!\nAsk me another concept whenever you're ready."]
        yield only_result("\n\n".join(md), S); return

    update_profile(S["profile"], cur["style"], False, cur["attempt"])
    if cur["attempt"] + 1 >= MAX_ATTEMPTS:
        cur["done"] = True
        md = [review_md(cur, g, short, False),
              "### Let's narrow it down\nTry asking about the exact part that's confusing, e.g. *\"just the how of "
              f"{cur['concept']}\"* or *\"{cur['concept']} with a simpler example\"*."]
        yield only_result("\n\n".join(md), S); return

    old = cur["style"]
    new = pick_next_style(cur["tried"], g["weak"], S["profile"])
    focus = focus_text(g["rows"], g["short"], cur["quiz"]["short"]["tests"], cur["quiz"]["short"]["q"], short)
    review = review_md(cur, g, short, True)               # built BEFORE the new quiz replaces cur["quiz"]
    cur["style"], cur["attempt"] = new, cur["attempt"] + 1
    cur["tried"].append(new)
    weak_names = ", ".join(TAG_NAMES.get(t, t) for t in g["weak"]) or "a few details"
    switch = (f"> \U0001F504 **Let me explain it differently.** You found *{weak_names}* tricky, so I'm switching from "
              f"**{STYLES[old]['label']}** to **{STYLES[new]['label']}**. Each mix-up above is fixed inside the matching section below "
              f"(look for the \U0001F527 callouts), and the diagram is redrawn to make the right version obvious.")
    yield from run_explanation(cur["concept"], new, focus, S, note=review + "\n\n" + switch)

# --- Converted from notebook code cell 9 ---
# Downloads: pretty HTML (diagram included, print-to-PDF ready) + plain Markdown
EXPORT_DIR = "cc_downloads"
os.makedirs(EXPORT_DIR, exist_ok=True)

def _slug(t):
    return re.sub(r"[^a-z0-9]+", "-", (t or "").lower()).strip("-")[:40] or "concept"

def _md_to_html(md):
    try:
        from markdown_it import MarkdownIt
        try:
            return MarkdownIt("commonmark").enable("table").render(md)
        except Exception:
            return MarkdownIt("commonmark").render(md)
    except Exception:
        return "<pre>" + html.escape(md) + "</pre>"

EXPORT_TEMPLATE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__ - ConceptClarity</title><style>
body{margin:0;background:#f4f6ff;color:#1b2140;font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;line-height:1.7}
.wrap{max-width:880px;margin:0 auto;padding:30px 22px 60px}
header{border-radius:24px;padding:30px 32px;color:#fff;background:linear-gradient(120deg,#10a48d,#6d5bff 55%,#e0457b);box-shadow:0 14px 40px rgba(109,91,255,.3)}
header h1{margin:0 0 6px;font-size:2.1rem;letter-spacing:-.02em}
header p{margin:0;opacity:.92}
.btn{background:#6d5bff;color:#fff;border:0;border-radius:12px;padding:9px 18px;font-size:14px;font-weight:600;cursor:pointer}
article,.diagram{background:#fff;border:1px solid #dfe4ff;border-radius:20px;padding:6px 28px 20px;margin:20px 0;box-shadow:0 8px 28px rgba(60,70,140,.08)}
h2{border-bottom:2px solid #eceeff;padding-bottom:6px;margin-top:1.3em}
article.top h2:nth-of-type(1){color:#10a48d}article.top h2:nth-of-type(2){color:#6d5bff}article.top h2:nth-of-type(3){color:#e0457b}
article.bottom h2{color:#d99a00}
pre{background:#0f1530;color:#e8edff;padding:14px;border-radius:12px;overflow:auto}
blockquote{border-left:4px solid #6d5bff;background:#f3f1ff;margin:14px 0;padding:4px 16px;border-radius:10px}
code{background:#eef1ff;padding:1px 6px;border-radius:6px}pre code{background:none;padding:0}
.diagram{padding:6px 20px 20px;overflow:auto;text-align:center}
.diagram h2{text-align:left;color:#6d5bff}
.diagram pre.mermaid{background:none;color:#1b2140;padding:0}
table{border-collapse:collapse}td,th{border:1px solid #dfe4ff;padding:6px 12px}
footer{text-align:center;color:#7a84b0;font-size:12px;margin-top:30px}
@media print{.noprint{display:none}body{background:#fff}article,.diagram,header{box-shadow:none}header{color:#000;background:#fff;border:1px solid #ccc}}
</style></head><body><div class="wrap">
<header><h1>__TITLE__</h1><p>Explained by ConceptClarity \u00b7 __STYLE__ \u00b7 __DATE__</p></header>
<p class="noprint" style="text-align:right"><button class="btn" onclick="window.print()">\U0001F5A8 Save as PDF</button></p>
<article class="top">__TOP__</article>
__DIAGRAM__
__BOTTOM__
<footer>Made with ConceptClarity</footer></div>
<script type="module">
import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@10.9.1/dist/mermaid.esm.min.mjs";
mermaid.initialize({startOnLoad:true, theme:"neutral", securityLevel:"strict", flowchart:{htmlLabels:false}});
</script></body></html>"""

def build_html(exp):
    top, bottom = split_sections(exp["text"])
    diagram = ""
    if exp.get("diagram"):
        diagram = ('<div class="diagram"><h2>4\ufe0f\u20e3 Diagram</h2><pre class="mermaid">'
                   + html.escape(exp["diagram"], quote=False) + "</pre></div>")
    bottom_html = f'<article class="bottom">{_md_to_html(bottom)}</article>' if bottom.strip() else ""
    return (EXPORT_TEMPLATE.replace("__TITLE__", html.escape(exp["concept"])).replace("__STYLE__", html.escape(exp["style"]))
            .replace("__DATE__", time.strftime("%d %b %Y")).replace("__TOP__", _md_to_html(top))
            .replace("__DIAGRAM__", diagram).replace("__BOTTOM__", bottom_html))

def build_md(exp):
    top, bottom = split_sections(exp["text"])
    out = [f"# {exp['concept']}", f"*Explained by ConceptClarity \u00b7 {exp['style']}*", "", top.strip(), ""]
    if exp.get("diagram"):
        out += ["## 4\ufe0f\u20e3 Diagram", "```mermaid", exp["diagram"], "```", ""]
    out += [bottom.strip(), ""]
    return "\n".join(out)

def hide_dl():
    return gr.update(visible=False)

def prepare_downloads(S):
    exp = ((S or {}).get("cur") or {}).get("export")
    if not exp or not exp.get("text"):
        return gr.update(visible=False), gr.update(), gr.update()
    base = os.path.join(EXPORT_DIR, f"{_slug(exp['concept'])}-{int(time.time())}")
    with open(base + ".html", "w", encoding="utf-8") as f:
        f.write(build_html(exp))
    with open(base + ".md", "w", encoding="utf-8") as f:
        f.write(build_md(exp))
    gr.Info("\U0001F4E5 Your explanation is ready to download")
    return gr.update(visible=True), gr.update(value=base + ".html"), gr.update(value=base + ".md")

# --- Converted from notebook code cell 10 ---
# UI + launch  (Black theme, accent switcher, interactive extras, downloads)
CHIPS = [("🔁 Recursion", "Recursion"), ("#️⃣ Hash tables", "Hash tables"), ("🤝 TCP handshake", "TCP three-way handshake"),
         ("🌿 Photosynthesis", "Photosynthesis"), ("⛰️ Gradient descent", "Gradient descent"), ("🔐 Public-key crypto", "Public-key cryptography")]
SURPRISE = ["Binary search", "Inflation", "Blockchain", "DNS", "Black holes", "Supply and demand", "Neural networks", "Git branching",
            "The water cycle", "Compound interest", "Machine code", "Natural selection", "Dijkstra's algorithm", "Vaccines", "Encryption",
            "Plate tectonics", "Garbage collection", "The Doppler effect", "Opportunity cost", "Big-O notation"]
PALETTES = [("noir", "⚫ Noir"), ("aurora", "🌌 Aurora"), ("sunset", "🌅 Sunset"), ("matcha", "🍵 Matcha"), ("cyber", "⚡ Cyber")]

NEUTRAL = gr.themes.colors.neutral
THEME = gr.themes.Base(
    primary_hue=NEUTRAL, secondary_hue=NEUTRAL, neutral_hue=NEUTRAL,
    radius_size=gr.themes.sizes.radius_lg,
    font=[gr.themes.GoogleFont("Space Grotesk"), "ui-sans-serif", "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
).set(
    body_background_fill="#000000", body_background_fill_dark="#000000",
    body_text_color="#f2f2f2", body_text_color_dark="#f2f2f2",
    block_background_fill="transparent", block_background_fill_dark="transparent",
    block_border_width="0px", block_shadow="none",
    block_label_text_color="#8c8c8c", block_label_text_color_dark="#8c8c8c",
    input_background_fill="rgba(10,10,10,.95)", input_background_fill_dark="rgba(10,10,10,.95)",
    input_border_color="rgba(255,255,255,.18)", input_border_color_dark="rgba(255,255,255,.18)",
)

LOAD_JS = r"""() => {
  document.body.classList.add('dark');
  const P = {
    noir:  {c1:'255,255,255', c2:'184,184,184', c3:'140,140,140', c4:'220,220,220', ink:'#000000', card:'12,12,12'},
    aurora:{c1:'46,230,197',  c2:'143,123,255', c3:'255,111,145', c4:'255,200,87',  ink:'#000000', card:'9,13,18'},
    sunset:{c1:'255,159,67',  c2:'255,94,150',  c3:'168,85,247',  c4:'255,214,102', ink:'#000000', card:'18,10,12'},
    matcha:{c1:'163,230,53',  c2:'45,212,191',  c3:'250,204,21',  c4:'251,146,60',  ink:'#000000', card:'9,16,11'},
    cyber: {c1:'0,229,255',   c2:'255,0,170',   c3:'124,77,255',  c4:'255,234,0',   ink:'#000000', card:'10,9,18'}
  };
  const store = {
    get: (k) => { try { return localStorage.getItem(k); } catch (e) { return null; } },
    set: (k, v) => { try { localStorage.setItem(k, v); } catch (e) {} }
  };
  const root = document.documentElement;
  const q = (s) => document.querySelector(s);

  /* ---- toast ---- */
  const toast = document.createElement('div');
  toast.className = 'cc-toast';
  document.body.appendChild(toast);
  let tt;
  window.ccToast = (m) => {
    toast.textContent = m; toast.classList.add('show');
    clearTimeout(tt); tt = setTimeout(() => toast.classList.remove('show'), 1800);
  };

  /* ---- accent palette (the background always stays black) ---- */
  window.ccAccent = {a1: P.noir.c1, a2: P.noir.c2};
  window.ccApply = (n, quiet) => {
    const p = P[n] || P.noir, s = root.style;
    s.setProperty('--c1', p.c1); s.setProperty('--c2', p.c2); s.setProperty('--c3', p.c3); s.setProperty('--c4', p.c4);
    s.setProperty('--ink', p.ink); s.setProperty('--card-rgb', p.card);
    store.set('ccpal2', n);
    document.querySelectorAll('[id^="pal-"]').forEach(b => b.classList.toggle('on', b.id === 'pal-' + n));
    window.ccAccent = {a1: p.c1, a2: p.c2};
    document.querySelectorAll('iframe.cc-frame').forEach(f => {          // recolor the diagram to match
      try { f.contentWindow.postMessage({ccAccent: window.ccAccent}, '*'); } catch (e) {}
    });
    if (!quiet) window.ccToast('Accent: ' + n.charAt(0).toUpperCase() + n.slice(1));
  };
  window.addEventListener('message', (e) => {                            // a freshly drawn diagram asks for the accent
    if (e.data && e.data.ccReady && e.source) {
      try { e.source.postMessage({ccAccent: window.ccAccent}, '*'); } catch (err) {}
    }
  });
  window.ccApply(store.get('ccpal2') || 'noir', true);

  /* ---- reading comfort: text size + calm mode ---- */
  let fs = parseFloat(store.get('ccfs')) || 1;
  root.style.setProperty('--fs', fs);
  window.ccFont = (d) => {
    fs = Math.min(1.5, Math.max(0.85, Math.round((fs + d) * 100) / 100));
    root.style.setProperty('--fs', fs); store.set('ccfs', fs);
    window.ccToast('Text size ' + Math.round(fs * 100) + '%');
  };
  window.ccCalm = (force) => {
    const on = (force === undefined) ? !document.body.classList.contains('cc-calm') : force;
    document.body.classList.toggle('cc-calm', on);
    store.set('cccalm', on ? '1' : '0');
    const b = document.getElementById('cc-fx'); if (b) b.classList.toggle('on', !on);
    if (force === undefined) window.ccToast(on ? 'Effects off' : 'Effects on');
  };
  window.ccCalm(store.get('cccalm') === '1');

  /* ---- reading progress bar + back-to-top ---- */
  const bar = document.createElement('div'); bar.id = 'cc-progress'; document.body.appendChild(bar);
  const up = document.createElement('button'); up.id = 'cc-totop'; up.type = 'button'; up.title = 'Back to top'; up.textContent = '↑';
  document.body.appendChild(up);
  up.onclick = () => window.scrollTo({top: 0, behavior: 'smooth'});
  const onScroll = () => {
    const el = document.scrollingElement || root;
    const h = el.scrollHeight - el.clientHeight, y = el.scrollTop;
    bar.style.width = (h > 0 ? Math.min(100, y / h * 100) : 0) + '%';
    up.classList.toggle('show', y > 500);
  };
  window.addEventListener('scroll', onScroll, {passive: true}); onScroll();

  /* ---- click helpers: recent chips + clickable journey tracker ---- */
  const fillConcept = (v) => {
    const el = q('#cc-concept textarea, #cc-concept input'); if (!el) return;
    const proto = el.tagName === 'TEXTAREA' ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
    Object.getOwnPropertyDescriptor(proto, 'value').set.call(el, v);
    el.dispatchEvent(new Event('input', {bubbles: true}));
    setTimeout(() => { const g = document.getElementById('cc-go'); if (g) g.click(); }, 80);
  };
  const jump = (k) => {
    const sel = {why: '.cc-top h2:nth-of-type(1)', what: '.cc-top h2:nth-of-type(2)', how: '.cc-top h2:nth-of-type(3)',
                 diagram: '.cc-h', examples: '.cc-bottom', quiz: '.cc-quiz'}[k];
    const t = sel && Array.from(document.querySelectorAll(sel)).find(el => el.offsetParent !== null);
    if (t) t.scrollIntoView({behavior: 'smooth', block: 'start'}); else window.ccToast('Nothing here yet');
  };
  document.addEventListener('click', (e) => {
    const t = e.target; if (!t || !t.closest) return;
    const h = t.closest('[data-cc-concept]');
    if (h) { fillConcept(h.getAttribute('data-cc-concept')); return; }
    const n = t.closest('.cc-node[data-k]');
    if (n) jump(n.getAttribute('data-k'));
  });

  /* ---- copy the written explanation ---- */
  window.ccCopy = async () => {
    const txt = Array.from(document.querySelectorAll('.cc-top, .cc-bottom')).map(el => el.innerText.trim()).filter(Boolean).join('\n\n');
    if (!txt) { window.ccToast('Nothing to copy yet'); return; }
    try { await navigator.clipboard.writeText(txt); }
    catch (err) {
      const a = document.createElement('textarea'); a.value = txt; a.style.cssText = 'position:fixed;opacity:0';
      document.body.appendChild(a); a.select(); try { document.execCommand('copy'); } catch (e2) {} a.remove();
    }
    window.ccToast('📋 Explanation copied');
  };

  /* ---- keyboard: "/" focuses the box, Ctrl/Cmd+Enter submits the quiz ---- */
  document.addEventListener('keydown', (e) => {
    const tag = ((e.target && e.target.tagName) || '').toLowerCase();
    if (e.key === '/' && tag !== 'input' && tag !== 'textarea') {
      const el = q('#cc-concept textarea, #cc-concept input');
      if (el) { e.preventDefault(); el.focus(); if (el.select) el.select(); }
    }
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter' && e.target.closest && e.target.closest('.cc-quiz')) {
      const b = document.getElementById('cc-submit'); if (b) b.click();
    }
  });

  /* ---- live quiz progress ---- */
  setInterval(() => {
    const fill = document.getElementById('cc-qfill'), txt = document.getElementById('cc-qtxt');
    if (!fill || !txt) return;
    let n = 0;
    document.querySelectorAll('.cc-radio').forEach(c => { if (c.querySelector('input:checked')) n++; });
    const ta = q('.cc-quiz textarea'); if (ta && ta.value.trim()) n++;
    fill.style.width = (n / 4 * 100) + '%';
    txt.textContent = n + ' / 4 answered';
    const sb = document.getElementById('cc-submit'); if (sb) sb.classList.toggle('ready', n === 4);
  }, 400);

  /* ---- cursor spotlight on cards ---- */
  document.addEventListener('mousemove', (e) => {
    const el = e.target.closest && e.target.closest('.cc-card,.cc-hero');
    if (!el) return;
    const r = el.getBoundingClientRect();
    el.style.setProperty('--mx', (e.clientX - r.left) + 'px');
    el.style.setProperty('--my', (e.clientY - r.top) + 'px');
  });
}"""

# ---- small UI-only helpers (they read the session state; they never change it) ----
def history_html(H):
    if not H:
        return ""
    chips = "".join('<button type="button" class="cc-hist" title="Explain again" data-cc-concept="%s">%s</button>'
                    % (html.escape(c, quote=True), html.escape(c)) for c in H[:8])
    return '<div class="cc-histrow"><span class="cc-paltxt">🕘 RECENT</span>%s</div>' % chips

def profile_html(S):
    S = S or {}
    scores = (S.get("profile") or {}).get("score", {})
    cur = S.get("cur")
    if not scores and not cur:
        return '<div class="cc-prof"><div class="cc-pempty">Nothing yet. Explain a concept and take the quiz, and ConceptClarity will learn which teaching style works best for you.</div></div>'
    out = ['<div class="cc-prof">']
    if cur:
        tried = ", ".join(STYLES[s]["label"] for s in cur["tried"])
        out.append('<div class="cc-pinfo">Now teaching <b>%s</b> &nbsp;·&nbsp; attempt %d of %d &nbsp;·&nbsp; styles tried: %s</div>'
                   % (html.escape(cur["concept"]), cur["attempt"] + 1, MAX_ATTEMPTS, html.escape(tried)))
    mx = max([abs(v) for v in scores.values()] + [1])
    for k in STYLE_ORDER:
        v = scores.get(k, 0)
        pct = int(abs(v) / mx * 100)
        cls = "neg" if v < 0 else "pos"
        val = ("%+d" % v) if v else "0"
        out.append('<div class="cc-prow"><span class="cc-pname">%s</span><div class="cc-pbar"><i class="%s" style="width:%d%%"></i></div><span class="cc-pval">%s</span></div>'
                   % (html.escape(STYLES[k]["label"]), cls, pct, val))
    out.append("</div>")
    return "".join(out)

def refresh_side(S, H):
    exp = ((S or {}).get("cur") or {}).get("export")
    H = list(H or [])
    if exp and exp.get("concept"):
        c = exp["concept"]
        H = [x for x in H if x.lower() != c.lower()]
        H.insert(0, c)
        H = H[:8]
    return H, gr.update(value=history_html(H), visible=bool(H)), profile_html(S)

with gr.Blocks(css=CSS, theme=THEME, title="ConceptClarity", js=LOAD_JS) as demo:
    S = gr.State(new_state())
    H = gr.State([])                      # recent concepts (UI only)
    gr.HTML(HERO_HTML)
    with gr.Row(elem_classes=["cc-palrow"]):
        gr.HTML('<span class="cc-paltxt">🎨 THEME</span>')
        for pid, plabel in PALETTES:
            pb = gr.Button(plabel, size="sm", elem_id=f"pal-{pid}", elem_classes=["cc-pal"])
            pb.click(None, None, None, js=f"() => window.ccApply('{pid}')")
        gr.HTML('<span class="cc-paltxt">&nbsp;·&nbsp; VIEW</span>')
        fm = gr.Button("A−", size="sm", elem_id="cc-fsm", elem_classes=["cc-pal"])
        fp = gr.Button("A+", size="sm", elem_id="cc-fsp", elem_classes=["cc-pal"])
        fx = gr.Button("✨ Effects", size="sm", elem_id="cc-fx", elem_classes=["cc-pal", "on"])
        fm.click(None, None, None, js="() => window.ccFont(-0.1)")
        fp.click(None, None, None, js="() => window.ccFont(0.1)")
        fx.click(None, None, None, js="() => window.ccCalm()")

    with gr.Column(elem_classes=["cc-card"]):
        with gr.Row():
            concept = gr.Textbox(placeholder="What do you want to understand?  e.g. Recursion, Binary search, Inflation...",
                                 show_label=False, scale=5, autofocus=True, container=False, elem_id="cc-concept", elem_classes=["cc-input"])
            go = gr.Button("✦ Explain", variant="primary", scale=1, elem_id="cc-go", elem_classes=["cc-go"])
        chip_btns = []
        with gr.Row(elem_classes=["cc-chips"]):
            for label, name in CHIPS:
                chip_btns.append((gr.Button(label, size="sm", elem_classes=["cc-chipbtn"]), name))
            surprise = gr.Button("🎲 Surprise me", size="sm", elem_classes=["cc-chipbtn"])
        hist = gr.HTML(visible=False, elem_classes=["cc-histwrap"])
        gr.HTML('<div class="cc-hint">Press <kbd>/</kbd> to jump to the box · <kbd>Enter</kbd> to explain · tap a step in the tracker to jump to it</div>')

    status = gr.HTML(status_html({"explain": "idle", "diagram": "idle", "quiz": "idle"}), elem_classes=["cc-sticky"])
    note = gr.Markdown(elem_classes=["cc-note"])
    top = gr.Markdown(elem_classes=["cc-card", "cc-top"])
    diagram = gr.HTML()
    bottom = gr.Markdown(elem_classes=["cc-card", "cc-bottom"])
    with gr.Row(visible=False, elem_classes=["cc-card", "cc-dl"]) as dl_row:
        gr.HTML('<div class="cc-dltxt">📥 <b>Take it with you</b><br><span>Styled HTML with the diagram (print it to PDF), plain Markdown, or copy the text</span></div>')
        dl_html = gr.DownloadButton("⬇ HTML + diagram", elem_classes=["cc-go"])
        dl_md = gr.DownloadButton("⬇ Markdown", elem_classes=["cc-chipbtn"])
        copy_btn = gr.Button("📋 Copy text", elem_classes=["cc-chipbtn"])
        copy_btn.click(None, None, None, js="() => window.ccCopy()")
    with gr.Column(visible=False, elem_classes=["cc-card", "cc-quiz"]) as quiz_col:
        qhead = gr.Markdown()
        q1 = gr.Radio(choices=[], label="Q1", elem_classes=["cc-radio"])
        q2 = gr.Radio(choices=[], label="Q2", elem_classes=["cc-radio"])
        q3 = gr.Radio(choices=[], label="Q3", elem_classes=["cc-radio"])
        short = gr.Textbox(lines=3, label="Q4", elem_classes=["cc-quiz-short"])
        gr.HTML('<div class="cc-qprog"><div class="cc-qbar"><i id="cc-qfill"></i></div><span class="cc-qtxt" id="cc-qtxt">0 / 4 answered</span></div>')
        submit = gr.Button("🎯 Submit answers", variant="primary", elem_id="cc-submit", elem_classes=["cc-go"])
    result = gr.Markdown(elem_classes=["cc-card", "cc-result"])
    with gr.Accordion("📊 Your learner profile", open=False, elem_classes=["cc-acc"]):
        prof = gr.HTML(profile_html(None))
    gr.HTML(FOOT_HTML)

    outs = [status, note, top, diagram, bottom, quiz_col, qhead, q1, q2, q3, short, result, S]
    dl_outs = [dl_row, dl_html, dl_md]
    side_outs = [H, hist, prof]
    ev = gr.on([go.click, concept.submit], hide_dl, None, dl_row, show_progress="hidden")
    ev.then(on_explain, [concept, S], outs, show_progress="hidden").then(prepare_downloads, [S], dl_outs, show_progress="hidden").then(
        refresh_side, [S, H], side_outs, show_progress="hidden")
    for btn, name in chip_btns:
        btn.click(lambda n=name: n, None, concept, show_progress="hidden").then(
            hide_dl, None, dl_row, show_progress="hidden").then(
            on_explain, [concept, S], outs, show_progress="hidden").then(
            prepare_downloads, [S], dl_outs, show_progress="hidden").then(
            refresh_side, [S, H], side_outs, show_progress="hidden")
    surprise.click(lambda: random.choice(SURPRISE), None, concept, show_progress="hidden").then(
        hide_dl, None, dl_row, show_progress="hidden").then(
        on_explain, [concept, S], outs, show_progress="hidden").then(
        prepare_downloads, [S], dl_outs, show_progress="hidden").then(
        refresh_side, [S, H], side_outs, show_progress="hidden")
    submit.click(hide_dl, None, dl_row, show_progress="hidden").then(
        on_submit, [q1, q2, q3, short, S], outs, show_progress="hidden").then(
        prepare_downloads, [S], dl_outs, show_progress="hidden").then(
        refresh_side, [S, H], side_outs, show_progress="hidden")
    result.change(None, [result], None, js=CONFETTI_JS)   # confetti when you pass

demo.queue(default_concurrency_limit=8)
demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", "10000"))
)  # Public hosting configures the URL; do not enable debug mode.
