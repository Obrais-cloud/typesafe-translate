#!/usr/bin/env python3
"""
typesafe-translate — escribe algo normal y lo transforma a las variables de TypeSafe.

Un LLM local (flota Ollama: Nemotron 3.5 Super @ corsair) compila tu texto en un request
TypeSafe válido {state, questions{...}} y, por defecto, lo ejecuta contra la API.

Uso:
    export TYPESAFE_API_KEY=...
    typesafe-translate "dime si este ticket es urgente (1-5) y su tono (calmado/enfadado): 'me cobraron dos veces!!'"
    echo "<texto a evaluar>" | typesafe-translate "clasifica el tono calmado/enfadado y si es de facturación"
    typesafe-translate "..." --json          # solo genera el esquema, no ejecuta

Motor y API son configurables por env/flags (ver --help).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request

# --- defaults (flota) --------------------------------------------------------
LLM_URL = os.environ.get("TS_TRANSLATE_LLM_URL", "http://100.94.117.48:11434/api/chat")  # corsair
LLM_MODEL = os.environ.get("TS_TRANSLATE_LLM_MODEL", "nemotron-3.5-super:120b-a12b-q4_K_M")
TS_URL = os.environ.get("TS_TRANSLATE_TS_URL", "https://api.typesafe.ai/v1/systemone")
TS_MODEL = (os.environ.get("TS_TRANSLATE_TS_MODEL") or "jev-1.13.0")

# ollawake (mac mini) — best-effort wake of a sleeping host; ignored if unreachable.
OLLAWAKE = os.environ.get("TS_TRANSLATE_OLLAWAKE", "http://100.70.244.85:11436")

# Fallback chain: tried in order until one host answers with a compilable request.
# Override wholesale with TS_TRANSLATE_LLM_CHAIN (JSON list of {label,url,model,wake?}).
DEFAULT_CHAIN = [
    # 2026-10-16: Corsair solo tiene Nemotron 3.5 Super (sustituye a qwen3.8 y qwen3.6)
    {"label": "corsair",    "url": "http://100.94.117.48:11434/api/chat", "model": "nemotron-3.5-super:120b-a12b-q4_K_M", "wake": "corsair"},
    {"label": "mac-studio", "url": "http://100.68.94.14:11434/api/chat",  "model": "ornith-1.5:35b"},
    {"label": "alien18",    "url": "http://100.87.2.47:11434/api/chat",   "model": "qwen3:14b"},
]

VALID_TYPES = {"noul", "choice", "score"}

COMPILER_SYSTEM = """\
You compile a user's natural-language request into a TypeSafe "System One" request.
TypeSafe does NOT generate text: it answers small typed judgments over a `state`.

Output ONLY a JSON object with exactly two keys:
  "state":     the content to be evaluated. If the user's text embeds the thing to
               judge (an email, a ticket, a message), put that verbatim here. If the
               user only describes the judgments and gives no content, use null.
  "questions": an object mapping a short snake_case id -> a question object.

Each question object has:
  "type":         one of "noul", "choice", "score".
  "instructions": a clear sentence describing the single judgment to make.
  "criteria":     depends on type (see below).

Choose the type by meaning:
  - "noul"   = a yes/no condition. Probability of yes. criteria may be omitted, or
               {"true": "...", "false": "..."} describing each side. Use one noul per
               label when several could apply at once.
  - "choice" = pick exactly one option from an unordered set. criteria is an object
               {"option_label": null, ...}; give a short description string instead of
               null only if the label alone is ambiguous. Add an "other"/"none" option
               when nothing may fit.
  - "score"  = a position on an ORDERED scale (urgency, severity, satisfaction...).
               criteria is an ARRAY of ordered level descriptions, lowest first, each
               concrete and independently understandable (e.g. "Broken but workaround
               exists"), NOT bare words like "medium".

Rules: one narrow judgment per question. Split independent dimensions into separate
questions. Do not invent judgments the user did not ask for. Keep ids short.

Return ONLY the JSON object. No prose, no markdown fences.
"""

FEWSHOT_USER = ('clasifica el tono (calmado/frustrado/enfadado) y dime si es urgente de baja a alta: '
                '"me habéis cobrado dos veces y nadie contesta!!"')
FEWSHOT_ASSISTANT = json.dumps({
    "state": "me habéis cobrado dos veces y nadie contesta!!",
    "questions": {
        "tono": {"type": "choice", "instructions": "¿Cuál es el tono del cliente?",
                 "criteria": {"calmado": None, "frustrado": None, "enfadado": None}},
        "urgencia": {"type": "score", "instructions": "¿Cómo de urgente es el mensaje?",
                     "criteria": ["No urgente: consulta general sin impacto.",
                                  "Media: problema molesto pero operativo.",
                                  "Alta: bloqueo grave o pérdida activa de dinero/datos."]},
    },
}, ensure_ascii=False)


# --- LLM call ----------------------------------------------------------------
def _post_json(url: str, payload: dict, timeout: float = 120.0) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _balance_repair(s: str) -> str:
    """Close any unclosed { [ (LLMs sometimes drop a trailing brace). Ignores strings."""
    stack, in_str, esc = [], False, False
    for ch in s:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack:
                stack.pop()
    if in_str:  # unterminated string — can't safely repair
        return s
    return s + "".join("}" if c == "{" else "]" for c in reversed(stack))


def _extract_json(text: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    start = text.find("{")
    if start > 0:
        text = text[start:]
    for candidate in (text, _balance_repair(text)):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise json.JSONDecodeError("no valid JSON after repair", text, 0)


def _reachable(chat_url: str) -> bool:
    """Fast liveness check on the host (short timeout) before a heavy generate call."""
    base = chat_url.rsplit("/api/", 1)[0]
    try:
        urllib.request.urlopen(base + "/api/tags", timeout=4).read()
        return True
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def _try_wake(host: str) -> None:
    """Best-effort: ask ollawake to wake/ensure a host. Silent if ollawake is down."""
    try:
        req = urllib.request.Request(f"{OLLAWAKE}/ensure/{host}", data=b"", method="POST")
        urllib.request.urlopen(req, timeout=10).read()
    except Exception:
        pass


def _compile_with(url: str, model: str, messages: list, state_override, ts_model: str) -> dict:
    """One host attempt. Raises URLError/Timeout (network) or JSONDecodeError/SystemExit (bad output)."""
    if not _reachable(url):
        raise urllib.error.URLError("host unreachable")
    payload = {
        "model": model, "think": False, "stream": False, "format": "json",
        "options": {"temperature": 0, "num_predict": 2048, **({} if "nemotron" in model else {"num_ctx": 8192})},  # Nemotron: no cambiar ctx (recargaría 90 GB)
        "messages": messages,
    }
    resp = _post_json(url, payload, timeout=300.0 if "nemotron" in model else 90.0)
    raw = resp.get("message", {}).get("content", "")
    compiled = _extract_json(raw)                       # JSONDecodeError -> try next host
    state = state_override if state_override is not None else compiled.get("state")
    questions = _validate_questions(compiled.get("questions"))
    return {"model": ts_model, "state": state, "questions": questions}


def compile_request(text: str, *, state_override: str | None = None,
                    candidates: list | None = None, ts_model: str = TS_MODEL,
                    verbose: bool = True) -> dict:
    """Natural-language text -> normalized TypeSafe request, with fleet fallback.

    Tries each candidate host in order; on network failure of a wake-capable host it
    asks ollawake to wake it and retries once, then falls through to the next host.
    """
    user = text if state_override is None else f"{text}\n\n<content to evaluate>\n{state_override}"
    messages = [
        {"role": "system", "content": COMPILER_SYSTEM},
        {"role": "user", "content": FEWSHOT_USER},
        {"role": "assistant", "content": FEWSHOT_ASSISTANT},
        {"role": "user", "content": user},
    ]
    chain = candidates if candidates is not None else _load_chain()
    errors: list[str] = []
    waked = False
    i = 0
    while i < len(chain):
        c = chain[i]
        label = c.get("label", c["url"])
        try:
            req = _compile_with(c["url"], c["model"], messages, state_override, ts_model)
            if verbose and i > 0:
                print(f"[llm] fallback → {label} ({c['model']})", file=sys.stderr)
            return req
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, SystemExit) as e:
            errors.append(f"{label}/{c['model']}: {type(e).__name__}")
            is_net = isinstance(e, (urllib.error.URLError, TimeoutError))
            if is_net and c.get("wake") and not waked:
                waked = True
                if verbose:
                    print(f"[llm] {label} no responde; despertando vía ollawake…", file=sys.stderr)
                _try_wake(c["wake"])
                continue  # retry this same candidate once
            i += 1
    raise SystemExit("[error] ningún host de la flota respondió/compiló:\n  "
                     + "\n  ".join(errors)
                     + "\n  Ajusta la cadena (TS_TRANSLATE_LLM_CHAIN) o usa --llm-url/--llm-model.")


def _load_chain() -> list:
    env = os.environ.get("TS_TRANSLATE_LLM_CHAIN")
    if env:
        try:
            return json.loads(env)
        except json.JSONDecodeError:
            raise SystemExit("[error] TS_TRANSLATE_LLM_CHAIN no es JSON válido.")
    return DEFAULT_CHAIN


def _validate_questions(q) -> dict:
    if not isinstance(q, dict) or not q:
        raise SystemExit("[error] el LLM no produjo 'questions'. Reformula el texto.")
    out: dict = {}
    for qid, spec in q.items():
        if not isinstance(spec, dict):
            raise SystemExit(f"[error] pregunta '{qid}' mal formada.")
        t = spec.get("type")
        if t not in VALID_TYPES:
            raise SystemExit(f"[error] pregunta '{qid}': type '{t}' inválido (noul/choice/score).")
        instr = spec.get("instructions")
        if not instr:
            raise SystemExit(f"[error] pregunta '{qid}': faltan 'instructions'.")
        crit = spec.get("criteria")
        if t == "score" and not isinstance(crit, list):
            raise SystemExit(f"[error] pregunta '{qid}' (score): 'criteria' debe ser lista ordenada.")
        if t == "choice" and not isinstance(crit, dict):
            raise SystemExit(f"[error] pregunta '{qid}' (choice): 'criteria' debe ser objeto de opciones.")
        item = {"type": t, "instructions": instr}
        if crit is not None:
            item["criteria"] = crit
        out[str(qid)] = item
    return out


# --- TypeSafe execution ------------------------------------------------------
def run_typesafe(request: dict) -> dict:
    key = os.environ.get("TYPESAFE_API_KEY")
    if not key:
        raise SystemExit("[error] falta TYPESAFE_API_KEY en el entorno.")
    if request.get("state") is None:
        raise SystemExit("[error] no hay 'state' que evaluar. Incluye el contenido en el "
                         "texto, pásalo por stdin, o usa --content.")
    req = urllib.request.Request(
        TS_URL, data=json.dumps(request).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise SystemExit(f"[error] TypeSafe {e.code}: {e.read().decode()[:300]}")


def format_answers(resp: dict) -> str:
    lines = [f"modelo: {resp.get('model')}   tokens_in: {resp.get('usage',{}).get('input_tokens')}"]
    for qid, a in resp.get("answers", {}).items():
        t = a.get("type")
        if t == "noul":
            lines.append(f"  • {qid:16} [noul]   {'sí' if a['noul']>=0.5 else 'no'}  ({a['noul']:.2f})")
        elif t == "choice":
            dist = " ".join(f"{k}={v:.2f}" for k, v in sorted(a["probabilities"].items(),
                                                              key=lambda kv: -kv[1]))
            lines.append(f"  • {qid:16} [choice] {a['choice']}  (conf {a['confidence']:.2f})  [{dist}]")
        elif t == "score":
            top = max(a["probabilities"], key=a["probabilities"].get)
            lines.append(f"  • {qid:16} [score]  {a['score']:.2f}/{len(a['legend'])-1} "
                         f"→ {a['legend'][top]}  (conf {a['confidence']:.2f})")
    return "\n".join(lines)


# --- CLI ---------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(description="Traduce texto normal → variables de TypeSafe y las ejecuta.")
    p.add_argument("text", help="lo que quieres juzgar, en lenguaje normal")
    p.add_argument("-c", "--content", help="contenido a evaluar (si no, se toma de stdin o del propio texto)")
    p.add_argument("--json", action="store_true", help="solo imprime el request generado (no ejecuta)")
    p.add_argument("--show", action="store_true", help="imprime también el request generado al ejecutar")
    p.add_argument("--llm-url", help="forzar un host LLM concreto (desactiva el fallback)")
    p.add_argument("--llm-model", help="modelo LLM para --llm-url (o el primero de la cadena)")
    p.add_argument("--ts-model", default=TS_MODEL, help="modelo TypeSafe (default jev-1.13.0, fijado)")
    p.add_argument("--chain", action="store_true", help="imprime la cadena de fallback efectiva y sale")
    args = p.parse_args(argv)

    if args.chain:
        print(json.dumps(_load_chain(), ensure_ascii=False, indent=2))
        return

    # Explicit host override -> single candidate, no fallback. Else the fleet chain.
    candidates = None
    if args.llm_url or args.llm_model:
        candidates = [{"label": "custom",
                       "url": args.llm_url or DEFAULT_CHAIN[0]["url"],
                       "model": args.llm_model or DEFAULT_CHAIN[0]["model"]}]

    state_override = args.content
    if state_override is None and not sys.stdin.isatty():
        piped = sys.stdin.read().strip()
        state_override = piped or None

    request = compile_request(args.text, state_override=state_override,
                              candidates=candidates, ts_model=args.ts_model)

    if args.json:
        print(json.dumps(request, ensure_ascii=False, indent=2))
        return
    if args.show:
        print(json.dumps(request, ensure_ascii=False, indent=2), file=sys.stderr)

    resp = run_typesafe(request)
    print(format_answers(resp))


if __name__ == "__main__":
    main()
