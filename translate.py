#!/usr/bin/env python3
"""
typesafe-translate — escribe algo normal y lo transforma a las variables de TypeSafe.

Un LLM local (flota Ollama: qwen3.8:27b @ corsair) compila tu texto en un request
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
LLM_MODEL = os.environ.get("TS_TRANSLATE_LLM_MODEL", "qwen3.8:27b")
TS_URL = os.environ.get("TS_TRANSLATE_TS_URL", "https://api.typesafe.ai/v1/systemone")
TS_MODEL = os.environ.get("TS_TRANSLATE_TS_MODEL", "jev-latest")

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


def compile_request(text: str, *, state_override: str | None = None,
                    llm_url: str = LLM_URL, llm_model: str = LLM_MODEL,
                    ts_model: str = TS_MODEL) -> dict:
    """Natural-language text -> normalized TypeSafe request {model, state, questions}."""
    user = text if state_override is None else f"{text}\n\n<content to evaluate>\n{state_override}"
    payload = {
        "model": llm_model, "think": False, "stream": False, "format": "json",
        "options": {"temperature": 0, "num_ctx": 8192, "num_predict": 2048},
        "messages": [
            {"role": "system", "content": COMPILER_SYSTEM},
            {"role": "user", "content": FEWSHOT_USER},
            {"role": "assistant", "content": FEWSHOT_ASSISTANT},
            {"role": "user", "content": user},
        ],
    }
    try:
        resp = _post_json(llm_url, payload)
    except (urllib.error.URLError, TimeoutError) as e:
        raise SystemExit(f"[error] LLM no accesible en {llm_url}: {e}\n"
                         f"        ¿corsair despierto? prueba otro --llm-url/--llm-model.")
    raw = resp.get("message", {}).get("content", "")
    try:
        compiled = _extract_json(raw)
    except json.JSONDecodeError:
        raise SystemExit(f"[error] el LLM no devolvió JSON válido. Salida:\n{raw[:500]}")

    state = state_override if state_override is not None else compiled.get("state")
    questions = _validate_questions(compiled.get("questions"))
    return {"model": ts_model, "state": state, "questions": questions}


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
    p.add_argument("--llm-url", default=LLM_URL)
    p.add_argument("--llm-model", default=LLM_MODEL)
    p.add_argument("--ts-model", default=TS_MODEL, help="modelo TypeSafe (default jev-latest)")
    args = p.parse_args(argv)

    state_override = args.content
    if state_override is None and not sys.stdin.isatty():
        piped = sys.stdin.read().strip()
        state_override = piped or None

    request = compile_request(args.text, state_override=state_override,
                              llm_url=args.llm_url, llm_model=args.llm_model,
                              ts_model=args.ts_model)

    if args.json:
        print(json.dumps(request, ensure_ascii=False, indent=2))
        return
    if args.show:
        print(json.dumps(request, ensure_ascii=False, indent=2), file=sys.stderr)

    resp = run_typesafe(request)
    print(format_answers(resp))


if __name__ == "__main__":
    main()
