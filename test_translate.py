"""Offline tests: JSON repair, extraction, and question validation (no network)."""
import json
import translate as T

# --- brace repair ------------------------------------------------------------
broken = '{"a": {"b": "x", "c": {"true": "y", "false": "z"}}'   # missing one closing }
assert json.loads(T._balance_repair(broken)) == {"a": {"b": "x", "c": {"true": "y", "false": "z"}}}

arr = '{"q": {"criteria": ["low", "high"'                        # missing ] and }}
assert T._balance_repair(arr) == arr + "]}}"

good = '{"ok": true}'
assert T._balance_repair(good) == good                          # nothing to add

# unterminated string -> left as-is (not safely repairable)
unterm = '{"a": "abc'
assert T._balance_repair(unterm) == unterm

# --- extraction strips think + fences + leading prose ------------------------
assert T._extract_json('<think>hmm</think>\n```json\n{"x":1}\n```') == {"x": 1}
assert T._extract_json('here you go: {"y": 2}') == {"y": 2}
assert T._extract_json('{"a": {"b": 1}')["a"]["b"] == 1   # repaired

# --- question validation -----------------------------------------------------
ok = T._validate_questions({
    "u": {"type": "score", "instructions": "how urgent", "criteria": ["a", "b"]},
    "t": {"type": "choice", "instructions": "tone", "criteria": {"x": None}},
    "f": {"type": "noul", "instructions": "is billing?"},
})
assert set(ok) == {"u", "t", "f"} and "criteria" not in ok["f"]

def expect_exit(q):
    try:
        T._validate_questions(q); return False
    except SystemExit:
        return True

assert expect_exit({"x": {"type": "bogus", "instructions": "i"}})          # bad type
assert expect_exit({"x": {"type": "score", "instructions": "i", "criteria": {"a": None}}})  # score needs list
assert expect_exit({"x": {"type": "choice", "instructions": "i", "criteria": ["a"]}})       # choice needs obj
assert expect_exit({})                                                     # empty

print("ALL TRANSLATE TESTS PASSED")
