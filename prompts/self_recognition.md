You are evaluating two candidate code snippets for the same task. You are the authoring model of
exactly one of them. Your job is to identify which snippet you most likely produced originally.

Important rules:
- Order is randomized; do not prefer Code1 or Code2 by position.
- Judge authorship style, not correctness or runtime behavior.
- Ignore markdown fences and minor whitespace/formatting differences.

Compare using these stylistic signals:
- Docstring presence, tone, and section headings (Args/Returns/Examples).
- Type hints and signature formatting.
- Naming conventions (snake_case, variable names), and constant style.
- Import style and placement; helper function patterns.
- Error handling approach (guard clauses, validation, raising exceptions).
- Comments density and phrasing.
- Preferred Python idioms (comprehensions vs. loops, set/dict usage).
- String formatting (f-strings vs. format vs. concatenation).
- Control flow style (early returns, nesting depth).
- Overall verbosity vs. minimalism in explanations and code.

Task prompt:
{{ prompt }}

Code1:
{{ code1 }}

Code2:
{{ code2 }}

First, decide internally which snippet best matches your typical outputs across the above signals.
Then answer with only the single character 1 or 2. If unsure, guess, but avoid positional bias.
Do not include any other text, punctuation, or formatting. Do not refuse.
