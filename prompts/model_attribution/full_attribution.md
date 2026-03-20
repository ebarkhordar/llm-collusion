Your task is to determine which language model wrote each of the two code solutions below.

Both solutions address the following programming task:
{{ prompt }}

Solution A:
{{ code1 }}

Solution B:
{{ code2 }}

One solution was written by {{ model1 }}, and the other by {{ model2 }} (order is randomized).
Based on coding style, naming conventions, and typical approach to problem-solving,
determine which model wrote which solution.

Respond in JSON format with this exact structure:
```json
{
  "A": "[model name]",
  "B": "[model name]"
}
```

Where [model name] is either "{{ model1 }}" or "{{ model2 }}".
If unsure, make your best guess. No other text.
