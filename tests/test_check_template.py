from src.evaluation.check_template import compare, prompt_part, render

CHATML = ("{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"
          "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")
MSGS = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]
TRAINED = '<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nU<|im_end|>\n<|im_start|>assistant\n{"pass": true}<|im_end|>\n'


def test_matching_template_passes():
    assert compare(prompt_part(TRAINED), render(CHATML, MSGS)) == ""


def test_drifted_template_is_caught():
    drifted = CHATML.replace("{{ m.content }}", "{{ m.content }}\n")  # one extra newline per turn
    assert compare(prompt_part(TRAINED), render(drifted, MSGS)) != ""
