from src.data_pipeline.format_chat import build_messages, number_diff

DIFF = (
    "--- a/app.py\n"
    "+++ b/app.py\n"
    "@@ -1,4 +1,5 @@\n"
    " import os\n"
    "-import sys\n"
    "+import json\n"
    "+import re\n"
    " \n"
    " def f():\n"
    "@@ -10,3 +11,3 @@ def f():\n"
    "     x = 1\n"
    "-    return x\n"
    "\\ No newline at end of file\n"
    "+    return x + 1\n"
    "     pass\n"
)


def test_number_diff():
    assert number_diff(DIFF) == (
        "--- a/app.py\n"
        "+++ b/app.py\n"
        "@@ -1,4 +1,5 @@\n"
        "1|  import os\n"
        "   | -import sys\n"
        "2| +import json\n"
        "3| +import re\n"
        "4|  \n"
        "5|  def f():\n"
        "@@ -10,3 +11,3 @@ def f():\n"
        "11|      x = 1\n"
        "   | -    return x\n"
        "\\ No newline at end of file\n"
        "12| +    return x + 1\n"
        "13|      pass"
    )


def test_build_messages():
    msgs = build_messages("app.py", DIFF)
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert msgs[1]["content"].startswith("File: app.py\n<git_diff>\n--- a/app.py")
    assert msgs[1]["content"].endswith("13|      pass\n</git_diff>")
