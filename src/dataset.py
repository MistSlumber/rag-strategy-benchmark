# -*- coding: utf-8 -*-
"""自包含的数据加载：不依赖其它仓库，保证本仓库可独立 clone 运行。"""
import json
import re


def load_sections(path):
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    sections = []
    for part in re.split(r"(?m)^##\s+", text)[1:]:
        lines = part.strip().splitlines()
        if not lines:
            continue
        match = re.match(r"^(\S+)\s+(.*)$", lines[0].strip())
        sid, title = (match.group(1), match.group(2)) if match else (lines[0], lines[0])
        body = "\n".join(lines[1:]).strip()
        if body:
            sections.append({"id": sid, "title": title or sid, "text": body})
    return sections


def load_questions(path):
    questions = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions
