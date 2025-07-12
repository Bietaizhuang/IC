"""
Evaluate SmartCourse AI advice quality under 4 context settings
using automatic precision-like metric.

Files needed:
  - course_list.txt           : full course catalog
  - <major>_plan.txt          : four-year plan (e.g. cps_plan.txt)
  - enrolled_courses.txt      : (username, course) lines
  - evaluation_questions.txt  : test queries

Metric:
  Score = (# recommended courses that are in plan AND not already taken)
          / (# total recommended courses)
"""
BOOT_ITER = 1000 # bootstrap iterations for 95% CI

import re, csv, time, difflib, json, requests, pathlib
from course_manager import CourseManager

# ------------- CONFIG -------------
TEST_STUDENT  = "miy@kean.edu"
QUESTION_FILE = "evaluation_questions.txt"
OUT_CSV       = "relevance_scores.csv"
MODEL_NAME    = "deepseek-r1:1.5b"          # 评测用的模型
OLLAMA_URL    = "http://localhost:11434"  # Ollama 端口
# ----------------------------------

## ---------- load data ----------
mgr = CourseManager()
student = mgr.get_student_by_username(TEST_STUDENT)
if not student:
    raise ValueError(f"Student {TEST_STUDENT} not found in account list.")

# ——— four-year plan
plan_file = f"{student.major}_plan.txt"
if not pathlib.Path(plan_file).exists():
    raise FileNotFoundError(f"{plan_file} missing. Upload the plan file.")
plan_courses = set()
with open(plan_file, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if ":" in line and re.match(r"[A-Z]{2,4}\s*\d{3,4}", line):
            plan_courses.add(line)

# ——— full course catalog
all_courses = []
with open("course_list.txt", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            all_courses.append(line)
all_codes = {c.split(":")[0].strip(): c for c in all_courses}  # code → full

# ——— courses already taken
taken_courses = set(mgr.get_student_courses(TEST_STUDENT).keys())

## ---------- helper ----------
def ask_ai(prompt: str) -> tuple[str, float]:
    start = time.time()
    resp = requests.post(f"{OLLAMA_URL}/api/generate",
                         json={"model": MODEL_NAME,
                               "prompt": prompt,
                               "stream": False},
                         timeout=600)
    txt = resp.json().get("response", "")
    latency = time.time() - start
    return txt.strip(), latency

def extract_courses(text: str) -> set[str]:
    found = set()

    # 1) 精确匹配课程代码
    for code, full in all_codes.items():
        if re.search(rf"\b{re.escape(code)}\b", text, flags=re.I):
            found.add(full)

    # 2) 若仍为空 → 使用模糊匹配（≥0.8 相似度）
    if not found:
        for full in all_courses:
            if difflib.SequenceMatcher(None,
                                       text.lower(), full.lower()).ratio() > 0.8:
                found.add(full)
    return found

def summarize(rows):
    """计算均值、bootstrap 95% CI 并打印。"""
    import random, statistics
    grouped = {}
    for q, mode, rec, good, score, recall, *_ in rows:
        grouped.setdefault(mode, []).append((score, recall))
    print("\n=== Aggregate metrics ===")
    for mode, values in grouped.items():
        scores  = [v[0] for v in values]
        recalls = [v[1] for v in values]
        mean_s  = statistics.mean(scores)
        mean_r  = statistics.mean(recalls)
        # bootstrap
        boot_s = []
        for _ in range(BOOT_ITER):
            samp = random.choices(scores, k=len(scores))
            boot_s.append(statistics.mean(samp))
        ci_low, ci_high = (statistics.quantiles(boot_s, n=20)[1],
                           statistics.quantiles(boot_s, n=20)[-2])  # ≈95 % CI
        print(f"{mode:9}  Precision {mean_s:.3f} 95%CI[{ci_low:.3f},{ci_high:.3f}]  "
              f"Recall {mean_r:.3f}")

def build_prompt(mode: str, q: str) -> str:
    course_info = "\n".join(f"{c} - {g or 'Not assigned'}"
                            for c, g in mgr.get_student_courses(TEST_STUDENT).items())
    plan_text = "\n".join(plan_courses)
    suffix = ("\n\n请严格按照以下格式给出 3-5 门课程，每行一个完整课程名称，例如：\n"
              "CPS 2232: Data Structure\nMATH 2110: Discrete Structure\n")
    if mode == "full":
        return (f'Student question: "{q}"\n'
                f"My course history:\n{course_info}\n"
                f"My 4-year plan:\n{plan_text}\n"
                "Based on ALL information, recommend courses." + suffix)
    if mode == "noGrades":
        return (f'"{q}"\nMy 4-year plan:\n{plan_text}\n'
                "Based on plan, recommend courses." + suffix)
    if mode == "noPlan":
        return (f'"{q}"\nMy course history:\n{course_info}\n'
                "Based on history, recommend courses." + suffix)
    return q + suffix

## ---------- run evaluation ----------
rows = []
with open(QUESTION_FILE, encoding="utf-8") as f:
    questions = [l.strip() for l in f if l.strip()]

for q in questions:
    for mode in ("full", "noGrades", "noPlan", "question"):
        prompt = build_prompt(mode, q)
        reply, lat = ask_ai(prompt)
        recs = extract_courses(reply)
        good = {c for c in recs if c in plan_courses and c not in taken_courses}
        score = round(len(good)/len(recs), 3) if recs else 0.0
        todo = plan_courses - taken_courses
        recall = round(len(good) / len(todo), 3) if todo else 0.0
        invalid_taken = len({c for c in recs if c in taken_courses})
        invalid_offplan = len(recs) - len(good) - invalid_taken
        rows.append([q, mode, len(recs), len(good), score,
                     recall, invalid_taken, invalid_offplan, f"{lat:.2f}s"])

        print(f"[{mode}] {q[:40]}…  Rec:{len(recs)}  Good:{len(good)}  Score:{score}")

# save
with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerow(
        ["Question", "Mode", "#Rec", "#Good", "Score",
         "Recall", "Invalid_Taken", "Invalid_OffPlan", "Latency"]
    )
    csv.writer(f).writerows(rows)

print(f"Saved results → {OUT_CSV}")
summarize(rows)
