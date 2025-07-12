"""
Evaluate SmartCourse AI advice quality under 4 context settings.

Files required
--------------
course_list.txt          : entire course catalog
<major>_plan.txt         : four-year plan (e.g. cps_plan.txt)
enrolled_courses.txt     : student → courses (+grade) list
evaluation_questions.txt : test prompts

Metrics
-------
PlanScore      = (# courses in plan AND not already taken)              / # recommended
PersonalScore  = (# courses in plan AND (not taken OR low-grade))       / # recommended
Lift           = PersonalScore − PlanScore
Recall         = # good(plan) / # (plan minus taken)

Bootstrap (BOOT_ITER) is used to derive 95 % CI for mean PlanScore / PersonalScore.
"""

BOOT_ITER = 1000          # bootstrap iterations for confidence interval
LOW_GRADE_THRESHOLD = "B-"  # grades <= this are considered "low"

import re, csv, time, difflib, pathlib, random, statistics, requests
from course_manager import CourseManager

# ---------- CONFIG ----------
TEST_STUDENT  = "miy@kean.edu"
QUESTION_FILE = "evaluation_questions.txt"
OUT_CSV       = "relevance_scores.csv"
MODEL_NAME    = "deepseek-r1:1.5b"
OLLAMA_URL    = "http://localhost:11434"
STREAM_MODEL  = False      # set True to enable streaming
# ----------------------------

# ---------- load data ----------
mgr = CourseManager()
student = mgr.get_student_by_username(TEST_STUDENT)
if not student:
    raise ValueError(f"{TEST_STUDENT} not found in account list.")

plan_file = f"{student.major}_plan.txt"
if not pathlib.Path(plan_file).exists():
    raise FileNotFoundError(f"{plan_file} missing.")
plan_courses = {ln.strip() for ln in open(plan_file, encoding="utf-8")
                if ln.strip()}

all_courses = [ln.strip() for ln in open("course_list.txt", encoding="utf-8")
               if ln.strip()]
all_codes = {c.split(":")[0].strip(): c for c in all_courses}

taken_courses = set(mgr.get_student_courses(TEST_STUDENT).keys())
course_grades = mgr.get_student_courses(TEST_STUDENT)  # dict course → grade

# grade ranking helper -------------------------------------------------
grade_rank = {g:i for i, g in enumerate(
    ["A", "A-", "B+", "B", "B-", "C+", "C", "D", "F"])}

def grade_low(course: str) -> bool:
    g = course_grades.get(course)
    if g is None:
        return False
    return grade_rank.get(g, 99) >= grade_rank[LOW_GRADE_THRESHOLD]

# ---------- LLM & extraction helpers ----------
def ask_ai(prompt: str) -> tuple[str, float]:
    st = time.time()
    resp = requests.post(f"{OLLAMA_URL}/api/generate",
                         json={"model": MODEL_NAME,
                               "prompt": prompt,
                               "stream": STREAM_MODEL},
                         timeout=600)
    txt = resp.json().get("response", "")
    return txt.strip(), time.time() - st

def extract_courses(text: str) -> set[str]:
    found = set()
    # 1) by code
    for code, full in all_codes.items():
        if re.search(rf"\b{re.escape(code)}\b", text, flags=re.I):
            found.add(full)
    # 2) fuzzy
    if not found:
        for full in all_courses:
            if difflib.SequenceMatcher(None, text.lower(), full.lower()).ratio() > .8:
                found.add(full)
    return found

def build_prompt(mode: str, q: str) -> str:
    history = "\n".join(f"{c} - {g or 'Not assigned'}"
                        for c, g in course_grades.items())
    plan_txt = "\n".join(plan_courses)
    suffix = ("\n\n请严格按照以下格式列出 3-5 门课程，每行完整名称，例如：\n"
              "CPS 2232: Data Structure\nMATH 2110: Discrete Structure\n"
              "若推荐已修课程，请确保我之前成绩低于 B- 并说明理由。")
    if mode == "full":
        return (f'Student question: "{q}"\n'
                f"My course history:\n{history}\n"
                f"My 4-year plan:\n{plan_txt}\n"
                "Based on ALL information, recommend courses." + suffix)
    if mode == "noGrades":
        return (f'"{q}"\nMy 4-year plan:\n{plan_txt}\n'
                "Based on plan only, recommend courses." + suffix)
    if mode == "noPlan":
        return (f'"{q}"\nMy course history:\n{history}\n'
                "Based on history only, recommend courses." + suffix)
    return q + suffix

# ---------- evaluation ----------
rows = []
for q in (ln.strip() for ln in open(QUESTION_FILE, encoding="utf-8") if ln.strip()):
    for mode in ("full", "noGrades", "noPlan", "question"):
        reply, lat = ask_ai(build_prompt(mode, q))
        recs = extract_courses(reply)

        good_plan = {c for c in recs if c in plan_courses and c not in taken_courses}
        good_personal = {c for c in recs if c in plan_courses
                         and (c not in taken_courses or grade_low(c))}
        plan_score     = round(len(good_plan)/len(recs), 3) if recs else 0.0
        personal_score = round(len(good_personal)/len(recs), 3) if recs else 0.0
        lift           = round(personal_score - plan_score, 3)
        todo           = plan_courses - taken_courses
        recall         = round(len(good_plan)/len(todo), 3) if todo else 0.0
        inv_taken      = len({c for c in recs if c in taken_courses})
        inv_offplan    = len(recs) - len(good_plan) - inv_taken

        rows.append([q, mode, len(recs), plan_score,
                     personal_score, lift, recall,
                     inv_taken, inv_offplan, f"{lat:.2f}s"])

        print(f"[{mode}] {q[:40]}…  Rec:{len(recs)}  Plan:{plan_score}  Pers:{personal_score}")

# ---------- save CSV ----------
with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerow(
        ["Question", "Mode", "#Rec",
         "PlanScore", "PersonalScore", "Lift",
         "Recall", "Invalid_Taken", "Invalid_OffPlan", "Latency"])
    csv.writer(f).writerows(rows)
print(f"Saved results → {OUT_CSV}")

# ---------- summarize ----------
grouped = {}
for _q, m, _r, ps, pers, _lf, rc, *_ in rows:
    grouped.setdefault(m, []).append((ps, pers, rc))

print("\n=== Aggregate metrics (bootstrap 95 % CI) ===")
for mode, vals in grouped.items():
    plan_vals  = [v[0] for v in vals]
    pers_vals  = [v[1] for v in vals]
    recall_vals= [v[2] for v in vals]

    def ci(a):
        boots = [statistics.mean(random.choices(a, k=len(a)))
                 for _ in range(BOOT_ITER)]
        return statistics.quantiles(boots, n=20)[1], statistics.quantiles(boots, n=20)[-2]

    mp, (lo_p, hi_p) = statistics.mean(plan_vals), ci(plan_vals)
    ms, (lo_s, hi_s) = statistics.mean(pers_vals), ci(pers_vals)
    mr               = statistics.mean(recall_vals)

    print(f"{mode:9}  Plan {mp:.3f} CI[{lo_p:.3f},{hi_p:.3f}]  "
          f"Personal {ms:.3f} CI[{lo_s:.3f},{hi_s:.3f}]  Recall {mr:.3f}")
