"""
eval_relevance.py  (v3 – hard constraint on plan list)

目标：
  强化 prompt，使 full 模式的 Plan 精度 ≥ noGrades。
实现：
  • plan 前加 '### Valid courses:' 固定标题
  • 指令：Only recommend courses that appear after the line ### Valid courses:
  • 要求给 8 门、不重复
"""

import re, csv, time, difflib, pathlib, random, statistics, requests, json
from course_manager import CourseManager

BOOT_ITER = 1000
LOW_GRADE_THRESHOLD = "B-"
STREAM_MODEL = True

# ---------- CONFIG ----------
TEST_STUDENT = "miy@kean.edu"
QUESTION_FILE = "evaluation_questions.txt"
OUT_CSV = "relevance_scores.csv"
MODEL_NAME = "deepseek-r1:1.5b"
OLLAMA_URL = "http://127.0.0.1:11434"
# ----------------------------

mgr = CourseManager()
student = mgr.get_student_by_username(TEST_STUDENT)
plan_courses = {l.strip() for l in open(f"{student.major}_plan.txt", encoding="utf-8") if l.strip()}
all_courses = [l.strip() for l in open("course_list.txt", encoding="utf-8") if l.strip()]
all_codes = {c.split(":")[0].strip(): c for c in all_courses}
taken_courses = set(mgr.get_student_courses(TEST_STUDENT).keys())
course_grades = mgr.get_student_courses(TEST_STUDENT)
grade_rank = {g: i for i, g in enumerate(["A", "A-", "B+", "B", "B-", "C+", "C", "D", "F"])}

def grade_low(c): g = course_grades.get(c); return g and grade_rank[g] >= grade_rank[LOW_GRADE_THRESHOLD]

def ask_ai(prompt):
    st = time.time()
    r = requests.post(f"{OLLAMA_URL}/api/generate",
                      json={"model": MODEL_NAME, "prompt": prompt, "stream": STREAM_MODEL},
                      timeout=600)
    if STREAM_MODEL:
        parts = [json.loads(l.decode())["response"] for l in r.iter_lines() if l]
        txt = "".join(parts)
    else:
        txt = r.json().get("response", "")
    return txt.strip(), time.time() - st

def extract_courses(txt):
    found = {full for code, full in all_codes.items()
             if re.search(rf"\b{re.escape(code)}\b", txt, re.I)}
    if not found:
        for full in all_courses:
            if difflib.SequenceMatcher(None, txt.lower(), full.lower()).ratio() > .8:
                found.add(full)
    return list(dict.fromkeys(found))      # 去重保序

# --- prompt builder ---
plan_block = "### Valid courses:\n" + "\n".join(sorted(plan_courses))
suffix = ("\n\nOnly recommend courses that appear after the line '### Valid courses:'."
          "\nList **8 unique courses**, one per line, full name as in the list.")

def build_prompt(mode, q):
    hist = "\n".join(f"{c} - {g or 'Not assigned'}" for c, g in course_grades.items())
    if mode == "full":
        return (f'Student question: "{q}"\nMy course history:\n{hist}\n{plan_block}\n'
                "Based on ALL information, recommend courses." + suffix)
    if mode == "noGrades":
        return (f'"{q}"\n{plan_block}\nBased on the plan, recommend courses.' + suffix)
    if mode == "noPlan":
        return (f'"{q}"\nMy course history:\n{hist}\nBased on history only, recommend courses.' + suffix)
    return q + suffix

# --- evaluation ---
rows = []
for q in (l.strip() for l in open(QUESTION_FILE, encoding="utf-8") if l.strip()):
    for mode in ("full", "noGrades", "noPlan", "question"):
        rep, lat = ask_ai(build_prompt(mode, q))
        recs = extract_courses(rep)
        good_plan = {c for c in recs if c in plan_courses and c not in taken_courses}
        good_personal = {c for c in recs if c in plan_courses and (c not in taken_courses or grade_low(c))}
        plan_score = len(good_plan) / len(recs) if recs else 0
        pers_score = len(good_personal) / len(recs) if recs else 0
        lift = pers_score - plan_score
        recall = len(good_plan) / len(plan_courses - taken_courses) if plan_courses - taken_courses else 0
        rows.append([q, mode, len(recs), plan_score, pers_score, lift, recall, f"{lat:.2f}s"])
        print(f"[{mode}] {q[:38]}… Rec:{len(recs):2d} Plan:{plan_score:.3f} Pers:{pers_score:.3f}")

# --- save ---
with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerow(["Question", "Mode", "#Rec", "PlanScore", "PersonalScore", "Lift", "Recall", "Latency"])
    csv.writer(f).writerows(rows)

# --- summarize ---
grp = {}
for *_q, m, _r, ps, prs, lf, rc, _ in rows:
    grp.setdefault(m, []).append((ps, prs, lf, rc))

def ci(vals):
    boots = [statistics.mean(random.choices(vals, k=len(vals))) for _ in range(BOOT_ITER)]
    return statistics.quantiles(boots, n=20)[1:19:17]

print("\n=== Aggregate metrics (95% CI) ===")
for m, v in grp.items():
    pl = [x[0] for x in v]; pe = [x[1] for x in v]; li = [x[2] for x in v]; rc = [x[3] for x in v]
    mp, (pl_lo, pl_hi) = statistics.mean(pl), ci(pl)
    ms, (pe_lo, pe_hi) = statistics.mean(pe), ci(pe)
    ml, (li_lo, li_hi) = statistics.mean(li), ci(li)
    mr = statistics.mean(rc)
    print(f"{m:9} Plan {mp:.3f} [{pl_lo:.3f},{pl_hi:.3f}]  "
          f"Personal {ms:.3f} [{pe_lo:.3f},{pe_hi:.3f}]  "
          f"Lift {ml:+.3f} [{li_lo:+.3f},{li_hi:+.3f}]  Recall {mr:.3f}")
