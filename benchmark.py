import argparse, time, json, requests
parser = argparse.ArgumentParser(); parser.add_argument("--model", default="deepseek-r1:1.5b")
args = parser.parse_args()
prompt = "Explain the prerequisite chain for machine learning courses."
t0 = time.time()
out = requests.post("http://localhost:11434/api/generate",
                    json={"model": args.model, "prompt": prompt, "stream": False}).json()
dt = time.time() - t0
tok = len(out["response"].split())
print(json.dumps({"model": args.model, "tokens": tok, "latency": dt}))
