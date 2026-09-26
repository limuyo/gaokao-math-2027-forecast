# -*- coding: utf-8 -*-
"""
2027 高考数学选择压轴题 · 重型预测引擎
数据：core_questions.json（全国Ⅰ卷/新高考Ⅰ卷选择压轴，2007理 + 2017-2026 共 21 题）
输出：engine_output.json
方法：频率衰减 / 马尔可夫(1+2阶) / 生存分析hazard / 文本风格指纹 / 命题人分段
     + 滚动回测(expanding window) + 网格穷举调参 + 10000次bootstrap置信区间 + BMA融合
     + 答案选项指纹分析（单选字母 / 多选组合 / 错项倾向）
仅用 Python 标准库；随机种子固定，结果可复现。
"""
import json, math, random, itertools
from collections import Counter, defaultdict

random.seed(2027)
TARGET = 2027
CATS = ["立体几何", "函数与导数", "解析几何", "概率统计", "三角与解三角形", "数列"]
LETTERS = ["A", "B", "C", "D"]

corpus = json.load(open("core_questions.json", encoding="utf-8"))
QS = corpus["questions"]

def build_slot(slot):
    m, stems = defaultdict(set), defaultdict(str)
    for q in QS:
        if q["slot"] != slot:
            continue
        m[q["year"]].add(q["l1"])
        stems[q["year"]] += q["stem"]
    return {y: sorted(v) for y, v in sorted(m.items())}, dict(stems)

SLOT, STEMS = {}, {}
for s in ("single", "multi"):
    SLOT[s], STEMS[s] = build_slot(s)

def norm(d):
    t = sum(d.values()) or 1.0
    return {c: d.get(c, 0.0) / t for c in CATS}

def targets_of(slot):
    years = sorted(SLOT[slot])
    return [y for y in years if len([x for x in years if x < y]) >= 2]

# ---------------- 模型池 ----------------
def m_freq(train, target, p, slot):
    DECAY = [1, 0.7, 0.5, 0.35, 0.25, 0.18, 0.12, 0.08, 0.05]
    sc = {}
    for c in CATS:
        ys = [y for y in train if c in train[y]]
        if not ys:
            sc[c] = p["filler"]; continue
        base = sum(p.get("topW", 3) if (target - y) <= p["window"]
                   else DECAY[min(target - y - p["window"] - 1, len(DECAY) - 1)] for y in ys)
        gap = target - max(ys)
        s = base * (1 + p["gapBoost"] * gap)
        if gap == 1:
            s *= p["repPenalty"]
        sc[c] = s
    return norm(sc)

def m_markov(train, target, p, slot):
    """1阶全权 + 2阶半权转移，拉普拉斯平滑"""
    years = sorted(train)
    cnt = {a: {b: p["alpha"] for b in CATS} for a in CATS}
    tot = {a: p["alpha"] * len(CATS) for a in CATS}
    for y in years:
        if y >= target:
            break
        for lag, w in ((1, 1.0), (2, 0.5)):
            nxt = y + lag
            if nxt in train and nxt < target:
                for a in train[y]:
                    for b in train[nxt]:
                        cnt[a][b] += w
                        tot[a] += w
    last = train[max(y for y in years if y < target)]
    sc = {c: sum(cnt[a][c] / tot[a] for a in last) / len(last) for c in CATS}
    return norm(sc)

def m_hazard(train, target, p, slot):
    """考点"回归间隔"生存分布：距上次出现 g 年时，按历史回归间隔分布打分"""
    gaps = Counter()
    for c in CATS:
        ys = sorted(y for y in train if c in train[y])
        for a, b in zip(ys, ys[1:]):
            if b - a <= 8:
                gaps[b - a] += 1
    tot = sum(gaps.values())
    sc = {}
    for c in CATS:
        ys = [y for y in train if c in train[y]]
        if not ys:
            sc[c] = p["alpha"] / (tot + p["alpha"] * 6 + 1)
            continue
        g = target - max(ys)
        sc[c] = (gaps.get(g, 0) + p["alpha"]) / (tot + p["alpha"] * 8) + 0.02
    return norm(sc)

def _bigrams(s):
    s = "".join(ch for ch in s if not ch.isspace())
    return Counter(s[i:i + 2] for i in range(len(s) - 1))

def m_fingerprint(train, target, p, slot):
    """风格指纹：目标前两年题面的字符 bigram TF-IDF 向量，与历史各年题面算余弦，
    相似度加权该年考点（捕捉同一命题人的语言风格）"""
    win = "".join(STEMS[slot][y] for y in (target - 2, target - 1) if y in STEMS[slot])
    if not win:
        return norm({c: 1 for c in CATS})
    wv = _bigrams(win)
    wn = math.sqrt(sum(v * v for v in wv.values())) or 1
    sc = defaultdict(float)
    for y in train:
        if target - y > 12:
            continue
        tv = _bigrams(STEMS[slot][y])
        tn = math.sqrt(sum(v * v for v in tv.values())) or 1
        cos = sum(wv.get(k, 0) * v for k, v in tv.items()) / (wn * tn)
        for c in train[y]:
            sc[c] += cos
    for c in CATS:
        sc.setdefault(c, 0.02)
    return norm(sc)

def m_era(train, target, p, slot):
    """命题人分段：当前命题人窗（目标前 window 年）内纯频数 + 拉普拉斯"""
    lo = target - p["window"]
    cnt = Counter(c for y in train if y >= lo for c in train[y])
    n = sum(cnt.values())
    return norm({c: (cnt.get(c, 0) + 1) / (n + len(CATS)) for c in CATS})

MODELS = [
    ("频率衰减", m_freq, {"window": [1, 2], "gapBoost": [0.1, 0.2, 0.3],
                          "repPenalty": [0.5, 0.65, 0.8], "filler": [0.05, 0.1, 0.2]}),
    ("马尔可夫", m_markov, {"alpha": [0.3, 0.8, 1.5]}),
    ("hazard", m_hazard, {"alpha": [0.5, 1.0, 2.0]}),
    ("风格指纹", m_fingerprint, {}),
    ("命题人分段", m_era, {"window": [2, 3]}),
]
NAMES = [m[0] for m in MODELS]
FN = {name: fn for name, fn, _ in MODELS}

def grid_params():
    per_model = []
    for _, _, grid in MODELS:
        keys = sorted(grid)
        if not keys:
            per_model.append([{}])
        else:
            per_model.append([dict(zip(keys, v))
                              for v in itertools.product(*[grid[k] for k in keys])])
    return [dict(zip(NAMES, combo)) for combo in itertools.product(*per_model)]

def bern_ll(prob, truth):
    ll = 0.0
    for c in CATS:
        p = min(max(prob[c], 1e-3), 1 - 1e-3)
        ll += math.log(p) if c in truth else math.log(1 - p)
    return -ll / len(CATS)

def brier(prob, truth):
    return sum((prob[c] - (1 if c in truth else 0)) ** 2 for c in CATS) / len(CATS)

def rank_top(prob):
    return sorted(CATS, key=lambda c: -prob[c])

# ---------------- 滚动回测缓存 ----------------
CACHE = {}
for slot in ("single", "multi"):
    for t in targets_of(slot):
        train = {y: v for y, v in SLOT[slot].items() if y < t}
        truth = set(SLOT[slot][t])
        d = {}
        for name, fn, grid in MODELS:
            keys = sorted(grid)
            plist = [{}] if not keys else [dict(zip(keys, v))
                                           for v in itertools.product(*[grid[k] for k in keys])]
            for pp in plist:
                pr = fn(train, t, pp, slot)
                rk = rank_top(pr)
                d.setdefault(name, []).append({
                    "params": pp, "prob": pr, "ll": bern_ll(pr, truth), "brier": brier(pr, truth),
                    "top1": rk[0] in truth, "top3": any(x in truth for x in rk[:3])})
        CACHE[(slot, t)] = d

TARGETS = sorted(CACHE.keys(), key=lambda k: (k[0], k[1]))

def row_entries(pmap):
    rows = {}
    for k in TARGETS:
        rows[k] = {n: next(e for e in CACHE[k][n] if e["params"] == pmap[n]) for n in NAMES}
    return rows

def fused_scores(pmap, tau):
    rows = row_entries(pmap)
    mean_ll = {n: sum(rows[k][n]["ll"] for k in TARGETS) / len(TARGETS) for n in NAMES}
    w = {n: math.exp(-mean_ll[n] / tau) for n in NAMES}
    tw = sum(w.values())
    w = {n: v / tw for n, v in w.items()}
    out = []
    for k in TARGETS:
        slot, t = k
        prob = norm({c: sum(w[n] * rows[k][n]["prob"][c] for n in NAMES) for c in CATS})
        truth = set(SLOT[slot][t])
        rk = rank_top(prob)
        out.append({"slot": slot, "year": t, "truth": SLOT[slot][t],
                    "top3": [{"cat": c, "p": round(prob[c], 4)} for c in rk[:3]],
                    "hit1": rk[0] in truth, "hit3": any(x in truth for x in rk[:3]),
                    "brier": brier(prob, truth), "ll": bern_ll(prob, truth)})
    return {"w": w, "rows": out,
            "brier": sum(r["brier"] for r in out) / len(out),
            "logloss": sum(r["ll"] for r in out) / len(out),
            "top1": sum(r["hit1"] for r in out) / len(out),
            "top3": sum(r["hit3"] for r in out) / len(out)}

print("网格穷举调参（%d 组参数 × 4 档 τ × %d 目标年）..." % (len(grid_params()), len(TARGETS)))
BEST = None
for pmap in grid_params():
    for tau in (0.05, 0.1, 0.2, 0.4):
        r = fused_scores(pmap, tau)
        key = (round(r["brier"], 6), round(r["logloss"], 6), round(r["top1"], 6))
        if BEST is None or key < BEST[0]:
            BEST = (key, pmap, tau, r)
KEY, PMAP, TAU, SC = BEST
print("最优参数：", json.dumps(PMAP, ensure_ascii=False), "tau =", TAU)
print("融合回测：Brier %.4f  LogLoss %.4f  Top1 %.2f  Top3 %.2f（%d 个目标年）"
      % (SC["brier"], SC["logloss"], SC["top1"], SC["top3"], len(SC["rows"])))

# ---------------- 模型成绩单（最优参数下） ----------------
rows_best = row_entries(PMAP)
model_board = {}
for n in NAMES:
    lls = [rows_best[k][n]["ll"] for k in TARGETS]
    top1 = [rows_best[k][n]["top1"] for k in TARGETS]
    top3 = [rows_best[k][n]["top3"] for k in TARGETS]
    br = [rows_best[k][n]["brier"] for k in TARGETS]
    model_board[n] = {"top1": round(sum(top1) / len(top1), 3), "top3": round(sum(top3) / len(top3), 3),
                      "brier": round(sum(br) / len(br), 4), "logloss": round(sum(lls) / len(lls), 4)}

# ---------------- 2027 最终分布 ----------------
def weights_from(sample_keys):
    mean_ll = {n: sum(rows_best[k][n]["ll"] for k in sample_keys) / len(sample_keys) for n in NAMES}
    w = {n: math.exp(-mean_ll[n] / TAU) for n in NAMES}
    tw = sum(w.values())
    return {n: v / tw for n, v in w.items()}

W0 = weights_from(TARGETS)
finals = {}
for slot in ("single", "multi"):
    train = {y: v for y, v in SLOT[slot].items() if y < TARGET}
    finals[slot] = norm({c: sum(W0[n] * FN[n](train, TARGET, PMAP[n], slot)[c]
                                for n in NAMES) for c in CATS})

# ---------------- 情景：命题人连续性视角（era+频率各半） ----------------
scenario_era = {}
for slot in ("single", "multi"):
    train = {y: v for y, v in SLOT[slot].items() if y < TARGET}
    era_p = m_era(train, TARGET, {"window": 2}, slot)
    freq_p = m_freq(train, TARGET, PMAP["频率衰减"], slot)
    scenario_era[slot] = norm({c: 0.5 * era_p[c] + 0.5 * freq_p[c] for c in CATS})

# ---------------- bootstrap 10000 ----------------
BOOT = 10000
boot_final = {s: [] for s in ("single", "multi")}
bt1, bt3 = [], []
rows_flat = SC["rows"]
for _ in range(BOOT):
    ks = [random.choice(TARGETS) for _ in range(len(TARGETS))]
    w = weights_from(ks)
    for slot in ("single", "multi"):
        train = {y: v for y, v in SLOT[slot].items() if y < TARGET}
        p = norm({c: sum(w[n] * FN[n](train, TARGET, PMAP[n], slot)[c] for n in NAMES) for c in CATS})
        boot_final[slot].append(p)
    smp = [random.choice(rows_flat) for _ in range(len(rows_flat))]
    bt1.append(sum(r["hit1"] for r in smp) / len(smp))
    bt3.append(sum(r["hit3"] for r in smp) / len(smp))

def ci(vs):
    vs = sorted(vs)
    return round(vs[int(0.025 * len(vs))], 3), round(vs[int(0.975 * len(vs)) - 1], 3)

final_out = {slot: {c: {"p": round(finals[slot][c], 4),
                        "lo": sorted(b[c] for b in boot_final[slot])[int(0.025 * BOOT)],
                        "hi": sorted(b[c] for b in boot_final[slot])[int(0.975 * BOOT) - 1]}
                    for c in CATS}
             for slot in ("single", "multi")}
t1lo, t1hi = ci(bt1)
t3lo, t3hi = ci(bt3)

# ---------------- 单选×多选联合 ----------------
joint = {f"{s}|{m}": finals["single"][s] * finals["multi"][m] * (0.15 if s == m else 1.0)
         for s in CATS for m in CATS}
jt = sum(joint.values())
joint = {k: v / jt for k, v in joint.items()}
joint_top = sorted(({"s": k.split("|")[0], "m": k.split("|")[1], "p": round(v, 4)}
                    for k, v in joint.items()), key=lambda x: -x["p"])[:6]

# ---------------- 答案选项指纹 ----------------
single_qs = [q for q in QS if q["slot"] == "single"]
multi_qs = [q for q in QS if q["slot"] == "multi"]
sl = Counter(q["answer"] for q in single_qs)
combos = [q["answer"] for q in multi_qs]
sizes = Counter(len(c) for c in combos)
ml = Counter(L for c in combos for L in set(c))
wrong = Counter(L for q in multi_qs for L in LETTERS if L not in q["answer"])
wrong_only = Counter("".join(sorted(set(LETTERS) - set(q["answer"]))) for q in multi_qs)
combo_hist = Counter("".join(sorted(c)) for c in combos)

def lap(k, n, alpha, K):
    return round((k + alpha) / (n + alpha * K), 3)

answers_out = {
    "single_letters": {L: {"n": sl.get(L, 0), "p": lap(sl.get(L, 0), len(single_qs), 1, 4)} for L in LETTERS},
    "multi_letters": {L: {"n": ml.get(L, 0), "p": lap(ml.get(L, 0), len(multi_qs), 1, 2)} for L in LETTERS},
    "multi_sizes": {str(k): lap(sizes.get(k, 0), len(multi_qs), 1, 3) for k in (1, 2, 3)},
    "wrong_letter": {L: wrong.get(L, 0) for L in LETTERS},
    "wrong_only_hist": dict(wrong_only),
    "combo_hist": dict(combo_hist),
    "likely_combos": [{"combo": c, "n": n, "p": lap(n, len(multi_qs), 0.5, 15)}
                      for c, n in combo_hist.most_common()],
}

# ---------------- 答案组合预测器（2027 多选） ----------------
ALL_COMBOS = []
for _k in (1, 2, 3, 4):
    for _c in itertools.combinations(LETTERS, _k):
        ALL_COMBOS.append("".join(_c))
sizes_cnt = Counter(len(c) for c in combos)
letter_p = {L: (ml.get(L, 0) + 1) / (len(multi_qs) + 2) for L in LETTERS}
AB_PENALTY = 0.5   # 外部先验（用户提供）：多选一般不选 AB；可调整或置 1 关闭

def sym_diff(a, b):
    return len(set(a) ^ set(b))

def combo_raw(freq_counter):
    raw = {}
    for S in ALL_COMBOS:
        k = len(S)
        size_p = (sizes_cnt.get(k, 0) + 0.5) / (len(multi_qs) + 2)
        base = (freq_counter.get(S, 0) + 0.3) / (len(multi_qs) + 15 * 0.3)
        mix = 1.0
        for L in LETTERS:
            mix *= letter_p[L] if L in S else (1 - letter_p[L])
        kernel = sum(math.exp(-sym_diff(S, a) / 1.0) * (1.0 if i == 0 else 0.5)
                     for i, a in enumerate([combos[-1], combos[-2]]))   # 2026 BCD / 2025 ABC
        ext = AB_PENALTY if S == "AB" else 1.0
        raw[S] = size_p * base * mix * kernel * ext
    t = sum(raw.values())
    return {S: v / t for S, v in raw.items()}

combo_dist = combo_raw(combo_hist)
boot_combo = {S: [] for S in ALL_COMBOS}
for _ in range(BOOT):
    smp = Counter(random.choice(combos) for _ in range(len(combos)))
    rr = combo_raw(smp)
    for S in ALL_COMBOS:
        boot_combo[S].append(rr[S])
combo_pred = {
    "model": "频率(Laplace) × 大小先验 × 字母基率 × 编辑距离核(锚:2026 BCD,2025 ABC) × 外部先验(不选AB ×0.5)",
    "top": [{"combo": S, "p": round(combo_dist[S], 4),
             "lo": sorted(boot_combo[S])[int(0.025 * BOOT)],
             "hi": sorted(boot_combo[S])[int(0.975 * BOOT) - 1]}
            for S in sorted(ALL_COMBOS, key=lambda s: -combo_dist[s])[:8]],
    "full": {S: round(combo_dist[S], 4) for S in ALL_COMBOS},
    "extern_prior": "多选一般不选 AB（用户提供的外部统计）",
}

# ---------------- 输出 ----------------
out = {
    "generated": "2026-09-13",
    "target": TARGET,
    "corpus": {"core": len(QS),
               "slots": {s: SLOT[s] for s in SLOT},
               "missing": corpus["missing"]},
    "params": {"chosen": PMAP, "tau": TAU, "grid_combos": len(grid_params()) * 4},
    "weights": {n: round(v, 4) for n, v in W0.items()},
    "models": model_board,
    "backtest": {"rows": SC["rows"],
                 "summary": {k: round(SC[k], 4) for k in ("brier", "logloss", "top1", "top3")},
                 "top1_ci95": [t1lo, t1hi], "top3_ci95": [t3lo, t3hi],
                 "n_targets": len(TARGETS)},
    "final": final_out,
    "scenario_era": {s: {c: round(v, 4) for c, v in scenario_era[s].items()} for s in scenario_era},
    "joint": {"grid": {k: round(v, 4) for k, v in joint.items()}, "top": joint_top},
    "answers": answers_out,
    "combo_pred": combo_pred,
    "honesty": {"ceiling": "单选 Top1 现实上限约 50-60%，Top3 约 85-90%",
                "note": "瓶颈为数据量（每年 2-3 点）与命题人意图不可观测；引擎优化概率校准，不承诺押中原题",
                "uniform_brier": 0.1389, "uniform_top1": 0.1667},
}
json.dump(out, open("engine_output.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("engine_output.json 已生成")
print("模型成绩单：")
for n, b in model_board.items():
    print("  %-6s Top1 %.2f Top3 %.2f Brier %.4f LL %.4f" % (n, b["top1"], b["top3"], b["brier"], b["logloss"]))
print("权重：", {n: round(v, 3) for n, v in W0.items()})
for r in SC["rows"][:14]:
    print("  %s %d 真实=%s 预测Top3=%s %s%s" % (r["slot"][:1], r["year"], ",".join(r["truth"]),
          ",".join(x["cat"][:2] for x in r["top3"]), "✓1" if r["hit1"] else "✗1", " ✓3" if r["hit3"] else " ✗3"))
for slot in ("single", "multi"):
    print(slot, {c: "%.3f [%.2f,%.2f]" % (v["p"], v["lo"], v["hi"]) for c, v in final_out[slot].items()})
print("联合 Top3：", [(j["s"], j["m"], j["p"]) for j in joint_top[:3]])
print("2027 多选组合预测 Top5：", [(x["combo"], x["p"]) for x in combo_pred["top"][:5]])
