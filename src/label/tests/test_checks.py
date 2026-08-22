#!/usr/bin/env python3
# coding: utf-8
"""八类检查的回归。**用真实发生过的坏数据当夹具，不调模型、不需要视频。**

    python3 src/label/tests/test_checks.py

为什么这一套必须存在
--------------------
`validate.py` 是整个 benchmark 的**守卫**：七个题型的真值全部源自它守着的那批
标注，「标注错一处，题就错一片，而且错得不报错」。

而在这个文件之前，**守卫自己没有守卫** —— `src/label/tests/` 里只有
`test_core_replay.py`，它测的是 `core.describe()` 的词表重放，
既不碰 `validate` 也不碰 `serve`。于是：

- 改判据没有回归拦着
- 而 4.3 已经发生了：`serve.py` 抄了一份只有三类的判据，
  同时四处文档写着「共用同一份判据」

**在没有守卫的情况下重构守卫本身，是把风险叠起来。** 所以这一步排在
「判据收敛」（4.3 / 4.4）之前 —— 先立回归，再动刀。

夹具从哪来
----------
**`data/label/*/segments.before_*` 与 `segments.polluted` 不只是留档，
它们是现成的回归夹具** —— 每一份都是一次真实数据事故的现场，
而每一条检查当初就是为了抓那次事故才加的。

    污染 / 引用 / 重叠   stack_cubes/segments.polluted        P-03 出题产物回写
    可疑                pen_inbox/segments.before_zerolen_fix 标注连按两次 K
    序列                wash/segments.before_seqfix           两处标错物体 / 漏标
    歧义                wash/segments.before_merge026         同集重复 subtask
    引用（v1 形状）      */segments.before_subtask_id          存 narration 而非 id

只有 `派生` 用合成夹具 —— 它的历史证据没有单独留档。

**`覆盖` 这一条测不了，而且是数据的原因不是偷懒。** 它只在「一个视频装多集」时
才会触发（`len(eps) > 1`），而唯一那样的族 tea2 已经移出（视角错位，D-42）——
现存六族的「打包视频」列全是 0。造合成夹具也没意义：那要伪造 `meta/episodes`
的 parquet，测出来的是伪造得像不像，不是判据对不对。
**tea2 回来的时候要记得补上这一条。**

**两个方向都要测。** 只测「会报」的话，一个永远报错的校验器也能全过；
只测「不报」的话，一个永远沉默的校验器也能全过。
所以每条都测：**在对应夹具上必须报，在当前数据上必须不报。**
"""

from __future__ import annotations

import collections
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src" / "label"))
sys.path.insert(0, str(ROOT / "src"))

import validate as V  # noqa: E402

REAL = ROOT / "data" / "label"


def run_on(family: str, segments: list[Path]) -> collections.Counter:
    """把给定的一批 segments 文件当成 `family` 跑一遍检查，返回各类发现的条数。

    只替换 `validate.LABEL`；`RAW` 与 `episode_bounds` 仍指向真实数据 ——
    「覆盖」「歧义」需要真实的 episode 边界，拿假的测等于没测。
    """
    tmp = Path(tempfile.mkdtemp())
    try:
        fam = tmp / family
        (fam / "segments").mkdir(parents=True)
        shutil.copy(REAL / family / "subtasks.json", fam / "subtasks.json")
        for path in segments:
            shutil.copy(path, fam / "segments" / path.name)
        old, V.LABEL = V.LABEL, tmp
        try:
            report = V.Report()
            V.check_family(family, report)
        finally:
            V.LABEL = old
        return collections.Counter(f.kind for f in report.findings)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def archived(family: str, name: str) -> list[Path]:
    return sorted((REAL / family / name).glob("*_segments.json"))


def synth(family: str, mutate: Any) -> list[Path]:
    """拿当前数据的一份文档改一处，造合成夹具。写进临时目录，调用方负责清理。"""
    src = sorted((REAL / family / "segments").glob("*_segments.json"))[0]
    doc = json.loads(src.read_text(encoding="utf-8"))
    mutate(doc)
    out = Path(tempfile.mkdtemp()) / src.name
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    return [out]


# ── 合成夹具的三个改法 ────────────────────────────────────────────────
def _break_derived(doc: dict[str, Any]) -> None:
    """把 start 改成与帧号不符 —— 上游 end=(f+1)/fps 那个隐含语义踩错时的样子。"""
    doc["segments"][0]["start"] = round(doc["segments"][0]["start"] + 1.5, 3)


def _break_overlap(doc: dict[str, Any]) -> None:
    """让两段真重叠（**走帧号**，不走秒 —— 秒区间本来就会重叠一帧，全量 631 处）。"""
    a, b = doc["segments"][0], doc["segments"][1]
    b["start_frame"] = a["end_frame"] - 5


def _break_reference(doc: dict[str, Any]) -> None:
    doc["segments"][0]["subtask"] = "no_such_subtask_id"


CASES: list[tuple[str, str, Any]] = [
    # (要验的检查, 说明, 取夹具的方法)
    ("污染", "stack_cubes/segments.polluted —— P-03 出题产物回写",
     lambda: ("stack_cubes", archived("stack_cubes", "segments.polluted"))),
    ("重叠", "同上（那批数据同时带着帧重叠）",
     lambda: ("stack_cubes", archived("stack_cubes", "segments.polluted"))),
    ("可疑", "pen_inbox/segments.before_zerolen_fix —— 标注连按两次 K",
     lambda: ("pen_inbox", archived("pen_inbox", "segments.before_zerolen_fix"))),
    ("序列", "wash/segments.before_seqfix —— 标错物体 / 漏标一段",
     lambda: ("wash", archived("wash", "segments.before_seqfix"))),
    ("歧义", "wash/segments.before_merge026 —— 同集重复 subtask",
     lambda: ("wash", archived("wash", "segments.before_merge026"))),
    ("引用", "v1 形状：段里存 narration 而非 subtask id",
     lambda: ("wash", archived("wash", "segments.before_subtask_id"))),
    ("引用", "合成：subtask 指向未定义的 id",
     lambda: ("wash", synth("wash", _break_reference))),
    ("派生", "合成：start 与 start_frame 不自洽",
     lambda: ("wash", synth("wash", _break_derived))),
    ("重叠", "合成：两段帧号真重叠",
     lambda: ("wash", synth("wash", _break_overlap))),
]


def check_schema_enforced() -> list[str]:
    """`schemas/segments.json` 必须真的被执行（4.4）。

    这个契约此前**没有任何代码加载** —— 它定义了 `id` / `subtask` 的 pattern
    与 `source` 的必填字段，而校验器只手抄了「多余字段」那一条，
    README 却写着「每个层间边界都有 schema + 校验器」。

    这里逐条造违规，确认它真的报 —— **不能只验「当前数据通过」**，
    那样的话把 schema 换成一个空对象也能全过。
    """
    from checks import check_schema
    bad: list[str] = []
    src = sorted((REAL / "wash" / "segments").glob("*_segments.json"))[0]
    doc = json.loads(src.read_text(encoding="utf-8"))

    if check_schema(doc):
        bad.append(f"当前数据本该通过 schema，却报了：{[f.detail for f in check_schema(doc)][:2]}")

    cases = [
        ("id 的 pattern", lambda d: d["segments"][0].__setitem__("id", "不合规的id")),
        ("subtask 的 pattern", lambda d: d["segments"][0].__setitem__("subtask", "Bad Subtask!")),
        ("source 的必填字段", lambda d: d["source"].pop("fps", None)),
        ("拒绝出题产物回写", lambda d: d["segments"][0].__setitem__("metadata", {"window_type": "x"})),
    ]
    for label, mutate in cases:
        broken = json.loads(json.dumps(doc))
        mutate(broken)
        if not check_schema(broken):
            bad.append(f"schema 没抓到「{label}」")
    return bad


def check_video_probe() -> list[str]:
    """与真实视频对一遍的那三条（`--probe-video`）。

    **这三条是从 v1 的 `migrate/check_labels.py` 移植过来的**，
    移植完那个脚本才被删掉 —— 它读的 `narration` 字段早已不存在（输出是假发现），
    但它探真实视频的能力此前没有替代者，而
    「元表声称的 fps 与视频实际 fps 是否一致」正是当年发现 tea2 问题的那类检查。

    与「派生」的区别：那条核的是 `start` 与 `start_frame` **内部自洽**；
    这里核的是**元表与盘上的文件对不对得上**。元表整个错了的话，内部再自洽也没用。
    """
    from checks import check_against_video, probe_video
    video = ROOT / "data" / "source" / "wash" / "file-000" / "main.mp4"
    if not video.exists():
        return []          # 没有 data/source 就跳过，不算失败
    if probe_video(video) == (0.0, 0):
        return []          # 没装 ffprobe，同上

    bad: list[str] = []
    src = REAL / "wash" / "segments" / "file-000_segments.json"
    doc = json.loads(src.read_text(encoding="utf-8"))
    if check_against_video(doc, video):
        bad.append(f"当前数据本该通过：{[f.detail for f in check_against_video(doc, video)]}")

    cases = [
        ("fps 与实际不符", lambda d: d["source"].__setitem__("fps", 30)),
        ("帧号越界", lambda d: d["segments"][-1].__setitem__("end_frame", 999999)),
        ("标注跨度过低", lambda d: d.__setitem__("segments", d["segments"][:1])),
    ]
    for label, mutate in cases:
        broken = json.loads(json.dumps(doc))
        mutate(broken)
        if not check_against_video(broken, video):
            bad.append(f"没抓到「{label}」")
    return bad


def check_online_parity() -> list[str]:
    """在线版（serve.review）与离线版必须给出同一批发现。

    **这是 4.3 的回归。** 此前 `serve.review()` 是另写的一份，只覆盖三类；
    「污染」「引用」「派生」「序列」「可疑」在保存时拦不下 ——
    而「污染」抓的正是 P-03 那类出题产物回写。
    四处文档同时写着「共用同一份判据」，而 serve.py 根本没 import validate。

    这里拿真实的污染数据过一遍在线版，断言它**确实拦得下**。
    """
    import importlib
    bad: list[str] = []
    try:
        serve = importlib.import_module("serve")
    except Exception as exc:  # noqa: BLE001
        return [f"serve.py 导入失败：{type(exc).__name__}: {exc}"]

    doc = json.loads((REAL / "stack_cubes" / "segments.polluted"
                      / sorted(p.name for p in (REAL / "stack_cubes" / "segments.polluted")
                               .glob("*_segments.json"))[0]).read_text(encoding="utf-8"))
    problems = serve.review(doc, "stack_cubes")
    kinds = {p["text"].split("：")[0] for p in problems}
    for want in ("污染", "引用"):
        if want not in kinds:
            bad.append(f"在线版没拦下「{want}」—— 它只报了 {sorted(kinds)}")
    if not any(p["level"] == "block" for p in problems):
        bad.append("在线版没有任何 block 级发现 —— 污染数据应当拦下不写盘")
    return bad


def main() -> int:
    if not REAL.exists():
        print(f"跳过：没有 {REAL}")
        return 0

    failures = 0
    print("一 · 每条检查在【它当初要抓的那次事故】上必须报")
    print(f"{'检查':<6}{'条数':>5}  夹具")
    print("-" * 74)
    for kind, label, get in CASES:
        family, segments = get()
        if not segments:
            print(f"{kind:<6}{'—':>5}  ✗ 夹具不存在：{label}")
            failures += 1
            continue
        counts = run_on(family, segments)
        hit = counts.get(kind, 0)
        print(f"{kind:<6}{hit:>5}  {'✓' if hit else '✗'} {label}")
        failures += not hit

    print()
    print("二 · 当前数据上必须【一条都不报】")
    print(f"{'族':<14}{'段数':>6}  结果")
    print("-" * 74)
    families = sorted(p.name for p in REAL.iterdir()
                      if p.is_dir() and (p / "subtasks.json").exists())
    for family in families:
        segments = sorted((REAL / family / "segments").glob("*_segments.json"))
        counts = run_on(family, segments)
        n = sum(len(json.loads(p.read_text(encoding="utf-8"))["segments"]) for p in segments)
        ok = not counts
        print(f"{family:<14}{n:>6}  {'✓ 零条' if ok else '✗ ' + str(dict(counts))}")
        failures += not ok

    print()
    print("三 · schemas/segments.json 必须真的被执行")
    print("-" * 74)
    bad = check_schema_enforced()
    print(f"{'✓ 契约生效：id / subtask 的 pattern、source 必填、拒绝回写' if not bad else '✗ 契约没生效'}")
    for line in bad:
        print(f"    {line}")
    failures += bool(bad)

    print()
    print("四 · 与真实视频对一遍（--probe-video，从 v1 的 check_labels.py 移植）")
    print("-" * 74)
    bad = check_video_probe()
    print(f"{'✓ fps 自洽 / 帧号越界 / 跨度覆盖率' if not bad else '✗ 视频核对失效'}")
    for line in bad:
        print(f"    {line}")
    failures += bool(bad)

    print()
    print("五 · 在线版（serve.review）与离线版共用同一份判据")
    print("-" * 74)
    bad = check_online_parity()
    print(f"{'✓ 在线版拦得下污染与引用' if not bad else '✗ 在线版与离线版不一致'}")
    for line in bad:
        print(f"    {line}")
    failures += bool(bad)

    print()
    if failures:
        print(f"❌ {failures} 处不符。**两个方向都要成立** —— "
              "只会报的校验器和只会沉默的校验器一样没用。")
        return 1
    print("七类检查：在历史事故上都报，在当前数据上都不报。")
    print("⚠ 「覆盖」没测 —— 它只在多集打包的视频上触发，而唯一那样的族 tea2 已移出。"
          "\n   tea2 回来时要补。")
    print("在线版与离线版 import 同一个 checks.check_document。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
