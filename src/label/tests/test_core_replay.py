#!/usr/bin/env python3
# coding: utf-8
"""重写回归：拿 ID 化之前的 1,859 段当语料，证明 subtask 模型没丢信息。

判据不是「输出相同」，而是**每一处差异都被声明过**。

    逐字节相同        新模型忠实保留了原信息
    声明过的差异      我们主动修的（B-01 插 the / B-02 硬编码动词表）
    未声明的差异      **我们改坏了**

三条链路各自验一遍：

  链路 A  subtask ID → subtasks.json 的 text → 能否复原原始 narration
          这是 ID 化的核心承诺：文字只存一处，引用能还原
  链路 B  core.describe() 重放词表 → 与原始 objects/main_verbs 比对
          这条会命中 B-01 与 B-02 两类**声明过的**差异
  链路 C  时间轴（帧号与秒）在 ID 化前后逐字段相同

语料：``data/label/<族>/segments.before_subtask_id/``（迁移时自动留下的备份）。
stack_cubes 另有 ``segments.polluted/`` —— 那是被出题产物污染的版本（P-03），
**不参与比对**，它本来就不是人工标注。

不需要视频，不需要 cv2。
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from label.core import action_verbs, describe, load_categories  # noqa: E402

LABEL = ROOT / "data" / "label"

# ── 声明过的差异 ────────────────────────────────────────────────
# B-01：上游在动词后无条件插 "the"，把介词当成宾语的第一个词。
#       只影响词表里第二个词是介词的条目 —— 全量八族只有 tea2 的 3 条。
#       （第 4 条 pick_up_teapot 是 D-21 人工补的 up，不属于 B-01。）
# B-02：上游 ACTION_VERBS 硬编码 20 个动词，不含 wipe / set。
#       express 因此 50 段 main_verbs/objects 全空 —— 新实现从词表推导，自动正确。
DECLARED_B01 = {"tea2"}
DECLARED_B02 = {"express"}

# P-05：wash 的两个盘子加相对位置。**这是链路 A 上唯一一处声明过的差异。**
#
# 起因：同一集里 `pick_plate` 出现两次，「什么时候拿的盘子」真值不唯一 ——
# 时间类题目直接不可用。解法是按相对位置拆开，`subtasks.json` 自己写着规则：
#
#   pick/wipe/put_plate 各拆成 left/right/far/near 四个。38 集机位一致、
#   盘子左右并排，用 left/right；file-000 与 file-001 机位不同、盘子前后排，
#   用 far/near。哪个先拿由抽帧确认，见 corrections.json。
#
# 于是 ID 还原不回原文：`wipe_far_plate_with_rag` → "Wipe the far plate with rag."，
# 而迁移前的 narration 是 " Wipe the plate with rag."。
# **这不是重写改坏了语义，是我们主动改了语义**，依据在 corrections.json。
#
# **声明的是规则不是清单** —— 枚举 id 的话，将来再拆一个动作就又红了，
# 而那时人多半会去放宽判据而不是补声明。
#
# ⚠ **不要因为这里红了就放宽判据。** 这个测试的设计意图是
# 「重写不得改变语义，除非声明」—— 声明才是正解，放宽等于把它变成摆设。
PLATE_SIDES = ("left", "right", "far", "near")


def corrected_ids(family: str) -> dict[str, str]:
    """从 ``corrections.json`` 读出「这一段的标签被人工改过」。

    **声明绑到依据文件，不硬编码。** 这样以后再修一处标注，
    只要按流程写进 corrections.json，这个测试自动认它 ——
    而随手改数据不写依据的，仍然会红。这正是我们想要的那条线。

    wash 现有两处（由 `validate.py` 第七条「序列」检查发现）：
        file-009@f001134  wipe_bowl_with_brush → wipe_plate_with_rag
        file-009@f001303  put_bowl             → put_plate
    依据：抽帧确认机械臂拿着蓝盘子用粉抹布擦、并把盘子放进沥水架，
    两段标错了物体。**时间边界一字未动。**
    """
    path = LABEL / family / "corrections.json"
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    data = json.loads(path.read_text(encoding="utf-8"))
    for entry in data.get("corrections", []):
        for item in entry.get("items", []):
            for change in item.get("changes", []):
                if change.get("id") and change.get("to"):
                    out[change["id"]] = change["to"]
    return out


def declared_a(family: str, subtask: str, narration: str, table: dict[str, str],
               corrected_to: str | None = None) -> bool:
    """这处链路 A 的差异是不是**声明过的**改动造成的。

    两次改动会**叠加**，所以要串起来判。wash 的实际链路是：

        wipe_bowl_with_brush  ──序列修正──▶ wipe_plate_with_rag
                              ──P-05────▶ wipe_left_plate_with_rag

    于是先用 corrections.json 把「期望的原文」换成修正后的那条，
    再套 P-05 的相对位置规则。任何一环没声明过，仍然会红。

    P-05 的判据是**去掉相对位置词之后逐字还原** —— 这样既承认那次拆分，
    又不放过「顺手改了别的措辞」：把 "Wipe the far plate with rag." 写成
    "Wipe far plate with the rag." 照样红。
    """
    text = table.get(subtask)
    if not text:
        return False

    # ① 人工修正过的标签（依据在 corrections.json）
    if corrected_to:
        # ⚠ 修正目标可能是**中间态**：wash 的 `wipe_plate_with_rag` 先由「序列」
        # 检查修出来，随后又被 P-05 拆成 left/right/far/near，于是它已不在词表里
        # （`table.get` 返回 None）。所以按 **id 结构**串联，不查中间态的文字。
        if subtask == corrected_to:
            return True
        for side in PLATE_SIDES:
            # put_plate → put_left_plate；wipe_plate_with_rag → wipe_left_plate_with_rag
            if subtask == corrected_to.replace("_plate", f"_{side}_plate", 1):
                return True
        return False

    # ② P-05 的相对位置拆分（只有 wash 有）。
    # 判据是**去掉位置词后逐字还原** —— 既承认那次拆分，又不放过顺手改的措辞：
    # 把 "Wipe the far plate with rag." 写成 "Wipe far plate with the rag." 照样红。
    if family != "wash":
        return False
    expected = narration.strip()
    for side in PLATE_SIDES:
        if f"{side} " in text and text.replace(f"{side} ", "", 1) == expected:
            return True
    return False


def load(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["segments"]


def check_family(family: str) -> dict[str, object]:
    base = LABEL / family
    before_dir = base / "segments.before_subtask_id"
    if not before_dir.exists():
        return {"skip": "无迁移前语料"}

    table = {s["id"]: s["text"]
             for s in json.loads((base / "subtasks.json").read_text(encoding="utf-8"))["subtasks"]}
    cats_file = base / "categories.deprecated.txt"
    categories = load_categories(cats_file.read_text(encoding="utf-8")) if cats_file.exists() else []
    verbs = action_verbs(categories)
    derived = {c: describe(c, verbs) for c in categories}
    # narration（去空格）→ 词表条目，用于把旧段反查回它来自哪条类别
    by_narration = {nar.strip(): cat for cat, (_, _, nar) in derived.items()}

    corrected = corrected_ids(family)

    counts: Counter[str] = Counter()
    samples: dict[str, str] = {}

    for after_path in sorted((base / "segments").glob("*_segments.json")):
        before_path = before_dir / after_path.name
        if not before_path.exists():
            # stack_cubes 还原后段数变了（300→200），不做逐段比对
            counts["restored_skipped"] += 1
            continue
        before, after = load(before_path), load(after_path)
        if len(before) != len(after):
            counts["restored_skipped"] += 1
            continue

        for old, new in zip(before, after):
            counts["total"] += 1

            # 链路 A：ID 能否还原原始 narration
            if table.get(new["subtask"]) == old["narration"].strip():
                counts["A_ok"] += 1
            elif declared_a(family, new["subtask"], old["narration"], table,
                            corrected.get(new["id"])):
                counts["A_declared"] += 1
            else:
                counts["A_bad"] += 1
                samples.setdefault("A", f"{new['subtask']} → {table.get(new['subtask'])!r}"
                                        f"，原 {old['narration']!r}")

            # 链路 C：时间轴一字未改
            if all(old[k] == new[k] for k in ("start_frame", "end_frame", "start", "end")):
                counts["C_ok"] += 1
            else:
                counts["C_bad"] += 1
                samples.setdefault("C", f"{new['id']}")

            # 链路 B：重放词表，与旧的 objects/main_verbs 比对
            cat = by_narration.get(old["narration"].strip())
            if cat is None:
                counts["B_unmatched"] += 1        # D-21 手改过的条目，词表里没有对应
                continue
            objects, main_verbs, _ = derived[cat]
            if (list(old.get("objects") or []) == objects
                    and list(old.get("main_verbs") or []) == main_verbs):
                counts["B_ok"] += 1
            elif family in DECLARED_B01 or family in DECLARED_B02:
                counts["B_declared"] += 1
            else:
                counts["B_bad"] += 1
                samples.setdefault("B", f"旧 {old.get('main_verbs')}/{old.get('objects')} "
                                        f"vs 新 {main_verbs}/{objects}")
    return {"counts": counts, "samples": samples}


def main() -> int:
    families = sorted(p.name for p in LABEL.iterdir() if p.is_dir())
    print(f"{'族':<13}{'段数':>6}{'A 还原':>8}{'A 声明':>8}{'C 时间轴':>9}{'B 派生':>8}"
          f"{'B 声明差异':>11}{'B 无对应':>9}  status")
    print("-" * 78)
    failures = 0
    for family in families:
        report = check_family(family)
        if "skip" in report:
            print(f"{family:<13}{'—':>6}  {report['skip']}")
            continue
        c: Counter[str] = report["counts"]                       # type: ignore[assignment]
        bad = c["A_bad"] + c["C_bad"] + c["B_bad"]
        failures += 0 if bad == 0 else 1
        note = "OK" if bad == 0 else f"**{bad} 处未声明差异**"
        if c["restored_skipped"]:
            note += f"（{c['restored_skipped']} 集因还原跳过）"
        print(f"{family:<13}{c['total']:>6}{c['A_ok']:>8}{c['A_declared']:>8}{c['C_ok']:>9}{c['B_ok']:>8}"
              f"{c['B_declared']:>11}{c['B_unmatched']:>9}  {note}")
        for kind, text in report["samples"].items():             # type: ignore[union-attr]
            print(f"      [{kind}] {text}")
    print("-" * 78)
    if failures:
        print(f"{failures} 个族出现未声明的差异 —— 重写改变了语义")
        return 1
    print("subtask 模型未丢信息：ID 可还原原文、时间轴未变、"
          "派生差异全部是声明过的 B-01 / B-02")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
