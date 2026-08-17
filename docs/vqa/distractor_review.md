# 干扰项人工复核清单

盲测只能测「零样本泄漏」。以下两类它测不到，必须人读：

1. **跨题可学的线索** —— 某条干扰项在本族每道题里都出现、且永远不是答案。
   零样本模型学不到，但题目公开后会进训练数据。
2. **读起来别扭的重组** —— 语法闸门与合理性闸门都放过了，但人一眼觉得不自然。

判断标准只有一条：**这个选项必须只能靠看视频排除。**
如果读一遍文字就觉得「数据集里不会有这个」，它就是一条捷径。

---

## airpods

场景物体：`airpods case` `left earphone` `right earphone`　真实动作 5 个　题目 520 道

| 重组干扰项 | 出现在几道题 | 占比 | 读起来自然吗（请填） |
| --- | ---: | ---: | --- |
| （不需要重组项，真实动作足够填满） | — | — | — |

样例题（各取一道）：

- `understanding` ✓**Pick the airpods case.**　/　Open the airpods case.　/　Pick the left earphone.　/　Pick the right earphone.　/　Close the airpods case.
- `planning` ✓**Open the airpods case.**　/　Pick the left earphone.　/　Pick the right earphone.　/　Close the airpods case.　/　Pick the airpods case.
- `planning_2` ✓**Open the airpods case.**　/　Close the airpods case.　/　Pick the airpods case.　/　Pick the left earphone.　/　Pick the right earphone.

## gift_inhand

场景物体：`gift`　真实动作 3 个　题目 210 道

| 重组干扰项 | 出现在几道题 | 占比 | 读起来自然吗（请填） |
| --- | ---: | ---: | --- |
| `Close the gift.` | 210 | 100% ⚠ **几乎每题都用** |  |
| `Open the gift.` | 210 | 100% ⚠ **几乎每题都用** |  |

样例题（各取一道）：

- `understanding` ✓**Pick the gift.**　/　Move the gift.　/　Place the gift.　/　Close the gift. ⟵重组　/　Open the gift. ⟵重组
- `planning` ✓**Move the gift.**　/　Place the gift.　/　Pick the gift.　/　Close the gift. ⟵重组　/　Open the gift. ⟵重组
- `planning_2` ✓**Move the gift.**　/　Place the gift.　/　Pick the gift.　/　Close the gift. ⟵重组　/　Open the gift. ⟵重组

## pen_inbox

场景物体：`box` `pen`　真实动作 3 个　题目 350 道

| 重组干扰项 | 出现在几道题 | 占比 | 读起来自然吗（请填） |
| --- | ---: | ---: | --- |
| `Place the box.` | 350 | 100% ⚠ **几乎每题都用** |  |
| `Close the box.` | 350 | 100% ⚠ **几乎每题都用** |  |

样例题（各取一道）：

- `understanding` ✓**Pick the pen.**　/　Place the pen.　/　Pick the box.　/　Place the box. ⟵重组　/　Close the box. ⟵重组
- `planning` ✓**Pick the box.**　/　Pick the pen.　/　Place the pen.　/　Place the box. ⟵重组　/　Close the box. ⟵重组
- `planning_2` ✓**Pick the box.**　/　Place the pen.　/　Pick the pen.　/　Place the box. ⟵重组　/　Close the box. ⟵重组

## stack_cubes

场景物体：`red cube` `yellow cube`　真实动作 4 个　题目 500 道

| 重组干扰项 | 出现在几道题 | 占比 | 读起来自然吗（请填） |
| --- | ---: | ---: | --- |
| `Wipe the red cube.` | 500 | 100% ⚠ **几乎每题都用** |  |

样例题（各取一道）：

- `understanding` ✓**Pick the red cube.**　/　Place the red cube.　/　Pick the yellow cube.　/　Place the yellow cube.　/　Wipe the red cube. ⟵重组
- `planning` ✓**Place the red cube.**　/　Pick the red cube.　/　Pick the yellow cube.　/　Place the yellow cube.　/　Wipe the red cube. ⟵重组
- `planning_2` ✓**Place the red cube.**　/　Pick the yellow cube.　/　Place the yellow cube.　/　Pick the red cube.　/　Wipe the red cube. ⟵重组

## tea

场景物体：`stirrer` `tea` `tea leaves` `teacup` `teapot lid`　真实动作 6 个　题目 624 道

| 重组干扰项 | 出现在几道题 | 占比 | 读起来自然吗（请填） |
| --- | ---: | ---: | --- |
| `Open the teacup.` | 351 | 56% ⚠ **几乎每题都用** |  |
| `Put the teacup.` | 351 | 56% ⚠ **几乎每题都用** |  |
| `Open the tea leaves.` | 273 | 44% |  |
| `Wipe the tea leaves.` | 273 | 44% |  |

样例题（各取一道）：

- `understanding` ✓**Open the teapot lid.**　/　Close the teapot lid.　/　Put the tea leaves.　/　Open the tea leaves. ⟵重组　/　Wipe the tea leaves. ⟵重组
- `planning` ✓**Put the tea leaves.**　/　Close the teapot lid.　/　Open the teapot lid.　/　Open the tea leaves. ⟵重组　/　Wipe the tea leaves. ⟵重组
- `planning_2` ✓**Put the tea leaves.**　/　Open the teapot lid.　/　Close the teapot lid.　/　Open the tea leaves. ⟵重组　/　Wipe the tea leaves. ⟵重组

## tea2

场景物体：`kettle` `lid` `tea` `tea bag` `teapot` `teapot lid` `water`　真实动作 8 个　题目 440 道

| 重组干扰项 | 出现在几道题 | 占比 | 读起来自然吗（请填） |
| --- | ---: | ---: | --- |
| `Place the lid in teapot.` | 140 | 32% |  |
| `Put the kettle on the teapot.` | 140 | 32% |  |
| `Put the tea bag.` | 120 | 27% |  |
| `Put the teapot lid.` | 120 | 27% |  |
| `Wipe the tea bag.` | 120 | 27% |  |
| `Pour the teapot.` | 120 | 27% |  |
| `Put the kettle.` | 120 | 27% |  |
| `Put the tea.` | 120 | 27% |  |
| `Place the teapot lid in teapot.` | 60 | 14% |  |
| `Put the tea bag in teapot.` | 60 | 14% |  |

样例题（各取一道）：

- `understanding` ✓**Pick up the teapot lid.**　/　Pick up the tea bag.　/　Put the lid on the teapot.　/　Place the lid in teapot. ⟵重组　/　Put the kettle on the teapot. ⟵重组
- `planning` ✓**Pick up the tea bag.**　/　Put the lid on the teapot.　/　Pick up the teapot lid.　/　Place the lid in teapot. ⟵重组　/　Put the kettle on the teapot. ⟵重组
- `planning_2` ✓**Pick up the tea bag.**　/　Pick up the teapot lid.　/　Put the lid on the teapot.　/　Place the lid in teapot. ⟵重组　/　Put the kettle on the teapot. ⟵重组

## wash

场景物体：`bowl` `brush` `plate` `rag`　真实动作 10 个　题目 1471 道

| 重组干扰项 | 出现在几道题 | 占比 | 读起来自然吗（请填） |
| --- | ---: | ---: | --- |
| `Wipe the bowl with rag.` | 360 | 24% |  |
| `Wipe the brush with rag.` | 360 | 24% |  |
| `Wipe the plate with brush.` | 360 | 24% |  |

样例题（各取一道）：

- `understanding` ✓**Pick the brush.**　/　Put the plate.　/　Put the rag.　/　Pick the bowl.　/　Put the brush.
- `planning` ✓**Pick the bowl.**　/　Put the brush.　/　Put the bowl.　/　Pick the plate.　/　Pick the rag.
- `planning_2` ✓**Pick the bowl.**　/　Put the rag.　/　Pick the brush.　/　Put the brush.　/　Put the bowl.

