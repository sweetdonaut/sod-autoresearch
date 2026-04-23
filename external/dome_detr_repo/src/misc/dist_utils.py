"""Local stub for Dome-DETR's `src.misc.dist_utils.all_gather`.

Used only by the `merge()` function in coco_eval_visdrone.py, which in
turn is used by `VisdroneCocoEvaluator.synchronize_between_processes()`
(multi-GPU eval result merging). We run single-process eval, so
all_gather just wraps the input in a 1-element list.

If the merge path is ever hit (shouldn't be — we don't call the wrapper),
this fallback keeps behavior consistent with single-process rank=0.
"""


def all_gather(x):
    return [x]
