"""Local stub for Dome-DETR's `src.core.register`.

Dome-DETR's register() is used to decorate classes for insertion into
their framework's class-registry. For our standalone eval we don't run
their framework, so register() is a no-op.

The only consumer of `register()` in coco_eval_visdrone.py is the
`VisdroneCocoEvaluator` wrapper class (used for distributed training eval
in Dome-DETR's own training loop). Our eval script uses the underlying
`VisdroneCOCOeval_faster` class and `detections_in_ignore_regions()`
helper directly, so this stub never affects the numbers reported.
"""


def register():
    return lambda cls: cls
