"""Print the small semantic parity fixture using the actual neutral mifrost engine.

Run with mifrost installed, or pass --native-module /path/to/_neutral_core.so
with the interpreter matching that module. No planning backend or PyG is used.
"""

import argparse
import importlib.util
import json

import numpy as np


class CapsuleArray:
    def __init__(self, capsule):
        self.capsule = capsule

    def __dlpack__(self, **kwargs):
        return self.capsule

    def __dlpack_device__(self):
        return 1, 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-module")
    arguments = parser.parse_args()
    if arguments.native_module:
        specification = importlib.util.spec_from_file_location("_neutral_core", arguments.native_module)
        core = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(core)
    else:
        from mifrost import _neutral_core as core

    category = core.SemanticPredicateCategory
    predicates = [
        core.SemanticPredicateSpec(category.fluent, "flag", 1),
        core.SemanticPredicateSpec(getattr(category, "static"), "fixed", 1),
        core.SemanticPredicateSpec(category.derived, "available", 1),
        core.SemanticPredicateSpec(category.fluent, "ready", 0),
    ]
    config = core.FlatRelationEncoderConfig(
        support_literals=True,
        ignore_zero_arity_relations=False,
        goal_derivations={core.GoalDerivation.satisfied, core.GoalDerivation.unsatisfied},
    )
    engine = core.SemanticFlatRelationEncoderEngine(predicates, [], config)
    value = core.SemanticFlatRelationInput()
    value.objects = ["a", "b"]
    value.state_facts = [core.SemanticAtom(predicate, [0]) for predicate in range(3)]
    value.goals = [
        core.SemanticLiteral(core.SemanticAtom(predicate, objects), positive)
        for predicate, objects, positive in [(0, [1], True), (0, [0], False), (1, [0], True),
                                              (1, [1], False), (2, [1], True), (2, [0], False), (3, [], True)]
    ]
    batch = engine.encode(value)
    tensors = {
        key: np.from_dlpack(payload if hasattr(payload, "__dlpack__") else CapsuleArray(payload))
        for key, payload in batch.as_dict()["tensors"].items()
        if key in {"__graph__/relation_args", "__graph__/relation_counts"}
    }
    counts = tensors["__graph__/relation_counts"][0]
    packed = tensors["__graph__/relation_args"]
    output = {}
    offset = 0
    for name, arity, count in zip(engine.relation_names, batch.graph_attrs["relation_arities"], counts):
        count, arity = int(count), int(arity)
        if count:
            output[name] = packed[offset : offset + count * arity].reshape(count, arity).tolist()
        offset += count * arity
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
