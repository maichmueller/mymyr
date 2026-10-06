"""Problems of the table instance sets for domains whose instances in the fork's data dir are
all small: gripper (data/gripper/domain.pddl), logistics (data/logistics/domain.pddl) and miconic-simpleadl
(data/miconic-simpleadl/domain.pddl, conditional effects). The problems use those domain files unchanged and follow the
layout of the fork's own problems there. Deterministic (no randomness): python3 tests/data/table_instances/gen.py writes the files
next to this script.
"""
import pathlib

HERE = pathlib.Path(__file__).resolve().parent


def gripper(n):
    """n balls in rooma, both grippers free (data/gripper/test_problem4.pddl for n = 4)."""
    balls = [f"ball{i}" for i in range(1, n + 1)]
    lines = [f"(define (problem gripper-{n})", "    (:domain gripper-strips)", "    (:objects",
             "        left right " + " ".join(balls), "    )", "    (:init", "        (room rooma)", "        (room roomb)",
             "        (gripper left)", "        (gripper right)"]
    lines += [f"        (ball {b})" for b in balls]
    lines += ["        (free left)", "        (free right)"]
    lines += [f"        (at {b} rooma)" for b in balls]
    lines += ["        (at-robby rooma)", "    )", "    (:goal", "        (and"]
    lines += [f"            (at {b} roomb)" for b in balls]
    lines += ["        )", "    )", ")"]
    return "\n".join(lines) + "\n"


def logistics(cities, size, packages, planes):
    """`cities` cities of `size` locations each (location 0 an airport), one truck per city, `planes` airplanes and
    `packages` packages (data/logistics/test_problem.pddl's layout); positions and goals by a fixed arithmetic rule."""
    locs = [[f"l{c}-{k}" for k in range(size)] for c in range(cities)]
    objs = ([f"a{i}" for i in range(planes)] + [f"c{c}" for c in range(cities)] + [f"t{c}" for c in range(cities)]
            + [l for ls in locs for l in ls] + [f"p{i}" for i in range(packages)])
    name = f"logistics-c{cities}-s{size}-p{packages}-a{planes}"
    lines = [f"(define (problem {name})", "(:domain logistics-strips)", "(:objects " + " ".join(objs), ")", "(:init"]
    lines += [f"    (AIRPLANE a{i})" for i in range(planes)]
    for c in range(cities):
        lines += [f"    (CITY c{c})", f"    (TRUCK t{c})"]
        for l in locs[c]:
            lines += [f"    (LOCATION {l})", f"    (in-city  {l} c{c})"]
        lines.append(f"    (AIRPORT {locs[c][0]})")
    lines += [f"    (OBJ p{i})" for i in range(packages)]
    lines += [f"    (at t{c} {locs[c][(c + 1) % size]})" for c in range(cities)]
    for i in range(packages):
        c = (3 * i + 1) % cities
        lines.append(f"    (at p{i} {locs[c][(5 * i + 2) % size]})")
    lines += [f"    (at a{i} {locs[i % cities][0]})" for i in range(planes)]
    lines += [")", "(:goal", "    (and"]
    for i in range(packages):
        c = (7 * i + 2) % cities
        lines.append(f"        (at p{i} {locs[c][(3 * i + 1) % size]})")
    lines += ["    )", ")", ")"]
    return "\n".join(lines) + "\n"


def miconic_simpleadl(floors, passengers, quantified):
    """The layout of pddl-generators' miconic-simpleadl (bench/cuda/conditional_tasks/simple-f24-p12.pddl): the lift at f0,
    passenger i from floor (3i + 1) % F to (5i + 2) % F (moved one floor up where the two coincide). The goal is
    `(forall (?p - passenger) (served ?p))` (a goal axiom after normalization) or the conjunction of the served atoms."""
    name = f"simple-f{floors}-p{passengers}-{'q' if quantified else 'c'}"
    ps = [f"p{i}" for i in range(passengers)]
    fs = [f"f{i}" for i in range(floors)]
    lines = [f"(define (problem {name})", "   (:domain miconic)", "   (:objects " + " ".join(ps) + " - passenger",
             "             " + " ".join(fs) + " - floor)", "", "(:init"]
    for a in range(floors):
        for b in range(a + 1, floors):
            lines.append(f"(above f{a} f{b})")
    for i in range(passengers):
        o = (3 * i + 1) % floors
        d = (5 * i + 2) % floors
        if d == o:
            d = (d + 1) % floors
        lines += [f"(origin p{i} f{o})", f"(destin p{i} f{d})"]
    lines += ["(lift-at f0)", ")", ""]
    if quantified:
        lines.append("(:goal (forall (?p - passenger) (served ?p)))")
    else:
        lines += ["(:goal (and"] + [f"  (served p{i})" for i in range(passengers)] + ["))"]
    lines.append(")")
    return "\n".join(lines) + "\n"


def main():
    for n in (4, 10, 20, 40, 80):
        (HERE / "gripper").mkdir(exist_ok=True)
        (HERE / "gripper" / f"gripper-{n}.pddl").write_text(gripper(n))
    for c, s, p, a in ((1, 2, 2, 1), (2, 3, 4, 1), (3, 3, 8, 2), (4, 4, 16, 3), (6, 5, 30, 4)):
        (HERE / "logistics").mkdir(exist_ok=True)
        (HERE / "logistics" / f"logistics-c{c}-s{s}-p{p}-a{a}.pddl").write_text(logistics(c, s, p, a))
    for f, p, q in ((3, 2, False), (6, 4, True), (12, 8, False), (24, 16, True), (40, 50, True), (60, 100, False)):
        (HERE / "miconic-simpleadl").mkdir(exist_ok=True)
        (HERE / "miconic-simpleadl" / f"simple-f{f}-p{p}-{'q' if q else 'c'}.pddl").write_text(miconic_simpleadl(f, p, q))


if __name__ == "__main__":
    main()
