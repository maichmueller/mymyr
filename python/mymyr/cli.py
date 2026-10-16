"""The ``mymyr`` command line (also ``python -m mymyr``): plan a PDDL task with one of mymyr's searches, validate a
plan, write the normalized task, or summarize a domain and problem.

    mymyr plan domain.pddl problem.pddl                      # A* with h_max; the plan to stdout
    mymyr plan domain.pddl problem.pddl --search gbfs -o plan.txt -v
    mymyr plan domain.pddl problem.pddl --search astar --heuristic ff --weight 2
    mymyr plan domain.pddl problem.pddl --search anytime --weights 5,3,2,1 --max-seconds 60
    mymyr validate domain.pddl problem.pddl plan.txt
    mymyr write domain.pddl problem.pddl --format pddl -o normalized/
    mymyr info domain.pddl problem.pddl
    mymyr --version

Exit codes of ``mymyr plan``: 0 a plan was found (by the anytime search also when a budget stopped it later), 1 the
search proved the task unsolvable or exhausted its search space without a plan, 2 invalid usage, an unreadable file or
PDDL mymyr cannot read (and any other error), 3 a budget (``--max-states``, ``--max-seconds``) stopped the search.
``mymyr validate``: 0 the plan is valid, 1 it is not, 2 as above. Errors are one line on stderr; ``--debug`` adds the
traceback.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import threading
import time
import traceback
from collections.abc import Callable, Sequence
from typing import Any, NoReturn

import mymyr
from mymyr import search

__all__ = ["main"]

EXIT_SOLVED = 0
EXIT_NO_PLAN = 1
EXIT_ERROR = 2
EXIT_BUDGET = 3
EXIT_INTERRUPTED = 130

SEARCHES = ("astar", "gbfs", "ehc", "anytime", "brfs", "iw", "siw", "astar_iw")
HEURISTICS = ("blind", "max", "add", "ff", "h2", "set_additive", "lmcut", "lmcount", "perfect")
ADMISSIBLE = ("blind", "max", "h2", "lmcut", "perfect")
DEFAULT_HEURISTIC = {"astar": "max", "gbfs": "ff", "ehc": "ff", "anytime": "ff", "astar_iw": "max"}
WIDTH_SEARCHES = ("iw", "siw", "astar_iw")


class UsageError(Exception):
    """A combination of options that does not apply (exit code 2)."""


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        # one line, like the other errors (argparse prints the usage too)
        self.exit(EXIT_ERROR, f"{self.prog}: error: {message} (see {self.prog} --help)\n")


def _parser() -> argparse.ArgumentParser:
    p = _Parser(
        prog="mymyr",
        description="Lifted classical and numeric planning: plan a PDDL task, validate a plan, write the normalized "
        "task or summarize it.",
    )
    p.add_argument("--version", action="version", version=f"mymyr {mymyr.__version__}")
    sub = p.add_subparsers(dest="command", metavar="COMMAND")

    plan = sub.add_parser(
        "plan",
        help="find a plan",
        description="Search for a plan and write it in the IPC plan format (to stdout unless -o is given).",
        epilog="Exit codes: 0 solved; 1 proved unsolvable or search space exhausted without a plan; 2 invalid usage, "
        "unreadable file or unsupported PDDL; 3 --max-states or --max-seconds stopped the search.",
    )
    plan.add_argument("domain", metavar="DOMAIN", help="the PDDL domain file")
    plan.add_argument("problem", metavar="PROBLEM", help="the PDDL problem file")
    plan.add_argument(
        "-s",
        "--search",
        choices=SEARCHES,
        default="astar",
        help="astar (A*, optimal with an admissible heuristic; weighted with --weight), gbfs (greedy best-first), "
        "ehc (enforced hill-climbing with helpful actions, greedy best-first when it fails), anytime (restarting "
        "weighted A*: better plans until the budget ends or the plan is proven optimal; the best is written), brfs "
        "(breadth-first: shortest plans), iw (iterated width), siw (serialized IW), astar_iw (A* with novelty "
        "pruning) (default: astar)",
    )
    plan.add_argument(
        "-H",
        "--heuristic",
        choices=HEURISTICS,
        help="the heuristic of astar, gbfs, ehc, anytime and astar_iw (default: max for astar and astar_iw, ff for "
        f"the others). A* finds optimal plans with an admissible one: {', '.join(ADMISSIBLE)}; add, ff, set_additive "
        "and lmcount are not admissible. perfect computes the state space first.",
    )
    plan.add_argument(
        "--weight",
        type=float,
        help="the weight w of astar: f = g + w * h, at least 1; with an admissible heuristic the plan costs at most "
        "w times the optimum (default: 1, A*)",
    )
    plan.add_argument(
        "--weights",
        type=_weights,
        metavar="W1,W2,...",
        help="the weights of the anytime search's iterations, non-increasing and at least 1; the last one repeats "
        "(default: 5,3,2,1)",
    )
    plan.add_argument(
        "-w",
        "--width",
        type=int,
        help="the largest novelty width of iw and siw (default: 2), the novelty width of astar_iw (default: 1)",
    )
    plan.add_argument(
        "-j", "--threads", type=int, default=1, help="threads of brfs (the other searches run on one thread; default: 1)"
    )
    plan.add_argument(
        "--atoms",
        choices=("lazy", "auto", "reachable", "dense"),
        default="lazy",
        help="atom slots of the task (Task.from_pddl): lazy gives slots to the atoms the search meets, the narrowest "
        "states on large tasks; auto, reachable and dense fix them before the search (default: lazy)",
    )
    plan.add_argument("--max-seconds", type=float, help="stop the search after this many seconds (exit code 3)")
    plan.add_argument(
        "--max-states", type=int, help="stop the search after storing this many states (per IW pass; exit code 3)"
    )
    plan.add_argument("-o", "--plan-file", metavar="FILE", help="write the plan to FILE instead of stdout")
    plan.add_argument(
        "-v", "--verbose", action="store_true",
        help="print statistics (expanded, generated, stored and pruned states, time, memory, cost) to stderr"
    )
    plan.add_argument("--debug", action="store_true", help="show the traceback of an error")
    plan.set_defaults(run=_plan)

    validate = sub.add_parser(
        "validate",
        help="check a plan",
        description="Replay a plan on the task: check that each action names the task's objects with the declared "
        "types and is applicable, apply its effects (conditional and numeric ones, derived predicates through the "
        "axioms), check the goal at the end and compute the plan's cost under the task's metric.",
        epilog="Exit codes: 0 the plan is valid; 1 it is not (the first failing action and the reason go to stderr); "
        "2 invalid usage, an unreadable file or unsupported PDDL.",
    )
    validate.add_argument("domain", metavar="DOMAIN", help="the PDDL domain file")
    validate.add_argument("problem", metavar="PROBLEM", help="the PDDL problem file")
    validate.add_argument(
        "plan", metavar="PLAN", help="the plan in the IPC format: one '(action object ...)' per line, ';' comments"
    )
    validate.add_argument("--debug", action="store_true", help="show the traceback of an error")
    validate.set_defaults(run=_validate)

    write = sub.add_parser(
        "write",
        help="write the normalized task",
        description="Write tasks as mymyr normalizes them: as mymyr's task text or as PDDL. The PDDL of several "
        "problems shares one domain file; each problem parses back with it to the same task.",
    )
    write.add_argument("domain", metavar="DOMAIN", help="the PDDL domain file")
    write.add_argument("problems", metavar="PROBLEM", nargs="+", help="PDDL problem files of the domain")
    write.add_argument(
        "-f", "--format", choices=("text", "pddl"), default="text", help="mymyr's task text or PDDL (default: text)"
    )
    write.add_argument(
        "-o",
        "--output",
        metavar="DIR",
        help="write files into DIR (created when missing): <problem>.txt per problem, or domain.pddl and "
        "<problem>.pddl; without it one problem's text, or its domain and problem, go to stdout",
    )
    write.add_argument("--debug", action="store_true", help="show the traceback of an error")
    write.set_defaults(run=_write)

    info = sub.add_parser(
        "info",
        help="summarize a domain and problem",
        description="Summarize a domain (and a problem) as mymyr reads them: requirements, types, predicates, the "
        "schemas that normalization made of each action, objects, initial atoms, numeric fluents, axioms and "
        "conditional effects.",
    )
    info.add_argument("domain", metavar="DOMAIN", help="the PDDL domain file")
    info.add_argument("problem", metavar="PROBLEM", nargs="?", help="a PDDL problem file of the domain")
    info.add_argument("--debug", action="store_true", help="show the traceback of an error")
    info.set_defaults(run=_info)
    return p


# ----------------------------------------------------------------------------------------------------------- plan


def _weights(text: str) -> tuple[float, ...]:
    """The --weights list: comma-separated numbers."""
    try:
        return tuple(float(w) for w in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a comma-separated list of numbers: {text!r}") from None


def _check_options(a: argparse.Namespace) -> None:
    if a.heuristic is not None and a.search not in DEFAULT_HEURISTIC:
        raise UsageError(f"--heuristic applies to astar, gbfs, ehc, anytime and astar_iw, not to {a.search}")
    if a.weight is not None and a.search != "astar":
        raise UsageError(f"--weight applies to astar, not to {a.search} (the anytime search takes --weights)")
    if a.weight is not None and not 1 <= a.weight < float("inf"):
        raise UsageError("--weight must be a finite number of at least 1")
    if a.weights is not None and a.search != "anytime":
        raise UsageError(f"--weights applies to anytime, not to {a.search}")
    if a.weights is not None and not all(1 <= w < float("inf") for w in a.weights):
        raise UsageError("--weights must be finite numbers of at least 1")
    if a.weights is not None and any(x < y for x, y in zip(a.weights, a.weights[1:])):
        raise UsageError("--weights must not increase")
    if a.width is not None and a.search not in WIDTH_SEARCHES:
        raise UsageError(f"--width applies to iw, siw and astar_iw, not to {a.search}")
    if a.width is not None and a.width < 0:
        raise UsageError("--width must be at least 0")
    if a.threads != 1 and a.search != "brfs":
        raise UsageError(f"--threads applies to brfs; {a.search} runs on one thread")
    if a.threads < 0:
        raise UsageError("--threads must be at least 0 (0: one per core)")
    if a.max_states is not None and a.max_states < 0:
        raise UsageError("--max-states must be at least 0")
    if a.max_seconds is not None and not a.max_seconds >= 0:
        raise UsageError("--max-seconds must be at least 0")


def _search(task: mymyr.Task, a: argparse.Namespace, cancel: search.CancelToken) -> Any:
    budget = {"max_states": a.max_states, "max_seconds": a.max_seconds, "cancel": cancel}
    h = a.heuristic or DEFAULT_HEURISTIC.get(a.search)
    if a.search == "astar":
        return search.astar(task, heuristic=h, weight=1.0 if a.weight is None else a.weight, **budget)
    if a.search == "gbfs":
        return search.gbfs(task, heuristic=h, **budget)
    if a.search == "ehc":
        return search.ehc(task, heuristic=h, **budget)
    if a.search == "anytime":
        return search.anytime_astar(task, heuristic=h, weights=a.weights or (5, 3, 2, 1), **budget)
    if a.search == "brfs":
        return search.brfs(task, threads=a.threads, **budget)
    width = {} if a.width is None else ({"width": a.width} if a.search == "astar_iw" else {"max_width": a.width})
    if a.search == "iw":
        return search.iw(task, **width, **budget)
    if a.search == "siw":
        return search.siw(task, **width, **budget)
    return search.astar_iw(task, heuristic=h, **width, **budget)


def _cancellable(run: Callable[[search.CancelToken], Any]) -> Any:
    """Runs a search on a worker thread so that Ctrl-C (KeyboardInterrupt on this thread) cancels it."""
    token = search.CancelToken()
    out: dict[str, Any] = {}

    def work() -> None:
        try:
            out["result"] = run(token)
        except BaseException as e:  # re-raised on the calling thread
            out["error"] = e

    worker = threading.Thread(target=work, name="mymyr-search", daemon=True)
    worker.start()
    try:
        while worker.is_alive():
            worker.join(0.1)
    except KeyboardInterrupt:
        token.request()
        worker.join()
        raise
    if "error" in out:
        raise out["error"]
    return out["result"]


def _plan(a: argparse.Namespace) -> int:
    _check_options(a)
    t0 = time.perf_counter()
    task = mymyr.Task.from_pddl(a.domain, a.problem, atoms=a.atoms)
    parse_s = time.perf_counter() - t0
    r = _cancellable(lambda token: _search(task, a, token))
    status = r.status
    text = None
    if status == search.Status.SOLVED:
        text = search.format_plan(task, r.plan)
        if a.plan_file:
            with open(a.plan_file, "w") as f:
                f.write(text)
        else:
            sys.stdout.write(text)
            sys.stdout.flush()
    if a.verbose:
        st = r.stats
        h = a.heuristic or DEFAULT_HEURISTIC.get(a.search)
        weight = f", weight {a.weight:g}" if a.weight is not None else ""
        lines = [
            f"search: {a.search}" + (f" with heuristic {h}" if h else "") + weight,
            f"status: {status}",
            f"expanded: {st.expanded}",
            f"generated: {st.generated}",
            f"states: {st.states}",
            f"pruned: {st.pruned}",
            f"time: {st.seconds:.3f} s search, {parse_s:.3f} s parse",
        ]
        if st.memory_bytes:
            lines.append(f"memory: {st.memory_bytes / 2**20:.1f} MiB")
        if isinstance(r, search.EhcResult):
            e = r.ehc
            lines.append(
                f"hill-climbing: {e.lookaheads} lookaheads, {e.improvements} improvements, {e.retries} retries"
                + (", failed; greedy best-first search ran" if e.fallback else (", failed" if e.failed else ""))
            )
        if isinstance(r, search.AnytimeResult):
            for x in r.solutions:
                lines.append(f"plan: cost {x.cost:g} at {x.seconds:.3f} s (weight {x.weight:g}, length {x.length})")
            lines.append(f"iterations: {len(r.iterations)}, optimal: {'yes' if r.optimal else 'not proven'}")
        if text is not None:
            lines.append(f"plan length: {len(r.plan)}")
            lines.append(f"cost: {r.cost:g}")
        print("\n".join(lines), file=sys.stderr)
    if status == search.Status.SOLVED:
        return EXIT_SOLVED
    if status in (search.Status.EXHAUSTED, search.Status.UNSOLVABLE):
        what = "the goal is unreachable" if status == search.Status.UNSOLVABLE else "the search space is exhausted"
        print(f"mymyr: no plan: {what} ({status})", file=sys.stderr)
        return EXIT_NO_PLAN
    if status in (search.Status.OUT_OF_STATES, search.Status.OUT_OF_TIME):
        print(f"mymyr: no plan: the search stopped at its budget ({status})", file=sys.stderr)
        return EXIT_BUDGET
    if status == search.Status.CANCELLED:
        print("mymyr: no plan: the search was cancelled", file=sys.stderr)
        return EXIT_INTERRUPTED
    message = getattr(r, "message", "") or str(status)
    raise UsageError(f"{a.search} cannot run on this task: {message}")


# ------------------------------------------------------------------------------------------------------- validate


def _validate(a: argparse.Namespace) -> int:
    task = mymyr.Task.from_pddl(a.domain, a.problem)
    v = search.validate_plan(task, pathlib.Path(a.plan))
    if v.valid:
        n = len(v.plan)
        print(f"valid plan: {n} action{'' if n == 1 else 's'}, cost {v.cost:g}")
        return EXIT_SOLVED
    print(f"mymyr: invalid plan: {v.message}", file=sys.stderr)
    return EXIT_NO_PLAN


# ---------------------------------------------------------------------------------------------------------- write


def _write(a: argparse.Namespace) -> int:
    if a.output is None and len(a.problems) > 1:
        raise UsageError("several problems need --output DIR")
    tasks = [mymyr.Task.from_pddl(a.domain, p) for p in a.problems]
    stems = [pathlib.Path(p).stem for p in a.problems]
    if len(set(stems)) < len(stems):
        raise UsageError("the problem files' names (without suffix) must differ")
    if a.format == "text":
        files = {f"{s}.txt": t.to_text() for s, t in zip(stems, tasks)}
    else:
        domain, problems = mymyr.write_pddl(tasks)
        if "domain" in stems:
            raise UsageError("a problem file named domain.pddl would overwrite the domain")
        files = {"domain.pddl": domain} | {f"{s}.pddl": p for s, p in zip(stems, problems)}
    if a.output is None:
        sys.stdout.write("\n".join(files.values()))
        return EXIT_SOLVED
    out = pathlib.Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (out / name).write_text(text)
    return EXIT_SOLVED


# ----------------------------------------------------------------------------------------------------------- info


def _requirement(r: str) -> str:
    return ":" + r.lstrip(":")


def _names(items: Sequence[Any], limit: int = 12) -> str:
    names = [str(x) for x in items]
    shown = ", ".join(names[:limit])
    return shown + (f", ... ({len(names) - limit} more)" if len(names) > limit else "")


def _is_conditional(effect: Any) -> bool:
    c = effect.condition
    return bool(effect.parameters) or bool(c.literals) or bool(c.numeric_constraints)


def _info(a: argparse.Namespace) -> int:
    if a.problem is None:
        if not hasattr(mymyr, "Domain"):
            raise UsageError("this build has no PDDL front end")
        f = mymyr.Domain.from_file(a.domain).formalism
    else:
        f = mymyr.Task.from_pddl(a.domain, a.problem).formalism
    out = [f"domain: {f.domain_name} ({a.domain})"]
    if a.problem is not None:
        out.append(f"problem: {f.problem_name} ({a.problem})")
    out.append("requirements: " + (" ".join(_requirement(r) for r in f.requirements) or "(none)"))
    types = [t.name for t in f.types if t.name not in ("object", "number")]
    out.append(f"types: {len(types)}" + (f" ({_names(types)})" if types else ""))
    kinds = {}
    for p in f.predicates:
        kinds.setdefault(p.kind, []).append(f"{p.name}/{p.arity}")
    out.append(
        f"predicates: {len(f.predicates)} ("
        + ", ".join(f"{len(v)} {k}" for k, v in sorted(kinds.items()))
        + ")"
        if f.predicates
        else "predicates: 0"
    )
    for k, v in sorted(kinds.items()):
        out.append(f"  {k}: {_names(v)}")
    fluent_functions = [fn for fn in f.functions if fn.kind == "fluent"]
    out.append(
        f"functions: {len(f.functions)}" + (f" ({_names([f'{fn.name}/{fn.arity} {fn.kind}' for fn in f.functions])})"
                                            if f.functions else "")
    )
    numeric = bool(fluent_functions)
    out.append(f"numeric: {'yes' if numeric else 'no'}")
    # schemas: normalization splits an action into several schemas of its name (e.g. one per branch of an `or`)
    by_action: dict[str, list[Any]] = {}
    for s in f.schemas:
        by_action.setdefault(s.name, []).append(s)
    split = sum(1 for v in by_action.values() if len(v) > 1)
    out.append(
        f"actions: {len(by_action)} in the source, {len(f.schemas)} schemas after normalization"
        + (f" ({split} split)" if split else "")
    )
    for name, schemas in by_action.items():
        arities = sorted({s.arity for s in schemas})
        line = f"  {name}/{schemas[0].original_arity}: {len(schemas)} schema" + ("s" if len(schemas) > 1 else "")
        if arities != [schemas[0].original_arity]:
            line += f" (arity {', '.join(map(str, arities))} after normalization)"
        out.append(line)
    conditional = sum(1 for s in f.schemas for e in s.effects if _is_conditional(e))
    with_conditional = sum(1 for s in f.schemas if any(_is_conditional(e) for e in s.effects))
    out.append(
        f"conditional effects: {conditional}"
        + (f" (in {with_conditional} schema{'s' if with_conditional > 1 else ''})" if conditional else "")
    )
    from_problem = sum(1 for x in f.axioms if x.from_problem)
    out.append(f"axioms: {len(f.axioms)}" + (f" ({from_problem} from the problem)" if from_problem else ""))
    if a.problem is not None:
        out.append(f"objects: {len(f.objects)}")
        out.append(f"initial atoms: {len(f.static_init) + len(f.fluent_init)} "
                   f"({len(f.static_init)} static, {len(f.fluent_init)} fluent)")
        if numeric or f.static_values or f.fluent_values:
            out.append(f"initial function values: {len(f.static_values) + len(f.fluent_values)} "
                       f"({len(f.static_values)} static, {len(f.fluent_values)} fluent)")
        out.append(f"goal: {f.goal}")
        if f.metric is not None:
            out.append(f"metric: {f.metric[0]} {f.metric[1]}")
    else:
        out.append(f"constants: {len(f.objects)}")
    print("\n".join(out))
    return EXIT_SOLVED


# ----------------------------------------------------------------------------------------------------------- main


def _one_line(e: BaseException) -> str:
    """The message of `e` on one line, without the library's own "mymyr: " prefix (the CLI prints its own)."""
    if isinstance(e, OSError) and e.strerror:
        return f"{e.strerror}: {e.filename}" if e.filename else e.strerror
    text = " ".join(str(e).split())
    for prefix in ("mymyr: ", "mymyr "):  # "mymyr: ..." and "mymyr brfs: ..."
        text = text.removeprefix(prefix)
    return text or type(e).__name__


def main(argv: Sequence[str] | None = None) -> int:
    """Runs the command line with ``argv`` (default: ``sys.argv[1:]``) and returns the exit code."""
    parser = _parser()
    a = parser.parse_args(argv)
    if a.command is None:
        parser.print_help(sys.stderr)
        return EXIT_ERROR
    try:
        return a.run(a)
    except KeyboardInterrupt:
        print("mymyr: interrupted", file=sys.stderr)
        return EXIT_INTERRUPTED
    except Exception as e:
        if a.debug:
            traceback.print_exc()
        print(f"mymyr: error: {_one_line(e)}", file=sys.stderr)
        return EXIT_ERROR


def console_main() -> NoReturn:
    """Entry point of the ``mymyr`` console script."""
    sys.exit(main())
