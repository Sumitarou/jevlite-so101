"""scripts/eval_samples.py, scripts/contact_sheet.py and jevlite/samples.py on a fake session
(no camera, GPU or Ollama)."""

import csv
import json
from pathlib import Path

import httpx
import numpy as np
import pytest
from PIL import Image

from jevlite.api import Choice, decision_from_probs, start_judging
from jevlite.config import load_config
from jevlite.samples import load_session
from scripts import contact_sheet, eval_samples

LABELS = ["none", "edge", "center"]
LETTER = {"none": "A", "edge": "B", "center": "C"}
SHADE = {"none": 10, "edge": 120, "center": 240}  # pixel value encodes the true label


def make_session(root: Path, name="s1", counts=(2, 2, 3)) -> Path:
    d = root / "samples" / "private" / name
    rows = []
    for label, n in zip(LABELS, counts, strict=True):
        (d / label).mkdir(parents=True)
        for i in range(1, n + 1):
            f = f"{label}/{i:05d}.jpg"
            Image.new("RGB", (64, 48), (SHADE[label],) * 3).save(d / f)
            rows.append({"file": f, "label": label, "teacher_choice": LETTER[label]})
    with (d / "metadata.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["file", "label", "teacher_choice", "seq"])
        w.writeheader()
        for k, r in enumerate(rows):
            w.writerow({**r, "seq": k + 1})
    return d


class ShadeJudge:
    """Predicts from the image brightness; `wrong` labels are mapped to CONTINUE on purpose."""

    def __init__(self, wrong=()):
        self.wrong = set(wrong)
        self.calls = 0

    def judge(self, image, t_capture=None):
        self.calls += 1
        v = int(image.mean())
        label = min(SHADE, key=lambda k: abs(SHADE[k] - v))
        choice = {"none": "CONTINUE", "edge": "SLOW", "center": "STOP"}[label]
        if label in self.wrong:
            choice = "CONTINUE"
        probs = {c.name: (0.8 if c.name == choice else 0.1) for c in Choice}
        return decision_from_probs(probs, "fake", start_judging(t_capture))


@pytest.fixture
def repo(tmp_path):
    make_session(tmp_path)
    return tmp_path


def run_eval(repo, argv, make_judge, msgs=None, **kw):
    msgs = [] if msgs is None else msgs
    rc = eval_samples.main(
        ["--session", "s1", *argv],
        make_judge=make_judge,
        repo_root=repo,
        check_ignored=lambda p: True,
        out=msgs.append,
        **kw,
    )
    return rc, repo / "outputs" / "eval" / "s1", msgs


# --- session loading -------------------------------------------------------------------


def test_load_session_reads_metadata_in_label_order(repo):
    s = load_session(repo / "samples" / "private", "s1", LABELS)
    assert [x.file for x in s.samples][:3] == ["none/00001.jpg", "none/00002.jpg", "edge/00001.jpg"]
    assert s.counts() == {"none": 2, "edge": 2, "center": 3}
    assert s.problems == []


def test_load_session_reports_missing_and_moved_files(repo):
    d = repo / "samples" / "private" / "s1"
    (d / "edge" / "00002.jpg").unlink()
    with (d / "metadata.csv").open("a", newline="") as fh:
        fh.write("center/00001.jpg,edge,B,99\n")  # metadata says edge, folder says center
    s = load_session(repo / "samples" / "private", "s1", LABELS)
    assert any("missing" in p for p in s.problems)
    assert any("folder 'center' but metadata label 'edge'" in p for p in s.problems)
    assert len(s.samples) == 7  # 7 rows - 1 missing + 1 extra


def test_missing_session_is_a_clear_error(repo):
    with pytest.raises(FileNotFoundError, match="session name"):
        load_session(repo / "samples" / "private", "nope", LABELS)


# --- evaluation -----------------------------------------------------------------------------


def test_perfect_judge_scores_one_and_writes_all_outputs(repo):
    judges = {}

    def make(model, prompt):
        judges[(model, prompt)] = ShadeJudge()
        return judges[(model, prompt)]

    rc, out, _ = run_eval(repo, ["--models", "v0", "--prompts", "baseline"], make)
    assert rc == 0
    summary = json.loads((out / "summary.json").read_text())
    assert list(summary["results"]["v0 | baseline"]) == ["all"]  # no subsets for this session
    r = summary["results"]["v0 | baseline"]["all"]
    assert r["accuracy"] == 1.0 and r["n"] == 7
    assert r["confusion"]["matrix"] == [[2, 0, 0], [0, 2, 0], [0, 0, 3]]
    assert r["latency_ms"]["p50"] is not None and r["latency_ms"]["p95"] is not None
    assert judges[("v0", "baseline")].calls == 7 + 1  # 1 warm-up call, not recorded
    for name in ("predictions.csv", "summary.json", "summary.md", "prompts.md"):
        assert (out / name).exists()


def test_confusion_matrix_rows_are_truth(repo):
    rc, out, _ = run_eval(
        repo, ["--models", "gemma3:4b", "--prompts", "def-v1"], lambda m, p: ShadeJudge({"edge"})
    )
    r = json.loads((out / "summary.json").read_text())["results"]["gemma3:4b | def-v1"]["all"]
    assert r["confusion"]["matrix"] == [[2, 0, 0], [2, 0, 0], [0, 0, 3]]  # edge -> CONTINUE
    assert r["accuracy"] == pytest.approx(5 / 7, abs=1e-4)
    assert r["recall"] == {"CONTINUE": 1.0, "SLOW": 0.0, "STOP": 1.0}


def test_predictions_csv_has_one_row_per_image_model_prompt(repo):
    rc, out, _ = run_eval(
        repo,
        ["--models", "v0,gemma3:4b", "--prompts", "baseline,def-v1"],
        lambda m, p: ShadeJudge(),
    )
    with (out / "predictions.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 7 * 2 * 2
    assert list(rows[0]) == eval_samples.PRED_COLUMNS
    r = rows[0]
    assert r["truth_letter"] == "A" and r["pred"] == "CONTINUE" and r["correct"] == "1"
    assert float(r["p_CONTINUE"]) == pytest.approx(0.8)


def test_default_models_and_prompts_exclude_the_27b(repo):
    seen = []

    def make(model, prompt):
        seen.append((model, prompt))
        return ShadeJudge()

    run_eval(repo, [], make)
    models = {m for m, _ in seen}
    assert models == {"v0", "qwen2.5vl:7b", "gemma3:4b"}
    assert {p for _, p in seen} == {"def-v2-ref"}
    run_eval(repo, ["--models", "qwen3.5:27b", "--prompts", "baseline"], make)
    assert ("qwen3.5:27b", "baseline") in seen


def test_unknown_prompt_is_rejected(repo):
    rc, _, msgs = run_eval(repo, ["--prompts", "nope"], lambda m, p: ShadeJudge())
    assert rc == 2 and "unknown" in msgs[0]


def test_a_skipped_model_does_not_stop_the_others(repo):
    def make(model, prompt):
        if model == "qwen2.5vl:7b":
            raise eval_samples.Skip("Ollama not reachable at http://x: ConnectError")
        return ShadeJudge()

    rc, out, msgs = run_eval(repo, ["--models", "v0,qwen2.5vl:7b", "--prompts", "baseline"], make)
    s = json.loads((out / "summary.json").read_text())
    assert rc == 0
    assert "v0 | baseline" in s["results"]
    assert s["skipped"] == {
        "qwen2.5vl:7b | baseline": "Ollama not reachable at http://x: ConnectError"
    }
    assert "Skipped" in (out / "summary.md").read_text()


def test_refuses_an_output_dir_git_would_track(repo):
    rc = eval_samples.main(
        ["--session", "s1"], make_judge=lambda m, p: ShadeJudge(), repo_root=repo,
        check_ignored=lambda p: False, out=lambda m: None,
    )  # fmt: skip
    assert rc == 2
    assert not (repo / "outputs").exists()


# --- real factories with fake Ollama / no SigLIP -----------------------------------------


def tags(*names):
    def get(url, timeout):
        return httpx.Response(
            200, json={"models": [{"name": n, "digest": f"d-{n}"} for n in names]},
            request=httpx.Request("GET", url),
        )  # fmt: skip

    return get


def factories(http_get):
    cfg = load_config("eval.yaml")
    return eval_samples.real_factories(
        cfg, load_config("teacher.yaml"), load_config("choices.yaml"), http_get
    )


def test_ollama_down_skips_teachers_with_the_reason():
    def down(url, timeout):
        raise httpx.ConnectError("connection refused")

    make, _ = factories(down)
    with pytest.raises(eval_samples.Skip, match="not reachable.*ConnectError"):
        make("gemma3:4b", "baseline")


def test_missing_ollama_model_is_skipped():
    make, _ = factories(tags("gemma3:4b"))
    with pytest.raises(eval_samples.Skip, match="qwen2.5vl:7b is not installed"):
        make("qwen2.5vl:7b", "baseline")


def test_teacher_gets_the_chosen_prompt_and_digest_is_recorded():
    make, info = factories(tags("gemma3:4b"))
    t_base = make("gemma3:4b", "baseline")
    t_def = make("gemma3:4b", "def-v1")
    assert t_base.cfg["prompt"] == load_config("teacher.yaml")["prompt"]
    assert "WHITE surface" in t_def.cfg["prompt"] and "GREEN" in t_def.cfg["prompt"]
    assert info["ollama"]["gemma3:4b"] == "d-gemma3:4b"


def test_v0_label_variants():
    cfg, choices = load_config("eval.yaml"), load_config("choices.yaml")
    assert eval_samples.v0_choices("baseline", cfg, choices) == choices
    v = eval_samples.v0_choices("def-v1", cfg, choices)
    assert [c["name"] for c in v["choices"]] == ["CONTINUE", "SLOW", "STOP"]
    assert any("green" in p for c in v["choices"] for p in c["prompts"])


def test_prompts_markdown_has_full_texts():
    cfg = load_config("eval.yaml")
    md = eval_samples.prompts_markdown(
        ["baseline", "def-v1"], cfg, load_config("teacher.yaml"), load_config("choices.yaml")
    )
    assert load_config("teacher.yaml")["prompt"].strip().splitlines()[0] in md
    assert cfg["teacher_prompts"]["def-v1"].strip().splitlines()[-1] in md
    assert "v0 text labels" in md


def test_label_mapping_matches_the_labeling_doc():
    m = load_config("eval.yaml")["label_to_choice"]
    assert m == {"none": "CONTINUE", "edge": "SLOW", "center": "STOP"}
    capture = load_config("capture.yaml")["labels"]
    letters = {Choice[c]: capture[label]["teacher"] for label, c in m.items()}
    assert letters == eval_samples.LETTERS  # A/B/C agree with capture.yaml


# --- contact sheet -------------------------------------------------------------------------


def test_contact_sheet_groups_by_label(repo):
    msgs = []
    rc = contact_sheet.main(["--session", "s1"], repo_root=repo, check_ignored=lambda p: True,
                            out=msgs.append)  # fmt: skip
    assert rc == 0
    path = repo / "outputs" / "contact_s1.jpg"
    img = np.asarray(Image.open(path))
    cfg = load_config("eval.yaml")["contact_sheet"]
    tw, cols = cfg["thumb_width"], cfg["columns"]
    assert img.shape[1] == cols * (tw + contact_sheet.GAP) + contact_sheet.GAP
    th = round(48 * tw / 64)
    expected_h = contact_sheet.HEADER_H + 3 * (
        contact_sheet.HEADER_H + th + contact_sheet.CAPTION_H + contact_sheet.GAP
    )
    assert img.shape[0] == expected_h  # one row per label (2, 2, 3 images < 8 columns)
    assert "7 images" in msgs[-1]


def test_contact_sheet_refuses_tracked_output(repo):
    rc = contact_sheet.main(["--session", "s1"], repo_root=repo, check_ignored=lambda p: False,
                            out=lambda m: None)  # fmt: skip
    assert rc == 2
    assert not (repo / "outputs").exists()


def test_repo_eval_outputs_are_git_ignored():
    from scripts.capture_samples import REPO_ROOT, is_git_ignored

    cfg = load_config("eval.yaml")
    assert is_git_ignored(REPO_ROOT / cfg["out_root"] / "20261011-desk" / "summary.json")
    assert is_git_ignored(REPO_ROOT / cfg["contact_sheet"]["out_dir"] / "contact_x.jpg")


def test_empty_session_does_not_crash(tmp_path):
    make_session(tmp_path, counts=(0, 0, 0))
    rc, out, _ = run_eval(tmp_path, ["--models", "v0", "--prompts", "baseline"],
                          lambda m, p: ShadeJudge())  # fmt: skip
    assert rc == 0
    s = json.loads((out / "summary.json").read_text())
    assert s["results"]["v0 | baseline"]["all"]["accuracy"] is None
    assert "| v0 | baseline | 全体 | 0 | - |" in (out / "summary.md").read_text()


# --- boundary pairs (subsets) --------------------------------------------------------------


def cfg_with_subsets(spec):
    cfg = load_config("eval.yaml")
    return {**cfg, "subsets": {"s1": spec}}


def test_resolve_subsets_expands_ranges_and_reports_missing(repo):
    from jevlite.samples import resolve_subsets

    s = load_session(repo / "samples" / "private", "s1", LABELS)
    subsets, problems = resolve_subsets(
        {"boundary": ["edge/00001-00002", "center/00003", "center/00009"]}, s
    )
    assert subsets == {"boundary": {"edge/00001.jpg", "edge/00002.jpg", "center/00003.jpg"}}
    assert problems == ["subset boundary: center/00009.jpg is not in session s1"]


@pytest.mark.parametrize("bad", ["edge", "edge/1-x", "edge/00001-"])
def test_bad_subset_entries_are_errors(repo, bad):
    from jevlite.samples import resolve_subsets

    s = load_session(repo / "samples" / "private", "s1", LABELS)
    with pytest.raises(ValueError):
        resolve_subsets({"boundary": [bad]}, s)


def test_reserved_subset_names_are_rejected(repo):
    from jevlite.samples import resolve_subsets

    s = load_session(repo / "samples" / "private", "s1", LABELS)
    with pytest.raises(ValueError, match="reserved"):
        resolve_subsets({"other": ["edge/00001"]}, s)


def test_results_are_split_into_all_boundary_and_other(repo):
    # edge 1-2 and center 1-2 are the pairs; the judge gets every edge wrong
    cfg = cfg_with_subsets({"boundary": ["edge/00001-00002", "center/00001-00002"]})
    rc, out, msgs = run_eval(repo, ["--models", "v0", "--prompts", "baseline"],
                             lambda m, p: ShadeJudge({"edge"}), eval_cfg=cfg)  # fmt: skip
    assert rc == 0
    s = json.loads((out / "summary.json").read_text())
    groups = s["results"]["v0 | baseline"]
    assert list(groups) == ["all", "boundary", "other"]
    assert (groups["all"]["n"], groups["boundary"]["n"], groups["other"]["n"]) == (7, 4, 3)
    assert groups["boundary"]["accuracy"] == 0.5  # 2 edge wrong, 2 center right
    assert groups["other"]["accuracy"] == 1.0  # 2 none + 1 center
    assert groups["all"]["accuracy"] == pytest.approx(5 / 7, abs=1e-4)
    assert s["subsets"]["boundary"] == ["center/00001.jpg", "center/00002.jpg",
                                        "edge/00001.jpg", "edge/00002.jpg"]  # fmt: skip
    md = (out / "summary.md").read_text()
    for title in ("全体", "boundary", "それ以外"):
        assert f"| v0 | baseline | {title} |" in md
    assert any("全体" in m and "boundary" in m and "それ以外" in m for m in msgs)
    with (out / "predictions.csv").open() as fh:
        rows = {r["file"]: r["subsets"] for r in csv.DictReader(fh)}
    assert rows["edge/00001.jpg"] == "boundary" and rows["none/00001.jpg"] == ""


def test_subset_entries_missing_from_the_session_are_reported(repo):
    cfg = cfg_with_subsets({"boundary": ["edge/00001-00005"]})
    rc, out, msgs = run_eval(repo, ["--models", "v0", "--prompts", "baseline"],
                             lambda m, p: ShadeJudge(), eval_cfg=cfg)  # fmt: skip
    s = json.loads((out / "summary.json").read_text())
    assert s["results"]["v0 | baseline"]["boundary"]["n"] == 2
    assert sum("not in session" in p for p in s["problems"]) == 3
    assert any("not in session" in m for m in msgs)


def test_repo_boundary_pairs_are_the_agreed_18(tmp_path):
    """configs/eval.yaml marks edge 1-9 and center 1-9 of 20261011-desk."""
    from jevlite.samples import resolve_subsets

    make_session(tmp_path, name="20261011-desk", counts=(9, 9, 19))
    s = load_session(tmp_path / "samples" / "private", "20261011-desk", LABELS)
    spec = load_config("eval.yaml")["subsets"]["20261011-desk"]
    subsets, problems = resolve_subsets(spec, s)
    assert problems == []
    assert sorted(subsets["boundary"]) == sorted(
        [f"edge/{i:05d}.jpg" for i in range(1, 10)] + [f"center/{i:05d}.jpg" for i in range(1, 10)]
    )


def test_contact_sheet_tags_subset_images(repo):
    from jevlite.samples import resolve_subsets

    s = load_session(repo / "samples" / "private", "s1", LABELS)
    subsets, _ = resolve_subsets({"boundary": ["edge/00001"]}, s)
    cfg = load_config("eval.yaml")["contact_sheet"]
    plain = np.asarray(contact_sheet.make_sheet(s, LABELS, cfg))
    tagged = np.asarray(contact_sheet.make_sheet(s, LABELS, cfg, subsets))
    assert plain.shape == tagged.shape
    assert not np.array_equal(plain, tagged)  # the caption of edge/00001 changed


# --- def-v2: attribute mode, reference image, extras ----------------------------------


class AttrJudge(ShadeJudge):
    """ShadeJudge that also exposes last_extra like OllamaAttrTeacher."""

    def judge(self, image, t_capture=None):
        d = super().judge(image, t_capture)
        self.last_extra = {
            "reason": f"saw {d.choice.name}", "object_present": d.choice.name != "CONTINUE",
            "touches_white": d.choice.name == "STOP", "contradiction": False,
        }  # fmt: skip
        return d


def test_extras_are_written_to_predictions(repo):
    rc, out, _ = run_eval(repo, ["--models", "gemma3:4b", "--prompts", "def-v2-attr"],
                          lambda m, p: AttrJudge())  # fmt: skip
    with (out / "predictions.csv").open() as fh:
        rows = {r["file"]: r for r in csv.DictReader(fh)}
    r = rows["edge/00001.jpg"]
    assert (r["reason"], r["object_present"], r["touches_white"], r["contradiction"]) == (
        "saw SLOW", "1", "0", "0"
    )  # fmt: skip
    assert rows["none/00001.jpg"]["object_present"] == "0"


def test_judges_without_extras_leave_the_columns_empty(repo):
    rc, out, _ = run_eval(
        repo, ["--models", "v0", "--prompts", "baseline"], lambda m, p: ShadeJudge()
    )
    with (out / "predictions.csv").open() as fh:
        r = next(csv.DictReader(fh))
    assert (r["reason"], r["object_present"], r["touches_white"]) == ("", "", "")


def test_stop_as_continue_counts_center_judged_continue(repo):
    rc, out, _ = run_eval(repo, ["--models", "v0", "--prompts", "baseline"],
                          lambda m, p: ShadeJudge({"center"}))  # fmt: skip
    s = json.loads((out / "summary.json").read_text())["results"]["v0 | baseline"]["all"]
    assert s["stop_as_continue"] == 3
    assert "C→A" in (out / "summary.md").read_text()


def test_reference_image_is_excluded_from_every_run(repo):
    seen = []

    def make(model, prompt):
        seen.append((model, prompt))
        return AttrJudge()

    rc, out, msgs = run_eval(
        repo, ["--models", "gemma3:4b", "--prompts", "def-v2-attr,def-v2-ref"], make
    )
    assert rc == 0
    s = json.loads((out / "summary.json").read_text())
    assert s["excluded"] == ["none/00001.jpg"]
    assert s["images"] == 6  # 7 - the reference
    for key in ("gemma3:4b | def-v2-attr", "gemma3:4b | def-v2-ref"):
        assert s["results"][key]["all"]["n"] == 6
    with (out / "predictions.csv").open() as fh:
        assert "none/00001.jpg" not in {r["file"] for r in csv.DictReader(fh)}
    assert "Excluded: none/00001.jpg" in (out / "summary.md").read_text()


def test_missing_reference_image_is_an_error(repo):
    (repo / "samples" / "private" / "s1" / "none" / "00001.jpg").unlink()
    rc, _, msgs = run_eval(repo, ["--models", "gemma3:4b", "--prompts", "def-v2-ref"],
                           lambda m, p: AttrJudge())  # fmt: skip
    assert rc == 2 and "reference image" in msgs[-1]


def test_real_factory_builds_attr_teacher_with_reference_and_timeout():
    from jevlite.teacher_ollama import OllamaAttrTeacher, OllamaTeacher

    ref = np.zeros((48, 64, 3), np.uint8)
    cfg = load_config("eval.yaml")
    make, _ = eval_samples.real_factories(
        cfg, load_config("teacher.yaml"), load_config("choices.yaml"),
        tags("gemma3:4b", "qwen3.5:27b"), references={"def-v2-ref": ref},
    )  # fmt: skip
    t_attr = make("gemma3:4b", "def-v2-attr")
    t_ref = make("gemma3:4b", "def-v2-ref")
    t_base = make("gemma3:4b", "baseline")
    assert isinstance(t_attr, OllamaAttrTeacher) and t_attr.reference is None
    assert isinstance(t_ref, OllamaAttrTeacher) and t_ref.reference is ref
    assert type(t_base) is OllamaTeacher
    assert "FIRST image is a reference" in t_ref.cfg["prompt"]
    big = make("qwen3.5:27b", "def-v2-ref")
    assert big.cfg["timeout_s"] == 900 and big.client.timeout.read == 900
    assert big.build_request(np.zeros((8, 8, 3), np.uint8))["think"] is False


def test_v0_is_skipped_for_prompts_without_v0_labels():
    make, _ = factories(tags())
    with pytest.raises(eval_samples.Skip, match="no text labels for prompt def-v2-attr"):
        make("v0", "def-v2-attr")


def test_def_v2_prompts_treat_the_table_as_background():
    p = load_config("eval.yaml")["teacher_prompts"]
    for name in ("def-v2-attr", "def-v2-ref"):
        text = p[name].lower()
        assert "paper" not in text
        assert "part of the table" in text and "never objects" in text
        assert text.index("reason") < text.index("object_present") < text.index("touches_white")


def test_tag_writes_to_a_subdirectory_and_keeps_earlier_runs(repo):
    run_eval(repo, ["--models", "v0", "--prompts", "baseline"], lambda m, p: ShadeJudge())
    rc, _, _ = run_eval(repo, ["--models", "v0", "--prompts", "def-v1", "--tag", "run2"],
                        lambda m, p: ShadeJudge())  # fmt: skip
    base = repo / "outputs" / "eval" / "s1"
    assert rc == 0
    assert "v0 | baseline" in json.loads((base / "summary.json").read_text())["results"]
    assert "v0 | def-v1" in json.loads((base / "run2" / "summary.json").read_text())["results"]
    rc, _, _ = run_eval(repo, ["--tag", "../x"], lambda m, p: ShadeJudge())
    assert rc == 2


def test_def_v2_1_ref_only_changes_the_shadow_rule():
    cfg = load_config("eval.yaml")
    p = cfg["teacher_prompts"]
    v2, v21 = p["def-v2-ref"].splitlines(), p["def-v2.1-ref"].splitlines()
    changed = [ln for ln in v21 if ln not in v2]
    assert changed and all("shadow" in ln or "safer" in ln for ln in changed)
    assert "treat it as an object" in p["def-v2.1-ref"]
    assert cfg["teacher_modes"]["def-v2.1-ref"] == cfg["teacher_modes"]["def-v2-ref"]


def test_teacher_default_is_def_v2_ref():
    cfg = load_config("eval.yaml")
    assert cfg["teacher_default"] == {
        "prompt": "def-v2-ref", "model_offline": "qwen3.5:27b", "model_quick": "qwen2.5vl:7b"
    }  # fmt: skip
    assert cfg["default_prompts"] == ["def-v2-ref"]
    assert cfg["teacher_default"]["prompt"] in cfg["teacher_prompts"]
