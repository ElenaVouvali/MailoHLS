"""Explicit TCAD train/inference entry point; legacy entry points are unchanged.

Default is a policy preflight, not training or GPU/model loading. Runtime binding
is scoped to the imported workflow module and never edits checkpoint contracts.
"""
from __future__ import annotations

import argparse
from contextvars import ContextVar
from functools import wraps
import importlib
import inspect
import json
from pathlib import Path
import sys

from . import directive_domains as legacy, mailohls_contract as mc
from .directive_domains import tcad

ACTIVE = ContextVar("tcad_source_actions", default=None)


class ContractView:
    def __getattr__(self, name):
        return getattr(mc, name)

    filter_semantic_candidates = staticmethod(tcad.filter_semantic_candidates)

    @staticmethod
    def validate_directive_assignments(assignments):
        mc.validate_directive_assignments(assignments)
        if ACTIVE.get() is not None:
            tcad.validate_assignments(assignments, ACTIVE.get())


def install_policy(
    module,
    application_dataset_dir,
    *,
    workflow,
    trained_policy=None,
    allow_domain_shift=False
):
    """Adapt existing per-source constrained decoders without changing their files."""
    shifted = workflow == "infer" and trained_policy != tcad.SOURCE_DOMAIN_POLICY
    if shifted and not allow_domain_shift:
        raise ValueError("tcad evaluation of a historical checkpoint requires --allow-domain-shift")
    # Inference keeps original contract/registry authentication. Domain expansion
    # occurs only inside the decoder, after those checks have passed.
    if workflow == "train" or trained_policy == tcad.SOURCE_DOMAIN_POLICY:
        module.directive_domains = tcad
    module.mailohls_contract = ContractView()
    if workflow == "train":
        original_loader = module.load_directive_domain_registry
        def load_registry(path):
            payload = json.loads(Path(path).read_text())
            if payload.get("generation_policy") != tcad.SOURCE_DOMAIN_POLICY:
                raise ValueError("tcad training requires a tcad registry, not a relabeled legacy registry")
            result = original_loader(path)
            for kernel, values in result.items():
                actions = legacy.load_source_actions(kernel, application_dataset_dir)
                expected = tcad.domains_from_actions(actions)
                if values != expected:
                    raise ValueError("Registry does not match exact source-derived tcad values")
                result[kernel] = expected
            return result
        module.load_directive_domain_registry = load_registry
    count = 0
    for name, function in list(vars(module).items()):
        if not inspect.isfunction(function) or function.__module__ != module.__name__:
            continue
        signature = inspect.signature(function)
        if not {"source_text", "kernel_name", "directive_domain_registry"} <= set(signature.parameters):
            continue
        def bind(original, signature):
            @wraps(original)
            def wrapped(*args, **kwargs):
                bound = signature.bind(*args, **kwargs)
                bound.apply_defaults()
                kernel, source = bound.arguments["kernel_name"], bound.arguments["source_text"]
                if not kernel or not source:
                    raise ValueError("tcad decoding/training requires explicit kernel and source metadata")
                domains = tcad.source_site_domains(kernel, source, application_dataset_dir)
                registry = dict(bound.arguments["directive_domain_registry"] or {})
                registry[legacy.normalize_kernel_name(kernel)] = domains
                bound.arguments["directive_domain_registry"] = registry
                token = ACTIVE.set(domains.actions)
                try:
                    result = original(*bound.args, **bound.kwargs)
                    if isinstance(result, dict):
                        result.update(evaluation_domain_policy=tcad.SOURCE_DOMAIN_POLICY,
                            checkpoint_domain_policy=trained_policy, domain_shift_evaluation=shifted)
                    return result
                finally:
                    ACTIVE.reset(token)
            return wrapped
        setattr(module, name, bind(function, signature))
        count += 1
    if not count:
        raise ValueError("Workflow has no compatible source-aware constrained decoder")
    # Preserve policy disclosure on outer inference records too.
    if workflow == "infer" and hasattr(module, "predict_case"):
        original = module.predict_case
        def predict(*args, **kwargs):
            result = original(*args, **kwargs)
            result.update(evaluation_domain_policy=tcad.SOURCE_DOMAIN_POLICY,
                checkpoint_domain_policy=trained_policy, domain_shift_evaluation=shifted)
            return result
        module.predict_case = predict
    return count


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow", choices=["train", "infer"], required=True)
    parser.add_argument("--domain-policy", choices=["tcad"], default="tcad")
    parser.add_argument("--allow-domain-shift", action="store_true")
    parser.add_argument("--execute", action="store_true", help="Explicitly invoke the existing model workflow")
    parser.add_argument("--policy-record", help="Required provenance output when executing")
    parser.add_argument(
        "--domain-audit-report",
        help="Required zero-exclusion dataset audit for executing tcad training"
    )
    args, forwarded = parser.parse_known_args(argv)
    if forwarded[:1] == ["--"]:
        forwarded = forwarded[1:]
    legacy_parser = argparse.ArgumentParser(add_help=False)
    legacy_parser.add_argument("--adapter_dir")
    legacy_parser.add_argument("--application_dataset_dir", default=str(legacy.DEFAULT_APPLICATION_DATASET_DIR))
    legacy_parser.add_argument("--dataset")
    known, _ = legacy_parser.parse_known_args(forwarded)
    trained_policy, checkpoint = None, None
    if args.workflow == "infer":
        if not known.adapter_dir:
            parser.error("infer preflight requires the original --adapter_dir")
        path = Path(known.adapter_dir) / "training_contract.json"
        checkpoint = json.loads(path.read_text())
        trained_policy = checkpoint.get("directive_domain_registry_policy")
    record = {"runtime_domain_policy": tcad.SOURCE_DOMAIN_POLICY, "checkpoint_domain_policy": trained_policy,
        "domain_shift_evaluation": args.workflow == "infer" and trained_policy != tcad.SOURCE_DOMAIN_POLICY,
        "workflow": args.workflow, "uncapped_factor_policy": True, "source_dimension_aware": True,
        "execute_requested": args.execute, "checkpoint_contract_unmodified": True}
    if args.policy_record:
        target = Path(args.policy_record)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, sort_keys=True))
    if not args.execute:
        return 0
    if not args.policy_record:
        parser.error("--execute requires --policy-record so evaluation/training policy cannot be implicit")
    if record["domain_shift_evaluation"] and not args.allow_domain_shift:
        parser.error("historical checkpoint evaluation requires explicit --allow-domain-shift")
    if args.workflow == "train":
        if not args.domain_audit_report or not known.dataset:
            parser.error(
                "tcad training requires --dataset and --domain-audit-report; labels must not be silently dropped"
            )
        import hashlib
        audit = json.loads(Path(args.domain_audit_report).read_text())
        digest = hashlib.sha256(Path(known.dataset).read_bytes()).hexdigest()
        coverage = audit.get("datasets", {}).get("sft", {})
        if coverage.get("sha256") != digest or coverage.get("tcad_invalid", 1) or coverage.get("unverifiable", 1):
            parser.error("dataset is unaudited or contains tcad-invalid/unverifiable labels; training is blocked")
    module = importlib.import_module("LLM_branch.train.train_SFT_xattn_new" if args.workflow == "train"
        else "LLM_branch.inference.eval_stage1_stage2_stage3")
    install_policy(module, known.application_dataset_dir, workflow=args.workflow,
        trained_policy=trained_policy, allow_domain_shift=args.allow_domain_shift)
    sys.argv = [module.__file__, *forwarded]
    module.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
