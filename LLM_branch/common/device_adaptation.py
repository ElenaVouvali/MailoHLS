"""Self-contained, frozen-parent plus residual-LoRA Stage-1 checkpoints.

Only the residual LoRA is optimized. The parent's token deltas remain active;
the residual must never create a second set of overlapping token deltas.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import warnings

ADAPTERS = ["mailohls_base", "device_adapt"]
KEY = "device_adapt_trainable_parameter_contract"
SCHEMA = "mailohls-device-adapt-trainables-v1"


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def weights_path(root):
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        path = Path(root) / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"Missing adapter weights: {root}")


def frozen_parent_hash(model):
    import torch
    digest = hashlib.sha256()
    count = 0
    for name, parameter in sorted(model.named_parameters()):
        if ".mailohls_base" not in name:
            continue
        if parameter.requires_grad:
            raise ValueError(f"Frozen parent parameter became trainable: {name}")
        tensor = parameter.detach().cpu().contiguous()
        digest.update(f"{name}:{tuple(tensor.shape)}:{tensor.dtype}".encode())
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
        count += 1
    if not count:
        raise ValueError("Missing frozen parent tensors")
    return digest.hexdigest()


def residual_parameter(name):
    return any(f".{kind}.device_adapt." in name for kind in ("lora_A", "lora_B"))


def isolate_parent_token_serialization(model):
    """PEFT 0.18.1 assumes every LoRA owns every auxiliary token wrapper.

    Scope the compatibility fix to this model's wrappers. A residual with no
    token rows must neither save nor load the parent's token delta under its
    own name. Do not patch the installed PEFT package or discard parent rows.
    """
    from peft.utils.other import TrainableTokensWrapper

    class ParentTokensWrapper(TrainableTokensWrapper):
        def adapter_state_dict(self, adapter_name, state_dict):
            if adapter_name not in self.token_adapter.token_indices:
                return {}
            return super().adapter_state_dict(adapter_name, state_dict)

        def adapter_state_dict_load_map(self, adapter_name):
            if adapter_name not in self.token_adapter.token_indices:
                return {}
            return super().adapter_state_dict_load_map(adapter_name)

    for module in model.modules():
        if isinstance(module, TrainableTokensWrapper):
            module.__class__ = ParentTokensWrapper


def activate(model, trainable=False):
    isolate_parent_token_serialization(model)
    model.base_model.set_adapter(ADAPTERS)
    # PEFT activation can re-enable parent gradients. Freeze AFTER activation.
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(trainable and residual_parameter(name))
    # A frozen parent must also be deterministic during residual training.
    for name, module in model.named_modules():
        if name.endswith("lora_dropout.mailohls_base"):
            module.p = 0.0


def trainables_contract(model):
    selected = {name: list(p.shape) for name, p in model.named_parameters() if p.requires_grad}
    if not selected or any(not residual_parameter(name) for name in selected):
        raise ValueError("Device adaptation must train only nonempty residual LoRA A/B tensors")
    if model.peft_config["device_adapt"].trainable_token_indices:
        raise ValueError("Residual adapter must not contain trainable token rows")
    return {"schema": SCHEMA, "active_adapters": ADAPTERS,
            "frozen_parent": True, "frozen_embeddings": True,
            "trainable_parameters": selected,
            "trainable_numel": sum(p.numel() for p in model.parameters() if p.requires_grad)}


def parent_contract(parent_dir, expected):
    parent = json.loads((Path(parent_dir) / "training_contract.json").read_text())
    if parent.get("stage") != "stage1" or parent.get("device_mode") == "device_adapt":
        raise ValueError("Device adaptation requires an unadapted Stage-1 parent")
    if parent.get("stage1_trainable_parameter_contract", {}).get("schema") != "mailohls-stage1-trainables-v1":
        raise ValueError("Parent lacks the validated Stage-1 trainables contract")
    fields = ("model", "model_revision", "tokenizer_size", "special_tokens", "special_token_ids",
              "embedding_weights_tied", "prompt_schema_version", "response_prefix_policy",
              "clock_supervision_policy", "directive_supervision_policy", "semantic_domain_policy", "objective")
    mismatches = [key for key in fields if key not in parent or parent[key] != expected.get(key)]
    if mismatches:
        raise ValueError(f"Adaptation parent contract mismatch: {mismatches}")
    return {"path": str(Path(parent_dir).resolve()),
            "training_contract_sha256": file_hash(Path(parent_dir) / "training_contract.json"),
            "adapter_sha256": file_hash(weights_path(parent_dir))}


def validate_contract(contract):
    c = contract.get(KEY, {})
    if (contract.get("stage") != "stage1" or contract.get("device_mode") != "device_adapt"
            or c.get("schema") != SCHEMA or c.get("active_adapters") != ADAPTERS
            or c.get("frozen_parent") is not True or c.get("frozen_embeddings") is not True):
        raise ValueError("Invalid device-adaptation trainables contract")
    names = c.get("trainable_parameters", {})
    if not names or any(not residual_parameter(n) for n in names):
        raise ValueError("Invalid residual trainable parameter inventory")
    target = contract.get("device_adaptation", {})
    if (not target.get("device") or set(target.get("device_capacities", {})) != {"BRAM_18K", "DSP", "FF", "LUT"}
            or any(type(v) is not int or v <= 0 for v in target["device_capacities"].values())
            or not target.get("parent", {}).get("adapter_sha256")):
        raise ValueError("Missing device-adaptation target or parent lineage")
    return target


def save_bundle_manifest(root, contract):
    validate_contract(contract)
    root = Path(root)
    files = ["training_contract.json"]
    for adapter in ADAPTERS:
        files.extend([str(weights_path(root / adapter).relative_to(root)), f"{adapter}/adapter_config.json"])
    payload = {"schema": "mailohls-device-adaptation-bundle-v1", "active_adapters": ADAPTERS,
               "files": {name: file_hash(root / name) for name in files}}
    (root / "device_adaptation.json").write_text(json.dumps(payload, indent=2) + "\n")


def load_bundle(base, root, trainable=False):
    from peft import PeftModel
    root = Path(root)
    contract = json.loads((root / "training_contract.json").read_text())
    validate_contract(contract)
    manifest = json.loads((root / "device_adaptation.json").read_text())
    required = {"training_contract.json"}
    for adapter in ADAPTERS:
        required.update({f"{adapter}/adapter_config.json", str(weights_path(root / adapter).relative_to(root))})
    if (manifest.get("schema") != "mailohls-device-adaptation-bundle-v1"
            or manifest.get("active_adapters") != ADAPTERS or set(manifest.get("files", {})) != required):
        raise ValueError("Invalid device-adaptation bundle manifest")
    for name, digest in manifest["files"].items():
        if file_hash(root / name) != digest:
            raise ValueError(f"Adaptation bundle checksum mismatch: {name}")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = PeftModel.from_pretrained(base, str(root / ADAPTERS[0]), adapter_name=ADAPTERS[0], is_trainable=False)
        isolate_parent_token_serialization(model)
        model.load_adapter(str(root / ADAPTERS[1]), adapter_name=ADAPTERS[1], is_trainable=trainable)
    if any("missing adapter keys" in str(w.message).lower() for w in caught):
        raise ValueError("Device adaptation adapter weights did not load completely")
    if model.peft_config[ADAPTERS[1]].trainable_token_indices:
        raise ValueError("Residual adapter contains overlapping token parameters")
    activate(model, trainable=True)
    if trainables_contract(model) != contract[KEY]:
        raise ValueError("Loaded residual tensors differ from the trainables contract")
    activate(model, trainable=trainable)
    return model
