"""Small CPU/stdlib regression tests. Does not need a GPU or real dataset."""
from __future__ import annotations

import ast
import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import freeze_native_contract as freezer
import native_contract as contract
from model_api import clean_graph


class FakeGraph:
    def __init__(self):
        self.stores = [{"name": "hidden-label", "original_center": [1, 2, 3], "native_mdn_prepared": True},
                       {"orig_pos": [9, 9, 9], "rmsd": 0.0, "pos": [1, 1, 1], "x": [4],
                        "reference_path": "do-not-read", "pose_uid": "label"}]

    def clone(self):
        return copy.deepcopy(self)


class NativeContractTests(unittest.TestCase):
    def test_all_sources_parse_without_importing_torch(self):
        for path in Path(__file__).parent.glob("*.py"):
            ast.parse(path.read_text(), filename=str(path), feature_version=(3, 10))

    def test_label_stripping_does_not_mutate_graph(self):
        graph = FakeGraph()
        cleaned = clean_graph(graph)
        self.assertEqual(cleaned.stores[0], {})
        self.assertEqual(cleaned.stores[1], {"pos": [1, 1, 1], "x": [4]})
        self.assertEqual(graph.stores[1]["rmsd"], 0.0)

    def test_one_attempt_receipt_is_not_overwritable(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "attempt.json"
            contract.exclusive_json(path, {"status": "STARTED"})
            with self.assertRaises(FileExistsError):
                contract.exclusive_json(path, {"status": "RETRY"})
            self.assertEqual(contract.read_json(path)["status"], "STARTED")

    def test_release_escape_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(RuntimeError):
                contract.release_path(temporary, "../outside")

    def test_duplicate_members_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "train.txt"
            path.write_text("A\nA\n")
            with self.assertRaises(RuntimeError):
                contract.member_names(path)

    def test_scaler_cannot_include_validation(self):
        scaler = self.scaler(["A"], "sha")
        contract.validate_scaler(scaler, ["A"], "sha")
        scaler["fit_policy"]["validation_surfaces_used"] = True
        with self.assertRaises(RuntimeError):
            contract.validate_scaler(scaler, ["A"], "sha")

    def test_scaler_must_cover_exact_train_id_set(self):
        scaler = self.scaler(["A", "B"], "sha")
        with self.assertRaises(RuntimeError):
            contract.validate_scaler(scaler, ["A", "C"], "sha")

    def test_fixed_recipe_and_two_stage_boundary(self):
        r = contract.RECIPE
        self.assertEqual((r["max_epochs"], r["min_epochs"], r["patience"]), (20, 3, 4))
        self.assertEqual((r["n_gaussians"], r["scorer_hidden_dim"], r["mdn_dropout"]), (20, 192, 0.1))
        self.assertEqual((r["batch_size"], r["lr"], r["weight_decay"]), (4, 2e-4, 1e-5))
        self.assertFalse(r["native_k8_evaluation"])
        self.assertFalse(r["test_opened"])
        self.assertIn("already_centered", contract.COORDINATE_POLICY["receptor.nucleic_anchor_pos"])

    @staticmethod
    def scaler(names, train_sha):
        return {"fit_policy": {"fit_split": "training_only", "fit_population": "new_L2_NA_train_only",
                "protein_surfaces_used": False, "validation_surfaces_used": False, "test_surfaces_used": False},
                "source": {"na": {"ply_count": len(names), "train_manifest_sha256": train_sha,
                                  "train_surface_files": [{"complex_id": name} for name in names]}}}

    def fixture(self, base):
        release = base / "release"
        source = base / "old_code" / "source_snapshot"
        root = base / "new_run"
        files = []

        def write(path, content):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
            return {"path": str(path), "sha256": contract.sha(path), "bytes": path.stat().st_size}

        def asset(relative, content):
            value = write(release / relative, content)
            files.append(dict(value, path=relative))
            return value

        splits, tables, audits = {}, {}, {}
        for split, names in (("train", ["A", "B"]), ("val", ["C"])):
            split_file = asset("splits/" + split + ".txt", "\n".join(names) + "\n")
            splits[split] = {"path": "splits/" + split + ".txt", "sha256": split_file["sha256"], "count": len(names)}
            table = asset("membership/" + split + ".tsv", "complex_id\tparent_pdb\tsplit_group_id\n" +
                          "".join(f"{name}\tPDB_{name}\tGROUP_{name}\n" for name in names))
            tables[split] = {"path": "membership/" + split + ".tsv", "sha256": table["sha256"]}
            cache = "cache/" + split
            cached = [asset(cache + "/" + n, "fixture, not a loadable graph") for n in ("heterographs.pkl", "rdkit_ligands.pkl")]
            audits[split] = {"actual": len(names), "expected": len(names), "inference_list_actual": len(names),
                "raw_graph_invalid": 0, "loader_invalid": 0, "ligand_matching_fallback_count": 0,
                "automatic_rebuild_disabled": True, "cache_directory": cache,
                "files_sha256": {str(Path(x["path"]).relative_to(release)): x["sha256"] for x in cached}}
            for name in names:
                asset(f"data/trainval/{name}/{name}_ligand.sdf", "fixture, not a loadable molecule")
        asset("scalers/l2.json", json.dumps(self.scaler(["A", "B"], splits["train"]["sha256"])))
        asset("audit/GRAPH_LOAD_AUDIT.json", json.dumps({"status": "verified", "raw_checks_before_sanitation": True, "splits": audits}))
        runtime = {"files": files, "counts": {"train": 2, "val": 1}, "splits": splits,
                   "membership": {"columns": {"id": "complex_id", "pdb": "parent_pdb"}, "tables": tables},
                   "layout": {"data_dir": "data/trainval", "surface_dir": "surfaces/trainval", "scaler": "scalers/l2.json"}}
        write(release / "PORTABLE_RUNTIME.json", json.dumps(runtime))
        source_manifest = {}
        for relative in ("models/surface_score_model_v3.py", "models/surface_interaction_backbone_v2.py",
                         "models/surfna_v2_scorer_components.py", "models/surfna_v2_transfer.py",
                         "utils/scorer_factory_v2.py", "utils/scorer_config_v2.py"):
            record = write(source / relative, "# architecture hash fixture\n")
            source_manifest[relative] = {"sha256": record["sha256"]}
        write(source.parent / "source_snapshot_manifest.json", json.dumps(source_manifest))
        protein_config = write(base / "weights/config.yml", "ns: 48\n")
        diffusion = write(base / "weights/diff.pt", "diffusion hash fixture")
        mdn = write(base / "weights/mdn.pt", "protein MDN hash fixture")
        for name in contract.TABLE_NAMES:
            write(base / "tables" / name, "precomputed hash fixture")
        parent = {"dataset": {"release_root": str(release),
                  "surface_scaler": {"path": str(release / "scalers/l2.json"), "sha256": contract.sha(release / "scalers/l2.json")}}}
        for split in ("train", "val"):
            parent["dataset"][split] = dict(splits[split], path=str(release / splits[split]["path"]))
        write(root / "contract.json", json.dumps(parent))
        args = SimpleNamespace(root=str(root), release_root=str(release), code_root=str(source),
            protein_model_config=protein_config["path"], protein_diffusion_checkpoint=diffusion["path"],
            protein_mdn_checkpoint=mdn["path"], precomputed_dir=str(base / "tables"))
        return args, diffusion["sha256"], mdn["sha256"]

    def test_freeze_readback_and_tampering_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, diffusion, mdn = self.fixture(Path(temporary))
            with patch.object(freezer, "DIFFUSION_SHA256", diffusion), patch.object(freezer, "MDN_SHA256", mdn), \
                 patch.object(contract, "DIFFUSION_SHA256", diffusion), patch.object(contract, "MDN_SHA256", mdn):
                freezer.freeze(args)
                frozen = contract.read_config(args.root)
                self.assertEqual([frozen["dataset"][s]["count"] for s in ("train", "val")], [2, 1])
                self.assertTrue(all(m["split_group"] for m in frozen["members"]))
                with self.assertRaises(RuntimeError):
                    freezer.freeze(args)
                Path(args.release_root, "splits/train.txt").write_text("A\nC\n")
                with self.assertRaises(RuntimeError):
                    contract.read_config(args.root)


class NativeCoordinateCPUTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("numpy") and
                         importlib.util.find_spec("torch_geometric"), "optional CPU torch/numpy/PyG unavailable")
    def test_anchor_not_double_centered_and_marker_rejects_repeat(self):
        import torch
        from torch_geometric.data import HeteroData
        from prepare_native import center_auxiliary_once
        graph = HeteroData()
        graph["receptor"].pos = torch.tensor([[0., 0., 0.], [2., 0., 0.]])
        graph["receptor"].center_pos = torch.tensor([[11., 10., 10.], [13., 10., 10.]])
        graph["receptor"].nucleic_anchor_pos = torch.tensor([[0.1, 0., 0.], [2.1, 0., 0.]])
        graph["receptor"].nucleic_anchor_parent_index = torch.tensor([0, 1])
        original = graph["receptor"].nucleic_anchor_pos.clone()
        center = torch.tensor([[10., 10., 10.]])
        audit = center_auxiliary_once(graph, center)
        self.assertTrue(torch.equal(original, graph["receptor"].nucleic_anchor_pos))
        self.assertEqual(audit["nucleic_anchor_pos"]["subtractions"], 0)
        self.assertEqual(audit["center_pos"]["subtractions"], 1)
        with self.assertRaises(RuntimeError):
            center_auxiliary_once(graph, center)


@unittest.skipUnless(importlib.util.find_spec("rdkit"), "optional CPU RDKit unavailable")
class NativeChemistryComparisonTests(unittest.TestCase):
    def compare(self, native, cached):
        from prepare_native import compare_native_cache_chemistry
        return compare_native_cache_chemistry(native, cached)

    def test_stale_achiral_quaternary_N_tag_is_normalized_on_copies(self):
        from rdkit import Chem
        native = Chem.MolFromSmiles("C[N+](C)(CC)CCC")
        cached = Chem.Mol(native)
        cached.GetAtomWithIdx(1).SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
        native_before = [str(a.GetChiralTag()) for a in native.GetAtoms()]
        cached_before = [str(a.GetChiralTag()) for a in cached.GetAtoms()]
        audit = self.compare(native, cached)
        self.assertTrue(all(audit["normalized_gates"].values()))
        self.assertFalse(audit["raw_atom_signature_equal"])
        self.assertEqual(audit["cached_normalized_tag_changes"], [1])
        self.assertEqual(audit["assigned_CIP_centers"], 0)
        self.assertEqual([str(a.GetChiralTag()) for a in native.GetAtoms()], native_before)
        self.assertEqual([str(a.GetChiralTag()) for a in cached.GetAtoms()], cached_before)

    def test_real_enantiomers_are_rejected(self):
        from rdkit import Chem
        with self.assertRaisesRegex(RuntimeError, "normalized chemistry differs"):
            self.compare(Chem.MolFromSmiles("F[C@](Cl)(Br)I"), Chem.MolFromSmiles("F[C@@](Cl)(Br)I"))

    def test_real_EZ_difference_is_rejected(self):
        from rdkit import Chem
        with self.assertRaisesRegex(RuntimeError, "normalized chemistry differs"):
            self.compare(Chem.MolFromSmiles("F/C=C/F"), Chem.MolFromSmiles("F/C=C\\F"))

    def test_charge_difference_is_rejected(self):
        from rdkit import Chem
        with self.assertRaisesRegex(RuntimeError, "normalized chemistry differs"):
            self.compare(Chem.MolFromSmiles("C[NH3+]"), Chem.MolFromSmiles("CN"))

    def test_isotope_difference_is_rejected(self):
        from rdkit import Chem
        with self.assertRaisesRegex(RuntimeError, "normalized chemistry differs"):
            self.compare(Chem.MolFromSmiles("[13CH3]CO"), Chem.MolFromSmiles("CCO"))

    def test_different_atom_order_is_rejected(self):
        from rdkit import Chem
        native = Chem.MolFromSmiles("CCO")
        cached = Chem.RenumberAtoms(native, [2, 1, 0])
        self.assertEqual(Chem.MolToSmiles(native), Chem.MolToSmiles(cached))
        with self.assertRaisesRegex(RuntimeError, "normalized chemistry differs"):
            self.compare(native, cached)

    def test_different_bond_order_is_rejected(self):
        from rdkit import Chem
        with self.assertRaisesRegex(RuntimeError, "normalized chemistry differs"):
            self.compare(Chem.MolFromSmiles("CCC"), Chem.MolFromSmiles("CC=C"))

    def test_single_bond_drawing_direction_is_audit_only(self):
        from rdkit import Chem
        native = Chem.MolFromSmiles("CCC")
        cached = Chem.Mol(native)
        cached.GetBondWithIdx(0).SetBondDir(Chem.BondDir.ENDUPRIGHT)
        audit = self.compare(native, cached)
        self.assertTrue(all(audit["normalized_gates"].values()))
        self.assertEqual(len(audit["raw_bond_direction_differences"]), 1)
        self.assertFalse(audit["graph_features_modified"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
