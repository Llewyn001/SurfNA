"""Synthetic CPU tests only. Never reads experiment data or initializes CUDA."""
from __future__ import annotations

import argparse
import ast
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import extract_features
import rank_common as common
import train_head

HAVE_TORCH = importlib.util.find_spec('torch') is not None


def population():
    split_ids, groups = {}, []
    for split, count in common.COUNTS.items():
        split_ids[split] = [f'{split}_{i:04d}' for i in range(count)]
        for i, name in enumerate(split_ids[split]):
            good = i % 9
            poses = [dict(ordinal=k, pose_uid=f'{name}:{k}', rmsd=1.0 if k < good else 3.0,
                          rmsd_method='symmetry_aware_no_alignment', sampling_seed=20260826,
                          atom_count=3, atom_order_sha256='a'*64) for k in range(8)]
            groups.append(dict(name=name, split=split, split_group=f'group_{name}',
                               parent_pdb=f'pdb_{name}', poses=poses))
    return groups, split_ids


class ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original, cls.ids = population()

    def setUp(self):
        self.groups = copy.deepcopy(self.original)

    def validate(self):
        return common.validate_groups(self.groups, self.ids)

    def test_all_890_7120_including_zero_good_and_all_good(self):
        histogram = self.validate()
        self.assertEqual(sum(histogram['train']), 801)
        self.assertEqual(sum(histogram['val']), 89)
        self.assertGreater(histogram['train'][0], 0)
        self.assertGreater(histogram['train'][8], 0)

    def test_missing_group_rejected(self):
        self.groups.pop()
        with self.assertRaises(RuntimeError):
            self.validate()

    def test_duplicate_group_rejected(self):
        self.groups[-1] = copy.deepcopy(self.groups[0])
        with self.assertRaises(RuntimeError):
            self.validate()

    def test_missing_pose_rejected(self):
        self.groups[0]['poses'].pop()
        with self.assertRaises(RuntimeError):
            self.validate()

    def test_duplicate_pose_uid_rejected(self):
        self.groups[1]['poses'][0]['pose_uid'] = self.groups[0]['poses'][0]['pose_uid']
        with self.assertRaises(RuntimeError):
            self.validate()

    def test_duplicate_ordinal_rejected(self):
        self.groups[0]['poses'][-1]['ordinal'] = 0
        with self.assertRaises(RuntimeError):
            self.validate()

    def test_test_or_wrong_split_rejected(self):
        self.groups[0]['split'] = 'test'
        with self.assertRaises(RuntimeError):
            self.validate()

    def test_parent_pdb_overlap_rejected(self):
        self.groups[-1]['parent_pdb'] = self.groups[0]['parent_pdb']
        with self.assertRaises(RuntimeError):
            self.validate()

    def test_split_group_overlap_rejected(self):
        self.groups[-1]['split_group'] = self.groups[0]['split_group']
        with self.assertRaises(RuntimeError):
            self.validate()

    def test_candidate_atom_order_drift_rejected(self):
        self.groups[0]['poses'][0]['atom_order_sha256'] = 'b'*64
        with self.assertRaises(RuntimeError):
            self.validate()

    def test_nan_label_rejected(self):
        self.groups[0]['poses'][0]['rmsd'] = float('nan')
        with self.assertRaises(RuntimeError):
            self.validate()

    def test_aligned_rmsd_rejected(self):
        self.groups[0]['poses'][0]['rmsd_method'] = 'rigid_aligned'
        with self.assertRaises(RuntimeError):
            self.validate()

    def test_sampling_seed_drift_rejected(self):
        self.groups[0]['poses'][0]['sampling_seed'] += 1
        with self.assertRaises(RuntimeError):
            self.validate()

    def test_non_approved_member_rejected(self):
        self.groups[0]['name'] = 'unknown_from_other_dataset'
        with self.assertRaises(RuntimeError):
            self.validate()


class IOAndRecipeTests(unittest.TestCase):
    def test_tampered_file_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'input.json'
            common.atomic_json(path, {'immutable': 1}, new=True)
            expected = common.sha(path)
            common.check_hash(path, expected)
            common.atomic_json(path, {'immutable': 2})
            with self.assertRaises(RuntimeError):
                common.check_hash(path, expected)

    def test_stage_and_json_overwrite_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            stage = common.reserve_stage(tmp, 'features')
            with self.assertRaises(FileExistsError):
                common.reserve_stage(tmp, 'features')
            with self.assertRaises(FileExistsError):
                common.atomic_json(stage/'STARTED.json', {}, new=True)

    def test_path_escape_and_external_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            inside = base/'inside'
            inside.mkdir()
            common.atomic_json(base/'outside.json', {}, new=True)
            (inside/'escape.json').symlink_to(base/'outside.json')
            for path in ('../outside.json', 'escape.json'):
                with self.assertRaises(RuntimeError):
                    common.safe_path(inside, path)

    def test_hash_pinned_original_architecture_bytes(self):
        self.assertEqual(common.sha(Path(__file__).with_name('head.py')), common.SOURCE_RECIPE['head.py'])

    def test_original_fixed_recipe_and_baseline_eligibility(self):
        self.assertEqual([common.RECIPE[k] for k in ('epochs', 'min_epochs', 'patience', 'batch_groups')], [30, 5, 5, 16])
        self.assertEqual([common.RECIPE[k] for k in ('lr', 'weight_decay', 'residual_l2')], [.001, .001, .01])
        self.assertTrue(common.RECIPE['baseline_selectable'])
        self.assertEqual(common.RECIPE['baseline_epoch'], -1)
        self.assertEqual(common.RECIPE['trainable_parameters'], 4225)
        a = {'top1_lt2': .5, 'regret': 1., 'ndcg': .6}
        b = dict(a, ndcg=.7)
        self.assertGreater(train_head.select_key(b), train_head.select_key(a))
        self.assertGreater(train_head.select_key(dict(a, regret=.9)), train_head.select_key(b))
        self.assertGreater(train_head.select_key(dict(a, top1_lt2=.6, regret=9.)), train_head.select_key(b))
        self.assertFalse(train_head.select_key(a) > train_head.select_key(a))

    def test_feature_api_cannot_receive_labels_or_identity(self):
        tree = ast.parse(Path(extract_features.__file__).read_text())
        node = next(item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == 'evidence')
        self.assertEqual([arg.arg for arg in node.args.args],
                         ['model', 'graph', 'surface', 'surface_pos', 'log_probability', 'device'])
        forbidden = {'rmsd', 'name', 'ordinal', 'native_path', 'native_global_coordinates', 'pose_uid', 'pose_path', 'orig_pos'}
        constants = {item.value for item in ast.walk(node) if isinstance(item, ast.Constant) and isinstance(item.value, str)}
        self.assertFalse(constants & forbidden)
        source = ast.unparse(node)
        self.assertEqual(source.count('encode_ligand_intra('), 1)

    def test_both_default_dryruns_without_model_imports_or_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = argparse.Namespace(root=tmp, execute=False, device='cuda')
            with mock.patch.object(extract_features, 'validate_contract', return_value=({}, [], {})):
                extract_features.run(args)
            with mock.patch.object(train_head, 'validate_contract', return_value=({}, [], {})), \
                    mock.patch.object(train_head, 'validate_features', return_value=({}, [])):
                train_head.run(args)
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_existing_failed_stage_cannot_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            common.reserve_stage(tmp, 'features')
            common.reserve_stage(tmp, 'model')
            args = argparse.Namespace(root=tmp, execute=True, device='cuda')
            with mock.patch.object(extract_features, 'validate_contract', return_value=({}, [], {})):
                with self.assertRaises(RuntimeError):
                    extract_features.run(args)
            with mock.patch.object(train_head, 'validate_contract', return_value=({}, [], {})), \
                    mock.patch.object(train_head, 'validate_features', return_value=({}, [])):
                with self.assertRaises(RuntimeError):
                    train_head.run(args)


@unittest.skipUnless(HAVE_TORCH, 'torch is unavailable locally; CPU numerical tests not executed')
class NumericalCPUTests(unittest.TestCase):
    def setUp(self):
        import torch
        torch.manual_seed(1)
        torch.set_num_threads(1)
        self.torch = torch

    def model(self):
        from head import ContactEvidenceResidual
        return ContactEvidenceResidual(self.torch.zeros(32), self.torch.ones(32), .3).eval()

    def test_parameter_count_and_initial_exact_zero_residual(self):
        t, model = self.torch, self.model()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 4225)
        baseline = t.randn(16)
        score, residual = model(t.randn(16, 7, 32), t.ones(16, 7, dtype=t.bool), baseline)
        self.assertTrue(t.equal(score, baseline/.3))
        self.assertTrue(t.equal(residual, t.zeros(16)))

    def test_nonzero_atom_pose_permutation_and_padding(self):
        t, model = self.torch, self.model()
        t.nn.init.normal_(model.pose[-1].weight)
        x, mask, baseline = t.randn(3, 7, 32), t.ones(3, 7, dtype=t.bool), t.randn(3)
        mask[0, 3:] = False
        original, _ = model(x, mask, baseline)
        padded = x.clone()
        padded[~mask] = 99999
        altered, _ = model(padded, mask, baseline)
        reversed_score, _ = model(x.flip(1), mask.flip(1), baseline)
        pose_order, _ = model(x.flip(0), mask.flip(0), baseline.flip(0))
        self.assertTrue(t.allclose(original, altered, atol=1e-6))
        self.assertTrue(t.allclose(original, reversed_score, atol=1e-6))
        self.assertTrue(t.allclose(original.flip(0), pose_order, atol=1e-6))

    def test_correct_pair_order_lowers_loss(self):
        from head import pair_loss
        rmsd = self.torch.tensor([[.5, 1., 1.5, 2., 3., 4., 5., 6.]])
        good, _ = pair_loss(-rmsd, rmsd)
        bad, _ = pair_loss(rmsd, rmsd)
        self.assertLess(float(good), float(bad))

    def test_exact_original_gap_and_cross2_weight_formula(self):
        from head import pair_loss
        t = self.torch
        rmsd = t.tensor([[1., 1.25, 1.5, 2., 2.1, 3., 4., 5.]])
        scores = t.tensor([[.1, .4, .6, .3, .2, -.2, .1, -.1]])
        expected, total = 0., 0.
        for i in range(8):
            for j in range(8):
                gap = float(rmsd[0, j] - rmsd[0, i])
                if gap > .25:
                    weight = min(gap/2, 1.) * (2 if rmsd[0, i] < 2 <= rmsd[0, j] else 1)
                    expected += weight * float(t.nn.functional.softplus(-(scores[0, i]-scores[0, j])))
                    total += weight
        actual, _ = pair_loss(scores, rmsd)
        self.assertAlmostEqual(float(actual), expected/total, places=6)

    def test_equal_rmsd_no_nan_or_false_gradient(self):
        from head import pair_loss
        t = self.torch
        scores = t.zeros((2, 8), requires_grad=True)
        loss, active = pair_loss(scores, t.ones(2, 8))
        loss.backward()
        self.assertFalse(bool(active.any()))
        self.assertEqual(float(loss), 0.)
        self.assertTrue(t.equal(scores.grad, t.zeros(2, 8)))

    def test_residual_bound_and_empty_mask(self):
        t, model = self.torch, self.model()
        t.nn.init.constant_(model.pose[-1].weight, 100)
        x = t.ones(2, 4, 32)
        _, residual = model(x, t.ones(2, 4, dtype=t.bool), t.zeros(2))
        self.assertLessEqual(float(residual.abs().max()), 2.)
        with self.assertRaises(ValueError):
            model(x, t.zeros(2, 4, dtype=t.bool), t.zeros(2))

    def test_train_only_scaler_rejects_val_or_missing_groups(self):
        t = self.torch
        groups = [dict(name=f'train_{i}', split='train', tokens=t.ones(8, 1, 32)*i,
                       baseline=t.arange(8).float()) for i in range(801)]
        mean, std, scale = train_head.fit_scaler(groups)
        self.assertTrue(t.equal(mean, t.ones(32)*400))
        self.assertGreater(float(std.min()), 0)
        self.assertGreater(float(scale), 0)
        groups[-1]['split'] = 'val'
        with self.assertRaises(RuntimeError):
            train_head.fit_scaler(groups)
        with self.assertRaises(RuntimeError):
            train_head.fit_scaler(groups[:-1])

    def test_candidate_overrides_native_before_cleaning(self):
        t = self.torch
        class Store:
            pass
        class Graph:
            def __init__(self):
                self.original_center = t.tensor([[10., 20., 30.]])
                self.ligand = Store()
                self.ligand.pos = t.zeros(2, 3)
            def __getitem__(self, name):
                assert name == 'ligand'
                return self.ligand
            def clone(self):
                return copy.deepcopy(self)
        template = Graph()
        xyz = t.tensor([[12., 23., 34.], [15., 26., 37.]])
        def cleaner(graph):
            self.assertTrue(t.equal(graph.ligand.pos, xyz-template.original_center))
            del graph.original_center
            return graph
        graph = extract_features.candidate_graph(template, xyz, cleaner)
        self.assertFalse(hasattr(graph, 'original_center'))
        self.assertTrue(t.equal(template.ligand.pos, t.zeros(2, 3)))


if __name__ == '__main__':
    unittest.main()
