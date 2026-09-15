#!/usr/bin/env python3
"""Small chemistry and topology unit tests for Surface-v2."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

import generate_surfaces_v2 as v2


class ChemistryTests(unittest.TestCase):
    def features(self, residue: str, atom: str) -> tuple[float, float, float]:
        name = f"A_1_x_{residue}_{atom}_V2"
        _hbond, _hphob, donor, acceptor, apolar = v2.atom_propensities([name])
        return float(donor[0]), float(acceptor[0]), float(apolar[0])

    def test_guanine_n1_is_donor_not_acceptor(self) -> None:
        self.assertEqual(self.features("G", "N1"), (1.0, 0.0, 0.0))

    def test_uracil_n3_is_donor_not_acceptor(self) -> None:
        self.assertEqual(self.features("U", "N3"), (1.0, 0.0, 0.0))

    def test_rna_two_prime_oxygen_is_both(self) -> None:
        self.assertEqual(self.features("A", "O2'"), (1.0, 1.0, 0.0))

    def test_adenine_n1_is_acceptor(self) -> None:
        self.assertEqual(self.features("DA", "N1"), (0.0, 1.0, 0.0))

    def test_protein_sidechain_roles(self) -> None:
        self.assertEqual(self.features("LYS", "NZ"), (1.0, 0.0, 0.0))
        self.assertEqual(self.features("ASP", "OD1"), (0.0, 1.0, 0.0))

    def test_carbon_is_shared_apolar_type(self) -> None:
        self.assertEqual(self.features("A", "C8"), (0.0, 0.0, 1.0))
        self.assertEqual(self.features("PHE", "CZ"), (0.0, 0.0, 1.0))


class TopologyTests(unittest.TestCase):
    def test_boundary_mask(self) -> None:
        vertices = np.zeros((4, 3), dtype=float)
        faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
        mask, edge_count = v2.mesh_boundary(vertices, faces)
        self.assertEqual(edge_count, 4)
        np.testing.assert_array_equal(mask, np.ones(4))

    def test_closed_tetrahedron_has_no_boundary(self) -> None:
        vertices = np.zeros((4, 3), dtype=float)
        faces = np.asarray([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]], dtype=np.int64)
        mask, edge_count = v2.mesh_boundary(vertices, faces)
        self.assertEqual(edge_count, 0)
        np.testing.assert_array_equal(mask, np.zeros(4))


class PqrParsingTests(unittest.TestCase):
    def test_combined_chain_and_four_digit_residue_id(self) -> None:
        with TemporaryDirectory() as directory:
            pqr = Path(directory) / "input.pqr"
            pdb = Path(directory) / "output.pdb"
            pqr.write_text(
                "ATOM      1  P    RG  A1403       1.000    2.000    3.000  1.0000 2.1000\n"
            )
            self.assertEqual(v2.write_standardized_pdb_from_pqr(pqr, pdb), 1)
            line = pdb.read_text().splitlines()[0]
            self.assertEqual(line[17:20].strip(), "G")
            self.assertEqual(line[21], "A")
            self.assertEqual(int(line[22:26]), 1403)


if __name__ == "__main__":
    unittest.main()
