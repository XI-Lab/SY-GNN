import logging
import os
import shutil
import tempfile

import numpy as np
from ase.io.extxyz import read_xyz
from pack.data import AtomsData
from pack.environment import SimpleEnvironmentProvider

__all__ = ['QM_sym']
properties = ['energy_U0', 'energy_U', 'enthalpy_H', 'free_energy', 'heat_capacity',
              'isotropic_polarizability', 'electronic_spatial_extent', 'zpve',
              'gap', 'lumo', 'homo', 'dipole_moment']

atomref = np.zeros([36, len(properties)])  # largest atomistic number is 35, U0 U H,G  need atomref
atomref[6, 0] = -38.08212814
atomref[1, 0] = -0.601967955
atomref[8, 0] = -75.16077019
atomref[17, 0] = -460.1589048
atomref[35, 0] = -2571.561142
atomref[9, 0] = -99.80889669
atomref[5, 0] = -79.66130988
atomref[7, 0] = -0  # since N always show with B together

atomref[6, 1] = -37.93934655
atomref[1, 1] = -0.515270609
atomref[8, 1] = -74.83047132
atomref[17, 1] = -460.0680846
atomref[35, 1] = -2571.467721
atomref[9, 1] = -99.72059438
atomref[5, 1] = -79.42408612
atomref[7, 1] = -0  # since N always show with B together

atomref[6, 2] = -38.08162248
atomref[1, 2] = -0.601597709
atomref[8, 2] = -75.15870704
atomref[17, 2] = -460.1574369
atomref[35, 2] = -2571.559389
atomref[9, 2] = -99.80787199
atomref[5, 2] = -79.66002671
atomref[7, 2] = -0  # since N always show with B together

atomref[6, 3] = -38.08661108
atomref[1, 3] = -0.601088457
atomref[8, 3] = -75.16388831
atomref[17, 3] = -460.1589339
atomref[35, 3] = -2571.562024
atomref[9, 3] = -99.808346
atomref[5, 3] = -79.67131417
atomref[7, 3] = -0  # since N always show with B together


class QM_sym(AtomsData):
    """
        Args:
            subset (list): indices of subset. Set to None for entire dataset (default: None)
            properties (list): properties in QM_sym, e.g. U0
    """

    # properties list
    # sym = 'symmetry_group'
    gap = 'gap'  # Band Gap (eV)
    lumo = 'lumo'  # LUMO (eV)
    homo = 'homo'  # HOMO (eV)
    A = 'rotational_constant_A'  # Rotationl Constant (GHz) 1
    B = 'rotational_constant_B'  # Rotationl Constant (GHz) 2
    C = 'rotational_constant_C'  # Rotationl Constant (GHz) 3
    mu_X = 'dipole_moment_X'  # Dipole Moment (D) x
    mu_Y = 'dipole_moment_Y'  # Dipole Moment (D) y
    mu_Z = 'dipole_moment_Z'  # Dipole Moment (D) z
    mu = 'dipole_moment'  # Dipole Moment (D) total
    alpha = 'isotropic_polarizability'  # Isotropic Polarizability (a0^3)
    r2 = 'electronic_spatial_extent'  # Electronic Spatial Extent (au)
    zpve = 'zpve'  # Zero-point Vibrational energy (J/mol)
    zpve_kcal = 'zpve_kcal'  # Zero-point Vibrational energy (Kcal/mol)
    U0 = 'energy_U0'  # Sum of electronic and zero-point energies (Ha)
    U = 'energy_U'  # Sum of electronic and thermal energies (Ha)
    H = 'enthalpy_H'  # Sum of electronic and thermal enthalpies (Ha)
    G = 'free_energy'  # Sum of electronic and thermal free energies (Ha)
    Cv = 'heat_capacity'  # Heat Capacity (J*K^-1)
    D_m5 = 'degeneracy_-5'  # Degeneracy of orbitals -5
    D_m4 = 'degeneracy_-4'  # Degeneracy of orbitals -4
    D_m3 = 'degeneracy_-3'  # Degeneracy of orbitals -3
    D_m2 = 'degeneracy_-2'  # Degeneracy of orbitals -2
    D_m1 = 'degeneracy_-1'  # Degeneracy of orbitals -1
    D_homo = 'degeneracy_homo'  # Degeneracy of orbitals HOMO
    D_lumo = 'degeneracy_lumo'  # Degeneracy of orbitals LUMO
    D_p1 = 'degeneracy_+1'  # Degeneracy of orbitals +1
    D_p2 = 'degeneracy_+2'  # Degeneracy of orbitals +2
    D_p3 = 'degeneracy_+3'  # Degeneracy of orbitals +3
    D_p4 = 'degeneracy_+4'  # Degeneracy of orbitals +4
    D_p5 = 'degeneracy_+5'  # Degeneracy of orbitals +5
    sym_m5 = 'symmetry_-5'  # Symmetry of orbitals -5
    sym_m4 = 'symmetry_-4'  # Symmetry of orbitals -4
    sym_m3 = 'symmetry_-3'  # Symmetry of orbitals -3
    sym_m2 = 'symmetry_-2'  # Symmetry of orbitals -2
    sym_m1 = 'symmetry_-1'  # Symmetry of orbitals -1
    sym_homo = 'symmetry_HOMO'  # Symmetry of orbitals HOMO
    sym_lumo = 'symmetry_LUMO'  # Symmetry of orbitals LUMO
    sym_p1 = 'symmetry_+1'  # Symmetry of orbitals +1
    sym_p2 = 'symmetry_+2'  # Symmetry of orbitals +2
    sym_p3 = 'symmetry_+3'  # Symmetry of orbitals +3
    sym_p4 = 'symmetry_+4'  # Symmetry of orbitals +4
    sym_p5 = 'symmetry_+5'  # Symmetry of orbitals +5

    available_properties = [gap, lumo, homo,
                            A, B, C,
                            mu_X, mu_Y, mu_Z, mu,
                            alpha,
                            r2,
                            zpve, zpve_kcal,
                            U0, U, H, G,
                            Cv,
                            D_m5, D_m4, D_m3, D_m2, D_m1, D_homo,
                            D_lumo, D_p1, D_p2, D_p3, D_p4, D_p5,
                            sym_m5, sym_m4, sym_m3, sym_m2, sym_m1, sym_homo,
                            sym_lumo, sym_p1, sym_p2, sym_p3, sym_p4, sym_p5]

    def __init__(self, dbpath, xyzpath=None, load_from_file=True, subset=None, properties=available_properties,
                 collect_triples=False, sym_tags=False):
        self.dbpath = dbpath
        self.required_properties = properties
        self.sym_tags = sym_tags
        environment_provider = SimpleEnvironmentProvider()

        if not os.path.exists(dbpath) and load_from_file:
            self._load(xyzpath)

        super().__init__(self.dbpath, subset, self.required_properties,
                         environment_provider,
                         collect_triples, sym_tags=sym_tags)

    def create_subset(self, idx):
        idx = np.array(idx)
        subidx = idx if self.subset is None else np.array(self.subset)[idx]

        return QM_sym(self.dbpath, None, False, subidx, self.required_properties,
                      self.collect_triples, self.sym_tags)

    def _load(self, xyzpath):
        evilmols = None
        self._load_data(xyzpath, evilmols)

    def _load_data(self, xyzpath, evilmols=None, ):
        tmpdir = tempfile.mkdtemp('sym')
        raw_path = os.path.join(xyzpath)

        ordered_files = []
        logging.info('Parse xyz files...')
        for fpathe, dirs, fs in os.walk(raw_path):
            for f in fs:
                ordered_files.append(os.path.join(fpathe, f))

        ordered_files = sorted(ordered_files)

        all_atoms = []
        all_properties = []

        irange = np.arange(len(ordered_files), dtype=np.int)
        if evilmols is not None:
            irange = np.setdiff1d(irange, evilmols - 1)

        sym_dict = dict((c, i) for i, c in enumerate(['BU', 'BG', 'AU', 'AG', 'EU', 'EG', 'A"', 'E"', "A'", "E'"]))

        # parse XYZ file
        for i in irange:
            xyzfile = ordered_files[i]

            if (i + 1) % 1000 == 0:
                logging.info('Parsed: {:6d}'.format(i + 1))
            properties = {}
            try:
                with open(xyzfile, 'r') as f:
                    lines = f.readlines()
                    info = lines[1].strip().split('|')

                    for j in range(1, 20):
                        properties[self.required_properties[j - 1]] = np.array(float(info[j])).reshape(1)
                    for j in range(20, 32):  # degeneracy
                        properties[self.required_properties[j - 1]] = np.array(int(info[j])).reshape(1)
                    for j in range(32, 44):  # symmetry
                        properties[self.required_properties[j - 1]] = np.array(sym_dict[info[j]]).reshape(1)

                    def to_idx(idx_str):
                        return int(idx_str[1:])

                    # the tage for each atoms, idicate the index of the atoms in which primitive cell
                    tags = list(map(to_idx, info[44].strip().split(' ')))

                    tmp = os.path.join(tmpdir, 'tmp.xyz')
                    with open(tmp, 'wt') as fout:
                        fout.write(lines[0])
                        fout.write('**\n')
                        for line in lines[2:]:
                            fout.write(line[:line.rfind(' ')] + '\n')  # remove charge
                    with open(tmp, 'r') as ftmp:
                        ats = list(read_xyz(ftmp, 0))[0]

                ats.set_tags(tags=tags)
                all_atoms.append(ats)
                all_properties.append(properties)
            except:
                print(xyzfile, info)

        logging.info('Write atoms to db...')
        self.add_systems(all_atoms, all_properties)
        logging.info('Done.')

        shutil.rmtree(tmpdir)

        return True


if __name__ == '__main__':
    """
    Should run this code to generate database file from XYZ files beform training network.
    """
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
    data = QM_sym(dbpath=r'../../database/QM_sym.db', xyzpath=r'../../../QM-sym-database/xyz')
