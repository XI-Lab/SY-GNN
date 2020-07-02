import logging
import os
import re
import shutil
import tarfile
import tempfile
from urllib import request as request

import numpy as np
from ase.io.extxyz import read_xyz
from ase.units import Debye, Bohr, Hartree, eV

from pack.data import DownloadableAtomsData

__all__ = ['QM9']

properties = ['energy_U0', 'energy_U', 'enthalpy_H', 'free_energy', 'heat_capacity',
              'isotropic_polarizability', 'electronic_spatial_extent', 'zpve',
              'gap', 'lumo', 'homo', 'dipole_moment']

aggregate = ['sum', 'sum', 'sum', 'sum', 'sum',
             'sum', 'sum', 'avg',
             'avg', 'avg', 'avg', 'sum']

atomref = np.zeros([36, len(properties)])  # largest atomistic number is 35, U0 U H, G need atomref
atomref[1, 0] = -0.500273  # H
atomref[6, 0] = -37.846772  # C
atomref[7, 0] = -54.583861  # N
atomref[8, 0] = -75.064579  # O
atomref[9, 0] = -99.718730  # F

atomref[1, 1] = -0.498857  # H
atomref[6, 1] = -37.845355  # C
atomref[7, 1] = -54.582445  # N
atomref[8, 1] = -75.063163  # O
atomref[9, 1] = -99.717314  # F

atomref[1, 2] = -0.497912  # H
atomref[6, 2] = -37.844411  # C
atomref[7, 2] = -54.581501  # N
atomref[8, 2] = -75.062219  # O
atomref[9, 2] = -99.716370  # F

atomref[1, 3] = -0.510927  # H
atomref[6, 3] = -37.861317  # C
atomref[7, 3] = -54.598897  # N
atomref[8, 3] = -75.079532  # O
atomref[9, 3] = -99.733544  # F


class QM9(DownloadableAtomsData):
    """ QM9 benchmark dataset for organic molecules with up to nine heavy atoms from {C, O, N, F}.

        This class adds convenience functions to download QM9 from figshare and load the data into pytorch.

        Args:
            path (str): path to directory containing qm9 database.
            download (bool): enable downloading if database does not exists (default: True)
            subset (list): indices of subset. Set to None for entire dataset (default: None)
            properties (list): properties in qm9, e.g. U0
            pair_provider (BaseEnvironmentProvider):
            remove_uncharacterized (bool): remove uncharacterized molecules from dataset (according to [#qm9_1]_)

        References:
            .. [#qm9_1] https://ndownloader.figshare.com/files/3195404

    """

    # properties
    A = 'rotational_constant_A'
    B = 'rotational_constant_B'
    C = 'rotational_constant_C'
    mu = 'dipole_moment'
    alpha = 'isotropic_polarizability'
    homo = 'homo'
    lumo = 'lumo'
    gap = 'gap'
    r2 = 'electronic_spatial_extent'
    zpve = 'zpve'
    U0 = 'energy_U0'
    U = 'energy_U'
    H = 'enthalpy_H'
    G = 'free_energy'
    Cv = 'heat_capacity'

    available_properties = [
        A, B, C,
        mu, alpha, homo, lumo, gap,
        r2, zpve,
        U0, U, H, G, Cv
    ]

    # reference = {
    #     zpve: 0, U0: 1, U: 2, H: 3, G: 4, Cv: 5
    # }

    # units = dict(
    #     zip(available_properties,
    #         [1., 1., 1.,
    #          Debye, Bohr ** 3, Hartree, Hartree, Hartree,
    #          Bohr ** 2, Hartree,
    #          Hartree, Hartree, Hartree, Hartree, 1.
    #          ])
    # )

    def __init__(self, dbpath, download=True, subset=None, properties=None,
                 collect_triples=False, remove_uncharacterized=False):

        self.remove_uncharacterized = remove_uncharacterized

        super().__init__(dbpath=dbpath, subset=subset,
                         required_properties=properties,
                         collect_triples=collect_triples, download=download)

    def create_subset(self, idx):
        idx = np.array(idx)
        subidx = idx if self.subset is None else np.array(self.subset)[idx]

        return QM9(self.dbpath, False, subidx, self.required_properties,
                   self.collect_triples, False)

    def _download(self):
        if self.remove_uncharacterized:
            evilmols = self._load_evilmols()
        else:
            evilmols = None

        self._load_data(evilmols)

    def _load_evilmols(self):
        logging.info('Downloading list of uncharacterized molecules...')
        at_url = 'https://ndownloader.figshare.com/files/3195404'
        tmpdir = tempfile.mkdtemp('gdb9')
        tmp_path = os.path.join(tmpdir, 'uncharacterized.txt')

        request.urlretrieve(at_url, tmp_path)
        logging.info("Done.")

        evilmols = []
        with open(tmp_path) as f:
            lines = f.readlines()
            for line in lines[9:-1]:
                evilmols.append(int(line.split()[0]))
        return np.array(evilmols)

    def _load_data(self, evilmols=None):
        tmpdir = tempfile.mkdtemp('gdb9')
        raw_path = os.path.join(r'../../database/qm9')

        logging.info('Parse xyz files...')
        ordered_files = sorted(os.listdir(raw_path),
                               key=lambda x: (int(re.sub('\D', '', x)), x))

        all_atoms = []
        all_properties = []

        irange = np.arange(len(ordered_files), dtype=np.int)
        if evilmols is not None:
            irange = np.setdiff1d(irange, evilmols - 1)

        for i in irange:
            xyzfile = os.path.join(raw_path, ordered_files[i])

            if (i + 1) % 10000 == 0:
                logging.info('Parsed: {:6d}'.format(i + 1))
            properties = {}
            tmp = os.path.join(tmpdir, 'tmp.xyz')

            with open(xyzfile, 'r') as f:
                lines = f.readlines()
                l = lines[1].split()[2:]
                for pn, p in zip(QM9.available_properties, l):
                    properties[pn] = np.array([float(p)])  # * self.units[pn]])
                with open(tmp, "wt") as fout:
                    have_tag = False
                    for line in lines:
                        fout.write(line.replace('*^', 'e'))
                        if not have_tag:
                            tags = list(range(1, int(line) + 1))
                            have_tag = True
            with open(tmp, 'r') as f:
                ats = list(read_xyz(f, 0))[0]
            ats.set_tags(tags=tags)

            all_atoms.append(ats)
            all_properties.append(properties)

        logging.info('Write atoms to db...')
        self.add_systems(all_atoms, all_properties)
        logging.info('Done.')

        shutil.rmtree(tmpdir)

        return True


if __name__ == '__main__':
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
    # data = QM9(r'../../database/qm9.db')
    data = QM9(r'../../database/qm9.db')
