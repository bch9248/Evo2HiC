import numpy as np
from cooler import Cooler
from cooltools.lib.numutils import adaptive_coarsegrain
from scipy.ndimage import binary_dilation

from hic_utils import *
import utils
from utils import *
from config import *

class Cool_Loader:
    def __init__(
            self,

            cell_line,

            resolution,

            **kwargs
        ) -> None:

        self.cell_line = cell_line
        assert cell_line in cool_files 
        self.resolution = resolution
        self.base_resolution = 1000

        self.D = self.resolution // self.base_resolution

        cool_file = cool_files[cell_line].replace('$RES', str(self.base_resolution))
        expect_file = expect_files[cell_line].replace('$RES', str(self.base_resolution))

        self.cooler = Cooler(cool_file)

        smooth_diag = np.load(expect_file)
        self.normmat = np.exp(smooth_diag)[np.abs(np.arange(4000000//self.base_resolution)[:, None] - np.arange(4000000//self.base_resolution)[None, :])]
        self.normmat = np.mean(self.normmat.reshape(4000000//resolution, self.D, 4000000//resolution, self.D), axis=(-1, -3))
        self.eps = np.min(self.normmat[:1000000//resolution, :1000000//resolution])

        self.read_count = 1

        self.chrs = self.cooler.chromnames

    def get(self, chr0, s0, e0, strand0, chr1, s1, e1, strand1, norm=False):
        c0, c1 = find_chr_in_set(chr0, self.chrs), find_chr_in_set(chr1, self.chrs)

        assert chr0 == chr1 and s0 == s1 and e0 == e1 and strand0 == strand1

        d = e0//self.resolution - s0//self.resolution

        matrix_r = self.cooler.matrix(balance=norm).fetch(f"{c0}:{s0}-{e0}")

        matrix_r = np.nanmean(np.reshape(matrix_r, (d, self.D, d, self.D)), axis=(-1, -3))

        matrix = matrix_r

        if strand0:
            matrix = np.flip(matrix, -2)
        if strand1:
            matrix = np.flip(matrix, -1)

        matrix = np.expand_dims(matrix, axis=0)

        return matrix.copy()

    def get_expected(self, chr0, s0, e0, strand0, chr1, s1, e1, strand1):
        d = e0//self.resolution - s0//self.resolution

        matrix = self.normmat[:d, :d]

        if strand0:
            matrix = np.flip(matrix, -2)
        if strand1:
            matrix = np.flip(matrix, -1)

        matrix = np.expand_dims(matrix, axis=0)

        return matrix.copy()