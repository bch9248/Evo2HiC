from tqdm import tqdm
import numpy as np
import scipy
from dataset.data_arguments import add_hic_dataset_arguments

def data_predict_parser():
    
    parser = add_hic_dataset_arguments()

    #inference arguments

    infer_args = parser.add_argument_group('inference Arguments')
    infer_args.add_argument('--save-dir', type=str, required=True, help='the dir to save results')
    infer_args.add_argument('--seed', type=int, default=0, help='set the seed to avoid randomness')
    infer_args.add_argument('--split', type=str, default='test', help='split to inference')
    infer_args.add_argument('--batch-size', type=int, default = 64, help='batch size')

    # for resolution enhancement
    infer_args.add_argument('-di', '--data-index', type=int, default = 0, help='start from hic with this index.')
    infer_args.add_argument('-tr', '--target-resolution', type=int, default = None, help='store results in this resolution')

    return parser

def construct_hic_matrices(submats, submat_poses, chrom0, chrom1, chrom_size0, chrom_size1, resolution, target_resolution, max_seperation = 1e100):

    if target_resolution is None:
        target_resolution=resolution
    
    assert target_resolution % resolution == 0

    d = target_resolution // resolution

    chrom_size0, chrom_size1 = int(np.ceil(chrom_size0/resolution)), int(np.ceil(chrom_size1/resolution))

    CHUNK_SIZE = (8000 // d) * d

    csubs = []
    cposs = []

    for mat, pos in zip(submats, submat_poses):
        chr0, s0, e0, strand0, chr1, s1, e1, strand1 = pos
        if chr0 == chrom0 and chr1 == chrom1:
            csubs.append(mat)
            cposs.append(pos)
    
    X, Y, data = [], [], []

    for i in tqdm(range(0, chrom_size0, CHUNK_SIZE)):
        for j in tqdm(range(i, chrom_size1, CHUNK_SIZE), leave=False):
            ii = min(i + CHUNK_SIZE, chrom_size0)
            jj = min(j + CHUNK_SIZE, chrom_size1)

            if (j - ii)*resolution > max_seperation:
                break

            sums = np.zeros((CHUNK_SIZE, CHUNK_SIZE))
            cnts = np.zeros((CHUNK_SIZE, CHUNK_SIZE))

            at_least_one = False

            for mat, pos in zip(csubs, cposs):
                chr0, s0, e0, strand0, chr1, s1, e1, strand1 = pos

                s0, e0, s1, e1 = s0//resolution, e0//resolution, s1//resolution, e1//resolution

                # print(s0, e0, s1, e1)
                # print(mat)

                weight = np.abs(np.arange(e0 - s0) - (e0-s0)/2)[:, None] + np.abs(np.arange(e1 - s1) - (e1-s1)/2)[None, :]
                weight = np.max(weight) - weight + 1

                if s0 >= ii or e0 <= i: continue
                if s1 >= jj or e1 <= j: continue

                at_least_one = True

                mat = mat.toarray()

                if len(mat.shape) > 2:
                    assert mat.shape[0] == 1
                    mat = mat.squeeze(0)

                if mat.shape[0]<weight.shape[0]:
                    s = weight.shape[0] - mat.shape[0]
                    assert s%2 == 0
                    p = s//2
                    mat = np.pad(mat, ((p, p), (0,0)))
                    weight[:p-1, :] = 0
                    weight[weight.shape[0] - p:, :] = 0

                if mat.shape[1]<weight.shape[1]:
                    s = weight.shape[1] - mat.shape[1]
                    assert s%2 == 0
                    p = s//2
                    mat = np.pad(mat, ((0,0), (p, p)))
                    weight[:, :p-1] = 0
                    weight[:, weight.shape[1]-p:] = 0

                if strand0 == '-':
                    mat = np.flip(mat, 0)
                    weight = np.flip(weight, 0)
                if strand1 == '-':
                    mat = np.flip(mat, 1)
                    weight = np.flip(weight, 1)

                if s0 < i:
                    mat = mat[i-s0:, :]
                    weight = weight[i-s0:, :]
                    s0 = i
                if e0 > ii:
                    mat = mat[:ii-e0, :]
                    weight = weight[:ii-e0, :]
                    e0 = ii
                if s1 < j:
                    mat = mat[:, j-s1:]
                    weight = weight[:, j-s1:]
                    s1 = j
                if e1 > jj:
                    mat = mat[:, :jj-e1]
                    weight = weight[:, :jj-e1]
                    e1 = jj
                
                s0 -= i
                e0 -= i
                s1 -= j
                e1 -= j

                sums[s0:e0, s1:e1] = sums[s0:e0, s1:e1] + mat * weight
                cnts[s0:e0, s1:e1] = cnts[s0:e0, s1:e1] + weight
            
            if not at_least_one: continue
            
            sums = sums + sums.T
            cnts = cnts + cnts.T
            Cmat = np.nan_to_num(sums/cnts)
            if i==j: Cmat = np.triu(Cmat)

            Cmat = Cmat.reshape(CHUNK_SIZE//d, d, CHUNK_SIZE//d, d).sum(axis=(1,3))

            Cmat = np.maximum(Cmat, 0)

            sX, sY = np.nonzero(Cmat)
            sdata = Cmat[sX, sY]

            for x, y, da in zip(((sX+i//d) * target_resolution).tolist(), ((sY+j//d) * target_resolution).tolist(), sdata.tolist()):
                if abs(x-y)>max_seperation: continue
                X.append(x)
                Y.append(y)
                data.append(da)

    C = scipy.sparse.coo_matrix((data, (X, Y)))
    return C

def construct_1d_tracks(preds, pred_poses, chrom, chrom_size, resolution, channels):

    chrom_size = int(np.ceil(chrom_size/resolution))

    CHUNK_SIZE = 8000

    cpreds = []
    cposs = []

    for pred, pos in zip(preds, pred_poses):
        chr0, s0, e0, strand0, *_ = pos
        if chr0 == chrom:
            cpreds.append(pred)
            cposs.append(pos)
    
    C = np.zeros((channels, chrom_size))

    for i in tqdm(range(0, chrom_size, CHUNK_SIZE)):
        ii = min(i + CHUNK_SIZE, chrom_size)

        sums = np.zeros((channels, ii-i))
        cnts = np.zeros((channels, ii-i))

        at_least_one = False

        for pred, pos in zip(cpreds, cposs):
            chr0, s0, e0, strand0, *_ = pos

            s, e = int(s0//resolution), int(e0//resolution)

            weight = np.abs(np.arange(e - s) - (e-s)/2)
            weight = np.max(weight) - weight + 1

            if s >= ii or e <= i: continue

            at_least_one = True

            pred = np.array(pred)

            assert pred.shape[0] == channels
            assert pred.shape[1] == weight.shape[0]

            if strand0 == '-':
                pred = np.flip(pred, 1)
                weight = np.flip(weight, 0)

            if s < i:
                pred   = pred[:,i-s:]
                weight = weight[i-s:]
                s = i
            if e > ii:
                pred   = pred[:,:ii-e]
                weight = weight[:ii-e]
                e = ii
            
            s -= i
            e -= i

            sums[:, s:e] = sums[:, s:e] + pred * weight[None, :]
            cnts[:, s:e] = cnts[:, s:e] + weight[None, :]
            
            if not at_least_one: continue
            
            Cpred = np.nan_to_num(sums/cnts)

            C[:, i:ii] = Cpred

    return C