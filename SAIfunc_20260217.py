import numpy as np
from numpy.strings import mod as percent
import pandas as pd
import json
import time
from pathlib import Path

rng = np.random.default_rng()



class Sim:

    def __init__(self, *, nV, p0, pE, a0, a1, nT_ini, nT_ext):
        self.timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())

        self.nV = np.uint16(nV)
        self.p0 = np.float64(p0)
        self.pE = np.float64(pE)
        self.A = np.array((a0,a1), dtype=np.float64)

        self.nT_ini = np.uint16(nT_ini)
        self.nT_ext = np.uint16(nT_ext)

        if self.nT_ini == 0 or self.nT_ext == 0:
            raise ValueError('Chunk size cannot be zero!')

        self.V = [np.zeros((self.nT_ini, self.nV), dtype=np.bool)]
        self.E = [np.zeros((self.nT_ini, self.nV,self.nV), dtype=np.bool)]

        self.V[0][0,:] = rng.random(self.nV) > self.p0
        edges_init = np.tril(
            rng.random((self.nV,self.nV))  <  self.pE
        )
        self.E[0][0,:,:] = edges_init | edges_init.T

        self.len = 1
        self.curr_memchunk = 0
        self.curr_memitem = 0



    @classmethod
    def fromfile(Sim_, fname, nT_extend_by):
        # np.lib.npyio.NpzFile().get()
        path = Path(fname).with_suffix('.npz')
        with np.load(path) as npz:
            sim_params = {
                'nV': npz.get('nV').item(),
                'p0': npz.get('p0').item(),
                'pE': npz.get('pE').item(),
                'a0': npz.get('a0').item(),
                'a1': npz.get('a1').item()
            }
            nT = npz.get('len').item()
            print('Read', nT, 'from', path, '\n', sim_params)
            result = Sim(
                **sim_params,
                nT_ini = nT,
                nT_ext = np.uint16(nT_extend_by)
            )
            result.V[0] = npz.get('V')
            result.E[0] = npz.get('E')
            result.len = nT
            result.timestamp = npz.get('timestamp').item()
            result.curr_memitem = nT - 1
        return result
            

    
    def create_memchunk(self):
        self.V.append(
            np.zeros((self.nT_ext, self.nV), dtype=np.bool),
        )
        self.E.append(
            np.zeros((self.nT_ext, self.nV,self.nV), dtype=np.bool)
        )
        self.curr_memchunk += 1


    def get_memloc(self, idx):
        "#### takes an integer index idx and returns a 2-tuple (chunki, itemi) that can be used to access the idx'th state"
        if idx < 0:
            # for negative idx, bring it into [0, known_states_n)
            if idx < -self.len:
                raise IndexError
            else:
                idx += self.len
            
        if idx < self.nT_ini:
            # idx in the initial chunk itself; simple case
            return 0, idx
        else: # search for the extn chunk holding the idx'th state
            chunki, itemi = divmod(idx - self.nT_ini, self.nT_ext)
            chunki += 1
            if chunki == self.curr_memchunk:
                if itemi <= self.curr_memitem:
                    return chunki, itemi
            elif chunki < self.curr_memchunk:
                if itemi < self.nT_ini:
                    return chunki, itemi
            raise IndexError
        


    def __getitem__(self, idx):
        if type(idx) is not int:
            raise TypeError("index must be int")
        chunki, itemi = self.get_memloc(idx)
        return self.V[chunki][itemi], self.E[chunki][itemi]



    def push(self, vertices, edges):
        if self.curr_memchunk == 0: 
            limit = self.nT_ini
        else:
            limit = self.nT_ext

        next_item_i = self.curr_memitem + 1
        if next_item_i == limit: # Full
            self.create_memchunk()
            # curr_chk_i incremented within method
            next_item_i = 0

        self.V[self.curr_memchunk][next_item_i] = vertices
        self.E[self.curr_memchunk][next_item_i] = edges
        self.curr_memitem = next_item_i
        self.len += 1



    tweaked_proportion = staticmethod(lambda sub, total: (sub + 1/2)/(total + 1))


    def next(self):
        v, e = self[-1]

        n_type0, n_type1 = np.count_nonzero((~v, v), axis=1)
        edges_bw_unlike =  e & (v ^ v[:,np.newaxis])

        i,j = np.indices(e.shape, sparse=True)

        nNeighbours = np.count_nonzero(e, axis=1)
        nUnlikeNeighbours = np.count_nonzero(edges_bw_unlike, axis=1)
        nMutualNeighbours = np.count_nonzero(e[i,:] & e[j,:], axis=-1)

        unlikes_twpr = self.tweaked_proportion(nUnlikeNeighbours, nNeighbours)
        mutual_twpr = self.tweaked_proportion(nMutualNeighbours, nNeighbours)

        type0_twpr = self.tweaked_proportion(n_type0, self.nV)
        type1_twpr = self.tweaked_proportion(n_type1, self.nV)
        opptype_twprs = np.where(v, type0_twpr, type1_twpr)


        vertex_change_ps = unlikes_twpr ** self.A[v.astype(np.uint8)]
        vertex_changes = rng.random(self.nV) < vertex_change_ps
        vertices_next = np.where(vertex_changes, ~v, v)

        connectivities = (mutual_twpr + mutual_twpr.T)/2
        affinities = np.sqrt(opptype_twprs * opptype_twprs[:, np.newaxis])
        p_off = 1 - affinities
        p_on = np.float_power(connectivities, 1/affinities)
        p_flip = np.where(e, p_off, p_on)
        edges_changes = rng.random(e.shape) < p_flip
        edges_next = np.tril(np.where(edges_changes, ~e, e), k=-1)
        edges_next |= edges_next.T

        self.push(vertices_next, edges_next)




    def all(self):
        "#### returns a 2-tuple of arrays (V,E) with all computed states"
        if self.curr_memchunk == 0:
            return self.V[0][0:self.len], self.E[0][0:self.len]
        else:
            allV = np.zeros((self.len, self.nV), dtype=np.bool)
            allE = np.zeros((self.len, self.nV,self.nV), dtype=np.bool)
            s = 0
            for chk_i in range(self.curr_memchunk):
            # leave out the last (possibly incomplete) chunk
                l = self.nT_ini if chk_i==0 else self.nT_ext
                allV[s:s+l] = self.V[chk_i]
                allE[s:s+l] = self.E[chk_i]
                s += l
            l = self.curr_memitem + 1
            allV[s:s+l] = self.V[-1][:l]
            allE[s:s+l] = self.E[-1][:l]
            return allV, allE



    def save(self, fname=None, *, overwrite=False):
        if fname is None:
            path = Path(f'data/SAIsim-{self.timestamp}.npz')
        else:
            path = Path(fname).with_suffix('.npz')
        if path.is_file() and not overwrite:
            raise FileExistsError('Set overwrite=True to replace contents')

        V, E = self.all()
        np.savez(
            path,
            nV = self.nV,
            p0 = self.p0,
            pE = self.pE,
            a0 = self.A[0],
            a1 = self.A[1],
            len = self.len,
            timestamp = self.timestamp,
            V = V,
            E = E
        )

        # with open('simlog.csv', 'at') as log:
        #     log.write(
        #         f'\n{self.len}, {self.nV}, {self.p0}, {self.pE}, {self.A[0]}, {self.A[1]}, {self.timestamp}, {path}'
        #     )

        simlog = pd.read_csv('simlog.csv', index_col='timestamp')
        if self.timestamp in simlog.index:
            simlog.loc[self.timestamp, 'len'] = self.len
        else:
            simlog.loc[self.timestamp] = {
                'nV': self.nV,
                'p0': self.p0,
                'pE': self.pE,
                'a0': self.A[0],
                'a1': self.A[1],
                'len': self.len,
                'location':path
            }

        simlog.to_csv('simlog.csv')
        print('Saved!', path)
        return None



    def construct_svg(self, idx, size_radius):
        innermargin, outermargin = 10, 5
        svg_halfside = (box_halfside := size_radius + innermargin) + outermargin
        svg_side, box_side = 2*svg_halfside, 2*box_halfside

        svgouter = ('<svg width="%spx" height="%spx" version="1.1" xmlns="http://www.w3.org/2000/svg">' % (svg_side,svg_side), '</svg>')
        g_transform = ('<g transform="translate(%s,%s)">' % (svg_halfside, svg_halfside), '</g>')
        border = '<path d="M -%s -%s h %s v %s h -%s v -%s" fill="#bbbbbb" stroke="black" stroke-width="2px" />' % (2*(box_halfside,) + 4*(box_side,))
        style = """
<style> 
    circle {stroke-width: 1px; stroke: black; fill:none} 
    ._0 {fill: red} 
    ._1 {fill: blue} 
    line {stroke: #444444; stroke-width: 1px} 
    .on {display: inline;} 
    .of {display: none;} 
</style>"""

        v,e = self[idx]
        V_class = np.where(v, '_1', '_0')
        E_class = np.where(e, 'on', 'of')
        i_idx, j_idx = np.triu_indices(self.nV, k=1) 
        V_id = 'V' + percent('%03d', np.arange(self.nV))
        E_id = 'E' + percent('%03d', i_idx) + percent('%03d', j_idx)

        theta = np.linspace(np.pi*1/2, np.pi*-3/2, self.nV, endpoint=False)
        cx = percent('%04d',
            (np.cos(theta) * size_radius).astype(np.int16)
        )
        cy = percent('%04d',
            (np.sin(theta) * size_radius).astype(np.int16)
        )
        x1, y1 = cx[i_idx], cy[i_idx]
        x2, y2 = cx[j_idx], cy[j_idx]

        circles = '<circle id="' + V_id + '" class="' + V_class + '" cx="' + cx + '" cy="' + cy + '" r="3" />'
        lines = (
            '<line id="' + E_id +
            '" class="' + E_class[i_idx, j_idx] +
            '" x1="' + x1 + '" y1="' + y1 +
            '" x2="' + x2 + '" y2="' + y2 +
            '" />'
        )
        return svgouter[0] + g_transform[0] + style + border + '\n'.join(lines) + '\n'.join(circles) + g_transform[1] + svgouter[1]



    def class_diffs_json(self, ref_idx, target_idx):
        v_ref, e_ref = self[ref_idx]
        v_tar, e_tar = self[target_idx]
        v_to1 = np.nonzero((~v_ref) & (v_tar))
        v_to0 = np.nonzero((v_ref) & (~v_tar))

        e_ref_triu = np.triu(e_ref)
        e_tar_triu = np.triu(e_tar)
        e_to1_idxs = percent('%03d', np.nonzero((~e_ref_triu) & (e_tar_triu)))
        e_to0_idxs = percent('%03d', np.nonzero((e_ref_triu) & (~e_tar_triu)))

        return json.dumps({
            '_1': list('V' + percent('%03d', v_to1[0])),
            '_0': list('V' + percent('%03d', v_to0[0])),
            'on': list('E' + e_to1_idxs[0] + e_to1_idxs[1]),
            'of': list('E' + e_to0_idxs[0] + e_to0_idxs[1])
        })



    def all_classes_json(self, target_idx):
        v,e = self[target_idx]
        v_1_idxs = percent('%03d', np.nonzero(v))
        v_0_idxs = percent('%03d', np.nonzero(~v))
        e_1_idxs = percent('%03d', np.nonzero(e))
        e_0_idxs = percent('%03d', np.nonzero(~e))

        return json.dumps({
                '_1': list('V' + v_1_idxs[0]),
                '_0': list('V' + v_0_idxs[0]),
                'on': list('E' + e_1_idxs[0] + e_1_idxs[1]),
                'of': list('E' + e_0_idxs[0] + e_0_idxs[1])
            })
