import numpy as np

class simmem:
    """
    Provides functionality to enable storage and
    convenient access/retrieval of the results from SAIfunc. 
    Does not validate shape of data. \n
    Append newly computed states using the push method.
    """
    def __init__(self, n_timepoints, ext_timepoints_n, n_vertices, vertices_init, edges_init):
        """
        ### n_timepoints: 
        how many states to allocate space for initially
        ### ext_timepoints_n:
        whenever allocated memory is full, how many states to extend memory by
        """
        self.nV = n_vertices
        self._V = [np.zeros((n_timepoints, n_vertices), dtype=np.bool)]
        self._V[0][0,:] = vertices_init
        self._E = [np.zeros((n_timepoints, n_vertices,n_vertices), dtype=np.bool)]
        self._E[0][0,:,:] = edges_init        

        self._ini_chk_size = n_timepoints
        self._ext_chk_size = np.uint8(ext_timepoints_n)

        self._known_states_n = 1
        # following two vars can be deduced easily enough from _V and known_states_n
        # keeping them for clarity/convenience, and to avoid repeatedly accessing _V for lengths
        self._curr_chk_i = 0
        self._curr_item_i = 0
        # access most recently filled state with
        #  self._*[self._curr_chk_i][self._curr_item_i]


    def alternate_constructor():
        "for a sim that's been read from file. the standard fresh sim __init__ allows for population of only the very first state"
        pass



    def __len__(self):
        # enables stuff like:
        #  for x in range(len(my_simmem)): foo(*my_simmem[x])
        return self._known_states_n



    def __getitem__(self, idx):
        if type(idx) is not int:
            raise TypeError("index must be int")
        chunki, itemi = self._get_loc(idx)
        return self._V[chunki][itemi], self._E[chunki][itemi]
        

    def _get_loc(self, idx):
        "#### takes an integer index idx and returns a 2-tuple (chunki, itemi) that can be used to access the idx'th state"
        if idx < 0:
            # for negative idx, bring it into [0, known_states_n)
            if idx < -self._known_states_n:
                raise IndexError
            else:
                idx += self._known_states_n
            
        if idx < self._ini_chk_size:
            # idx in the initial chunk itself; simple case
            return 0, idx
        else: # search for the extn chunk holding the idx'th state
            chunki, itemi = divmod(idx - self._ini_chk_size, self._ext_chk_size)
            chunki += 1
            # EQN:- idx = ini_chk_size + (chunki-1)*ext_chk_size + itemi
            # WHERE 1 ≤ chunki ≤ curr_chunk 
            # AND, IF chunki = curr_chunk, 0 ≤ itemi < curr_idx
            # ELSE, IF chunki < curr_chunk, 0 ≤ itemi < _ext_chk_size

            if chunki == self._curr_chk_i:
                if itemi <= self._curr_item_i:
                    return chunki, itemi
            elif chunki < self._curr_chk_i:
                if itemi < self._ext_chk_size:
                    return chunki, itemi
            raise IndexError
            

    def push(self, vertices, edges):
        if self._curr_chk_i == 0: 
            limit = self._ini_chk_size
        else:
            limit = self._ext_chk_size

        next_item_i = self._curr_item_i + 1
        if next_item_i == limit: # Full
            self._create_ext_chunk()
            # curr_chk_i incremented within method
            next_item_i = 0

        self._V[self._curr_chk_i][next_item_i] = vertices
        self._E[self._curr_chk_i][next_item_i] = edges
        self._curr_item_i = next_item_i
        self._known_states_n += 1


    def _create_ext_chunk(self):
        self._V.append(np.zeros((self._ext_chk_size, self.nV)))
        self._E.append(np.zeros((self._ext_chk_size, self.nV,self.nV)))
        self._curr_chk_i += 1


    def all(self):
        "#### returns a 2-tuple of arrays (V,E) with all computed states"
        total_states = self._known_states_n
        if total_states < self._ini_chk_size:
            return self._V[0:total_states], self._E[0:total_states]
        else:
            allV = np.zeros((total_states, self.nV), dtype=np.bool)
            allE = np.zeros((total_states, self.nV,self.nV), dtype=np.bool)
            s = 0
            for ch in range(self._curr_chk_i - 1):
                # leave out the last (possibly incomplete) chunk
                l = self._ini_chk_size if ch==0 else self._ext_chk_size
                allV[s:s+l] = self._V[ch]
                allE[s:s+l] = self._E[ch]
                s += l
            l = self._curr_item_i + 1
            allV[s:s+l] = self._V[-1]
            allE[s:s+l] = self._E[-1]
            return allV, allE
