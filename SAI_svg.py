import numpy as np
from numpy.strings import mod as npformat
import json

def svg_setup(size_radius, n_vertices, vertex_class, edge_class):
    innermargin, outermargin = 10, 5
    svg_halfside = (box_halfside := size_radius + innermargin) + outermargin
    svg_side, box_side = 2*svg_halfside, 2*box_halfside

    svgouter = ('<svg width="%spx" height="%spx" version="1.1" xmlns="http://www.w3.org/2000/svg">' % (svg_side,svg_side), '</svg>')
    g_transform = ('<g transform="translate(%s,%s)">' % (svg_halfside, svg_halfside), '</g>')
    border = '<path d="M -%s -%s h %s v %s h -%s v -%s" fill="#bbbbbb" stroke="black" stroke-width="2px" />' % (2*(box_halfside,) + 4*(box_side,))

    style = '<style> circle {stroke-width: 1px; stroke: black; fill:none} ._0 {fill: red} ._1 {fill: blue} line {stroke: #555555; stroke-width: 1px} .on {display: inline;} .of {display: none;} </style>'

    theta = np.linspace(np.pi*1/2, np.pi*-3/2, n_vertices, endpoint=False)
    cx = (np.cos(theta) * size_radius).astype(np.int16)
    cy = (np.sin(theta) * size_radius).astype(np.int16)

    i_ = npformat('%03d', np.arange(n_vertices))
    cx_ = npformat('%04d', cx)
    cy_ = npformat('%04d', cy)
    circles = '<circle id="V' + i_ + '" class="' + vertex_class + '" cx="' + cx_ + '" cy="' + cy_ + '" r="3" />'

    i_idx, j_idx = np.triu_indices(n_vertices, k=1) # shape (n^2-n), i<j
    # 2️⃣ Gather coordinates
    x1 = npformat('%04d', cx[i_idx])
    y1 = npformat('%04d', cy[i_idx])
    x2 = npformat('%04d', cx[j_idx])
    y2 = npformat('%04d', cy[j_idx])
    # 3️⃣ Build deterministic DOM ids: 'E__i__j', i<j
    id_i = npformat('%03d', i_idx)
    id_j = npformat('%03d', j_idx)
    classes = edge_class[i_idx, j_idx]
    # 4️⃣ Construct <line /> elements vectorized
    lines = (
        '<line id="' + 'E' + id_i + id_j +
        '" class="' + classes +
        '" x1="' + x1 + '" y1="' + y1 +
        '" x2="' + x2 + '" y2="' + y2 +
        '" />'
    )

    return svgouter[0] + g_transform[0] + style + border + '\n'.join(lines) + '\n'.join(circles) + g_transform[1] + svgouter[1]



def style_updates_as_json(vertices_ref, edges_ref, vertices_target, edges_target):
    v_to1, = np.nonzero((~vertices_ref) & (vertices_target))
    v_to0, = np.nonzero((vertices_ref) & (~vertices_target))

    eref = np.triu(edges_ref)
    etar = np.triu(edges_target)
    e_to1_idxs = npformat('%03d', np.nonzero((~eref) & (etar)))
    e_to0_idxs = npformat('%03d', np.nonzero((eref) & (~etar)))

    return json.dumps({
    # lists of svg element ids that need to be updates to the corresponding class
        '_1': list('V' + npformat('%03d', v_to1)),
        '_0': list('V' + npformat('%03d', v_to0)),
        'on': list('E' + e_to1_idxs[0] + e_to1_idxs[1]),
        'of': list('E' + e_to0_idxs[0] + e_to0_idxs[1])
    })


