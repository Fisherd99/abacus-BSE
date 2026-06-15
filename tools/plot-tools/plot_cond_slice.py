#!/usr/bin/env python3
"""
Plot conditional exciton density on a 2D cross-section plane.

Reads a data file produced by ABACUS's ExcitonPlotter::plot_cond_slice(),
which contains the conditional density |psi_cond(r)|^2 on a regular 2D grid
spanning the BvK supercell (with padding to verify periodicity).

The plot shows:
  - Density heatmap (log scale)
  - BvK supercell boundaries (thick solid lines, spacing = nk mesh)
  - Unit cell boundaries (thin dashed lines)
  - Atom positions projected onto the cross-section plane, with depth coloring
  - Fixed hole position (green dot, home cell only)

Usage: python3 plot_cond_slice.py <data_file.dat>
"""
import sys, numpy as np, matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

def parse(fn):
    """Parse the data file header into a metadata dictionary."""
    m = {}
    with open(fn) as f:
        for line in f:
            if not line.startswith('#'): break
            parts = line[1:].strip().split(None, 1)
            if len(parts) == 2: m[parts[0]] = parts[1]
    for k in ['state', 'grid_nu', 'grid_nv', 'BvK_u', 'BvK_v',
              'u_min', 'u_max', 'v_min', 'v_max', 'n_atoms', 'has_hole']:
        if k in m: m[k] = int(m[k])
    m['density_kind'] = m.get('density_kind', 'conditional')
    m['has_hole'] = m.get('has_hole', 1 if m['density_kind'] == 'conditional' else 0)
    for k in ['energy_Ry', 'slice_pos', 'hole_fix_x', 'hole_fix_y', 'hole_fix_z',
              'u_vec_x', 'u_vec_y', 'u_vec_z', 'v_vec_x', 'v_vec_y', 'v_vec_z',
              'cell_a_x', 'cell_a_y', 'cell_a_z', 'cell_b_x', 'cell_b_y', 'cell_b_z',
              'cell_c_x', 'cell_c_y', 'cell_c_z']:
        if k in m: m[k] = float(m[k])
    m['nu'] = m.get('grid_nu', 100); m['nv'] = m.get('grid_nv', 100)
    m['nk_u'] = m.get('BvK_u', 1); m['nk_v'] = m.get('BvK_v', 1)
    m['u_vec'] = np.array([m['u_vec_x'], m['u_vec_y'], m['u_vec_z']])
    m['v_vec'] = np.array([m['v_vec_x'], m['v_vec_y'], m['v_vec_z']])
    m['a'] = np.array([m['cell_a_x'], m['cell_a_y'], m['cell_a_z']])
    m['b'] = np.array([m['cell_b_x'], m['cell_b_y'], m['cell_b_z']])
    m['c'] = np.array([m['cell_c_x'], m['cell_c_y'], m['cell_c_z']])
    m['hole'] = np.array([m['hole_fix_x'], m['hole_fix_y'], m['hole_fix_z']])
    m['atoms'] = []
    for ia in range(m['n_atoms']):
        m['atoms'].append({
            'label': m[f'atom_label_{ia}'],
            'pos': np.array([float(m[f'atom_x_{ia}']), float(m[f'atom_y_{ia}']), float(m[f'atom_z_{ia}'])])
        })
    return m

def load_data(fn):
    return np.array([[float(x) for x in l.split()] for l in open(fn) if not l.startswith('#') and l.strip()])

def plot(data, m, out=None):
    """
    Generate the 2D cross-section plot.

    Uses Gram-Schmidt to handle non-orthogonal lattice vectors:
      X_plot = r · u_hat       (distance along u)
      Y_plot = r · v_perp_hat  (distance perpendicular to u, in uv-plane)
    Atom and boundary coordinates are projected using the same transformation.
    """
    nu, nv = data.shape
    nk_u, nk_v = m['nk_u'], m['nk_v']
    u_vec, v_vec = m['u_vec'], m['v_vec']
    umin, umax = m['u_min'], m['u_max']
    vmin, vmax = m['v_min'], m['v_max']

    # Non-orthogonal uv → 2D plot coordinates via Gram-Schmidt
    u_len = np.linalg.norm(u_vec); v_len = np.linalg.norm(v_vec)
    u_hat = u_vec / u_len; v_hat = v_vec / v_len
    v_perp = v_vec - np.dot(v_vec, u_hat) * u_hat  # component of v ⊥ u
    v_perp_len = np.linalg.norm(v_perp); v_perp_hat = v_perp / v_perp_len
    v_dot_u = np.dot(v_vec, u_hat)  # cross-term for non-orthogonal X coordinate
    angle_uv = np.degrees(np.arccos(max(-1, min(1, np.dot(u_hat, v_hat)))))

    # Grid edge coordinates for pcolormesh
    du = (umax - umin) / (nu - 1.0)
    dv = (vmax - vmin) / (nv - 1.0)
    Xe = np.zeros((nu+1, nv+1)); Ye = np.zeros((nu+1, nv+1))
    for iu in range(nu+1):
        uf = umin + (iu - 0.5) * du
        for iv in range(nv+1):
            vf = vmin + (iv - 0.5) * dv
            r = uf * u_vec + vf * v_vec
            Xe[iu, iv] = np.dot(r, u_hat)
            Ye[iu, iv] = np.dot(r, v_perp_hat)

    fig, ax = plt.subplots(figsize=(12, 10))

    vmx = data.max()
    if vmx <= 0:
        raise ValueError("Density data contains no positive values")
    vmn = max(data[data > 0].min(), vmx * 1e-6)
    im = ax.pcolormesh(Xe, Ye, data.T, cmap='hot', norm=LogNorm(vmin=vmn, vmax=vmx),
                       shading='flat', rasterized=True)
    density_kind = m.get('density_kind', 'conditional')
    if density_kind in ('conditional', 'conditional_elec'):
        cbar_label = r'$|\psi_{\rm cond}|^2$'
        title_kind = 'Conditional Electron Density'
    elif density_kind == 'conditional_hole':
        cbar_label = r'$|\psi_{\rm cond}|^2$'
        title_kind = 'Conditional Hole Density'
    elif density_kind == 'average_hole':
        cbar_label = r'$\rho_{\rm h}^{\rm avg}$'
        title_kind = 'Average Hole Density'
    elif density_kind == 'average_elec':
        cbar_label = r'$\rho_{\rm e}^{\rm avg}$'
        title_kind = 'Average Electron Density'
    else:
        cbar_label = density_kind
        title_kind = density_kind.replace('_', ' ').title()
    plt.colorbar(im, ax=ax, label=cbar_label, shrink=0.78)

    # ---- Unit cell boundaries: white dashed, ALL cells within view range ----
    for iu in range(int(np.floor(umin)), int(np.ceil(umax)) + 1):
        rs = iu * u_vec + vmin * v_vec
        re = iu * u_vec + vmax * v_vec
        ax.plot([np.dot(rs, u_hat), np.dot(re, u_hat)],
                [np.dot(rs, v_perp_hat), np.dot(re, v_perp_hat)],
                color='black', ls='--', lw=0.6, alpha=0.55)
    for iv in range(int(np.floor(vmin)), int(np.ceil(vmax)) + 1):
        rs = umin * u_vec + iv * v_vec
        re = umax * u_vec + iv * v_vec
        ax.plot([np.dot(rs, u_hat), np.dot(re, u_hat)],
                [np.dot(rs, v_perp_hat), np.dot(re, v_perp_hat)],
                color='black', ls='--', lw=0.6, alpha=0.55)

    # ---- BvK supercell boundaries: solid, centered on home cell ----
    # Home cell at [0,1); BvK period = nk_u. Center block: [-nk_u//2, nk_u//2 + nk_u%2)
    bvk_u0 = -(nk_u // 2)
    bvk_v0 = -(nk_v // 2)
    for iu in [bvk_u0, bvk_u0 + nk_u]:
        rs = iu * u_vec + vmin * v_vec
        re = iu * u_vec + vmax * v_vec
        ax.plot([np.dot(rs, u_hat), np.dot(re, u_hat)],
                [np.dot(rs, v_perp_hat), np.dot(re, v_perp_hat)],
                'k-', lw=2.8, alpha=0.94)
    for iv in [bvk_v0, bvk_v0 + nk_v]:
        rs = umin * u_vec + iv * v_vec
        re = umax * u_vec + iv * v_vec
        ax.plot([np.dot(rs, u_hat), np.dot(re, u_hat)],
                [np.dot(rs, v_perp_hat), np.dot(re, v_perp_hat)],
                'k-', lw=2.8, alpha=0.94)

    # ---- Atom projections (replicated over BvK supercell) ----
    plane = m.get('plane', 'ab')
    perp = m['c'] if plane == 'ab' else (m['a'] if plane == 'bc' else m['b'])
    p_len = np.linalg.norm(perp); p_hat = perp / p_len

    labels = sorted(set(a['label'] for a in m['atoms']))
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(labels))))
    lc = {l: colors[i] for i, l in enumerate(labels)}

    for atom in m['atoms']:
        pos = atom['pos']
        X_home = np.dot(pos, u_hat); Y_home = np.dot(pos, v_perp_hat)
        depth = np.dot(pos, p_hat) - m.get('slice_pos', 0)
        depth = ((depth + p_len/2) % p_len) - p_len/2
        alpha = max(0.2, 1.0 - abs(depth) / (p_len * 0.5))
        # Replicate over ALL visible cells (not just BvK)
        for ni in range(int(np.floor(umin)), int(np.ceil(umax))):
            for nj in range(int(np.floor(vmin)), int(np.ceil(vmax))):
                ax.plot(X_home + ni*u_len + nj*v_dot_u,
                        Y_home + nj*v_perp_len, 'o', color=lc[atom['label']],
                        ms=6, alpha=alpha, mec='white', mew=0.5)

    fixed_particle = m.get('fixed_particle', 'hole')
    # ---- Fixed-particle position: ONLY in home cell, as a green circle ----
    if m.get('has_hole', 1):
        hole = m['hole']
        X_hole_home = np.dot(hole, u_hat) % u_len
        Y_hole_home = np.dot(hole, v_perp_hat) % v_perp_len
        ax.plot(X_hole_home, Y_hole_home, 'o', color='lime', ms=10,
                mec='black', mew=1.5, zorder=10, label=f'Fixed {fixed_particle}')

    # Auto-crop to density extent, keeping home cell centered
    # Find data extents where density > threshold
    thresh = vmx * 1e-4
    nonzero_u = np.any(data > thresh, axis=1)
    nonzero_v = np.any(data > thresh, axis=0)
    if np.any(nonzero_u) and np.any(nonzero_v):
        iu_min = max(0, np.argmax(nonzero_u) - 2)
        iu_max = min(nu-1, nu - np.argmax(nonzero_u[::-1]) + 2)
        iv_min = max(0, np.argmax(nonzero_v) - 2)
        iv_max = min(nv-1, nv - np.argmax(nonzero_v[::-1]) + 2)
        # Convert to plot coords
        uf = lambda i: umin + (umax-umin)*i/(nu-1.)
        vf = lambda i: vmin + (vmax-vmin)*i/(nv-1.)
        d_Xmin = uf(iu_min)*u_len + vf(iv_min)*v_dot_u
        d_Xmax = uf(iu_max)*u_len + vf(iv_max)*v_dot_u
        d_Ymin = vf(iv_min)*v_perp_len
        d_Ymax = vf(iv_max)*v_perp_len
        # Expand symmetrically around home cell center
        hc_X = 0.5 * u_len; hc_Y = 0.5 * v_perp_len
        hw = max(d_Xmax - hc_X, hc_X - d_Xmin, d_Ymax - hc_Y, hc_Y - d_Ymin) + u_len * 0.6
        ax.set_xlim(hc_X - hw, hc_X + hw)
        ax.set_ylim(hc_Y - hw, hc_Y + hw)

    ax.set_xlabel(f'Distance along u = a  (|u| = {u_len:.2f} Bohr)', fontsize=11)
    ax.set_ylabel(f'Distance ⊥ u in uv-plane  (|v| = {v_len:.2f} Bohr, '
                  f'∠(u,v) = {angle_uv:.0f}°)', fontsize=11)
    ax.set_aspect('equal')

    h = m['hole']
    hole_text = (f'{fixed_particle.title()} at ({h[0]:.2f}, {h[1]:.2f}, {h[2]:.2f}) Bohr  '
                 if m.get('has_hole', 1) else '')
    marker_text = '  Fixed-hole marker: green circle' if m.get('has_hole', 1) else ''
    ax.set_title(f'{title_kind} | State {m["state"]} | {plane}-plane\n'
                 f'{hole_text}E = {m["energy_Ry"]:.4f} Ry = {m["energy_Ry"]*13.6057:.3f} eV\n'
                 f'BvK {nk_u}x{nk_v}  ({nu}x{nv} pts)  '
                 f'|a|={np.linalg.norm(m["a"]):.1f}  |b|={np.linalg.norm(m["b"]):.1f} Bohr\n'
                 f'Solid lines: BvK supercell. Dashed lines: unit cells.{marker_text}',
                 fontsize=11)

    # Legend
    handles = []
    if m.get('has_hole', 1):
        handles.append(Line2D([0],[0], marker='o', color='w', markerfacecolor='lime',
                              markersize=8, markeredgecolor='black', markeredgewidth=1.5,
                              label=f'Fixed {fixed_particle} (home cell)'))
    for l in labels:
        handles.append(Line2D([0],[0], marker='o', color='w', markerfacecolor=lc[l],
                              markersize=7, markeredgecolor='white', markeredgewidth=0.5,
                              label=l))
    ax.legend(loc='upper right', fontsize=9, handles=handles)
    plt.tight_layout()
    fname = out
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f"Saved {fname}")
    plt.close(fig)

if __name__ == '__main__':
    fn = sys.argv[1] if len(sys.argv) > 1 else 'Exciton_cond_slice_state0.dat'
    meta = parse(fn); data = load_data(fn)
    print(f"{meta.get('density_kind', 'conditional')}  Grid {data.shape}  "
          f"BvK {meta['nk_u']}x{meta['nk_v']}  "
          f"|a|={np.linalg.norm(meta['a']):.1f}  |b|={np.linalg.norm(meta['b']):.1f} Bohr  "
          f"∠(a,b)={np.degrees(np.arccos(np.dot(meta['a'],meta['b']) / (np.linalg.norm(meta['a'])*np.linalg.norm(meta['b'])))):.0f}°")
    outfn = fn.replace('.dat', '.png')
    plot(data, meta, outfn)
