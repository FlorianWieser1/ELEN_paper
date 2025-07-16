import os
import tempfile
import imageio
import MDAnalysis as mda
from MDAnalysis.analysis import align
import matplotlib.pyplot as plt

def create_md_video(topology_file,
                    trajectory_file,
                    output_video="out.mp4",
                    fps=10,
                    frame_stride=1,
                    style='lines',
                    align_selection='protein and name CA',
                    do_alignment=True,
                    atom_selection='not resname SOL'):
    """
    Creates a 3D video from an MD trajectory using Matplotlib.

    Parameters:
    -----------
    topology_file : str
        Path to the topology file (e.g. .tpr, .pdb, .gro).
    trajectory_file : str
        Path to the trajectory file (e.g. .xtc, .trr).
    output_video : str
        Filename for the output video (e.g. out.mp4).
    fps : int
        Frames per second for the output video.
    frame_stride : int
        Process every n-th frame in the trajectory to reduce total frames.
    style : str
        Visualization style. Options: ['sticks', 'lines', 'spheres'].
    align_selection : str
        Atom selection for alignment, if do_alignment=True.
    do_alignment : bool
        Whether to align each frame to the first frame using align_selection.
    atom_selection : str
        Atom selection for visualization (e.g., "protein", "not resname SOL").
        Default is 'all' (show everything).
    """

    # 1) Load the MDAnalysis Universe
    u = mda.Universe(topology_file, trajectory_file)
    print("Number of frames in the trajectory:", len(u.trajectory))

    # 2) Optionally align all frames to the first frame
    if do_alignment:
        align.AlignTraj(u, u, select=align_selection, in_memory=True).run()

    # 3) Define your visualization selection
    selection_group = u.select_atoms(atom_selection)
    if len(selection_group) == 0:
        print(f"WARNING: Selection '{atom_selection}' returned 0 atoms.")
        print("Check your residue names or selection string.")
        return

    # 4) Create a temporary directory to store intermediate frame images
    with tempfile.TemporaryDirectory() as tmpdirname:
        frame_files = []

        # Check whether the selection has bond information
        # (some file formats may not provide it)
        has_bonds = (len(selection_group.bonds) > 0)

        # 5) Loop over the trajectory
        for ts in u.trajectory[::frame_stride]:
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')

            # Get coordinates from the selection (not the full universe)
            coords = selection_group.positions

            # Simple 3D visualization
            if style == 'sticks' and has_bonds:
                # Draw each bond in the selection
                for bond in selection_group.bonds:
                    atom1_pos = bond.atoms[0].position
                    atom2_pos = bond.atoms[1].position
                    ax.plot([atom1_pos[0], atom2_pos[0]],
                            [atom1_pos[1], atom2_pos[1]],
                            [atom1_pos[2], atom2_pos[2]],
                            color='gray')
                # Also scatter the atom positions
                ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c='blue', s=20)

            elif style == 'lines':
                ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], color='blue')

            elif style == 'spheres':
                ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c='red', s=50)

            else:
                # Fallback style: simple scatter
                ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c='blue', s=20)

            # 6) Define plot limits with a small buffer
            buffer = 2.0
            x_min, x_max = coords[:, 0].min() - buffer, coords[:, 0].max() + buffer
            y_min, y_max = coords[:, 1].min() - buffer, coords[:, 1].max() + buffer
            z_min, z_max = coords[:, 2].min() - buffer, coords[:, 2].max() + buffer
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_zlim(z_min, z_max)

            # Hide axes and label
            ax.set_title(f"Frame {ts.frame}")
            ax.axis('off')

            # 7) Save the current frame to a temporary image
            frame_path = os.path.join(tmpdirname, f"frame_{ts.frame:05d}.png")
            plt.savefig(frame_path, dpi=100)
            plt.close(fig)
            frame_files.append(frame_path)

        # 8) Compile all saved frames into a video using imageio
        with imageio.get_writer(output_video, mode='I', fps=fps) as writer:
            for frame_file in frame_files:
                image = imageio.v2.imread(frame_file)
                writer.append_data(image)

    print(f"Video saved to {output_video}")
    
import os
import tempfile

import MDAnalysis as mda
from MDAnalysis.analysis import align

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
import matplotlib.collections as mcoll

import imageio


def create_md_video2(topology_file,
                    trajectory_file,
                    output_video="out.mp4",
                    fps=10,
                    frame_stride=1,
                    style='sticks',
                    align_selection='protein and name CA',
                    do_alignment=True,
                    atom_selection='not resname SOL',
                    color_by='element',
                    background_color='black',
                    bond_color='white',
                    atom_size=20,
                    dpi=150):
    """
    Creates a 3D video from an MD trajectory using Matplotlib with improved visuals.

    Parameters
    ----------
    topology_file : str
        Path to the topology file (e.g. .tpr, .pdb, .gro).
    trajectory_file : str
        Path to the trajectory file (e.g. .xtc, .trr).
    output_video : str
        Filename for the output video (e.g. out.mp4).
    fps : int
        Frames per second for the output video.
    frame_stride : int
        Process every n-th frame in the trajectory to reduce total frames.
    style : str
        Visualization style. Options: ['sticks', 'lines', 'spheres'].
    align_selection : str
        Atom selection for alignment, if do_alignment=True.
    do_alignment : bool
        Whether to align each frame to the first frame using align_selection.
    atom_selection : str
        Atom selection for visualization (e.g., "protein", "not resname SOL").
        Default is 'all' (show everything).
    color_by : str
        How to color the atoms. Options: ['element', 'single_color', 'residue'].
    background_color : str
        Matplotlib color for background (e.g., 'black', 'white', '#222222').
    bond_color : str
        Matplotlib color for bonds (applicable to 'sticks' style).
    atom_size : float
        Marker size for the atoms in scatter plots.
    dpi : int
        Resolution in dots per inch for each saved frame (higher -> sharper image).
    """

    # A simple dictionary for coloring by element
    # Extend this dictionary for more elements as needed
    element_colors = {
        'H': 'white',
        'C': 'gray',
        'N': 'blue',
        'O': 'red',
        'S': 'yellow',
        'P': 'orange'
    }

    # 1) Load the MDAnalysis Universe
    u = mda.Universe(topology_file, trajectory_file)
    print("Number of frames in the trajectory:", len(u.trajectory))

    # 2) Optionally align all frames to the first frame
    if do_alignment:
        print(f"Aligning all frames to the first frame using selection: {align_selection}")
        align.AlignTraj(u, u, select=align_selection, in_memory=True).run()

    # 3) Define your visualization selection
    selection_group = u.select_atoms(atom_selection)
    if len(selection_group) == 0:
        print(f"WARNING: Selection '{atom_selection}' returned 0 atoms.")
        print("Check your residue names or selection string.")
        return

    # Check whether the selection has bond information
    has_bonds = (len(selection_group.bonds) > 0)

    # 4) Prepare a temporary directory for frame images
    with tempfile.TemporaryDirectory() as tmpdirname:
        frame_files = []

        # 5) Loop over the trajectory frames
        for ts in u.trajectory[::frame_stride]:
            fig = plt.figure(figsize=(8, 8), dpi=dpi)
            ax = fig.add_subplot(111, projection='3d')

            # Set a consistent camera angle for all frames
            ax.view_init(elev=30, azim=-60)

            # Set background color (axis face color)
            ax.set_facecolor(background_color)
            fig.patch.set_facecolor(background_color)

            # Extract coordinates
            coords = selection_group.positions

            # Optionally define a color array for each atom
            if color_by == 'element':
                # MDA tries to guess the element via atom name by default.
                # Fallback to 'C' if unknown.
                atom_elements = [a.element.upper() if a.element else 'C'
                                 for a in selection_group.atoms]
                colors = [element_colors.get(el, 'magenta') for el in atom_elements]
            elif color_by == 'single_color':
                colors = ['blue'] * len(selection_group)
            elif color_by == 'residue':
                # Each residue gets a unique color
                unique_resids = sorted(set(selection_group.resids))
                cmap = plt.cm.get_cmap('rainbow', len(unique_resids))
                resid_to_color = {r: cmap(i) for i, r in enumerate(unique_resids)}
                colors = [resid_to_color[r] for r in selection_group.resids]
            else:
                # Fallback
                colors = ['blue'] * len(selection_group)

            # 6) Render the selection in the chosen style
            if style == 'sticks' and has_bonds:
                # Draw bonds
                lines = []
                for bond in selection_group.bonds:
                    at1, at2 = bond.atoms
                    lines.append([at1.position, at2.position])

                # Create a Line3DCollection for all bonds at once
                bond_collection = mcoll.Line3DCollection(lines, colors=bond_color, linewidths=1)
                ax.add_collection3d(bond_collection)

                # Draw atoms (small spheres)
                ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                           c=colors, s=atom_size, alpha=1.0)

            elif style == 'lines':
                # Connect all selected atoms in order
                # Typically not very meaningful for all atoms,
                # but can be interesting for C-alpha backbone, etc.
                ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], color='white', linewidth=2)

            elif style == 'spheres':
                # Scatter with bigger spheres
                ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                           c=colors, s=atom_size, alpha=1.0)

            else:
                # Fallback style: scatter
                ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                           c=colors, s=atom_size, alpha=1.0)

            # 7) Define plot limits with a small buffer
            buffer = 2.0
            x_min, x_max = coords[:, 0].min() - buffer, coords[:, 0].max() + buffer
            y_min, y_max = coords[:, 1].min() - buffer, coords[:, 1].max() + buffer
            z_min, z_max = coords[:, 2].min() - buffer, coords[:, 2].max() + buffer
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_zlim(z_min, z_max)

            # Hide axis lines and ticks
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.set_title(f"Frame {ts.frame}", color='white', pad=20)

            # 8) Save the current frame to a temporary image
            frame_path = os.path.join(tmpdirname, f"frame_{ts.frame:05d}.png")
            plt.savefig(frame_path, bbox_inches='tight', facecolor=fig.get_facecolor())
            plt.close(fig)
            frame_files.append(frame_path)

        # 9) Compile all saved frames into a video using imageio
        with imageio.get_writer(output_video, mode='I', fps=fps) as writer:
            for frame_file in frame_files:
                image = imageio.v2.imread(frame_file)
                writer.append_data(image)

    print(f"Video saved to {output_video}")