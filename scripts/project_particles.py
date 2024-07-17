from argparse import ArgumentParser
import numpy as np
import pandas as pd
import os
import sys
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src'))
sys.path.append(src_dir)
from minislab import make_particle_projections
from settings import ProcessingConfigMinislab

def parse_args():
    """ Parser for command line arguments.
    """
    parser = ArgumentParser(description="Generate starfile based on cryosparc-curated picks.")
    # basic input/output arguments
    parser.add_argument("--in_coords", type=str, required=True,
                        help="Coordinate file(s) as one or multiple starfiles or a copick config file")
    parser.add_argument("--in_vol", type=str, required=True,
                        help="Directory containing volumes or a copick config file")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Output directory for galleries and/or particle stack")

    # additional arguments related to input format
    parser.add_argument("--extract_shape", type=int, nargs=3, required=True,
                        help="Subvolume extraction shape in (x,y,z) in Angstrom")
    parser.add_argument("--voxel_spacing", type=float, required=True,
                        help="Pixel size of tomograms to extract minislabs from in Angstrom")
    parser.add_argument("--coords_scale", type=float, required=False, default=1,
                        help="Multiplicative factor to convert input coords to Angstrom")
    parser.add_argument("--col_name", type=str, required=False, default='rlnTomoName',
                        help="Tomogram column name in starfile(s)")
    parser.add_argument("--tomo_type", type=str, required=False,
                        help="Tomogram type if extracting from copick")
    parser.add_argument("--user_id", type=str, required=False,
                         help="User ID, required if coordinates from copick")
    parser.add_argument("--session_id", type=str, required=False,
                        help="Session ID, required if coordinates from copick")
    parser.add_argument("--particle_name", type=str, required=False,
                        help="Particle name, required if coordinates from copick")
    
    # optional arguments for formatting output
    parser.add_argument("--out_format", required=False, nargs="+", default=['gallery', 'stack'],
                        help="Specify output mode -- gallery and/or stack")
    parser.add_argument("--gallery_shape", type=int, nargs=2, required=False, default=[16,15],
                        help="Number of gallery particles in (row,col) format")
    parser.add_argument("--one_per_vol", required=False, action="store_true",
                        help="Generate one gallery per volume, not applicable to stacks")
    parser.add_argument("--normalize", required=False, action="store_true",
                        help="Normalize particle stacks")
    parser.add_argument("--radius", type=float, required=False, default=0.9,
                        help="Fractional radius for normalizing particle stacks")
    parser.add_argument("--invert", required=False, action="store_true",
                        help="Invert contrast of particle stacks")
    
    # arguments related to real-time mode
    parser.add_argument("--live", required=False, action="store_true",
                        help="Live processing mode, generating one gallery per tomogram")
    parser.add_argument("--t_interval", required=False, type=float, default=300,
                        help="Interval in seconds between checking for new files")
    parser.add_argument("--t_exit", required=False, type=float, default=1800,
                        help="Interval in seconds after which to exit if new files not found")
    
    return parser.parse_args()

def generate_config(config):
    """
    Store command line arguments in a json file.
    """
    d_config = vars(config)
    
    reconfig = {}
    reconfig['software'] = {'name': 'slabpick', 'version':'0.1.0'}
    reconfig['input'] = {k: d_config[k] for k in ('in_coords','in_vol')}
    reconfig['output'] = {k: d_config[k] for k in ('out_dir','out_format')}
    reconfig['mode'] = {k: d_config[k] for k in ('live','t_interval', 't_exit')}

    used_keys = [list(reconfig[key].keys()) for key in reconfig.keys()]
    used_keys = [p for param in used_keys for p in param]
    param_keys = [key for key in d_config.keys() if key not in used_keys]
    reconfig['parameters'] = {k: d_config[k] for k in param_keys}
    
    reconfig = ProcessingConfigMinislab(**reconfig)
    
    os.makedirs(config.out_dir, exist_ok=True)
    with open(os.path.join(config.out_dir, "config.json"), "w") as f:
        f.write(reconfig.model_dump_json(indent=4))

def main(config):
    
    generate_config(config)
    
    as_stack = True if 'stack' in config.out_format else False
    as_gallery = True if 'gallery' in config.out_format else False
    if not as_stack and not as_gallery:
        raise ValueError("out_format argument must contain at least one of gallery or stack")
    
    if os.path.splitext(config.in_coords)[-1] == '.json':
        if None in [config.user_id, config.session_id, config.particle_name]:
            raise ValueError("Missing session ID and/or particle name")
    if os.path.splitext(config.in_vol)[-1] == '.json':
        if config.tomo_type is None:
            raise ValueError("Missing tomogram type")
    if os.path.splitext(config.in_vol)[-1] == '.json' and os.path.splitext(config.in_coords)[-1] == '.json':
        if config.in_vol != config.in_coords:
            print("Warning! in_vol and in_coords correspond to different copick config files")

    make_particle_projections(config.in_vol,
                              config.in_coords,
                              config.out_dir,
                              config.extract_shape,
                              config.voxel_spacing,
                              coords_scale=config.coords_scale,
                              col_name=config.col_name,
                              tomo_type=config.tomo_type,
                              particle_name=config.particle_name,
                              user_id=config.user_id,
                              session_id=config.session_id,
                              as_gallery=as_gallery,
                              as_stack=as_stack,
                              gallery_shape=tuple(config.gallery_shape),
                              one_per_vol=config.one_per_vol,
                              normalize=config.normalize,
                              invert=config.invert,
                              radius=config.radius,
                              live=config.live,
                              t_interval=config.t_interval,
                              t_exit=config.t_exit)
                
if __name__ == "__main__":
    config = parse_args()
    main(config)
