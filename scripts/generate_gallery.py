import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from argparse import ArgumentParser
import os
import glob
import time
import datetime

import sys
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src'))
sys.path.append(src_dir)
from gallery import Gallery

def parse_args():
    """ Parser for command line arguments.
    """
    parser = ArgumentParser(description="Generate galleries of per-particle minislabs.")
    parser.add_argument("--stardir", type=str, required=True, 
                        help="Path to directory containing star files")
    parser.add_argument("--volumes", type=str, required=True,
                        help="Tomogram directory")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output folder for micrographs and bookkeeping")
    parser.add_argument("--extract_shape", type=int, nargs=3, required=True,
                        help="Subvolume extraction shape in (x,y,z) in Angstrom")
    parser.add_argument("--gallery_shape", type=int, nargs=2, required=False, default=[16,15],
                        help="Number of gallery particles in (row,col) format")
    parser.add_argument("--coords_scale_factor", type=float, required=False, default=1,
                        help="Multiplicative factor for coords if picked and extract tomograms don't match")
    parser.add_argument("--live", required=False, action="store_true",
                        help="Live processing mode, generating one gallery per tomogram")
    parser.add_argument("--t_interval", required=False, type=float, default=300,
                        help="Interval in seconds between checking for new files")
    parser.add_argument("--t_exit", required=False, type=float, default=1800,
                        help="Interval in seconds after which to exit if new files not found")
    
    return parser.parse_args()

def main(config):

    if config.live:
        start_time = time.time()
        processed = []
        
        while True:
            fnames = glob.glob(f"{config.stardir}/*/*.star")
            fnames = [fn for fn in fnames if os.path.basename(fn) not in processed]
            print(f"{datetime.datetime.fromtimestamp(time.time())}: Found {len(fnames)} new files to process")
            if len(fnames) > 0:
                for star_path in fnames:
                    filetag = star_path.split("/")[-2]
                    tomo_path = os.path.join(config.volumes, f"{filetag}_Vol.mrc")
                    if not os.path.exists(tomo_path):
                        print(f"Warning: could not find {tomo_path}")
                        continue
                    mosaic = Gallery(config.extract_shape, coords_scale_factor=config.coords_scale_factor)
                    mosaic.generate_slabs(tomo_path, star_path)
                    mosaic.generate_galleries(config.gallery_shape, config.output_dir, filename=filetag)
                    processed.append(star_path)
                start_time = time.time()
            time.sleep(config.t_interval)
            t_elapsed = time.time() - start_time
            if t_elapsed > config.t_exit:
                break

    else:
        raise NotImplementedError
    
if __name__ == "__main__":
    config = parse_args()
    main(config)
