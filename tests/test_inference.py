#!/home/florian_wieser/miniconda3/envs/elen_inference/bin/python3
import os
import sys
import argparse
import pandas as pd
from elen.inference.run_elen_inference import run_inference
from elen.config import PATH_ELEN_MODELS

def test_run_inference_smoketest(inpath, outpath, elen_models):
    print("Running ELEN inference smoketest...")
    # Model and config paths (mock or point to test models/weights)
    # feature modes: ['full', 'no_saprot', 'geom_only', 'saprot_only']
    run_inference(
        input_dir=inpath,
        output_dir=outpath,
        elen_models_dir=PATH_ELEN_MODELS,
        elen_models=elen_models,
        feature_mode="full",
        elen_score_to_pdb="lddt_cad",  # Use a valid score type
        pocket_type="RP",
        loop_max_size=10,
        ss_frag_size=2,
        nr_residues=30,
        batch_size=1,
        num_workers=0,
        overwrite=True,
        path_saprot_embeddings=None,  # Assuming no embeddings for smoketest
        save_features=False,  # Print features for testing
    )
        
def test_elen_scores_exact(outpath, elen_models):
    df_ref = pd.read_csv("elen_scores_RP_1ubq_save.csv")
    print("Reference ELEN scores:")
    print(df_ref)
    df = pd.read_csv(f"{outpath}/elen_results_ELEN_full/elen_scores_RP.csv")
    print(df)
    pd.testing.assert_frame_equal(df, df_ref)
    
###############################################################################
def main(args):
    test_run_inference_smoketest(args.inpath, args.outpath, args.elen_models)
    test_elen_scores_exact(args.outpath, args.elen_models)
    print("Inference test passed successfully.")

############################################################################### 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ELEN inference test")
    parser.add_argument("--inpath", type=str, default="input", help="Input path for test files")
    parser.add_argument("--outpath", type=str, default="out", help="Output path for results")
    parser.add_argument('--elen_models', nargs='+', default=None)
    args = parser.parse_args()
    main(args) 
