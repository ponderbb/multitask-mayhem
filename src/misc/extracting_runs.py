from pathlib import Path
from tqdm import tqdm
import shutil
import logging
import pandas as pd

WORKDIR_HPC = "/work3/s202821/models/"
OUTDIR_HPC = "/work3/s202821/emp_models/"

RUNS_LIST = [
    "frcnn-hybrid_2v1_23-02-21T182529",
    "frcnn-hybrid_dwa-10v1-cagrad_23-02-21T182427",
    "frcnn-hybrid_dwa-10v1-graddrop_23-02-21T182345",
    "frcnn-hybrid_dwa-10v1_23-02-21T182344",
    "frcnn-hybrid_dwa-10v1-pcgrad_23-02-21T144644",
    "frcnn-hybrid_bmtl-graddrop_23-02-20T140735",
    "frcnn-hybrid_15v1_23-02-20T132858",
    "frcnn-hybrid_bmtl-cagrad_23-02-20T132609",
    "frcnn-hybrid_bmtl-pcgrad_23-02-20T132549",
    "frcnn-hybrid_10v1-graddrop_23-02-19T151850",
    "frcnn-hybrid_10v1-cagrad_23-02-19T151821",
    "frcnn-hybrid_10v1-pcgrad_23-02-19T151738",
    "frcnn-hybrid_dwa-cagrad_23-02-16T235038",
    "frcnn-hybrid_dwa-pcgrad_23-02-16T235026",
    "frcnn-hybrid_dwa_23-02-16T234921",
    "frcnn-hybrid_dwa-graddrop_23-02-16T234854",
    "frcnn-hybrid_relative_23-02-16T004645",
    "frcnn-hybrid_bmtl_23-02-16T003206",
    "frcnn-hybrid_5v1_23-02-16T001810",
    "frcnn-hybrid_10v1_23-02-16T001236",
    "frcnn-hybrid_cagrad_23-02-15T235953",
    "frcnn-hybrid_graddrop_23-02-15T235924",
    "frcnn-hybrid_pcgrad_23-02-15T155512",
    "frcnn-hybrid_equal_23-02-14T235356",
    "ssdlite-hybrid_equal_23-02-14T235500",
    "lraspp-hybrid_equal_23-02-14T235345",
    "deeplabv3_23-02-14T235445",
    "ssdlite_23-02-14T235429",
    "frcnn_23-02-14T235417",
    "lraspp_23-02-25T215937",
    "frcnn-hybrid_uc-cagrad_23-02-25T204529",
    "frcnn-hybrid_uc-pcgrad_23-02-25T204444",
    "frcnn-hybrid_uc-graddrop_23-02-25T204407",
    "frcnn-hybrid_uc_23-02-25T204205"
]

def extract_runs(workdir: str , outdir: str , runs_list: list = RUNS_LIST, force_rewrite: bool = False):

    for run in tqdm(runs_list):
        assert Path(workdir + run).exists(), f"{run} does not exist"
        outdir_exists = Path(outdir + run).exists()

        if force_rewrite or not(outdir_exists):
            logging.info(f"Copy {run} to {outdir}")
            shutil.copytree(workdir + run, outdir + run)
        else:
            logging.info(f"Skipping {run}")
    
    df = pd.DataFrame(runs_list)
    df.to_csv(outdir+"runs_list.csv", header=["name"])

    
