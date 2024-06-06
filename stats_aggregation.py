import os
import glob
import numpy as np
import pandas as pd
import argparse
import json
from scipy import stats as st


def parse(f):
    with open(f, "r") as fd:
        data = json.load(fd)
    if "test" in data:
        sc = data["test"]['test_score']
    elif "aggregated_scores" in data:
        sc = data['aggregated_scores']['test_score_mean']
    else:
        raise ValueError("not found ")
    return sc


def get_info(f):
    f = f.replace("_bugfix", "")
    splitted = f.split("/")[:-1]
    dataset = splitted[-1]
    exp = splitted[-2]
    print(dataset, exp)
    print(exp)
    exp_info = exp.split("_")[-1]
    # print(exp_info)
    if "r" not in exp_info:
        runid = int(exp_info)
        exp = "_".join(exp.split("_")[:-1])
    else:
        # print("HAD RUN ID==1")
        runid = 1
    print(exp, runid)

    ssl_run_id = int(exp[-1])
    print(ssl_run_id)
    exp = exp[:-3]

    return {
        "exp": exp,
        "ssl_run": ssl_run_id,
        "run": runid,
        "dataset": dataset
    }
    # return exp, ssl_run_id, runid, dataset


def get_stats(x, conf=0.95):
    l, h = st.t.interval(conf, len(x)-1, loc=np.mean(x), scale=st.sem(x))
    s = (h - l) / 2
    m = h - s
    return m, s


def get_overall_stats(df, exp, ssl_run=None, run=None, dataset=None):
    subdf = df[df['exp'] == exp]
    if dataset:
        subdf = subdf[subdf['dataset'] == dataset]
    if ssl_run:
        subdf = subdf[subdf['ssl_run'] == ssl_run]
    if run:
        subdf = subdf[subdf['run'] == run]

    mean, std = get_stats(subdf['scores'].values)
    return mean, std, subdf


parser = argparse.ArgumentParser()
parser.add_argument("--base_dir", type=str)
parser.add_argument("--output_dir", type=str)

if __name__ == "__main__":
    args = parser.parse_args()
    files = glob.glob(os.path.join(args.base_dir, "*", "*/test.predicted-scores.json"))
    records = []
    for f in files:
        rec = get_info(f)
        sc = parse(f)
        rec['scores'] = sc
        records.append(rec)

    df = pd.DataFrame(records)
    unique_exps = np.unique(df['exp'].values)
    exp_map = {}
    for ix in range(len(unique_exps)):
        exp_map[unique_exps[ix]] = ix
    df['mapped_exp'] = df['exp'].apply(lambda x: exp_map[x])
    unique_datasets = np.unique(df['dataset'].values)
    exp_dset = {}
    for ix in range(len(unique_datasets)):
        exp_dset[unique_datasets[ix]] = ix

    dset_map = {
        "beijing_opera-v1.0-hear2021-full": "Beijing-Opera",
        "esc50-v2.0.0-full": "ESC-50",
        "libricount-v1.0.0-hear2021-full": "LibriCount",
        "mridangam_stroke-v1.5-full": "Mridangam-S",
        "mridangam_tonic-v1.5-full": "Mridangam-T",
        "nsynth_pitch-v2.2.3-5h": "NSynth-Pitch-5h",
        "speech_commands-v0.0.2-5h": "SpeechCommands-5h",
        "tfds_crema_d-1.0.0-full": "CREMA-D",
        "vox_lingua_top10-hear2021-full": "VoxLingua",
        "fsd50k-v1.0-full": "FSD50k",
        "gunshot_triangulation-v1.0-full": "Gunshot"
    }
    output_fld = args.output_dir
    os.makedirs(output_fld, exist_ok=True)

    for uq_exp in unique_exps:
        out_dir = os.path.join(output_fld, uq_exp)
        os.makedirs(out_dir, exist_ok=True)
        for uq_ds in unique_datasets:
            mean, std, subdf = get_overall_stats(df, uq_exp, dataset=uq_ds)
            # print("\t", uq_exp, uq_ds, mean, std)
            if len(subdf) == 0:
                print("NO DATA FOR ", uq_exp, uq_ds)
                continue
            all_scores = subdf.scores.values
            scores_txt = os.path.join(out_dir, f"{dset_map[uq_ds]}_scores.txt")
            stats_txt = os.path.join(out_dir, f"{dset_map[uq_ds]}_stats.txt")
            with open(scores_txt, "w") as fd:
                fd.writelines(["{}\n".format(sc) for sc in all_scores])
            with open(stats_txt, "w") as fd:
                lines = [
                    "{:.05f}\n".format(mean),
                    "{:.05f}\n".format(std)
                ]
                fd.writelines(lines)

    df.to_csv(os.path.join(output_fld, "all_results.csv"), index=False)
