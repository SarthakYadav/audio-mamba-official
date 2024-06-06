import os
import glob
import subprocess as sp
import tqdm
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--base_dir", type=str)
parser.add_argument("--tgt_dir", type=str)


args = parser.parse_args()
base_dir = args.base_dir
tgt_dir = args.tgt_dir

if not os.path.exists(tgt_dir):
   os.makedirs(tgt_dir)


def setup_paths(fld, num_seeds):
    print("Setting up fld:", fld)
    for ix in range(num_seeds):
        fldname = fld.split("/")[-1]
        # 
        print("\tfldname:", fldname)
        for task in glob.glob(os.path.join(fld, "*")):
            out_base_dir = os.path.join(tgt_dir, "{}_r1_{:02d}".format(fldname, ix), task.split("/")[-1])
            if os.path.exists(out_base_dir):
               print(f"{out_base_dir} exists..")
               continue
            os.makedirs(out_base_dir)
            print("\t\tout_base_dir:", out_base_dir)

            for file in glob.glob(os.path.join(task, "*")):
                tgt_p = file.replace(task, out_base_dir)
                sp.call("ln -s {} {}".format(file, tgt_p), shell=True)

print("BASE_DIR:", base_dir)
print("TGT_DIR:", tgt_dir)

flds = glob.glob(os.path.join(base_dir, "*"))
for fld in flds:
    setup_paths(fld, 10)
