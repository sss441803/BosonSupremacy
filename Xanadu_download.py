import os
import requests
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, help="Experiment directory.")
args = vars(parser.parse_args())

dir = args['dir']

# s3 bucket where data is stored
url = "https://qca-data.s3.amazonaws.com/"


# remote list of folders and files to be downloaded
# You may only need fig3a, fig3b, fig4, figS15
folders_list = ["fig2/", "fig3a/", "fig3b/", "fig4/", "figS15/"]


# You may only need samples.npy
files_list = ["program_params.json", "U.npy", "T.npy", "r.npy", "samples.npy"]

for folder in folders_list:
    local_folder_path = dir+folder
    if not os.path.exists(local_folder_path):
        os.makedirs(local_folder_path)
    for file in files_list:
        file_path = folder+file
        local_file_path = local_folder_path + file
        r = requests.get(url+file_path)
        open(local_file_path,"wb").write(r.content)