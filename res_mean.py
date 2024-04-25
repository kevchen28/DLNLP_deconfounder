import pandas as pd
import numpy as np
import glob
import argparse
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", help="Output directory", required=True)
    parser.add_argument(
        "-d",
        "--directory",
        help="Input directory name containing experiment csv",
        required=True,
    )
    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # Read all the csv files in the input directory
    path = args.directory
    all_files = glob.glob(path + "/*.csv")

    # Create empty lists to store the mean pehe and mean mae_ate values
    mean_pehe_list = []
    mean_mae_ate_list = []

    experiments_df = pd.DataFrame(
        columns=[
            "hidden",
            "dropout",
            "epochs",
            "weight_decay",
            "nin",
            "nout",
            "alpha",
            "ipm",
            "mean_pehe",
            "mean_mae_ate",
        ]
    )

    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)

        # Caluculate the mean of the second last column
        mean_pehe = np.mean(df.iloc[:, -2])
        mean_mae_ate = np.mean(df.iloc[:, -1])

        hidden = df.iloc[0, 1]
        dropout = df.iloc[0, 2]
        epochs = df.iloc[0, 3]
        weight_decay = df.iloc[0, 4]
        nin = df.iloc[0, 5]
        nout = df.iloc[0, 6]
        alpha = df.iloc[0, 7]
        ipm = df.iloc[0, 8]

        temp_df = pd.DataFrame(
            {
                "hidden": hidden,
                "dropout": dropout,
                "epochs": epochs,
                "weight_decay": weight_decay,
                "nin": nin,
                "nout": nout,
                "alpha": alpha,
                "ipm": ipm,
                "mean_pehe": mean_pehe,
                "mean_mae_ate": mean_mae_ate,
            },
            index=[0],
        )
        experiments_df = pd.concat([experiments_df, temp_df], ignore_index=True)

    # Save the dataframe to a csv file
    experiments_df.to_csv(os.path.join(args.output, "experiments.csv"), index=False)
    
    # Code to run this script
    # python res_mean.py -d new_results/BlogCatalog2 -o new_results/BlogCatalog2
