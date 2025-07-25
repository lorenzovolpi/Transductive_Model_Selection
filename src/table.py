import os
from argparse import ArgumentParser
from typing import Literal

import pandas as pd
from pandatex import Format, Table

import env
from config import get_acc_names, get_all_dataset_names
from results import Results
from util import decorate_dataset, decorate_datasets, rename_datasets, rename_methods

method_map = {
    "LR": "$\\varnothing$-LR",
    "kNN": "$\\varnothing$-$k$NN",
    "SVM": "$\\varnothing$-SVM",
    "MLP": "$\\varnothing$-MLP",
    "SVM-t": "$\\varnothing$-TSVM",
    "Naive-LR": "IMS-LR",
    "Naive-kNN": "IMS-$k$NN",
    "Naive-SVM": "IMS-SVM",
    "Naive-MLP": "IMS-MLP",
    "Naive": "IMS-$\\forall$",
    "O-LEAP(KDEy)": "LEAP-$\\forall$",
}

dataset_map = {
    "poker_hand": "poker-hand",
    "hand_digits": "hand-digits",
    "page_block": "page-block",
    "image_seg": "image-seg",
}


def tables(experiment: Literal["transd", "hoptim"]):
    if experiment == "transd":
        ea_label = "estim_accs"
    else:
        print(f"Invalid experiment '{experiment}'; aborting.")
        return

    base_dir = os.path.join(env.root_dir, experiment)

    def add_to_table(tbl: Table, df: pd.DataFrame, dataset, methods):
        for method in methods:
            values = df.loc[df["method"] == method, "true_accs"].to_numpy()
            for v in values:
                tbl.add(dataset, method, v)
        return tbl

    tbls = []
    accs = get_acc_names()
    datasets = get_all_dataset_names()
    for acc in accs:
        name = f"{experiment}_{acc}"
        tbl = Table(name=name)
        tbl.format = Format(
            lower_is_better=False,
            mean_prec=3,
            show_std=True,
            remove_zero=True,
            with_rank_mean=False,
            with_mean=True,
            mean_macro=False,
            color=True,
            color_mode="local",
            simple_stat=True,
        )
        for dataset in datasets:
            res = (
                Results.load(base_dir=base_dir, acc_name=acc, dataset=dataset, set_problem=False)
                # .split_by_shift(prevs=0.5)
                .model_selection(oracle=False, only_default=True, ea_label=ea_label)
                .map_column_values("method", method_map)
                .map_column_values("dataset", dataset_map)
                .apply_to_column("dataset", decorate_dataset)
            )
            _methods = [method_map.get(m, m) for m in res.unique_column_values("method")]
            _dataset = decorate_dataset(dataset_map.get(dataset, dataset))
            tbl = add_to_table(tbl, res.df, _dataset, _methods)
            print(f"{dataset} done")

        tbls.append(tbl)
        print(f"table {name} genned")

    table_dir = os.path.join(env.root_dir, "tables")
    os.makedirs(table_dir, exist_ok=True)
    pdf_path = os.path.join(table_dir, f"{experiment}.pdf")
    new_commands = [
        "\\newcommand{\\leapacc}{LEAP$_\\mathrm{ACC}$}",
        "\\newcommand{\\leapplus}{LEAP$_\\mathrm{KDEy}$}",
        "\\newcommand{\\leapppskde}{S-LEAP$_\\mathrm{KDEy}$}",
        "\\newcommand{\\oleapkde}{O-LEAP$_\\mathrm{KDEy}$}",
    ]
    column_alignment = [5, 5, 1], "c"
    additional_headers = [("$\\varnothing$", 5), ("IMS", 5), ("TMS", 1)]
    Table.LatexPDF(
        pdf_path,
        tables=tbls,
        landscape=False,
        new_commands=new_commands,
        column_alignment=column_alignment,
        additional_headers=additional_headers,
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-e",
        "--experiment",
        action="store",
        help="The type of experiment for which to generate tables",
        choices=["transd"],
        default="transd",
    )
    args = parser.parse_args()

    tables(args.experiment)
