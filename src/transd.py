import os
from dataclasses import dataclass
from traceback import print_exception
from typing import Iterable

import cap
import numpy as np
import pandas as pd
import quapy as qp
from cap.models.cont_table import LEAP
from cap.utils.commons import get_shift, parallel, true_acc
from quapy.data import LabelledCollection
from sklearn.base import BaseEstimator

from config import (
    ClsVariant,
    DatasetBundle,
    gen_acc_measure,
    gen_CAP_methods,
    gen_classifiers,
    gen_datasets,
    get_acc_names,
    get_CAP_method_names,
)
from env import PROJECT
from util import (
    all_exist_pre_check,
    fit_or_switch,
    get_ct_predictions,
    get_logger,
    get_plain_prev,
    is_excluded,
    local_path,
    timestamp,
)

EXPERIMENT = "transd"
log = get_logger(id=f"{PROJECT}.{EXPERIMENT}")

qp.environ["SAMPLE_SIZE"] = 100


class NoMSException(Exception):
    def __init__(self, message="Classifier does not support MS"):
        super().__init__(message)


def get_extra_from_method(df, method):
    if isinstance(method, LEAP):
        df["true_solve"] = method._true_solve_log[-1]


def gen_method_df(df_len, **data):
    data = data | {k: [v] * df_len for k, v in data.items() if not isinstance(v, list)}
    return pd.DataFrame.from_dict(data, orient="columns")


@dataclass
class EXP:
    code: int
    clsf: ClsVariant
    dataset_name: str
    acc_name: str
    method_name: str
    df: pd.DataFrame = None
    t_train: float = None
    t_test_ave: float = None
    err: Exception = None

    @classmethod
    def SUCCESS(cls, *args, **kwargs):
        return EXP(200, *args, **kwargs)

    @classmethod
    def EXISTS(cls, *args, **kwargs):
        return EXP(300, *args, **kwargs)

    @classmethod
    def ERROR(cls, e, *args, **kwargs):
        return EXP(400, *args, err=e, **kwargs)

    @property
    def ok(self):
        return self.code == 200

    @property
    def old(self):
        return self.code == 300

    def error(self):
        return self.code == 400


def exp_protocol(
    args: tuple[
        ClsVariant,
        str,
        DatasetBundle,
        np.ndarray,
        str,
        BaseEstimator,
        LabelledCollection,
        np.ndarray,
    ],
) -> list[EXP]:
    clsf, dataset_name, D, true_accs, method_name, method, val, val_posteriors = args
    results = []

    L_prev = get_plain_prev(D.L_prevalence)
    val_prev = get_plain_prev(val.prevalence())
    t_train = None
    for acc_name, acc_fn in gen_acc_measure():
        if is_excluded(clsf.name, dataset_name, method_name, acc_name):
            continue
        path = local_path(dataset_name, clsf.file_name, method_name, acc_name, experiment=EXPERIMENT)
        if os.path.exists(path):
            results.append(EXP.EXISTS(clsf, dataset_name, acc_name, method_name))
            continue

        df_len = D.test_prot.total()
        test_shift = get_shift(np.array([Ui.prevalence() for Ui in D.test_prot()]), D.L_prevalence).tolist()

        # df_len = len(true_accs[acc_name])
        # t_train, t_test_ave = 0, 0
        # method_df = gen_method_df(
        #     df_len,
        #     uids=np.arange(df_len).tolist(),
        #     shifts=test_shift,
        #     true_accs=true_accs[acc_name],
        #     estim_accs=[None] * df_len,
        #     acc_err=[None] * df_len,
        #     classifier=clsf.name,
        #     classifier_class=clsf.class_name,
        #     default_c=[clsf.default] * df_len,
        #     ms_ignore=[True] * df_len,
        #     method=method_name,
        #     dataset=dataset_name,
        #     acc_name=acc_name,
        #     train_prev=[L_prev] * df_len,
        #     val_prev=[val_prev] * df_len,
        #     t_train=t_train,
        #     t_test_ave=t_test_ave,
        # )
        #
        # results.append(
        #     EXP.SUCCESS(
        #         clsf, dataset_name, acc_name, method_name, df=method_df, t_train=t_train, t_test_ave=t_test_ave
        #     )
        # )
        # continue

        try:
            if clsf.ms_ignore:
                raise NoMSException()

            method, _t_train = fit_or_switch(method, val, val_posteriors, acc_fn, t_train is not None)
            t_train = t_train if _t_train is None else _t_train

            estim_accs, _, t_test_ave = get_ct_predictions(method, D.test_prot, D.test_prot_posteriors)
            ae = cap.error.ae(np.array(true_accs[acc_name]), np.array(estim_accs)).tolist()
        except NoMSException:
            estim_accs = [None] * df_len
            ae = [None] * df_len
        except Exception as e:
            print_exception(e)
            results.append(EXP.ERROR(e, clsf, dataset_name, acc_name, method_name))
            continue

        # df_len = len(estim_accs)
        method_df = gen_method_df(
            df_len,
            uids=np.arange(df_len).tolist(),
            shifts=test_shift,
            true_accs=true_accs[acc_name],
            estim_accs=estim_accs,
            acc_err=ae,
            classifier=clsf.name,
            classifier_class=clsf.class_name,
            default_c=[clsf.default] * df_len,
            ms_ignore=[clsf.ms_ignore] * df_len,
            method=method_name,
            dataset=dataset_name,
            acc_name=acc_name,
            train_prev=[L_prev] * df_len,
            val_prev=[val_prev] * df_len,
            t_train=t_train,
            t_test_ave=t_test_ave,
        )

        results.append(
            EXP.SUCCESS(
                clsf,
                dataset_name,
                acc_name,
                method_name,
                df=method_df,
                t_train=t_train,
                t_test_ave=t_test_ave,
            )
        )

    return results


def train_cls(args):
    orig_clsf, (dataset_name, (L, V, U)) = args
    #
    # check if all results for current combination already exist
    # if so, skip the combination
    if all_exist_pre_check(
        dataset_name=dataset_name,
        cls_name=orig_clsf.file_name,
        method_names=get_CAP_method_names(),
        acc_names=get_acc_names(),
        experiment=EXPERIMENT,
    ):
        return (orig_clsf, dataset_name, None, None)
    else:
        # clone model from the original one
        clsf = orig_clsf.clone()
        # fit model
        clsf.h.fit(*L.Xy)
        # create dataset bundle
        D = DatasetBundle(L.prevalence(), V, U).create_bundle(clsf.h)
        # compute true accs for h on dataset
        true_accs = {}
        for acc_name, acc_fn in gen_acc_measure():
            true_accs[acc_name] = [true_acc(clsf.h, acc_fn, Ui) for Ui in D.test_prot()]
        # store h-dataset combination
        return (clsf, dataset_name, D, true_accs)


def experiments():
    # cls_train_args = list(gen_model_dataset(gen_classifiers, gen_datasets))
    cls_train_args = []
    for dataset in gen_datasets():
        _, (L, _, _) = dataset
        for model in gen_classifiers(L.n_classes):
            cls_train_args.append((model, dataset))
    cls_dataset_gen = parallel(
        func=train_cls,
        args_list=cls_train_args,
        n_jobs=cap.env["N_JOBS"],
        return_as="generator_unordered",
    )
    cls_dataset = []
    for clsf, dataset_name, D, true_accs in cls_dataset_gen:
        if D is None:
            log.info(f"All results for {clsf.name} over {dataset_name} exist, skipping")
        else:
            log.info(f"Trained {clsf.name} over {dataset_name}")
            cls_dataset.append((clsf, dataset_name, D, true_accs))

    exp_prot_args_list = []
    for clsf, dataset_name, D, true_accs in cls_dataset:
        for method_name, method, val, val_posteriors in gen_CAP_methods(clsf.h, D):
            exp_prot_args_list.append(
                (
                    clsf,
                    dataset_name,
                    D,
                    true_accs,
                    method_name,
                    method,
                    val,
                    val_posteriors,
                )
            )

    results_gen: Iterable[list[EXP]] = parallel(
        func=exp_protocol,
        args_list=exp_prot_args_list,
        n_jobs=cap.env["N_JOBS"],
        return_as="generator_unordered",
        max_nbytes=None,
    )

    for res in results_gen:
        for r in res:
            if r.ok:
                path = local_path(
                    r.dataset_name,
                    r.clsf.file_name,
                    r.method_name,
                    r.acc_name,
                    experiment=EXPERIMENT,
                )
                r.df.to_json(path)
                log.info(
                    f"[{r.clsf.name}@{r.dataset_name}] {r.method_name} on {r.acc_name} done [{timestamp(r.t_train, r.t_test_ave)}]"
                )
            elif r.old:
                log.info(f"[{r.clsf.name}@{r.dataset_name}] {r.method_name} on {r.acc_name} exists, skipping")
            elif r.error:
                log.warning(
                    f"[{r.clsf.name}@{r.dataset_name}] {r.method_name}: {r.acc_name} gave error '{r.err}' - skipping"
                )


if __name__ == "__main__":
    try:
        log.info("-" * 31 + "  start  " + "-" * 31)
        experiments()
        log.info("-" * 32 + "  end  " + "-" * 32)
    except Exception as e:
        log.error(e)
        print_exception(e)
