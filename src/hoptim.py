import itertools as IT
import os
from dataclasses import dataclass
from traceback import print_exception
from typing import Callable, Iterable

import cap
import env
import numpy as np
import pandas as pd
import quapy as qp
from cap.models.cont_table import LEAP
from cap.utils.commons import get_shift, parallel, true_acc
from env import PROJECT
from quapy.data import LabelledCollection
from quapy.protocol import UPP
from sklearn.base import BaseEstimator
from sklearn.metrics import confusion_matrix

from config import (
    ClsVariant,
    gen_acc_measure,
    gen_CAP_methods,
    gen_classifiers,
    gen_datasets,
    get_acc_names,
    get_CAP_method_names,
)
from util import (
    all_exist_pre_check,
    fit_or_switch,
    get_ct_predictions,
    get_logger,
    get_plain_prev,
    is_excluded,
    local_path,
    split_validation,
    timestamp,
)

EXPERIMENT = "hoptim"
log = get_logger(id=f"{PROJECT}.{EXPERIMENT}")

qp.environ["SAMPLE_SIZE"] = 1000
MS_SIZE = 100


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


class MSProtocol:
    def __init__(self, test_prot: Callable[[], Iterable[LabelledCollection]], ms_indexes: list[np.ndarray]):
        self.test_prot = test_prot
        self.ms_indexes = ms_indexes

    def __call__(self) -> Iterable[LabelledCollection]:
        for Ui, _idx in zip(self.test_prot(), self.ms_indexes):
            yield Ui.sampling_from_index(_idx)

    def total(self):
        return len(self.ms_indexes)

    @classmethod
    def empty(self):
        return MSProtocol(lambda: [], [])


class HOptimDatasetBundle:
    def __init__(self, L_prevalence: np.ndarray, V: LabelledCollection, U: LabelledCollection):
        self.L_prevalence: np.ndarray = L_prevalence
        self.V: LabelledCollection = V
        self.U: LabelledCollection = U
        self.V1: LabelledCollection = None
        self.V2_prot: Callable[[], Iterable[LabelledCollection]] | None = None
        self.test_prot: Callable[[], Iterable[LabelledCollection]] = lambda: []
        self.V_posteriors: np.ndarray = None
        self.V1_posteriors: np.ndarray = None
        self.V2_prot_posteriors: list[np.ndarray] = None
        self.test_prot_posteriors: list[np.ndarray] = None
        self.test_prot_y_hat: list[np.ndarray] = None
        self.true_accs: dict[str, list[float]] = None
        self.ms_indexes: list[np.ndarray] = None
        self.ms_prot: MSProtocol = MSProtocol.empty()
        self.ms_prot_posteriors: list[np.ndarray] = None
        self.ms_prot_y_hat: list[np.ndarray] = None
        self.ms_true_accs: dict[str, list[float]] = None

    def create_bundle(self, h: BaseEstimator):
        # generate test protocol
        self.test_prot = UPP(
            self.U,
            repeats=env.NUM_TEST,
            return_type="labelled_collection",
            random_state=qp.environ["_R_SEED"],
        )

        # split validation set
        self.V1, self.V2_prot = split_validation(self.V)

        # precomumpute model posteriors for validation sets
        self.V_posteriors = h.predict_proba(self.V.X)
        self.V1_posteriors = h.predict_proba(self.V1.X)
        self.V2_prot_posteriors = []
        for sample in self.V2_prot():
            self.V2_prot_posteriors.append(h.predict_proba(sample.X))

        # precomumpute model posteriors for test samples
        self.test_prot_posteriors, self.test_prot_y_hat, self.test_prot_true_cts = [], [], []
        for sample in self.test_prot():
            P = h.predict_proba(sample.X)
            self.test_prot_posteriors.append(P)
            y_hat = np.argmax(P, axis=-1)
            self.test_prot_y_hat.append(y_hat)

        self.true_accs = {}
        for acc_name, acc_fn in gen_acc_measure():
            self.true_accs[acc_name] = [
                self._get_true_acc(acc_fn, Ui.y, y_hat, labels=Ui.classes_)
                for Ui, y_hat in zip(self.test_prot(), self.test_prot_y_hat)
            ]

        with qp.util.temp_seed(qp.environ["_R_SEED"]):
            self.ms_indexes = [Ui.uniform_sampling_index(MS_SIZE) for Ui in self.test_prot()]

        self.ms_prot = MSProtocol(self.test_prot, self.ms_indexes)
        self.ms_prot_posteriors = [P[_idx, :] for P, _idx in zip(self.test_prot_posteriors, self.ms_indexes)]
        self.ms_prot_y_hat = [y_hat[_idx] for y_hat, _idx in zip(self.test_prot_y_hat, self.ms_indexes)]
        self.ms_true_accs = {}
        for acc_name, acc_fn in gen_acc_measure():
            self.ms_true_accs[acc_name] = [
                self._get_true_acc(acc_fn, Ui.y, y_hat, labels=Ui.classes_)
                for Ui, y_hat in zip(self.ms_prot(), self.ms_prot_y_hat)
            ]
        # self.ms_true_accs = {
        #     n: np.array(tas)[_idx].tolist() for (n, tas), _idx in zip(self.true_accs.items(), self.ms_indexes)
        # }

        return self

    def _get_true_acc(self, acc_fn, y, y_hat, labels=None):
        conf_table = confusion_matrix(y, y_pred=y_hat, labels=labels)
        return acc_fn(conf_table)

    @classmethod
    def mock(cls):
        return HOptimDatasetBundle(None, None, None)


def exp_protocol(
    args: tuple[ClsVariant, str, HOptimDatasetBundle, str, BaseEstimator, LabelledCollection, np.ndarray],
) -> list[EXP]:
    clsf, dataset_name, D, method_name, method, val, val_posteriors = args
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

        test_shift = get_shift(np.array([Ui.prevalence() for Ui in D.test_prot()]), D.L_prevalence).tolist()

        if clsf.ms_ignore:
            df_len = len(D.true_accs[acc_name])
            t_train, t_test_ave = 0, 0
            method_df = gen_method_df(
                df_len,
                uids=np.arange(df_len).tolist(),
                shifts=test_shift,
                true_accs=D.true_accs[acc_name],
                ms_true_accs=D.ms_true_accs[acc_name],
                ms_estim_accs=[None] * df_len,
                ms_acc_err=[None] * df_len,
                classifier=clsf.name,
                classifier_class=clsf.class_name,
                default_c=[clsf.default] * df_len,
                ms_ignore=[True] * df_len,
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
                    clsf, dataset_name, acc_name, method_name, df=method_df, t_train=t_train, t_test_ave=t_test_ave
                )
            )
            continue

        try:
            method, _t_train = fit_or_switch(method, val, val_posteriors, acc_fn, t_train is not None)
            t_train = t_train if _t_train is None else _t_train

            ms_estim_accs, _, t_test_ave = get_ct_predictions(method, D.ms_prot, D.ms_prot_posteriors)
        except Exception as e:
            print_exception(e)
            results.append(EXP.ERROR(e, clsf, dataset_name, acc_name, method_name))
            continue

        ms_ae = cap.error.ae(np.array(D.ms_true_accs[acc_name]), np.array(ms_estim_accs)).tolist()

        df_len = len(ms_estim_accs)
        method_df = gen_method_df(
            df_len,
            uids=np.arange(df_len).tolist(),
            shifts=test_shift,
            true_accs=D.true_accs[acc_name],
            ms_true_accs=D.ms_true_accs[acc_name],
            ms_estim_accs=ms_estim_accs,
            ms_acc_err=ms_ae,
            classifier=clsf.name,
            classifier_class=clsf.class_name,
            default_c=[clsf.default] * df_len,
            ms_ignore=[False] * df_len,
            method=method_name,
            dataset=dataset_name,
            acc_name=acc_name,
            train_prev=[L_prev] * df_len,
            val_prev=[val_prev] * df_len,
            t_train=t_train,
            t_test_ave=t_test_ave,
        )

        results.append(
            EXP.SUCCESS(clsf, dataset_name, acc_name, method_name, df=method_df, t_train=t_train, t_test_ave=t_test_ave)
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
        D = HOptimDatasetBundle(L.prevalence(), V, U).create_bundle(clsf.h)
        # compute true accs for h on dataset
        true_accs = {}
        for acc_name, acc_fn in gen_acc_measure():
            true_accs[acc_name] = [true_acc(clsf.h, acc_fn, Ui) for Ui in D.test_prot()]
        # store h-dataset combination
        return (clsf, dataset_name, D)


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
    for clsf, dataset_name, D in cls_dataset_gen:
        if D is None:
            log.info(f"All results for {clsf.name} over {dataset_name} exist, skipping")
        else:
            log.info(f"Trained {clsf.name} over {dataset_name}")
            cls_dataset.append((clsf, dataset_name, D))

    exp_prot_args_list = []
    for clsf, dataset_name, D in cls_dataset:
        for method_name, method, val, val_posteriors in gen_CAP_methods(clsf.h, D):
            exp_prot_args_list.append(
                (
                    clsf,
                    dataset_name,
                    D,
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
                path = local_path(r.dataset_name, r.clsf.file_name, r.method_name, r.acc_name, experiment=EXPERIMENT)
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
