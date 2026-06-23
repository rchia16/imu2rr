import numpy as np
import os
import glob
import json
import pandas as pd
import ipdb

from typing import Dict

from sklearn.metrics import (
    r2_score, root_mean_squared_error, mean_absolute_error
)
from scipy.stats import pearsonr, f_oneway, mannwhitneyu

from config import N_MARKERS

class Evaluation():
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred
        self.condition = None

    def rmse(self):
        rmse = root_mean_squared_error(self.y_true, self.y_pred)
        return rmse

    def r2_score(self):
        return r2_score(self.y_true, self.y_pred)
    
    def mae(self):
        return mean_absolute_error(self.y_true, self.y_pred)

    def calculate_pearsonr(self):
        ''' looking for positive correlation p < 0.01 '''
        return pearsonr(self.y_true, self.y_pred)

    def err_std(self):
        return np.std(np.abs(self.y_true - self.y_pred))

    def get_stats(self, **kwargs):
        lbls_mu = np.mean(self.y_true, **kwargs)
        lbls_sd = np.std( self.y_true, **kwargs)

        pred_mu = np.mean(self.y_pred, **kwargs)
        pred_sd = np.std(self.y_pred, **kwargs)

        _, pval = f_oneway(self.y_pred, self.y_true)
        _, mpval = mannwhitneyu(self.y_pred, self.y_true)

        stats = {
            'lbls_mu' : lbls_mu,
            'lbls_sd' : lbls_sd,
            'pred_mu' : pred_mu,
            'pred_sd' : pred_sd,
            'anova'   : pval,
            'mannwu'  : mpval,
        }
        return stats

    def bland_altman(self):
        samples = np.array([self.y_true, self.y_pred]).T
        avgs = np.mean(samples, axis=-1)
        diffs = samples[:,0] - samples[:,1]
        return avgs, diffs

    def get_evals(self):
        coeff, pval = self.calculate_pearsonr()
        if coeff.ndim > 0:
            coeff = coeff.mean()
        if pval.ndim > 0:
            pval = pval.mean()

        my_evals = {
            'mae': self.mae(),
            'std': self.err_std(),
            'rmse': self.rmse(),
            'r2': self.r2_score(),
            'pearsonr_coeff': coeff,
            'pearsonr_pval': pval,
        }
        return my_evals

class MarkerEvaluation():
    def __init__(self, y_true, y_pred):
        if len(y_pred.shape) > len(y_true.shape):
            n_elem = y_pred.shape[-1]
            y_true = np.repeat(y_true.reshape(-1,1), n_elem, axis=1)
        self.y_true = y_true
        self.y_pred = y_pred
        self.n_elem = 1
        try:
            self.n_elem = y_pred.shape[1]
        except: pass

    def get_stats(self, ind=None):
        if ind is not None:
            evals = Evaluation(self.y_true[:, ind], self.y_pred[:, ind])
        else:
            evals = Evaluation(self.y_true, self.y_pred)
        stats = evals.get_stats(axis=0)
        return stats

    def get_evals(self):
        marker_evals_dict = []
        for n in range(self.n_elem):
            evals = Evaluation(self.y_true[:,n], self.y_pred[:,n])
            marker_evals_dict.append(evals.get_evals())
        return marker_evals_dict

    def bland_altman(self):
        avgs, diff = [], []
        for n in range(self.n_elem):
            evals = Evaluation(self.y_true[:,n], self.y_pred[:,n])
            avg_vec, dif_vec = evals.bland_altman()
            avgs.append(avg_vec)
            diff.append(dif_vec)
        return np.array(avgs), np.array(diff)

def simple_regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
    ev = Evaluation(y_true, y_pred)
    out = ev.get_evals()
    return {
        "mae": float(out["mae"]),
        "rmse": float(out["rmse"]),
        "pearsonr": float(out["pearsonr_coeff"]),
    }

if __name__ == '__main__':
    from config import USER, SEAT_DATA_DIR, BR_FS
    from os.path import join, sep
    from copy import copy
    from digitalsignalprocessing import movingaverage
    from digitalsignalprocessing import do_pad_fft, get_max_freq, hernandez_sp
    import pickle
    
    from scipy.stats import normaltest
    from statsmodels.formula.api import ols
    import statsmodels.api as sm

    eval_dir = f"/projects/BLVMob/imu-rr-seated/models"
    imu_issues = [17, 26, 30]
    subjects = [
        'S' + str(i).zfill(2)
        for i in range(12, 31)
        if i not in imu_issues
    ]
    strategy = 'vit'
    data_str = 'imu_filt'
    device = None
    debug = False
    freeze = True

    baseline = {'data_str': data_str,
                'use_classifier': False,
                'use_flow': False,
                'use_ttt': False,
                'use_ttt_flow': False,
                'use_tttflow_rr_prior': False,
                'use_mae_training': False,
                'use_temporal_encoder': False,
                'freeze': freeze,
                'use_style_views': False,
                'strategy': strategy,
                'use_autoencoder': False,
                'use_ssa': False,
                'use_cmt': False, }

    flow = {'data_str': data_str,
            'use_classifier': False,
            'use_flow': True,
            'use_ttt': False,
            'use_ttt_flow': True,
            'use_tttflow_rr_prior': False,
            'use_mae_training': False,
            'use_temporal_encoder': True,
            'freeze': freeze,
            'use_style_views': False,
            'strategy': strategy,
            'use_autoencoder': False,
            'use_ssa': False,
            'use_cmt': False,}

    ssa = {'data_str': data_str,
            'use_classifier': False,
            'use_flow': True,
            'use_ttt': False,
            'use_ttt_flow': True,
            'use_tttflow_rr_prior': False,
            'use_mae_training': False,
            'use_temporal_encoder': True,
            'freeze': freeze,
            'use_style_views': False,
            'strategy': strategy,
            'use_autoencoder': False,
            'use_ssa': True,
            'use_cmt': False,}

    cmt = {'data_str': data_str,
            'use_classifier': False,
            'use_flow': True,
            'use_ttt': False,
            'use_ttt_flow': True,
            'use_tttflow_rr_prior': False,
            'use_mae_training': False,
            'use_temporal_encoder': True,
            'freeze': freeze,
            'use_style_views': False,
            'strategy': strategy,
            'use_autoencoder': False,
            'use_ssa': False,
            'use_cmt': True,}

    ssa_cmt = {'data_str': data_str,
            'use_classifier': False,
            'use_flow': True,
            'use_ttt': False,
            'use_ttt_flow': True,
            'use_tttflow_rr_prior': False,
            'use_mae_training': False,
            'use_temporal_encoder': True,
            'freeze': freeze,
            'use_style_views': False,
            'strategy': strategy,
            'use_autoencoder': False,
            'use_ssa': True,
            'use_cmt': True,}
    
    m_dir = join(eval_dir, data_str)
    if m_dir == SEAT_DATA_DIR:
        model_parent_directory = sep.join(m_dir.split(sep)[:-1]+['loocv'])
    else:
        model_parent_directory = join(m_dir, 'loocv')

    def get_eval_fname(
        sbj,
        data_str='imu_filt',
        use_classifier=False,
        use_ttt=False,
        use_flow=False,
        use_ttt_flow=False,
        use_tttflow_rr_prior=False,
        use_mae_training=False,
        use_style_views=False,
        use_temporal_encoder=False,
        freeze=True,
        strategy='vit',
        use_autoencoder=False,
        use_ssa=False,
        use_cmt=False,
    ):
        prefix = f'{sbj}_{data_str}_cls_{int(use_classifier)}_'\
                f'ttt_{int(use_ttt)}_'\
                f'flow_{int(use_ttt_flow)}_'\
                f'prior_{int(use_tttflow_rr_prior)}_'\
                f'mae_{int(use_mae_training)}_'\
                f'temporal_{int(use_temporal_encoder)}_'\
                f'freeze_{int(freeze)}_'\
                f'style_{int(use_style_views)}_'\
                f'autoencode_{int(use_autoencoder)}_'\
                f'ssa_{int(use_ssa)}_'\
                f'cmt_{int(use_cmt)}_'

        s_mdl_dir = join(model_parent_directory, sbj, strategy)
        eval_file = join(
            s_mdl_dir,
            prefix+'eval.pkl')

        pss_file = join(
            s_mdl_dir,
            prefix+'pss.pkl')

        downstream_eval_file = join(
            s_mdl_dir,
            prefix+'downstream_eval.pkl')
        return eval_file, pss_file, downstream_eval_file

    def load_pickle(m_file):
        with open(m_file, 'rb') as f:
            data = pickle.load(f)
        return data


    sbj_results = []

    methods = ['baseline', 'flow', 'ssa', 'cmt', 'ssa_cmt']
    eval_strs = ['mae', 'sd', 'cc', 'method']
    configs = [baseline, flow, ssa, cmt, ssa_cmt]

    method_map = {k: idx for idx, k in enumerate(methods)}

    result_str = 'br'

    for sbj in subjects:
        # sbj_results[sbj] = {e_str: [] for e_str in eval_strs}
        # sbj_results[sbj] = []
        for method, cfg in zip(methods, configs):
            tmp = {}
            eval_file, pss_file, br_file = get_eval_fname(sbj, **cfg)

            eval_results = load_pickle(eval_file)
            pss_results = load_pickle(pss_file)
            br_results = load_pickle(br_file)

            if result_str == 'pss':
                results = copy(eval_results)
            else:
                downstream_signals = pss_results['preds']
                chest_labels = pss_results['labels']
                labels = np.array(
                    [get_max_freq(win, fs=BR_FS) * 60.0 for win in chest_labels],
                    dtype=np.float32
                )
                downstream_dsp = np.array(
                    [get_max_freq(win, fs=BR_FS) * 60.0 for win in downstream_signals],
                    dtype=np.float32
                )
                dsp_evals = Evaluation(labels, downstream_dsp).get_evals()
                results = copy(br_results)

            tmp['mae'] = results[0]['mae']
            tmp['sd']  = results[0]['std']
            tmp['cc']  = results[0]['pearsonr_coeff']
            tmp['method']  = method
            tmp['subject'] = sbj
            sbj_results.append(tmp)

    result_df = pd.DataFrame(sbj_results)
    result_df['method'] = result_df['method'].map(lambda x: method_map[x])

    # model = 
    model = sm.OLS(result_df['mae'], sm.add_constant(result_df['method']))
    results = model.fit()
    print(results.summary())
    # results_lm = ols('mae ~ C(method)', data=result_df)
    # table = sm.stats.anova_lm(results_lm, typ=2)

    # result_df.to_csv("vit_experiment_results_br.csv")
    print(result_df)
    my_sub_result = {}
    for method, idx in method_map.items():
        tmp = result_df[result_df['method'] == idx]
        my_sub_result[method] = tmp[['mae','cc']].agg(['mean', 'std'])

    for k, v in my_sub_result.items():
        print(f"{k}: \n {v}")
