#!/usr/bin/env python3
"""Reproduce fixed CHIIR URL-feature audit experiments from the extracted feature cache.

Run from package root:
  python scripts/run_experiments.py --features data/deduplicated_features_for_audit.csv.gz --out results

The included source-verification report confirms that label 1 in the supplied artifact
corresponds to canonical phishing. Metrics therefore treat supplied label 1 as positive.
"""
from __future__ import annotations
import argparse, json, platform, sys
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn, shap, xgboost, idna
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, precision_recall_curve, auc, confusion_matrix)
from url_features import PublicSuffixList

SEED = 42
FEATURES = ["URLLength_recomputed", "HostnameLength", "HostnameLabelCount",
            "NoOfLettersInURL_recomputed", "NoOfDigitsInURL_recomputed",
            "NoOfOtherSpecialCharsInURL_recomputed"]
SIGMAS = [0.0, 0.1, 0.3, 0.5]

def model_factories():
    return {
        "Logistic Regression": lambda: LogisticRegression(max_iter=1000, C=0.01, random_state=SEED),
        "Decision Tree": lambda: DecisionTreeClassifier(max_depth=3, random_state=SEED),
        "Random Forest": lambda: RandomForestClassifier(n_estimators=20, max_depth=5, random_state=SEED, n_jobs=1),
        "XGBoost": lambda: XGBClassifier(n_estimators=20, max_depth=2, learning_rate=0.03,
            subsample=0.6, colsample_bytree=0.6, reg_alpha=2, reg_lambda=2,
            random_state=SEED, n_jobs=1, eval_metric="logloss", tree_method="hist")
    }

def make_splits(frame, psl):
    idx=np.arange(len(frame)); y=frame["label"].astype(int).to_numpy(); out={}
    trainval, test = train_test_split(idx, test_size=.20, random_state=42, stratify=y)
    core, val = train_test_split(trainval, test_size=.25, random_state=43, stratify=y[trainval])
    out["random"]=(core,val,test,{"method":"stratified train_test_split twice","test_fraction":0.2,"seed_test":42,"validation_fraction_of_trainval":0.25,"seed_validation":43})
    host_values=frame["ascii_hostname"].fillna("").astype(str).tolist()
    for mode in ["exact-hostname","registered-domain"]:
        groups=[]
        for i,h in enumerate(host_values):
            if not h: groups.append(f"__NOHOST_ROW_{i}")
            elif mode=="exact-hostname": groups.append(h)
            else: groups.append(psl.registrable_domain(h) or h)
        groups=np.asarray(groups,dtype=object)
        first=GroupShuffleSplit(n_splits=1,test_size=.20,random_state=42)
        trainval_idx,test_idx=next(first.split(idx,y,groups))
        trainval, test=idx[trainval_idx],idx[test_idx]
        second=GroupShuffleSplit(n_splits=1,test_size=.25,random_state=43)
        core_rel,val_rel=next(second.split(trainval,y[trainval],groups[trainval]))
        core,val=trainval[core_rel],trainval[val_rel]
        out[mode]=(core,val,test,{"method":"GroupShuffleSplit by exact ASCII hostname" if mode=="exact-hostname" else "GroupShuffleSplit by libpsl registrable domain","test_fraction_of_groups":0.20,"seed_test":42,"validation_fraction_of_remaining_groups":0.25,"seed_validation":43,"libpsl_version":psl.version,"group_overlap_core_validation":len(set(groups[core])&set(groups[val])),"group_overlap_core_test":len(set(groups[core])&set(groups[test])),"group_overlap_validation_test":len(set(groups[val])&set(groups[test]))})
    return out

def binary_metrics(y, prob):
    pred=(prob>=0.5).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {"n":int(len(y)),"TN":int(tn),"FP":int(fp),"FN":int(fn),"TP":int(tp),
      "accuracy":float(accuracy_score(y,pred)),"precision":float(precision_score(y,pred,zero_division=0)),
      "recall":float(recall_score(y,pred,zero_division=0)),"F1":float(f1_score(y,pred,zero_division=0)),
      "FPR":float(fp/(fp+tn)) if fp+tn else float("nan"),"FNR":float(fn/(fn+tp)) if fn+tp else float("nan"),
      "ROC_AUC":float(roc_auc_score(y,prob)),"PR_AUC_trapezoid":float(auc(*precision_recall_curve(y,prob)[1::-1])),
      "AP":float(average_precision_score(y,prob))}

def bootstrap_ci(y, prob, B=1000, seed=9001):
    rng=np.random.default_rng(seed); n=len(y); pred=(prob>=.5).astype(int)
    accs=[]; f1s=[]
    # Vectorized in bounded batches to keep memory reasonable for 50k-row test sets.
    for start in range(0,B,25):
        b=min(25,B-start); ii=rng.integers(0,n,size=(b,n)); yy=y[ii]; pp=pred[ii]
        accs.extend(np.mean(yy==pp,axis=1).tolist())
        tp=np.sum((yy==1)&(pp==1),axis=1); fp=np.sum((yy==0)&(pp==1),axis=1); fn=np.sum((yy==1)&(pp==0),axis=1)
        den=2*tp+fp+fn; f1s.extend(np.divide(2*tp,den,out=np.zeros_like(den,dtype=float),where=den>0).tolist())
    return {"accuracy_ci95_low":float(np.quantile(accs,.025)),"accuracy_ci95_high":float(np.quantile(accs,.975)),
      "F1_ci95_low":float(np.quantile(f1s,.025)),"F1_ci95_high":float(np.quantile(f1s,.975)),"bootstrap_resamples":B,"bootstrap_seed":seed}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--features',required=True,help='Deduplicated cache from scripts/extract_features.py')
    ap.add_argument('--out',required=True,help='Directory for metrics, manifests, predictions, and SHAP outputs')
    ap.add_argument('--bootstrap-resamples',type=int,default=1000)
    args=ap.parse_args(); out=Path(args.out); out.mkdir(parents=True,exist_ok=True); pred_dir=out/'predictions'; pred_dir.mkdir(exist_ok=True)
    frame=pd.read_csv(args.features, usecols=lambda c:c in set(['URL','label','ascii_hostname']+FEATURES))
    if not set(['URL','label','ascii_hostname']+FEATURES).issubset(frame.columns): raise ValueError('Feature cache lacks required columns')
    frame['label']=pd.to_numeric(frame['label'],errors='raise').astype(int)
    if set(frame['label'].unique()) != {0,1}: raise ValueError('Expected source labels {0,1}')
    y=frame['label'].to_numpy(); X=frame[FEATURES].to_numpy(float); psl=PublicSuffixList(); splits=make_splits(frame,psl); factories=model_factories()
    metrics_rows=[]; split_manifest={}
    # First: only the random validation split is used for the Gaussian noise selection sweep.
    core,val,test,_=splits['random']; core=np.asarray(core); val=np.asarray(val); scaler=StandardScaler().fit(X[core]); Xc=scaler.transform(X[core]); Xv=scaler.transform(X[val])
    noise_rows=[]
    for name,factory in factories.items():
        for sigma in SIGMAS:
            if sigma==0: noisy=Xc
            else:
                rng=np.random.default_rng(SEED + int(sigma*1000) + sum(ord(c) for c in 'random') + sum(ord(c) for c in name))
                noisy=Xc+rng.normal(0,sigma,size=Xc.shape)
            model=factory(); model.fit(noisy,y[core]); prob=model.predict_proba(Xv)[:,list(model.classes_).index(1)]
            noise_rows.append({'split':'random','model':name,'sigma':sigma,'validation_AP':float(average_precision_score(y[val],prob))})
    noise_df=pd.DataFrame(noise_rows); noise_df.to_csv(out/'random_noise_validation.csv',index=False)
    # Four configurations choose sigma=0; grouped splits reuse that setting and do not retune.
    selected=noise_df.sort_values(['validation_AP','sigma'],ascending=[False,True]).groupby('model',sort=False).first()['sigma'].to_dict()
    # In this experimental design the random model chooses each model's sigma independently.
    for split_name,(core,val,test,desc) in splits.items():
        core=np.asarray(core); val=np.asarray(val); test=np.asarray(test); train=np.concatenate([core,val])
        stats={}
        for part,ix in [('core',core),('validation',val),('train_plus_validation',train),('test',test)]:
            vals,cts=np.unique(y[ix],return_counts=True)
            stats[part]={'n':int(len(ix)),'class_counts_csv_label':{str(int(v)):int(c) for v,c in zip(vals,cts)}}
        if split_name != 'random':
            group_ids=[]
            for i,h in enumerate(frame['ascii_hostname'].fillna('').astype(str)):
                group_ids.append(f'__NOHOST_ROW_{i}' if not h else (h if split_name=='exact-hostname' else (psl.registrable_domain(h) or h)))
            groups=np.asarray(group_ids,dtype=object)
            desc.update({'group_overlap_core_validation':len(set(groups[core])&set(groups[val])), 'group_overlap_core_test':len(set(groups[core])&set(groups[test])), 'group_overlap_validation_test':len(set(groups[val])&set(groups[test]))})
            desc['unique_groups_by_partition']={part:int(len(set(groups[ix]))) for part,ix in [('core',core),('validation',val),('test',test)]}
        split_manifest[split_name]={'algorithm':desc,'partitions':stats,'target':'Phishing-positive target: supplied label 1 = phishing (canonical label inversion verified)','models':{}}
        scaler=StandardScaler().fit(X[train]); Xtr=scaler.transform(X[train]); Xte=scaler.transform(X[test])
        for name,factory in factories.items():
            model=factory(); model.fit(Xtr,y[train]); classes=list(model.classes_); prob=model.predict_proba(Xte)[:,classes.index(1)]
            metrics=binary_metrics(y[test],prob); ci=bootstrap_ci(y[test],prob,B=args.bootstrap_resamples,seed=9001+sum(map(ord,split_name+name)))
            row={'split':split_name,'model':name,'selected_sigma':float(selected.get(name,0.0)),**metrics,**ci}; metrics_rows.append(row)
            split_manifest[split_name]['models'][name]={'configuration': {'Logistic Regression':'C=0.01; max_iter=1000; random_state=42','Decision Tree':'max_depth=3; random_state=42','Random Forest':'n_estimators=20; max_depth=5; random_state=42; n_jobs=1','XGBoost':'n_estimators=20; max_depth=2; learning_rate=0.03; subsample=0.6; colsample_bytree=0.6; reg_alpha=2; reg_lambda=2; random_state=42; n_jobs=1; eval_metric=logloss; tree_method=hist'}[name], 'selected_sigma':float(selected.get(name,0.0)),'scaler':'StandardScaler fitted on core + validation','threshold':0.5}
            pd.DataFrame({'row_id':test,'URL':frame.iloc[test]['URL'].astype(str).to_numpy(),'y_true_csv_label':y[test],'y_pred_at_0_5':(prob>=.5).astype(int),'probability_csv_label_1':prob}).to_csv(pred_dir/f'{split_name}_{name.replace(" ","_")}_test_predictions.csv',index=False)
            print(f'{split_name:20s} {name:21s} n={len(test)} acc={metrics["accuracy"]:.4f} F1={metrics["F1"]:.4f} AP={metrics["AP"]:.4f}',flush=True)
            if split_name=='random' and name=='Random Forest':
                # Explain 250 fixed random test examples in probability space.
                import shap
                rng_bg=np.random.default_rng(4242); bg_idx=rng_bg.choice(len(train),size=min(100,len(train)),replace=False)
                rng_test=np.random.default_rng(4243); te_idx=rng_test.choice(len(test),size=min(250,len(test)),replace=False)
                explainer=shap.TreeExplainer(model, data=Xtr[bg_idx], feature_perturbation='interventional', model_output='probability')
                sv=explainer.shap_values(Xte[te_idx], check_additivity=True)
                if isinstance(sv,list): sv=sv[1]
                sv=np.asarray(sv)
                # SHAP versions differ in binary output dimensions; choose positive-class slice if present.
                if sv.ndim==3: sv=sv[:,:,1]
                exp=explainer.expected_value
                if np.ndim(exp)>0: exp=np.asarray(exp).reshape(-1)[-1]
                recon=np.asarray(exp)+sv.sum(axis=1); model_prob=model.predict_proba(Xte[te_idx])[:,classes.index(1)]
                shap_rows=pd.DataFrame(sv,columns=FEATURES); shap_rows.insert(0,'test_row_id',test[te_idx])
                shap_rows.to_csv(out/'random_RF_shap_values.csv',index=False)
                pd.DataFrame({'feature':FEATURES,'mean_abs_SHAP_probability':np.mean(np.abs(sv),axis=0)}).sort_values('mean_abs_SHAP_probability',ascending=False).to_csv(out/'random_RF_shap_global.csv',index=False)
                pd.DataFrame({'test_row_id':test[te_idx],'true_csv_label':y[test[te_idx]],'model_probability_csv_label_1':model_prob,'shap_base_value':float(exp),'sum_base_plus_shap':recon,'absolute_additivity_error':np.abs(recon-model_prob)}).to_csv(out/'random_RF_shap_reconstruction.csv',index=False)
                (out/'shap_metadata.json').write_text(json.dumps({'shap_version':shap.__version__,'explainer':'shap.TreeExplainer','feature_perturbation':'interventional','model_output':'probability','background_n':int(len(bg_idx)),'background_seed':4242,'explained_test_n':int(len(te_idx)),'test_sample_seed':4243,'preprocessing':'StandardScaler fitted on core+validation; SHAP receives scaled features as model inputs','max_abs_additivity_error':float(np.max(np.abs(recon-model_prob))),'mean_abs_additivity_error':float(np.mean(np.abs(recon-model_prob)))},indent=2))
    pd.DataFrame(metrics_rows).to_csv(out/'all_model_metrics.csv',index=False)
    (out/'split_manifest.json').write_text(json.dumps(split_manifest,indent=2))
    (out/'software_versions.json').write_text(json.dumps({'python':platform.python_version(),'pandas':pd.__version__,'numpy':np.__version__,'scikit_learn':sklearn.__version__,'xgboost':xgboost.__version__,'shap':shap.__version__,'idna':idna.__version__,'libpsl':psl.version},indent=2))
    print('Saved metrics, split manifest, predictions, validation noise sweep, and SHAP checks to',out,flush=True)
if __name__=='__main__': main()
