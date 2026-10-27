#!/usr/bin/env python3
"""Deduplicate by exact URL, extract explicit pre-fetch features, and write audit outputs."""
from __future__ import annotations
import argparse, hashlib, json, platform
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn, shap, xgboost, idna
from url_features import extract_url_features

PAIRS=[('URLLength','URLLength_recomputed','URL length'),('DomainLength','HostnameLength','Hostname length'),('NoOfSubDomain','HostnameLabelCount','Hostname-label count'),('NoOfLettersInURL','NoOfLettersInURL_recomputed','Letters'),('NoOfDegitsInURL','NoOfDigitsInURL_recomputed','Digits'),('NoOfOtherSpecialCharsInURL','NoOfOtherSpecialCharsInURL_recomputed','Other characters')]

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--csv',required=True,help='Path to source/supplied PhiUSIIL CSV')
    p.add_argument('--out',required=True,help='Output directory')
    args=p.parse_args(); src=Path(args.csv); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(src)
    if not {'URL','label'}.issubset(df.columns): raise ValueError('Input must have URL and label columns')
    raw_n=len(df); raw_cols=len(df.columns)
    repeated=df[df.duplicated('URL',keep=False)].groupby('URL')['label'].nunique()
    conflict_groups=int((repeated>1).sum())
    df=df.drop_duplicates(subset=['URL'],keep='first').copy().reset_index(drop=True)
    extracted=pd.DataFrame([extract_url_features(u) for u in df['URL'].astype(str)])
    frame=pd.concat([df,extracted],axis=1)
    frame['label']=pd.to_numeric(frame['label'],errors='raise').astype(int)
    frame.to_csv(out/'deduplicated_features_for_audit.csv.gz',index=False,compression='gzip')
    audits=[]
    for stored,rebuilt,name in PAIRS:
        a=pd.to_numeric(frame[stored],errors='coerce').to_numpy() if stored in frame else np.full(len(frame),np.nan)
        b=pd.to_numeric(frame[rebuilt],errors='coerce').to_numpy(); valid=np.isfinite(a)&np.isfinite(b)
        audits.append({'stored_column':stored,'recomputed_feature':rebuilt,'feature':name,'valid_n':int(valid.sum()),
          'exact_match_rate':float(np.mean(a[valid]==b[valid])),'mean_absolute_difference':float(np.mean(np.abs(a[valid]-b[valid]))),
          'pearson_correlation':float(np.corrcoef(a[valid],b[valid])[0,1])})
    pd.DataFrame(audits).to_csv(out/'feature_agreement.csv',index=False)
    examples=[]
    for i in [0,1,2]:
        r=frame.iloc[i]; raw=str(r['URL']); sl=raw[12:-1]
        examples.append({'URL':raw,'stored_URLLength':int(r['URLLength']),'recomputed_URLLength':int(r['URLLength_recomputed']),
          'stored_DomainLength':int(r['DomainLength']),'recomputed_HostnameLength':int(r['HostnameLength']),
          'stored_NoOfSubDomain':int(r['NoOfSubDomain']),'recomputed_HostnameLabelCount':int(r['HostnameLabelCount']),
          'stored_letters':int(r['NoOfLettersInURL']),'recomputed_letters':int(r['NoOfLettersInURL_recomputed']),
          'stored_digits':int(r['NoOfDegitsInURL']),'recomputed_digits':int(r['NoOfDigitsInURL_recomputed']),
          'stored_other':int(r['NoOfOtherSpecialCharsInURL']),'recomputed_other':int(r['NoOfOtherSpecialCharsInURL_recomputed']),
          'URL_slice_12_to_minus1':sl,'slice_letters':sum(ch.isascii() and ch.isalpha() for ch in sl),
          'slice_digits':sum(ch.isascii() and ch.isdigit() for ch in sl),
          'slice_other':sum(not(ch.isascii() and (ch.isalpha() or ch.isdigit())) for ch in sl)})
    pd.DataFrame(examples).to_csv(out/'feature_examples.csv',index=False)
    urls=frame['URL'].astype(str); mask=urls.str.startswith('https://www.'); sub=urls[mask]; cohort=frame.loc[mask]
    signature={'https_www_subset_n':int(mask.sum()),
      'stored_URLLength_equals_raw_length_minus1_rate':float((pd.to_numeric(cohort['URLLength']).to_numpy()==sub.str.len().to_numpy()-1).mean())}
    for stored,fun in [('NoOfLettersInURL',lambda s:sum(c.isascii() and c.isalpha() for c in s)),
      ('NoOfDegitsInURL',lambda s:sum(c.isascii() and c.isdigit() for c in s)),
      ('NoOfOtherSpecialCharsInURL',lambda s:sum(not(c.isascii() and (c.isalpha() or c.isdigit())) for c in s))]:
        calculated=sub.str.slice(12,-1).map(fun).to_numpy()
        signature[stored+'_matches_URL_slice_12_minus1_rate']=float((pd.to_numeric(cohort[stored]).to_numpy()==calculated).mean())
    (out/'feature_signature_summary.json').write_text(json.dumps({'representative_examples':examples,'https_www_signature':signature},indent=2,ensure_ascii=False))
    edge_urls=['https://例え.テスト/パス?q=1','https://example.com/%E2%9C%93?q=%252F#frag','http://192.0.2.10:8080/login','https://Example.COM.:443/a','http://[::1']
    pd.DataFrame([{'input':u,**extract_url_features(u)} for u in edge_urls]).to_csv(out/'url_normalisation_examples.csv',index=False)
    report={'input_file':src.name,'input_sha256':sha256(src),'input_rows':raw_n,'input_columns':raw_cols,
      'unique_URL_rows':len(frame),'duplicate_rows_removed':raw_n-len(frame),'duplicate_URL_groups_with_conflicting_labels':conflict_groups,
      'class_counts_after_dedup_csv_label':{str(int(k)):int(v) for k,v in frame['label'].value_counts().sort_index().items()},
      'feature_definition':'NFC+trim; parse host before percent-encoding; UTS46 IDNA; URL lexical counts from percent-encoded full URL; no page fetch',
      'python':platform.python_version(),'pandas':pd.__version__,'numpy':np.__version__,'scikit_learn':sklearn.__version__,'xgboost':xgboost.__version__,'shap':shap.__version__,'idna':idna.__version__}
    (out/'environment_and_source_audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
if __name__=='__main__': main()
