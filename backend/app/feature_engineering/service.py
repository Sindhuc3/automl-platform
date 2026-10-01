from __future__ import annotations
import hashlib, json, uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict
import numpy as np
import pandas as pd
from app.database.mongodb import datasets_collection, preprocessing_runs_collection, feature_engineering_runs_collection
from .analysis import feature_statistics, redundancy_pairs
from .filtering import safe_filter
from .engineering import engineer_features
from .selection import selector_capability_report, build_compact_spec
from .constants import RANDOM_SEED, MAX_GENERATED_FEATURES, CORRELATION_FLAG, CORRELATION_PRUNE


def _now(): return datetime.now(timezone.utc)
def _safe(x): return json.loads(json.dumps(x, default=str))
def _hash_file(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()

def _artifact(folder,name,value):
    p=folder/name; p.write_text(json.dumps(_safe(value),indent=2),encoding="utf-8"); return p

def _load_module3(dataset_id, preprocessing_run_id=None):
    q={"dataset_id":dataset_id}
    if preprocessing_run_id: q["run_id"]=preprocessing_run_id
    else:
        docs=list(preprocessing_runs_collection.find(q,{"_id":0}).sort("created_at",-1).limit(1))
        if not docs: raise ValueError("M4_PREPROCESSING_NOT_FOUND: Run Module 3 preprocessing first.")
        return docs[0]
    doc=preprocessing_runs_collection.find_one(q,{"_id":0})
    if not doc: raise ValueError("M4_PREPROCESSING_NOT_FOUND: Module 3 run not found.")
    return doc

def _load_train_frame(dataset, pre):
    path=Path(dataset["validated_storage_path"])
    if not path.exists(): raise ValueError("M4_DATASET_MISSING: Validated dataset file not found.")
    df=pd.read_csv(path)
    split_path=Path((pre.get("artifact_paths") or {}).get("split_row_ids", ""))
    if not split_path.exists(): raise ValueError("M4_SPLIT_MISSING: Module 3 split artifact is missing.")
    split=json.loads(split_path.read_text(encoding="utf-8"))
    train_ids=split.get("train",[])
    train=df.loc[df.index.isin(train_ids)].copy()
    if train.empty: raise ValueError("M4_EMPTY_TRAIN: Module 3 training partition is empty.")
    target=pre["inputs"]["target_column"]
    if target not in train.columns: raise ValueError("M4_TARGET_MISSING: Target is missing from training data.")
    # Reconstruct Module 3's safe feature input, but retain datetime/high-cardinality source columns as engineering parents.
    report=(pre.get("report") or {})
    decisions=report.get("column_decisions") or []
    excluded={d.get("column") for d in decisions if d.get("action")=="exclude" and d.get("category")!="target"}
    safe=[d.get("column") for d in decisions if d.get("action")=="use" and d.get("column") in train.columns]
    source_cols=[]
    for c in train.columns:
        if c==target: continue
        d=next((x for x in decisions if x.get("column")==c),None)
        # Datetime and high-cardinality columns may be useful as parents for safe derived features.
        if c in safe or (d and d.get("reason_code") in {"EXCL_HIGH_CARDINALITY","EXCL_DATETIME"}):
            source_cols.append(c)
    # Module 3's missing-target/duplicate cleaning is reflected in split IDs.
    return train[source_cols + [target]].copy(), target, decisions

def _lineage_sources(source_cols, generated):
    registry=[]
    for i,c in enumerate(source_cols,1):
        registry.append({"feature_id":f"src_{i:04d}","name":c,"origin":"source","parents":[],"transform":"identity","status":"available"})
    registry.extend(generated)
    return registry

def run_feature_engineering(dataset_id: str, mode="automatic", version=1, preprocessing_run_id=None, overrides=None):
    overrides=overrides or {}
    if mode not in {"automatic","guided"}: raise ValueError("M4_INVALID_MODE: mode must be automatic or guided.")
    dataset=datasets_collection.find_one({"dataset_id":dataset_id},{"_id":0})
    if not dataset: raise ValueError("M4_DATASET_NOT_FOUND: Dataset not found.")
    pre=_load_module3(dataset_id,preprocessing_run_id)
    if pre.get("status")!="completed": raise ValueError("M4_PREPROCESSING_NOT_COMPLETE: Module 3 has not completed successfully.")
    path=Path(dataset["validated_storage_path"])
    if _hash_file(path)!=(pre.get("inputs") or {}).get("validated_dataset_fingerprint"):
        raise ValueError("M4_DATASET_CHANGED: Validated dataset fingerprint no longer matches Module 3.")
    train,target,decisions=_load_train_frame(dataset,pre)
    problem=(pre.get("inputs") or {}).get("problem_type","").lower()
    y=train[target]
    X=train.drop(columns=[target])
    # Basic statistics are target-blind; relevance is a diagnostic on TRAIN only.
    stats=feature_statistics(X)
    # No target-dependent feature decision is made in Module 4-Forward.
    # Supervised relevance/selection is executed inside Module 5/6 CV.
    relevance=[]
    corr_pairs=redundancy_pairs(X,float(overrides.get("correlation_flag_threshold",CORRELATION_FLAG)))
    safe_df, filter_log, redundant=safe_filter(X,None,float(overrides.get("correlation_prune_threshold",CORRELATION_PRUNE)))
    engineered_df, generated=engineer_features(safe_df,stats,int(overrides.get("max_generated_features",MAX_GENERATED_FEATURES)))
    generated_names=[g["name"] for g in generated]
    generated_sanity=[]
    for g in generated:
        s=engineered_df[g["name"]]
        if not np.isfinite(pd.to_numeric(s,errors="coerce").fillna(0)).all() or s.nunique(dropna=True)<=1:
            generated_sanity.append({"feature":g["name"],"decision":"EXCLUDE","rule_id":"GENERATED_SANITY","explanation":"Generated feature contained invalid or constant values and was removed."})
            engineered_df=engineered_df.drop(columns=[g["name"]])
    generated=[g for g in generated if g["name"] in engineered_df.columns]
    baseline_cols=list(safe_df.columns)
    engineered_cols=list(engineered_df.columns)
    # Compact is a recipe, not a prematurely materialized feature list.
    # Redundancy and supervised selection must be resolved inside CV folds.
    compact_cols=list(engineered_df.columns)
    compact_spec=build_compact_spec(compact_cols,len(train))
    warnings=[]
    if len(train)<200: warnings.append({"code":"M4_SMALL_DATASET","message":"Supervised feature selection can be unstable on small training sets; candidate generation continues without relying on model-dependent selection."})
    if len(engineered_cols)>len(baseline_cols): warnings.append({"code":"M4_ENGINEERED_FEATURES_ADDED","message":f"Created {len(generated)} dataset-driven companion features; Module 5 will evaluate whether they improve model performance."})
    if len(X.columns)>len(train): warnings.append({"code":"M4_WIDE_FEATURE_SPACE","message":"The feature space is wider than the available training rows; Compact selection is available as a cross-validation recipe."})
    run_id=f"FE-{_now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8].upper()}"
    folder=Path(dataset["validated_storage_path"]).parent/"feature_engineering"/run_id
    folder.mkdir(parents=True,exist_ok=True)
    feature_sets=[
        {"set_id":"A","name":"Baseline","description":"Module 3 usable features after target-blind safety filtering.","is_baseline":True,"features":baseline_cols,"selection":None},
        {"set_id":"B","name":"Engineered","description":"Baseline plus safe dataset-driven engineered features that survive sanity checks.","is_baseline":False,"features":engineered_cols,"selection":None},
    ]
    if compact_spec["enabled"]:
        feature_sets.append({"set_id":"C","name":"Compact","description":"Engineered candidate plus redundancy pruning and supervised MI-null selection executed inside Module 5 CV.","is_baseline":False,"features":compact_cols,"selection":compact_spec})
    else:
        warnings.append({"code":"M4_COMPACT_SKIPPED","message":compact_spec["reason"]})
    lineage=_lineage_sources(list(X.columns),generated)
    report={
        "summary":{"training_rows":len(train),"input_features":len(X.columns),"safe_features":len(baseline_cols),"removed_features":max(0, len(X.columns)-len(baseline_cols)),"engineered_features_added":len(generated),"engineered_feature_count":len(engineered_cols),"candidate_sets":len(feature_sets),"redundant_pairs":len(corr_pairs)},
        "analysis":{"feature_statistics":stats,"redundancy_pairs":corr_pairs,"target_relevance":[],
                    "target_dependent_analysis": "Deferred to Module 5/6 cross-validation."},
        "filtering":{"decisions":filter_log,"generated_sanity":generated_sanity},
        "engineering":{"created":generated,"max_generated_features":int(overrides.get("max_generated_features",MAX_GENERATED_FEATURES))},
        "feature_sets":feature_sets,
        "selector_capabilities":selector_capability_report(),
        "decision_log":filter_log+generated+generated_sanity,
        "lineage":lineage,
        "warnings":warnings,
        "reproducibility":{"run_id":run_id,"module3_run_id":pre.get("run_id"),"dataset_fingerprint":pre.get("inputs",{}).get("validated_dataset_fingerprint"),"random_seed":RANDOM_SEED,"ruleset_version":"m4-1.0","correlation_flag_threshold":float(overrides.get("correlation_flag_threshold",CORRELATION_FLAG)),"correlation_prune_threshold":float(overrides.get("correlation_prune_threshold",CORRELATION_PRUNE))},
        "graphs":{
            "funnel":[
                {"label":"Original", "value":len(X.columns)},
                {"label":"After safe filtering", "value":len(baseline_cols)},
                {"label":"After engineering", "value":len(engineered_cols)}
            ],
            "candidate_composition":[{"name":s["name"],"count":len(s["features"])} for s in feature_sets],
            "relevance":sorted(relevance,key=lambda x:x.get("mutual_information",0),reverse=True)[:12],
            "redundancy":corr_pairs[:20]
        },
        "final_selection": None,
        "boundary":{"test_partition_used":False,"supervised_selection_execution":"Module 5/6 cross-validation","final_feature_set_decided_by":"Arbiter using Module 5/6 evidence; Module 4-Finalize materializes the winning recipe"},
    }
    _artifact(folder,"feature_engineering_report.json",report)
    _artifact(folder,"feature_registry.json",lineage)
    _artifact(folder,"candidate_feature_sets.json",feature_sets)
    # Store a human-readable CSV for the training semantic frame used by Module 4 diagnostics.
    engineered_df.to_csv(folder/"engineered_training_features.csv",index=False)
    result={"run_id":run_id,"dataset_id":dataset_id,"version":version,"mode":mode,"status":"completed","inputs":{"target_column":target,"problem_type":problem,"module3_run_id":pre.get("run_id")},"report":report,"artifact_paths":{"report":str(folder/"feature_engineering_report.json"),"feature_registry":str(folder/"feature_registry.json"),"candidate_feature_sets":str(folder/"candidate_feature_sets.json"),"engineered_training_features":str(folder/"engineered_training_features.csv")},"created_at":_now()}
    feature_engineering_runs_collection.insert_one(_safe(result))
    return _safe(result)

def get_feature_engineering(dataset_id,run_id=None):
    q={"dataset_id":dataset_id}
    if run_id:
        q["run_id"]=run_id
        return feature_engineering_runs_collection.find_one(q,{"_id":0})
    return list(feature_engineering_runs_collection.find(q,{"_id":0}).sort("created_at",-1).limit(20))
