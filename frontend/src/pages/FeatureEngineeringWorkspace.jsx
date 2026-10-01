import {useEffect,useMemo,useState} from "react";
import {useNavigate,useParams} from "react-router-dom";
import {getLatestFeatureEngineering,runFeatureEngineering,featureDownloadUrl} from "../services/featureApi";
import "../styles/featureEngineering.css";

function Section({title,label,children,actions}){return <section className="m4-section"><div className="m4-heading"><div><span className="section-label">{label}</span><h2>{title}</h2></div>{actions}</div>{children}</section>}
function Download({href,children}){return <a className="m4-download" href={href} target="_blank" rel="noreferrer">{children}</a>}
function fmt(v){return typeof v==="number"?v.toFixed(v<1?3:2):String(v??"—")}

function FunnelGraph({funnel=[]}){
 const max=Math.max(...funnel.map(x=>x.value),1);
 return <div className="m4-chart-card"><div className="m4-chart-title">Feature funnel</div><div className="m4-funnel">{funnel.map((x,i)=><div className="m4-funnel-row" key={x.label}><div className="m4-funnel-label"><span>{x.label}</span><b>{x.value}</b></div><div className="m4-bar-track"><div className="m4-bar" style={{width:`${Math.max(8,(x.value/max)*100)}%`}}/></div>{i<funnel.length-1&&<span className="m4-down">↓</span>}</div>)}</div></div>
}
function RelevanceGraph({items=[]}){
 const top=items.slice(0,8); const max=Math.max(...top.map(x=>x.mutual_information||0),0.000001);
 return <div className="m4-chart-card"><div className="m4-chart-title">Target relevance — training data</div><div className="m4-hbars">{top.map(x=><div className="m4-hbar-row" key={x.feature}><div><span>{x.feature}</span><b>{fmt(x.mutual_information)}</b></div><div className="m4-bar-track"><div className="m4-bar" style={{width:`${Math.max(2,(x.mutual_information/max)*100)}%`}}/></div></div>)}</div></div>
}
function CandidateGraph({sets=[]}){
 const max=Math.max(...sets.map(x=>x.count),1);
 return <div className="m4-chart-card"><div className="m4-chart-title">Candidate-set composition</div><div className="m4-candidate-bars">{sets.map(x=><div key={x.name} className="m4-candidate-bar"><div className="m4-candidate-value"><b>{x.count}</b><span>{x.name}</span></div><div className="m4-vertical-track"><div className="m4-vertical-fill" style={{height:`${Math.max(8,(x.count/max)*100)}%`}}/></div></div>)}</div></div>
}
function RedundancyGraph({pairs=[]}){
 const names=[...new Set(pairs.flatMap(x=>[x.left,x.right]))].slice(0,10);
 const lookup=new Map(pairs.map(x=>[`${x.left}|||${x.right}`,x.value]));
 const value=(a,b)=>lookup.get(`${a}|||${b}`)??lookup.get(`${b}|||${a}`)??null;
 return <div className="m4-chart-card"><div className="m4-chart-title">Redundancy map <span>Spearman / Cramér's V ≥ configured flag threshold</span></div>{!names.length?<div className="m4-empty-chart">No high-redundancy pairs crossed the reporting threshold.</div>:<div className="m4-heatmap"><div></div>{names.map(n=><span className="m4-axis" key={`x-${n}`}>{n}</span>)}{names.map(a=><div className="m4-heat-row" key={a}><span className="m4-axis y">{a}</span>{names.map(b=>{const v=a===b?1:value(a,b);return <span key={b} className="m4-cell" title={v===null?"No flagged relationship":`${a} ↔ ${b}: ${v}`} style={{opacity:v===null?.08:.15+Math.min(v,1)*.85}}>{a===b?"—":v===null?"":v.toFixed(2)}</span>})}</div>)}</div>}</div>
}

function DecisionDetail({item,onClose}){
 if(!item)return null;
 const ev=item.evidence||{}; const th=item.threshold||{}; const eff=item.effect||{}; const scope=item.fit_scope||{}; const sets=item.candidate_sets||{};
 return <div className="m4-detail-overlay" onClick={onClose}><div className="m4-detail" onClick={e=>e.stopPropagation()}><button className="m4-close" onClick={onClose}>×</button><span className="section-label">FEATURE DECISION</span><h2>{item.feature||item.name}</h2><div className="m4-detail-status"><span>Decision</span><strong>{item.decision||"—"}</strong></div><div className="m4-detail-grid">
 <div><label>Evidence</label><p>{Object.entries(ev).map(([k,v])=><span key={k}><b>{k.replaceAll("_"," ")}:</b> {typeof v==="object"?JSON.stringify(v):String(v)}</span>)}</p></div>
 <div><label>Why?</label><p>{item.explanation||"No explanation recorded."}</p></div>
 <div><label>What happens next?</label><p>{item.what_happens_next||"The recorded decision is passed to the candidate-set construction stage."}</p></div>
 <div><label>Threshold</label><p>{Object.keys(th).length?Object.entries(th).map(([k,v])=><span key={k}><b>{k.replaceAll("_"," ")}:</b> {String(v)}</span>):"No threshold — direct rule."}</p></div>
 <div><label>Threshold type</label><p>{th.kind||"Not applicable"}{th.user_adjustable===true?" · user adjustable in Guided Mode":""}</p></div>
 <div><label>Learned from</label><p>{scope.learned_from||"Training data only"}{scope.n_rows_used?` · ${scope.n_rows_used} rows`:""}</p></div>
 <div><label>Effect</label><p>{Object.keys(eff).length?Object.entries(eff).map(([k,v])=><span key={k}><b>{k.replaceAll("_"," ")}:</b> {typeof v==="object"?JSON.stringify(v):String(v)}</span>):"Recorded in candidate construction."}</p></div>
 <div><label>Candidate sets</label><p>{Object.keys(sets).length?Object.entries(sets).map(([k,v])=><span key={k}><b>{k}:</b> {String(v)}</span>):"Recorded in the feature-set registry."}</p></div>
 </div><div className="m4-detail-footer"><span>Mode: {item.mode||"AUTO"}</span><span>Origin: {item.origin||"source"}</span><span>Reversible: {item.reversible===false?"No":"Yes"}</span></div></div></div>
}

export default function FeatureEngineeringWorkspace(){
 const {datasetId}=useParams(); const navigate=useNavigate(); const [result,setResult]=useState(null); const [loading,setLoading]=useState(false); const [error,setError]=useState(""); const [selected,setSelected]=useState(null); const [selectedSet,setSelectedSet]=useState(null);
 useEffect(()=>{getLatestFeatureEngineering(datasetId).then(r=>{if(r.length)setResult(r[0])}).catch(()=>{})},[datasetId]);
 async function run(){setLoading(true);setError("");try{const pre=JSON.parse(sessionStorage.getItem("automlPreprocessing")||"null");const r=await runFeatureEngineering(datasetId,{mode:"automatic",preprocessingRunId:pre?.run_id});setResult(r);sessionStorage.setItem("automlFeatureEngineering",JSON.stringify(r));}catch(e){setError(e.message||"Feature engineering failed.")}finally{setLoading(false)}}
 const report=result?.report||{}; const summary=report.summary||{}; const sets=report.feature_sets||[]; const analysis=report.analysis||{}; const logs=report.decision_log||[]; const graphs=report.graphs||{}; const final=report.final_selection; const download=n=>featureDownloadUrl(datasetId,result.run_id,n);
 const funnel=graphs.funnel||[{label:"Original",value:summary.input_features||0},{label:"After safe filtering",value:summary.safe_features||0},{label:"After engineering",value:summary.engineered_feature_count||0}];
 const compactSet=sets.find(s=>s.name==="Compact");
 const displayLogs=useMemo(()=>logs.slice(0,100),[logs]);
 return <div className="m4-page">
 {!result&&!loading&&<section className="m4-start"><span className="section-label">MODULE 04 · FEATURE ENGINEERING & SELECTION</span><h2>Discover, engineer and select useful features</h2><p>Module 4 analyses the training partition, applies target-blind safety filters, creates common dataset-driven features, measures relevance and redundancy, and packages candidate feature sets for Module 5.</p><div className="m4-note"><strong>Automatic Mode</strong><span>No decisions are requested during the first run. The same feature engines will be available in Guided Mode with configurable controls.</span></div><button className="primary-button" onClick={run}>Run Automatic Feature Engineering</button></section>}
 {loading&&<div className="m4-loading"><span className="spinner"/><div><strong>Analysing training features...</strong><span>Filtering, engineering, measuring relevance, generating graphs and building candidate sets.</span></div></div>}
 {error&&<div className="message error-message">{error}</div>}
 {result&&<>
 <section className="m4-header"><div><span className="section-label">MODULE 04 · COMPLETE</span><h2>Feature Engineering & Selection</h2><p>Dataset: <b>{result.inputs?.dataset_name||datasetId}</b> · Target: <b>{result.inputs?.target_column||"—"}</b> · Problem: <b>{result.inputs?.problem_type||"—"}</b></p><p>Only the Module 3 training partition was used. The test partition remains sealed for final evaluation.</p></div><span className="status-tag valid">Version {result.version}</span></section>
 <section className="m4-metrics"><div><span>Original features</span><strong>{summary.input_features}</strong></div><div><span>Safe features</span><strong>{summary.safe_features}</strong></div><div><span>Removed features</span><strong>{summary.removed_features??Math.max(0,(summary.input_features||0)-(summary.safe_features||0))}</strong></div><div><span>Engineered features</span><strong>{summary.engineered_features_added}</strong></div><div><span>Candidate sets</span><strong>{summary.candidate_sets}</strong></div></section>
 <Section label="FEATURE FUNNEL" title="How the feature space changed"><div className="m4-funnel-head"><span>Original → safe filtering → engineering</span><b>{summary.engineered_feature_count||0} features after engineering</b></div><FunnelGraph funnel={funnel}/></Section>
 <Section label="GRAPHS" title="Feature analysis visuals"><div className="m4-chart-grid"><RelevanceGraph items={graphs.relevance||analysis.target_relevance||[]}/><CandidateGraph sets={graphs.candidate_composition||sets.map(s=>({name:s.name,count:s.features.length}))}/><RedundancyGraph pairs={graphs.redundancy||analysis.redundancy_pairs||[]}/><div className="m4-chart-card"><div className="m4-chart-title">Feature-selection capability map</div><div className="m4-capability-graph">{(report.selector_capabilities||[]).map(x=><div key={x.id}><span>{x.name}</span><b className={x.execution.includes("module5")?"later":"now"}>{x.execution.includes("module5")?"Module 5 CV":"Module 4"}</b></div>)}</div></div></div></Section>
 <Section label="CANDIDATE FEATURE SETS" title="What Module 5 will evaluate"><div className="m4-set-grid">{sets.map(s=><button className="m4-set" key={s.set_id} onClick={()=>setSelectedSet(s)}><div><span>{s.set_id}</span><strong>{s.name}</strong></div><p>{s.description}</p><b>{s.features.length} features</b><small>View →</small></button>)}</div></Section>
 <Section label="FEATURE DECISIONS" title="Evidence for every keep, exclude, flag and create decision"><div className="m4-decision-table">{displayLogs.map((x,i)=><button className="m4-log-card" key={`${x.feature}-${i}`} onClick={()=>setSelected(x)}><div className="m4-log-main"><strong>{x.feature||x.name}</strong><span>{x.rule_id||x.transform||"ENGINEERING"}</span></div><b className={`decision-${String(x.decision||"").toLowerCase()}`}>{x.decision}</b><p>{x.explanation||"Recorded evidence-based decision."}</p><span className="m4-view">View →</span></button>)}</div></Section>
 <Section label="FEATURE ANALYSIS" title="Training-data evidence"><div className="m4-analysis-table"><div className="m4-analysis-head"><b>Feature</b><b>MI</b><b>Univariate F</b><b>Missing</b><b>Unique</b><b>Skew</b></div>{(analysis.feature_statistics||[]).map(st=>{const rel=(analysis.target_relevance||[]).find(x=>x.feature===st.feature)||{};return <div className="m4-analysis-row" key={st.feature}><strong>{st.feature}</strong><span>{fmt(rel.mutual_information)}</span><span>{fmt(rel.univariate_f)}</span><span>{st.missing_percent}%</span><span>{st.unique_count}</span><span>{fmt(st.skewness)}</span></div>})}</div></Section>
 <Section label="FINAL FEATURE SET" title="The final features selected by Module 5"><div className={`m4-final ${final?"ready":"pending"}`}>{final?<><div className="m4-final-head"><div><span>Final set: <b>{final.set_name||final.set_id}</b></span><small>{final.selection_method||"Selected after model × feature-set cross-validation"}</small></div><strong>{(final.features||[]).length} features</strong></div><div className="m4-feature-chips">{(final.features||[]).map(f=><span key={f}>{f}</span>)}</div><p>{final.explanation||"Module 5 selected this feature set after cross-validation and final refit."}</p></>:<><strong>Pending Module 5</strong><span>Module 4 creates the candidate sets; Module 5 evaluates them against the model families and writes the final feature set here. This section will become populated when Module 5 completes.</span><small>Candidate sets currently available: {sets.map(s=>s.name).join(" · ")||"—"}</small></>}</div></Section>
 {(report.warnings||[]).length>0&&<Section label="WARNINGS" title="Recorded observations"><div className="m4-warnings">{report.warnings.map((w,i)=><div key={i}><b>!</b><span><strong>{w.code}</strong>{w.message}</span></div>)}</div></Section>}
 <Section label="ARTIFACTS" title="Module 4 outputs"><div className="m4-download-grid"><Download href={download("report")}>Feature engineering report</Download><Download href={download("feature_registry")}>Feature registry / lineage</Download><Download href={download("candidate_feature_sets")}>Candidate feature sets</Download><Download href={download("engineered_training_features")}>Engineered training features</Download></div></Section>
 <div className="m4-actions"><button className="secondary-button" onClick={()=>navigate(`/datasets/${datasetId}/preprocessing`)}>Back to Preprocessing</button><button className="primary-button" onClick={()=>navigate(`/datasets/${datasetId}/model-selection`)}>Continue to Model Selection</button></div>
 </>}
 {selected&&<DecisionDetail item={selected} onClose={()=>setSelected(null)}/>} 
 {selectedSet&&<div className="m4-detail-overlay" onClick={()=>setSelectedSet(null)}><div className="m4-detail" onClick={e=>e.stopPropagation()}><button className="m4-close" onClick={()=>setSelectedSet(null)}>×</button><span className="section-label">CANDIDATE SET</span><h2>{selectedSet.name}</h2><p className="m4-set-description">{selectedSet.description}</p><div className="m4-set-feature-list">{selectedSet.features.map(f=><span key={f}>{f}</span>)}</div>{selectedSet.selection&&<div className="m4-set-selection"><b>Selection recipe</b><p>{selectedSet.selection.method} · {selectedSet.selection.fit_scope}</p><p>{selectedSet.selection.reason}</p></div>}</div></div>}
 </div>
}
