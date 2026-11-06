"""Recalculate the report analyses from frozen hourly predictions and metadata."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .recovery_metrics import complete, hit_metrics, stats
from .peer_selection import build as select_peers
from .service_metrics import process, full

EPS = 1e-12


def read(path):
    """Read a numeric CSV without rounding its stored floating-point values."""
    return pd.read_csv(path, float_precision="round_trip", low_memory=False)


def write(folder, name, frame):
    """Write one analysis table, retaining unavailable observations as empty cells."""
    folder.mkdir(parents=True, exist_ok=True)
    frame.to_csv(folder/name, index=False,
                 compression={"method":"gzip", "mtime":0} if name.endswith(".gz") else None)


def activity_gaps(hourly, center="b", usable="raw_usable"):
    """Calculate 1-Y/P and its normal-reference-centered counterpart."""
    h = hourly.copy()
    ok = h[usable].eq(True) if usable in h else h.raw_status.eq("VALID")
    ok &= np.isfinite(h.Y) & np.isfinite(h.P) & h.P.ge(10)
    h["gap"] = np.where(ok, 1-h.Y/h.P, np.nan)
    h["gap_centered"] = h.gap-h[center]
    if "centered_usable" in h:
        h.loc[~h.centered_usable.eq(True), "gap_centered"] = np.nan
    elif "centered_status" in h:
        h.loc[~h.centered_status.eq("VALID"), "gap_centered"] = np.nan
    return h


def recovery(data, out):
    """Measure each registered event on its own rain-ending clock."""
    registry = read(data/"event_registry.csv")
    h = activity_gaps(read(data/"hourly_predictions.csv.gz"))
    catalogue = read(data/"weather_groups.csv")
    ids = set(registry.loc[registry.primary_model_weather_eligible, "canonical_group"])
    split = catalogue[catalogue.canonical_group.isin(ids)].sort_values(["start_utc","canonical_group"]).copy()
    split["time_rank"] = np.arange(len(split))+1
    split["period"] = np.where(split.time_rank <= 2*len(split)//3,"exploration","later")
    periods = split.set_index("canonical_group").period
    rows = []
    for eid, w in h.groupby("event_id", sort=False):
        w = w.sort_values("offset_from_end")
        ix = w.set_index("offset_from_end")
        post = ix.reindex(range(6))
        rain = post.rain_mean_mm.to_numpy(float)
        pre = w.loc[w.segment.eq("pre"),"rain_mean_mm"].to_numpy(float)
        known = np.isfinite(rain); wet = known & (rain > .100001)
        r = dict(event_id=eid,
                 pre6_rain_mm=pre.sum() if complete(pre,6) else np.nan,
                 post13_rain_mm=rain[:3].sum() if complete(rain[:3],3) else np.nan,
                 post46_rain_mm=rain[3:].sum() if complete(rain[3:],3) else np.nan,
                 post6_peak_mm=rain.max() if complete(rain,6) else np.nan,
                 post6_weather_complete=bool(known.all()),
                 rain_class="RERAIN" if wet.any() else "NO_RERAIN" if known.all() else "UNKNOWN",
                 rerain_first_hour=float(np.flatnonzero(wet)[0]+1) if wet.any() else np.nan,
                 rerain_first_exact=bool(known[:np.flatnonzero(wet)[0]].all()) if wet.any() else False)
        for mode, field in [("raw","gap"),("centered","gap_centered")]:
            d = ix.reindex([-2,-1])[field].to_numpy(float)
            d0 = float(d.mean()) if complete(d,2) else np.nan
            g = post[field].to_numpy(float)
            r[f"{mode}_D0"] = d0
            for H in [3,6]:
                v=g[:H]; late=g[1:3] if H==3 else g[3:6]
                r[f"{mode}_A{H}"] = float(v.sum()) if complete(v,H) else np.nan
                r[f"{mode}_late{H}"] = float(late.mean()) if complete(late,len(late)) else np.nan
                r[f"{mode}_Qlate{H}"] = d0-r[f"{mode}_late{H}"]
                r[f"{mode}_H80_{H}"] = float((v>.2*d0).sum()) if d0>0 and complete(v,H) else np.nan
                for label,factor in [("T50",.5),("T80",.2)]:
                    r.update({f"{mode}_{label}_{H}_{k}":val for k,val in hit_metrics(v,d0,rain[:H],factor,H).items()})
            for k in range(6):
                r[f"{mode}_g_h{k+1}"]=g[k]
        rows.append(r)
    events = registry.merge(pd.DataFrame(rows),on="event_id",validate="one_to_one")
    events["period"] = events.canonical_group.map(periods).fillna("DESCRIPTIVE_ONLY")
    events["inference_scope"] = np.where(events.primary_model_weather_eligible,"MAIN_VERIFIED_CLOSED","DESCRIPTIVE_ONLY")
    write(out,"hourly_curves.csv.gz",h)
    write(out,"recovery_event_metrics.csv",events)
    write(out,"temporal_split_groups.csv",split)
    return events,h,split


def recurrence(data, out, events):
    """Apply the original deterministic matching and equal-weight recurrence rule."""
    candidates,support,_ = select_peers(events)
    peers = candidates[candidates.selected].copy()
    write(out,"peer_selection.csv",peers)
    write(out,"event_peer_support.csv",support)
    ev = events.set_index("event_id")
    selected = peers[peers.inference_scope.eq("MAIN_VERIFIED_CLOSED") &
                     peers.comparison_pool.eq("SAME_KNOWN_RAIN_CLASS")].copy()
    local_tables=[];group_tables=[]
    keys=["target_zone","canonical_group","period","inference_scope","comparison_pool"]
    for mode in ["centered","raw"]:
        for H in [6,3]:
            p=selected.copy()
            a=ev.loc[p.target_event].reset_index(drop=True)
            b=ev.loc[p.peer_event].reset_index(drop=True)
            p=p.reset_index(drop=True)
            p["S80"]=a[f"{mode}_H80_{H}"]-b[f"{mode}_H80_{H}"]
            applicable=a[f"{mode}_D0"].gt(0)&b[f"{mode}_D0"].gt(0)
            p["Sprogress"]=(b[f"{mode}_Qlate{H}"]-a[f"{mode}_Qlate{H}"]).where(applicable)
            local=p.groupby(["target_event"]+keys,dropna=False).agg(
                S80=("S80","mean"),Sprogress=("Sprogress","mean"),selected_peers=("peer_event","size"),
                S80_contributing_peers=("S80","count"),Sprogress_contributing_peers=("Sprogress","count")).reset_index()
            group=local.groupby(keys,dropna=False).agg(
                S80=("S80","mean"),Sprogress=("Sprogress","mean"),selected_local_events=("target_event","size"),
                S80_contributing_events=("S80","count"),Sprogress_contributing_events=("Sprogress","count"),
                selected_peers=("selected_peers","sum"),S80_contributing_peers=("S80_contributing_peers","sum"),
                Sprogress_contributing_peers=("Sprogress_contributing_peers","sum")).reset_index()
            for frame in [local,group]:
                frame["mode"]=mode;frame["H"]=H;frame["subset"]="all_selected"
            local_tables.append(local);group_tables.append(group)
    groups=pd.concat(group_tables,ignore_index=True)
    write(out,"event_peer_recovery.csv",pd.concat(local_tables,ignore_index=True))
    write(out,"zone_weather_group_recovery.csv",groups)
    cohort=read(data/"recovery_cohort.csv")
    labels=read(data/"zone_labels.csv")
    if "zone_name" not in cohort or "borough" not in cohort:
        cohort=cohort[["zone_id"]].merge(labels[["zone_id","zone_name","borough"]],on="zone_id",validate="one_to_one")
    primary=groups[groups["mode"].eq("centered")&groups.H.eq(6)]
    periods=[]
    for z in cohort.itertuples(index=False):
        for metric in ["S80","Sprogress"]:
            for period in ["exploration","later","all"]:
                q=primary[primary.target_zone.eq(z.zone_id)]
                if period!="all":q=q[q.period.eq(period)]
                x=q[metric].dropna().to_numpy();n=len(x);positive=int((x>EPS).sum())
                periods.append(dict(zone_id=z.zone_id,zone_name=z.zone_name,borough=z.borough,metric=metric,
                    period=period,selected_groups=len(q),valid_groups=n,mean=x.mean() if n else np.nan,
                    positive_groups=positive,negative_groups=int((x<-EPS).sum()),tied_groups=int((abs(x)<=EPS).sum()),
                    positive_fraction=positive/n if n else np.nan,
                    rule=bool(n>=3 and positive/n>=2/3-EPS and x.mean()>0)))
    summaries=pd.DataFrame(periods)
    records=[]
    for z in cohort.itertuples(index=False):
        for metric in ["S80","Sprogress"]:
            q=summaries[summaries.zone_id.eq(z.zone_id)&summaries.metric.eq(metric)].set_index("period")
            a,b=q.loc["exploration"],q.loc["later"];n=int(b.valid_groups)
            direction="UNKNOWN" if n==0 else "POSITIVE" if b["mean"]>EPS else "NEGATIVE" if b["mean"]<-EPS else "TIED"
            state=("NOT_IN_FROZEN_LIST" if not a.rule else "NO_LATER_COMPARISON" if b.selected_groups==0 else
                "LATER_METRIC_UNAVAILABLE" if n==0 else "ONE_TWO_LATER_GROUPS" if n<3 else
                "REPEATS_ORIGINAL_POINT_RULE" if b.rule else "POSITIVE_NOT_RULE" if direction=="POSITIVE" else "NEGATIVE_OR_TIED")
            records.append(dict(zone_id=z.zone_id,zone_name=z.zone_name,borough=z.borough,metric=metric,
                exploration_member=bool(a.rule),exploration_groups=int(a.valid_groups),later_selected_groups=int(b.selected_groups),
                later_valid_groups=n,exploration_mean=a["mean"],later_mean=b["mean"],
                later_mean_display=b["mean"]*(100 if metric=="Sprogress" else 1),later_direction=direction,
                later_support_bin=str(n) if n<3 else ">=3",replication_status=state,later_rule=bool(b.rule) if a.rule else None,
                later_positive_groups=int(b.positive_groups)))
    status=pd.DataFrame(records)
    write(out,"recurrence_period_summary.csv",summaries)
    write(out,"replication_status_long.csv",status)
    return peers,groups,status


def service_comparison(data,out):
    """Recompute separate complete H80 and Qlate samples from fixed service cases."""
    cases=read(data/"service_cases.csv")
    h=activity_gaps(read(data/"service_hourly_predictions.csv.gz"),center="normal_error_center",usable="raw_status_unused")
    rows=[]
    for ev in cases.itertuples(index=False):
        v=h[h.event_id.eq(ev.event_id)]
        for mode,col in [("raw","gap"),("centered","gap_centered")]:
            r=dict(event_id=ev.event_id,zone_id=ev.zone_id,zone_name=ev.zone_name,canonical_group=ev.canonical_group,
                   h1_utc=ev.h1_utc,h1_NYC=ev.h1_NYC,mode=mode)
            measured={}
            for service in ["Taxi","RideHailing"]:
                sv=v[v.service.eq(service)].set_index("offset_from_end")
                d0=full(sv.reindex([-2,-1])[col]);g=sv.reindex(range(24))[col].to_numpy(float)
                for H in [6,24]:
                    met=process(g,d0,H);measured[(service,H)]=met
                    r.update({f"{service}_H{H}_{k}":value for k,value in met.items()})
            td,rd=measured[("Taxi",6)]["D0"],measured[("RideHailing",6)]["D0"]
            eligible=td>0 and rd>0 and abs(td-rd)<=.1+EPS
            r["own_start_eligible"]=bool(eligible)
            for H in [6,24]:
                tm,rm=measured[("Taxi",H)],measured[("RideHailing",H)]
                r[f"H{H}_S80"]=tm["H80"]-rm["H80"] if eligible else np.nan
                r[f"H{H}_Sprogress_pp"]=100*(rm["Qlate"]-tm["Qlate"]) if eligible else np.nan
            rows.append(r)
    measured=pd.DataFrame(rows);groups=[];summary=[]
    for mode,v in measured.groupby("mode"):
        for metric,field in [("S80","H80"),("Sprogress_pp","Qlate")]:
            for sample,H,q in [("ORIGINAL20_H6",6,v),("FIXED20_H24_AVAILABLE",24,v),
                    ("H24_METRIC_COMMON_H6",6,v[v[f"H24_{metric}"].notna()]),
                    ("H24_METRIC_COMMON_H24",24,v[v[f"H24_{metric}"].notna()])]:
                key=f"H{H}_{metric}";q=q[q[key].notna()].copy();scale=100 if metric=="Sprogress_pp" else 1
                q["Taxi_value"]=q[f"Taxi_H{H}_{field}"]*scale
                q["RideHailing_value"]=q[f"RideHailing_H{H}_{field}"]*scale
                vals=[]
                for cg in cases.canonical_group.unique():
                    g=q[q.canonical_group.eq(cg)];z=g.groupby("zone_id")[[key,"Taxi_value","RideHailing_value"]].mean()
                    difference=z[key].mean()
                    groups.append(dict(mode=mode,metric=metric,sample=sample,horizon=H,canonical_group=cg,
                        events=len(g),zones=len(z),difference=difference,Taxi_value=z.Taxi_value.mean(),RideHailing_value=z.RideHailing_value.mean(),
                        event_ids=";".join(g.event_id),fixed_cohort_events=int(cases.canonical_group.eq(cg).sum()),
                        unit="hours" if metric=="S80" else "percentage_points"))
                    if len(g):vals.append(difference)
                a=np.asarray(vals)
                summary.append(dict(mode=mode,metric=metric,sample=sample,horizon=H,events=len(q),zones=q.zone_id.nunique(),
                    groups=q.canonical_group.nunique(),equal_group_mean=full(a),positive_groups=int((a>EPS).sum()),
                    negative_groups=int((a<-EPS).sum()),tied_groups=int((abs(a)<=EPS).sum())))
    write(out,"service_event_process.csv",measured)
    write(out,"service_weather_group_summary.csv",pd.DataFrame(groups))
    write(out,"service_comparison_summary.csv",pd.DataFrame(summary))
    return measured,pd.DataFrame(summary)


def normal_excess(data,out):
    """Recalculate paired storm/normal metrics with fixed leave-date centers and peers."""
    hourly=read(data/"normal_hourly_predictions.csv.gz")
    valid=np.isfinite(hourly.Y)&np.isfinite(hourly.P)&hourly.P.ge(10)&hourly.source_available
    hourly["gap_centered_lodo"]=np.where(valid,1-hourly.Y/hourly.P,np.nan)-hourly.b_minus_d
    meta=read(data/"normal_window_metadata.csv.gz")
    rows=[]
    for wid,w in hourly.groupby("normal_window_id",sort=False):
        r=dict(normal_window_id=wid)
        for trajectory in ["storm","normal"]:
            g=w[w.trajectory.eq(trajectory)].set_index("offset_from_end").reindex(range(-2,6)).gap_centered_lodo.to_numpy()
            r.update({trajectory+"_"+k:v for k,v in stats(g).items()})
        matched=r["storm_D0"]>0 and r["normal_D0"]>0 and abs(r["storm_D0"]-r["normal_D0"])<=.1+EPS
        r["start_matched"]=bool(matched)
        for met,base,direction in [("E80","H80",1),("EQ","Qlate",-1)]:
            r[met]=direction*(r["storm_"+base]-r["normal_"+base]) if matched else np.nan
        rows.append(r)
    windows=meta.merge(pd.DataFrame(rows),on="normal_window_id",how="left",validate="one_to_one")
    for m in ["E80","EQ"]:
        windows.loc[~windows.calendar_weather_source_eligible,m]=np.nan
    event=windows.groupby("event_id")[["E80","EQ"]].mean()
    p=read(data/"normal_fixed_peers.csv");membership=read(data/"normal_target_membership.csv").set_index("zone_id")
    pairs=[]
    for metric,field,scale in [("S80","E80",1),("Sprogress","EQ",100)]:
        q=p.copy();q["metric"]=metric
        q["value"]=(q.target_event.map(event[field])-q.peer_event.map(event[field]))*scale
        pairs.append(q)
    pairs=pd.concat(pairs,ignore_index=True)
    local=pairs.groupby(["target_event","target_zone","canonical_group","period","metric"],dropna=False).value.mean().reset_index()
    groups=local.groupby(["target_zone","canonical_group","period","metric"],dropna=False).value.mean().reset_index().rename(columns={"target_zone":"zone_id"})
    summaries=[]
    for (z,period,metric),g in groups.groupby(["zone_id","period","metric"]):
        x=g.value.dropna();n=len(x)
        summaries.append(dict(zone_id=z,zone_name=membership.loc[z,"zone_name"],period=period,metric=metric,stage="D",
            valid_groups=n,positive_groups=int((x>EPS).sum()),negative_groups=int((x<-EPS).sum()),
            tied_groups=int((abs(x)<=EPS).sum()),mean=x.mean(),positive_fraction=float((x>EPS).mean()) if n else np.nan,
            frozen_metric_member=bool(membership.loc[z,metric+"_point_clue"]),
            descriptive_original_direction_pattern=bool(n>=3 and (x>EPS).mean()>=2/3 and x.mean()>0)))
    write(out,"normal_window_metrics.csv.gz",windows)
    write(out,"normal_excess_event_metrics.csv",event.reset_index())
    write(out,"normal_excess_group_scores.csv",groups)
    write(out,"five_zone_period_summary.csv",pd.DataFrame(summaries))
    return groups,pd.DataFrame(summaries)


def local_reference_comparison(data,out):
    """Apply the existing local reference correction to its fixed pair roster only."""
    h=read(data/"local_reference_centers.csv");p=read(data/"local_reference_fixed_peers.csv")
    metrics=[]
    for eid,w in h.groupby("event_id",sort=False):
        w=w.set_index("offset_from_end").reindex(range(-2,6))
        for label,b in [("original","old_b"),("corrected","corrected_b")]:
            met=stats(w.raw_gap.to_numpy()-w[b].to_numpy())
            metrics.append(dict(event_id=eid,version=label,**met))
    e=pd.DataFrame(metrics);rows=[]
    for version in ["original","corrected"]:
        s=e[e.version.eq(version)].set_index("event_id")
        for metric,field,direction,scale in [("S80","H80",1,1),("Sprogress","Qlate",-1,100)]:
            q=p[p.target_event.isin(s.index)&p.peer_event.isin(s.index)].copy()
            q["value"]=direction*(q.target_event.map(s[field])-q.peer_event.map(s[field]))*scale
            q["version"]=version;q["metric"]=metric;rows.append(q)
    pairs=pd.concat(rows,ignore_index=True)
    groups=pairs.groupby(["target_zone","canonical_group","metric","version"]).value.mean().reset_index()
    write(out,"local_reference_event_metrics.csv",e)
    write(out,"local_reference_group_scores.csv",groups)


def run(data_dir, output_dir):
    """Run every final analysis from the supplied frozen input directory."""
    data=Path(data_dir);out=Path(output_dir)/"results"
    events,h,split=recovery(data,out)
    peers,groups,status=recurrence(data,out,events)
    service,ss=service_comparison(data,out)
    ng,np_=normal_excess(data,out)
    local_reference_comparison(data,out)
    core=events[events.core_weather_verified]
    rebound=core[core.raw_D0.gt(0)&core.raw_T50_6_hour.notna()]
    reversals=int(rebound.raw_T50_6_relapse.eq(True).sum())
    summary={"candidate_events":len(events),"verified_liquid_events":len(core),
        "observed_half_gap_rebounds":len(rebound),"subsequent_reversals":reversals,
        "reversal_fraction":reversals/len(rebound),"weather_groups":len(split),
        "exploration_groups":int(split.period.eq("exploration").sum()),"later_groups":int(split.period.eq("later").sum()),
        "recurring_zones":{m:status.loc[status.metric.eq(m)&status.replication_status.eq("REPEATS_ORIGINAL_POINT_RULE"),"zone_name"].tolist() for m in ["S80","Sprogress"]}}
    (out/"analysis_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    return summary
