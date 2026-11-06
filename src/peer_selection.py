"""Select up to five weather- and starting-gap-matched neighborhoods."""
import numpy as np
import pandas as pd
EPS=1e-12
RAIN_FIELDS=["peak_mm","total_mm","duration_h","pre6_rain_mm","post13_rain_mm","post46_rain_mm","post6_peak_mm"]
INPUTS=["event_id","zone_id","canonical_group","start_utc","end_utc","period","primary_model_weather_eligible","centered_D0","raw_D0","rain_class","rerain_first_hour","rerain_first_exact","post6_weather_complete"]+RAIN_FIELDS

def build(e):
    """Return candidate pairs, event support and exposure scales from event metrics.

    Selection is deterministic within each weather group and rain class,
    with at most five distinct peer zones per target local episode.
    """
    e = e[INPUTS].copy()
    for k in ["start_utc","end_utc"]:
        e[k] = pd.to_datetime(e[k],utc=True)
    scales = {}
    for k in RAIN_FIELDS:
        x = np.log1p(e[k].dropna().to_numpy(float))
        s = np.subtract(*np.quantile(x,[.75,.25]))
        if s<=EPS: s=x.std()
        scales[k] = float(s if s>EPS else 1.)
    allp = []
    support = []
    for cg, allg in e.groupby("canonical_group",sort=True):
        d = allg[allg.centered_D0.gt(0)]
        for a in allg.itertuples(index=False):
            stat = dict(event_id=a.event_id,canonical_group=cg,zone_id=a.zone_id,
                        weather_and_centered_start_candidates=0,
                        same_known_class_candidates=0,cross_known_class_candidates=0,
                        unknown_class_candidates=0,selected_main=0)
            if not a.centered_D0>0:
                stat["peer_status"]="D0_NONPOSITIVE_OR_UNKNOWN"
                support.append(stat); continue
            q=d[d.zone_id.ne(a.zone_id)].copy()
            if a.primary_model_weather_eligible:
                q=q[q.primary_model_weather_eligible]
            overlap=(q.end_utc.clip(upper=a.end_utc)-q.start_utc.clip(lower=a.start_utc)).dt.total_seconds()/3600
            endgap=(q.end_utc-a.end_utc).abs().dt.total_seconds()/3600
            startgap=(q.start_utc-a.start_utc).abs().dt.total_seconds()/3600
            ratios={k:np.maximum(q[k],getattr(a,k))/np.minimum(q[k],getattr(a,k))
                    for k in ["total_mm","peak_mm","duration_h"]}
            mask=(overlap>0)&(endgap<=6)&((q.centered_D0-a.centered_D0).abs()<=.1+EPS)
            for v in ratios.values():mask &= v<=2+1e-8
            q=q[mask]
            stat["weather_and_centered_start_candidates"]=len(q)
            if not len(q):
                stat["peer_status"]="NO_WEATHER_START_COMPARABLE_PEER"
                support.append(stat);continue
            p=pd.DataFrame(dict(target_event=a.event_id,peer_event=q.event_id,
                target_zone=a.zone_id,peer_zone=q.zone_id,canonical_group=cg,
                period=a.period,
                inference_scope="MAIN_VERIFIED_CLOSED" if a.primary_model_weather_eligible else "DESCRIPTIVE_ONLY",
                overlap_h=overlap.loc[q.index],end_gap_h=endgap.loc[q.index],start_gap_h=startgap.loc[q.index],
                target_D0=a.centered_D0,peer_D0=q.centered_D0,
                centered_start_difference=a.centered_D0-q.centered_D0,
                target_raw_D0=a.raw_D0,peer_raw_D0=q.raw_D0,
                both_raw_positive=(a.raw_D0>0)&q.raw_D0.gt(0),
                raw_start_caliper=(a.raw_D0-q.raw_D0).abs()<=.1+EPS,
                target_rain_class=a.rain_class,peer_rain_class=q.rain_class,
                target_post_weather_complete=a.post6_weather_complete,
                peer_post_weather_complete=q.post6_weather_complete))
            unknown=(a.rain_class=="UNKNOWN")|q.rain_class.eq("UNKNOWN")
            p["comparison_pool"]=np.where(unknown,"UNKNOWN_RAIN_CLASS",
                 np.where(q.rain_class.eq(a.rain_class),"SAME_KNOWN_RAIN_CLASS","CROSS_RAIN_CLASS"))
            comps=[abs(a.centered_D0-q.centered_D0).to_numpy()/.1,
                   endgap.loc[q.index].to_numpy()/6,startgap.loc[q.index].to_numpy()/6]
            missing=np.zeros(len(q),int)
            for k in RAIN_FIELDS:
                p["target_"+k]=getattr(a,k)
                p["peer_"+k]=q[k]
                delta=abs(np.log1p(q[k].to_numpy())-np.log1p(getattr(a,k)))/scales[k]
                missing+=~np.isfinite(delta)
                comps.append(np.where(np.isfinite(delta),delta,1.))
                p[k+"_distance"]=delta
            at=7. if a.rain_class=="NO_RERAIN" else a.rerain_first_hour
            bt=np.where(q.rain_class.eq("NO_RERAIN"),7.,q.rerain_first_hour)
            td=abs(at-bt)/6
            missing+=~np.isfinite(td)
            comps.append(np.where(np.isfinite(td),td,1.))
            p["target_rerain_hour"]=a.rerain_first_hour
            p["peer_rerain_hour"]=q.rerain_first_hour
            p["rerain_time_distance"]=td
            p["exposure_distance_missing_dimensions"]=missing
            p["distance"]=np.mean(comps,axis=0)
            for k,v in ratios.items():p[k+"_ratio"]=v.loc[q.index]
            p["selected"]=False;p["selected_rank"]=np.nan
            for pool,pp in p.groupby("comparison_pool"):
                key={"SAME_KNOWN_RAIN_CLASS":"same_known_class_candidates",
                     "CROSS_RAIN_CLASS":"cross_known_class_candidates","UNKNOWN_RAIN_CLASS":"unknown_class_candidates"}[pool]
                stat[key]=len(pp)
                take=pp.sort_values(["distance","end_gap_h","start_gap_h","peer_event"]).drop_duplicates("peer_zone").head(5).index
                p.loc[take,"selected"]=True
                p.loc[take,"selected_rank"]=np.arange(len(take))+1
                if pool=="SAME_KNOWN_RAIN_CLASS":stat["selected_main"]=len(take)
            stat["peer_status"]="MAIN_PEERS_SELECTED" if stat["selected_main"] else "ONLY_CROSS_OR_UNKNOWN_RAIN_CLASS"
            support.append(stat);allp.append(p)
    return pd.concat(allp,ignore_index=True),pd.DataFrame(support),scales
