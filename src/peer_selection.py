"""Select up to five weather- and starting-gap-matched neighborhoods."""
import numpy as np  # Calculate exposure distances, scaled differences and deterministic peer ranks.
import pandas as pd  # Group event records, compare timestamps and assemble the pair and support tables.
EPS=1e-12  # Allow a small floating-point tolerance in the exposure scales and starting-gap criterion.
# Name the seven rainfall and duration measures used to score exposure similarity.
RAIN_FIELDS=["peak_mm","total_mm","duration_h","pre6_rain_mm","post13_rain_mm","post46_rain_mm","post6_peak_mm"]
# Specify the identifiers, time windows, eligibility fields and exposures required for matching.
INPUTS=["event_id","zone_id","canonical_group","start_utc","end_utc","period","primary_model_weather_eligible","centered_D0","raw_D0","rain_class","rerain_first_hour","rerain_first_exact","post6_weather_complete"]+RAIN_FIELDS

def build(e):  # Match each local episode to distinct peer zones within its weather group.
    """Return candidate pairs, event support and exposure scales from event metrics.

    Selection is deterministic within each weather group and rain class,
    with at most five distinct peer zones per target local episode.
    """
    e = e[INPUTS].copy()  # Work on a copy containing only the columns needed to construct matches.
    # Normalize both ends of each local episode before comparing event windows.
    for k in ["start_utc","end_utc"]:
        e[k] = pd.to_datetime(e[k],utc=True)  # Parse this boundary as a timezone-aware UTC timestamp.
    scales = {}  # Collect one scale per exposure so their differences can be combined.
    for k in RAIN_FIELDS:  # Estimate the spread of each rainfall or duration measure separately.
        # Log-transform available exposure values to reduce the influence of large episodes.
        x = np.log1p(e[k].dropna().to_numpy(float))
        # Use the interquartile range of the transformed values as the primary scale.
        s = np.subtract(*np.quantile(x,[.75,.25]))
        # Fall back to the standard deviation when the quartiles coincide within tolerance.
        if s<=EPS: s=x.std()
        # Use unit scale if neither spread estimate is large enough to divide by.
        scales[k] = float(s if s>EPS else 1.)
    allp = []  # Collect the candidate-pair tables for all targets with possible peers.
    support = []  # Collect a support record even for targets with no usable peer.
    # Restrict each matching search to episodes in one common weather process.
    for cg, allg in e.groupby("canonical_group",sort=True):
        # Consider only peer episodes with a positive centered starting deficit.
        d = allg[allg.centered_D0.gt(0)]
        # Evaluate support for every target in the group, including ineligible targets.
        for a in allg.itertuples(index=False):
            # Start a support record keyed by the target episode, zone and weather group.
            stat = dict(event_id=a.event_id,canonical_group=cg,zone_id=a.zone_id,
                        # Initialize the count passing both the weather and starting-deficit criteria.
                        weather_and_centered_start_candidates=0,
                        # Track candidates with matching and differing known subsequent-rain classes separately.
                        same_known_class_candidates=0,cross_known_class_candidates=0,
                        # Track unknown-class candidates and the number selected from the main pool.
                        unknown_class_candidates=0,selected_main=0)
            # Check whether the target itself has a positive centered starting deficit.
            if not a.centered_D0>0:
                # Record why matching cannot proceed for this target.
                stat["peer_status"]="D0_NONPOSITIVE_OR_UNKNOWN"
                # Retain the unsupported target in the audit and advance to the next episode.
                support.append(stat); continue
            q=d[d.zone_id.ne(a.zone_id)].copy()  # Exclude other episodes from the target's own zone.
            # Apply the main-study weather eligibility restriction when the target qualifies.
            if a.primary_model_weather_eligible:
                # Require candidate peers to meet the same main-study eligibility condition.
                q=q[q.primary_model_weather_eligible]
            # Measure shared rain duration as the earlier end minus the later start, in hours.
            overlap=(q.end_utc.clip(upper=a.end_utc)-q.start_utc.clip(lower=a.start_utc)).dt.total_seconds()/3600
            # Measure absolute separation between local rain-ending times, in hours.
            endgap=(q.end_utc-a.end_utc).abs().dt.total_seconds()/3600
            # Measure absolute separation between local rain-starting times, in hours.
            startgap=(q.start_utc-a.start_utc).abs().dt.total_seconds()/3600
            # Express exposure mismatch as the larger value divided by the smaller one.
            ratios={k:np.maximum(q[k],getattr(a,k))/np.minimum(q[k],getattr(a,k))
                    # Compare event totals, hourly peaks and episode durations on this ratio scale.
                    for k in ["total_mm","peak_mm","duration_h"]}
            # Require positive overlap, endings within six hours and centered deficits within 0.10.
            mask=(overlap>0)&(endgap<=6)&((q.centered_D0-a.centered_D0).abs()<=.1+EPS)
            # Apply the twofold upper bound, with numerical tolerance, to all three exposure ratios.
            for v in ratios.values():mask &= v<=2+1e-8
            q=q[mask]  # Keep only candidates passing every initial matching condition.
            # Save the number that passed before dividing candidates into rain classes.
            stat["weather_and_centered_start_candidates"]=len(q)
            if not len(q):  # Handle targets for which the initial criteria eliminate every candidate.
                # Record the absence of a weather- and start-comparable peer.
                stat["peer_status"]="NO_WEATHER_START_COMPARABLE_PEER"
                # Save the support record and skip ranking for this target.
                support.append(stat);continue
            # Identify both local episodes in each candidate pair.
            p=pd.DataFrame(dict(target_event=a.event_id,peer_event=q.event_id,
                # Retain both zones and their common weather group for later aggregation.
                target_zone=a.zone_id,peer_zone=q.zone_id,canonical_group=cg,
                period=a.period,  # Carry the target's chronological study period into the pair table.
                # Mark whether this target can enter the main analysis or only the descriptive pool.
                inference_scope="MAIN_VERIFIED_CLOSED" if a.primary_model_weather_eligible else "DESCRIPTIVE_ONLY",
                # Save actual overlap and start/end separations for each surviving candidate.
                overlap_h=overlap.loc[q.index],end_gap_h=endgap.loc[q.index],start_gap_h=startgap.loc[q.index],
                # Retain both centered starting deficits for inspection.
                target_D0=a.centered_D0,peer_D0=q.centered_D0,
                # Record the signed target-minus-peer centered starting gap.
                centered_start_difference=a.centered_D0-q.centered_D0,
                # Keep both uncentered starting deficits for raw-gap comparisons.
                target_raw_D0=a.raw_D0,peer_raw_D0=q.raw_D0,
                # Flag pairs in which both raw starting deficits are positive.
                both_raw_positive=(a.raw_D0>0)&q.raw_D0.gt(0),
                # Flag raw starting deficits within the same 0.10 absolute tolerance.
                raw_start_caliper=(a.raw_D0-q.raw_D0).abs()<=.1+EPS,
                # Record the target and candidate's observed subsequent-rain classes.
                target_rain_class=a.rain_class,peer_rain_class=q.rain_class,
                # Record whether the target has complete six-hour follow-up rainfall.
                target_post_weather_complete=a.post6_weather_complete,
                # Preserve the corresponding six-hour rainfall coverage flag for each candidate.
                peer_post_weather_complete=q.post6_weather_complete))
            # Detect pairs for which either subsequent-rain class is unknown.
            unknown=(a.rain_class=="UNKNOWN")|q.rain_class.eq("UNKNOWN")
            # Assign such pairs to the unknown-class comparison pool.
            p["comparison_pool"]=np.where(unknown,"UNKNOWN_RAIN_CLASS",
                 # Split the remaining pairs according to whether their known rain classes agree.
                 np.where(q.rain_class.eq(a.rain_class),"SAME_KNOWN_RAIN_CLASS","CROSS_RAIN_CLASS"))
            # Start the distance with the absolute centered-gap difference in units of 0.10.
            comps=[abs(a.centered_D0-q.centered_D0).to_numpy()/.1,
                   # Add end-time and start-time separations in units of six hours.
                   endgap.loc[q.index].to_numpy()/6,startgap.loc[q.index].to_numpy()/6]
            # Count missing exposure-distance components separately for each candidate.
            missing=np.zeros(len(q),int)
            # Add one scaled difference for every named rainfall or duration measure.
            for k in RAIN_FIELDS:
                # Copy the target's exposure into the pair-level audit fields.
                p["target_"+k]=getattr(a,k)
                p["peer_"+k]=q[k]  # Record the corresponding exposure observed for the peer.
                # Scale the absolute difference of log-transformed exposures by their reference spread.
                delta=abs(np.log1p(q[k].to_numpy())-np.log1p(getattr(a,k)))/scales[k]
                missing+=~np.isfinite(delta)  # Count pairs missing a finite distance for this exposure.
                # Use a unit contribution when this exposure distance cannot be measured.
                comps.append(np.where(np.isfinite(delta),delta,1.))
                # Preserve the unfilled exposure distance so missing components remain visible.
                p[k+"_distance"]=delta
            # Represent a target with no renewed rain as hour seven, just beyond the window.
            at=7. if a.rain_class=="NO_RERAIN" else a.rerain_first_hour
            # Apply the same hour-seven convention to peers with no renewed rain.
            bt=np.where(q.rain_class.eq("NO_RERAIN"),7.,q.rerain_first_hour)
            td=abs(at-bt)/6  # Scale the target-peer difference in renewed-rain timing by six hours.
            missing+=~np.isfinite(td)  # Count unknown timing as another missing distance component.
            # Include the timing difference, assigning one unit when it is missing.
            comps.append(np.where(np.isfinite(td),td,1.))
            # Save the original target renewed-rain hour alongside its ranking inputs.
            p["target_rerain_hour"]=a.rerain_first_hour
            # Save the original peer renewed-rain hour for the pair.
            p["peer_rerain_hour"]=q.rerain_first_hour
            # Retain the scaled renewed-rain timing difference before missing-value substitution.
            p["rerain_time_distance"]=td
            # Record how many components of the ranking distance were missing.
            p["exposure_distance_missing_dimensions"]=missing
            # Average the component distances with equal weight for each candidate.
            p["distance"]=np.mean(comps,axis=0)
            # Attach the three original exposure ratios to the matched candidate rows.
            for k,v in ratios.items():p[k+"_ratio"]=v.loc[q.index]
            # Start every candidate as unselected with no within-pool rank.
            p["selected"]=False;p["selected_rank"]=np.nan
            # Rank candidates independently within each subsequent-rain comparison pool.
            for pool,pp in p.groupby("comparison_pool"):
                # Choose the support-count field for the matching known rain class.
                key={"SAME_KNOWN_RAIN_CLASS":"same_known_class_candidates",
                     # Choose the corresponding support field for cross-class and unknown-class pools.
                     "CROSS_RAIN_CLASS":"cross_known_class_candidates","UNKNOWN_RAIN_CLASS":"unknown_class_candidates"}[pool]
                # Record how many candidates were available in this pool before ranking.
                stat[key]=len(pp)
                # Sort by distance, timing and event ID, then keep the first five distinct zones.
                take=pp.sort_values(["distance","end_gap_h","start_gap_h","peer_event"]).drop_duplicates("peer_zone").head(5).index
                p.loc[take,"selected"]=True  # Mark those retained candidate rows as selected.
                # Number the selected candidates from one in their sorted order.
                p.loc[take,"selected_rank"]=np.arange(len(take))+1
                # Record the selected count from the matching known-rain pool for the main comparison.
                if pool=="SAME_KNOWN_RAIN_CLASS":stat["selected_main"]=len(take)
            # Distinguish main-pool support from matches available only in other pools.
            stat["peer_status"]="MAIN_PEERS_SELECTED" if stat["selected_main"] else "ONLY_CROSS_OR_UNKNOWN_RAIN_CLASS"
            # Save the target's support record and its complete candidate-pair table.
            support.append(stat);allp.append(p)
    # Return the combined pairs, support counts and exposure scales.
    return pd.concat(allp,ignore_index=True),pd.DataFrame(support),scales
