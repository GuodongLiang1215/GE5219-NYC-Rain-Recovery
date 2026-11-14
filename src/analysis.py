"""Recalculate the report analyses from frozen hourly predictions and metadata."""
from pathlib import Path  # Construct paths to the frozen input files and the generated analysis tables.
import json  # Serialize the main analysis counts and recurrence summary.
import numpy as np  # Calculate finite-value masks, hourly metrics and numerical summaries.
import pandas as pd  # Join event catalogues and aggregate episodes, zones and weather groups.
# Reuse the hourly completeness, threshold-crossing and six-hour recovery calculations.
from .recovery_metrics import complete, hit_metrics, stats
from .peer_selection import build as select_peers  # Import the study's fixed peer-matching procedure.
# Reuse horizon-specific service measures and complete-window averages.
from .service_metrics import process, full

EPS = 1e-12  # Allow for floating-point roundoff when classifying ties and comparing starting gaps.


def read(path):  # Load a frozen CSV while preserving the precision used in the original estimates.
    """Read a numeric CSV without rounding its stored floating-point values."""
    # Read the whole file with round-trip parsing of floating-point values.
    return pd.read_csv(path, float_precision="round_trip", low_memory=False)


def write(folder, name, frame):  # Save a result table using the requested filename and compression.
    """Write one analysis table, retaining unavailable observations as empty cells."""
    # Create the analysis destination including any missing parent directories.
    folder.mkdir(parents=True, exist_ok=True)
    frame.to_csv(folder/name, index=False,  # Write the DataFrame without an extra row-index column.
                 # Apply gzip only to .gz outputs and fix its timestamp for reproducible bytes.
                 compression={"method":"gzip", "mtime":0} if name.endswith(".gz") else None)


# Derive raw and reference-centered hourly gaps with the input's eligibility fields.
def activity_gaps(hourly, center="b", usable="raw_usable"):
    """Calculate 1-Y/P and its normal-reference-centered counterpart."""
    h = hourly.copy()  # Work on a copy so the caller's hourly observations remain available.
    # Use the named Boolean eligibility field when present or the service table's VALID status.
    ok = h[usable].eq(True) if usable in h else h.raw_status.eq("VALID")
    # Require finite observations and predictions, with at least ten predicted pickups.
    ok &= np.isfinite(h.Y) & np.isfinite(h.P) & h.P.ge(10)
    # Express the raw activity deficit as the fraction of predicted pickups missing.
    h["gap"] = np.where(ok, 1-h.Y/h.P, np.nan)
    # Subtract the normal reference residual to center the raw deficit.
    h["gap_centered"] = h.gap-h[center]
    # Check for the Boolean centered eligibility flag used in the main hourly input.
    if "centered_usable" in h:
        # Mask centered values for rows outside that eligibility rule.
        h.loc[~h.centered_usable.eq(True), "gap_centered"] = np.nan
    # Check for the alternative centered status field used in the service input.
    elif "centered_status" in h:
        # Mask every centered service observation whose status is not VALID.
        h.loc[~h.centered_status.eq("VALID"), "gap_centered"] = np.nan
    return h  # Return the original hourly fields together with the two calculated gaps.


# Calculate event-level weather, starting deficits, crossings and late-window recovery.
def recovery(data, out):
    """Measure each registered event on its own rain-ending clock."""
    # Read the local event catalogue and its weather and baseline eligibility fields.
    registry = read(data/"event_registry.csv")
    # Construct both gap trajectories for every registered event with hourly predictions.
    h = activity_gaps(read(data/"hourly_predictions.csv.gz"))
    # Read the common weather groups and their start times for the chronological split.
    catalogue = read(data/"weather_groups.csv")
    # Collect groups containing at least one weather-eligible primary-model event.
    ids = set(registry.loc[registry.primary_model_weather_eligible, "canonical_group"])
    # Sort those groups by UTC start, breaking ties by group identifier.
    split = catalogue[catalogue.canonical_group.isin(ids)].sort_values(["start_utc","canonical_group"]).copy()
    # Number the eligible groups chronologically starting at one.
    split["time_rank"] = np.arange(len(split))+1
    # Assign the first two thirds of groups, rounded down, to exploration and the rest to later evaluation.
    split["period"] = np.where(split.time_rank <= 2*len(split)//3,"exploration","later")
    # Build a weather-group lookup for the study-period label.
    periods = split.set_index("canonical_group").period
    rows = []  # Collect one set of recovery and rainfall measurements per local event.
    # Evaluate each event's hourly observations without reordering the event identifiers.
    for eid, w in h.groupby("event_id", sort=False):
        w = w.sort_values("offset_from_end")  # Order readings relative to the local rain-ending hour.
        # Index observations by offset so unobserved hours can be retained as missing.
        ix = w.set_index("offset_from_end")
        post = ix.reindex(range(6))  # Align the six post-rain bins to offsets zero through five.
        # Extract follow-up rainfall in the same order as the recovery hours.
        rain = post.rain_mean_mm.to_numpy(float)
        # Extract the hourly rainfall readings tagged as preceding the episode.
        pre = w.loc[w.segment.eq("pre"),"rain_mean_mm"].to_numpy(float)
        # Mark known rainfall and identify renewed wet hours above 0.100001 mm.
        known = np.isfinite(rain); wet = known & (rain > .100001)
        r = dict(event_id=eid,  # Start the event's measurements with its catalogue identifier.
                 # Sum preceding rainfall only when all six required readings are available.
                 pre6_rain_mm=pre.sum() if complete(pre,6) else np.nan,
                 # Sum rainfall during post-rain hours one through three if all three are observed.
                 post13_rain_mm=rain[:3].sum() if complete(rain[:3],3) else np.nan,
                 # Sum rainfall during post-rain hours four through six if all three are observed.
                 post46_rain_mm=rain[3:].sum() if complete(rain[3:],3) else np.nan,
                 # Report the maximum follow-up rainfall only over a complete six-hour window.
                 post6_peak_mm=rain.max() if complete(rain,6) else np.nan,
                 # Record whether rainfall is known for every follow-up hour.
                 post6_weather_complete=bool(known.all()),
                 # Identify renewed rain if observed and reserve the dry class for complete follow-up.
                 rain_class="RERAIN" if wet.any() else "NO_RERAIN" if known.all() else "UNKNOWN",
                 # Store the first observed renewed-rain hour on a one-based clock.
                 rerain_first_hour=float(np.flatnonzero(wet)[0]+1) if wet.any() else np.nan,
                 # Confirm its timing only when every earlier rainfall reading was available.
                 rerain_first_exact=bool(known[:np.flatnonzero(wet)[0]].all()) if wet.any() else False)
        # Measure recovery once for raw deficits and once for centered deficits.
        for mode, field in [("raw","gap"),("centered","gap_centered")]:
            # Select the final two hourly deficits before the local rain ending.
            d = ix.reindex([-2,-1])[field].to_numpy(float)
            # Define D0 as their mean only when both starting values are finite.
            d0 = float(d.mean()) if complete(d,2) else np.nan
            # Extract the ordered post-rain trajectory for the current gap definition.
            g = post[field].to_numpy(float)
            r[f"{mode}_D0"] = d0  # Save the event's starting deficit under the corresponding mode.
            for H in [3,6]:  # Measure progress at both three and six hours.
                # Select the full horizon and its late portion, hours 2 to 3 or 4 to 6.
                v=g[:H]; late=g[1:3] if H==3 else g[3:6]
                # Sum hourly deficits only if the whole selected horizon is observed.
                r[f"{mode}_A{H}"] = float(v.sum()) if complete(v,H) else np.nan
                # Average late-window deficits only when every late hour is observed.
                r[f"{mode}_late{H}"] = float(late.mean()) if complete(late,len(late)) else np.nan
                # Compute late gap closure as the starting deficit minus the late mean.
                r[f"{mode}_Qlate{H}"] = d0-r[f"{mode}_late{H}"]
                # Count hours still above one fifth of positive D0 across a complete window.
                r[f"{mode}_H80_{H}"] = float((v>.2*d0).sum()) if d0>0 and complete(v,H) else np.nan
                # Evaluate thresholds corresponding to 50 percent and 80 percent gap closure.
                for label,factor in [("T50",.5),("T80",.2)]:
                    # Attach first crossing, reversal and rainfall measurements for this mode and horizon.
                    r.update({f"{mode}_{label}_{H}_{k}":val for k,val in hit_metrics(v,d0,rain[:H],factor,H).items()})
            for k in range(6):  # Export the six individual hourly deficits in their chronological order.
                # Store the current gap under its mode and one-based hour label.
                r[f"{mode}_g_h{k+1}"]=g[k]
        rows.append(r)  # Append the completed weather and recovery record for this event.
    # Join one measurement record to each matching catalogue event by event ID.
    events = registry.merge(pd.DataFrame(rows),on="event_id",validate="one_to_one")
    # Attach the chronological period or label events outside the split as descriptive.
    events["period"] = events.canonical_group.map(periods).fillna("DESCRIPTIVE_ONLY")
    # Mark the analysis scope using the event's primary-model weather eligibility.
    events["inference_scope"] = np.where(events.primary_model_weather_eligible,"MAIN_VERIFIED_CLOSED","DESCRIPTIVE_ONLY")
    # Save hourly raw and centered curves for later plots and case comparisons.
    write(out,"hourly_curves.csv.gz",h)
    # Save the joined event metrics and their eligibility information.
    write(out,"recovery_event_metrics.csv",events)
    # Save the chronological group assignments that define exploration and later periods.
    write(out,"temporal_split_groups.csv",split)
    return events,h,split  # Return the event metrics, hourly curves and weather-group split.


# Compare each target with matched peers and evaluate recurrence in the later period.
def recurrence(data, out, events):
    """Apply the original deterministic matching and equal-weight recurrence rule."""
    # Calculate candidate matches and counts of peer support for every target.
    candidates,support,_ = select_peers(events)
    # Keep the pairs selected by the deterministic ranking within each comparison pool.
    peers = candidates[candidates.selected].copy()
    # Save the selected pairs with their ranking and matching details.
    write(out,"peer_selection.csv",peers)
    # Save target-level support, including targets without a usable match.
    write(out,"event_peer_support.csv",support)
    # Index event metrics so target and peer rows can be retrieved in pair order.
    ev = events.set_index("event_id")
    # Limit the recurrence analysis to pairs within the verified main-study scope.
    selected = peers[peers.inference_scope.eq("MAIN_VERIFIED_CLOSED") &
                     # Require target and peer to share a known subsequent-rain class.
                     peers.comparison_pool.eq("SAME_KNOWN_RAIN_CLASS")].copy()
    # Collect separate tables for local-episode scores and weather-group scores.
    local_tables=[];group_tables=[]
    # Identify the target zone, group, period, scope and rain pool for each comparison.
    keys=["target_zone","canonical_group","period","inference_scope","comparison_pool"]
    # Calculate relative scores under both centered and raw activity gaps.
    for mode in ["centered","raw"]:
        for H in [6,3]:  # Evaluate six-hour and three-hour comparisons separately.
            # Start this mode and horizon with an independent copy of the selected pairs.
            p=selected.copy()
            # Align target event metrics with the order of the selected pair rows.
            a=ev.loc[p.target_event].reset_index(drop=True)
            # Align peer event metrics with those same rows.
            b=ev.loc[p.peer_event].reset_index(drop=True)
            # Reset pair indices so arithmetic and assignment use matching positions.
            p=p.reset_index(drop=True)
            # Define S80 as target shortfall hours minus peer shortfall hours.
            p["S80"]=a[f"{mode}_H80_{H}"]-b[f"{mode}_H80_{H}"]
            # Require both starting deficits to be positive for the progress comparison.
            applicable=a[f"{mode}_D0"].gt(0)&b[f"{mode}_D0"].gt(0)
            # Define Sprogress as peer late closure minus target late closure for eligible pairs.
            p["Sprogress"]=(b[f"{mode}_Qlate{H}"]-a[f"{mode}_Qlate{H}"]).where(applicable)
            # Aggregate the selected peer comparisons within each target episode.
            local=p.groupby(["target_event"]+keys,dropna=False).agg(
                # Average peer differences for each metric and count all selected peers.
                S80=("S80","mean"),Sprogress=("Sprogress","mean"),selected_peers=("peer_event","size"),
                # Count the peers supplying a finite score separately for each metric.
                S80_contributing_peers=("S80","count"),Sprogress_contributing_peers=("Sprogress","count")).reset_index()
            # Aggregate the local-episode scores within each zone and weather group.
            group=local.groupby(keys,dropna=False).agg(
                # Average episode-level scores with equal weight and count selected episodes.
                S80=("S80","mean"),Sprogress=("Sprogress","mean"),selected_local_events=("target_event","size"),
                # Record how many episodes actually contributed to each group metric.
                S80_contributing_events=("S80","count"),Sprogress_contributing_events=("Sprogress","count"),
                # Total the selected peers and those contributing to the S80 comparisons.
                selected_peers=("selected_peers","sum"),S80_contributing_peers=("S80_contributing_peers","sum"),
                # Total the peers contributing to the group progress score.
                Sprogress_contributing_peers=("Sprogress_contributing_peers","sum")).reset_index()
            # Label both aggregation levels with the analysis settings that produced them.
            for frame in [local,group]:
                # Store gap definition, horizon length and the all-selected subset label.
                frame["mode"]=mode;frame["H"]=H;frame["subset"]="all_selected"
            # Collect the finished event-level and group-level tables for this combination.
            local_tables.append(local);group_tables.append(group)
    # Combine group scores across modes and horizons in a single table.
    groups=pd.concat(group_tables,ignore_index=True)
    # Save the corresponding target-event peer averages.
    write(out,"event_peer_recovery.csv",pd.concat(local_tables,ignore_index=True))
    # Save the zone-by-weather-group recovery comparisons.
    write(out,"zone_weather_group_recovery.csv",groups)
    # Read the fixed cohort, including zones without usable later comparisons.
    cohort=read(data/"recovery_cohort.csv")
    labels=read(data/"zone_labels.csv")  # Read the zone names and boroughs used to label the cohort.
    # Check whether the cohort needs descriptive geography from the zone lookup.
    if "zone_name" not in cohort or "borough" not in cohort:
        # Attach one verified name and borough to each cohort zone identifier.
        cohort=cohort[["zone_id"]].merge(labels[["zone_id","zone_name","borough"]],on="zone_id",validate="one_to_one")
    # Use centered six-hour group scores to evaluate the primary recurrence rule.
    primary=groups[groups["mode"].eq("centered")&groups.H.eq(6)]
    periods=[]  # Collect one recurrence summary per zone, metric and study period.
    for z in cohort.itertuples(index=False):  # Evaluate every zone in the fixed cohort.
        for metric in ["S80","Sprogress"]:  # Summarize shortfall duration and gap closure independently.
            # Evaluate exploration, later observations and the combined period.
            for period in ["exploration","later","all"]:
                # Select primary group scores for this target zone.
                q=primary[primary.target_zone.eq(z.zone_id)]
                # Restrict to the requested period unless the combined summary is being built.
                if period!="all":q=q[q.period.eq(period)]
                # Drop missing scores and count valid and positive weather groups, allowing for roundoff.
                x=q[metric].dropna().to_numpy();n=len(x);positive=int((x>EPS).sum())
                # Identify the target zone and metric for this period summary.
                periods.append(dict(zone_id=z.zone_id,zone_name=z.zone_name,borough=z.borough,metric=metric,
                    # Store selected and valid group counts, with the mean over valid scores.
                    period=period,selected_groups=len(q),valid_groups=n,mean=x.mean() if n else np.nan,
                    # Count groups with positive, negative and effectively tied contrasts.
                    positive_groups=positive,negative_groups=int((x<-EPS).sum()),tied_groups=int((abs(x)<=EPS).sum()),
                    # Calculate the positive fraction only when a valid group exists.
                    positive_fraction=positive/n if n else np.nan,
                    # Apply the rule requiring three valid groups, at least two thirds positive and a positive mean.
                    rule=bool(n>=3 and positive/n>=2/3-EPS and x.mean()>0)))
    # Assemble the period summaries for subsequent earlier-to-later evaluation.
    summaries=pd.DataFrame(periods)
    records=[]  # Collect one later-period classification per zone and metric.
    # Classify all zones, retaining zones outside the earlier candidate list.
    for z in cohort.itertuples(index=False):
        # Assess the two recurrence measures separately for the same zone.
        for metric in ["S80","Sprogress"]:
            # Look up the exploration and later summaries for this zone and measure.
            q=summaries[summaries.zone_id.eq(z.zone_id)&summaries.metric.eq(metric)].set_index("period")
            # Retrieve the two periods and the number of valid later groups.
            a,b=q.loc["exploration"],q.loc["later"];n=int(b.valid_groups)
            # Classify the later mean as unknown, positive, negative or tied using the tolerance.
            direction="UNKNOWN" if n==0 else "POSITIVE" if b["mean"]>EPS else "NEGATIVE" if b["mean"]<-EPS else "TIED"
            # First distinguish earlier nonmembers and candidates with no selected later group.
            state=("NOT_IN_FROZEN_LIST" if not a.rule else "NO_LATER_COMPARISON" if b.selected_groups==0 else
                # Distinguish selected groups lacking a metric from only one or two valid later groups.
                "LATER_METRIC_UNAVAILABLE" if n==0 else "ONE_TWO_LATER_GROUPS" if n<3 else
                # For well-supported candidates, record a repeated rule or the remaining direction category.
                "REPEATS_ORIGINAL_POINT_RULE" if b.rule else "POSITIVE_NOT_RULE" if direction=="POSITIVE" else "NEGATIVE_OR_TIED")
            # Create the classification record for this zone and metric.
            records.append(dict(zone_id=z.zone_id,zone_name=z.zone_name,borough=z.borough,metric=metric,
                # Store earlier membership and counts of earlier valid and later selected groups.
                exploration_member=bool(a.rule),exploration_groups=int(a.valid_groups),later_selected_groups=int(b.selected_groups),
                # Keep the valid later count and both period means in their native units.
                later_valid_groups=n,exploration_mean=a["mean"],later_mean=b["mean"],
                # Scale progress to percentage points for display and keep the later direction label.
                later_mean_display=b["mean"]*(100 if metric=="Sprogress" else 1),later_direction=direction,
                # Group later support counts and evaluate the later rule only for earlier candidates.
                later_support_bin=str(n) if n<3 else ">=3",replication_status=state,later_rule=bool(b.rule) if a.rule else None,
                # Store the number of positive later groups behind the classification.
                later_positive_groups=int(b.positive_groups)))
    status=pd.DataFrame(records)  # Assemble the later recurrence status table.
    # Save all period-specific recurrence summaries and rule flags.
    write(out,"recurrence_period_summary.csv",summaries)
    # Save the earlier membership and later evaluation for each zone and metric.
    write(out,"replication_status_long.csv",status)
    # Return the selected pairs, group scores and replication classifications.
    return peers,groups,status


# Compare taxi and ride-hailing recovery on the fixed matched service cases.
def service_comparison(data,out):
    """Recompute separate complete H80 and Qlate samples from fixed service cases."""
    # Load the fixed case list with zone, group and recovery start time.
    cases=read(data/"service_cases.csv")
    # Recalculate service gaps using the service-status fields and its normal reference center.
    h=activity_gaps(read(data/"service_hourly_predictions.csv.gz"),center="normal_error_center",usable="raw_status_unused")
    rows=[]  # Collect recovery measurements for each service case and gap definition.
    # Visit each fixed case, retaining its original event identity.
    for ev in cases.itertuples(index=False):
        v=h[h.event_id.eq(ev.event_id)]  # Select hourly observations for both services in that case.
        # Evaluate both raw and normal-centered gap trajectories.
        for mode,col in [("raw","gap"),("centered","gap_centered")]:
            # Identify this result by event, zone, zone name and common weather process.
            r=dict(event_id=ev.event_id,zone_id=ev.zone_id,zone_name=ev.zone_name,canonical_group=ev.canonical_group,
                   # Preserve both the UTC and New York recovery start times and the gap definition.
                   h1_utc=ev.h1_utc,h1_NYC=ev.h1_NYC,mode=mode)
            measured={}  # Store each service's measurements by observation horizon for later pairing.
            # Calculate the two services independently on the same local event clock.
            for service in ["Taxi","RideHailing"]:
                # Index one service's hourly gaps by offset from the end of local rain.
                sv=v[v.service.eq(service)].set_index("offset_from_end")
                # Average the two starting hours and align the 24 hourly post-rain gaps.
                d0=full(sv.reindex([-2,-1])[col]);g=sv.reindex(range(24))[col].to_numpy(float)
                # Evaluate both the short six-hour follow-up and the full 24-hour window.
                for H in [6,24]:
                    # Calculate this service's process measures and keep them for the paired contrast.
                    met=process(g,d0,H);measured[(service,H)]=met
                    # Add the measures to the case record under service- and horizon-specific names.
                    r.update({f"{service}_H{H}_{k}":value for k,value in met.items()})
            # Retrieve the two services' own starting deficits from the six-hour records.
            td,rd=measured[("Taxi",6)]["D0"],measured[("RideHailing",6)]["D0"]
            # Require positive initial deficits within 0.10 of each other, allowing for roundoff.
            eligible=td>0 and rd>0 and abs(td-rd)<=.1+EPS
            # Record whether this pair passes the starting-gap comparison.
            r["own_start_eligible"]=bool(eligible)
            for H in [6,24]:  # Calculate the two paired service contrasts over each horizon.
                # Retrieve taxi and ride-hailing results at the same follow-up horizon.
                tm,rm=measured[("Taxi",H)],measured[("RideHailing",H)]
                # Subtract ride-hailing H80 from taxi H80, in hours, when starting gaps match.
                r[f"H{H}_S80"]=tm["H80"]-rm["H80"] if eligible else np.nan
                # Convert ride-hailing minus taxi late closure to percentage points for eligible pairs.
                r[f"H{H}_Sprogress_pp"]=100*(rm["Qlate"]-tm["Qlate"]) if eligible else np.nan
            rows.append(r)  # Collect the event's paired service results for this gap definition.
    # Assemble event records and prepare the weather-group and overall summaries.
    measured=pd.DataFrame(rows);groups=[];summary=[]
    for mode,v in measured.groupby("mode"):  # Summarize raw and centered results independently.
        # Pair each contrast with its underlying service measure, duration or late closure.
        for metric,field in [("S80","H80"),("Sprogress_pp","Qlate")]:
            # Evaluate the six-hour and available 24-hour results on the original fixed case list.
            for sample,H,q in [("ORIGINAL20_H6",6,v),("FIXED20_H24_AVAILABLE",24,v),
                    # Evaluate H6 on the subset with a valid 24-hour contrast for this metric.
                    ("H24_METRIC_COMMON_H6",6,v[v[f"H24_{metric}"].notna()]),
                    # Evaluate H24 on that same metric-specific supported subset.
                    ("H24_METRIC_COMMON_H24",24,v[v[f"H24_{metric}"].notna()])]:
                # Drop unavailable contrasts at this horizon and choose units for the individual service measures.
                key=f"H{H}_{metric}";q=q[q[key].notna()].copy();scale=100 if metric=="Sprogress_pp" else 1
                # Express taxi's own measure in hours or percentage points as appropriate.
                q["Taxi_value"]=q[f"Taxi_H{H}_{field}"]*scale
                # Apply the same reporting units to the ride-hailing service measure.
                q["RideHailing_value"]=q[f"RideHailing_H{H}_{field}"]*scale
                vals=[]  # Collect one contrast per weather group with usable cases.
                # Include all weather groups represented in the fixed case list in the group table.
                for cg in cases.canonical_group.unique():
                    # Select this group's cases and average event differences and service measures within zones.
                    g=q[q.canonical_group.eq(cg)];z=g.groupby("zone_id")[[key,"Taxi_value","RideHailing_value"]].mean()
                    # Give each represented zone equal weight in the weather-group contrast.
                    difference=z[key].mean()
                    # Identify this group estimate by mode, metric, sample, horizon and weather group.
                    groups.append(dict(mode=mode,metric=metric,sample=sample,horizon=H,canonical_group=cg,
                        # Record contributing event and zone counts and the zone-averaged service measures.
                        events=len(g),zones=len(z),difference=difference,Taxi_value=z.Taxi_value.mean(),RideHailing_value=z.RideHailing_value.mean(),
                        # List the contributing event IDs and count all original cases in this weather group.
                        event_ids=";".join(g.event_id),fixed_cohort_events=int(cases.canonical_group.eq(cg).sum()),
                        # Label H80 contrasts in hours and progress contrasts in percentage points.
                        unit="hours" if metric=="S80" else "percentage_points"))
                    # Use groups with contributing cases when forming the across-group average.
                    if len(g):vals.append(difference)
                # Convert the observed group differences to an array for sign counts and the mean.
                a=np.asarray(vals)
                # Record the number of contributing events and distinct zones for the service summary.
                summary.append(dict(mode=mode,metric=metric,sample=sample,horizon=H,events=len(q),zones=q.zone_id.nunique(),
                    # Count weather groups, average them with equal weight and count positive differences.
                    groups=q.canonical_group.nunique(),equal_group_mean=full(a),positive_groups=int((a>EPS).sum()),
                    # Count negative and effectively tied differences at the same numerical tolerance.
                    negative_groups=int((a<-EPS).sum()),tied_groups=int((abs(a)<=EPS).sum())))
    # Save each service case's detailed recovery measurements and paired contrasts.
    write(out,"service_event_process.csv",measured)
    # Save group-level service estimates and their supporting cases.
    write(out,"service_weather_group_summary.csv",pd.DataFrame(groups))
    # Save the metric-specific across-group service comparisons.
    write(out,"service_comparison_summary.csv",pd.DataFrame(summary))
    # Return the case-level measurements and the across-group summary.
    return measured,pd.DataFrame(summary)


# Estimate storm excess relative to paired normal windows and their fixed peers.
def normal_excess(data,out):
    """Recalculate paired storm/normal metrics with fixed leave-date centers and peers."""
    # Load hourly storm and normal-day observations, predictions and leave-date-out centers.
    hourly=read(data/"normal_hourly_predictions.csv.gz")
    # Require available source data, finite counts and at least ten predicted pickups.
    valid=np.isfinite(hourly.Y)&np.isfinite(hourly.P)&hourly.P.ge(10)&hourly.source_available
    # Center each eligible raw gap using the reference that excludes the paired normal date.
    hourly["gap_centered_lodo"]=np.where(valid,1-hourly.Y/hourly.P,np.nan)-hourly.b_minus_d
    # Load each window's fixed event linkage and calendar, weather and source eligibility.
    meta=read(data/"normal_window_metadata.csv.gz")
    rows=[]  # Collect metrics for each paired storm and normal window.
    # Evaluate the trajectories sharing one normal-window identifier.
    for wid,w in hourly.groupby("normal_window_id",sort=False):
        r=dict(normal_window_id=wid)  # Start a metric record keyed to this particular pairing.
        # Measure both the storm window and the corresponding normal window.
        for trajectory in ["storm","normal"]:
            # Align its centered gaps to two starting hours and six post-rain or comparison hours.
            g=w[w.trajectory.eq(trajectory)].set_index("offset_from_end").reindex(range(-2,6)).gap_centered_lodo.to_numpy()
            # Keep each trajectory's recovery measures under its own prefix.
            r.update({trajectory+"_"+k:v for k,v in stats(g).items()})
        # Match windows with positive starting deficits separated by no more than 0.10.
        matched=r["storm_D0"]>0 and r["normal_D0"]>0 and abs(r["storm_D0"]-r["normal_D0"])<=.1+EPS
        r["start_matched"]=bool(matched)  # Save the starting-gap eligibility flag with the window.
        # Define excess duration as storm minus normal and excess shortfall in closure as normal minus storm.
        for met,base,direction in [("E80","H80",1),("EQ","Qlate",-1)]:
            # Calculate each signed excess only for start-matched trajectories.
            r[met]=direction*(r["storm_"+base]-r["normal_"+base]) if matched else np.nan
        rows.append(r)  # Collect this pairing's start eligibility and measured excesses.
    # Preserve every metadata window and attach at most one measured record by ID.
    windows=meta.merge(pd.DataFrame(rows),on="normal_window_id",how="left",validate="one_to_one")
    for m in ["E80","EQ"]:  # Apply calendar, weather and source eligibility to both excess measures.
        # Set excess values missing for windows that failed those eligibility checks.
        windows.loc[~windows.calendar_weather_source_eligible,m]=np.nan
    # Average each event's eligible window excesses before comparing events with peers.
    event=windows.groupby("event_id")[["E80","EQ"]].mean()
    # Read the fixed peer roster and index the original candidate membership by zone.
    p=read(data/"normal_fixed_peers.csv");membership=read(data/"normal_target_membership.csv").set_index("zone_id")
    pairs=[]  # Collect peer contrasts for both excess measures.
    # Keep duration differences in hours and convert closure contrasts to percentage points.
    for metric,field,scale in [("S80","E80",1),("Sprogress","EQ",100)]:
        q=p.copy();q["metric"]=metric  # Label a copy of the fixed roster with the comparison metric.
        # Calculate scaled target-minus-peer differences in the event-level excess.
        q["value"]=(q.target_event.map(event[field])-q.peer_event.map(event[field]))*scale
        pairs.append(q)  # Collect all pair scores for this metric.
    # Combine the duration and progress comparisons into one pair table.
    pairs=pd.concat(pairs,ignore_index=True)
    # Average over fixed peers within the same target episode and comparison settings.
    local=pairs.groupby(["target_event","target_zone","canonical_group","period","metric"],dropna=False).value.mean().reset_index()
    # Give local episodes equal weight within each zone, weather group, period and metric.
    groups=local.groupby(["target_zone","canonical_group","period","metric"],dropna=False).value.mean().reset_index().rename(columns={"target_zone":"zone_id"})
    summaries=[]  # Collect period summaries for the fixed target zones.
    # Evaluate valid group contrasts within each zone, period and metric.
    for (z,period,metric),g in groups.groupby(["zone_id","period","metric"]):
        x=g.value.dropna();n=len(x)  # Drop unavailable group values and count the remaining groups.
        # Identify the target, period and metric in the stage-D normal-day comparison.
        summaries.append(dict(zone_id=z,zone_name=membership.loc[z,"zone_name"],period=period,metric=metric,stage="D",
            # Record total valid groups and counts above and below the sign tolerance.
            valid_groups=n,positive_groups=int((x>EPS).sum()),negative_groups=int((x<-EPS).sum()),
            # Count ties and compute the across-group mean and fraction positive.
            tied_groups=int((abs(x)<=EPS).sum()),mean=x.mean(),positive_fraction=float((x>EPS).mean()) if n else np.nan,
            # Attach the metric-specific earlier membership from the frozen candidate roster.
            frozen_metric_member=bool(membership.loc[z,metric+"_point_clue"]),
            # Flag the descriptive pattern with at least three groups, two thirds positive and positive mean.
            descriptive_original_direction_pattern=bool(n>=3 and (x>EPS).mean()>=2/3 and x.mean()>0)))
    # Save paired-window metrics and the eligibility fields used to screen them.
    write(out,"normal_window_metrics.csv.gz",windows)
    # Save the storm excesses averaged within each event.
    write(out,"normal_excess_event_metrics.csv",event.reset_index())
    # Save the fixed-peer differences aggregated by zone and weather group.
    write(out,"normal_excess_group_scores.csv",groups)
    # Save each target's period-level normal-day comparison.
    write(out,"five_zone_period_summary.csv",pd.DataFrame(summaries))
    return groups,pd.DataFrame(summaries)  # Return the weather-group scores and their period summaries.


# Compare original and corrected reference centers for the fixed local comparisons.
def local_reference_comparison(data,out):
    """Apply the existing local reference correction to its fixed pair roster only."""
    # Read the hourly centers and the fixed local target-peer roster.
    h=read(data/"local_reference_centers.csv");p=read(data/"local_reference_fixed_peers.csv")
    metrics=[]  # Collect event metrics under each version of the center.
    # Compare the two centers on the same rows for a given event.
    for eid,w in h.groupby("event_id",sort=False):
        # Align the event to two starting and six post-rain hours.
        w=w.set_index("offset_from_end").reindex(range(-2,6))
        # Recompute recovery once with the old center and once with the corrected center.
        for label,b in [("original","old_b"),("corrected","corrected_b")]:
            # Subtract the chosen center from each raw gap before calculating six-hour metrics.
            met=stats(w.raw_gap.to_numpy()-w[b].to_numpy())
            # Store the event identifier and center version with its recovery measures.
            metrics.append(dict(event_id=eid,version=label,**met))
    # Assemble event results and prepare a collection for the pair scores.
    e=pd.DataFrame(metrics);rows=[]
    # Evaluate the local peer contrasts separately for each center version.
    for version in ["original","corrected"]:
        # Index the version-specific recovery measures by event identifier.
        s=e[e.version.eq(version)].set_index("event_id")
        # Use target minus peer for H80 and peer minus target for closure in percentage points.
        for metric,field,direction,scale in [("S80","H80",1,1),("Sprogress","Qlate",-1,100)]:
            # Restrict the fixed pair list to events present under this reference version.
            q=p[p.target_event.isin(s.index)&p.peer_event.isin(s.index)].copy()
            # Calculate the signed and scaled pair contrast for the chosen metric.
            q["value"]=direction*(q.target_event.map(s[field])-q.peer_event.map(s[field]))*scale
            # Attach the center version and metric before collecting these pair rows.
            q["version"]=version;q["metric"]=metric;rows.append(q)
    # Combine the pair scores across metrics and center versions.
    pairs=pd.concat(rows,ignore_index=True)
    # Average fixed-pair contrasts within each target zone, weather group, metric and version.
    groups=pairs.groupby(["target_zone","canonical_group","metric","version"]).value.mean().reset_index()
    # Save the local event metrics for both reference centers.
    write(out,"local_reference_event_metrics.csv",e)
    # Save the corresponding local group comparisons.
    write(out,"local_reference_group_scores.csv",groups)


# Run all frozen-input analyses and assemble the report's headline counts.
def run(data_dir, output_dir):
    """Run every final analysis from the supplied frozen input directory."""
    # Resolve the input directory and place numerical results under the requested output root.
    data=Path(data_dir);out=Path(output_dir)/"results"
    # Calculate local event recovery metrics, hourly trajectories and the temporal split.
    events,h,split=recovery(data,out)
    # Build matched peers, zone-group contrasts and later recurrence classifications.
    peers,groups,status=recurrence(data,out,events)
    # Calculate the matched taxi and ride-hailing service comparison.
    service,ss=service_comparison(data,out)
    # Evaluate the fixed normal-day pairs and their weather-group excess scores.
    ng,np_=normal_excess(data,out)
    # Calculate the local sensitivity comparison under the corrected reference center.
    local_reference_comparison(data,out)
    # Restrict rebound counts to the core weather-verified event sample.
    core=events[events.core_weather_verified]
    # Select positive raw starting deficits with an observed half-gap rebound within six hours.
    rebound=core[core.raw_D0.gt(0)&core.raw_T50_6_hour.notna()]
    # Count those rebounds followed by an observed reversal within the window.
    reversals=int(rebound.raw_T50_6_relapse.eq(True).sum())
    # Start the summary with all candidate events and the verified liquid-weather count.
    summary={"candidate_events":len(events),"verified_liquid_events":len(core),
        # Record the observed half-gap rebound count and the subset with a later reversal.
        "observed_half_gap_rebounds":len(rebound),"subsequent_reversals":reversals,
        # Calculate reversals per observed rebound and count weather groups used for the split.
        "reversal_fraction":reversals/len(rebound),"weather_groups":len(split),
        # Record how many weather groups belong to the exploration and later periods.
        "exploration_groups":int(split.period.eq("exploration").sum()),"later_groups":int(split.period.eq("later").sum()),
        # List zones repeating the original rule separately for duration and late closure.
        "recurring_zones":{m:status.loc[status.metric.eq(m)&status.replication_status.eq("REPEATS_ORIGINAL_POINT_RULE"),"zone_name"].tolist() for m in ["S80","Sprogress"]}}
    # Write the analysis counts and recurring-zone lists to the summary JSON.
    (out/"analysis_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    return summary  # Return the same summary to the command-line driver.
