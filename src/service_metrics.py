"""Compare cumulative recovery and late gap closure at H6 and H24."""
import numpy as np  # Use finite-value masks and array indices to count observed service recovery hours.
import pandas as pd  # Use pandas' nullable value when reversal occurrence has no preceding crossing.

def full(v,operation="mean"):  # Summarize a service window only when its observations are complete.
    """Return a mean or sum only when every supplied hourly value is finite."""
    a=np.asarray(v,float)  # Convert the supplied hourly service values to floating point.
    # Use a sum when requested or otherwise a mean, leaving incomplete and empty windows missing.
    return float(a.sum() if operation=="sum" else a.mean()) if len(a) and np.isfinite(a).all() else np.nan

def hours(a):  # Encode recovery-hour lists for the service comparison tables.
    """Serialize observed one-based hour numbers for output tables."""
    # Join integer hour labels with semicolons in their existing order.
    return ";".join(str(int(x)) for x in a)

def process(g,d0,Horizon):  # Calculate the recovery record for one service at its selected horizon.
    """Measure a ratio-gap vector against its own D0 over H6 or H24.

    Return a dictionary of cumulative H80, late-window closure, observed
    threshold crossings and reversals, with explicit missing-hour states.
    """
    # Take the first six or 24 gaps and mark the observed hours.
    p=np.asarray(g[:Horizon],float);valid=np.isfinite(p)
    # Count consecutive valid hours from the start up to the first missing observation.
    prefix=next((i for i,v in enumerate(valid) if not v),Horizon)
    # Require a finite positive starting deficit to interpret proportional closure.
    applicable=np.isfinite(d0) and d0>0
    # Locate observed gaps at or below one fifth of the service's own D0.
    hit=np.flatnonzero(valid&(p<=.2*d0)) if applicable else np.array([],int)
    # Express the earliest observed crossing as a one-based hour.
    first=int(hit[0])+1 if len(hit) else np.nan
    # Confirm its timing only when every hour through the crossing is observed.
    confirmed=bool(len(hit) and valid[:int(first)].all())
    # Isolate follow-up strictly after the first crossing.
    after=p[int(first):] if len(hit) else np.array([])
    # Find observed recrossings above threshold and convert their positions back to recovery hours.
    relapse=np.flatnonzero(np.isfinite(after)&(after>.2*d0))+int(first)+1 if len(hit) else np.array([],int)
    # Count the usable readings available to detect a subsequent reversal.
    obs_after=int(np.isfinite(after).sum())
    # Classify a missing or nonpositive baseline deficit before assessing crossings.
    state=("D0_MISSING" if not np.isfinite(d0) else "D0_NONPOSITIVE" if d0<=0 else
           # Separate exact crossing times from observed crossings preceded by missing hours.
           "CONFIRMED_FIRST_HOURLY_HIT" if confirmed else "FIRST_OBSERVED_HIT_PRIOR_MISSING" if len(hit) else
           # Distinguish complete non-crossing from an unresolved incomplete window.
           f"NOT_REACHED_COMPLETE{Horizon}" if valid.all() else "NOT_OBSERVED_INCOMPLETE_WINDOW")
    # Export the start, applicable threshold and number of observed recovery hours.
    return dict(D0=d0,threshold80=.2*d0 if applicable else np.nan,valid_hours=int(valid.sum()),
        # Record which hours are missing, whether the window is complete and its uninterrupted prefix length.
        missing_hours=hours(np.flatnonzero(~valid)+1),complete=bool(valid.all()),continuous_valid_prefix=prefix,
        # Give the last hour of continuous observation when that sequence has at least one hour.
        continuous_observation_stop_hour=prefix if prefix else np.nan,
        # Count all shortfall hours only for an applicable and complete window.
        H80=float((p>.2*d0).sum()) if applicable and valid.all() else np.nan,
        # Count observed exceedances even when the full-window duration is unavailable.
        observed_insufficient_hours=int((p[valid]>.2*d0).sum()) if applicable else np.nan,
        # Compute total gap, mean of the final three hours and applicable late gap closure.
        A=full(p,"sum"),late=full(p[-3:]),Qlate=d0-full(p[-3:]) if applicable else np.nan,
        # Store the first observed crossing hour, its confirmation flag and its status.
        first_observed_hit=first,first_is_confirmed=confirmed,T80_status=state,
        # List each observed hour that still exceeds the service-specific shortfall threshold.
        insufficient_hours=hours(np.flatnonzero(valid&(p>.2*d0))+1) if applicable else "",
        # Export the crossing-hour and later reversal-hour lists.
        hit_hours=hours(hit+1),relapse_hours=hours(relapse),
        # Leave the reversal indicator undefined when no initial crossing was observed.
        relapse_observed=bool(len(relapse)) if len(hit) else pd.NA,
        # Count observed and missing follow-up hours after an observed crossing.
        after_hit_valid_hours=obs_after,after_hit_missing_hours=int((~np.isfinite(after)).sum()) if len(hit) else np.nan,
        # Distinguish a window without any crossing from a crossing with no observed follow-up.
        after_hit_observation_status="NOT_REACHED" if not len(hit) else "NO_VALID_FOLLOWUP" if not obs_after else
            # Identify an observed reversal or a complete remainder with no reversal.
            "RELAPSE_OBSERVED" if len(relapse) else "NO_RELAPSE_IN_COMPLETE_REMAINING_WINDOW" if np.isfinite(after).all() else
            # Label a partially observed remainder in which no reversal was seen.
            "NO_OBSERVED_RELAPSE_FOLLOWUP_PARTIAL")
