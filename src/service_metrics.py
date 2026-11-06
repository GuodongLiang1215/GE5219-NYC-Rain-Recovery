"""Compare cumulative recovery and late gap closure at H6 and H24."""
import numpy as np
import pandas as pd

def full(v,operation="mean"):
    """Return a mean or sum only when every supplied hourly value is finite."""
    a=np.asarray(v,float)
    return float(a.sum() if operation=="sum" else a.mean()) if len(a) and np.isfinite(a).all() else np.nan

def hours(a):
    """Serialize observed one-based hour numbers for output tables."""
    return ";".join(str(int(x)) for x in a)

def process(g,d0,Horizon):
    """Measure a ratio-gap vector against its own D0 over H6 or H24.

    Return a dictionary of cumulative H80, late-window closure, observed
    threshold crossings and reversals, with explicit missing-hour states.
    """
    p=np.asarray(g[:Horizon],float);valid=np.isfinite(p)
    prefix=next((i for i,v in enumerate(valid) if not v),Horizon)
    applicable=np.isfinite(d0) and d0>0
    hit=np.flatnonzero(valid&(p<=.2*d0)) if applicable else np.array([],int)
    first=int(hit[0])+1 if len(hit) else np.nan
    confirmed=bool(len(hit) and valid[:int(first)].all())
    after=p[int(first):] if len(hit) else np.array([])
    relapse=np.flatnonzero(np.isfinite(after)&(after>.2*d0))+int(first)+1 if len(hit) else np.array([],int)
    obs_after=int(np.isfinite(after).sum())
    state=("D0_MISSING" if not np.isfinite(d0) else "D0_NONPOSITIVE" if d0<=0 else
           "CONFIRMED_FIRST_HOURLY_HIT" if confirmed else "FIRST_OBSERVED_HIT_PRIOR_MISSING" if len(hit) else
           f"NOT_REACHED_COMPLETE{Horizon}" if valid.all() else "NOT_OBSERVED_INCOMPLETE_WINDOW")
    return dict(D0=d0,threshold80=.2*d0 if applicable else np.nan,valid_hours=int(valid.sum()),
        missing_hours=hours(np.flatnonzero(~valid)+1),complete=bool(valid.all()),continuous_valid_prefix=prefix,
        continuous_observation_stop_hour=prefix if prefix else np.nan,
        H80=float((p>.2*d0).sum()) if applicable and valid.all() else np.nan,
        observed_insufficient_hours=int((p[valid]>.2*d0).sum()) if applicable else np.nan,
        A=full(p,"sum"),late=full(p[-3:]),Qlate=d0-full(p[-3:]) if applicable else np.nan,
        first_observed_hit=first,first_is_confirmed=confirmed,T80_status=state,
        insufficient_hours=hours(np.flatnonzero(valid&(p>.2*d0))+1) if applicable else "",
        hit_hours=hours(hit+1),relapse_hours=hours(relapse),
        relapse_observed=bool(len(relapse)) if len(hit) else pd.NA,
        after_hit_valid_hours=obs_after,after_hit_missing_hours=int((~np.isfinite(after)).sum()) if len(hit) else np.nan,
        after_hit_observation_status="NOT_REACHED" if not len(hit) else "NO_VALID_FOLLOWUP" if not obs_after else
            "RELAPSE_OBSERVED" if len(relapse) else "NO_RELAPSE_IN_COMPLETE_REMAINING_WINDOW" if np.isfinite(after).all() else
            "NO_OBSERVED_RELAPSE_FOLLOWUP_PARTIAL")
