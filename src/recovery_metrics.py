"""Recovery arithmetic for hourly raw and centered activity gaps."""
import numpy as np
import pandas as pd

def complete(a, n):
    """Whether the window has exactly n finite hourly observations."""
    a = np.asarray(a, float)
    return len(a) == n and np.isfinite(a).all()

def hit_metrics(g, d0, rain, factor, H):
    """Return first observed threshold crossing and any later reversal at horizon H."""
    r = dict(hour=np.nan, status="D0_MISSING", exact_first=False,
             earlier_missing=False, relapse=np.nan, relapse_hour=np.nan,
             relapse_rain_mm=np.nan, relapse_rain_state="NOT_APPLICABLE",
             hit_rain_mm=np.nan, hit_rain_state="NOT_APPLICABLE")
    if not np.isfinite(d0):
        return r
    if d0 <= 0:
        r["status"] = "NOT_APPLICABLE_D0_NONPOSITIVE"
        return r
    hits = np.flatnonzero(np.isfinite(g) & (g <= factor*d0))
    if not len(hits):
        r["status"] = "NOT_REACHED_COMPLETE_WINDOW" if complete(g,H) else "UNRESOLVED_INCOMPLETE_WINDOW"
        return r
    k = int(hits[0])
    before = not np.isfinite(g[:k]).all()
    r.update(hour=k+1, status="FIRST_OBSERVED_EARLIER_MISSING" if before else "FIRST_OBSERVED_EXACT_HOURLY",
             earlier_missing=before, exact_first=not before,
             hit_rain_mm=rain[k], hit_rain_state=rainstate(rain[k]))
    later = np.flatnonzero(np.isfinite(g[k+1:]) & (g[k+1:] > factor*d0))
    if len(later):
        j = k+1+int(later[0])
        r.update(relapse=True, relapse_hour=j+1, relapse_rain_mm=rain[j],
                 relapse_rain_state=rainstate(rain[j]))
    elif np.isfinite(g[k+1:]).all():
        r.update(relapse=False, relapse_rain_state="NO_OBSERVED_RELAPSE")
    else:
        r["relapse_rain_state"] = "RELAPSE_UNKNOWN_LATER_MISSING"
    return r

def rainstate(v):
    """Classify one hourly rainfall value without treating missing rain as dry."""
    return "UNKNOWN" if not np.isfinite(v) else "RAIN" if v > .100001 else "DRY_OR_TRACE"

def full(v):
    """Return the mean only for a nonempty, complete finite vector."""
    a=np.asarray(v,float);return float(a.mean()) if len(a) and np.isfinite(a).all() else np.nan

def serial(a):return ";".join(map(str,a))

def stats(v):
    """Measure two starting hours and six post-rain hours from an eight-value vector."""
    v=np.asarray(v,float);assert len(v)==8
    d=full(v[:2]);p=v[2:];pos=np.isfinite(d) and d>0
    hit=np.flatnonzero(np.isfinite(p)&(p<=.2*d)) if pos else []
    first=int(hit[0])+1 if len(hit) else np.nan
    relapse=[i+1 for i in range(int(first),6) if np.isfinite(p[i]) and p[i]>.2*d] if len(hit) else []
    return dict(D0=d,H80=float((p>.2*d).sum()) if pos and np.isfinite(p).all() else np.nan,
        Qlate=d-full(p[3:]) if pos else np.nan,complete6=bool(np.isfinite(p).all()),valid_post=int(np.isfinite(p).sum()),
        first_hit=first,first_confirmed=bool(len(hit) and np.isfinite(p[:int(first)]).all()),
        first_hit_status="START_MISSING" if not np.isfinite(d) else "NONPOSITIVE_START_NOT_APPLICABLE" if not pos else
            "CONFIRMED_FIRST_OBSERVED" if len(hit) and np.isfinite(p[:int(first)]).all() else
            "HIT_OBSERVED_WITH_PRIOR_MISSING" if len(hit) else
            "COMPLETE_NOT_REACHED" if np.isfinite(p).all() else "INCOMPLETE_NO_OBSERVED_HIT",
        insufficient_hours=serial(np.flatnonzero(np.isfinite(p)&(p>.2*d))+1) if pos else "",
        relapse_hours=serial(relapse),after_hit_valid=int(np.isfinite(p[int(first):]).sum()) if len(hit) else np.nan)
