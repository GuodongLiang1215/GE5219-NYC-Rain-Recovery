"""Recovery arithmetic for hourly raw and centered activity gaps."""
import numpy as np  # Use array operations to locate observed crossings and check finite hourly gaps.
import pandas as pd  # Load the pandas dependency alongside the numerical helpers in this module.

def complete(a, n):  # Test whether every hour required for a window is available.
    """Whether the window has exactly n finite hourly observations."""
    # Convert the supplied observations to a floating-point array for the validity check.
    a = np.asarray(a, float)
    # Require both the expected number of hours and a finite value for each one.
    return len(a) == n and np.isfinite(a).all()

# Trace the first crossing of a fraction of D0 and any subsequent reversal.
def hit_metrics(g, d0, rain, factor, H):
    """Return first observed threshold crossing and any later reversal at horizon H."""
    # Start with unknown crossing time and a missing-start status.
    r = dict(hour=np.nan, status="D0_MISSING", exact_first=False,
             # Leave reversal occurrence and timing unresolved until follow-up has been examined.
             earlier_missing=False, relapse=np.nan, relapse_hour=np.nan,
             # Reserve fields for the rainfall observed when a reversal occurs.
             relapse_rain_mm=np.nan, relapse_rain_state="NOT_APPLICABLE",
             # Reserve separate rainfall fields for the first observed crossing.
             hit_rain_mm=np.nan, hit_rain_state="NOT_APPLICABLE")
    # Check whether the initial deficit is available before calculating its threshold.
    if not np.isfinite(d0):
        return r  # Return the missing-start record when the threshold cannot be calculated.
    if d0 <= 0:  # Handle events whose starting activity has no positive deficit to close.
        # Record why a recovery threshold does not apply to this event.
        r["status"] = "NOT_APPLICABLE_D0_NONPOSITIVE"
        return r  # Return the nonpositive-start status without classifying a crossing.
    # Locate finite gaps at or below the specified fraction of the starting deficit.
    hits = np.flatnonzero(np.isfinite(g) & (g <= factor*d0))
    if not len(hits):  # Handle a window with no observed crossing.
        # Use completeness to distinguish a missed threshold from inconclusive follow-up.
        r["status"] = "NOT_REACHED_COMPLETE_WINDOW" if complete(g,H) else "UNRESOLVED_INCOMPLETE_WINDOW"
        return r  # Return the appropriate no-crossing status.
    k = int(hits[0])  # Select the earliest observed crossing in the ordered post-rain series.
    before = not np.isfinite(g[:k]).all()  # Check for a gap in observation before that crossing.
    # Store the one-based crossing hour and whether earlier missing values limit its timing.
    r.update(hour=k+1, status="FIRST_OBSERVED_EARLIER_MISSING" if before else "FIRST_OBSERVED_EXACT_HOURLY",
             # Retain both timing flags so downstream tables can filter for exact crossings.
             earlier_missing=before, exact_first=not before,
             # Attach the concurrent rainfall amount and wetness class to the crossing.
             hit_rain_mm=rain[k], hit_rain_state=rainstate(rain[k]))
    # Search only subsequent observed hours for gaps back above the same threshold.
    later = np.flatnonzero(np.isfinite(g[k+1:]) & (g[k+1:] > factor*d0))
    if len(later):  # Handle a measured reversal after the first crossing.
        # Translate the first reversal's position back to the full post-rain window.
        j = k+1+int(later[0])
        # Store the reversal flag, its one-based hour and the rainfall in that hour.
        r.update(relapse=True, relapse_hour=j+1, relapse_rain_mm=rain[j],
                 # Classify the reversal-hour rainfall with the same rule as the crossing hour.
                 relapse_rain_state=rainstate(rain[j]))
    # Check whether every remaining hour was observed when no reversal was found.
    elif np.isfinite(g[k+1:]).all():
        # Mark a fully observed remainder as having no observed reversal.
        r.update(relapse=False, relapse_rain_state="NO_OBSERVED_RELAPSE")
    else:  # Handle missing follow-up when the observed hours show no reversal.
        # Flag the reversal as unresolved in the incomplete remainder.
        r["relapse_rain_state"] = "RELAPSE_UNKNOWN_LATER_MISSING"
    return r  # Return all event-level crossing, reversal and rainfall fields.

def rainstate(v):  # Assign a wetness category to the rainfall at an observed crossing or reversal.
    """Classify one hourly rainfall value without treating missing rain as dry."""
    # Keep missing values unknown and separate rainfall above 0.100001 mm from dry or trace amounts.
    return "UNKNOWN" if not np.isfinite(v) else "RAIN" if v > .100001 else "DRY_OR_TRACE"

def full(v):  # Calculate a mean for a complete vector of hourly gaps.
    """Return the mean only for a nonempty, complete finite vector."""
    # Average only a nonempty vector of finite gaps and otherwise return a missing value.
    a=np.asarray(v,float);return float(a.mean()) if len(a) and np.isfinite(a).all() else np.nan

def serial(a):  # Format hour lists for storage in a single CSV field.
    """Join a sequence of hour labels with semicolons for storage in one output-table cell."""
    return ";".join(map(str,a))  # Preserve the hour order with a semicolon between successive labels.

def stats(v):  # Derive the six-hour measures used in the paired reference comparisons.
    """Measure two starting hours and six post-rain hours from an eight-value vector."""
    # Require two starting observations and six post-rain observations in hourly order.
    v=np.asarray(v,float);assert len(v)==8
    # Average the starting pair, isolate follow-up and check for a positive starting deficit.
    d=full(v[:2]);p=v[2:];pos=np.isfinite(d) and d>0
    # Find observed follow-up hours with at least 80 percent closure of a positive deficit.
    hit=np.flatnonzero(np.isfinite(p)&(p<=.2*d)) if pos else []
    # Convert the earliest qualifying position to a one-based recovery hour.
    first=int(hit[0])+1 if len(hit) else np.nan
    # List all observed hours after that crossing with renewed shortfall.
    relapse=[i+1 for i in range(int(first),6) if np.isfinite(p[i]) and p[i]>.2*d] if len(hit) else []
    # Report the initial deficit and count shortfall hours only over a complete applicable window.
    return dict(D0=d,H80=float((p>.2*d).sum()) if pos and np.isfinite(p).all() else np.nan,
        # Calculate closure in hours 4 to 6 and record total observed follow-up.
        Qlate=d-full(p[3:]) if pos else np.nan,complete6=bool(np.isfinite(p).all()),valid_post=int(np.isfinite(p).sum()),
        # Mark the first crossing exact only when all preceding hours through the crossing are observed.
        first_hit=first,first_confirmed=bool(len(hit) and np.isfinite(p[:int(first)]).all()),
        # Distinguish a missing start from a nonpositive starting deficit.
        first_hit_status="START_MISSING" if not np.isfinite(d) else "NONPOSITIVE_START_NOT_APPLICABLE" if not pos else
            # Label a crossing with uninterrupted observation from the first post-rain hour.
            "CONFIRMED_FIRST_OBSERVED" if len(hit) and np.isfinite(p[:int(first)]).all() else
            # Preserve an observed crossing even when intervening gaps obscure its first possible time.
            "HIT_OBSERVED_WITH_PRIOR_MISSING" if len(hit) else
            # Distinguish complete non-crossing from an incomplete window with no detected crossing.
            "COMPLETE_NOT_REACHED" if np.isfinite(p).all() else "INCOMPLETE_NO_OBSERVED_HIT",
        # List observed one-based hours above the shortfall threshold for an applicable event.
        insufficient_hours=serial(np.flatnonzero(np.isfinite(p)&(p>.2*d))+1) if pos else "",
        # Export reversal hours and count observed hours strictly after the first crossing.
        relapse_hours=serial(relapse),after_hit_valid=int(np.isfinite(p[int(first):]).sum()) if len(hit) else np.nan)
