"""Reproduce the report's six figures and analytical-sample table."""
from pathlib import Path  # Resolve the input tables and the three report figure formats on disk.
import json  # Read geometry and period metadata and export the values underlying the figures.
import numpy as np  # Manipulate map vertices, hourly curves and arrays of plotted values.
import pandas as pd  # Read the calculated results and assemble each plot's supporting table.
import matplotlib  # Configure the plotting backend and construct color mappings.
# Select a noninteractive backend so an extracted package can render without a display.
matplotlib.use('Agg')
# Create the figures, panels and annotations after the backend has been selected.
import matplotlib.pyplot as plt
# Draw polygon areas, legend swatches and the locator map's extent rectangle.
from matplotlib.patches import PathPatch, Patch, Rectangle
# Construct compound paths for zone polygons and interior holes.
from matplotlib.path import Path as MplPath
from matplotlib.lines import Line2D  # Make line and marker samples for legends shared across panels.
# Normalize pickup changes and create the map's diverging color scale.
from matplotlib.colors import Normalize, LinearSegmentedColormap
from shapely.geometry import shape  # Turn the stored zone geometries into Shapely geometry objects.
# Ensure polygon exteriors and holes have the winding required for filled paths.
from shapely.geometry.polygon import orient
from shapely.ops import unary_union  # Dissolve adjacent zone polygons into borough outlines.

FT_TO_M = 0.3048006096012192  # Convert US survey feet in EPSG:2263 to metres for scale-bar lengths.
# Set the projected citywide x and y bounds used by the main and locator maps.
FULL = (907000, 1075000, 113000, 281000)
FOCUS = (968000, 1055000, 183000, 279000)  # Set the projected inset bounds for the two recurrence maps.

BLUE = '#0072B2'  # Use blue for the plotted recovery series, ride-hailing gains and group means.
TEAL = '#277E87'  # Define the report palette's teal accent.
PURPLE = '#9B3B75'  # Use purple for repeated shortfall classes and the Mount Hope case series.
ORANGE = '#B96821'  # Use orange for pickup declines, taxi bars and recovery thresholds.
GRAY = '#6B747B'  # Use neutral gray for reference lines, supporting labels and fixed-peer curves.
LIGHT = '#F0F1F2'  # Use a pale fill for NYC zones outside the fixed recovery cohort on the study map.
EDGE = '#B8BDC1'  # Give zone edges and legend swatches a common border color.
TEXT = '#20262B'  # Use a dark tone for text, map labels and scale-bar marks.
STYLE = {  # Collect the rendering settings applied across all six figures.
    # Use Arial and set the default report text size in points.
    'font.family': 'Arial', 'font.size': 8.5,
    # Use compact bold titles aligned to the left edge of each panel.
    'axes.titlesize': 9.5, 'axes.titleweight':'bold', 'axes.titlelocation':'left',
    # Set type sizes independently for axis labels and for x and y tick labels.
    'axes.labelsize': 8.5, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    # Set legend type size and the color and width of axis borders.
    'legend.fontsize': 8, 'axes.edgecolor': '#68737B', 'axes.linewidth': .6,
    # Match labels, tick text and general text to the report's main text color.
    'text.color':TEXT, 'axes.labelcolor':TEXT, 'xtick.color':TEXT, 'ytick.color':TEXT,
    # Use white backgrounds for both figure canvases and plot panels.
    'figure.facecolor':'white', 'axes.facecolor':'white',
    # Embed TrueType fonts in PDF and PostScript and preserve editable SVG text.
    'pdf.fonttype':42, 'ps.fonttype':42, 'svg.fonttype':'none',
    # Use a white export background and a default resolution of 300 dpi.
    'savefig.facecolor':'white', 'savefig.dpi':300,
    # Keep fine gridlines visible behind the plotted observations.
    'grid.color':'#E1E5E8','grid.linewidth':.5,'axes.axisbelow':True,
    # Use thin map hatching and rounded line ends.
    'hatch.linewidth':.45, 'lines.solid_capstyle':'round',
}  # Finish the settings applied inside the report's plotting context.

def _number_word(number):  # Format a calculated small count for use in figure text.
    """Format a small count in a sentence without fixing its value."""
    # Provide the spellings used for counts from zero through ten.
    words = ('zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten')
    # Use a word within the lookup range and digits for other counts.
    return words[number] if 0 <= number < len(words) else str(number)


# Isolate the weather-group scores used to label and summarize the recurrence maps.
def _main_group_scores(frame):
    """Select centered six-hour comparisons used by the recurrence maps."""
    selected = frame.copy()  # Start from a copy so the source weather-group table is preserved.
    # Specify centered gaps and the six-hour horizon used in the maps.
    for column, value in [('mode', 'centered'), ('H', 6),
                          ('subset', 'all_selected'),  # Use the results for all selected comparisons.
                          # Require peers with the same known subsequent-rain class.
                          ('comparison_pool', 'SAME_KNOWN_RAIN_CLASS'),
                          # Restrict comparisons to the verified main-study weather groups.
                          ('inference_scope', 'MAIN_VERIFIED_CLOSED')]:
        if column in selected:  # Test whether this filter's field is present in the supplied table.
            # Keep rows matching the corresponding map setting.
            selected = selected[selected[column].eq(value)]
    return selected  # Return the filtered group scores used by the figure routines.


def _case_curves(hourly, peers):  # Build the hourly paths for the Belmont and Mount Hope case figures.
    """Calculate plotted rebound and cumulative-shortfall paths from hourly gaps."""
    def event_rows(event_id):  # Retrieve the aligned starting and post-rain rows for one plotted case.
        """Return two starting bins and six follow-up bins for a case, rejecting duplicate offsets."""
        # Order a single event's rows by their offset from local rain ending.
        selected = hourly[hourly.event_id.eq(event_id)].set_index('offset_from_end').sort_index()
        # Check whether any recovery offset has more than one observation.
        if selected.index.duplicated().any():
            # Stop if duplicate offsets would make the plotted trajectory ambiguous.
            raise ValueError(f'Duplicate hourly rows for {event_id}')
        # Extract the two starting hours followed by all six post-rain hours.
        return selected.loc[list(range(-2, 6))].copy()

    belmont = event_rows('N020E0010')  # Load the fixed Belmont episode shown in the rebound figure.
    # Average its two starting raw deficits to obtain the initial shortfall.
    initial = belmont.loc[[-2, -1], 'gap'].mean()
    threshold = 0.5 * initial  # Use half that initial deficit as the rebound threshold.
    # Select the six observed follow-up gaps in chronological order.
    post = belmont.loc[list(range(6)), 'gap']
    # Find the first post-rain offset at or below the half-gap threshold.
    first = int(post[post.le(threshold)].index[0])
    later = post[post.index > first]  # Restrict the reversal search to hours after that crossing.
    # Find the first later offset where the raw deficit again exceeds the threshold.
    recross = int(later[later.gt(threshold)].index[0])
    # Restore offsets as a column for plotting and exporting the hourly series.
    records = belmont.reset_index()
    # Convert raw deficit ratios to percentages for the vertical axis.
    records['raw_deficit_percent'] = records.gap * 100
    # Convert UTC bins to New York time for the chart's horizontal labels.
    records['hour_NYC'] = pd.to_datetime(records.hour_utc, utc=True).dt.tz_convert('America/New_York')
    # Store the Belmont hourly series under the report case's plotting key.
    recovery = {'slides': {'6': {'plot_records': records,
        # Save the half-gap threshold in the same percentage units as the plot.
        'recommended_case': {'half_gap_threshold_percent': threshold * 100,
                             # Shift both offsets by two to align the markers after the two starting observations.
                             'first_index': first + 2, 'recross_index': recross + 2,
                             # Count follow-up hours with positive measured rainfall for the case note.
                             'post_rain_hours': int(belmont.loc[list(range(6)), 'rain_mean_mm'].gt(0).sum())}}}}

    # Copy the selected peer roster before applying the case-figure restrictions.
    selected_peers = peers.copy()
    # Select the main pool of peers sharing the subsequent-rain class.
    for column, value in [('comparison_pool', 'SAME_KNOWN_RAIN_CLASS'),
                          # Restrict the case to comparisons in the verified main-study scope.
                          ('inference_scope', 'MAIN_VERIFIED_CLOSED')]:
        # Check whether the peer table contains the current selection field.
        if column in selected_peers:
            # Retain only peer rows matching the case-figure setting.
            selected_peers = selected_peers[selected_peers[column].eq(value)]
    cases = []  # Collect the two Mount Hope cases for the paired panels.
    # Build a curve for each of the two designated Mount Hope event identifiers.
    for target in ['Z169E0385', 'L3Z169_20250504T02_20250504T04']:
        # Get unique selected peers for this target in their existing order.
        peer_ids = selected_peers.loc[selected_peers.target_event.eq(target), 'peer_event'].drop_duplicates().tolist()
        if not peer_ids:  # Check that a fixed peer comparison is available for the target.
            # Stop if the figure would display a peer mean without any peers.
            raise ValueError(f'No selected peers for {target}')
        paths = []  # Collect the cumulative shortfall curves with the target first.
        # Apply the same six-hour calculation to the target and each selected peer.
        for event_id in [target, *peer_ids]:
            rows = event_rows(event_id)  # Load this episode's eight aligned hourly observations.
            # Average the two centered gaps at the end of rain for this episode's own start.
            d0 = rows.loc[[-2, -1], 'gap_centered'].mean()
            # Check that its centered initial deficit is positive and finite.
            if not np.isfinite(d0) or d0 <= 0:
                # Reject a relative closure threshold when that starting gap is unavailable or nonpositive.
                raise ValueError(f'Nonpositive starting gap for {event_id}')
            # Read the six centered follow-up gaps in post-rain order.
            values = rows.loc[list(range(6)), 'gap_centered'].to_numpy()
            # Check for missing values before making a cumulative shortfall path.
            if not np.isfinite(values).all():
                # Stop when an incomplete curve would change the duration comparison.
                raise ValueError(f'Incomplete recovery curve for {event_id}')
            # Accumulate hours exceeding one fifth of the episode's own starting deficit.
            paths.append(np.cumsum(values > 0.2 * d0))
        # Retrieve the target again to assemble the values shown in its panels.
        rows = event_rows(target)
        # Calculate the target's centered starting deficit for its threshold line.
        d0 = rows.loc[[-2, -1], 'gap_centered'].mean()
        # Extract the target's post-rain gaps for the upper panel.
        post = rows.loc[list(range(6)), 'gap_centered'].to_numpy()
        # Give the fixed peers equal weight at each point on their cumulative trajectories.
        peer_mean = np.mean(paths[1:], axis=0)
        # Express the first post-rain observation in New York time for the case date.
        local = pd.Timestamp(rows.loc[0, 'hour_utc']).tz_convert('America/New_York')
        # Number the plotted follow-up hours from one.
        plotted = pd.DataFrame({'h': np.arange(1, len(post) + 1),
            # Retain the target's actual centered-gap trajectory as well as the derived counts.
            'target_centered_gap': post,
            # Flag each hour that contributes to the target's cumulative shortfall.
            'target_shortfall_this_hour': post > 0.2 * d0,
            'target_cumulative_H80': paths[0],  # Store the target's running count of shortfall hours.
            # Store the equal-weight peer mean at each recovery hour.
            'equal_peer_mean_cumulative_H80': peer_mean,
            # Calculate the target's excess cumulative duration relative to its peers.
            'target_minus_peer_cumulative_H80': paths[0] - peer_mean})
        # Record the target identifier and local date for the panel label.
        cases.append({'event_id': target, 'local_date': local.strftime('%Y-%m-%d'),
                      # Mark the case for display and retain its distinct peer count.
                      'display_priority': 'main', 'peer_count': len(peer_ids),
                      # Store the target's 80 percent closure threshold in centered-gap units.
                      'target': {'threshold_centered_80': 0.2 * d0},
                      # Attach all ordered plotted records to the case summary.
                      'plot_cumulative_records': plotted.to_dict('records')})
    # Return the Belmont rebound data and the two Mount Hope case summaries.
    return recovery, {'cases': cases}


# Build the report's analytical sample table from its actual study inputs and scores.
def _sample_table(snapshot, events, scores, service_groups, cohort, periods):
    """Return Table 1 and its numeric support, calculated from the sample tables."""
    def year_span(start, end):  # Format a pair of sample dates as a year or an inclusive range of years.
        """Format the years bounding a sample period, collapsing a single-year span to one label."""
        # Extract the calendar year from each boundary of the sample period.
        first, last = pd.Timestamp(start).year, pd.Timestamp(end).year
        # Show the year once when both boundaries fall in the same year.
        return str(first) if first == last else f'{first}–{last}'

    # Read the full source panel's first and last timestamps.
    first, last = pd.Timestamp(periods['source_start']), pd.Timestamp(periods['source_end'])
    # Display its first year and the month and year of its final observation period.
    recovery_period = f"{first.year}–{last.strftime('%B %Y')}"
    # Format the period covered by the normal-centered comparisons.
    centered_period = year_span(periods['centered_start'], periods['centered_end'])
    # Format the dates covered by the paired service comparison.
    service_period = year_span(periods['service_start'], periods['service_end'])
    # Select weather-verified events with a positive raw starting deficit.
    rebounds = events[events.core_weather_verified.eq(True) & events.raw_D0.gt(0)
                      # Retain only events with an observed half-gap rebound in the six-hour window.
                      & events.raw_T50_6_hour.notna()]
    # Record cohort size and the observed rebound count as support for Table 1.
    support = {'cohort_zones': len(cohort), 'half_gap_rebounds': len(rebounds)}
    # Calculate weather-group and service-case support for each recurrence metric.
    for metric in ['S80', 'Sprogress']:
        # Count distinct weather groups with a usable primary score for this metric.
        support[f'recurrence_{metric}_groups'] = int(scores.loc[scores[metric].notna(), 'canonical_group'].nunique())
        # Select the percentage-point field name used for service closure scores.
        service_metric = 'Sprogress_pp' if metric == 'Sprogress' else metric
        # Begin with centered service comparisons.
        common = service_groups[service_groups['mode'].eq('centered')
                 # Match the service table's measure to the metric being summarized.
                 & service_groups.metric.eq(service_metric)
                 # Use the 24-hour supported sample shared across both service horizons.
                 & service_groups['sample'].eq('H24_METRIC_COMMON_H24')
                 # Require a measured group difference when counting common sample support.
                 & service_groups.difference.notna()]
        # Split the stored ID lists into unique nonempty contributing event IDs.
        ids = set(';'.join(common.event_ids.fillna('')).split(';')) - {''}
        selected = events[events.event_id.isin(ids)]  # Find the matching event metadata for those cases.
        # Verify that every service case is represented in the main event table.
        if selected.event_id.nunique() != len(ids):
            # Stop if missing event records would understate zone or group support.
            raise ValueError('Service-comparison cases are missing from the event table')
        # Store distinct matched cases and zones separately for this service metric.
        support[f'service_{metric}'] = {'cases': len(ids), 'zones': int(selected.zone_id.nunique()),
                                         # Count weather groups contributing measured common-sample contrasts.
                                         'groups': int(common.canonical_group.nunique())}
    # Select the ride-hailing records for the Pilot spatial snapshot.
    rh = snapshot[snapshot.service.eq('Ride-hailing')]
    # Count zones with a usable ride-hailing pickup change on the Pilot map.
    support['snapshot_rh_zones'] = int(rh.map_change_pct.notna().sum())
    # Format the observed local snapshot date for the table.
    snapshot_date = pd.Timestamp(snapshot.hour.iloc[0]).strftime('%d %B %Y').lstrip('0')
    # Retrieve the two metric-specific service samples for the final rows.
    s80, progress = support['service_S80'], support['service_Sprogress']
    rows = [  # Assemble the analytical samples in the order used by the report.
        ['Spatial snapshot', snapshot_date,  # Start the Pilot snapshot row with its observation date.
         # Describe the snapshot unit and include its usable ride-hailing zone count.
         f"Zone × service × hour; {support['snapshot_rh_zones']} usable ride-hailing zones"],
        # Add the source period for the recovery-reversal analysis.
        ['Recovery reversals', recovery_period,
         # Report the number of zone episodes with observed half-gap rebounds.
         f"{len(rebounds):,} zone–episode records with an observed half-gap rebound"],
        # Label the recurrence row with the centered comparison period.
        ['Relative recurrence', centered_period + ' centered results',
         # Report cohort size and distinct group support for shortfall duration.
         f"Fixed {len(cohort)}-zone cohort; {support['recurrence_S80_groups']} weather groups for S80, "
         # Complete the recurrence entry with group support for late closure.
         f"{support['recurrence_Sprogress_groups']} for Sprogress"],
        # Identify the matched six-hour and 24-hour duration comparison and its period.
        ['Matched H6/H24 services', service_period,
         # Report the common cases, zones and groups for the service duration measure.
         f"Same {s80['cases']} zone–episode cases, {s80['zones']} zones and {s80['groups']} weather groups"],
        # Identify the late-window closure comparison over the same service data period.
        ['H6/H24 late gap closure', service_period,
         # Report the closure-specific matched cases, zones and groups.
         f"Same {progress['cases']} cases, {progress['zones']} zones and {progress['groups']} weather groups"],
    ]  # Finish the ordered set of sample rows for Table 1.
    # Return the formatted three-column table and the numeric sample counts used to construct it.
    return pd.DataFrame(rows, columns=['Analysis', 'Period', 'Observation and support']), support


# Produce the report figures and export their shared tables from the analysis outputs.
def generate_report_outputs(data_dir, results_dir, output_dir):
    """Write six PNG/SVG/PDF figures and Table 1 from the computed results."""
    # Normalize the three directory arguments to filesystem paths.
    data_dir, results_dir, output_dir = map(Path, [data_dir, results_dir, output_dir])
    FIG = output_dir / 'figures'  # Set the destination for figure images and vector exports.
    BUILD = output_dir / 'tables'  # Set the destination for Table 1 and its supporting numerical files.
    # Create the figure export directory including any missing parents.
    FIG.mkdir(parents=True, exist_ok=True)
    # Create the table export directory independently of the figure directory.
    BUILD.mkdir(parents=True, exist_ok=True)
    # Load the stored taxi-zone polygons in projected EPSG:2263 coordinates.
    geometry = json.loads((data_dir / 'map_geometry_2263.json').read_text(encoding='utf8'))
    # Map each numeric TLC zone identifier to its Shapely geometry.
    GEOMS = {int(f['properties']['zone_id']): shape(f['geometry'])
             # Keep the NYC zones by excluding identifier 1, Newark Airport, from the map geometry.
             for f in geometry['features'] if int(f['properties']['zone_id']) != 1}
    # Index the zone names, boroughs and label coordinates by TLC zone ID.
    labels = pd.read_csv(data_dir / 'zone_labels.csv').set_index('zone_id')
    # Build a map lookup containing the identifying geography for each zone.
    PROPS = {int(zone): {'zone_name': row.zone_name, 'borough': row.borough,
                       # Store each label's x and y anchor in the same projected coordinates as the map.
                       'label_point_2263': [row.label_x_2263, row.label_y_2263]}
             for zone, row in labels.iterrows()}  # Populate that lookup from the zone-label rows.
    # Read the fixed recovery cohort as a set for map membership checks.
    ANALYSIS = set(pd.read_csv(data_dir / 'recovery_cohort.csv').zone_id)
    # Load the freshly computed hourly trajectories for the two case figures.
    hourly = pd.read_csv(results_dir / 'hourly_curves.csv.gz')
    # Load the selected target-peer pairs used to construct comparison curves.
    peers = pd.read_csv(results_dir / 'peer_selection.csv')
    # Read event measures for sample counts and the rebound analysis.
    events = pd.read_csv(results_dir / 'recovery_event_metrics.csv')
    # Filter the newly calculated group scores to the primary map configuration.
    scores = _main_group_scores(pd.read_csv(results_dir / 'zone_weather_group_recovery.csv'))
    # Read the service-specific group estimates for Table 1 and the H6/H24 chart.
    service_groups = pd.read_csv(results_dir / 'service_weather_group_summary.csv')
    # Read each zone's later-period classification for the recurrence maps.
    rep = pd.read_csv(results_dir / 'replication_status_long.csv')
    # Load the previously calculated Pilot snapshot observations and baselines.
    records = pd.read_csv(results_dir / 'pilot_snapshot.csv')
    # Express Pilot pickup changes as percentages of baseline for the figure.
    records['map_change_pct'] = 100 * (records.actual / records.baseline - 1)
    # Require four or more references and a baseline of at least ten pickups.
    eligible = records.reference_n.ge(4) & records.baseline.ge(10)
    # Mask unsupported Pilot changes so the legend can distinguish insufficient baselines.
    records.loc[~eligible, 'map_change_pct'] = np.nan
    # Pass the prepared Pilot rows to the spatial response plotting code.
    GIS = {'slide5': {'records': records}}
    # Calculate both the Belmont rebound example and the Mount Hope peer paths.
    REC, MH = _case_curves(hourly, peers)
    # Read source dates and sample boundaries for figure labels and Table 1.
    periods = json.loads((data_dir / 'study_periods.json').read_text(encoding='utf8'))
    # Build the study-sample descriptions and their supporting counts.
    table, support = _sample_table(records, events, scores, service_groups, ANALYSIS, periods)
    # Write Table 1 as a CSV with the three report columns.
    table.to_csv(BUILD / 'Table_1_Analytical_Samples.csv', index=False)
    # Assemble the Markdown header and column divider for Table 1.
    lines = ['| ' + ' | '.join(table.columns) + ' |', '| --- | --- | --- |']
    # Add one Markdown row per analytical sample, retaining the CSV column order.
    lines += ['| ' + ' | '.join(row) + ' |' for row in table.itertuples(index=False, name=None)]
    # Save the Markdown table with a final newline.
    (BUILD / 'Table_1_Analytical_Samples.md').write_text('\n'.join(lines) + '\n', encoding='utf8')
    # Start the machine-readable figure support with Table 1's underlying counts.
    numeric_values = {'table1': support}

    def polygon_path(poly):  # Convert a polygon, including any holes, into a single Matplotlib path.
        """Build a compound Matplotlib path from a polygon exterior and its oriented interior rings."""
        # Orient the exterior and interior rings consistently before creating a filled shape.
        poly = orient(poly, sign=1.0)
        verts, codes = [], []  # Collect the vertices and drawing commands for all rings in this polygon.
        # Process the exterior boundary first, then each interior hole.
        for ring in [poly.exterior, *poly.interiors]:
            # Read all vertices in the current ring in their winding order.
            coords = np.asarray(ring.coords)
            verts.extend(coords)  # Append the ring's coordinates to the compound vertex list.
            # Start the ring, connect its vertices and close it so interior holes render correctly.
            codes.extend([MplPath.MOVETO]+[MplPath.LINETO]*(len(coords)-2)+[MplPath.CLOSEPOLY])
        return MplPath(verts,codes)  # Return one compound path that Matplotlib can fill and outline.

    # Draw a zone's polygon parts using the specified fill, border and hatching.
    def draw_geom(ax, geom, facecolor=LIGHT, edgecolor=EDGE, linewidth=.25, zorder=1, hatch=None):
        """Add each part of a zone geometry to the map axes with the requested fill and border."""
        # Treat a single polygon and the parts of a multipolygon as an iterable of polygons.
        polygons = [geom] if geom.geom_type == 'Polygon' else list(geom.geoms)
        for poly in polygons:  # Draw each disconnected polygon part onto the same map axes.
            # Add the polygon path as a patch with the requested interior and boundary colors.
            ax.add_patch(PathPatch(polygon_path(poly),facecolor=facecolor,edgecolor=edgecolor,
                                  # Apply the line width, layer order and optional hatching to that part.
                                  linewidth=linewidth,zorder=zorder,hatch=hatch))

    def map_axis(ax, extent=FULL):  # Set the projected bounds and axes appearance for a map panel.
        """Set equal-scale projected bounds and hide coordinate ticks and the map frame."""
        # Use the supplied x and y limits with equal scales so zone shapes retain their proportions.
        ax.set_xlim(extent[:2]); ax.set_ylim(extent[2:]); ax.set_aspect('equal')
        ax.set_xticks([]); ax.set_yticks([])  # Suppress coordinate tick labels on both map axes.
        # Hide the axis frame around the mapped geometry.
        for s in ax.spines.values(): s.set_visible(False)

    # Outline the boroughs by dissolving the taxi zones belonging to each one.
    def borough_borders(ax, lw=.6):
        """Dissolve zones by borough and overlay the resulting boundary outlines on the map."""
        by = {}  # Collect a separate list of component zone geometries for each borough.
        # Group the plotted NYC polygons by the borough recorded in the zone lookup.
        for zid,geom in GEOMS.items(): by.setdefault(PROPS[zid]['borough'],[]).append(geom)
        # Dissolve each borough and draw its unfilled outline above the zone fills.
        for gs in by.values(): draw_geom(ax,unary_union(gs),facecolor='none',edgecolor='#5D6A71',linewidth=lw,zorder=3)

    # Add a scale bar in kilometres and a north arrow at the requested panel positions.
    def scale_north(ax, km=10, position=(.08,.08), north=(.93,.82), label_size=7.4):
        """Draw a kilometre scale bar in projected survey feet and a north arrow in axes coordinates."""
        # Read the current map limits and calculate the width and height in projection units.
        xmin,xmax=ax.get_xlim();ymin,ymax=ax.get_ylim(); dx=xmax-xmin;dy=ymax-ymin
        # Convert the relative scale-bar anchor to map coordinates and kilometres to survey feet.
        x=xmin+position[0]*dx;y=ymin+position[1]*dy; L=km*1000/FT_TO_M
        # Draw the horizontal scale-bar segment at its true projected length.
        ax.plot([x,x+L],[y,y],lw=1.2,color=TEXT,zorder=10)
        # Mark both ends and the halfway point of the scale bar.
        for xx in [x,x+L/2,x+L]:ax.plot([xx,xx],[y-.006*dy,y+.006*dy],lw=.8,color=TEXT,zorder=10)
        # Place a zero label under the near end of the scale.
        ax.text(x,y-.024*dy,'0',va='top',ha='center',fontsize=label_size)
        # Label the far end with the requested distance in kilometres.
        ax.text(x+L,y-.024*dy,f'{km:g} km',va='top',ha='center',fontsize=label_size)
        # Position the north arrow's tip above its base using relative panel locations.
        ax.annotate('',xy=(north[0],north[1]+.12),xytext=north,
                    # Use axes-fraction coordinates and a filled arrowhead for the north indicator.
                    xycoords='axes fraction',arrowprops={'arrowstyle':'-|>','color':TEXT,'lw':.9})
        # Place the north label above the arrowhead in the same panel coordinates.
        ax.text(north[0],north[1]+.13,'N',ha='center',va='bottom',transform=ax.transAxes,fontsize=8)

    def point(zid):  # Look up the projected annotation anchor for a zone.
        """Return the stored EPSG:2263 annotation coordinates for one taxi-zone identifier."""
        # Return the stored EPSG:2263 point for the supplied zone identifier.
        return PROPS[zid]['label_point_2263']

    # Attach a label at a chosen text position to the zone's projected anchor.
    def label_zone(ax,zid,label,xytext,fontsize=8,color=TEXT):
        """Connect a zone's projected anchor to its label at the requested map coordinates."""
        # Draw the supplied label from the zone anchor to its separate text location.
        ax.annotate(label,xy=point(zid),xytext=xytext,fontsize=fontsize,color=color,
                    # Left-align the label and allow it to extend beyond the map boundary.
                    ha='left',va='center',annotation_clip=False,
                    # Give the text a small white backing so zone boundaries do not obscure it.
                    bbox={'facecolor':'white','edgecolor':'none','pad':1.5},
                    # Draw a short, high-layer leader line from the backing to the zone anchor.
                    arrowprops={'arrowstyle':'-','color':color,'lw':.65,'shrinkA':1,'shrinkB':1},zorder=12)

    def save(fig,stem):  # Save a completed report figure under a shared filename stem.
        """Export one figure at its fixed physical size in three formats."""
        # Export the same canvas as a 300-dpi PNG and as PDF and SVG vector figures.
        for ext in ['png','pdf','svg']:fig.savefig(FIG/f'{stem}.{ext}',dpi=300)
        plt.close(fig)  # Release the finished figure before constructing the next one.

    def clean(ax):  # Apply the report's standard grid, frame and tick styling to a chart panel.
        """Style a chart panel with horizontal guides, short ticks and an open upper-right frame."""
        # Remove the top and right borders to leave the plotted data area open.
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y')  # Draw horizontal reference gridlines at the vertical tick values.
        ax.tick_params(length=3,width=.6)  # Set the common tick length and stroke width.

    # Label a highlighted zone with its positive and valid later weather-group counts.
    def recurring_label(metric, zone, display_name):
        """Format a zone name with counts of positive and valid later groups for one metric."""
        # Select later-period weather groups belonging to this target zone.
        later = scores[scores.period.eq('later') & scores.target_zone.eq(zone)]
        valid = later[metric].dropna()  # Remove groups lacking a value for the metric used in this map.
        # Append the count above zero and the total number of measured groups to the name.
        return f'{display_name} {int(valid.gt(0).sum())}/{len(valid)}'

    with plt.rc_context(STYLE):  # Apply the shared report style only while drawing these six figures.
        # Map the fixed cohort in the complete NYC geography.
        fig=plt.figure(figsize=(6.3,4.15))  # Create the landscape canvas for the study-area map.
        # Allocate space for the NYC map and a text panel with its axes hidden.
        ax=fig.add_axes([.015,.07,.66,.86]); side=fig.add_axes([.70,.07,.28,.86]);side.axis('off')
        # Draw each NYC zone, highlighting members of the fixed ride-hailing cohort.
        for zid,geom in GEOMS.items():draw_geom(ax,geom,facecolor='#80B8BD' if zid in ANALYSIS else LIGHT)
        # Overlay the borough boundaries and add the map extent, scale and north indicator.
        borough_borders(ax);map_axis(ax);scale_north(ax)
        # Set projected label positions for the Bronx and Manhattan.
        labels=[('Bronx',(1031000,267000)),('Manhattan',(957000,232000)),
                # Add projected label positions for Queens, Brooklyn and Staten Island.
                ('Queens',(1047000,205000)),('Brooklyn',(998000,165000)),('Staten Island',(944000,137000))]
        # Center each borough name in bold above its map position.
        for t,(x,y) in labels:ax.text(x,y,t,fontsize=8.4,ha='center',weight='bold',zorder=10,
            # Put a white backing behind each borough label.
            bbox={'facecolor':'white','edgecolor':'none','pad':1})
        # Identify the citywide taxi-zone map as panel a.
        ax.text(.02,.98,'a  New York City Taxi Zones',transform=ax.transAxes,ha='left',va='top',weight='bold',fontsize=9.5)
        # Title the right-hand geographical framework as panel b.
        side.text(0,.98,'b  Geographical framework',weight='bold',fontsize=9.2,va='top')
        # Show the count of NYC zone geometries actually drawn.
        side.text(0,.84,f'{len(GEOMS)} NYC zones',weight='bold',fontsize=10)
        # Show the number of zones in the fixed ride-hailing recovery cohort.
        side.text(0,.76,f'{len(ANALYSIS)} zones in the fixed\nride-hailing recovery cohort',fontsize=8.7,linespacing=1.4)
        # Create the legend symbol for the highlighted recovery cohort.
        side.legend(handles=[Patch(facecolor='#80B8BD',edgecolor=EDGE,label='Recovery cohort'),
                             # Pair it with the neutral symbol for other NYC zones.
                             Patch(facecolor=LIGHT,edgecolor=EDGE,label='Other NYC zones')],
                    # Position the cohort legend below the two zone counts in the side panel.
                    loc='upper left',bbox_to_anchor=(-.04,.67),frameon=False,handlelength=1.5,labelspacing=.8)
        # Lay out the three steps from hourly pickups and zone rainfall to aligned recovery.
        for y,title,desc in [(0.45,'Hourly mobility','Completed pickups by zone'),(.27,'Local rainfall','MRMS area-weighted to zones'),(.09,'Aligned recovery','Hours relative to local rain end')]:
            # Place each step's title at its assigned vertical position.
            side.text(0,y,title,weight='bold',fontsize=9)
            # Place its one-line description directly below the title.
            side.text(0,y-.06,desc,fontsize=8.2)
            # Connect the first two steps to the next one using downward arrows.
            if y>.10:side.annotate('',xy=(.06,y-.14),xytext=(.06,y-.09),
                                # Style the connectors as thin gray arrows.
                                arrowprops={'arrowstyle':'->','lw':.8,'color':GRAY})
        # Print the map's projected coordinate reference system beneath the figure.
        fig.text(.02,.012,'Projected coordinate system: NAD83 / New York Long Island (EPSG:2263).',fontsize=7.3,color=GRAY)
        save(fig,'Figure_1_Study_Area')  # Export the complete citywide study-area figure.

        # Retrieve the Pilot records prepared for the spatial response figure.
        records=pd.DataFrame(GIS['slide5']['records'])
        # Index the ride-hailing snapshot by zone ID for polygon coloring.
        rh=records[records.service=='Ride-hailing'].set_index('zone_id')
        # Build an orange-to-blue diverging color map with a pale neutral midpoint.
        cmap=LinearSegmentedColormap.from_list('diverge',[ORANGE,'#FCFCF8',BLUE])
        # Fix the symmetric percentage scale from minus 125 to plus 125 for the full map.
        norm=Normalize(-125,125)
        # Create the canvas for the Pilot map and its service comparison.
        fig=plt.figure(figsize=(6.3,4.35))
        # Put the city map on the left and the Midtown service bars on the right.
        ax=fig.add_axes([.015,.17,.61,.75]);bar=fig.add_axes([.73,.40,.25,.39])
        # Color each NYC polygon using the corresponding ride-hailing observation.
        for zid,geom in GEOMS.items():
            # Read the zone's eligible pickup change relative to its Pilot baseline.
            val=rh.loc[zid,'map_change_pct']
            # Use the color scale for supported observations and gray for missing values.
            draw_geom(ax,geom,facecolor=cmap(norm(val)) if pd.notna(val) else '#CFD3D6')
        # Add borough boundaries, citywide bounds, scale and north to the response map.
        borough_borders(ax,lw=.5);map_axis(ax);scale_north(ax)
        # Title panel a with the mapped service and measure.
        ax.set_title('a  Ride-hailing pickup change',pad=3)
        # Choose Midtown Center and LaGuardia Airport with fixed label positions.
        for zid,name,xy in [(161,'Midtown Center',(937000,248000)),(138,'LaGuardia Airport',(1022000,258000))]:
            # Format each highlighted zone's observed change with its sign and two decimals.
            lbl = f"{name}\n{rh.loc[zid, 'map_change_pct']:+.2f}%".replace('-', '−')
            # Outline the highlighted zone above its existing fill.
            draw_geom(ax,GEOMS[zid],facecolor='none',edgecolor='#20262B',linewidth=1,zorder=8)
            label_zone(ax,zid,lbl,xy,fontsize=8)  # Connect the zone label to its projected anchor.
        # Reserve a horizontal strip beneath the map for the color bar.
        cbax=fig.add_axes([.065,.103,.50,.027])
        # Draw the map's percentage scale with fixed negative, zero and positive ticks.
        cb=fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm,cmap=cmap),cax=cbax,orientation='horizontal',ticks=[-125,-75,0,75,125])
        # Set a light border and compact tick marks on the color bar.
        cb.outline.set_linewidth(.5);cb.ax.tick_params(length=2,labelsize=7.5)
        # Label the percentage changes relative to baseline.
        cb.set_label('Change in pickups relative to baseline (%)',fontsize=8,labelpad=2)
        # Create a missing-baseline legend entry with the calculated number of affected zones.
        fig.legend(handles=[Patch(facecolor='#CFD3D6',edgecolor=EDGE,label=f'Insufficient baseline ({rh.map_change_pct.isna().sum()} zones)')],
                   # Place the missing-data legend below the right side of the map.
                   loc='lower left',bbox_to_anchor=(.65,.015),frameon=False,fontsize=7.5,handlelength=1.2)
        # Retrieve Midtown Center's taxi and ride-hailing changes in the bar order.
        vals=[records.loc[records.zone_id.eq(161) & records.service.eq(service), 'map_change_pct'].item() for service in ['Taxi', 'Ride-hailing']]
        # Save the two Midtown comparisons and the LaGuardia ride-hailing map value.
        numeric_values['figure2'] = {'Midtown_Taxi_pct': vals[0], 'Midtown_RideHailing_pct': vals[1], 'LaGuardia_RideHailing_pct': float(rh.loc[138, 'map_change_pct'])}
        # Draw the two service changes as horizontal bars for Midtown Center.
        bar.barh([1,0],vals,color=[ORANGE,BLUE],height=.45)
        # Add a zero line, label the service rows and use symmetric limits for their changes.
        bar.axvline(0,color=GRAY,lw=.7);bar.set_yticks([1,0],['Taxi','Ride-hailing']);bar.set_xlim(-65,65)
        # Set percentage ticks and leave vertical room for bar annotations.
        bar.set_xticks([-60,0,60]);bar.set_xlabel('Pickup change (%)');bar.set_ylim(-.65,1.7)
        # Hide unnecessary borders and keep the service names without horizontal tick marks.
        bar.spines[['top','right','left']].set_visible(False);bar.tick_params(axis='y',length=0,labelsize=8)
        bar.grid(axis='x',linewidth=.5)  # Draw vertical guides to make bar lengths comparable.
        # Print each signed change above its service row.
        for y,v in zip([1,0],vals):bar.text(0,y+.30,f'{v:+.2f}%',ha='center',fontsize=9.2,weight='bold')
        # Title the paired Midtown service comparison as panel b.
        fig.text(.66,.905,'b  Same place, different services',fontsize=9.2,weight='bold')
        # Identify the location and zone number shared by the two plotted services.
        fig.text(.66,.848,'Midtown Center (Zone 161)',fontsize=8.5)
        # Display the Pilot snapshot's date and New York clock hour.
        fig.text(.66,.24,pd.Timestamp(records.hour.iloc[0]).strftime('%d %B %Y\n%H:%M New York time (EDT)'),fontsize=8.8,linespacing=1.6)
        # Report how many NYC zones have a usable change on the ride-hailing map.
        fig.text(.66,.11,f'Map: {rh.map_change_pct.notna().sum()} eligible zones\nof {len(GEOMS)} NYC zones',fontsize=8.2,linespacing=1.5,color=GRAY)
        # Export the spatial response figure including both services and the map.
        save(fig,'Figure_2_Spatial_Responses')

        # Retrieve the Belmont hourly data and its calculated crossing and reversal positions.
        rr=REC['slides']['6'];d=pd.DataFrame(rr['plot_records']);case=rr['recommended_case']
        # Set the canvas size and plotting area for the single-case recovery chart.
        fig=plt.figure(figsize=(6.3,3.25));ax=fig.add_axes([.10,.23,.86,.66])
        # Use one horizontal position per observation and the raw deficit percentage on the vertical axis.
        xx=np.arange(len(d));yy=d['raw_deficit_percent'].to_numpy()
        # Shade the two starting bins before the first post-rain observation.
        ax.axvspan(-.4,2,color='#EEF3F4',zorder=0)
        # Mark the local rain-ending boundary with a vertical dotted line.
        ax.axvline(2,color=GRAY,lw=.8,ls=':')
        # Draw the ordered observed deficits with a point at every hourly value.
        ax.plot(xx,yy,'o-',color=BLUE,lw=1.7,ms=4.3,zorder=4)
        # Retrieve the half-gap threshold calculated from the event's own starting shortfall.
        threshold=case['half_gap_threshold_percent']
        # Draw that threshold as a dashed horizontal reference line.
        ax.axhline(threshold,color=ORANGE,ls='--',lw=1.2)
        # Label the threshold at the right edge of the plotted series.
        ax.text(7.17,threshold,'Half-gap\nthreshold',ha='right',va='bottom',color=ORANGE,fontsize=8)
        # Position the first-rebound label at the first observed half-gap crossing.
        ax.annotate('First half-gap rebound',xy=(case['first_index'],yy[case['first_index']]),xytext=(3.65,1.1),ha='center',fontsize=8,
                    # Point a leader arrow from the rebound label to the corresponding observation.
                    arrowprops={'arrowstyle':'->','lw':.7,'color':TEXT})
        # Position the renewed-shortfall label at the first recrossing above threshold.
        ax.annotate('Renewed shortfall',xy=(case['recross_index'],yy[case['recross_index']]),xytext=(4.1,37.5),ha='center',fontsize=8,
                    # Connect the reversal label to the measured deficit at that hour.
                    arrowprops={'arrowstyle':'->','lw':.7,'color':TEXT})
        # Label the rain-ending boundary between starting and recovery bins.
        ax.text(2.06,37.5,'Rain ends',fontsize=7.8,va='top',color=GRAY)
        # Show local clock times and add one-based recovery-hour labels to post-rain bins.
        ax.set_xticks(xx,[t.strftime('%H:%M') + (f'\nh{int(offset)+1}' if offset >= 0 else '') for t,offset in zip(d.hour_NYC,d.offset_from_end)])
        # Set common plotting bounds and identify the vertical measure as percent of predicted activity.
        ax.set(xlim=(-.35,7.35),ylim=(0,40),ylabel='Activity deficit relative to prediction (%)')
        # Style the panel and title it with Belmont and the local recovery date.
        clean(ax);ax.set_title('Belmont, ' + d.hour_NYC.iloc[2].strftime('%d %B %Y').lstrip('0'),pad=8)
        # Save the threshold, crossing and reversal hours and the entire plotted percentage series.
        numeric_values['figure3'] = {'half_gap_threshold_pct': threshold, 'first_rebound_h': int(case['first_index']-1), 'recross_h': int(case['recross_index']-1), 'gap_percent': yy.tolist()}
        # Use the dry-follow-up annotation when none of the six hours had positive measured rain.
        rain_note = ('No rain was recorded during h1–h6.' if case['post_rain_hours'] == 0 else
                     # Otherwise give the actual count of follow-up hours with positive rain.
                     f"Rain was recorded in {case['post_rain_hours']} hours during h1–h6.")
        # Add the local-time convention and observed follow-up rainfall note.
        fig.text(.1,.035,'Hourly bins in New York time (EDT). ' + rain_note,fontsize=8,color=GRAY)
        save(fig,'Figure_3_Recovery_Reversal')  # Export the finished Belmont rebound and reversal plot.

        # Copy the weather-group estimates for the two-service recovery-window comparison.
        groups=service_groups.copy()
        # Keep finite centered S80 results on the metric-specific cases common to both horizons.
        groups=groups[(groups['mode']=='centered')&(groups.metric=='S80')&groups['sample'].isin(['H24_METRIC_COMMON_H6','H24_METRIC_COMMON_H24'])].dropna(subset=['difference'])
        # Align the H6 and H24 estimates in adjacent columns for each weather group.
        pivot=groups.pivot(index='canonical_group',columns='horizon',values='difference').sort_index()
        # Check for an empty panel or a weather group lacking either horizon.
        if pivot.empty or pivot.isna().any().any():
            # Stop if either horizon lacks the group pairs needed for the comparison.
            raise ValueError('The paired service windows do not have common group support')
        # Average each horizon across the same equally weighted weather groups.
        mean=pivot[[6,24]].mean().to_numpy()
        # Save the plotted mean contrasts, horizon labels and number of groups.
        numeric_values['figure4'] = {'horizons': [6,24], 'equal_group_means': mean.tolist(), 'groups': len(pivot)}
        # Prepare the canvas for the paired service-window line plot.
        fig=plt.figure(figsize=(6.3,3.25));ax=fig.add_axes([.12,.20,.80,.69])
        # Draw a thin two-point line connecting each weather group's H6 and H24 contrasts.
        for _,row in pivot.iterrows():ax.plot([0,1],row[[6,24]],color='#AEB8BF',lw=1,marker='o',ms=4,zorder=2)
        # Overlay the equal-group mean with a heavier blue line and square markers.
        ax.plot([0,1],mean,color=BLUE,lw=2.6,marker='s',ms=7,zorder=5)
        # Draw a horizontal reference at equal taxi and ride-hailing shortfall hours.
        ax.axhline(0,color=GRAY,lw=.9)
        # Annotate both mean differences with a sign and their hour units.
        for x,y in enumerate(mean):ax.annotate(f'{y:+.2f} h',xy=(x,y),xytext=(8,8),textcoords='offset points',color=BLUE,weight='bold',fontsize=10)
        # Label the two horizons and the signed taxi-minus-ride-hailing duration scale.
        ax.set(xticks=[0,1],xticklabels=['H6: first 6 hours','H24: first 24 hours'],xlim=(-.16,1.28),ylim=(-1.6,8.0),yticks=[-1,0,2,4,6,8],ylabel='Taxi − ride-hailing shortfall hours')
        # Title the panel with the calculated number of cases common to both windows.
        clean(ax);ax.set_title(f"The same {_number_word(support['service_S80']['cases'])} cases in both observation windows",pad=8)
        # Create the blue square-and-line legend entry for the equal-group mean.
        ax.legend(handles=[Line2D([0],[0],color=BLUE,marker='s',lw=2.4,label='Equal-group mean'),
                           # Add the lighter marker and line used for individual weather groups.
                           Line2D([0],[0],color='#AEB8BF',marker='o',lw=1,label='Individual weather group')],
                  # Place the two entries in the upper-left corner without a frame.
                  loc='upper left',frameon=False,fontsize=8)
        # Add the matched case, zone and group counts and the service study years.
        fig.text(.12,.035,f"{support['service_S80']['cases']} zone–rainfall cases  •  {support['service_S80']['zones']} zones  •  {len(pivot)} weather groups  •  {pd.Timestamp(periods['service_start']).year}–{pd.Timestamp(periods['service_end']).year}",fontsize=8,color=GRAY)
        save(fig,'Figure_4_Recovery_Windows')  # Export the H6 versus H24 service comparison figure.
        # Save the weather-group rows behind the plotted paired lines.
        groups.to_csv(BUILD/'figure4_matched_group_values.csv',index=False)

        status_colors={  # Associate each later recurrence classification with a consistent fill color.
            # Highlight zones repeating the original rule in purple.
            'REPEATS_ORIGINAL_POINT_RULE':PURPLE,
            # Use the same gray for adequately supported cases that miss the rule in either direction.
            'NEGATIVE_OR_TIED':'#8F989F', 'POSITIVE_NOT_RULE':'#8F989F',
            # Mark cases with only one or two valid later groups in tan.
            'ONE_TWO_LATER_GROUPS':'#E6BE91',
            # Assign distinct base fills to absent comparisons and unavailable later metrics.
            'NO_LATER_COMPARISON':'#FFFFFF', 'LATER_METRIC_UNAVAILABLE':'#D8EDF0',
            # Use pale gray for earlier nonmembers and white outside the analytical cohort.
            'NOT_IN_FROZEN_LIST':'#ECEFF1', 'OUTSIDE_COHORT':'#FFFFFF',
        }  # Finish the classification palette used by both metric maps.
        # Use hatch patterns to distinguish missing comparisons from unavailable metrics.
        hatches={'NO_LATER_COMPARISON':'///','LATER_METRIC_UNAVAILABLE':'xxx','OUTSIDE_COHORT':None}
        fig=plt.figure(figsize=(6.3,5.10))  # Create the canvas for the two later-period recurrence maps.
        # Place matched panels side by side above their shared legend.
        axs=[fig.add_axes([.015,.33,.48,.61]),fig.add_axes([.505,.33,.48,.61])]
        # Draw one panel for cumulative shortfall hours and one for late-window closure.
        for ax,metric,title in zip(axs,['S80','Sprogress'],['a  More cumulative shortfall hours','b  Less late-window gap closure']):
            # Index the current metric's classified zones by TLC identifier.
            sub=rep[rep.metric==metric].set_index('zone_id')
            # Include every NYC geometry, even zones outside the analytical cohort.
            for zid,geom in GEOMS.items():
                # Look up the zone's later status or classify an absent zone as outside the cohort.
                st=sub.loc[zid,'replication_status'] if zid in sub.index else 'OUTSIDE_COHORT'
                # Draw the status fill and matching hatch using the shared zone-border color.
                draw_geom(ax,geom,facecolor=status_colors[st],hatch=hatches.get(st),edgecolor=EDGE,
                          # Emphasize the boundary of zones repeating the original rule.
                          linewidth=.35 if st=='REPEATS_ORIGINAL_POINT_RULE' else .25)
            # Apply the same borough borders, focused extent and five-kilometre reference to each panel.
            borough_borders(ax,.5);map_axis(ax,FOCUS);scale_north(ax,km=5,position=(.07,.10),north=(.91,.80),label_size=8)
            # Label the panel with its corresponding shortfall measure.
            ax.set_title(title,pad=5,fontsize=9)
            if metric=='S80':  # Use the duration-specific highlighted zones on the S80 map.
                # Position the Belmont and Inwood duration labels with their later-group counts.
                for zid,label,xy in [(20,recurring_label(metric, 20, 'Belmont'),(1023500,262500)),(127,recurring_label(metric, 127, 'Inwood'),(972000,260000)),
                    # Add Mount Hope and Financial District South to the duration labels.
                    (169,recurring_label(metric, 169, 'Mount Hope'),(1028000,242000)),(88,recurring_label(metric, 88, 'Financial District\nSouth'),(1008000,195500))]:
                    # Draw the four zone labels with the recurrence highlight color.
                    label_zone(ax,zid,label,xy,fontsize=8.7,color=PURPLE)
            else:  # Use the separate set of highlighted zones for the late-closure map.
                # Position Mount Hope and West Chelsea / Hudson Yards with their closure counts.
                for zid,label,xy in [(169,recurring_label(metric, 169, 'Mount Hope'),(1022000,254000)),(246,recurring_label(metric, 246, 'West Chelsea /\nHudson Yards'),(970000,224000)),
                    # Include Financial District South and attach all closure labels to their map anchors.
                    (88,recurring_label(metric, 88, 'Financial District\nSouth'),(1008000,195500))]:label_zone(ax,zid,label,xy,fontsize=8.7,color=PURPLE)
        # Geographic locator shows the focus extent, retaining citywide context.
        loc=fig.add_axes([.835,.020,.145,.155])  # Reserve a small panel for the citywide locator map.
        # Draw all NYC zones as a neutral citywide background in the locator.
        for zid,geom in GEOMS.items():draw_geom(loc,geom,facecolor='#E0E5E8',edgecolor='none')
        # Mark the focused panels' projected extent with a rectangle on the full-city locator.
        map_axis(loc,FULL);loc.add_patch(Rectangle((FOCUS[0],FOCUS[2]),FOCUS[1]-FOCUS[0],FOCUS[3]-FOCUS[2],facecolor='none',edgecolor=GRAY,lw=.8))
        # Label the locator rectangle as the mapped extent.
        loc.text(.5,-.04,'Map extent',transform=loc.transAxes,ha='center',fontsize=7.8)
        handles=[  # Assemble one shared legend for the two recurrence maps.
            # Explain the purple fill for zones that meet the later recurrence rule.
            Patch(facecolor=PURPLE,edgecolor=EDGE,label='Meets later recurrence rule'),
            # Explain the gray fill for zones with enough groups that miss the rule.
            Patch(facecolor='#8F989F',edgecolor=EDGE,label='≥3 later groups; rule not met'),
            # Identify the tan class with only one or two valid later groups.
            Patch(facecolor='#E6BE91',edgecolor=EDGE,label='Only 1–2 valid later groups'),
            # Explain the diagonal hatching used when no later comparison was selected.
            Patch(facecolor='white',edgecolor=EDGE,hatch='///',label='No later comparison'),
            # Explain cross-hatching for selected comparisons with an unavailable later metric.
            Patch(facecolor='#D8EDF0',edgecolor=EDGE,hatch='xxx',label='Later metric unavailable'),
            # Identify the light-gray zones outside the earlier candidate list.
            Patch(facecolor='#ECEFF1',edgecolor=EDGE,label='Outside earlier candidate list'),
            # Label the unfilled zones outside the cohort and show the actual cohort size.
            Patch(facecolor='white',edgecolor=EDGE,label=f'Outside {len(ANALYSIS)}-zone cohort'),
        ]  # Finish the ordered recurrence legend entries.
        # Lay the two-column legend across the space below the maps.
        fig.legend(handles=handles,loc='lower left',bbox_to_anchor=(.02,.065),ncol=2,frameon=False,
                   # Set legend type size, intercolumn spacing, swatch length and row spacing.
                   fontsize=8.4,columnspacing=1.9,handlelength=1.6,labelspacing=.9)
        # Explain the positive-over-valid group labels and give the centered horizon and later years.
        fig.text(.02,.015,f"Labels: positive / valid later weather groups. Centered H6; later observations in {pd.Timestamp(periods['later_start']).year}–{pd.Timestamp(periods['later_end']).year}.",fontsize=7.5,color=GRAY)
        # Export both recurrence maps together with their supporting legend and locator.
        save(fig,'Figure_5_Recurring_Shortfalls')
        # Save the earlier membership, valid later count and mapped status for each zone and metric.
        rep[['zone_id','zone_name','borough','metric','exploration_member','later_valid_groups','replication_status']].to_csv(BUILD/'figure5_status_values.csv',index=False)

        # Select the Mount Hope cases designated for the main two-column display.
        cases=[c for c in MH['cases'] if c['display_priority']=='main']
        cases.sort(key=lambda c:c['local_date'])  # Order the panels by each case's local date.
        # Record the zone identifiers meeting the later recurrence rule for each mapped metric.
        numeric_values['figure5'] = {metric: sorted(rep.loc[rep.metric.eq(metric) & rep.replication_status.eq('REPEATS_ORIGINAL_POINT_RULE'), 'zone_id'].astype(int).tolist()) for metric in ['S80','Sprogress']}
        # Start a list for the numerical values behind the Mount Hope case panels.
        numeric_values['figure6'] = []
        # Create one column per event with shared hourly axes within each column.
        fig,axs=plt.subplots(2,2,figsize=(6.3,4.80),sharex='col')
        # Reserve space for row titles, axis labels and the shared case legend.
        fig.subplots_adjust(left=.10,right=.97,bottom=.15,top=.92,hspace=.30,wspace=.28)
        allrows=[]  # Collect the hourly records plotted for both Mount Hope events.
        # Assign each chronological event to a column in the two-by-two figure.
        for col,c in enumerate(cases):
            # Read the plotted case records and use post-rain hour on the horizontal axis.
            rows=pd.DataFrame(c['plot_cumulative_records']);x=rows.h
            # Express the target's hourly centered deficit and its threshold in percentage points.
            threshold=100*c['target']['threshold_centered_80'];y=100*rows.target_centered_gap
            # Draw the target's centered-gap line in the upper panel for this case.
            a=axs[0,col];a.plot(x,y,color=PURPLE,lw=1.5,zorder=3)
            # Mark zero centered activity gap as a reference in the upper panel.
            a.axhline(0,color='#AEB8BF',lw=.7)
            # Draw the event-specific 80 percent closure threshold in orange.
            a.axhline(threshold,color=ORANGE,ls='--',lw=1)
            # Identify hours that add to the cumulative shortfall count.
            states=rows.target_shortfall_this_hour.astype(bool)
            # Fill the markers for hours when the target gap exceeds the threshold.
            a.scatter(x[states],y[states],facecolor=PURPLE,edgecolor=PURPLE,s=21,zorder=4)
            # Use open markers for hours at or below the target threshold.
            a.scatter(x[~states],y[~states],facecolor='white',edgecolor=PURPLE,s=23,zorder=5)
            # Keep hourly and centered-gap scales comparable across the case panels.
            a.set(ylim=(-15,20),yticks=[-10,0,10,20],xlim=(.7,6.5),xticks=range(1,7))
            # Label the upper panel by letter and its local event date.
            a.set_title(chr(ord('a')+col) + '  ' + pd.Timestamp(c['local_date']).strftime('%d %B %Y').lstrip('0'),pad=6)
            # Put the centered-gap units on the left-hand panel for the row.
            if col==0:a.set_ylabel('Centered activity gap (pp)')
            clean(a)  # Apply the common chart grid and frame to the hourly gap panel.
            b=axs[1,col]  # Select the lower panel for the same dated event.
            # Plot Mount Hope's running total of shortfall hours.
            b.plot(x,rows.target_cumulative_H80,color=PURPLE,lw=1.8,marker='o',ms=4,zorder=3)
            # Plot the corresponding equal-weight fixed-peer mean as a dashed line.
            b.plot(x,rows.equal_peer_mean_cumulative_H80,color=GRAY,lw=1.4,ls='--',marker='s',ms=3.5,zorder=2)
            # Use hours on both axes, with the recovery clock along the horizontal axis.
            b.set(ylim=(-.2,5.1),yticks=range(0,6),xticks=range(1,7),xlabel='Post-rain hour')
            # Title and letter the lower panels in reading order.
            b.set_title(('c  Cumulative shortfall' if col==0 else 'd  Cumulative shortfall'),pad=6)
            # Put the common cumulative-duration label on the left-hand panel.
            if col==0:b.set_ylabel('Cumulative shortfall hours')
            final=rows.iloc[-1]  # Select the final recovery hour for the end-of-window labels.
            # Display the signed final target-minus-peer shortfall in hours.
            b.text(5.95,4.55,f"$S_{{80}}$ = {final.target_minus_peer_cumulative_H80:+.2f} h",ha='right',fontsize=8.5,weight='bold',color=PURPLE)
            # Label Mount Hope's final cumulative duration beside its last observation.
            b.annotate(f'{final.target_cumulative_H80:.1f}',xy=(final.h,final.target_cumulative_H80),xytext=(7,0),textcoords='offset points',fontsize=8,color=PURPLE,va='center')
            # Label the fixed peers' final mean duration beside the dashed curve.
            b.annotate(f'{final.equal_peer_mean_cumulative_H80:.1f}',xy=(6,final.equal_peer_mean_cumulative_H80),xytext=(7,-1),textcoords='offset points',fontsize=8,color=GRAY,va='center')
            clean(b)  # Apply the common grid and frame to the cumulative panel.
            # Save the event ID, date, peer count and final target, peer and relative durations.
            numeric_values['figure6'].append({'event_id': c['event_id'], 'date': c['local_date'], 'peer_count': c['peer_count'], 'target_final_hours': float(final.target_cumulative_H80), 'peer_final_mean_hours': float(final.equal_peer_mean_cumulative_H80), 'relative_shortfall_hours': float(final.target_minus_peer_cumulative_H80)})
            # Collect each plotted hourly row with its event and date for the supporting CSV.
            for rr in c['plot_cumulative_records']:allrows.append({'date':c['local_date'],'event_id':c['event_id'],**rr})
        # Use a purple line and circle for the observed Mount Hope series in the legend.
        fig.legend(handles=[Line2D([0],[0],color=PURPLE,marker='o',lw=1.6,label='Mount Hope'),
                            # Label the dashed peer series with the calculated number of fixed comparison zones.
                            Line2D([0],[0],color=GRAY,marker='s',ls='--',lw=1.4,label=f"Mean of {_number_word(cases[0]['peer_count'])} fixed peers"),
                            # Add the orange dashed reference for 80 percent closure.
                            Line2D([0],[0],color=ORANGE,ls='--',lw=1,label='80% closure threshold')],
                   # Center the three shared legend entries below the four panels.
                   loc='lower center',bbox_to_anchor=(.53,.008),ncol=3,frameon=False,fontsize=7.6,columnspacing=1.1)
        save(fig,'Figure_6_Mount_Hope_Recovery')  # Export the two-event Mount Hope recovery figure.
        # Write the hourly plotted values for both events to the supporting CSV.
        pd.DataFrame(allrows).to_csv(BUILD/'figure6_mount_hope_values.csv',index=False)

    # Record the counts of NYC zones and fixed-cohort zones drawn in the study-area figure.
    numeric_values['figure1'] = {'NYC_zones': len(GEOMS), 'cohort_zones': len(ANALYSIS)}
    # Write the figure and sample-table values together as indented UTF-8 JSON.
    (BUILD / 'report_values.json').write_text(json.dumps(numeric_values, indent=2), encoding='utf8')
    # Return those values to callers that use the report generator programmatically.
    return numeric_values
