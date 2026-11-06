"""Reproduce the report's six figures and analytical-sample table."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch, Patch, Rectangle
from matplotlib.path import Path as MplPath
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize, LinearSegmentedColormap
from shapely.geometry import shape
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

FT_TO_M = 0.3048006096012192
FULL = (907000, 1075000, 113000, 281000)
FOCUS = (968000, 1055000, 183000, 279000)

BLUE = '#0072B2'
TEAL = '#277E87'
PURPLE = '#9B3B75'
ORANGE = '#B96821'
GRAY = '#6B747B'
LIGHT = '#F0F1F2'
EDGE = '#B8BDC1'
TEXT = '#20262B'
STYLE = {
    'font.family': 'Arial', 'font.size': 8.5,
    'axes.titlesize': 9.5, 'axes.titleweight':'bold', 'axes.titlelocation':'left',
    'axes.labelsize': 8.5, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.fontsize': 8, 'axes.edgecolor': '#68737B', 'axes.linewidth': .6,
    'text.color':TEXT, 'axes.labelcolor':TEXT, 'xtick.color':TEXT, 'ytick.color':TEXT,
    'figure.facecolor':'white', 'axes.facecolor':'white',
    'pdf.fonttype':42, 'ps.fonttype':42, 'svg.fonttype':'none',
    'savefig.facecolor':'white', 'savefig.dpi':300,
    'grid.color':'#E1E5E8','grid.linewidth':.5,'axes.axisbelow':True,
    'hatch.linewidth':.45, 'lines.solid_capstyle':'round',
}

def _number_word(number):
    """Format a small count in a sentence without fixing its value."""
    words = ('zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten')
    return words[number] if 0 <= number < len(words) else str(number)


def _main_group_scores(frame):
    """Select centered six-hour comparisons used by the recurrence maps."""
    selected = frame.copy()
    for column, value in [('mode', 'centered'), ('H', 6),
                          ('subset', 'all_selected'),
                          ('comparison_pool', 'SAME_KNOWN_RAIN_CLASS'),
                          ('inference_scope', 'MAIN_VERIFIED_CLOSED')]:
        if column in selected:
            selected = selected[selected[column].eq(value)]
    return selected


def _case_curves(hourly, peers):
    """Calculate plotted rebound and cumulative-shortfall paths from hourly gaps."""
    def event_rows(event_id):
        selected = hourly[hourly.event_id.eq(event_id)].set_index('offset_from_end').sort_index()
        if selected.index.duplicated().any():
            raise ValueError(f'Duplicate hourly rows for {event_id}')
        return selected.loc[list(range(-2, 6))].copy()

    belmont = event_rows('N020E0010')
    initial = belmont.loc[[-2, -1], 'gap'].mean()
    threshold = 0.5 * initial
    post = belmont.loc[list(range(6)), 'gap']
    first = int(post[post.le(threshold)].index[0])
    later = post[post.index > first]
    recross = int(later[later.gt(threshold)].index[0])
    records = belmont.reset_index()
    records['raw_deficit_percent'] = records.gap * 100
    records['hour_NYC'] = pd.to_datetime(records.hour_utc, utc=True).dt.tz_convert('America/New_York')
    recovery = {'slides': {'6': {'plot_records': records,
        'recommended_case': {'half_gap_threshold_percent': threshold * 100,
                             'first_index': first + 2, 'recross_index': recross + 2,
                             'post_rain_hours': int(belmont.loc[list(range(6)), 'rain_mean_mm'].gt(0).sum())}}}}

    selected_peers = peers.copy()
    for column, value in [('comparison_pool', 'SAME_KNOWN_RAIN_CLASS'),
                          ('inference_scope', 'MAIN_VERIFIED_CLOSED')]:
        if column in selected_peers:
            selected_peers = selected_peers[selected_peers[column].eq(value)]
    cases = []
    for target in ['Z169E0385', 'L3Z169_20250504T02_20250504T04']:
        peer_ids = selected_peers.loc[selected_peers.target_event.eq(target), 'peer_event'].drop_duplicates().tolist()
        if not peer_ids:
            raise ValueError(f'No selected peers for {target}')
        paths = []
        for event_id in [target, *peer_ids]:
            rows = event_rows(event_id)
            d0 = rows.loc[[-2, -1], 'gap_centered'].mean()
            if not np.isfinite(d0) or d0 <= 0:
                raise ValueError(f'Nonpositive starting gap for {event_id}')
            values = rows.loc[list(range(6)), 'gap_centered'].to_numpy()
            if not np.isfinite(values).all():
                raise ValueError(f'Incomplete recovery curve for {event_id}')
            paths.append(np.cumsum(values > 0.2 * d0))
        rows = event_rows(target)
        d0 = rows.loc[[-2, -1], 'gap_centered'].mean()
        post = rows.loc[list(range(6)), 'gap_centered'].to_numpy()
        peer_mean = np.mean(paths[1:], axis=0)
        local = pd.Timestamp(rows.loc[0, 'hour_utc']).tz_convert('America/New_York')
        plotted = pd.DataFrame({'h': np.arange(1, len(post) + 1),
            'target_centered_gap': post,
            'target_shortfall_this_hour': post > 0.2 * d0,
            'target_cumulative_H80': paths[0],
            'equal_peer_mean_cumulative_H80': peer_mean,
            'target_minus_peer_cumulative_H80': paths[0] - peer_mean})
        cases.append({'event_id': target, 'local_date': local.strftime('%Y-%m-%d'),
                      'display_priority': 'main', 'peer_count': len(peer_ids),
                      'target': {'threshold_centered_80': 0.2 * d0},
                      'plot_cumulative_records': plotted.to_dict('records')})
    return recovery, {'cases': cases}


def _sample_table(snapshot, events, scores, service_groups, cohort, periods):
    """Return Table 1 and its numeric support, calculated from the sample tables."""
    def year_span(start, end):
        first, last = pd.Timestamp(start).year, pd.Timestamp(end).year
        return str(first) if first == last else f'{first}–{last}'

    first, last = pd.Timestamp(periods['source_start']), pd.Timestamp(periods['source_end'])
    recovery_period = f"{first.year}–{last.strftime('%B %Y')}"
    centered_period = year_span(periods['centered_start'], periods['centered_end'])
    service_period = year_span(periods['service_start'], periods['service_end'])
    rebounds = events[events.core_weather_verified.eq(True) & events.raw_D0.gt(0)
                      & events.raw_T50_6_hour.notna()]
    support = {'cohort_zones': len(cohort), 'half_gap_rebounds': len(rebounds)}
    for metric in ['S80', 'Sprogress']:
        support[f'recurrence_{metric}_groups'] = int(scores.loc[scores[metric].notna(), 'canonical_group'].nunique())
        service_metric = 'Sprogress_pp' if metric == 'Sprogress' else metric
        common = service_groups[service_groups['mode'].eq('centered')
                 & service_groups.metric.eq(service_metric)
                 & service_groups['sample'].eq('H24_METRIC_COMMON_H24')
                 & service_groups.difference.notna()]
        ids = set(';'.join(common.event_ids.fillna('')).split(';')) - {''}
        selected = events[events.event_id.isin(ids)]
        if selected.event_id.nunique() != len(ids):
            raise ValueError('Service-comparison cases are missing from the event table')
        support[f'service_{metric}'] = {'cases': len(ids), 'zones': int(selected.zone_id.nunique()),
                                         'groups': int(common.canonical_group.nunique())}
    rh = snapshot[snapshot.service.eq('Ride-hailing')]
    support['snapshot_rh_zones'] = int(rh.map_change_pct.notna().sum())
    snapshot_date = pd.Timestamp(snapshot.hour.iloc[0]).strftime('%d %B %Y').lstrip('0')
    s80, progress = support['service_S80'], support['service_Sprogress']
    rows = [
        ['Spatial snapshot', snapshot_date,
         f"Zone × service × hour; {support['snapshot_rh_zones']} usable ride-hailing zones"],
        ['Recovery reversals', recovery_period,
         f"{len(rebounds):,} zone–episode records with an observed half-gap rebound"],
        ['Relative recurrence', centered_period + ' centered results',
         f"Fixed {len(cohort)}-zone cohort; {support['recurrence_S80_groups']} weather groups for S80, "
         f"{support['recurrence_Sprogress_groups']} for Sprogress"],
        ['Matched H6/H24 services', service_period,
         f"Same {s80['cases']} zone–episode cases, {s80['zones']} zones and {s80['groups']} weather groups"],
        ['H6/H24 late gap closure', service_period,
         f"Same {progress['cases']} cases, {progress['zones']} zones and {progress['groups']} weather groups"],
    ]
    return pd.DataFrame(rows, columns=['Analysis', 'Period', 'Observation and support']), support


def generate_report_outputs(data_dir, results_dir, output_dir):
    """Write six PNG/SVG/PDF figures and Table 1 from the computed results."""
    data_dir, results_dir, output_dir = map(Path, [data_dir, results_dir, output_dir])
    FIG = output_dir / 'figures'
    BUILD = output_dir / 'tables'
    FIG.mkdir(parents=True, exist_ok=True)
    BUILD.mkdir(parents=True, exist_ok=True)
    geometry = json.loads((data_dir / 'map_geometry_2263.json').read_text(encoding='utf8'))
    GEOMS = {int(f['properties']['zone_id']): shape(f['geometry'])
             for f in geometry['features'] if int(f['properties']['zone_id']) != 1}
    labels = pd.read_csv(data_dir / 'zone_labels.csv').set_index('zone_id')
    PROPS = {int(zone): {'zone_name': row.zone_name, 'borough': row.borough,
                       'label_point_2263': [row.label_x_2263, row.label_y_2263]}
             for zone, row in labels.iterrows()}
    ANALYSIS = set(pd.read_csv(data_dir / 'recovery_cohort.csv').zone_id)
    hourly = pd.read_csv(results_dir / 'hourly_curves.csv.gz')
    peers = pd.read_csv(results_dir / 'peer_selection.csv')
    events = pd.read_csv(results_dir / 'recovery_event_metrics.csv')
    scores = _main_group_scores(pd.read_csv(results_dir / 'zone_weather_group_recovery.csv'))
    service_groups = pd.read_csv(results_dir / 'service_weather_group_summary.csv')
    rep = pd.read_csv(results_dir / 'replication_status_long.csv')
    records = pd.read_csv(results_dir / 'pilot_snapshot.csv')
    records['map_change_pct'] = 100 * (records.actual / records.baseline - 1)
    eligible = records.reference_n.ge(4) & records.baseline.ge(10)
    records.loc[~eligible, 'map_change_pct'] = np.nan
    GIS = {'slide5': {'records': records}}
    REC, MH = _case_curves(hourly, peers)
    periods = json.loads((data_dir / 'study_periods.json').read_text(encoding='utf8'))
    table, support = _sample_table(records, events, scores, service_groups, ANALYSIS, periods)
    table.to_csv(BUILD / 'Table_1_Analytical_Samples.csv', index=False)
    lines = ['| ' + ' | '.join(table.columns) + ' |', '| --- | --- | --- |']
    lines += ['| ' + ' | '.join(row) + ' |' for row in table.itertuples(index=False, name=None)]
    (BUILD / 'Table_1_Analytical_Samples.md').write_text('\n'.join(lines) + '\n', encoding='utf8')
    numeric_values = {'table1': support}

    def polygon_path(poly):
        poly = orient(poly, sign=1.0)
        verts, codes = [], []
        for ring in [poly.exterior, *poly.interiors]:
            coords = np.asarray(ring.coords)
            verts.extend(coords)
            codes.extend([MplPath.MOVETO]+[MplPath.LINETO]*(len(coords)-2)+[MplPath.CLOSEPOLY])
        return MplPath(verts,codes)

    def draw_geom(ax, geom, facecolor=LIGHT, edgecolor=EDGE, linewidth=.25, zorder=1, hatch=None):
        polygons = [geom] if geom.geom_type == 'Polygon' else list(geom.geoms)
        for poly in polygons:
            ax.add_patch(PathPatch(polygon_path(poly),facecolor=facecolor,edgecolor=edgecolor,
                                  linewidth=linewidth,zorder=zorder,hatch=hatch))

    def map_axis(ax, extent=FULL):
        ax.set_xlim(extent[:2]); ax.set_ylim(extent[2:]); ax.set_aspect('equal')
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(False)

    def borough_borders(ax, lw=.6):
        by = {}
        for zid,geom in GEOMS.items(): by.setdefault(PROPS[zid]['borough'],[]).append(geom)
        for gs in by.values(): draw_geom(ax,unary_union(gs),facecolor='none',edgecolor='#5D6A71',linewidth=lw,zorder=3)

    def scale_north(ax, km=10, position=(.08,.08), north=(.93,.82), label_size=7.4):
        xmin,xmax=ax.get_xlim();ymin,ymax=ax.get_ylim(); dx=xmax-xmin;dy=ymax-ymin
        x=xmin+position[0]*dx;y=ymin+position[1]*dy; L=km*1000/FT_TO_M
        ax.plot([x,x+L],[y,y],lw=1.2,color=TEXT,zorder=10)
        for xx in [x,x+L/2,x+L]:ax.plot([xx,xx],[y-.006*dy,y+.006*dy],lw=.8,color=TEXT,zorder=10)
        ax.text(x,y-.024*dy,'0',va='top',ha='center',fontsize=label_size)
        ax.text(x+L,y-.024*dy,f'{km:g} km',va='top',ha='center',fontsize=label_size)
        ax.annotate('',xy=(north[0],north[1]+.12),xytext=north,
                    xycoords='axes fraction',arrowprops={'arrowstyle':'-|>','color':TEXT,'lw':.9})
        ax.text(north[0],north[1]+.13,'N',ha='center',va='bottom',transform=ax.transAxes,fontsize=8)

    def point(zid):
        return PROPS[zid]['label_point_2263']

    def label_zone(ax,zid,label,xytext,fontsize=8,color=TEXT):
        ax.annotate(label,xy=point(zid),xytext=xytext,fontsize=fontsize,color=color,
                    ha='left',va='center',annotation_clip=False,
                    bbox={'facecolor':'white','edgecolor':'none','pad':1.5},
                    arrowprops={'arrowstyle':'-','color':color,'lw':.65,'shrinkA':1,'shrinkB':1},zorder=12)

    def save(fig,stem):
        """Export one figure at its fixed physical size in three formats."""
        for ext in ['png','pdf','svg']:fig.savefig(FIG/f'{stem}.{ext}',dpi=300)
        plt.close(fig)

    def clean(ax):
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y')
        ax.tick_params(length=3,width=.6)

    def recurring_label(metric, zone, display_name):
        later = scores[scores.period.eq('later') & scores.target_zone.eq(zone)]
        valid = later[metric].dropna()
        return f'{display_name} {int(valid.gt(0).sum())}/{len(valid)}'

    with plt.rc_context(STYLE):
        # Map the fixed cohort in the complete NYC geography.
        fig=plt.figure(figsize=(6.3,4.15))
        ax=fig.add_axes([.015,.07,.66,.86]); side=fig.add_axes([.70,.07,.28,.86]);side.axis('off')
        for zid,geom in GEOMS.items():draw_geom(ax,geom,facecolor='#80B8BD' if zid in ANALYSIS else LIGHT)
        borough_borders(ax);map_axis(ax);scale_north(ax)
        labels=[('Bronx',(1031000,267000)),('Manhattan',(957000,232000)),
                ('Queens',(1047000,205000)),('Brooklyn',(998000,165000)),('Staten Island',(944000,137000))]
        for t,(x,y) in labels:ax.text(x,y,t,fontsize=8.4,ha='center',weight='bold',zorder=10,
            bbox={'facecolor':'white','edgecolor':'none','pad':1})
        ax.text(.02,.98,'a  New York City Taxi Zones',transform=ax.transAxes,ha='left',va='top',weight='bold',fontsize=9.5)
        side.text(0,.98,'b  Geographical framework',weight='bold',fontsize=9.2,va='top')
        side.text(0,.84,f'{len(GEOMS)} NYC zones',weight='bold',fontsize=10)
        side.text(0,.76,f'{len(ANALYSIS)} zones in the fixed\nride-hailing recovery cohort',fontsize=8.7,linespacing=1.4)
        side.legend(handles=[Patch(facecolor='#80B8BD',edgecolor=EDGE,label='Recovery cohort'),
                             Patch(facecolor=LIGHT,edgecolor=EDGE,label='Other NYC zones')],
                    loc='upper left',bbox_to_anchor=(-.04,.67),frameon=False,handlelength=1.5,labelspacing=.8)
        for y,title,desc in [(0.45,'Hourly mobility','Completed pickups by zone'),(.27,'Local rainfall','MRMS area-weighted to zones'),(.09,'Aligned recovery','Hours relative to local rain end')]:
            side.text(0,y,title,weight='bold',fontsize=9)
            side.text(0,y-.06,desc,fontsize=8.2)
            if y>.10:side.annotate('',xy=(.06,y-.14),xytext=(.06,y-.09),
                                arrowprops={'arrowstyle':'->','lw':.8,'color':GRAY})
        fig.text(.02,.012,'Projected coordinate system: NAD83 / New York Long Island (EPSG:2263).',fontsize=7.3,color=GRAY)
        save(fig,'Figure_1_Study_Area')

        records=pd.DataFrame(GIS['slide5']['records'])
        rh=records[records.service=='Ride-hailing'].set_index('zone_id')
        cmap=LinearSegmentedColormap.from_list('diverge',[ORANGE,'#FCFCF8',BLUE])
        norm=Normalize(-125,125)
        fig=plt.figure(figsize=(6.3,4.35))
        ax=fig.add_axes([.015,.17,.61,.75]);bar=fig.add_axes([.73,.40,.25,.39])
        for zid,geom in GEOMS.items():
            val=rh.loc[zid,'map_change_pct']
            draw_geom(ax,geom,facecolor=cmap(norm(val)) if pd.notna(val) else '#CFD3D6')
        borough_borders(ax,lw=.5);map_axis(ax);scale_north(ax)
        ax.set_title('a  Ride-hailing pickup change',pad=3)
        for zid,name,xy in [(161,'Midtown Center',(937000,248000)),(138,'LaGuardia Airport',(1022000,258000))]:
            lbl = f"{name}\n{rh.loc[zid, 'map_change_pct']:+.2f}%".replace('-', '−')
            draw_geom(ax,GEOMS[zid],facecolor='none',edgecolor='#20262B',linewidth=1,zorder=8)
            label_zone(ax,zid,lbl,xy,fontsize=8)
        cbax=fig.add_axes([.065,.103,.50,.027])
        cb=fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm,cmap=cmap),cax=cbax,orientation='horizontal',ticks=[-125,-75,0,75,125])
        cb.outline.set_linewidth(.5);cb.ax.tick_params(length=2,labelsize=7.5)
        cb.set_label('Change in pickups relative to baseline (%)',fontsize=8,labelpad=2)
        fig.legend(handles=[Patch(facecolor='#CFD3D6',edgecolor=EDGE,label=f'Insufficient baseline ({rh.map_change_pct.isna().sum()} zones)')],
                   loc='lower left',bbox_to_anchor=(.65,.015),frameon=False,fontsize=7.5,handlelength=1.2)
        vals=[records.loc[records.zone_id.eq(161) & records.service.eq(service), 'map_change_pct'].item() for service in ['Taxi', 'Ride-hailing']]
        numeric_values['figure2'] = {'Midtown_Taxi_pct': vals[0], 'Midtown_RideHailing_pct': vals[1], 'LaGuardia_RideHailing_pct': float(rh.loc[138, 'map_change_pct'])}
        bar.barh([1,0],vals,color=[ORANGE,BLUE],height=.45)
        bar.axvline(0,color=GRAY,lw=.7);bar.set_yticks([1,0],['Taxi','Ride-hailing']);bar.set_xlim(-65,65)
        bar.set_xticks([-60,0,60]);bar.set_xlabel('Pickup change (%)');bar.set_ylim(-.65,1.7)
        bar.spines[['top','right','left']].set_visible(False);bar.tick_params(axis='y',length=0,labelsize=8)
        bar.grid(axis='x',linewidth=.5)
        for y,v in zip([1,0],vals):bar.text(0,y+.30,f'{v:+.2f}%',ha='center',fontsize=9.2,weight='bold')
        fig.text(.66,.905,'b  Same place, different services',fontsize=9.2,weight='bold')
        fig.text(.66,.848,'Midtown Center (Zone 161)',fontsize=8.5)
        fig.text(.66,.24,pd.Timestamp(records.hour.iloc[0]).strftime('%d %B %Y\n%H:%M New York time (EDT)'),fontsize=8.8,linespacing=1.6)
        fig.text(.66,.11,f'Map: {rh.map_change_pct.notna().sum()} eligible zones\nof {len(GEOMS)} NYC zones',fontsize=8.2,linespacing=1.5,color=GRAY)
        save(fig,'Figure_2_Spatial_Responses')

        rr=REC['slides']['6'];d=pd.DataFrame(rr['plot_records']);case=rr['recommended_case']
        fig=plt.figure(figsize=(6.3,3.25));ax=fig.add_axes([.10,.23,.86,.66])
        xx=np.arange(len(d));yy=d['raw_deficit_percent'].to_numpy()
        ax.axvspan(-.4,2,color='#EEF3F4',zorder=0)
        ax.axvline(2,color=GRAY,lw=.8,ls=':')
        ax.plot(xx,yy,'o-',color=BLUE,lw=1.7,ms=4.3,zorder=4)
        threshold=case['half_gap_threshold_percent']
        ax.axhline(threshold,color=ORANGE,ls='--',lw=1.2)
        ax.text(7.17,threshold,'Half-gap\nthreshold',ha='right',va='bottom',color=ORANGE,fontsize=8)
        ax.annotate('First half-gap rebound',xy=(case['first_index'],yy[case['first_index']]),xytext=(3.65,1.1),ha='center',fontsize=8,
                    arrowprops={'arrowstyle':'->','lw':.7,'color':TEXT})
        ax.annotate('Renewed shortfall',xy=(case['recross_index'],yy[case['recross_index']]),xytext=(4.1,37.5),ha='center',fontsize=8,
                    arrowprops={'arrowstyle':'->','lw':.7,'color':TEXT})
        ax.text(2.06,37.5,'Rain ends',fontsize=7.8,va='top',color=GRAY)
        ax.set_xticks(xx,[t.strftime('%H:%M') + (f'\nh{int(offset)+1}' if offset >= 0 else '') for t,offset in zip(d.hour_NYC,d.offset_from_end)])
        ax.set(xlim=(-.35,7.35),ylim=(0,40),ylabel='Activity deficit relative to prediction (%)')
        clean(ax);ax.set_title('Belmont, ' + d.hour_NYC.iloc[2].strftime('%d %B %Y').lstrip('0'),pad=8)
        numeric_values['figure3'] = {'half_gap_threshold_pct': threshold, 'first_rebound_h': int(case['first_index']-1), 'recross_h': int(case['recross_index']-1), 'gap_percent': yy.tolist()}
        rain_note = ('No rain was recorded during h1–h6.' if case['post_rain_hours'] == 0 else
                     f"Rain was recorded in {case['post_rain_hours']} hours during h1–h6.")
        fig.text(.1,.035,'Hourly bins in New York time (EDT). ' + rain_note,fontsize=8,color=GRAY)
        save(fig,'Figure_3_Recovery_Reversal')

        groups=service_groups.copy()
        groups=groups[(groups['mode']=='centered')&(groups.metric=='S80')&groups['sample'].isin(['H24_METRIC_COMMON_H6','H24_METRIC_COMMON_H24'])].dropna(subset=['difference'])
        pivot=groups.pivot(index='canonical_group',columns='horizon',values='difference').sort_index()
        if pivot.empty or pivot.isna().any().any():
            raise ValueError('The paired service windows do not have common group support')
        mean=pivot[[6,24]].mean().to_numpy()
        numeric_values['figure4'] = {'horizons': [6,24], 'equal_group_means': mean.tolist(), 'groups': len(pivot)}
        fig=plt.figure(figsize=(6.3,3.25));ax=fig.add_axes([.12,.20,.80,.69])
        for _,row in pivot.iterrows():ax.plot([0,1],row[[6,24]],color='#AEB8BF',lw=1,marker='o',ms=4,zorder=2)
        ax.plot([0,1],mean,color=BLUE,lw=2.6,marker='s',ms=7,zorder=5)
        ax.axhline(0,color=GRAY,lw=.9)
        for x,y in enumerate(mean):ax.annotate(f'{y:+.2f} h',xy=(x,y),xytext=(8,8),textcoords='offset points',color=BLUE,weight='bold',fontsize=10)
        ax.set(xticks=[0,1],xticklabels=['H6: first 6 hours','H24: first 24 hours'],xlim=(-.16,1.28),ylim=(-1.6,8.0),yticks=[-1,0,2,4,6,8],ylabel='Taxi − ride-hailing shortfall hours')
        clean(ax);ax.set_title(f"The same {_number_word(support['service_S80']['cases'])} cases in both observation windows",pad=8)
        ax.legend(handles=[Line2D([0],[0],color=BLUE,marker='s',lw=2.4,label='Equal-group mean'),
                           Line2D([0],[0],color='#AEB8BF',marker='o',lw=1,label='Individual weather group')],
                  loc='upper left',frameon=False,fontsize=8)
        fig.text(.12,.035,f"{support['service_S80']['cases']} zone–rainfall cases  •  {support['service_S80']['zones']} zones  •  {len(pivot)} weather groups  •  {pd.Timestamp(periods['service_start']).year}–{pd.Timestamp(periods['service_end']).year}",fontsize=8,color=GRAY)
        save(fig,'Figure_4_Recovery_Windows')
        groups.to_csv(BUILD/'figure4_matched_group_values.csv',index=False)

        status_colors={
            'REPEATS_ORIGINAL_POINT_RULE':PURPLE,
            'NEGATIVE_OR_TIED':'#8F989F', 'POSITIVE_NOT_RULE':'#8F989F',
            'ONE_TWO_LATER_GROUPS':'#E6BE91',
            'NO_LATER_COMPARISON':'#FFFFFF', 'LATER_METRIC_UNAVAILABLE':'#D8EDF0',
            'NOT_IN_FROZEN_LIST':'#ECEFF1', 'OUTSIDE_COHORT':'#FFFFFF',
        }
        hatches={'NO_LATER_COMPARISON':'///','LATER_METRIC_UNAVAILABLE':'xxx','OUTSIDE_COHORT':None}
        fig=plt.figure(figsize=(6.3,5.10))
        axs=[fig.add_axes([.015,.33,.48,.61]),fig.add_axes([.505,.33,.48,.61])]
        for ax,metric,title in zip(axs,['S80','Sprogress'],['a  More cumulative shortfall hours','b  Less late-window gap closure']):
            sub=rep[rep.metric==metric].set_index('zone_id')
            for zid,geom in GEOMS.items():
                st=sub.loc[zid,'replication_status'] if zid in sub.index else 'OUTSIDE_COHORT'
                draw_geom(ax,geom,facecolor=status_colors[st],hatch=hatches.get(st),edgecolor=EDGE,
                          linewidth=.35 if st=='REPEATS_ORIGINAL_POINT_RULE' else .25)
            borough_borders(ax,.5);map_axis(ax,FOCUS);scale_north(ax,km=5,position=(.07,.10),north=(.91,.80),label_size=8)
            ax.set_title(title,pad=5,fontsize=9)
            if metric=='S80':
                for zid,label,xy in [(20,recurring_label(metric, 20, 'Belmont'),(1023500,262500)),(127,recurring_label(metric, 127, 'Inwood'),(972000,260000)),
                    (169,recurring_label(metric, 169, 'Mount Hope'),(1028000,242000)),(88,recurring_label(metric, 88, 'Financial District\nSouth'),(1008000,195500))]:
                    label_zone(ax,zid,label,xy,fontsize=8.7,color=PURPLE)
            else:
                for zid,label,xy in [(169,recurring_label(metric, 169, 'Mount Hope'),(1022000,254000)),(246,recurring_label(metric, 246, 'West Chelsea /\nHudson Yards'),(970000,224000)),
                    (88,recurring_label(metric, 88, 'Financial District\nSouth'),(1008000,195500))]:label_zone(ax,zid,label,xy,fontsize=8.7,color=PURPLE)
        # Geographic locator shows the focus extent, retaining citywide context.
        loc=fig.add_axes([.835,.020,.145,.155])
        for zid,geom in GEOMS.items():draw_geom(loc,geom,facecolor='#E0E5E8',edgecolor='none')
        map_axis(loc,FULL);loc.add_patch(Rectangle((FOCUS[0],FOCUS[2]),FOCUS[1]-FOCUS[0],FOCUS[3]-FOCUS[2],facecolor='none',edgecolor=GRAY,lw=.8))
        loc.text(.5,-.04,'Map extent',transform=loc.transAxes,ha='center',fontsize=7.8)
        handles=[
            Patch(facecolor=PURPLE,edgecolor=EDGE,label='Meets later recurrence rule'),
            Patch(facecolor='#8F989F',edgecolor=EDGE,label='≥3 later groups; rule not met'),
            Patch(facecolor='#E6BE91',edgecolor=EDGE,label='Only 1–2 valid later groups'),
            Patch(facecolor='white',edgecolor=EDGE,hatch='///',label='No later comparison'),
            Patch(facecolor='#D8EDF0',edgecolor=EDGE,hatch='xxx',label='Later metric unavailable'),
            Patch(facecolor='#ECEFF1',edgecolor=EDGE,label='Outside earlier candidate list'),
            Patch(facecolor='white',edgecolor=EDGE,label=f'Outside {len(ANALYSIS)}-zone cohort'),
        ]
        fig.legend(handles=handles,loc='lower left',bbox_to_anchor=(.02,.065),ncol=2,frameon=False,
                   fontsize=8.4,columnspacing=1.9,handlelength=1.6,labelspacing=.9)
        fig.text(.02,.015,f"Labels: positive / valid later weather groups. Centered H6; later observations in {pd.Timestamp(periods['later_start']).year}–{pd.Timestamp(periods['later_end']).year}.",fontsize=7.5,color=GRAY)
        save(fig,'Figure_5_Recurring_Shortfalls')
        rep[['zone_id','zone_name','borough','metric','exploration_member','later_valid_groups','replication_status']].to_csv(BUILD/'figure5_status_values.csv',index=False)

        cases=[c for c in MH['cases'] if c['display_priority']=='main']
        cases.sort(key=lambda c:c['local_date'])
        numeric_values['figure5'] = {metric: sorted(rep.loc[rep.metric.eq(metric) & rep.replication_status.eq('REPEATS_ORIGINAL_POINT_RULE'), 'zone_id'].astype(int).tolist()) for metric in ['S80','Sprogress']}
        numeric_values['figure6'] = []
        fig,axs=plt.subplots(2,2,figsize=(6.3,4.80),sharex='col')
        fig.subplots_adjust(left=.10,right=.97,bottom=.15,top=.92,hspace=.30,wspace=.28)
        allrows=[]
        for col,c in enumerate(cases):
            rows=pd.DataFrame(c['plot_cumulative_records']);x=rows.h
            threshold=100*c['target']['threshold_centered_80'];y=100*rows.target_centered_gap
            a=axs[0,col];a.plot(x,y,color=PURPLE,lw=1.5,zorder=3)
            a.axhline(0,color='#AEB8BF',lw=.7)
            a.axhline(threshold,color=ORANGE,ls='--',lw=1)
            states=rows.target_shortfall_this_hour.astype(bool)
            a.scatter(x[states],y[states],facecolor=PURPLE,edgecolor=PURPLE,s=21,zorder=4)
            a.scatter(x[~states],y[~states],facecolor='white',edgecolor=PURPLE,s=23,zorder=5)
            a.set(ylim=(-15,20),yticks=[-10,0,10,20],xlim=(.7,6.5),xticks=range(1,7))
            a.set_title(chr(ord('a')+col) + '  ' + pd.Timestamp(c['local_date']).strftime('%d %B %Y').lstrip('0'),pad=6)
            if col==0:a.set_ylabel('Centered activity gap (pp)')
            clean(a)
            b=axs[1,col]
            b.plot(x,rows.target_cumulative_H80,color=PURPLE,lw=1.8,marker='o',ms=4,zorder=3)
            b.plot(x,rows.equal_peer_mean_cumulative_H80,color=GRAY,lw=1.4,ls='--',marker='s',ms=3.5,zorder=2)
            b.set(ylim=(-.2,5.1),yticks=range(0,6),xticks=range(1,7),xlabel='Post-rain hour')
            b.set_title(('c  Cumulative shortfall' if col==0 else 'd  Cumulative shortfall'),pad=6)
            if col==0:b.set_ylabel('Cumulative shortfall hours')
            final=rows.iloc[-1]
            b.text(5.95,4.55,f"$S_{{80}}$ = {final.target_minus_peer_cumulative_H80:+.2f} h",ha='right',fontsize=8.5,weight='bold',color=PURPLE)
            b.annotate(f'{final.target_cumulative_H80:.1f}',xy=(final.h,final.target_cumulative_H80),xytext=(7,0),textcoords='offset points',fontsize=8,color=PURPLE,va='center')
            b.annotate(f'{final.equal_peer_mean_cumulative_H80:.1f}',xy=(6,final.equal_peer_mean_cumulative_H80),xytext=(7,-1),textcoords='offset points',fontsize=8,color=GRAY,va='center')
            clean(b)
            numeric_values['figure6'].append({'event_id': c['event_id'], 'date': c['local_date'], 'peer_count': c['peer_count'], 'target_final_hours': float(final.target_cumulative_H80), 'peer_final_mean_hours': float(final.equal_peer_mean_cumulative_H80), 'relative_shortfall_hours': float(final.target_minus_peer_cumulative_H80)})
            for rr in c['plot_cumulative_records']:allrows.append({'date':c['local_date'],'event_id':c['event_id'],**rr})
        fig.legend(handles=[Line2D([0],[0],color=PURPLE,marker='o',lw=1.6,label='Mount Hope'),
                            Line2D([0],[0],color=GRAY,marker='s',ls='--',lw=1.4,label=f"Mean of {_number_word(cases[0]['peer_count'])} fixed peers"),
                            Line2D([0],[0],color=ORANGE,ls='--',lw=1,label='80% closure threshold')],
                   loc='lower center',bbox_to_anchor=(.53,.008),ncol=3,frameon=False,fontsize=7.6,columnspacing=1.1)
        save(fig,'Figure_6_Mount_Hope_Recovery')
        pd.DataFrame(allrows).to_csv(BUILD/'figure6_mount_hope_values.csv',index=False)

    numeric_values['figure1'] = {'NYC_zones': len(GEOMS), 'cohort_zones': len(ANALYSIS)}
    (BUILD / 'report_values.json').write_text(json.dumps(numeric_values, indent=2), encoding='utf8')
    return numeric_values
