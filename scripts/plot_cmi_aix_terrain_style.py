#!/usr/bin/env python3
"""Original terrain-renderer adapter with illustrative ACK-anchored phase envelopes."""

import argparse
import importlib.util
import json
import shlex
import sys
from pathlib import Path
from unittest.mock import patch

import plot_cmi_aix_timeline as base
import plot_cmi_aix_rank_timeline as rank
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, FuncFormatter, MaxNLocator
from matplotlib.transforms import Bbox

RENDERER = Path('/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/CPP-ML-Interface/scripts/plot_p2p_timeline.py')
PHASES = ('input', 'inference', 'output')
COLORS = ('#4c9ed9', '#8e6bbd', '#f4a261')
REF_COLORS = ('#23764d', '#67508e', '#ad592b')
REF_LABELS = ('Collective gather end', 'Collective inference end', 'Collective scatter end')
CAVEAT = ('Illustrative ACK-anchored alignment, NOT measured synchronization. '
          'Each WG independently zeros its controller ml_step_start; no cross-WG concurrency interpretation.')
SEMANTICS = """Worker time is shifted by c_input_ready - w_input_send_complete using the original
synchronize_step_clocks with the exact solver-log membership, then every event in each WG
subtracts that controller's ml_step_start. Equality ignores ACK/communication delay and is
not trustworthy globally. alignment.csv records additive worker offsets and causal bounds
(controller time = worker time + offset); bounds are evidence, not error bars or synchronization.
Some aligned worker receipts precede controller sends: signed output spans and validity flags
are exported, never clamped. Invalid output pairs are marked by red crosses in detail plots
at their aligned receipt times; the original renderer cannot draw their negative bars.
Envelope widths themselves are validated nonnegative; an envelope can conceal invalid pair order.
Negative worker starts are retained, without clipping. Approximate clock alignment can shift
workers before controller ml_step_start; these logs cannot diagnose early arrival versus an
alignment artifact. Negative coordinates do not establish physically early worker arrival.
Detailed figures use the original render_step directly,
including controller lanes and worker send/receive display. Its final earliest-event zero
is recorded and applied to all collective references; tick labels undo that display shift
so zero explicitly denotes each WG's own controller start. Original global start/end lines are
only extrema across independently zeroed WGs, not a global measured interval.

Phase envelopes are windows INCLUDING gaps, not sums: input = earliest worker input_send_start
to latest worker input_send_complete (equal to latest worker-targeted input_ready under the
anchor); inference = first range_inference_start to last range_inference_end, a CPU-host
ready-range window, NOT torch_forward-only; output = first worker-targeted controller
result_send_start to last worker result_received. Controller self-copy is excluded from
input/output; all controller ready-range inference is included. CUDA phases in detail retain
their original host anchors; inter-request GPU overlap is approximate.

Dashed references are collective controller durations only: g = gather_collective_ms,
d = device_cpu_ms, s = scatter_collective_ms. Ends are g, g+d, g+d+s, independently
zeroed and concatenated schematically; arrival waits and gaps are omitted. These are NOT
observed collective timestamps or globally synchronized clocks. No collective bars are drawn.
Comparisons match collective/P2P by logged WG ID, never by controller-rank intersection.
One selected measured call is shown, not repetition averages. World memberships come from
base.parse_mapping solver logs; local peer IDs come from actual rank CSVs, never modulo.
"""


def load_renderer(path):
    spec = importlib.util.spec_from_file_location('cmi_original_terrain', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def references(row):
    g, d, s = (base.number(row[k]) for k in
               ('gather_collective_ms', 'device_cpu_ms', 'scatter_collective_ms'))
    return (g, g+d, g+d+s)


def endpoint(rows, event):
    return base.number(base.unique([r for r in rows if r['event'] == event], event)['time_s'])


def prepare(comparison, metadata, renderer):
    """Validate arbitrary-sized WGs and explicitly preprocess the original anchor."""
    p2p = comparison['p2p']
    run = metadata['cells'][p2p[0]['cell']]['selected_row']
    mapping = base.parse_mapping(base.ROOT / p2p[0]['solver_log'], run)
    wg_map = {r['controller']: r['members'] for r in mapping.values()}
    members = [w for group in wg_map.values() for w in group]
    if len(members) != len(set(members)) or set(members) != set(range(int(run['solver_ranks']))):
        raise ValueError('Incomplete or overlapping world membership')
    events, alignments, origins = [], [], {}
    for p in p2p:
        group, ctrl = p['workgroup'], p['controller']
        mapped = mapping[group]
        if mapped['controller'] != ctrl or mapped['members'] != list(map(int, p['members'].split(';'))):
            raise ValueError('Metric/log membership mismatch')
        rows_by_rank, locals_seen = {}, {}
        for world in mapped['members']:
            path = base.ROOT / p['run_dir'] / 'aix_gpu_diagnostics' / f'aix_p2p_timeline_rank_{world}.csv'
            rows = [r for r in base.read_csv(path) if int(r['step']) == p['call']]
            local = {int(r['workgroup_rank']) for r in rows}
            if len(local) != 1 or any(int(r['world_rank']) != world or
                    int(r['is_controller']) != int(world == ctrl) for r in rows):
                raise ValueError(f'Invalid rank diagnostic membership: {path}')
            locals_seen[world] = local.pop()
            rows_by_rank[world] = rows
            metadata['sources'][str(path.relative_to(base.ROOT))] = base.sha256(path)
            for row in rows:
                events.append({k: (v if k == 'event' else float(v) if k == 'time_s' else int(v))
                               for k, v in row.items()})
        if set(locals_seen.values()) != set(range(len(mapped['members']))) or locals_seen[ctrl] != 0:
            raise ValueError('Incomplete/duplicate local membership')
        controller = rows_by_rank[ctrl]
        origins[ctrl] = endpoint(controller, 'ml_step_start')
        for world, rows in rows_by_rank.items():
            if world == ctrl:
                continue
            rank.worker_intervals(rows)
            local = locals_seen[world]
            marks = {e: rank.controller_endpoint(controller, e, local) for e in rank.MARKERS}
            if list(marks.values()) != sorted(marks.values()):
                raise ValueError('Invalid controller peer endpoint order')
            send, complete = (endpoint(rows, e) for e in ('input_send_start', 'input_send_complete'))
            low, high, _ = rank.causal_bounds(marks['input_credit_send'], marks['input_ready'], send, complete)
            offset = marks['input_ready'] - complete
            output_span = endpoint(rows, 'result_received') + offset - marks['result_send_start']
            received = endpoint(rows, 'result_received')
            output_low = marks['result_send_start'] - received
            tightened = max(low, output_low)
            alignments.append(dict(comparison=comparison['name'], cell=p['cell'], call=p['call'],
                workgroup=group, controller=ctrl, world_rank=world, workgroup_rank=local,
                worker_offset_s=offset, causal_low_s=low, causal_high_s=high,
                controller_origin_s=origins[ctrl], worker_send_start_s=send,
                worker_send_complete_s=complete, controller_input_ready_s=marks['input_ready'],
                controller_credit_send_s=marks['input_credit_send'],
                controller_result_send_s=marks['result_send_start'], worker_result_received_s=received,
                worker_ack_to_receipt_ms=(received-complete)*1000,
                controller_ready_to_send_ms=(marks['result_send_start']-marks['input_ready'])*1000,
                output_causal_low_s=output_low, tightened_causal_low_s=tightened,
                tightened_causal_width_ms=(high-tightened)*1000,
                tightened_causal_feasible=tightened <= high,
                ack_satisfies_input_bounds=low <= offset <= high,
                raw_same_clock_order_valid=True,
                aligned_output_span_ms=output_span*1000,
                output_span_valid=output_span >= 0,
                assumption='illustrative ACK equality; ignores delay; not measured synchronization'))
    if len(alignments) != int(run['solver_ranks']) - len(wg_map):
        raise ValueError('Worker count != total ranks minus controllers')
    aligned = renderer.synchronize_step_clocks(events, wg_map)
    owner = {w: c for c, workers in wg_map.items() for w in workers}
    for row in aligned:
        row['time_s'] -= origins[owner[row['world_rank']]]
    minimum = min(r['time_s'] for r in aligned)
    for row in alignments:
        row['renderer_origin_relative_s'] = minimum
    envelopes = []
    for p in p2p:
        ctrl = p['controller']
        controller = [r for r in aligned if r['world_rank'] == ctrl]
        workers = [r for r in aligned if r['world_rank'] in wg_map[ctrl] and r['world_rank'] != ctrl]
        local_workers = {r['workgroup_rank'] for r in workers}
        sends = [r for r in controller if r['peer_workgroup_rank'] in local_workers]
        for phase, starts, ends, a, b in (
                ('input', workers, workers, 'input_send_start', 'input_send_complete'),
                ('inference', controller, controller, 'range_inference_start', 'range_inference_end'),
                ('output', sends, workers, 'result_send_start', 'result_received')):
            start = min(r['time_s'] for r in starts if r['event'] == a)*1000
            end = max(r['time_s'] for r in ends if r['event'] == b)*1000
            if end < start:
                raise ValueError(f'Negative {phase} envelope')
            envelopes.append(dict(comparison=comparison['name'], workgroup=p['workgroup'], controller=ctrl,
                phase=phase, start_ms=start, end_ms=end, duration_ms=end-start,
                invalid_output_pairs=sum(not r['output_span_valid'] for r in alignments if r['controller'] == ctrl),
                controller_origin_s=origins[ctrl], renderer_origin_relative_s=minimum,
                origin='independent WG controller ml_step_start; illustrative ACK alignment'))
    return aligned, wg_map, alignments, envelopes


def save_cropped(fig, out, name):
    """Crop only empty bottom canvas, leaving all axes at their original scale."""
    fig.canvas.draw()
    bottom = max(0, fig.get_tightbbox(fig.canvas.get_renderer()).y0-.08)
    bounds = Bbox.from_extents(0, bottom, fig.get_figwidth(), fig.get_figheight())
    savefig = fig.savefig
    with patch.object(fig, 'savefig', side_effect=lambda *a, **kw: savefig(*a, **kw, bbox_inches=bounds)):
        base.save(fig, out, name)


def render_envelopes(comparison, envelopes, out):
    coll = {r['workgroup']: r for r in comparison['collective']}
    fig, ax = base.plt.subplots(figsize=(12, 1.65+.85*len(coll)))
    ticks, labels = [], []
    for group_index, p in enumerate(comparison['p2p']):
        selected = [r for r in envelopes if r['workgroup'] == p['workgroup']]
        start_y = group_index*3.5
        for lane, phase, color in zip(range(3), PHASES, COLORS):
            y = start_y+lane
            row = base.unique([r for r in selected if r['phase'] == phase], phase)
            ax.broken_barh([(row['start_ms'], row['duration_ms'])], (y-.28, .56), facecolors=color)
            ticks.append(y)
            labels.append(('Input', 'Inference (host)', 'Output')[lane])
        for x, color in zip(references(coll[p['workgroup']]), REF_COLORS):
            ax.vlines(x, start_y-.45, start_y+2.45, color=color, linestyle='--', linewidth=1)
        ax.text(-.14, start_y+1, f"WG {p['workgroup']}\nr{p['controller']} / {len(p['members'].split(';'))-1}w",
                transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=8)
        if group_index:
            ax.axhline(start_y-.75, color='.7', linewidth=.6)
    ax.set_yticks(ticks, labels, fontsize=8)
    ax.set_ylim(ticks[-1]+.6, -.6)
    ax.axvline(0, color='.35', linewidth=.7)
    ax.set_xlabel("ms from each WG's own controller ml_step_start (independent zero; estimated alignment)", fontsize=8)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
    ax.grid(axis='x', alpha=.2)
    handles = [Line2D([], [], color=c, linestyle='--', label=t) for c, t in zip(REF_COLORS,
               REF_LABELS)]
    fig.legend(handles=handles, loc='upper center', ncol=3, fontsize=8, frameon=False,
               bbox_to_anchor=(.59, 1-.30/fig.get_figheight()))
    fig.suptitle(title(comparison), fontsize=10, y=1-.04/fig.get_figheight())
    fig.subplots_adjust(left=.23, right=.99, bottom=.60/fig.get_figheight(), top=1-.58/fig.get_figheight())
    save_cropped(fig, out, comparison['name']+'_phase_envelopes')


def title(comparison):
    row = comparison['p2p'][0]
    model = {'mlp': 'MLP (ov4)', 'transformer': 'Transformer (ov0)'}[row['model']]
    return base.title(comparison).replace(row['model'], model, 1)


def stagger_range_labels(ax, ctrl_y):
    """Allocate rows using real text extents, reflowing the controller band only."""
    labels = [t for t in ax.texts if t.get_transform() == ax.transData
              and abs(t.get_position()[1] - (ctrl_y-.48)) < 1e-7]
    if not labels:
        return []
    fig = ax.figure
    for text in labels:
        text.set_fontsize(4.5)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    rows = []
    anchors = {}
    assignments = {}
    for text in sorted(labels, key=lambda t: t.get_position()[0]):
        anchors[text] = text.get_position()[0]
        box = text.get_window_extent(renderer)
        # Clamp horizontally to the axes before assigning a collision row.
        dx = max(ax.bbox.x0+2-box.x0, min(0, ax.bbox.x1-2-box.x1))
        x, y = text.get_position()
        text.set_x(ax.transData.inverted().transform((ax.transData.transform((x, y))[0]+dx, 0))[0])
        box = text.get_window_extent(renderer)
        row = next((i for i, right in enumerate(rows) if box.x0 > right+2), len(rows))
        if row == len(rows):
            rows.append(box.x1)
        else:
            rows[row] = box.x1
        assignments[text] = row
    pitch = max(t.get_window_extent(renderer).height for t in labels)+2
    needed = len(rows)*pitch+8
    bottom, top = ax.get_ylim()
    if needed >= ax.bbox.height*.8:
        raise ValueError('Controller label band exceeds available panel height')
    extra = (top-bottom)*needed/(ax.bbox.height-needed)
    threshold = ctrl_y-.5
    # Original renderer uses data-coordinate paths for bars and horizontal lines.
    for collection in ax.collections:
        for path in collection.get_paths():
            ys = path.vertices[:, 1]
            ys[ys >= threshold] += extra
    for line in ax.lines:
        if line.get_transform() == ax.transData:
            line.set_ydata([y+extra if y >= threshold else y for y in line.get_ydata()])
    for text in ax.texts:
        if text.get_transform() == ax.transData:
            x, y = text.get_position()
            text.set_position((x, y+extra if y >= threshold else y))
    ticks = ax.get_yticks()
    ticklabels = [t.get_text() for t in ax.get_yticklabels()]
    ax.set_yticks([y+extra if y >= threshold else y for y in ticks], ticklabels, fontsize=5)
    ax.set_ylim(bottom, top+extra)
    scale = ax.bbox.height/(top+extra-bottom)
    for text in labels:
        x, _ = text.get_position()
        y = ctrl_y+extra-.48-assignments[text]*pitch/scale
        text.set_position((x, y))
        text.set_clip_on(True)
        ax.plot([anchors[text], x], [ctrl_y+extra-.25, y], color='.5', alpha=.3,
                linewidth=.35, zorder=1)
    fig.canvas.draw()
    boxes = [t.get_window_extent(fig.canvas.get_renderer()) for t in labels]
    if any(a.overlaps(b) for i, a in enumerate(boxes) for b in boxes[i+1:]):
        raise ValueError('Overlapping ready-range labels after layout')
    if any(b.x0 < ax.bbox.x0 or b.x1 > ax.bbox.x1 or b.y0 < ax.bbox.y0 or b.y1 > ax.bbox.y1 for b in boxes):
        raise ValueError('Ready-range labels outside axes')
    return labels


def render_detail(comparison, events, wg_map, renderer, out):
    minimum = min(r['time_s'] for r in events)
    coll = {r['workgroup']: r for r in comparison['collective']}
    overlays = {p['controller']: [(x-minimum*1000, label, '--', color)
                for x, label, color in zip(references(coll[p['workgroup']]),
                    REF_LABELS, REF_COLORS)]
                for p in comparison['p2p']}
    original_savefig = Figure.savefig

    def save(figure, *args, **kwargs):
        # Compact only this adapter's saved canvas, never the external renderer.
        original_height = figure.get_figheight()
        height = min(original_height*.5, 1.5+.12*sum(map(len, wg_map.values())))
        figure.set_size_inches(12, height)
        figure._suptitle.set_text(title(comparison))
        figure._suptitle.set_fontsize(10)
        figure._suptitle.set_position((.5, 1-.04/height))
        for ax, ctrl in zip(figure.axes, sorted(wg_map)):
            p = base.unique([r for r in comparison['p2p'] if r['controller'] == ctrl], 'controller')
            ax.set_ylabel(f"WG {p['workgroup']}\nController r{ctrl}", fontsize=5, fontweight='normal')
            labels = [f'r{w} [GPU Ctrl]' if w == ctrl else f'r{w} [Worker]' for w in wg_map[ctrl]]
            ax.set_yticks(range(len(labels)), labels)
            for tick in ax.get_yticklabels():
                tick.set_color('#d62728' if 'Ctrl' in tick.get_text() else '#333333')
                tick.set_fontsize(5)
            for line in ax.lines:
                if line.get_marker() == 'v':
                    line.set_markersize(2.5)
            controller = [r for r in events if r['world_rank'] == ctrl]
            for y, world in enumerate(wg_map[ctrl]):
                if world == ctrl:
                    continue
                worker = [r for r in events if r['world_rank'] == world]
                local = worker[0]['workgroup_rank']
                sent = rank.controller_endpoint(controller, 'result_send_start', local)
                received = endpoint(worker, 'result_received')
                if received < sent:
                    ax.plot((received-minimum)*1000, y+.17, 'x', color='#ad322b', markersize=5)
            ax.set_xlim(right=max(ax.get_xlim()[1], max(x[0] for x in overlays[ctrl])*1.02))
            ax.axvline(-minimum*1000, color='.35', linewidth=.7)
            left, right = ax.get_xlim()
            ticks = MaxNLocator(nbins=7).tick_values(left+minimum*1000, right+minimum*1000)
            ax.set_xticks(ticks-minimum*1000)
            ax.set_xlim(left, right)
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x+minimum*1000:g}'))
        figure.axes[-1].set_xlabel("ms from each WG's own controller ml_step_start (independent zero; estimated alignment)", fontsize=8)
        short_labels = {
            'worker input send -> controller-ready (incl. ack)': 'Input + ACK',
            'worker waits for input credit': 'Credit wait',
            'controller result-send -> worker output received': 'Output',
            'selected ready-range inference': 'Host inference',
            'controller sends input credit to this worker': 'Credit send',
            'Torch forward execution': 'Torch forward',
        }
        for legend in list(figure.legends):
            handles = getattr(legend, 'legend_handles', None)
            if handles is None:
                handles = legend.legendHandles
            labels = [short_labels.get(t.get_text(), t.get_text().replace('controller ', '').replace('Torch ', ''))
                      for t in legend.get_texts()]
            kept = [(h, t) for h, t in zip(handles, labels) if 'Worker (' not in t]
            for handle, label in kept:
                if label == 'Credit send':
                    handle.set_markersize(2.5)
            legend.remove()
            figure.legend([h for h, t in kept], [t for h, t in kept], loc='upper center',
                          bbox_to_anchor=(.55, 1-.28/height), ncol=5, fontsize=6,
                          frameon=False, columnspacing=.8, labelspacing=.25, handlelength=1.2)
        figure.canvas.draw()
        legend_bottom = min((legend.get_window_extent(figure.canvas.get_renderer()).y0
                             for legend in figure.legends), default=figure.bbox.height-.5*figure.dpi)
        top = (figure.bbox.height-legend_bottom)/figure.dpi+.08
        figure.subplots_adjust(left=.12, right=.99, bottom=.55/height, top=1-top/height, hspace=.12)
        for ax, ctrl in zip(figure.axes, sorted(wg_map)):
            stagger_range_labels(ax, wg_map[ctrl].index(ctrl))
        with patch.object(Figure, 'savefig', original_savefig):
            save_cropped(figure, out, comparison['name']+'_terrain_detail')

    with patch.object(renderer, 'get_workgroup_mapping', return_value=wg_map), \
            patch.object(renderer, 'synchronize_step_clocks', side_effect=lambda rows, mapping: rows), \
            patch.object(Figure, 'savefig', save):
        renderer.render_step(events, comparison['p2p'][0]['call'], out,
                             comparison['p2p'][0]['model'], overlays=overlays)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag, choices in (('models', ['transformer', 'mlp']), ('placements', ['24c1g', '24p24c1g', '96c4g', '96p96c4g']),
                          ('layouts', ['default', 'numa']), ('modes', ['p2p1', 'p2pfull'])):
        flags = ['--placements', '--placement'] if flag == 'placements' else ['--'+flag]
        parser.add_argument(*flags, nargs='+', choices=choices, default=choices)
    parser.add_argument('--rep', type=int, default=1)
    parser.add_argument('--call', type=int, choices=range(7), default=6)
    parser.add_argument('--renderer', type=Path, default=RENDERER)
    parser.add_argument('--detailed-all', action='store_true', help='Render all details (already the default)')
    parser.add_argument('--out-dir', type=Path, default=base.ROOT / 'figures/aix_terrain_style_draft')
    args = parser.parse_args(argv)
    renderer = load_renderer(args.renderer)
    comparisons, metrics, _, metadata = base.collect(args.models, args.placements, args.layouts, args.modes, args.rep, args.call)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    alignment, envelopes, gallery = [], [], []
    for comparison in comparisons:
        events, mapping, offsets, windows = prepare(comparison, metadata, renderer)
        alignment.extend(offsets)
        envelopes.extend(windows)
        render_envelopes(comparison, windows, args.out_dir)
        names = [comparison['name']+'_phase_envelopes']
        render_detail(comparison, events, mapping, renderer, args.out_dir)
        names.append(comparison['name']+'_terrain_detail')
        gallery.extend(names)
    for name, rows in (('alignment', alignment), ('phase_envelopes', envelopes), ('workgroup_metrics', metrics)):
        base.write_csv(args.out_dir / (name+'.csv'), rows)
    source_root = args.renderer.parent.parent / 'extern/AIxeleratorService/src'
    for path in (Path(__file__), Path(__file__).with_name('test_plot_cmi_aix_terrain_style.py'), Path(rank.__file__), args.renderer, args.renderer.with_name('analyze_p2p_inference_phases.py'),
                 source_root / 'communicationStrategy/collectiveCommunication.cpp',
                 source_root / 'utils/p2pTimeline.h'):
        if not path.exists():
            continue
        metadata['sources'][str(path.resolve())] = base.sha256(path)
    command = shlex.join(['python', 'scripts/plot_cmi_aix_terrain_style.py', *(sys.argv[1:] if argv is None else argv)])
    metadata.update(clock_caveats=CAVEAT, semantics=SEMANTICS, command=command,
                    envelope_figures=len(comparisons), detailed_figures=len(gallery)-len(comparisons),
                    figure_files=2*len(gallery), alignment_rows=len(alignment), envelope_rows=len(envelopes))
    metadata['invalid_aligned_output_pairs'] = sum(not r['output_span_valid'] for r in alignment)
    metadata['causal_feasible_pairs'] = sum(r['tightened_causal_feasible'] for r in alignment)
    report = ['# Output Causality Audit', '',
        'Read-only analysis of raw timestamps; no offsets or durations are changed in the figures.', '',
        '## Event Evidence', '',
        'Source: `extern/AIxeleratorService/src/communicationStrategy/collectiveCommunication.cpp` '
        '(external renderer repository), lines 239-258, 323-331, 360-377, 398-409. '
        'Worker send start follows credit receipt; input send complete follows blocking input ACK receipt; '
        'result received follows blocking output receipt. Controller input ready precedes ACK Isend, '
        'and result send start precedes output Isend. Thus input ready and send complete are not simultaneous.', '',
        '`src/utils/p2pTimeline.h:16-21,90-106` uses MPI_Wtime plus optional '
        'AIX_P2P_CLOCK_OFFSET_SECONDS, sampled before taking the logging mutex. MPI_Wtime does not '
        'establish a common rank clock. No measured clock synchronization or drift estimate is available. '
        'Source inspection describes the available source, not independent proof of the exact binary used.', '',
        '## Bounds', '',
        'With controller time = worker time + delta, input causality gives '
        '`L=max(credit-send_start, ready-send_complete)`, `U=ready-send_start`. '
        'Output causality adds `delta >= result_send-result_received`; tightened '
        '`L2=max(L, result_send-result_received)`. Feasibility is `L2 <= U`. '
        'These are per-worker, per-call necessary causal bounds, not a measured synchronization; '
        'a midpoint is not used. Equality at ACK is only one component of the input lower bound, '
        'and can even violate the credit lower bound. It assumes zero ACK latency.', '',
        f'Workers: {len(alignment)}; negative ACK-aligned output spans: '
        f'{metadata["invalid_aligned_output_pairs"]}; feasible tightened intervals: '
        f'{metadata["causal_feasible_pairs"]}; infeasible: '
        f'{len(alignment)-metadata["causal_feasible_pairs"]}.', '',
        f'Validated raw same-clock worker and controller endpoint orders: {len(alignment)}; '
        f'ACK offsets satisfying all input bounds: {sum(r["ack_satisfies_input_bounds"] for r in alignment)}.', '']
    invalid = [r for r in alignment if not r['output_span_valid']]
    feasible = [r for r in alignment if r['tightened_causal_feasible']]
    for label, rows in [('All workers', feasible), ('Negative ACK-span workers', [r for r in invalid if r['tightened_causal_feasible']])]:
        if rows:
            widths = sorted(r['tightened_causal_width_ms'] for r in rows)
            report.append(f'{label}: tightened width min/max = {widths[0]:.9g}/{widths[-1]:.9g} ms.')
    report.extend(['', 'All negative cases below preserve raw same-clock durations. The signed span equals '
                   'worker ACK-to-receipt minus controller ready-to-send. Feasible intervals show that '
                   'these cases do not require a measured communication-order bug; ACK equality is '
                   'incompatible with output causality. They do not prove an exact offset or rule out other bugs.', '',
                   '| Comparison | WG | Worker | ACK-to-receipt ms | Ready-to-send ms | Signed output ms | L2 s | U s | Width ms | Feasible |',
                   '|---|---:|---:|---:|---:|---:|---:|---:|---:|---|'])
    for r in invalid:
        report.append(f'| {r["comparison"]} | {r["workgroup"]} | {r["world_rank"]} | '
                      f'{r["worker_ack_to_receipt_ms"]:.9g} | {r["controller_ready_to_send_ms"]:.9g} | '
                      f'{r["aligned_output_span_ms"]:.9g} | {r["tightened_causal_low_s"]:.12g} | '
                      f'{r["causal_high_s"]:.12g} | {r["tightened_causal_width_ms"]:.9g} | '
                      f'{r["tightened_causal_feasible"]} |')
    (args.out_dir / 'output_causality_report.md').write_text('\n'.join(report)+'\n')
    (args.out_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    lines = ['# AIx Terrain Style Draft', '', CAVEAT, '', '## Reproduce', '', '```sh', command, '```', '',
             'Narrative notes and statistics are README/caption-only in every graphic version. '
             'Axes retain independent-zero and estimated-alignment semantics; red crosses retain invalid receipt flags. '
             'Saved canvases crop unused bottom space without stretching axes.', '',
             f'Output: {len(comparisons)} timeline pairs (envelope/detail), '
             f'{len(gallery)} PNG and {len(gallery)} PDF files; no PGF. '
             f'{len(alignment)} workers; {metadata["invalid_aligned_output_pairs"]} invalid ACK-aligned output pairs; '
             f'{metadata["causal_feasible_pairs"]} feasible tightened causal intervals.', '',
             '## Semantics', '', SEMANTICS, '', 'All four placements get envelopes and details by default: '
             '24c1g, 24p24c1g, 96c4g, 96p96c4g (32 comparisons, 32 envelopes and 32 details, each PNG/PDF). '
             'Use --placement (alias --placements) to select placements. Details retain original lanes at save-time height '
              'min(original height / 2, 1.5 + 0.12 inches per rank), with uniform 5 pt rank labels. '
              'Ready-range labels are 4.5 pt, allocated by rendered bounding boxes into staggered rows '
              'in a reserved controller band with faint leaders. Legend space uses its actual bounding box. '
             'At up to 192 ranks, view the vector PDF at full page height to read individual ranks; a thesis-width reduction cannot preserve every label.', '',
              '[Alignment](alignment.csv) | [Causality audit](output_causality_report.md) | [Phase envelopes](phase_envelopes.csv) | [Metrics](workgroup_metrics.csv) | [Source hashes](metadata.json)', '', '## Gallery', '']
    for name in gallery:
        lines.extend([f'### {name}', '', f'[PDF]({name}.pdf)', f'![{name}]({name}.png)', ''])
    (args.out_dir / 'README.md').write_text('\n'.join(lines)+'\n')
    print(f"Rendered {len(comparisons)} envelopes, {len(gallery)-len(comparisons)} details, {2*len(gallery)} PNG/PDF files; {len(alignment)} workers")


if __name__ == '__main__':
    main()
