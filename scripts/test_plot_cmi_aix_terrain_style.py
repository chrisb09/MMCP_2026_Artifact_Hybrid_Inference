"""Adapter contracts and a real four-placement render smoke test."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import plot_cmi_aix_terrain_style as terrain
from PIL import Image


class TerrainTests(unittest.TestCase):
    def test_crop_preserves_axes_and_physical_scale(self):
        fig, ax = terrain.base.plt.subplots(figsize=(6, 4))
        fig.subplots_adjust(bottom=.3)
        ax.set_xlabel('independent zero; estimated alignment')
        fig.canvas.draw()
        before = ax.bbox.bounds
        with patch.object(terrain.Figure, 'savefig') as savefig:
            terrain.save_cropped(fig, Path('.'), 'synthetic')
        self.assertEqual(savefig.call_count, 2)
        bounds = savefig.call_args.kwargs['bbox_inches']
        self.assertGreater(bounds.y0, 0)
        self.assertEqual((bounds.x0, bounds.x1, bounds.y1), (0, 6, 4))
        self.assertEqual(ax.bbox.bounds, before)
        self.assertAlmostEqual(ax.get_tightbbox(fig.canvas.get_renderer()).y0/fig.dpi-bounds.y0, .08)

    def test_published_data_hashes_unchanged(self):
        expected = {
            'alignment.csv': '0e50b542283cdba1f52da5d775c31c3e6af14a5a5a6fe5d39d79a720eb4e9eb2',
            'phase_envelopes.csv': 'e88b845d1fa257248a3f0665576b7597e3570f5220720ca42c1feee0e817ec9e',
            'workgroup_metrics.csv': 'b39f3cacb167d7a73773001dd15d40f53b007c8b021c0d3b609add4d63cbd6eb',
        }
        for name, digest in expected.items():
            self.assertEqual(terrain.base.sha256(terrain.base.ROOT/'figures/aix_terrain_style_draft'/name), digest)

    def test_stagger_real_display_boxes_and_controller_band(self):
        fig, ax = terrain.base.plt.subplots(figsize=(6, 3))
        ax.set_xlim(0, 100)
        ax.set_ylim(-.8, 9.8)
        ax.set_yticks(range(10), [f'r{i}' for i in range(10)])
        for x in (0, .1, .2, .3, 50, 100):
            ax.text(x, 4-.48, '12-34', ha='center', va='top', fontsize=12)
        ax.broken_barh([(0, 10)], (4-.24, .22))
        worker_bar = ax.broken_barh([(0, 10)], (3+.04, .27))
        labels = terrain.stagger_range_labels(ax, 4)
        boxes = [t.get_window_extent(fig.canvas.get_renderer()) for t in labels]
        self.assertEqual(len(labels), 6)
        self.assertTrue(all(t.get_fontsize() == 4.5 for t in labels))
        self.assertGreater(len({t.get_position()[1] for t in labels}), 1)
        self.assertTrue(all(not a.overlaps(b) for i, a in enumerate(boxes) for b in boxes[i+1:]))
        self.assertTrue(all(b.y0 > ax.transData.transform((0, 3.8))[1] for b in boxes))
        self.assertTrue(all(ax.bbox.contains(b.x0, b.y0) and ax.bbox.contains(b.x1, b.y1) for b in boxes))
        ys = worker_bar.get_paths()[0].vertices[:, 1]
        self.assertAlmostEqual(max(ys)-min(ys), .27)
        terrain.base.plt.close(fig)

    def test_reference_durations_only(self):
        self.assertEqual(terrain.references(dict(gather_collective_ms=2, device_cpu_ms=3,
                                                scatter_collective_ms=4)), (2, 5, 9))
        with self.assertRaises(ValueError):
            terrain.references(dict(gather_collective_ms=-1, device_cpu_ms=3, scatter_collective_ms=4))

    def test_endpoint_requires_unique(self):
        self.assertEqual(terrain.endpoint([dict(event='x', time_s='2')], 'x'), 2)
        for rows in ([], [dict(event='x', time_s=1)]*2):
            with self.assertRaises(ValueError):
                terrain.endpoint(rows, 'x')

    def test_original_anchor_non_modulo_mapping(self):
        renderer = terrain.load_renderer(terrain.RENDERER)
        events = [dict(world_rank=7, workgroup_rank=0, is_controller=1,
                       event='input_ready', peer_workgroup_rank=1, time_s=10),
                  dict(world_rank=2, workgroup_rank=1, is_controller=0,
                       event='input_send_complete', peer_workgroup_rank=0, time_s=30),
                  dict(world_rank=9, workgroup_rank=0, is_controller=1,
                       event='input_ready', peer_workgroup_rank=1, time_s=100),
                  dict(world_rank=4, workgroup_rank=1, is_controller=0,
                       event='input_send_complete', peer_workgroup_rank=0, time_s=60)]
        aligned = renderer.synchronize_step_clocks(events, {7: [2, 7], 9: [4, 9]})
        self.assertEqual([r['time_s'] for r in aligned], [10, 10, 100, 100])
        self.assertEqual(events[1]['time_s'], 30)

    def test_real_smoke_four_placements(self):
        if not terrain.RENDERER.exists():
            self.skipTest('External original renderer unavailable')
        with tempfile.TemporaryDirectory() as directory:
            original_save = terrain.base.save
            def inspect_save(fig, out, name):
                self.assertIn('MLP (ov4)', fig._suptitle.get_text())
                self.assertEqual(fig.texts, [fig._suptitle])
                self.assertIn('estimated', fig.axes[-1].get_xlabel())
                self.assertFalse(any('invalid output pairs' in t.get_text() or 'violate ACK' in t.get_text()
                                     for ax in fig.axes for t in ax.texts))
                if name.endswith('_phase_envelopes'):
                    self.assertEqual(len(fig.axes), 1)
                    self.assertLessEqual(fig.get_figheight(), 5.1)
                    self.assertLess(fig.axes[0].get_xlim()[0], 0)
                    self.assertIn('independent zero', fig.axes[0].get_xlabel())
                else:
                    ranks = sum(len(ax.get_yticklabels()) for ax in fig.axes)
                    self.assertLessEqual(fig.get_figheight(), max(4, 1.2+.28*ranks)/2)
                    self.assertLessEqual(fig.get_figheight(), 1.5+.12*ranks)
                    self.assertTrue(all(t.get_fontsize() == 5 for ax in fig.axes
                                        for t in ax.get_yticklabels()))
                    self.assertTrue(all(ax.yaxis.label.get_fontsize() == 5 for ax in fig.axes))
                    self.assertLess((1-fig.subplotpars.top)*fig.get_figheight(), .75)
                    self.assertTrue(all(line.get_markersize() == 2.5 for ax in fig.axes
                                        for line in ax.lines if line.get_marker() == 'v'))
                    for ax in fig.axes:
                        formatter = ax.xaxis.get_major_formatter()
                        self.assertLess(float(formatter(0)), 0)
                    self.assertFalse(any('Local Worker' in t.get_text() or 'Remote Worker' in t.get_text()
                                         for legend in fig.legends for t in legend.get_texts()))
                original_save(fig, out, name)
                with Image.open(out/(name+'.png')) as image:
                    self.assertEqual(image.width, 3000)
                    self.assertLess(image.height, int(fig.get_figheight()*250))
            with patch.object(terrain.base, 'save', side_effect=inspect_save):
                terrain.main(['--models', 'mlp', '--layouts', 'default', '--modes', 'p2p1',
                              '--out-dir', directory])
            out = Path(directory)
            metadata = json.loads((out/'metadata.json').read_text())
            self.assertEqual(metadata['envelope_figures'], 4)
            self.assertEqual(metadata['detailed_figures'], 4)
            self.assertEqual(len(list(out.glob('*.png'))), 8)
            self.assertEqual(len(list(out.glob('*.pdf'))), 8)
            self.assertEqual(metadata['alignment_rows'], 23+47+92+188)
            rows = terrain.base.read_csv(out/'phase_envelopes.csv')
            self.assertEqual(len(rows), 30)
            self.assertTrue(all(float(r['duration_ms']) >= 0 for r in rows))
            self.assertTrue(any(float(r['renderer_origin_relative_s']) < 0 for r in rows))
            self.assertTrue(any(float(r['start_ms']) < 0 for r in rows))
            self.assertEqual(metadata['causal_feasible_pairs'], metadata['alignment_rows'])
            self.assertIn('cannot diagnose early arrival', (out/'README.md').read_text())
            for path, digest in metadata['sources'].items():
                self.assertEqual(terrain.base.sha256(terrain.base.ROOT / path), digest)

    def test_placement_alias(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(terrain, 'load_renderer'), \
                patch.object(terrain.base, 'write_csv'), \
                patch.object(terrain.base, 'collect', return_value=([], [], [], {'sources': {}})) as collect:
            terrain.main(['--placement', '24p24c1g', '--out-dir', directory])
            self.assertEqual(collect.call_args.args[1], ['24p24c1g'])

    def test_model_titles(self):
        for model, expected in [('mlp', 'MLP (ov4)'), ('transformer', 'Transformer (ov0)')]:
            with patch.object(terrain.base, 'title', return_value=model+' | placement'):
                self.assertEqual(terrain.title({'p2p': [{'model': model}]}), expected+' | placement')

    def test_detail_calls_original_and_shifts_refs(self):
        renderer = terrain.load_renderer(terrain.RENDERER)
        comparison = dict(name='synthetic', p2p=[dict(controller=8, workgroup=3, call=6, model='mlp')],
                          collective=[dict(workgroup=3, gather_collective_ms=2, device_cpu_ms=3, scatter_collective_ms=4)])
        events = [dict(time_s=-.001)]
        def check(rows, step, out, model, overlays):
            self.assertIs(rows, events)
            self.assertEqual(renderer.get_workgroup_mapping(rows), {8: [2, 8]})
            self.assertIs(renderer.synchronize_step_clocks(rows, {}), rows)
            self.assertEqual([r[0] for r in overlays[8]], [3, 6, 10])
        with patch.object(renderer, 'render_step', side_effect=check) as render:
            terrain.render_detail(comparison, events, {8: [2, 8]}, renderer, Path('.'))
        render.assert_called_once()

    def test_real_anchor_failure_is_exported_not_clamped(self):
        if not terrain.RENDERER.exists():
            self.skipTest('External original renderer unavailable')
        comparisons, _, _, metadata = terrain.base.collect(
            ['mlp'], ['96p96c4g'], ['numa'], ['p2pfull'])
        renderer = terrain.load_renderer(terrain.RENDERER)
        events, mapping, offsets, envelopes = terrain.prepare(comparisons[0], metadata, renderer)
        self.assertEqual(len(offsets), 188)
        self.assertTrue(any(not r['output_span_valid'] for r in offsets))
        self.assertTrue(all(r['duration_ms'] >= 0 for r in envelopes))
        for ctrl in mapping:
            self.assertEqual(terrain.endpoint([r for r in events if r['world_rank'] == ctrl], 'ml_step_start'), 0)
        for row in offsets:
            worker = [r for r in events if r['world_rank'] == row['world_rank']]
            controller = [r for r in events if r['world_rank'] == row['controller']]
            ready = terrain.rank.controller_endpoint(controller, 'input_ready', row['workgroup_rank'])
            self.assertAlmostEqual(terrain.endpoint(worker, 'input_send_complete'), ready)
            self.assertEqual(row['output_span_valid'], row['aligned_output_span_ms'] >= 0)
            self.assertTrue(row['tightened_causal_feasible'])
            self.assertAlmostEqual(row['aligned_output_span_ms'],
                                   row['worker_ack_to_receipt_ms']-row['controller_ready_to_send_ms'])
        def inspect(fig, out, name):
            self.assertEqual(fig.texts, [fig._suptitle])
            crosses = [line for ax in fig.axes for line in ax.lines
                       if line.get_marker() == 'x' and line.get_color() == '#ad322b']
            invalid = [r for r in offsets if not r['output_span_valid']]
            self.assertEqual(len(crosses), len(invalid))
            minimum = min(r['time_s'] for r in events)
            expected = sorted((r['worker_result_received_s']+r['worker_offset_s']
                               -r['controller_origin_s']-minimum)*1000 for r in invalid)
            for actual, wanted in zip(sorted(line.get_xdata()[0] for line in crosses), expected):
                self.assertAlmostEqual(actual, wanted)
            terrain.base.plt.close(fig)
        with patch.object(terrain.base, 'save', side_effect=inspect):
            terrain.render_detail(comparisons[0], events, mapping, renderer, Path('.'))


if __name__ == '__main__':
    unittest.main()
