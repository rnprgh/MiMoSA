"""Cross-run and cross-strain comparisons for exported MiMoSA statistics."""
from __future__ import annotations

import os
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import is_color_like
from matplotlib.lines import Line2D
from matplotlib.text import Text
from scipy.stats import gaussian_kde, norm


@dataclass
class _DatasetRecord:
    """Loaded files and provenance for one notebook/run."""

    dataset_id: str
    strain: str
    source: str
    dataframes: dict[str, pd.DataFrame]
    paths: dict[str, str]


class ComparativeStats:
    """
    Combine exported statistics from multiple runs and compare strains.

    Load each notebook/run separately so run identity is preserved::

        comparison = ComparativeStats()
        comparison.load_dataframes(run_1_folder, 'BD9422')
        comparison.load_dataframes(run_2_folder, 'BD9422')
        comparison.load_dataframes(run_3_folder, 'BD9571')
    """

    ANGULAR_MSD_DR_PARAMETER = 'Dr (rad²/s)'

    ANALYSIS_PLOT_DEFAULTS = {
        'title': None,
        'title_fontsize': None,
        'x_axis_fontsize': None,
        'y_axis_fontsize': None,
        'x_tick_fontsize': None,
        'y_tick_fontsize': None,
        'x_axis_titles': None,
        'y_axis_titles': None,
        'strain_colors': None,
        'font_family': 'Aptos',
        'dpi': 300,
    }

    DATAFRAME_FILES = {
        'fitted_mean_speeds': (
            '05_Basic_Stats/01_Fitted_Mean_Speed_Distribution.csv',
        ),
        'particle_characteristics': (
            '06_Additional_Analysis/01_Particle_Characteristics.csv',
        ),
        'turn_summary': (
            '06_Additional_Analysis/02_Turn_Summary.csv',
        ),
        'turn_angles': (
            '06_Additional_Analysis/03_Turn_Angles.csv',
        ),
        'angle_tumble_summary': (
            '06_Additional_Analysis/04_Angle_Tumble_Summary.csv',
        ),
        'angle_tumble_events': (
            '06_Additional_Analysis/05_Angle_Tumble_Events.csv',
        ),
        'angle_tumble_points': (
            '06_Additional_Analysis/06_Angle_Tumble_Points.csv',
        ),
        'angle_tumble_population': (
            '06_Additional_Analysis/07_Angle_Tumble_Population.csv',
        ),
        'angle_tumble_table_1': (
            '06_Additional_Analysis/07_Angle_Tumble_Table_1.csv',
        ),
        'velocity_tumble_summary': (
            '06_Additional_Analysis/08_Velocity_Tumble_Summary.csv',
        ),
        'velocity_tumble_events': (
            '06_Additional_Analysis/09_Velocity_Tumble_Events.csv',
        ),
        'velocity_tumble_points': (
            '06_Additional_Analysis/10_Velocity_Tumble_Points.csv',
        ),
        'velocity_tumble_population': (
            '06_Additional_Analysis/11_Velocity_Tumble_Population.csv',
        ),
        'velocity_tumble_table_1': (
            '06_Additional_Analysis/11_Velocity_Tumble_Table_1.csv',
        ),
        'individual_msd': (
            '06_Additional_Analysis/14_Individual_MSD.csv',
        ),
        'ensemble_msd': (
            '06_Additional_Analysis/15_Ensemble_MSD.csv',
        ),
        'msd_fit': (
            '06_Additional_Analysis/16_MSD_Fit.csv',
        ),
    }

    REQUIRED_COLUMNS = {
        'fitted_mean_speeds': {'mean_speed'},
        'particle_characteristics': {
            'particle', 'mean_major_axis_length', 'mean_minor_axis_length',
        },
        'turn_summary': {
            'particle', 'number_of_turns',
            'mean_absolute_turn_angle_degrees', 'total_path_length',
            'observed_duration_seconds', 'turns_per_distance',
            'turns_per_second',
        },
        'angle_tumble_summary': {
            'particle', 'number_of_runs', 'number_of_tumbles',
            'analyzed_tracking_time_seconds', 'tumble_frequency_per_second',
            'mean_run_speed', 'mean_tumble_speed',
            'mean_run_interval_seconds', 'mean_tumble_interval_seconds',
            'mean_run_angular_speed_degrees_per_point',
            'mean_tumble_angular_speed_degrees_per_point',
            'mean_direction_change_within_runs_degrees',
            'mean_run_to_run_direction_change_degrees',
        },
        'angle_tumble_events': {
            'particle', 'segment_id', 'event_type', 'interval_seconds',
        },
        'angle_tumble_points': {
            'particle', 'segment_id', 'frame', 'state',
            'instantaneous_speed',
        },
        'angle_tumble_population': {'particle_count'},
        'angle_tumble_table_1': {'Parameter', 'Value'},
        'velocity_tumble_summary': {
            'particle', 'v_R', 'v_R_support_seconds', 'v_T',
            'v_T_support_seconds', 't_R', 't_R_interval_count', 't_T',
            't_T_interval_count', 'p', 'p_direction_change_count', 'R',
            'R_run_transition_count',
        },
        'velocity_tumble_events': {
            'particle', 'segment_id', 'run_to_run_turn_angle_degrees',
            'run_to_run_directional_cosine',
        },
        'velocity_tumble_points': {
            'particle', 'segment_id', 'frame', 'state', 'speed',
            'elapsed_time_from_previous_seconds', 'elapsed_time_seconds',
            'event_id', 'smoothed_position_x_pixels',
            'smoothed_position_y_pixels',
            'angular_velocity_magnitude_radians_per_second',
        },
        'velocity_tumble_population': {
            'particle_count', 'v_R', 'v_R_support_seconds', 'v_T',
            'v_T_support_seconds', 't_R', 't_R_interval_count', 't_T',
            't_T_interval_count', 'p', 'p_direction_change_count', 'R',
            'R_run_transition_count',
        },
        'velocity_tumble_table_1': {'Parameter', 'Value'},
        'individual_msd': {'particle', 'lag_time', 'individual_msd'},
        'ensemble_msd': {'lag_time', 'ensemble_msd'},
        'msd_fit': {'max_lag_time_frames', 'time_unit', 'msd_unit'},
    }

    PARAMETER_COLUMNS = {
        'fitted_mean_speeds': (
            'source_dataframe', 'speed_unit', 'discard_initial_frames',
            'discard_final_frames', 'distribution_type', 'fit_range_source',
            'requested_fit_range_start', 'requested_fit_range_end',
            'ci_range_lower_percentile', 'ci_range_upper_percentile',
            'capture_speed_in_fps', 'pixel_scale_factor', 'scale_units',
        ),
        'particle_characteristics': (
            'source_dataframe', 'length_unit', 'discard_initial_frames',
            'discard_final_frames',
        ),
        'turn_summary': (
            'source_dataframe', 'distance_unit',
            'minimum_turn_angle_degrees', 'smoothing_method',
            'moving_average_window', 'triangular_smoothing_window',
            'sg_filter_window_length', 'sg_filter_polyorder',
            'rdp_epsilon_pixels', 'max_frame_gap',
            'discard_initial_frames', 'discard_final_frames',
        ),
        'angle_tumble_summary': (
            'source_dataframe', 'distance_unit', 'speed_unit',
            'smoothing_method', 'moving_average_window',
            'triangular_smoothing_window', 'sg_filter_window_length',
            'sg_filter_polyorder', 'rdp_epsilon_pixels',
            'discard_initial_frames', 'discard_final_frames',
            'tumble_threshold_angle_degrees', 'max_frame_gap',
        ),
        'velocity_tumble_summary': (
            'source_dataframe', 'distance_unit', 'speed_unit',
            'angular_velocity_unit', 'smoothing_method',
            'moving_average_window', 'triangular_smoothing_window',
            'sg_filter_window_length', 'sg_filter_polyorder',
            'rdp_epsilon_pixels', 'discard_initial_frames',
            'discard_final_frames', 'speed_drop_ratio_threshold',
            'speed_period_depth_fraction', 'angular_change_coefficient',
            'minimum_heading_speed', 'speed_extrema_prominence',
            'angular_velocity_extrema_prominence', 'extrema_min_distance',
            'persistence_interval_seconds', 'run_direction_fit_points',
            'max_frame_gap',
        ),
        'msd_fit': (
            'source_dataframe', 'max_lag_time_frames',
            'discard_initial_frames', 'discard_final_frames', 'time_unit',
            'msd_unit', 'requested_alpha_fit_lag_start',
            'requested_alpha_fit_lag_end', 'capture_speed_in_fps',
            'pixel_scale_factor', 'scale_units',
        ),
    }

    COMPARISON_PARAMETER_KEYS = {
        'angle': ('angle_tumble_summary', 'particle_characteristics'),
        'velocity': ('velocity_tumble_summary',),
        'angular_msd': (
            'velocity_tumble_table_1', 'velocity_tumble_summary',
        ),
        'speed': ('fitted_mean_speeds',),
        'turn': ('turn_summary',),
        'run_tumble_plot': (
            'velocity_tumble_summary', 'velocity_tumble_points',
            'velocity_tumble_events',
        ),
        'msd': ('msd_fit',),
    }

    COMPARISON_REQUIRED_DATAFRAMES = {
        'angle': ('angle_tumble_summary', 'particle_characteristics'),
        'velocity': ('velocity_tumble_summary',),
        'angular_msd': ('velocity_tumble_table_1',),
        'speed': ('fitted_mean_speeds',),
        'turn': ('turn_summary',),
        'run_tumble_plot': (
            'velocity_tumble_summary', 'velocity_tumble_points',
            'velocity_tumble_events',
        ),
        'msd': ('msd_fit', 'individual_msd'),
    }

    VALIDATION_COLUMNS = [
        'severity', 'status', 'dataframe', 'parameter', 'datasets', 'values',
        'message',
    ]

    def __init__(self) -> None:
        """Initialize an empty multi-run comparison collection."""
        self._datasets: list[_DatasetRecord] = []
        self._strain_validation_rows: list[dict] = []
        self._parameter_validation_dataframe = pd.DataFrame(
            columns=self.VALIDATION_COLUMNS
        )
        self._angle_comparison_dataframe = pd.DataFrame()
        self._velocity_comparison_dataframe = pd.DataFrame()
        self._fitted_speed_observations_dataframe = pd.DataFrame()
        self._fitted_speed_comparison_dataframe = pd.DataFrame()
        self._fitted_speed_export_dataframe = pd.DataFrame()
        self._turn_comparison_dataframe = pd.DataFrame()
        self._run_tumble_plot_data: dict[str, pd.DataFrame] = {}
        self._angular_msd_comparison_dataframe = pd.DataFrame()
        self._msd_comparison_dataframe = pd.DataFrame()
        self._analysis_plotting_parameters = dict(
            self.ANALYSIS_PLOT_DEFAULTS
        )

    def load_dataframes(
        self,
        dataset_path: str | os.PathLike | Mapping[str, str | os.PathLike] |
        Sequence[str | os.PathLike],
        strain: str,
        dataset_name: str | None = None,
    ) -> dict[str, pd.DataFrame]:
        """
        Load one exported notebook/run and associate it with a strain.

        ``dataset_path`` may be the run folder, its ``02_Outputs`` folder, or
        a mapping/sequence of full paths to specific exported CSV files. Call
        this method once for every independent notebook/run. ``dataset_name``
        is optional and is useful when multiple run folders share a basename.

        Embedded strain metadata is checked against ``strain``. Conflicting or
        absent metadata produce warnings and are retained in the validation
        report; a conflicting dataset is blocked when a comparison is run.
        Analysis-setting differences warn during loading and again when the
        affected comparison is requested.
        """
        declared_strain = self.__validate_strain(strain)
        for record in self._datasets:
            if record.strain.casefold() == declared_strain.casefold():
                declared_strain = record.strain
                break
        paths, source, inferred_name = self.__resolve_dataset_paths(
            dataset_path
        )
        resolved_dataset_name = (
            self.__validate_dataset_name(dataset_name)
            if dataset_name is not None else inferred_name
        )
        new_run_roots = {
            self.__direct_file_run_root(path).resolve()
            for path in paths.values()
        }
        for record in self._datasets:
            loaded_run_roots = {
                self.__direct_file_run_root(Path(path)).resolve()
                for path in record.paths.values()
            }
            if new_run_roots.intersection(loaded_run_roots):
                raise ValueError(
                    f"Notebook/run '{resolved_dataset_name}' has already been "
                    f"loaded as dataset '{record.dataset_id}'. Load every "
                    'required file for a run in one load_dataframes call.'
                )
        if any(
            record.dataset_id == resolved_dataset_name
            for record in self._datasets
        ):
            raise ValueError(
                f"dataset_name '{resolved_dataset_name}' is already loaded. "
                'Provide a unique dataset_name for each notebook/run.'
            )

        loaded_dataframes: dict[str, pd.DataFrame] = {}
        missing_strain_metadata: list[str] = []
        mismatched_strain_metadata: list[tuple[str, str, Path]] = []
        for dataframe_name, path in paths.items():
            dataframe = pd.read_csv(path)
            self.__validate_required_columns(
                dataframe_name, dataframe, path
            )
            strain_status, embedded_strain = self.__validate_embedded_strain(
                dataframe=dataframe,
                declared_strain=declared_strain,
            )
            if strain_status == 'unverified':
                missing_strain_metadata.append(dataframe_name)
            elif strain_status == 'mismatch':
                mismatched_strain_metadata.append(
                    (dataframe_name, embedded_strain, path)
                )
            dataframe = self.__add_provenance_columns(
                dataframe=dataframe,
                strain=declared_strain,
                dataset_id=resolved_dataset_name,
            )
            loaded_dataframes[dataframe_name] = dataframe

        record = _DatasetRecord(
            dataset_id=resolved_dataset_name,
            strain=declared_strain,
            source=source,
            dataframes=loaded_dataframes,
            paths={key: str(value) for key, value in paths.items()},
        )
        self._datasets.append(record)
        if missing_strain_metadata:
            missing_names = ', '.join(sorted(missing_strain_metadata))
            message = (
                f"Dataset '{resolved_dataset_name}' files lack verifiable "
                f"strain metadata: {missing_names}. The declared strain "
                f"'{declared_strain}' was attached, but should be confirmed "
                'by rerunning the source notebook with current exports.'
            )
            self._strain_validation_rows.append({
                'severity': 'warning',
                'status': 'strain_unverified',
                'dataframe': missing_names,
                'parameter': 'strain',
                'datasets': resolved_dataset_name,
                'values': declared_strain,
                'message': message,
            })
            warnings.warn(message, UserWarning, stacklevel=2)
        for dataframe_name, embedded_strain, path in (
            mismatched_strain_metadata
        ):
            message = (
                f"Dataset '{resolved_dataset_name}' declared strain "
                f"'{declared_strain}', but '{dataframe_name}' contains "
                f"'{embedded_strain}' ({path}). The dataset was loaded for "
                'inspection but will be blocked from comparisons until the '
                'declared strain or source export is corrected.'
            )
            self._strain_validation_rows.append({
                'severity': 'error',
                'status': 'strain_mismatch',
                'dataframe': dataframe_name,
                'parameter': 'strain',
                'datasets': resolved_dataset_name,
                'values': (
                    f'declared={declared_strain}; embedded={embedded_strain}'
                ),
                'message': message,
            })
            warnings.warn(message, UserWarning, stacklevel=2)

        self.__reset_comparison_results()
        self._parameter_validation_dataframe = (
            self.__build_parameter_validation_report()
        )
        if not self._parameter_validation_dataframe.empty:
            self.__emit_validation_warning(
                self._parameter_validation_dataframe,
                context='loading dataframes',
            )
        return {
            name: dataframe.copy()
            for name, dataframe in loaded_dataframes.items()
        }

    def clear_dataframes(self) -> None:
        """Remove all loaded runs and cached comparison results."""
        self._datasets = []
        self._strain_validation_rows = []
        self._parameter_validation_dataframe = pd.DataFrame(
            columns=self.VALIDATION_COLUMNS
        )
        self.__reset_comparison_results()

    def get_dataset_registry(self) -> pd.DataFrame:
        """Return one provenance row per loaded notebook/run."""
        rows = []
        for record in self._datasets:
            rows.append({
                'dataset_id': record.dataset_id,
                'strain': record.strain,
                'dataset_source': record.source,
                'loaded_dataframe_count': len(record.dataframes),
                'loaded_dataframes': ', '.join(sorted(record.dataframes)),
            })
        return pd.DataFrame(rows, columns=[
            'dataset_id', 'strain', 'dataset_source',
            'loaded_dataframe_count', 'loaded_dataframes',
        ])

    def get_validation_report(self) -> pd.DataFrame:
        """Return strain-verification and parameter-compatibility findings."""
        frames = []
        if self._strain_validation_rows:
            frames.append(pd.DataFrame(
                self._strain_validation_rows,
                columns=self.VALIDATION_COLUMNS,
            ))
        if not self._parameter_validation_dataframe.empty:
            frames.append(self._parameter_validation_dataframe.copy())
        if not frames:
            return pd.DataFrame(columns=self.VALIDATION_COLUMNS)
        return pd.concat(frames, ignore_index=True)[self.VALIDATION_COLUMNS]

    def set_analysis_plotting_parameters(
        self,
        title: str | Mapping[str, str | None] | None = None,
        title_fontsize: float | None = None,
        x_axis_fontsize: float | None = None,
        y_axis_fontsize: float | None = None,
        x_tick_fontsize: float | None = None,
        y_tick_fontsize: float | None = None,
        x_axis_titles: str | Sequence | Mapping[str, object] | None = None,
        y_axis_titles: str | Sequence | Mapping[str, object] | None = None,
        strain_colors: (
            Mapping[str, str] | Sequence[str] | str | None
        ) = None,
        font_family: str | None = 'Aptos',
        dpi: int = 300,
    ) -> None:
        """Set reusable defaults for all ``ComparativeStats`` plots.

        Plot calls inherit these settings when their corresponding argument
        is omitted or ``None``. An empty ``title`` or axis-title string removes
        that title. ``title``, ``x_axis_titles``, and ``y_axis_titles`` may be
        mappings keyed by plotting-method name when different defaults are
        needed for different comparison plots.

        A single-axis plot accepts an axis-title string. A twin-y-axis plot
        accepts two y titles in left/right order. For
        ``plot_run_and_tumble_comparison``, axis-title sequences follow panel
        order A through F; the panel-F y-title entry must itself contain the
        left and right titles, for example::

            y_axis_titles=[
                'A y', 'B y', 'C y', 'D y', 'E y',
                ['Left-side y-axis', 'Right-side y-axis'],
            ]

        ``strain_colors`` accepts a strain-to-color mapping, a reusable color
        sequence, a single Matplotlib color, or a colormap name. The angular
        MSD comparison uses strain colors by default; marker shape distinguishes
        its two metrics.
        """
        parameters = {
            'title': title,
            'title_fontsize': title_fontsize,
            'x_axis_fontsize': x_axis_fontsize,
            'y_axis_fontsize': y_axis_fontsize,
            'x_tick_fontsize': x_tick_fontsize,
            'y_tick_fontsize': y_tick_fontsize,
            'x_axis_titles': x_axis_titles,
            'y_axis_titles': y_axis_titles,
            'strain_colors': strain_colors,
            'font_family': font_family,
            'dpi': dpi,
        }
        self._analysis_plotting_parameters = (
            self.__validate_analysis_plotting_parameters(parameters)
        )

    def get_analysis_plotting_parameters(self) -> dict:
        """Return a copy of the current comparison-plot defaults."""
        parameters = getattr(
            self,
            '_analysis_plotting_parameters',
            self.ANALYSIS_PLOT_DEFAULTS,
        )
        return dict(parameters)

    def get_combined_dataframe(self, dataframe_name: str) -> pd.DataFrame:
        """Return all loaded rows for one canonical exported dataframe."""
        normalized_name = self.__normalize_dataframe_name(dataframe_name)
        dataframes = [
            record.dataframes[normalized_name]
            for record in self._datasets
            if normalized_name in record.dataframes
        ]
        if not dataframes:
            raise ValueError(
                f"No '{normalized_name}' dataframe has been loaded."
            )
        return pd.concat(dataframes, ignore_index=True).copy()

    def angle_tumble_analysis_comparison(
        self,
        export_csv_path: str | os.PathLike | None = None,
    ) -> pd.DataFrame:
        """
        Rebuild a combined Turner-style Table 1 for every strain.

        Finite per-cell means from all runs receive equal weight. Sample SD
        uses ``ddof=1`` and SEM uses the metric-specific finite cell count.
        Body dimensions come from the same composite cells in the particle-
        characteristics exports; notebook-level Table 1 means are not averaged.
        """
        self.__prepare_comparison('angle')
        summary = self.get_combined_dataframe('angle_tumble_summary')
        characteristics = self.get_combined_dataframe(
            'particle_characteristics'
        )
        self.__require_compatible_units(
            summary, ('distance_unit', 'speed_unit'),
            'angle-based tumble comparison'
        )
        self.__require_compatible_units(
            characteristics, ('length_unit',),
            'angle-based body-size comparison'
        )

        rows: list[dict] = []
        for strain, strain_summary in summary.groupby('strain', sort=False):
            dataset_count = int(strain_summary['dataset_id'].nunique())
            speed_unit = self.__single_unit(
                strain_summary, 'speed_unit', 'speed'
            )
            distance_unit = self.__single_unit(
                strain_summary, 'distance_unit', 'distance'
            )
            strain_characteristics = characteristics.loc[
                (characteristics['strain'] == strain) &
                (characteristics['cell_id'].isin(strain_summary['cell_id']))
            ]
            number_of_cells = int(len(strain_summary))
            number_tumbled = int((self.__numeric_series(
                strain_summary['number_of_tumbles']
            ).fillna(0) > 0).sum())
            total_runs = int(self.__numeric_series(
                strain_summary['number_of_runs']
            ).fillna(0).sum())
            total_tumbles = int(self.__numeric_series(
                strain_summary['number_of_tumbles']
            ).fillna(0).sum())
            total_tracking_time = float(self.__numeric_series(
                strain_summary['analyzed_tracking_time_seconds']
            ).fillna(0).sum())

            self.__append_table_value(
                rows, strain, 'Number of cells', number_of_cells,
                number_of_cells, dataset_count
            )
            self.__append_table_value(
                rows, strain, 'Number of cells that tumbled',
                number_tumbled, number_of_cells, dataset_count
            )
            table_metrics = [
                (f'Run speed ({speed_unit})', 'mean_run_speed'),
                (f'Tumble speed ({speed_unit})', 'mean_tumble_speed'),
                (
                    'Angular speed while running (°/point)',
                    'mean_run_angular_speed_degrees_per_point',
                ),
                (
                    'Angular speed while tumbling (°/point)',
                    'mean_tumble_angular_speed_degrees_per_point',
                ),
                ('Run interval (s)', 'mean_run_interval_seconds'),
                ('Tumble interval (s)', 'mean_tumble_interval_seconds'),
                (
                    'Change in direction from run to run (°)',
                    'mean_run_to_run_direction_change_degrees',
                ),
                (
                    'Change in direction during runs (°)',
                    'mean_direction_change_within_runs_degrees',
                ),
                ('Tumble frequency (s^-1)', 'tumble_frequency_per_second'),
            ]
            for label, column in table_metrics:
                self.__append_distribution_rows(
                    rows=rows,
                    strain=strain,
                    label=label,
                    values=self.__finite_values(strain_summary[column]),
                    dataset_count=dataset_count,
                )
            self.__append_table_value(
                rows,
                strain,
                'Number of events (runs, tumbles)',
                f'{total_runs}, {total_tumbles}',
                total_runs + total_tumbles,
                dataset_count,
            )
            self.__append_table_value(
                rows, strain, 'Total tracking time (s)',
                total_tracking_time, number_of_cells, dataset_count
            )
            body_diameter = self.__finite_values(
                strain_characteristics['mean_minor_axis_length']
            )
            body_length = self.__finite_values(
                strain_characteristics['mean_major_axis_length']
            )
            self.__append_table_value(
                rows, strain, f'Body diameter ({distance_unit})',
                self.__safe_mean(body_diameter), len(body_diameter),
                dataset_count,
            )
            self.__append_table_value(
                rows, strain, f'Body length ({distance_unit})',
                self.__safe_mean(body_length), len(body_length),
                dataset_count,
            )

        comparison = pd.DataFrame(rows, columns=[
            'strain', 'Parameter', 'Value', 'sample_count', 'dataset_count'
        ])
        self._angle_comparison_dataframe = comparison
        if export_csv_path is not None:
            self.__export_dataframe(
                comparison, export_csv_path,
                'Angle_Tumble_Analysis_Comparison.csv'
            )
        return comparison.copy()

    def velocity_tumble_analysis_comparison(
        self,
        export_csv_path: str | os.PathLike | None = None,
    ) -> pd.DataFrame:
        """
        Rebuild a support-weighted Najafi-style Table 1 per strain.

        Run/tumble speeds use support seconds; p and R use their exported pair
        counts. Run and tumble durations use exported complete-episode counts.
        Exact pooled tR SD/SEM requires the current ``std_t_R`` export.
        """
        self.__prepare_comparison('velocity')
        summary = self.get_combined_dataframe('velocity_tumble_summary')
        self.__require_compatible_units(
            summary, ('distance_unit', 'speed_unit', 'angular_velocity_unit'),
            'velocity-based tumble comparison'
        )
        rows: list[dict] = []
        for strain, strain_summary in summary.groupby('strain', sort=False):
            dataset_count = int(strain_summary['dataset_id'].nunique())
            speed_unit = self.__single_unit(
                strain_summary, 'speed_unit', 'speed'
            )
            v_r, _ = self.__weighted_mean_from_columns(
                strain_summary, 'v_R', 'v_R_support_seconds'
            )
            v_t, _ = self.__weighted_mean_from_columns(
                strain_summary, 'v_T', 'v_T_support_seconds'
            )
            if 'v_R_sample_count' in strain_summary.columns:
                v_r_count = int(self.__numeric_series(
                    strain_summary['v_R_sample_count']
                ).fillna(0).sum())
            else:
                v_r_count = np.nan
            if 'v_T_sample_count' in strain_summary.columns:
                v_t_count = int(self.__numeric_series(
                    strain_summary['v_T_sample_count']
                ).fillna(0).sum())
            else:
                v_t_count = np.nan
            t_r, t_r_count = self.__weighted_mean_from_columns(
                strain_summary, 't_R', 't_R_interval_count'
            )
            std_t_r, sem_t_r = self.__combine_group_standard_deviation(
                strain_summary, 't_R', 'std_t_R', 't_R_interval_count'
            )
            t_t, t_t_count = self.__weighted_mean_from_columns(
                strain_summary, 't_T', 't_T_interval_count'
            )
            persistence, persistence_count = (
                self.__weighted_mean_from_columns(
                    strain_summary, 'p', 'p_direction_change_count'
                )
            )
            run_persistence, run_persistence_count = (
                self.__weighted_mean_from_columns(
                    strain_summary, 'R', 'R_run_transition_count'
                )
            )

            self.__append_table_value(
                rows, strain, 'Strain', strain,
                int(strain_summary['cell_id'].nunique()), dataset_count
            )
            self.__append_table_value(
                rows, strain, f'vR ({speed_unit})', v_r,
                v_r_count, dataset_count
            )
            self.__append_table_value(
                rows, strain, f'vT ({speed_unit})', v_t,
                v_t_count, dataset_count
            )
            self.__append_table_value(
                rows, strain, 'Mean tR (s)', t_r,
                t_r_count, dataset_count
            )
            self.__append_table_value(
                rows, strain, '±std', std_t_r,
                t_r_count, dataset_count
            )
            self.__append_table_value(
                rows, strain, '±sem', sem_t_r,
                t_r_count, dataset_count
            )
            self.__append_table_value(
                rows, strain, 'Mean tT (s)', t_t,
                t_t_count, dataset_count
            )
            self.__append_table_value(
                rows, strain, 'p', persistence,
                persistence_count, dataset_count
            )
            self.__append_table_value(
                rows, strain, 'R', run_persistence,
                run_persistence_count, dataset_count
            )

        comparison = pd.DataFrame(rows, columns=[
            'strain', 'Parameter', 'Value', 'sample_count', 'dataset_count'
        ])
        self._velocity_comparison_dataframe = comparison
        if export_csv_path is not None:
            self.__export_dataframe(
                comparison, export_csv_path,
                'Velocity_Tumble_Analysis_Comparison.csv'
            )
        return comparison.copy()

    def plot_angular_msd_comparison(
        self,
        show_dataset_values: bool = True,
        show_sem: bool = True,
        figsize: tuple[float, float] = (9, 6),
        p_color: str | None = None,
        dr_color: str | None = None,
        save_path: str | os.PathLike | None = None,
        dpi: int | None = None,
        show: bool = True,
        *,
        title: str | None = None,
        title_fontsize: float | None = None,
        x_axis_fontsize: float | None = None,
        y_axis_fontsize: float | None = None,
        x_tick_fontsize: float | None = None,
        y_tick_fontsize: float | None = None,
        x_axis_titles: str | Sequence | None = None,
        y_axis_titles: str | Sequence | None = None,
        strain_colors: (
            Mapping[str, str] | Sequence[str] | str | None
        ) = None,
        font_family: str | None = None,
        legend_position: str | None = None,
    ) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
        """
        Plot run persistence and rotational diffusion by strain.

        This reproduces the quantities and marker convention of Najafi
        Figure 1D: within-run directional persistence ``p`` uses filled
        circles on the left axis, while the run-only angular-MSD coefficient
        ``Dr`` uses open squares on the right axis. Each exported Table 1 is
        one independent dataset value. Strain markers are equal-dataset means;
        colors identify strains by default. ``p_color`` and ``dr_color`` remain
        available as explicit legacy overrides for their respective metrics.
        ``legend_position`` accepts ``'middle top'``, ``'middle center'``, or
        ``'middle bottom'``; ``None`` keeps the upper-right position.
        Optional error bars are SEM across the loaded datasets and are a
        MiMoSA aggregation convention, not a reconstruction of the paper's
        unspecified error-bar calculation.

        Legacy velocity Table 1 files remain loadable, but this plot requires
        each loaded dataset to contain exactly one finite ``p`` row and one
        finite, nonnegative ``Dr (rad²/s)`` row. Missing per-particle velocity
        summaries or angular-MSD fit diagnostics warn because calculation
        compatibility or fit provenance cannot then be verified.
        """
        plotting_parameters = self.__resolve_analysis_plotting_parameters(
            title=title,
            title_fontsize=title_fontsize,
            x_axis_fontsize=x_axis_fontsize,
            y_axis_fontsize=y_axis_fontsize,
            x_tick_fontsize=x_tick_fontsize,
            y_tick_fontsize=y_tick_fontsize,
            x_axis_titles=x_axis_titles,
            y_axis_titles=y_axis_titles,
            strain_colors=strain_colors,
            font_family=font_family,
            dpi=dpi,
        )
        self.__prepare_comparison('angular_msd')
        if not isinstance(show_dataset_values, bool):
            raise TypeError('show_dataset_values must be a boolean.')
        if not isinstance(show_sem, bool):
            raise TypeError('show_sem must be a boolean.')
        if p_color is not None and not is_color_like(p_color):
            raise ValueError('p_color must be a Matplotlib-compatible color.')
        if dr_color is not None and not is_color_like(dr_color):
            raise ValueError('dr_color must be a Matplotlib-compatible color.')
        resolved_legend_position = self.__resolve_legend_position(
            legend_position
        )

        dataset_values = self.__angular_msd_dataset_values()
        missing_method_metadata = dataset_values.loc[
            ~dataset_values['velocity_tumble_summary_available'], 'dataset_id'
        ].tolist()
        if missing_method_metadata:
            warnings.warn(
                'Angular-MSD methodology metadata cannot be verified for '
                'datasets without Velocity Tumble Summary exports: ' +
                ', '.join(missing_method_metadata) + '.',
                UserWarning,
                stacklevel=2,
            )
        missing_fit_diagnostics = dataset_values.loc[
            ~dataset_values['Dr_fit_diagnostics_verified'], 'dataset_id'
        ].tolist()
        if missing_fit_diagnostics:
            warnings.warn(
                'Angular-MSD fit diagnostics are not exported for datasets: '
                + ', '.join(missing_fit_diagnostics) + '. Their finite Dr '
                'values can be plotted, but fit support cannot be verified.',
                UserWarning,
                stacklevel=2,
            )
        strain_order = list(dict.fromkeys(dataset_values['strain']))
        strain_color_map = self.__resolve_colors(
            strain_order, plotting_parameters['strain_colors']
        )
        summary_rows = []
        for strain in strain_order:
            strain_values = dataset_values.loc[
                dataset_values['strain'] == strain
            ]
            persistence_values = strain_values['dataset_p'].to_numpy(
                dtype=float
            )
            diffusion_values = strain_values['dataset_Dr'].to_numpy(
                dtype=float
            )
            summary_rows.append({
                'strain': strain,
                'dataset_count': int(len(strain_values)),
                'mean_p': self.__safe_mean(persistence_values),
                'std_p': self.__safe_std(persistence_values),
                'sem_p': self.__safe_sem(persistence_values),
                'mean_Dr': self.__safe_mean(diffusion_values),
                'std_Dr': self.__safe_std(diffusion_values),
                'sem_Dr': self.__safe_sem(diffusion_values),
                'Dr_unit': 'rad²/s',
                'aggregation_rule': (
                    'equal-dataset mean; SD and SEM are across loaded '
                    'datasets within each declared strain'
                ),
            })
        strain_summary = pd.DataFrame(summary_rows)
        dataset_export = dataset_values.copy()
        dataset_export.insert(0, 'record_type', 'dataset')
        dataset_export['Dr_unit'] = 'rad²/s'
        dataset_export['aggregation_rule'] = 'independent loaded dataset'
        summary_export = strain_summary.copy()
        summary_export.insert(0, 'record_type', 'combined_strain_mean')
        comparison = pd.concat(
            [dataset_export, summary_export],
            ignore_index=True,
            sort=False,
        )
        self._angular_msd_comparison_dataframe = comparison

        fig, persistence_axis = plt.subplots(figsize=figsize)
        diffusion_axis = persistence_axis.twinx()
        x_positions = np.arange(len(strain_order), dtype=float)
        for strain_index, strain in enumerate(strain_order):
            persistence_color = (
                strain_color_map[strain] if p_color is None else p_color
            )
            diffusion_color = (
                strain_color_map[strain] if dr_color is None else dr_color
            )
            strain_values = dataset_values.loc[
                dataset_values['strain'] == strain
            ].reset_index(drop=True)
            strain_result = strain_summary.loc[
                strain_summary['strain'] == strain
            ].iloc[0]
            if show_dataset_values and len(strain_values) > 1:
                jitter = np.linspace(-0.08, 0.08, len(strain_values))
                persistence_axis.plot(
                    strain_index + jitter,
                    strain_values['dataset_p'].to_numpy(dtype=float),
                    marker='o', markersize=4, linestyle='none',
                    color=persistence_color, alpha=0.28
                )
                diffusion_axis.plot(
                    strain_index + jitter,
                    strain_values['dataset_Dr'].to_numpy(dtype=float),
                    marker='s', markersize=4, linestyle='none',
                    markerfacecolor='none', color=diffusion_color, alpha=0.35
                )
            persistence_sem = float(strain_result['sem_p'])
            diffusion_sem = float(strain_result['sem_Dr'])
            persistence_axis.errorbar(
                strain_index,
                float(strain_result['mean_p']),
                yerr=(
                    persistence_sem
                    if show_sem and np.isfinite(persistence_sem) else None
                ),
                marker='o', markersize=8, capsize=4,
                markerfacecolor=persistence_color, color=persistence_color,
                linestyle='none'
            )
            diffusion_axis.errorbar(
                strain_index,
                float(strain_result['mean_Dr']),
                yerr=(
                    diffusion_sem
                    if show_sem and np.isfinite(diffusion_sem) else None
                ),
                marker='s', markersize=8, capsize=4,
                markerfacecolor='none', color=diffusion_color,
                linestyle='none'
            )

        persistence_axis.set_xticks(x_positions, strain_order)
        self.__center_categorical_strain_axis(
            persistence_axis, len(strain_order)
        )
        persistence_axis.set_ylabel('Directional persistence p')
        diffusion_axis.set_ylabel(
            'Rotational diffusion coefficient Dr (rad²/s)'
        )
        persistence_axis.set_title(
            'Run-phase persistence and rotational diffusion by strain'
        )
        persistence_axis.grid(alpha=0.2, axis='y')
        persistence_axis.legend(
            handles=[
                Line2D(
                    [], [], marker='o', markersize=8,
                    markerfacecolor='black', color='black',
                    linestyle='none', label='Directional persistence p'
                ),
                Line2D(
                    [], [], marker='s', markersize=8,
                    markerfacecolor='none', color='black',
                    linestyle='none', label='Rotational diffusion Dr'
                ),
            ],
            loc=resolved_legend_position
        )
        self.__apply_axis_title_settings(
            plotting_parameters,
            'plot_angular_msd_comparison',
            [(persistence_axis, diffusion_axis)],
        )
        self.__format_comparison_figure(
            fig,
            plotting_parameters,
            'plot_angular_msd_comparison',
            persistence_axis,
        )
        fig.tight_layout()
        self.__finalize_figure(
            fig, save_path, plotting_parameters['dpi'], show
        )
        return fig, (persistence_axis, diffusion_axis)

    def get_angular_msd_comparison_dataframe(self) -> pd.DataFrame:
        """Return separate dataset and combined-strain summary records."""
        if self._angular_msd_comparison_dataframe.empty:
            raise ValueError(
                'Run plot_angular_msd_comparison before requesting its data.'
            )
        return self._angular_msd_comparison_dataframe.copy()

    def fitted_mean_speed_distribution_comparison(
        self,
        distribution_type: str = 'norm',
        bins: int | None = 25,
        figsize: tuple[float, float] = (9, 6),
        colors: Mapping[str, str] | Sequence[str] | None = None,
        save_plot_path: str | os.PathLike | None = None,
        export_csv_path: str | os.PathLike | None = None,
        dpi: int | None = None,
        show: bool = True,
        *,
        show_histogram: bool | None = None,
        title: str | None = None,
        title_fontsize: float | None = None,
        x_axis_fontsize: float | None = None,
        y_axis_fontsize: float | None = None,
        x_tick_fontsize: float | None = None,
        y_tick_fontsize: float | None = None,
        x_axis_titles: str | Sequence | None = None,
        y_axis_titles: str | Sequence | None = None,
        strain_colors: (
            Mapping[str, str] | Sequence[str] | str | None
        ) = None,
        font_family: str | None = None,
    ) -> tuple[pd.DataFrame, plt.Figure, plt.Axes]:
        """
        Fit and plot the distribution of particle fitted means by strain.

        Input values are each particle's already fitted mean speed, not the
        original point-speed samples. The returned dataframe summarizes each
        strain. The CSV export contains every particle observation with the
        corresponding strain fit and strain mean repeated for direct reuse.
        ``show_histogram=False`` hides the observed step histograms while
        retaining the fitted curves. ``bins=None`` keeps the established
        25-bin default.
        """
        plotting_parameters = self.__resolve_analysis_plotting_parameters(
            title=title,
            title_fontsize=title_fontsize,
            x_axis_fontsize=x_axis_fontsize,
            y_axis_fontsize=y_axis_fontsize,
            x_tick_fontsize=x_tick_fontsize,
            y_tick_fontsize=y_tick_fontsize,
            x_axis_titles=x_axis_titles,
            y_axis_titles=y_axis_titles,
            strain_colors=strain_colors,
            font_family=font_family,
            dpi=dpi,
        )
        self.__prepare_comparison('speed')
        if distribution_type != 'norm':
            raise ValueError("distribution_type currently supports only 'norm'.")
        bins = 25 if bins is None else bins
        if isinstance(bins, bool) or not isinstance(bins, int) or bins < 1:
            raise ValueError('bins must be a positive integer.')
        show_histogram = True if show_histogram is None else show_histogram
        if not isinstance(show_histogram, bool):
            raise TypeError('show_histogram must be a boolean or None.')
        if colors is not None and strain_colors is not None:
            raise ValueError(
                'Pass either legacy colors or strain_colors, not both.'
            )
        observations = self.get_combined_dataframe('fitted_mean_speeds')
        self.__require_compatible_units(
            observations, ('speed_unit',),
            'fitted mean-speed comparison'
        )
        observations = observations.copy()
        observations['mean_speed'] = self.__numeric_series(
            observations['mean_speed']
        )
        observations = observations.loc[
            np.isfinite(observations['mean_speed'])
        ].reset_index(drop=True)
        if observations.empty:
            raise ValueError('No finite fitted mean-speed values are available.')

        strain_order = list(dict.fromkeys(observations['strain']))
        resolved_colors = (
            colors
            if colors is not None
            else plotting_parameters['strain_colors']
        )
        color_map = self.__resolve_colors(strain_order, resolved_colors)
        fig, ax = plt.subplots(figsize=figsize)
        summary_rows = []
        overall_values = observations['mean_speed'].to_numpy(dtype=float)
        histogram_edges = np.linspace(
            float(overall_values.min()), float(overall_values.max()), bins + 1
        )
        if np.isclose(histogram_edges[0], histogram_edges[-1]):
            histogram_edges = np.linspace(
                histogram_edges[0] - 0.5,
                histogram_edges[-1] + 0.5,
                bins + 1,
            )
        fit_x_values = np.linspace(
            histogram_edges[0], histogram_edges[-1], 300
        )
        for strain in strain_order:
            strain_rows = observations.loc[observations['strain'] == strain]
            values = strain_rows['mean_speed'].to_numpy(dtype=float)
            fitted_location, fitted_scale = norm.fit(values)
            speed_unit = self.__single_unit(
                strain_rows, 'speed_unit', 'speed'
            )
            dataset_means = self.__finite_values(
                strain_rows.groupby('dataset_id')['mean_speed'].mean()
            )
            summary_rows.append({
                'strain': strain,
                'dataset_count': int(strain_rows['dataset_id'].nunique()),
                'particle_count': int(len(values)),
                'distribution': distribution_type,
                'fitted_location': float(fitted_location),
                'fitted_scale': float(fitted_scale),
                'strain_mean': self.__safe_mean(values),
                'sample_std': self.__safe_std(values),
                'sem': self.__safe_sem(values),
                'median': float(np.median(values)),
                'mean_of_dataset_means': self.__safe_mean(dataset_means),
                'std_of_dataset_means': self.__safe_std(dataset_means),
                'sem_of_dataset_means': self.__safe_sem(dataset_means),
                'speed_unit': speed_unit,
                'histogram_shown': show_histogram,
                'histogram_bins': bins,
            })
            if show_histogram:
                ax.hist(
                    values, bins=histogram_edges, density=True,
                    histtype='step', linewidth=1.6, alpha=0.8,
                    color=color_map[strain],
                    label=f'{strain} observations (n={len(values)})'
                )
            if fitted_scale > 0:
                ax.plot(
                    fit_x_values, norm.pdf(
                        fit_x_values, loc=fitted_location, scale=fitted_scale
                    ),
                    color=color_map[strain], linewidth=2.2,
                    label=(
                        f'{strain} normal fit '
                        f'(mean={fitted_location:.3g})'
                    ),
                )
            else:
                ax.axvline(
                    fitted_location, color=color_map[strain], linewidth=2.2,
                    label=f'{strain} point-mass fit'
                )
        summary = pd.DataFrame(summary_rows)
        export_columns = [
            column for column in (
                'strain', 'dataset_id', 'cell_id', 'particle', 'mean_speed'
            ) if column in observations.columns
        ]
        distribution_export = observations[export_columns].merge(
            summary,
            on='strain',
            how='left',
            validate='many_to_one',
        )
        distribution_export = distribution_export.rename(columns={
            'mean_speed': 'particle_fitted_mean_speed'
        })
        distribution_export['fitted_probability_density'] = np.nan
        positive_scale = distribution_export['fitted_scale'] > 0
        distribution_export.loc[
            positive_scale, 'fitted_probability_density'
        ] = norm.pdf(
            distribution_export.loc[
                positive_scale, 'particle_fitted_mean_speed'
            ],
            loc=distribution_export.loc[
                positive_scale, 'fitted_location'
            ],
            scale=distribution_export.loc[positive_scale, 'fitted_scale'],
        )
        distribution_export['aggregation_rule'] = (
            'one row per particle; normal fit pooled across particles within '
            'each declared strain'
        )
        self._fitted_speed_observations_dataframe = observations
        self._fitted_speed_comparison_dataframe = summary
        self._fitted_speed_export_dataframe = distribution_export
        unit_label = self.__plot_unit_label(
            self.__single_unit(observations, 'speed_unit', 'speed')
        )
        ax.set_xlabel(f'Particle fitted mean speed ({unit_label})')
        ax.set_ylabel('Probability density')
        ax.set_title('Fitted mean-speed distributions by strain')
        ax.grid(alpha=0.2)
        ax.legend()
        self.__apply_axis_title_settings(
            plotting_parameters,
            'fitted_mean_speed_distribution_comparison',
            [(ax,)],
        )
        self.__format_comparison_figure(
            fig,
            plotting_parameters,
            'fitted_mean_speed_distribution_comparison',
            ax,
        )
        fig.tight_layout()
        self.__finalize_figure(
            fig, save_plot_path, plotting_parameters['dpi'], show
        )
        if export_csv_path is not None:
            self.__export_dataframe(
                distribution_export, export_csv_path,
                'Fitted_Mean_Speed_Distribution_Comparison.csv'
            )
        return summary.copy(), fig, ax

    def turn_analysis_comparison(
        self,
        export_csv_path: str | os.PathLike | None = None,
    ) -> pd.DataFrame:
        """
        Compare exposure-normalized geometric-turn results by strain.

        Pooled rates divide total turns by total observed path/time. Equal-cell
        means are also retained; raw turn count is descriptive because it is
        confounded by track length and duration.
        """
        self.__prepare_comparison('turn')
        summary = self.get_combined_dataframe('turn_summary')
        self.__require_compatible_units(
            summary, ('distance_unit',), 'turn comparison'
        )
        rows = []
        for strain, strain_rows in summary.groupby('strain', sort=False):
            turns = self.__numeric_series(
                strain_rows['number_of_turns']
            )
            path_lengths = self.__numeric_series(
                strain_rows['total_path_length']
            )
            durations = self.__numeric_series(
                strain_rows['observed_duration_seconds']
            )
            angles = self.__numeric_series(
                strain_rows['mean_absolute_turn_angle_degrees']
            )
            finite_turns = np.isfinite(turns)
            distance_mask = (
                finite_turns & np.isfinite(path_lengths) &
                (path_lengths > 0)
            )
            time_mask = (
                finite_turns & np.isfinite(durations) & (durations > 0)
            )
            angle_mask = finite_turns & np.isfinite(angles) & (turns > 0)
            total_turns = float(turns.loc[finite_turns].sum())
            distance_turns = float(turns.loc[distance_mask].sum())
            time_turns = float(turns.loc[time_mask].sum())
            total_path = float(path_lengths.loc[distance_mask].sum())
            total_duration = float(durations.loc[time_mask].sum())
            all_turn_angle = (
                float(np.average(
                    angles.loc[angle_mask], weights=turns.loc[angle_mask]
                )) if angle_mask.any() else np.nan
            )
            equal_cell_angles = self.__finite_values(angles)
            per_distance = self.__finite_values(
                strain_rows['turns_per_distance']
            )
            per_second = self.__finite_values(
                strain_rows['turns_per_second']
            )
            dataset_distance_rates = []
            dataset_time_rates = []
            for _, dataset_rows in strain_rows.groupby(
                'dataset_id', sort=False
            ):
                dataset_turns = self.__numeric_series(
                    dataset_rows['number_of_turns']
                )
                dataset_paths = self.__numeric_series(
                    dataset_rows['total_path_length']
                )
                dataset_durations = self.__numeric_series(
                    dataset_rows['observed_duration_seconds']
                )
                valid_distance = (
                    np.isfinite(dataset_turns) &
                    np.isfinite(dataset_paths) & (dataset_paths > 0)
                )
                valid_time = (
                    np.isfinite(dataset_turns) &
                    np.isfinite(dataset_durations) &
                    (dataset_durations > 0)
                )
                if valid_distance.any():
                    dataset_distance_rates.append(float(
                        dataset_turns.loc[valid_distance].sum() /
                        dataset_paths.loc[valid_distance].sum()
                    ))
                if valid_time.any():
                    dataset_time_rates.append(float(
                        dataset_turns.loc[valid_time].sum() /
                        dataset_durations.loc[valid_time].sum()
                    ))
            dataset_distance_rates = np.asarray(
                dataset_distance_rates, dtype=float
            )
            dataset_time_rates = np.asarray(dataset_time_rates, dtype=float)
            rows.append({
                'strain': strain,
                'dataset_count': int(strain_rows['dataset_id'].nunique()),
                'particle_count': int(strain_rows['cell_id'].nunique()),
                'total_turn_count': int(total_turns),
                'mean_turns_per_particle': self.__safe_mean(
                    self.__finite_values(turns)
                ),
                'std_turns_per_particle': self.__safe_std(
                    self.__finite_values(turns)
                ),
                'sem_turns_per_particle': self.__safe_sem(
                    self.__finite_values(turns)
                ),
                'equal_cell_mean_absolute_turn_angle_degrees': (
                    self.__safe_mean(equal_cell_angles)
                ),
                'std_cell_mean_absolute_turn_angle_degrees': (
                    self.__safe_std(equal_cell_angles)
                ),
                'sem_cell_mean_absolute_turn_angle_degrees': (
                    self.__safe_sem(equal_cell_angles)
                ),
                'all_turn_pooled_mean_absolute_turn_angle_degrees': (
                    all_turn_angle
                ),
                'mean_particle_turns_per_distance': self.__safe_mean(
                    per_distance
                ),
                'std_particle_turns_per_distance': self.__safe_std(
                    per_distance
                ),
                'sem_particle_turns_per_distance': self.__safe_sem(
                    per_distance
                ),
                'mean_particle_turns_per_second': self.__safe_mean(
                    per_second
                ),
                'std_particle_turns_per_second': self.__safe_std(
                    per_second
                ),
                'sem_particle_turns_per_second': self.__safe_sem(
                    per_second
                ),
                'pooled_turns_per_distance': (
                    distance_turns / total_path if total_path > 0 else np.nan
                ),
                'pooled_turns_per_second': (
                    time_turns / total_duration
                    if total_duration > 0 else np.nan
                ),
                'mean_dataset_pooled_turns_per_distance': (
                    self.__safe_mean(dataset_distance_rates)
                ),
                'sem_dataset_pooled_turns_per_distance': (
                    self.__safe_sem(dataset_distance_rates)
                ),
                'mean_dataset_pooled_turns_per_second': (
                    self.__safe_mean(dataset_time_rates)
                ),
                'sem_dataset_pooled_turns_per_second': (
                    self.__safe_sem(dataset_time_rates)
                ),
                'total_path_length_for_pooled_rate': total_path,
                'total_observed_duration_seconds_for_pooled_rate': (
                    total_duration
                ),
                'distance_rate_particle_count': int(distance_mask.sum()),
                'time_rate_particle_count': int(time_mask.sum()),
                'distance_unit': self.__single_unit(
                    strain_rows, 'distance_unit', 'distance'
                ),
            })
        comparison = pd.DataFrame(rows)
        self._turn_comparison_dataframe = comparison
        if export_csv_path is not None:
            self.__export_dataframe(
                comparison, export_csv_path,
                'Turn_Analysis_Comparison.csv'
            )
        return comparison.copy()

    def plot_run_and_tumble_comparison(
        self,
        speed_bins: int | None = 30,
        angle_bins: int = 30,
        figsize: tuple[float, float] = (17, 10),
        colors: Mapping[str, str] | Sequence[str] | None = None,
        save_path: str | os.PathLike | None = None,
        dpi: int | None = None,
        show: bool = True,
        *,
        split_figures: bool = False,
        panel_b_curve: str | None = None,
        panel_b_show_histogram: bool | None = None,
        panel_b_curve_resolution: int | None = None,
        panel_e_curve: str | None = None,
        panel_e_curve_factor: float | str | None = None,
        panel_e_curve_resolution: int | None = None,
        title: str | None = None,
        title_fontsize: float | None = None,
        x_axis_fontsize: float | None = None,
        y_axis_fontsize: float | None = None,
        x_tick_fontsize: float | None = None,
        y_tick_fontsize: float | None = None,
        x_axis_titles: str | Sequence | None = None,
        y_axis_titles: str | Sequence | None = None,
        strain_colors: (
            Mapping[str, str] | Sequence[str] | str | None
        ) = None,
        font_family: str | None = None,
        legend_position: str | None = None,
    ) -> tuple[plt.Figure | list[plt.Figure], np.ndarray]:
        """
        Plot Najafi Figure 2-inspired panels A through F.

        Panel B pools point-level speeds classified as run. Panel D uses
        complete run episodes reconstructed from the tumble-event boundaries
        and censoring convention used by ``Stats``.
        Panel E reconstructs signed run-to-run angles from the fitted run
        directions before and after each tumble. Panel F uses their absolute
        magnitude and the exported directional cosine.

        ``split_figures=True`` returns six figures and a one-dimensional axes
        array in panel order A through F. When saving split figures, panel
        suffixes are added to ``save_path``. Panel E keeps its established
        binned-density curve when ``panel_e_curve`` is None. It also supports
        ``'kde'`` and ``'normal'`` curves; ``panel_e_curve_resolution`` sets
        their number of evaluation points and ``panel_e_curve_factor`` sets
        the KDE bandwidth (a positive number, ``'scott'``, or
        ``'silverman'``). ``legend_position`` controls the panel-F metric
        legend and accepts ``'middle top'``, ``'middle center'``, or
        ``'middle bottom'``; ``None`` keeps the upper-right position.

        Panel B retains its established binned-density line when
        ``panel_b_curve`` and ``panel_b_show_histogram`` are both ``None``.
        Set ``panel_b_curve='normal'`` to fit a normal curve to each strain's
        pooled run-speed points. The observed step histogram is overlaid by
        default in normal mode and can be hidden with
        ``panel_b_show_histogram=False``. Setting only
        ``panel_b_show_histogram=True`` replaces the established line with
        step histograms. ``panel_b_curve_resolution`` controls the number of
        fitted-curve points, and ``speed_bins=None`` keeps the 30-bin default.
        Panel-B legends report the number of contributing cells and pooled
        mean run speed for each strain.
        """
        plotting_parameters = self.__resolve_analysis_plotting_parameters(
            title=title,
            title_fontsize=title_fontsize,
            x_axis_fontsize=x_axis_fontsize,
            y_axis_fontsize=y_axis_fontsize,
            x_tick_fontsize=x_tick_fontsize,
            y_tick_fontsize=y_tick_fontsize,
            x_axis_titles=x_axis_titles,
            y_axis_titles=y_axis_titles,
            strain_colors=strain_colors,
            font_family=font_family,
            dpi=dpi,
        )
        self.__prepare_comparison('run_tumble_plot')
        if not isinstance(split_figures, bool):
            raise TypeError('split_figures must be a boolean.')
        resolved_legend_position = self.__resolve_legend_position(
            legend_position
        )
        panel_b_curve_method, panel_b_histogram_mode, panel_b_resolution = (
            self.__resolve_panel_b_curve_parameters(
                panel_b_curve,
                panel_b_show_histogram,
                panel_b_curve_resolution,
            )
        )
        curve_method, curve_factor, curve_resolution = (
            self.__resolve_panel_e_curve_parameters(
                panel_e_curve,
                panel_e_curve_factor,
                panel_e_curve_resolution,
            )
        )
        if colors is not None and strain_colors is not None:
            raise ValueError(
                'Pass either legacy colors or strain_colors, not both.'
            )
        speed_bins = 30 if speed_bins is None else speed_bins
        if (
            isinstance(speed_bins, bool) or not isinstance(speed_bins, int) or
            speed_bins < 1
        ):
            raise ValueError('speed_bins must be a positive integer.')
        if (
            isinstance(angle_bins, bool) or not isinstance(angle_bins, int) or
            angle_bins < 1
        ):
            raise ValueError('angle_bins must be a positive integer.')
        summary = self.get_combined_dataframe('velocity_tumble_summary')
        points = self.get_combined_dataframe('velocity_tumble_points')
        events = self.get_combined_dataframe('velocity_tumble_events')
        self.__require_compatible_units(
            summary, ('speed_unit',), 'Najafi-style Figure 2 comparison'
        )
        episodes = self.__build_velocity_state_episodes()
        complete_episodes = episodes.loc[
            ~episodes['is_boundary_censored']
        ].copy()
        strain_order = list(dict.fromkeys(summary['strain']))
        resolved_colors = (
            colors
            if colors is not None
            else plotting_parameters['strain_colors']
        )
        color_map = self.__resolve_colors(strain_order, resolved_colors)
        dataset_summary = self.__build_velocity_dataset_summary(
            summary
        )
        signed_angles_available = {
            'preceding_run_direction_radians',
            'following_run_direction_radians',
        }.issubset(events.columns)
        if not signed_angles_available:
            warnings.warn(
                'Signed run-to-run angles cannot be reconstructed because '
                'the fitted run-direction columns are absent. Panel E falls '
                'back to the exported signed turn-angle column.',
                UserWarning,
                stacklevel=2,
            )

        if split_figures:
            split_figsize = (figsize[0] / 3, figsize[1] / 2)
            figure_axis_pairs = [
                plt.subplots(figsize=split_figsize) for _ in range(6)
            ]
            figures = [pair[0] for pair in figure_axis_pairs]
            panel_axes = [pair[1] for pair in figure_axis_pairs]
            ax_a, ax_b, ax_d, ax_e, ax_f, ax_g = panel_axes
            axes = np.asarray(panel_axes, dtype=object)
        else:
            fig, axes = plt.subplots(2, 3, figsize=figsize)
            figures = [fig]
            ax_a, ax_d, ax_f = axes[0]
            ax_b, ax_e, ax_g = axes[1]
        x_positions = np.arange(len(strain_order), dtype=float)
        state_legend_handles = [
            Line2D(
                [], [], marker='o', markerfacecolor='white',
                markeredgecolor='black', color='black', linestyle='none',
                markersize=8, label='Run'
            ),
            Line2D(
                [], [], marker='D', markerfacecolor='white',
                markeredgecolor='black', color='black', linestyle='none',
                markersize=8, label='Tumble'
            ),
        ]

        panel_a_rows = []
        for index, strain in enumerate(strain_order):
            strain_datasets = dataset_summary.loc[
                dataset_summary['strain'] == strain
            ]
            for metric, marker, offset, label in (
                ('v_R', 'o', -0.13, 'Run'),
                ('v_T', 'D', 0.13, 'Tumble'),
            ):
                values = self.__finite_values(strain_datasets[metric])
                if len(values):
                    jitter = np.linspace(-0.035, 0.035, len(values))
                    ax_a.scatter(
                        index + offset + jitter, values,
                        facecolors=color_map[strain],
                        edgecolors=color_map[strain],
                        marker=marker, alpha=0.8, s=42,
                    )
                    mean_value = self.__safe_mean(values)
                    sem_value = self.__safe_sem(values)
                    ax_a.errorbar(
                        index + offset, mean_value,
                        yerr=sem_value if np.isfinite(sem_value) else None,
                        color=color_map[strain], marker=marker,
                        markerfacecolor='white', capsize=4,
                        linestyle='none', markersize=8,
                    )
                    panel_a_rows.append({
                        'strain': strain,
                        'record_type': 'combined_strain_mean',
                        'dataset_id': np.nan,
                        'state': label.lower(),
                        'dataset_count': len(values),
                        'mean_speed': mean_value,
                        'sem_speed': sem_value,
                    })
                    panel_a_rows.extend({
                        'strain': strain,
                        'record_type': 'dataset',
                        'dataset_id': dataset_id,
                        'state': label.lower(),
                        'dataset_count': 1,
                        'mean_speed': value,
                        'sem_speed': np.nan,
                    } for dataset_id, value in zip(
                        strain_datasets.loc[
                            np.isfinite(self.__numeric_series(
                                strain_datasets[metric]
                            )), 'dataset_id'
                        ],
                        values,
                    ))
        speed_unit = self.__plot_unit_label(
            self.__single_unit(summary, 'speed_unit', 'speed')
        )
        ax_a.set_xticks(x_positions, strain_order)
        self.__center_categorical_strain_axis(ax_a, len(strain_order))
        ax_a.set_ylabel(f'Speed ({speed_unit})')
        ax_a.set_title('A  Mean run and tumble speeds')
        ax_a.legend(handles=state_legend_handles)
        ax_a.grid(alpha=0.2, axis='y')

        panel_b_rows = []
        panel_b_curve_rows = []
        all_run_speeds = self.__finite_values(points.loc[
            points['state'].astype(str).str.lower() == 'run', 'speed'
        ])
        if not len(all_run_speeds):
            raise ValueError('No finite run-speed points are available.')
        speed_edges = np.linspace(
            float(all_run_speeds.min()), float(all_run_speeds.max()),
            speed_bins + 1,
        )
        if np.isclose(speed_edges[0], speed_edges[-1]):
            speed_edges = np.linspace(
                speed_edges[0] - 0.5, speed_edges[-1] + 0.5,
                speed_bins + 1,
        )
        for strain in strain_order:
            strain_run_points = points.loc[
                (points['strain'] == strain) &
                (points['state'].astype(str).str.lower() == 'run')
            ].copy()
            numeric_run_speeds = self.__numeric_series(
                strain_run_points['speed']
            )
            finite_speed = np.isfinite(numeric_run_speeds)
            run_speeds = numeric_run_speeds.loc[finite_speed].to_numpy(
                dtype=float
            )
            if not len(run_speeds):
                continue
            cell_count = int(
                strain_run_points.loc[finite_speed, 'cell_id'].nunique()
            )
            density, edges = np.histogram(
                run_speeds, bins=speed_edges, density=True
            )
            centers = (edges[:-1] + edges[1:]) / 2
            mean_run_speed = self.__safe_mean(run_speeds)
            mean_label = f'{mean_run_speed:.3g} {speed_unit}'
            legend_label = (
                f'{strain}\n'
                f'n={cell_count}, mean={mean_label}'
            )
            if panel_b_histogram_mode == 'current':
                ax_b.plot(
                    centers, density, color=color_map[strain], linewidth=2,
                    label=legend_label,
                )
            elif panel_b_histogram_mode == 'histogram':
                histogram_label = (
                    '_nolegend_'
                    if panel_b_curve_method == 'normal'
                    else legend_label
                )
                ax_b.hist(
                    run_speeds, bins=speed_edges, density=True,
                    histtype='step', linewidth=1.6, alpha=0.8,
                    color=color_map[strain], label=histogram_label,
                )

            fitted_location = np.nan
            fitted_scale = np.nan
            if panel_b_curve_method == 'normal':
                fitted_location, fitted_scale = norm.fit(run_speeds)
                if fitted_scale > 0:
                    fit_x_values = np.linspace(
                        speed_edges[0], speed_edges[-1], panel_b_resolution
                    )
                    fit_y_values = norm.pdf(
                        fit_x_values,
                        loc=fitted_location,
                        scale=fitted_scale,
                    )
                    ax_b.plot(
                        fit_x_values, fit_y_values,
                        color=color_map[strain], linewidth=2.2,
                        label=legend_label,
                    )
                    panel_b_curve_rows.extend({
                        'strain': strain,
                        'speed': speed,
                        'probability_density': probability,
                        'curve_method': 'normal',
                        'fitted_location': float(fitted_location),
                        'fitted_scale': float(fitted_scale),
                        'point_count': len(run_speeds),
                        'particle_count': cell_count,
                        'speed_unit': speed_unit,
                        'curve_resolution': panel_b_resolution,
                    } for speed, probability in zip(
                        fit_x_values, fit_y_values
                    ))
                else:
                    ax_b.axvline(
                        fitted_location,
                        color=color_map[strain],
                        linewidth=2.2,
                        label=legend_label,
                    )
                    panel_b_curve_rows.append({
                        'strain': strain,
                        'speed': float(fitted_location),
                        'probability_density': np.nan,
                        'curve_method': 'point_mass',
                        'fitted_location': float(fitted_location),
                        'fitted_scale': float(fitted_scale),
                        'point_count': len(run_speeds),
                        'particle_count': cell_count,
                        'speed_unit': speed_unit,
                        'curve_resolution': 1,
                    })
            panel_b_rows.extend({
                'strain': strain,
                'speed_bin_center': center,
                'probability_density': probability,
                'point_count': len(run_speeds),
                'particle_count': cell_count,
                'mean_run_speed': mean_run_speed,
                'speed_unit': speed_unit,
                'curve_method': panel_b_curve_method,
                'histogram_mode': panel_b_histogram_mode,
                'histogram_bins': speed_bins,
                'fitted_location': fitted_location,
                'fitted_scale': fitted_scale,
                'curve_resolution': (
                    panel_b_resolution
                    if panel_b_curve_method == 'normal' else np.nan
                ),
            } for center, probability in zip(centers, density))
        ax_b.set_xlabel(f'Run speed ({speed_unit})')
        ax_b.set_ylabel('Probability density')
        ax_b.set_title('B  Run-speed distribution')
        ax_b.legend()
        ax_b.grid(alpha=0.2)

        panel_d_rows = []
        for index, strain in enumerate(strain_order):
            strain_datasets = dataset_summary.loc[
                dataset_summary['strain'] == strain
            ]
            for metric, marker, offset, label in (
                ('t_R', 'o', -0.13, 'Run'),
                ('t_T', 'D', 0.13, 'Tumble'),
            ):
                values = self.__finite_values(strain_datasets[metric])
                if len(values):
                    jitter = np.linspace(-0.035, 0.035, len(values))
                    ax_d.scatter(
                        index + offset + jitter, values,
                        facecolors=color_map[strain],
                        edgecolors=color_map[strain],
                        marker=marker, alpha=0.8, s=42,
                    )
                    mean_value = self.__safe_mean(values)
                    sem_value = self.__safe_sem(values)
                    ax_d.errorbar(
                        index + offset, mean_value,
                        yerr=sem_value if np.isfinite(sem_value) else None,
                        color=color_map[strain], marker=marker,
                        markerfacecolor='white', capsize=4,
                        linestyle='none', markersize=8,
                    )
                    panel_d_rows.append({
                        'strain': strain,
                        'record_type': 'combined_strain_mean',
                        'dataset_id': np.nan,
                        'state': label.lower(),
                        'dataset_count': len(values),
                        'mean_interval_seconds': mean_value,
                        'sem_interval_seconds': sem_value,
                    })
                    panel_d_rows.extend({
                        'strain': strain,
                        'record_type': 'dataset',
                        'dataset_id': dataset_id,
                        'state': label.lower(),
                        'dataset_count': 1,
                        'mean_interval_seconds': value,
                        'sem_interval_seconds': np.nan,
                    } for dataset_id, value in zip(
                        strain_datasets.loc[
                            np.isfinite(self.__numeric_series(
                                strain_datasets[metric]
                            )), 'dataset_id'
                        ],
                        values,
                    ))
        ax_d.set_xticks(x_positions, strain_order)
        self.__center_categorical_strain_axis(ax_d, len(strain_order))
        ax_d.set_ylabel('Time (s)')
        ax_d.set_title('C  Mean run and tumble times')
        ax_d.legend(handles=state_legend_handles)
        ax_d.grid(alpha=0.2, axis='y')

        panel_e_rows = []
        for strain in strain_order:
            run_intervals = self.__finite_values(complete_episodes.loc[
                (complete_episodes['strain'] == strain) &
                (complete_episodes['state'] == 'run'),
                'interval_seconds'
            ])
            if not len(run_intervals):
                continue
            times = np.sort(run_intervals)
            survival = (
                len(times) - np.arange(1, len(times) + 1, dtype=float)
            ) / len(times)
            positive_survival = survival > 0
            plot_times = np.concatenate((
                np.array([0.0]), times[positive_survival]
            ))
            plot_survival = np.concatenate((
                np.array([1.0]), survival[positive_survival]
            ))
            ax_e.step(
                plot_times, plot_survival,
                where='post', color=color_map[strain], linewidth=2,
                label=f'{strain} (n={len(times)})'
            )
            panel_e_rows.extend({
                'strain': strain,
                'run_interval_seconds': time,
                'survival_probability': probability,
                'interval_count': len(times),
            } for time, probability in zip(times, survival))
        ax_e.set_yscale('log')
        ax_e.set_xlabel('Run time (s)')
        ax_e.set_ylabel('P(run time > t)')
        ax_e.set_title('D  Run-time survival distribution')
        ax_e.legend()
        ax_e.grid(alpha=0.2, which='both')

        panel_f_rows = []
        panel_e_curve_rows = []
        for strain in strain_order:
            strain_events = events.loc[events['strain'] == strain]
            angles = self.__signed_run_to_run_angles(strain_events)
            if not len(angles):
                angles = self.__finite_values(
                    strain_events['run_to_run_turn_angle_degrees']
                )
            if not len(angles):
                continue
            density, edges = np.histogram(
                angles, bins=angle_bins, range=(-180, 180), density=True
            )
            centers = (edges[:-1] + edges[1:]) / 2
            curve_x, curve_y = self.__panel_e_curve_values(
                angles,
                centers,
                density,
                curve_method,
                curve_factor,
                curve_resolution,
            )
            ax_f.plot(
                curve_x, curve_y, color=color_map[strain], linewidth=2,
                label=f'{strain} (n={len(angles)})'
            )
            panel_f_rows.extend({
                'strain': strain,
                'signed_angle_bin_center_degrees': center,
                'probability_density': probability,
                'transition_count': len(angles),
            } for center, probability in zip(centers, density))
            panel_e_curve_rows.extend({
                'strain': strain,
                'angle_degrees': angle,
                'probability_density': probability,
                'curve_method': curve_method,
                'curve_factor': curve_factor,
                'curve_resolution': len(curve_x),
                'transition_count': len(angles),
            } for angle, probability in zip(curve_x, curve_y))
        ax_f.set_xlim(-180, 180)
        ax_f.set_xlabel(r'Run-to-run $\phi$ (degrees)')
        ax_f.set_ylabel('Probability density')
        ax_f.set_title(
            r'E  Turning-angle probability distribution $R(\phi)$'
        )
        ax_f.legend()
        ax_f.grid(alpha=0.2)

        panel_g_rows = []
        ax_g_r = ax_g.twinx()
        mean_angles = []
        mean_cosines = []
        sem_angles = []
        sem_cosines = []
        for strain in strain_order:
            strain_events = events.loc[events['strain'] == strain]
            signed_angles = self.__signed_run_to_run_angles(strain_events)
            angles = (
                np.abs(signed_angles)
                if len(signed_angles)
                else np.abs(self.__finite_values(
                    strain_events['run_to_run_turn_angle_degrees']
                ))
            )
            cosines = self.__finite_values(
                strain_events['run_to_run_directional_cosine']
            )
            mean_angle = self.__safe_mean(angles)
            mean_cosine = self.__safe_mean(cosines)
            mean_angles.append(mean_angle)
            mean_cosines.append(mean_cosine)
            angle_sem = self.__safe_sem(angles)
            cosine_sem = self.__safe_sem(cosines)
            sem_angles.append(angle_sem)
            sem_cosines.append(cosine_sem)
            panel_g_rows.append({
                'strain': strain,
                'angle_transition_count': len(angles),
                'persistence_transition_count': len(cosines),
                'mean_absolute_turn_angle_degrees': mean_angle,
                'sem_absolute_turn_angle_degrees': angle_sem,
                'R': mean_cosine,
                'sem_R': cosine_sem,
            })
        for index, strain in enumerate(strain_order):
            ax_g.errorbar(
                index, mean_angles[index],
                yerr=(
                    sem_angles[index]
                    if np.isfinite(sem_angles[index]) else None
                ),
                marker='^', markersize=9, capsize=4,
                color=color_map[strain], linestyle='none'
            )
            ax_g_r.errorbar(
                index, mean_cosines[index],
                yerr=(
                    sem_cosines[index]
                    if np.isfinite(sem_cosines[index]) else None
                ),
                marker='s', markersize=8, capsize=4,
                markerfacecolor='none', color=color_map[strain],
                linestyle='none'
            )
        ax_g.set_xticks(x_positions, strain_order)
        self.__center_categorical_strain_axis(ax_g, len(strain_order))
        ax_g.set_ylabel(r'$\langle|\phi|\rangle$ (degrees)')
        ax_g_r.set_ylabel(r'$R$')
        ax_g.set_title(
            r'F  Average turning angle $\langle|\phi|\rangle$ and '
            'directional change\n'
            r'$R=\langle\cos\phi\rangle$ between consecutive run phases'
        )
        ax_g.grid(alpha=0.2, axis='y')
        ax_g.legend(
            handles=[
                Line2D([], [], marker='^', color='black',
                       linestyle='none', label=r'Avg. turning angle $\langle|\phi|\rangle$'),
                Line2D([], [], marker='s', markerfacecolor='none',
                       color='black', linestyle='none', label=r'$R=\langle\cos\phi\rangle$'),
            ],
            loc=resolved_legend_position
        )

        self._run_tumble_plot_data = {
            'panel_A_speed_summary': pd.DataFrame(panel_a_rows),
            'panel_B_run_speed_pdf': pd.DataFrame(panel_b_rows),
            'panel_B_run_speed_curve': pd.DataFrame(panel_b_curve_rows),
            'panel_D_interval_summary': pd.DataFrame(panel_d_rows),
            'panel_E_run_survival': pd.DataFrame(panel_e_rows),
            'panel_F_turn_angle_pdf': pd.DataFrame(panel_f_rows),
            'panel_G_turn_persistence': pd.DataFrame(panel_g_rows),
            'panel_E_turn_angle_curve': pd.DataFrame(panel_e_curve_rows),
            'state_episodes': episodes.copy(),
        }
        axis_groups = [
            (ax_a,),
            (ax_b,),
            (ax_d,),
            (ax_e,),
            (ax_f,),
            (ax_g, ax_g_r),
        ]
        self.__apply_axis_title_settings(
            plotting_parameters,
            'plot_run_and_tumble_comparison',
            axis_groups,
        )
        if split_figures:
            save_paths = self.__split_figure_save_paths(
                save_path, ('A', 'B', 'C', 'D', 'E', 'F')
            )
            for figure, figure_save_path in zip(figures, save_paths):
                self.__format_comparison_figure(
                    figure,
                    plotting_parameters,
                    'plot_run_and_tumble_comparison',
                    None,
                )
                figure.tight_layout()
                self.__finalize_figure(
                    figure,
                    figure_save_path,
                    plotting_parameters['dpi'],
                    False,
                )
            if show:
                plt.show()
            return figures, axes

        figure = figures[0]
        figure.suptitle(
            'Run-and-tumble comparison by strain', fontsize=15
        )
        self.__format_comparison_figure(
            figure,
            plotting_parameters,
            'plot_run_and_tumble_comparison',
            None,
        )
        figure.tight_layout()
        self.__finalize_figure(
            figure, save_path, plotting_parameters['dpi'], show
        )
        return figure, axes

    def get_run_and_tumble_comparison_dataframes(
        self,
    ) -> dict[str, pd.DataFrame]:
        """Return copies of the data supporting the six Figure 2 panels."""
        if not self._run_tumble_plot_data:
            raise ValueError(
                'Run plot_run_and_tumble_comparison before requesting data.'
            )
        return {
            name: dataframe.copy()
            for name, dataframe in self._run_tumble_plot_data.items()
        }

    def plot_msd_comparison(
        self,
        show_dataset_curves: bool = True,
        show_sem: bool = True,
        lag_grid_points: int = 100,
        figsize: tuple[float, float] = (9, 7),
        colors: Mapping[str, str] | Sequence[str] | None = None,
        save_path: str | os.PathLike | None = None,
        dpi: int | None = None,
        show: bool = True,
        *,
        title: str | None = None,
        title_fontsize: float | None = None,
        x_axis_fontsize: float | None = None,
        y_axis_fontsize: float | None = None,
        x_tick_fontsize: float | None = None,
        y_tick_fontsize: float | None = None,
        x_axis_titles: str | Sequence | None = None,
        y_axis_titles: str | Sequence | None = None,
        strain_colors: (
            Mapping[str, str] | Sequence[str] | str | None
        ) = None,
        font_family: str | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Plot an equal-particle MSD comparison inspired by Najafi Fig. 3C.

        Each particle is interpolated in log-log space onto a common lag grid
        bounded by the interval supported by every loaded run. This prevents
        slightly different FPS-derived lag grids from being interleaved or
        extrapolated.
        """
        plotting_parameters = self.__resolve_analysis_plotting_parameters(
            title=title,
            title_fontsize=title_fontsize,
            x_axis_fontsize=x_axis_fontsize,
            y_axis_fontsize=y_axis_fontsize,
            x_tick_fontsize=x_tick_fontsize,
            y_tick_fontsize=y_tick_fontsize,
            x_axis_titles=x_axis_titles,
            y_axis_titles=y_axis_titles,
            strain_colors=strain_colors,
            font_family=font_family,
            dpi=dpi,
        )
        self.__prepare_comparison('msd')
        if colors is not None and strain_colors is not None:
            raise ValueError(
                'Pass either legacy colors or strain_colors, not both.'
            )
        if (
            isinstance(lag_grid_points, bool) or
            not isinstance(lag_grid_points, int) or
            lag_grid_points < 2
        ):
            raise ValueError('lag_grid_points must be an integer of at least 2.')
        individual = self.get_combined_dataframe('individual_msd')
        ensemble = self.__optional_combined_dataframe('ensemble_msd')
        self.__require_compatible_units(
            individual, ('time_unit', 'msd_unit'), 'MSD comparison'
        )
        if show_dataset_curves:
            missing_ensemble = [
                record.dataset_id for record in self._datasets
                if 'ensemble_msd' not in record.dataframes
            ]
            if missing_ensemble:
                warnings.warn(
                    'Per-dataset ensemble MSD curves are unavailable for: '
                    f'{", ".join(missing_ensemble)}. The combined '
                    'equal-particle strain curves still include those runs.',
                    UserWarning,
                    stacklevel=2,
                )
            if not ensemble.empty:
                self.__require_compatible_units(
                    ensemble, ('time_unit', 'msd_unit'),
                    'per-dataset ensemble MSD overlay'
                )
        individual = individual.copy()
        individual['lag_time'] = self.__numeric_series(
            individual['lag_time']
        )
        individual['individual_msd'] = self.__numeric_series(
            individual['individual_msd']
        )
        individual = individual.loc[
            np.isfinite(individual['lag_time']) &
            np.isfinite(individual['individual_msd']) &
            (individual['lag_time'] > 0) &
            (individual['individual_msd'] > 0)
        ].copy()
        if individual.empty:
            raise ValueError('No positive finite individual-MSD values exist.')
        dataset_ranges = individual.groupby('dataset_id')['lag_time'].agg(
            ['min', 'max']
        )
        common_minimum_lag = float(dataset_ranges['min'].max())
        common_maximum_lag = float(dataset_ranges['max'].min())
        if common_minimum_lag >= common_maximum_lag:
            raise ValueError(
                'Loaded runs do not share a positive MSD lag-time interval.'
            )
        lag_grid = np.geomspace(
            common_minimum_lag, common_maximum_lag, lag_grid_points
        )
        interpolated_rows = []
        for (strain, dataset_id, cell_id), particle_rows in individual.groupby(
            ['strain', 'dataset_id', 'cell_id'], sort=False
        ):
            particle_rows = (
                particle_rows.groupby('lag_time', as_index=False)[
                    'individual_msd'
                ].mean().sort_values('lag_time')
            )
            if len(particle_rows) < 2:
                continue
            lag_values = particle_rows['lag_time'].to_numpy(dtype=float)
            msd_values = particle_rows['individual_msd'].to_numpy(dtype=float)
            supports_grid = (
                (lag_grid >= lag_values.min()) &
                (lag_grid <= lag_values.max())
            )
            if not supports_grid.any():
                continue
            interpolated_values = np.exp(np.interp(
                np.log(lag_grid[supports_grid]),
                np.log(lag_values),
                np.log(msd_values),
            ))
            interpolated_rows.extend({
                'strain': strain,
                'dataset_id': dataset_id,
                'cell_id': cell_id,
                'lag_time': lag_time,
                'individual_msd': msd_value,
            } for lag_time, msd_value in zip(
                lag_grid[supports_grid], interpolated_values
            ))
        interpolated = pd.DataFrame(interpolated_rows)
        if interpolated.empty:
            raise ValueError(
                'No particle MSD curves could be interpolated to the common '
                'lag grid.'
            )

        rows = []
        time_unit = self.__single_unit(individual, 'time_unit', 'time')
        msd_unit = self.__single_unit(individual, 'msd_unit', 'MSD')
        for (strain, lag_time), lag_rows in interpolated.groupby(
            ['strain', 'lag_time'], sort=True
        ):
            values = lag_rows['individual_msd'].to_numpy(dtype=float)
            rows.append({
                'strain': strain,
                'lag_time': float(lag_time),
                'mean_msd': self.__safe_mean(values),
                'std_msd': self.__safe_std(values),
                'sem_msd': self.__safe_sem(values),
                'particle_count': int(lag_rows['cell_id'].nunique()),
                'dataset_count': int(lag_rows['dataset_id'].nunique()),
                'time_unit': time_unit,
                'msd_unit': msd_unit,
                'common_lag_minimum': common_minimum_lag,
                'common_lag_maximum': common_maximum_lag,
                'lag_grid_points': lag_grid_points,
                'aggregation_rule': (
                    'equal-particle mean after log-log interpolation to a '
                    'common non-extrapolated lag grid'
                ),
            })
        comparison = pd.DataFrame(rows)
        self._msd_comparison_dataframe = comparison
        strain_order = list(dict.fromkeys(comparison['strain']))
        resolved_colors = (
            colors
            if colors is not None
            else plotting_parameters['strain_colors']
        )
        color_map = self.__resolve_colors(strain_order, resolved_colors)
        fig, ax = plt.subplots(figsize=figsize)
        if show_dataset_curves and not ensemble.empty:
            for (strain, _), dataset_rows in ensemble.groupby(
                ['strain', 'dataset_id'], sort=False
            ):
                ordered = dataset_rows.sort_values('lag_time')
                ax.plot(
                    ordered['lag_time'], ordered['ensemble_msd'],
                    color=color_map[strain], alpha=0.22, linewidth=1
                )
        for strain in strain_order:
            strain_rows = comparison.loc[
                comparison['strain'] == strain
            ].sort_values('lag_time')
            x_values = strain_rows['lag_time'].to_numpy(dtype=float)
            y_values = strain_rows['mean_msd'].to_numpy(dtype=float)
            sem_values = strain_rows['sem_msd'].to_numpy(dtype=float)
            ax.plot(
                x_values, y_values, color=color_map[strain], linewidth=2.5,
                label=(
                    f'{strain} equal-particle mean '
                    f'({int(strain_rows.dataset_count.max())} runs)'
                )
            )
            finite_sem = np.isfinite(sem_values)
            if show_sem and finite_sem.any():
                lower = np.maximum(y_values - np.nan_to_num(sem_values), 0)
                upper = y_values + np.nan_to_num(sem_values)
                ax.fill_between(
                    x_values, lower, upper, color=color_map[strain],
                    alpha=0.16, linewidth=0
                )
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(f'Lag time ({self.__plot_unit_label(time_unit)})')
        ax.set_ylabel(f'MSD ({self.__plot_unit_label(msd_unit)})')
        ax.set_title('MSD comparison by strain')
        ax.grid(alpha=0.2, which='both')
        ax.legend()
        self.__apply_axis_title_settings(
            plotting_parameters,
            'plot_msd_comparison',
            [(ax,)],
        )
        self.__format_comparison_figure(
            fig,
            plotting_parameters,
            'plot_msd_comparison',
            ax,
        )
        fig.tight_layout()
        self.__finalize_figure(
            fig, save_path, plotting_parameters['dpi'], show
        )
        return fig, ax

    def export_angle_tumble_analysis_comparison(
        self, output_path: str | os.PathLike
    ) -> str:
        """Export the latest combined Turner-style comparison."""
        return self.__export_cached_dataframe(
            self._angle_comparison_dataframe, output_path,
            'Angle_Tumble_Analysis_Comparison.csv',
            'angle_tumble_analysis_comparison'
        )

    def export_velocity_tumble_analysis_comparison(
        self, output_path: str | os.PathLike
    ) -> str:
        """Export the latest combined Najafi-style comparison."""
        return self.__export_cached_dataframe(
            self._velocity_comparison_dataframe, output_path,
            'Velocity_Tumble_Analysis_Comparison.csv',
            'velocity_tumble_analysis_comparison'
        )

    def export_angular_msd_comparison(
        self, output_path: str | os.PathLike
    ) -> str:
        """Export dataset-level p/Dr values and their strain summaries."""
        return self.__export_cached_dataframe(
            self._angular_msd_comparison_dataframe, output_path,
            'Angular_MSD_Comparison.csv',
            'plot_angular_msd_comparison'
        )

    def export_fitted_mean_speed_distribution_comparison(
        self, output_path: str | os.PathLike
    ) -> str:
        """Export the latest per-particle fitted-speed comparison data."""
        return self.__export_cached_dataframe(
            self._fitted_speed_export_dataframe, output_path,
            'Fitted_Mean_Speed_Distribution_Comparison.csv',
            'fitted_mean_speed_distribution_comparison'
        )

    def export_turn_analysis_comparison(
        self, output_path: str | os.PathLike
    ) -> str:
        """Export the latest geometric-turn comparison."""
        return self.__export_cached_dataframe(
            self._turn_comparison_dataframe, output_path,
            'Turn_Analysis_Comparison.csv', 'turn_analysis_comparison'
        )

    def export_run_and_tumble_comparison_data(
        self, output_folder: str | os.PathLike
    ) -> list[str]:
        """Export one CSV for each cached Najafi Figure 2-style panel."""
        if not self._run_tumble_plot_data:
            raise ValueError(
                'Run plot_run_and_tumble_comparison before exporting data.'
            )
        folder = Path(output_folder).expanduser()
        if folder.suffix:
            raise ValueError('output_folder must be a directory path.')
        folder.mkdir(parents=True, exist_ok=True)
        output_paths = []
        for name, dataframe in self._run_tumble_plot_data.items():
            output_path = folder / f'{name}.csv'
            dataframe.to_csv(output_path, index=False)
            output_paths.append(str(output_path))
        return output_paths

    def export_msd_comparison(
        self, output_path: str | os.PathLike
    ) -> str:
        """Export the latest equal-particle MSD comparison data."""
        return self.__export_cached_dataframe(
            self._msd_comparison_dataframe, output_path,
            'MSD_Comparison.csv', 'plot_msd_comparison'
        )

    def __resolve_dataset_paths(
        self,
        dataset_path: str | os.PathLike | Mapping[str, str | os.PathLike] |
        Sequence[str | os.PathLike],
    ) -> tuple[dict[str, Path], str, str]:
        if isinstance(dataset_path, Mapping):
            if not dataset_path:
                raise ValueError('dataset_path mapping cannot be empty.')
            paths = {}
            for dataframe_name, file_path in dataset_path.items():
                normalized_name = self.__normalize_dataframe_name(
                    dataframe_name
                )
                path = self.__validate_explicit_file_path(file_path)
                paths[normalized_name] = path
            self.__validate_direct_file_run_roots(list(paths.values()))
            source = os.path.commonpath(
                [str(path.parent) for path in paths.values()]
            )
            inferred_name = self.__infer_direct_dataset_name(
                list(paths.values())
            )
            return paths, source, inferred_name

        if isinstance(dataset_path, Sequence) and not isinstance(
            dataset_path, (str, bytes, os.PathLike)
        ):
            if not dataset_path:
                raise ValueError('dataset_path file sequence cannot be empty.')
            paths = {}
            for file_path in dataset_path:
                path = self.__validate_explicit_file_path(file_path)
                dataframe_name = self.__name_from_filename(path.name)
                if dataframe_name in paths:
                    raise ValueError(
                        f"Multiple files resolve to '{dataframe_name}'."
                    )
                paths[dataframe_name] = path
            self.__validate_direct_file_run_roots(list(paths.values()))
            source = os.path.commonpath(
                [str(path.parent) for path in paths.values()]
            )
            inferred_name = self.__infer_direct_dataset_name(
                list(paths.values())
            )
            return paths, source, inferred_name

        root = Path(dataset_path).expanduser().resolve()
        if root.is_file():
            dataframe_name = self.__name_from_filename(root.name)
            return (
                {dataframe_name: root},
                str(root.parent),
                self.__infer_direct_dataset_name([root]),
            )
        if not root.is_dir():
            raise FileNotFoundError(f'Dataset folder does not exist: {root}')
        outputs_root = self.__resolve_outputs_root(root)
        paths: dict[str, Path] = {}
        for dataframe_name, relative_candidates in self.DATAFRAME_FILES.items():
            for relative_path in relative_candidates:
                candidate = outputs_root / relative_path
                if candidate.is_file():
                    paths[dataframe_name] = candidate.resolve()
                    break
        if not paths:
            raise FileNotFoundError(
                f'No recognized statistics CSV files were found under {root}.'
            )
        inferred_name = self.__infer_run_name(root, outputs_root)
        return paths, str(root), inferred_name

    @staticmethod
    def __resolve_outputs_root(root: Path) -> Path:
        if (root / '02_Outputs').is_dir():
            return root / '02_Outputs'
        if root.name == '02_Outputs':
            return root
        if root.name in {'05_Basic_Stats', '06_Additional_Analysis'}:
            return root.parent
        if (
            (root / '05_Basic_Stats').is_dir() or
            (root / '06_Additional_Analysis').is_dir()
        ):
            return root
        return root

    @staticmethod
    def __infer_run_name(root: Path, outputs_root: Path) -> str:
        if root.name in {'05_Basic_Stats', '06_Additional_Analysis'}:
            return root.parent.parent.name
        if root.name == '02_Outputs':
            return root.parent.name
        if outputs_root.name == '02_Outputs':
            return outputs_root.parent.name
        return root.name

    @staticmethod
    def __infer_direct_dataset_name(paths: list[Path]) -> str:
        first_parent = paths[0].parent
        if first_parent.name in {
            '05_Basic_Stats', '06_Additional_Analysis'
        }:
            return first_parent.parent.parent.name
        return first_parent.name

    @staticmethod
    def __direct_file_run_root(path: Path) -> Path:
        if path.parent.name in {
            '05_Basic_Stats', '06_Additional_Analysis'
        } and path.parent.parent.name == '02_Outputs':
            return path.parent.parent.parent
        return path.parent

    def __validate_direct_file_run_roots(
        self, paths: list[Path]
    ) -> None:
        run_roots = {
            self.__direct_file_run_root(path).resolve()
            for path in paths
        }
        if len(run_roots) > 1:
            raise ValueError(
                'All explicit dataframe files in one load_dataframes call '
                'must belong to the same notebook/run folder. Found: '
                + ', '.join(sorted(str(path) for path in run_roots))
            )

    @staticmethod
    def __validate_explicit_file_path(
        file_path: str | os.PathLike,
    ) -> Path:
        path = Path(file_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f'Dataframe file does not exist: {path}')
        if path.suffix.lower() != '.csv':
            raise ValueError(f'Only CSV dataframe files are supported: {path}')
        return path

    def __name_from_filename(self, filename: str) -> str:
        normalized_filename = filename.strip().lower()
        for dataframe_name, candidates in self.DATAFRAME_FILES.items():
            if any(
                Path(candidate).name.lower() == normalized_filename
                for candidate in candidates
            ):
                return dataframe_name
        raise ValueError(
            f"Unrecognized statistics filename '{filename}'. Use a mapping "
            'with a canonical dataframe name when the file was renamed.'
        )

    def __normalize_dataframe_name(self, dataframe_name: str) -> str:
        if not isinstance(dataframe_name, str):
            raise TypeError('dataframe_name must be a string.')
        normalized = dataframe_name.strip().lower()
        if normalized not in self.DATAFRAME_FILES:
            raise ValueError(
                f"Unknown dataframe_name '{dataframe_name}'. Available names: "
                f"{sorted(self.DATAFRAME_FILES)}"
            )
        return normalized

    @staticmethod
    def __validate_strain(strain: str) -> str:
        if not isinstance(strain, str):
            raise TypeError('strain must be a string.')
        normalized = strain.strip()
        if not normalized:
            raise ValueError('strain must be a non-empty string.')
        if normalized.casefold() == 'define cell strain':
            raise ValueError(
                "Replace 'define cell strain' with the actual strain name."
            )
        return normalized

    @staticmethod
    def __validate_dataset_name(dataset_name: str) -> str:
        if not isinstance(dataset_name, str):
            raise TypeError('dataset_name must be a string.')
        normalized = dataset_name.strip()
        if not normalized:
            raise ValueError('dataset_name must be a non-empty string.')
        return normalized

    def __validate_required_columns(
        self,
        dataframe_name: str,
        dataframe: pd.DataFrame,
        path: Path,
    ) -> None:
        required = self.REQUIRED_COLUMNS.get(dataframe_name, set())
        missing = sorted(required.difference(dataframe.columns))
        if missing:
            raise ValueError(
                f"{path} is missing required columns for '{dataframe_name}': "
                f'{missing}'
            )

    @staticmethod
    def __clean_strain_values(values: pd.Series) -> set[str]:
        cleaned = {
            str(value).strip()
            for value in values.dropna()
            if str(value).strip() and
            str(value).strip().casefold() != 'define cell strain'
        }
        return cleaned

    def __validate_embedded_strain(
        self,
        dataframe: pd.DataFrame,
        declared_strain: str,
    ) -> tuple[str, str]:
        embedded_values: set[str] = set()
        if 'strain' in dataframe.columns:
            embedded_values.update(self.__clean_strain_values(
                dataframe['strain']
            ))
        if {'Parameter', 'Value'}.issubset(dataframe.columns):
            strain_rows = dataframe.loc[
                dataframe['Parameter'].astype(str).str.strip().str.casefold()
                == 'strain',
                'Value'
            ]
            embedded_values.update(self.__clean_strain_values(strain_rows))
        if not embedded_values:
            return 'unverified', '<not exported>'
        if len(embedded_values) > 1:
            return 'mismatch', ', '.join(sorted(embedded_values))
        embedded_strain = next(iter(embedded_values))
        if embedded_strain.casefold() != declared_strain.casefold():
            return 'mismatch', embedded_strain
        return 'verified', embedded_strain

    @staticmethod
    def __particle_identifier(value: object) -> str:
        numeric_value = pd.to_numeric(value, errors='coerce')
        if pd.notna(numeric_value) and float(numeric_value).is_integer():
            return str(int(numeric_value))
        return str(value)

    @classmethod
    def __particle_ids(cls, particles: pd.Series) -> list[int | str]:
        identifiers = particles.map(cls.__particle_identifier).tolist()
        unique_identifiers = list(dict.fromkeys(identifiers))
        return [
            int(identifier)
            if identifier.lstrip('-').isdigit()
            else identifier
            for identifier in unique_identifiers
        ]

    def __add_provenance_columns(
        self,
        dataframe: pd.DataFrame,
        strain: str,
        dataset_id: str,
    ) -> pd.DataFrame:
        output = dataframe.copy()
        for column in ('strain', 'dataset_id', 'cell_id'):
            if column in output.columns:
                output = output.drop(columns=column)
        output.insert(0, 'strain', strain)
        output.insert(1, 'dataset_id', dataset_id)
        if 'particle' in output.columns:
            particle_labels = output['particle'].map(
                self.__particle_identifier
            )
            output.insert(
                2, 'cell_id', dataset_id + ':' + particle_labels
            )
        return output

    def __build_parameter_validation_report(self) -> pd.DataFrame:
        rows: list[dict] = []
        for dataframe_name in (
            'fitted_mean_speeds', 'particle_characteristics',
            'turn_summary', 'angle_tumble_summary',
            'velocity_tumble_summary',
        ):
            for record in self._datasets:
                dataframe = record.dataframes.get(dataframe_name)
                if dataframe is None or 'cell_id' not in dataframe.columns:
                    continue
                duplicate_count = int(dataframe['cell_id'].duplicated().sum())
                if not duplicate_count:
                    continue
                duplicate_particle_ids = self.__particle_ids(
                    dataframe.loc[
                        dataframe['cell_id'].duplicated(keep=False),
                        'particle',
                    ]
                )
                rows.append({
                    'severity': 'error',
                    'status': 'duplicate_cell_summary',
                    'dataframe': dataframe_name,
                    'parameter': 'cell_id uniqueness',
                    'datasets': record.dataset_id,
                    'values': (
                        f'{duplicate_count} duplicate rows; '
                        f'duplicate_particle_ids={duplicate_particle_ids}'
                    ),
                    'message': (
                        f"Dataset '{record.dataset_id}' {dataframe_name} "
                        f'contains {duplicate_count} duplicate cell rows.'
                    ),
                })
        for dataframe_name, parameters in self.PARAMETER_COLUMNS.items():
            records = [
                record for record in self._datasets
                if dataframe_name in record.dataframes
            ]
            if len(records) < 2:
                continue
            for parameter in parameters:
                signatures = {}
                for record in records:
                    dataframe = record.dataframes[dataframe_name]
                    signatures[record.dataset_id] = (
                        self.__parameter_signature(dataframe, parameter)
                    )
                for dataset_id, signature in signatures.items():
                    if (
                        signature != ('<not exported>',) and
                        len(signature) > 1
                    ):
                        message = (
                            f"{dataframe_name}.{parameter} contains multiple "
                            f"values within dataset '{dataset_id}': "
                            f'{signature}'
                        )
                        rows.append({
                            'severity': (
                                'error' if self.__is_unit_parameter(parameter)
                                else 'warning'
                            ),
                            'status': 'within_dataset_parameter_mismatch',
                            'dataframe': dataframe_name,
                            'parameter': parameter,
                            'datasets': dataset_id,
                            'values': str(signature),
                            'message': message,
                        })
                present_signatures = [
                    signature for signature in signatures.values()
                    if signature != ('<not exported>',)
                ]
                if not present_signatures:
                    values_text = '; '.join(
                        f'{dataset}=<not exported>'
                        for dataset in signatures
                    )
                    rows.append({
                        'severity': 'warning',
                        'status': 'not_verifiable',
                        'dataframe': dataframe_name,
                        'parameter': parameter,
                        'datasets': ', '.join(signatures),
                        'values': values_text,
                        'message': (
                            f'{dataframe_name}.{parameter} was not exported '
                            'by any participating run and cannot be verified.'
                        ),
                    })
                    continue
                missing_metadata = any(
                    signature == ('<not exported>',)
                    for signature in signatures.values()
                )
                first_signature = present_signatures[0]
                values_match = all(
                    self.__signatures_equal(first_signature, signature)
                    for signature in present_signatures[1:]
                )
                if values_match and not missing_metadata:
                    continue
                status = (
                    'not_verifiable' if missing_metadata
                    else 'parameter_mismatch'
                )
                severity = (
                    'error' if self.__is_unit_parameter(parameter)
                    and not values_match else 'warning'
                )
                values_text = '; '.join(
                    f'{dataset}={signature}'
                    for dataset, signature in signatures.items()
                )
                message = (
                    f"{dataframe_name}.{parameter} is not comparable across "
                    f'loaded runs: {values_text}'
                )
                rows.append({
                    'severity': severity,
                    'status': status,
                    'dataframe': dataframe_name,
                    'parameter': parameter,
                    'datasets': ', '.join(signatures),
                    'values': values_text,
                    'message': message,
                })
        rows.extend(self.__range_metadata_validation_rows())
        rows.extend(self.__calibration_metadata_validation_rows())
        rows.extend(self.__companion_validation_rows())
        return pd.DataFrame(rows, columns=self.VALIDATION_COLUMNS)

    def __range_metadata_validation_rows(self) -> list[dict]:
        """Validate paired requested-range metadata when it is exported."""
        rows = []
        range_specs = {
            'fitted_mean_speeds': (
                (
                    'requested_fit_range_start',
                    'requested_fit_range_end',
                    'requested fit range',
                    True,
                    None,
                ),
                (
                    'ci_range_lower_percentile',
                    'ci_range_upper_percentile',
                    'CI percentile range',
                    False,
                    (0.0, 100.0),
                ),
            ),
            'msd_fit': ((
                'requested_alpha_fit_lag_start',
                'requested_alpha_fit_lag_end',
                'requested alpha-fit lag range',
                True,
                None,
            ),),
        }
        for record in self._datasets:
            for dataframe_name, specifications in range_specs.items():
                dataframe = record.dataframes.get(dataframe_name)
                if dataframe is None:
                    continue
                for (
                    lower_column, upper_column, label, allow_blank, limits
                ) in specifications:
                    columns_present = {
                        lower_column: lower_column in dataframe.columns,
                        upper_column: upper_column in dataframe.columns,
                    }
                    if not any(columns_present.values()):
                        continue
                    if not all(columns_present.values()):
                        invalid_count = len(dataframe)
                    else:
                        lower = self.__numeric_series(
                            dataframe[lower_column]
                        )
                        upper = self.__numeric_series(
                            dataframe[upper_column]
                        )
                        lower_blank = dataframe[lower_column].isna() | (
                            dataframe[lower_column].astype(str).str.strip() == ''
                        )
                        upper_blank = dataframe[upper_column].isna() | (
                            dataframe[upper_column].astype(str).str.strip() == ''
                        )
                        valid = (
                            ~lower_blank & ~upper_blank &
                            np.isfinite(lower) & np.isfinite(upper) &
                            (lower < upper)
                        )
                        if limits is not None:
                            valid &= (
                                (lower >= limits[0]) &
                                (upper <= limits[1])
                            )
                        if allow_blank:
                            valid |= lower_blank & upper_blank
                        invalid_count = int((~valid).sum())
                    if not invalid_count:
                        continue
                    message = (
                        f"Dataset '{record.dataset_id}' {dataframe_name} has "
                        f'{invalid_count} row(s) with invalid {label} metadata.'
                    )
                    rows.append({
                        'severity': 'error',
                        'status': 'invalid_range_metadata',
                        'dataframe': dataframe_name,
                        'parameter': f'{lower_column}, {upper_column}',
                        'datasets': record.dataset_id,
                        'values': f'{invalid_count} invalid rows',
                        'message': message,
                    })
        return rows

    def __calibration_metadata_validation_rows(self) -> list[dict]:
        """Validate exported acquisition-calibration metadata."""
        rows = []
        dataframe_names = (
            'fitted_mean_speeds', 'individual_msd', 'ensemble_msd', 'msd_fit'
        )
        for record in self._datasets:
            for dataframe_name in dataframe_names:
                dataframe = record.dataframes.get(dataframe_name)
                if dataframe is None:
                    continue
                for parameter in (
                    'capture_speed_in_fps', 'pixel_scale_factor'
                ):
                    if parameter not in dataframe.columns:
                        continue
                    numeric = pd.to_numeric(
                        dataframe[parameter], errors='coerce'
                    )
                    valid = np.isfinite(numeric) & (numeric > 0)
                    invalid_count = int((~valid).sum())
                    if invalid_count:
                        rows.append({
                            'severity': 'error',
                            'status': 'invalid_calibration_metadata',
                            'dataframe': dataframe_name,
                            'parameter': parameter,
                            'datasets': record.dataset_id,
                            'values': f'{invalid_count} invalid rows',
                            'message': (
                                f"Dataset '{record.dataset_id}' "
                                f'{dataframe_name} has {invalid_count} row(s) '
                                f'with non-positive or non-finite {parameter}.'
                            ),
                        })
                if 'scale_units' not in dataframe.columns:
                    continue
                scale_units = dataframe['scale_units']
                valid = scale_units.notna() & (
                    scale_units.astype(str).str.strip() != ''
                )
                invalid_count = int((~valid).sum())
                if invalid_count:
                    rows.append({
                        'severity': 'error',
                        'status': 'invalid_calibration_metadata',
                        'dataframe': dataframe_name,
                        'parameter': 'scale_units',
                        'datasets': record.dataset_id,
                        'values': f'{invalid_count} invalid rows',
                        'message': (
                            f"Dataset '{record.dataset_id}' "
                            f'{dataframe_name} has {invalid_count} row(s) '
                            'with blank scale_units.'
                        ),
                    })
        return rows

    def __companion_validation_rows(self) -> list[dict]:
        rows = []
        paired_parameters = (
            ('source_dataframe', 'source_dataframe'),
            ('discard_initial_frames', 'discard_initial_frames'),
            ('discard_final_frames', 'discard_final_frames'),
            ('distance_unit', 'length_unit'),
        )
        for record in self._datasets:
            angle = record.dataframes.get('angle_tumble_summary')
            characteristics = record.dataframes.get(
                'particle_characteristics'
            )
            if angle is None or characteristics is None:
                continue
            for angle_parameter, characteristic_parameter in (
                paired_parameters
            ):
                left = self.__parameter_signature(angle, angle_parameter)
                right = self.__parameter_signature(
                    characteristics, characteristic_parameter
                )
                missing = (
                    left == ('<not exported>',) or
                    right == ('<not exported>',)
                )
                matches = (
                    not missing and self.__signatures_equal(left, right)
                )
                if matches:
                    continue
                message = (
                    f"Dataset '{record.dataset_id}' angle analysis "
                    f'{angle_parameter}={left} does not verifiably match '
                    f'Particle Characteristics '
                    f'{characteristic_parameter}={right}.'
                )
                rows.append({
                    'severity': 'warning' if missing else 'error',
                    'status': (
                        'not_verifiable' if missing
                        else 'companion_parameter_mismatch'
                    ),
                    'dataframe': 'angle_tumble_summary',
                    'parameter': (
                        f'{angle_parameter} vs '
                        f'particle_characteristics.{characteristic_parameter}'
                    ),
                    'datasets': record.dataset_id,
                    'values': f'angle={left}; characteristics={right}',
                    'message': message,
                })
            angle_ids = (
                set(angle['cell_id']) if 'cell_id' in angle.columns else set()
            )
            characteristic_ids = (
                set(characteristics['cell_id'])
                if 'cell_id' in characteristics.columns else set()
            )
            duplicate_angle = int(angle['cell_id'].duplicated().sum())
            duplicate_characteristics = int(
                characteristics['cell_id'].duplicated().sum()
            )
            angle_only_ids = angle_ids - characteristic_ids
            characteristics_only_ids = characteristic_ids - angle_ids
            angle_duplicate_particle_ids = self.__particle_ids(
                angle.loc[
                    angle['cell_id'].duplicated(keep=False), 'particle'
                ]
            )
            characteristics_duplicate_particle_ids = self.__particle_ids(
                characteristics.loc[
                    characteristics['cell_id'].duplicated(keep=False),
                    'particle',
                ]
            )
            angle_only_particle_ids = self.__particle_ids(
                angle.loc[angle['cell_id'].isin(angle_only_ids), 'particle']
            )
            characteristics_only_particle_ids = self.__particle_ids(
                characteristics.loc[
                    characteristics['cell_id'].isin(
                        characteristics_only_ids
                    ),
                    'particle',
                ]
            )
            if (
                duplicate_angle or duplicate_characteristics or
                angle_ids != characteristic_ids
            ):
                message = (
                    f"Dataset '{record.dataset_id}' angle summary and "
                    'Particle Characteristics do not have a one-to-one cell '
                    f'identity match (angle duplicates={duplicate_angle}, '
                    'characteristic duplicates='
                    f'{duplicate_characteristics}, angle-only='
                    f'{len(angle_only_ids)}, characteristic-only='
                    f'{len(characteristics_only_ids)}).'
                )
                rows.append({
                    'severity': 'error',
                    'status': 'companion_cell_mismatch',
                    'dataframe': 'angle_tumble_summary',
                    'parameter': 'cell_id coverage and uniqueness',
                    'datasets': record.dataset_id,
                    'values': (
                        'not one-to-one; angle_only_particle_ids='
                        f'{angle_only_particle_ids}; '
                        'characteristics_only_particle_ids='
                        f'{characteristics_only_particle_ids}; '
                        'angle_duplicate_particle_ids='
                        f'{angle_duplicate_particle_ids}; '
                        'characteristics_duplicate_particle_ids='
                        f'{characteristics_duplicate_particle_ids}'
                    ),
                    'message': message,
                })
        msd_pairs = (
            ('source_dataframe', 'source_dataframe'),
            ('time_unit', 'time_unit'),
            ('msd_unit', 'msd_unit'),
            ('capture_speed_in_fps', 'capture_speed_in_fps'),
            ('pixel_scale_factor', 'pixel_scale_factor'),
            ('scale_units', 'scale_units'),
        )
        for record in self._datasets:
            fit = record.dataframes.get('msd_fit')
            if fit is None:
                continue
            for companion_name in ('individual_msd', 'ensemble_msd'):
                companion = record.dataframes.get(companion_name)
                if companion is None:
                    continue
                for fit_parameter, companion_parameter in msd_pairs:
                    left = self.__parameter_signature(fit, fit_parameter)
                    right = self.__parameter_signature(
                        companion, companion_parameter
                    )
                    missing = (
                        left == ('<not exported>',) or
                        right == ('<not exported>',)
                    )
                    if (
                        left == ('<not exported>',) and
                        right == ('<not exported>',)
                    ):
                        continue
                    matches = (
                        not missing and self.__signatures_equal(left, right)
                    )
                    if matches:
                        continue
                    message = (
                        f"Dataset '{record.dataset_id}' MSD Fit "
                        f'{fit_parameter}={left} does not verifiably match '
                        f'{companion_name}.{companion_parameter}={right}.'
                    )
                    rows.append({
                        'severity': 'warning' if missing else 'error',
                        'status': (
                            'not_verifiable' if missing
                            else 'companion_parameter_mismatch'
                        ),
                        'dataframe': 'msd_fit',
                        'parameter': (
                            f'{fit_parameter} vs '
                            f'{companion_name}.{companion_parameter}'
                        ),
                        'datasets': record.dataset_id,
                        'values': f'fit={left}; {companion_name}={right}',
                        'message': message,
                    })
        velocity_pairs = (
            ('source_dataframe', 'source_dataframe'),
            ('distance_unit', 'distance_unit'),
            ('speed_unit', 'speed_unit'),
            ('angular_velocity_unit', 'angular_velocity_unit'),
            ('smoothing_method', 'smoothing_method'),
        )
        for record in self._datasets:
            summary = record.dataframes.get('velocity_tumble_summary')
            if summary is None:
                continue
            for companion_name in (
                'velocity_tumble_points', 'velocity_tumble_events'
            ):
                companion = record.dataframes.get(companion_name)
                if companion is None:
                    continue
                for summary_parameter, companion_parameter in velocity_pairs:
                    left = self.__parameter_signature(
                        summary, summary_parameter
                    )
                    right = self.__parameter_signature(
                        companion, companion_parameter
                    )
                    missing = (
                        left == ('<not exported>',) or
                        right == ('<not exported>',)
                    )
                    matches = (
                        not missing and self.__signatures_equal(left, right)
                    )
                    if matches:
                        continue
                    message = (
                        f"Dataset '{record.dataset_id}' Velocity Tumble "
                        f'Summary {summary_parameter}={left} does not '
                        f'verifiably match {companion_name}.'
                        f'{companion_parameter}={right}.'
                    )
                    rows.append({
                        'severity': 'warning' if missing else 'error',
                        'status': (
                            'not_verifiable' if missing
                            else 'companion_parameter_mismatch'
                        ),
                        'dataframe': companion_name,
                        'parameter': (
                            f'{companion_parameter} vs '
                            f'velocity_tumble_summary.{summary_parameter}'
                        ),
                        'datasets': record.dataset_id,
                        'values': (
                            f'summary={left}; {companion_name}={right}'
                        ),
                        'message': message,
                    })
        return rows

    @staticmethod
    def __parameter_signature(
        dataframe: pd.DataFrame,
        parameter: str,
    ) -> tuple:
        if parameter not in dataframe.columns:
            return ('<not exported>',)
        normalized_values = []
        for value in dataframe[parameter].drop_duplicates().tolist():
            if pd.isna(value) or (isinstance(value, str) and not value.strip()):
                normalized_value = None
            elif isinstance(value, (int, float, np.integer, np.floating)):
                normalized_value = float(value)
            else:
                normalized_value = str(value).strip().casefold()
            if normalized_value not in normalized_values:
                normalized_values.append(normalized_value)
        return tuple(normalized_values) if normalized_values else (None,)

    @staticmethod
    def __signatures_equal(left: tuple, right: tuple) -> bool:
        if len(left) != len(right):
            return False
        unmatched = list(right)
        for left_value in left:
            matched_index = None
            for index, right_value in enumerate(unmatched):
                if left_value is None and right_value is None:
                    matched_index = index
                    break
                if isinstance(left_value, float) and isinstance(
                    right_value, float
                ) and np.isclose(left_value, right_value, equal_nan=True):
                    matched_index = index
                    break
                if left_value == right_value:
                    matched_index = index
                    break
            if matched_index is None:
                return False
            unmatched.pop(matched_index)
        return not unmatched

    @staticmethod
    def __is_unit_parameter(parameter: str) -> bool:
        return parameter in {
            'distance_unit', 'speed_unit', 'angular_velocity_unit',
            'time_unit', 'msd_unit', 'length_unit',
        }

    def __angular_msd_dataset_values(self) -> pd.DataFrame:
        """Extract one validated p/Dr pair from each loaded Table 1."""
        rows = []
        for record in self._datasets:
            table = record.dataframes['velocity_tumble_table_1']
            parameters = table['Parameter'].astype(str).str.strip()

            def matching_values(parameter: str) -> pd.Series:
                matching_values = table.loc[
                    parameters == parameter, 'Value'
                ]
                if len(matching_values) > 1:
                    raise ValueError(
                        f"Dataset '{record.dataset_id}' must contain at most "
                        f"one '{parameter}' row in its Velocity Tumble Table "
                        f'1 export; found {len(matching_values)}.'
                    )
                return matching_values

            def table_value(
                parameter: str,
                required: bool = True,
            ) -> float:
                values = matching_values(parameter)
                if values.empty:
                    if not required:
                        return np.nan
                    raise ValueError(
                        f"Dataset '{record.dataset_id}' must contain exactly "
                        f"one '{parameter}' row in its Velocity Tumble Table "
                        '1 export; found 0.'
                    )
                numeric_value = pd.to_numeric(
                    values.iloc[0], errors='coerce'
                )
                if pd.isna(numeric_value) or not np.isfinite(numeric_value):
                    raise ValueError(
                        f"Dataset '{record.dataset_id}' has a nonfinite "
                        f"'{parameter}' value in its Velocity Tumble Table 1 "
                        'export.'
                    )
                return float(numeric_value)

            persistence = table_value('p')
            status_values = matching_values('Dr fit status')
            if status_values.empty:
                fit_status = 'not_exported'
            else:
                fit_status = str(status_values.iloc[0]).strip()
                if not fit_status:
                    raise ValueError(
                        f"Dataset '{record.dataset_id}' has a blank "
                        "'Dr fit status' value."
                    )
                if fit_status != 'fitted':
                    raise ValueError(
                        f"Dataset '{record.dataset_id}' has Dr fit status "
                        f"'{fit_status}', so no fitted rotational-diffusion "
                        'coefficient is available for comparison.'
                    )
            rotational_diffusion = table_value(
                self.ANGULAR_MSD_DR_PARAMETER
            )
            fit_max_lag = table_value('Dr fit max lag (s)', required=False)
            largest_fit_lag = table_value(
                'Dr largest fitted lag (s)', required=False
            )
            fit_lag_count = table_value(
                'Dr fit lag count', required=False
            )
            fit_pair_count = table_value(
                'Dr fit direction-pair count', required=False
            )
            fit_r_squared = table_value(
                'Dr fit uncentered R^2', required=False
            )
            if not -1 <= persistence <= 1:
                raise ValueError(
                    f"Dataset '{record.dataset_id}' has p={persistence}; "
                    'directional persistence must be between -1 and 1.'
                )
            if rotational_diffusion < 0:
                raise ValueError(
                    f"Dataset '{record.dataset_id}' has "
                    f'Dr={rotational_diffusion} rad²/s; rotational diffusion '
                    'must be nonnegative.'
                )
            diagnostic_values = (
                fit_max_lag, largest_fit_lag, fit_lag_count,
                fit_pair_count, fit_r_squared,
            )
            fit_diagnostics_verified = (
                fit_status == 'fitted' and
                all(np.isfinite(value) for value in diagnostic_values)
            )
            invalid_lag_counts = (
                np.isfinite(fit_lag_count) and (
                    fit_lag_count < 2 or
                    not float(fit_lag_count).is_integer()
                )
            )
            invalid_pair_count = (
                np.isfinite(fit_pair_count) and (
                    fit_pair_count < 2 or
                    not float(fit_pair_count).is_integer()
                )
            )
            inconsistent_counts = (
                np.isfinite(fit_lag_count) and
                np.isfinite(fit_pair_count) and
                fit_pair_count < fit_lag_count
            )
            invalid_fit_range = (
                (np.isfinite(fit_max_lag) and fit_max_lag <= 0) or
                (np.isfinite(largest_fit_lag) and largest_fit_lag <= 0) or
                (
                    np.isfinite(fit_max_lag) and
                    np.isfinite(largest_fit_lag) and
                    largest_fit_lag > fit_max_lag + 1e-12
                )
            )
            if (
                invalid_lag_counts or invalid_pair_count or
                inconsistent_counts or invalid_fit_range
            ):
                raise ValueError(
                    f"Dataset '{record.dataset_id}' has inconsistent Dr "
                    'fit-support diagnostics in its Velocity Tumble Table 1 '
                    'export.'
                )
            if np.isfinite(fit_r_squared) and fit_r_squared > 1 + 1e-12:
                raise ValueError(
                    f"Dataset '{record.dataset_id}' has an invalid "
                    f'uncentered Dr fit R^2 value of {fit_r_squared}.'
                )
            velocity_summary_available = (
                'velocity_tumble_summary' in record.dataframes
            )
            rows.append({
                'strain': record.strain,
                'dataset_id': record.dataset_id,
                'dataset_source': record.source,
                'velocity_tumble_table_1_path': record.paths[
                    'velocity_tumble_table_1'
                ],
                'dataset_p': persistence,
                'dataset_Dr': rotational_diffusion,
                'Dr_fit_status': fit_status,
                'Dr_fit_max_lag_seconds': fit_max_lag,
                'Dr_largest_fitted_lag_seconds': largest_fit_lag,
                'Dr_fit_lag_count': fit_lag_count,
                'Dr_fit_direction_pair_count': fit_pair_count,
                'Dr_fit_uncentered_r_squared': fit_r_squared,
                'Dr_fit_diagnostics_verified': fit_diagnostics_verified,
                'velocity_tumble_summary_available': (
                    velocity_summary_available
                ),
                'velocity_tumble_summary_path': record.paths.get(
                    'velocity_tumble_summary'
                ),
            })
        return pd.DataFrame(rows)

    def __prepare_comparison(self, comparison_name: str) -> None:
        if not self._datasets:
            raise ValueError('Load at least two strains before comparing them.')
        required_dataframes = self.COMPARISON_REQUIRED_DATAFRAMES[
            comparison_name
        ]
        missing_by_dataset = {
            record.dataset_id: [
                name for name in required_dataframes
                if name not in record.dataframes
            ]
            for record in self._datasets
        }
        missing_by_dataset = {
            dataset_id: names
            for dataset_id, names in missing_by_dataset.items()
            if names
        }
        if missing_by_dataset:
            details = '; '.join(
                f'{dataset_id}: {", ".join(names)}'
                for dataset_id, names in missing_by_dataset.items()
            )
            table_note = ''
            if comparison_name in {'angle', 'velocity'}:
                table_note = (
                    ' Final Table 1 CSV files alone cannot be pooled exactly; '
                    'load the underlying per-particle summary files'
                    + (
                        ' and Particle Characteristics.'
                        if comparison_name == 'angle' else '.'
                    )
                )
            message = (
                f'The {comparison_name} comparison cannot include every '
                f'loaded run because required dataframes are missing: '
                f'{details}.{table_note}'
            )
            warnings.warn(message, UserWarning, stacklevel=3)
            raise ValueError(message)
        loaded_strains = {record.strain for record in self._datasets}
        if len(loaded_strains) < 2:
            raise ValueError(
                'At least two distinct declared strains are required for a '
                'comparison.'
            )
        required_keys = self.COMPARISON_PARAMETER_KEYS[comparison_name]
        strain_issues = pd.DataFrame(
            self._strain_validation_rows,
            columns=self.VALIDATION_COLUMNS,
        )
        if not strain_issues.empty:
            strain_issues = strain_issues.loc[
                (strain_issues['parameter'] == 'strain') |
                (strain_issues['dataframe'].isin(required_keys))
            ]
        if not strain_issues.empty:
            self.__emit_validation_warning(
                strain_issues,
                context=f'{comparison_name} comparison',
            )
            if (strain_issues['status'] == 'strain_mismatch').any():
                raise ValueError(
                    'Cannot compare datasets whose declared strain conflicts '
                    'with embedded strain metadata. Review '
                    'get_validation_report().'
                )
        issues = self._parameter_validation_dataframe.loc[
            self._parameter_validation_dataframe['dataframe'].isin(
                required_keys
            )
        ]
        if not issues.empty:
            self.__emit_validation_warning(
                issues,
                context=f'{comparison_name} comparison',
            )
        unit_errors = issues.loc[issues['severity'] == 'error']
        if not unit_errors.empty:
            parameters = ', '.join(sorted(unit_errors['parameter'].unique()))
            raise ValueError(
                'Cannot pool numerical results with incompatible metadata: '
                f'{parameters}.'
            )

    @staticmethod
    def __emit_validation_warning(
        issues: pd.DataFrame,
        context: str,
    ) -> None:
        warnable = issues.loc[issues['severity'].isin(['warning', 'error'])]
        if warnable.empty:
            return
        messages = warnable['message'].drop_duplicates().tolist()
        preview = messages[:8]
        suffix = (
            f' (+{len(messages) - len(preview)} additional issues)'
            if len(messages) > len(preview) else ''
        )
        warnings.warn(
            f'Compatibility warning during {context}: ' +
            ' | '.join(preview) + suffix,
            UserWarning,
            stacklevel=3,
        )

    def __optional_combined_dataframe(
        self, dataframe_name: str
    ) -> pd.DataFrame:
        try:
            return self.get_combined_dataframe(dataframe_name)
        except ValueError:
            return pd.DataFrame()

    @staticmethod
    def __numeric_series(series: pd.Series) -> pd.Series:
        return pd.to_numeric(series, errors='coerce').replace(
            [np.inf, -np.inf], np.nan
        )

    def __finite_values(self, values: pd.Series | Sequence) -> np.ndarray:
        numeric = pd.to_numeric(
            pd.Series(values), errors='coerce'
        ).to_numpy(dtype=float)
        return numeric[np.isfinite(numeric)]

    @staticmethod
    def __safe_mean(values: np.ndarray) -> float:
        finite = values[np.isfinite(values)]
        return float(np.mean(finite)) if len(finite) else np.nan

    @staticmethod
    def __safe_std(values: np.ndarray) -> float:
        finite = values[np.isfinite(values)]
        return float(np.std(finite, ddof=1)) if len(finite) >= 2 else np.nan

    def __safe_sem(self, values: np.ndarray) -> float:
        finite = values[np.isfinite(values)]
        standard_deviation = self.__safe_std(finite)
        if len(finite) < 2 or not np.isfinite(standard_deviation):
            return np.nan
        return float(standard_deviation / np.sqrt(len(finite)))

    @staticmethod
    def __single_unit(
        dataframe: pd.DataFrame,
        column: str,
        fallback: str,
    ) -> str:
        if column not in dataframe.columns:
            return fallback
        values = [
            str(value).strip()
            for value in dataframe[column].dropna().unique()
            if str(value).strip()
        ]
        if not values:
            return fallback
        return values[0] if len(set(values)) == 1 else 'mixed'

    @staticmethod
    def __plot_unit_label(unit: object) -> str:
        """Use the Greek mu in plot-only micrometre unit labels."""
        return re.sub(
            r'(?<![A-Za-z])(?:u|\N{MICRO SIGN}|\N{GREEK SMALL LETTER MU})m'
            r'(?![A-Za-z])',
            '\N{GREEK SMALL LETTER MU}m',
            str(unit),
            flags=re.IGNORECASE,
        )

    def __require_compatible_units(
        self,
        dataframe: pd.DataFrame,
        unit_columns: Sequence[str],
        context: str,
    ) -> None:
        incompatible = []
        unavailable = []
        for column in unit_columns:
            if column not in dataframe.columns:
                unavailable.append(column)
                continue
            values = {
                str(value).strip()
                for value in dataframe[column].dropna()
                if str(value).strip()
            }
            if len(values) > 1:
                incompatible.append(f'{column}={sorted(values)}')
            elif not values:
                unavailable.append(column)
        if incompatible:
            raise ValueError(
                f'Cannot perform {context} with incompatible units: '
                + ', '.join(incompatible)
            )
        if unavailable:
            warnings.warn(
                f'Unit metadata could not be fully verified for {context}: '
                f'{unavailable}.',
                UserWarning,
                stacklevel=3,
            )

    @staticmethod
    def __append_table_value(
        rows: list[dict],
        strain: str,
        parameter: str,
        value: object,
        sample_count: int | float,
        dataset_count: int,
    ) -> None:
        rows.append({
            'strain': strain,
            'Parameter': parameter,
            'Value': value,
            'sample_count': sample_count,
            'dataset_count': dataset_count,
        })

    def __append_distribution_rows(
        self,
        rows: list[dict],
        strain: str,
        label: str,
        values: np.ndarray,
        dataset_count: int,
    ) -> None:
        sample_count = len(values)
        self.__append_table_value(
            rows, strain, label, self.__safe_mean(values),
            sample_count, dataset_count
        )
        self.__append_table_value(
            rows, strain, '±std', self.__safe_std(values),
            sample_count, dataset_count
        )
        self.__append_table_value(
            rows, strain, '±sem', self.__safe_sem(values),
            sample_count, dataset_count
        )

    def __weighted_mean_from_columns(
        self,
        dataframe: pd.DataFrame,
        value_column: str,
        weight_column: str,
    ) -> tuple[float, int]:
        if (
            value_column not in dataframe.columns or
            weight_column not in dataframe.columns
        ):
            return np.nan, 0
        values = self.__numeric_series(dataframe[value_column])
        weights = self.__numeric_series(dataframe[weight_column])
        valid = (
            np.isfinite(values) & np.isfinite(weights) & (weights > 0)
        )
        if not valid.any():
            return np.nan, 0
        weighted_mean = float(np.average(values[valid], weights=weights[valid]))
        return weighted_mean, int(round(float(weights[valid].sum())))

    def __combine_group_standard_deviation(
        self,
        dataframe: pd.DataFrame,
        mean_column: str,
        std_column: str,
        count_column: str,
    ) -> tuple[float, float]:
        if not {mean_column, std_column, count_column}.issubset(
            dataframe.columns
        ):
            warnings.warn(
                f'Exact pooled uncertainty is unavailable because '
                f'{std_column} was not exported. Rerun the source notebooks '
                'with the current Stats implementation.',
                UserWarning,
                stacklevel=3,
            )
            return np.nan, np.nan
        means = self.__numeric_series(dataframe[mean_column])
        standard_deviations = self.__numeric_series(dataframe[std_column])
        counts = self.__numeric_series(dataframe[count_column])
        valid = np.isfinite(means) & np.isfinite(counts) & (counts > 0)
        if not valid.any():
            return np.nan, np.nan
        means = means[valid].to_numpy(dtype=float)
        standard_deviations = standard_deviations[valid].to_numpy(dtype=float)
        counts = counts[valid].to_numpy(dtype=float)
        missing_standard_deviation = (
            (counts > 1) & ~np.isfinite(standard_deviations)
        )
        if missing_standard_deviation.any():
            affected = dataframe.loc[valid].loc[
                missing_standard_deviation, 'dataset_id'
            ].astype(str).unique()
            warnings.warn(
                f'Exact pooled uncertainty is unavailable because '
                f'{std_column} is missing for datasets: '
                f'{", ".join(affected)}. Rerun those source notebooks with '
                'the current Stats implementation.',
                UserWarning,
                stacklevel=3,
            )
            return np.nan, np.nan
        total_count = float(counts.sum())
        pooled_mean = float(np.average(means, weights=counts))
        within_sum_squares = np.where(
            counts > 1,
            (counts - 1) * np.square(standard_deviations),
            0.0,
        )
        total_sum_squares = float(np.sum(
            within_sum_squares + counts * np.square(means - pooled_mean)
        ))
        if total_count < 2:
            return np.nan, np.nan
        standard_deviation = float(np.sqrt(
            total_sum_squares / (total_count - 1)
        ))
        return standard_deviation, float(
            standard_deviation / np.sqrt(total_count)
        )

    def __build_velocity_state_episodes(self) -> pd.DataFrame:
        points = self.__optional_combined_dataframe(
            'velocity_tumble_points'
        )
        output_columns = [
            'strain', 'dataset_id', 'cell_id', 'particle', 'segment_id',
            'episode_index', 'state', 'start_frame', 'end_frame',
            'point_count', 'interval_seconds', 'is_boundary_censored',
        ]
        if points.empty:
            return pd.DataFrame(columns=output_columns)
        required = {
            'strain', 'dataset_id', 'particle', 'segment_id', 'frame',
            'elapsed_time_seconds', 'event_id',
            'smoothed_position_x_pixels', 'smoothed_position_y_pixels',
            'angular_velocity_magnitude_radians_per_second',
        }
        missing = required.difference(points.columns)
        if missing:
            raise ValueError(
                'Velocity point data are missing columns required to '
                f'reconstruct state episodes: {sorted(missing)}'
            )
        rows: list[dict] = []
        group_columns = ['strain', 'dataset_id', 'particle', 'segment_id']
        for group_key, segment_rows in points.groupby(
            group_columns, sort=False, dropna=False
        ):
            segment_rows = segment_rows.sort_values(
                'frame', kind='stable'
            ).reset_index(drop=True)
            elapsed = self.__numeric_series(
                segment_rows['elapsed_time_seconds']
            )
            smoothed_x = self.__numeric_series(
                segment_rows['smoothed_position_x_pixels']
            )
            smoothed_y = self.__numeric_series(
                segment_rows['smoothed_position_y_pixels']
            )
            angular_velocity = self.__numeric_series(
                segment_rows[
                    'angular_velocity_magnitude_radians_per_second'
                ]
            )
            supported = (
                np.isfinite(elapsed) & np.isfinite(smoothed_x) &
                np.isfinite(smoothed_y)
            )
            if int(np.isfinite(angular_velocity).sum()) < 5 or not supported.any():
                continue
            supported_rows = segment_rows.loc[supported].copy()
            supported_rows['_elapsed'] = elapsed.loc[supported].to_numpy()
            segment_start_time = float(supported_rows['_elapsed'].iloc[0])
            segment_end_time = float(supported_rows['_elapsed'].iloc[-1])
            segment_start_frame = int(supported_rows['frame'].iloc[0])
            segment_end_frame = int(supported_rows['frame'].iloc[-1])
            cell_id = (
                str(segment_rows['cell_id'].iloc[0])
                if 'cell_id' in segment_rows.columns
                else f'{group_key[1]}:{group_key[2]}'
            )
            event_rows = supported_rows.loc[
                pd.to_numeric(
                    supported_rows['event_id'], errors='coerce'
                ).notna()
            ]
            events = []
            for event_id, event_points in event_rows.groupby(
                'event_id', sort=False, dropna=True
            ):
                event_points = event_points.sort_values(
                    '_elapsed', kind='stable'
                )
                start_time = float(event_points['_elapsed'].iloc[0])
                end_time = float(event_points['_elapsed'].iloc[-1])
                events.append({
                    'event_id': event_id,
                    'start_time': start_time,
                    'end_time': end_time,
                    'start_frame': int(event_points['frame'].iloc[0]),
                    'end_frame': int(event_points['frame'].iloc[-1]),
                    'point_count': int(len(event_points)),
                    'is_complete': bool(
                        start_time > segment_start_time and
                        end_time < segment_end_time
                    ),
                })
            events.sort(key=lambda event: event['start_time'])

            segment_episode_rows: list[dict] = []
            cursor_time = segment_start_time
            cursor_frame = segment_start_frame
            previous_event_was_complete = False
            for event_position, event in enumerate(events):
                if event['start_time'] > cursor_time:
                    run_is_complete = bool(
                        event_position > 0 and
                        previous_event_was_complete and
                        event['is_complete']
                    )
                    segment_episode_rows.append({
                        'state': 'run',
                        'start_frame': cursor_frame,
                        'end_frame': event['start_frame'],
                        'point_count': int((
                            (supported_rows['_elapsed'] >= cursor_time) &
                            (supported_rows['_elapsed'] <= event['start_time'])
                        ).sum()),
                        'interval_seconds': float(
                            event['start_time'] - cursor_time
                        ),
                        'is_boundary_censored': not run_is_complete,
                    })
                segment_episode_rows.append({
                    'state': 'tumble',
                    'start_frame': event['start_frame'],
                    'end_frame': event['end_frame'],
                    'point_count': event['point_count'],
                    'interval_seconds': float(
                        event['end_time'] - event['start_time']
                    ),
                    'is_boundary_censored': not event['is_complete'],
                })
                if event['end_time'] >= cursor_time:
                    cursor_time = event['end_time']
                    cursor_frame = event['end_frame']
                previous_event_was_complete = bool(event['is_complete'])
            if segment_end_time > cursor_time or not events:
                segment_episode_rows.append({
                    'state': 'run',
                    'start_frame': cursor_frame,
                    'end_frame': segment_end_frame,
                    'point_count': int((
                        (supported_rows['_elapsed'] >= cursor_time) &
                        (supported_rows['_elapsed'] <= segment_end_time)
                    ).sum()),
                    'interval_seconds': float(
                        segment_end_time - cursor_time
                    ),
                    'is_boundary_censored': True,
                })
            for episode_index, episode_row in enumerate(segment_episode_rows):
                rows.append({
                    'strain': group_key[0],
                    'dataset_id': group_key[1],
                    'cell_id': cell_id,
                    'particle': group_key[2],
                    'segment_id': group_key[3],
                    'episode_index': episode_index,
                    **episode_row,
                })
        return pd.DataFrame(rows, columns=output_columns)

    def __signed_run_to_run_angles(
        self,
        events: pd.DataFrame,
    ) -> np.ndarray:
        required = {
            'preceding_run_direction_radians',
            'following_run_direction_radians',
        }
        if events.empty or not required.issubset(events.columns):
            return np.array([], dtype=float)
        preceding = self.__numeric_series(
            events['preceding_run_direction_radians']
        ).to_numpy(dtype=float)
        following = self.__numeric_series(
            events['following_run_direction_radians']
        ).to_numpy(dtype=float)
        valid = np.isfinite(preceding) & np.isfinite(following)
        if not valid.any():
            return np.array([], dtype=float)
        difference = following[valid] - preceding[valid]
        wrapped_difference = np.arctan2(
            np.sin(difference), np.cos(difference)
        )
        return np.degrees(wrapped_difference)

    def __build_velocity_dataset_summary(
        self,
        summary: pd.DataFrame,
    ) -> pd.DataFrame:
        rows = []
        for (strain, dataset_id), dataset_rows in summary.groupby(
            ['strain', 'dataset_id'], sort=False
        ):
            v_r, _ = self.__weighted_mean_from_columns(
                dataset_rows, 'v_R', 'v_R_support_seconds'
            )
            v_t, _ = self.__weighted_mean_from_columns(
                dataset_rows, 'v_T', 'v_T_support_seconds'
            )
            t_r, _ = self.__weighted_mean_from_columns(
                dataset_rows, 't_R', 't_R_interval_count'
            )
            t_t, _ = self.__weighted_mean_from_columns(
                dataset_rows, 't_T', 't_T_interval_count'
            )
            rows.append({
                'strain': strain,
                'dataset_id': dataset_id,
                'particle_count': int(dataset_rows['particle'].nunique()),
                'v_R': v_r,
                'v_T': v_t,
                't_R': t_r,
                't_T': t_t,
            })
        return pd.DataFrame(rows)

    def __resolve_analysis_plotting_parameters(
        self,
        **per_call_parameters: object,
    ) -> dict:
        """Merge per-call comparison-plot options with stored defaults."""
        resolved = dict(self.ANALYSIS_PLOT_DEFAULTS)
        resolved.update(self.get_analysis_plotting_parameters())
        for argument_name, value in per_call_parameters.items():
            if value is not None:
                resolved[argument_name] = value
        return self.__validate_analysis_plotting_parameters(resolved)

    def __validate_analysis_plotting_parameters(
        self,
        parameters: Mapping[str, object],
    ) -> dict:
        """Validate reusable comparison-plot configuration."""
        validated = dict(parameters)
        title = validated['title']
        if isinstance(title, Mapping):
            for method_name, method_title in title.items():
                if not isinstance(method_name, str):
                    raise TypeError('title mapping keys must be strings.')
                if method_title is not None and not isinstance(
                    method_title, str
                ):
                    raise TypeError(
                        'title mapping values must be strings or None.'
                    )
            validated['title'] = dict(title)
        elif title is not None and not isinstance(title, str):
            raise TypeError('title must be a string, mapping, or None.')

        for argument_name in ('x_axis_titles', 'y_axis_titles'):
            self.__validate_axis_title_setting(
                validated[argument_name], argument_name
            )

        for argument_name in (
            'title_fontsize',
            'x_axis_fontsize',
            'y_axis_fontsize',
            'x_tick_fontsize',
            'y_tick_fontsize',
        ):
            value = validated[argument_name]
            if value is None:
                continue
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, float, np.integer, np.floating)
            ):
                raise TypeError(f'{argument_name} must be numeric or None.')
            if not np.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(
                    f'{argument_name} must be finite and positive.'
                )
            validated[argument_name] = float(value)

        font_family = validated['font_family']
        if font_family is not None:
            if not isinstance(font_family, str):
                raise TypeError('font_family must be a string or None.')
            if not font_family.strip():
                raise ValueError('font_family must not be empty.')
            validated['font_family'] = font_family.strip()

        dpi = validated['dpi']
        if isinstance(dpi, (bool, np.bool_)) or not isinstance(
            dpi, (int, np.integer)
        ):
            raise TypeError('dpi must be a positive integer.')
        if int(dpi) < 1:
            raise ValueError('dpi must be a positive integer.')
        validated['dpi'] = int(dpi)
        validated['strain_colors'] = self.__validate_strain_colors(
            validated['strain_colors']
        )
        return validated

    @classmethod
    def __validate_axis_title_setting(
        cls,
        value: object,
        argument_name: str,
    ) -> None:
        """Validate strings, nested sequences, or per-method mappings."""
        if value is None or isinstance(value, str):
            return
        if isinstance(value, Mapping):
            for method_name, method_value in value.items():
                if not isinstance(method_name, str):
                    raise TypeError(
                        f'{argument_name} mapping keys must be strings.'
                    )
                cls.__validate_axis_title_setting(
                    method_value, argument_name
                )
            return
        if isinstance(value, Sequence) and not isinstance(value, bytes):
            for item in value:
                cls.__validate_axis_title_setting(item, argument_name)
            return
        raise TypeError(
            f'{argument_name} must contain only strings, sequences, '
            'mappings, or None.'
        )

    @staticmethod
    def __validate_strain_colors(
        colors: object,
    ) -> Mapping[str, str] | list[str] | str | None:
        """Validate and copy a reusable strain-color specification."""
        if colors is None:
            return None
        if isinstance(colors, Mapping):
            normalized = dict(colors)
            for strain, color in normalized.items():
                if not isinstance(strain, str):
                    raise TypeError(
                        'strain_colors mapping keys must be strings.'
                    )
                if not is_color_like(color):
                    raise ValueError(
                        f"strain_colors value for '{strain}' is not a valid "
                        'Matplotlib color.'
                    )
            return normalized
        if isinstance(colors, str):
            if is_color_like(colors):
                return colors
            try:
                plt.get_cmap(colors)
            except ValueError as error:
                raise ValueError(
                    f"strain_colors '{colors}' is neither a valid color nor "
                    'a Matplotlib colormap.'
                ) from error
            return colors
        if isinstance(colors, Sequence) and not isinstance(colors, bytes):
            normalized = list(colors)
            if not normalized:
                raise ValueError('strain_colors sequence cannot be empty.')
            for color in normalized:
                if not is_color_like(color):
                    raise ValueError(
                        f"strain_colors contains invalid color '{color}'."
                    )
            return normalized
        raise TypeError(
            'strain_colors must be a mapping, sequence, color, colormap, '
            'or None.'
        )

    @staticmethod
    def __plot_specific_setting(value: object, plot_name: str) -> object:
        """Select a plotting-method-specific value from an optional mapping."""
        if isinstance(value, Mapping):
            return value.get(plot_name)
        return value

    def __apply_axis_title_settings(
        self,
        parameters: Mapping[str, object],
        plot_name: str,
        axis_groups: Sequence[Sequence[plt.Axes]],
    ) -> None:
        """Apply validated x/y titles to one or more logical panels."""
        x_defaults = [[group[0].get_xlabel()] for group in axis_groups]
        y_defaults = [
            [axis.get_ylabel() for axis in group]
            for group in axis_groups
        ]
        x_titles = self.__resolve_axis_title_groups(
            parameters['x_axis_titles'], plot_name, x_defaults,
            'x_axis_titles'
        )
        y_titles = self.__resolve_axis_title_groups(
            parameters['y_axis_titles'], plot_name, y_defaults,
            'y_axis_titles'
        )
        for group, group_x_titles, group_y_titles in zip(
            axis_groups, x_titles, y_titles
        ):
            group[0].set_xlabel(group_x_titles[0])
            for axis, y_title in zip(group, group_y_titles):
                axis.set_ylabel(y_title)

    def __resolve_axis_title_groups(
        self,
        configured_titles: object,
        plot_name: str,
        default_titles: list[list[str]],
        argument_name: str,
    ) -> list[list[str]]:
        """Resolve scalar or panel-aligned axis titles."""
        selected = self.__plot_specific_setting(
            configured_titles, plot_name
        )
        if selected is None:
            return [list(group) for group in default_titles]
        if isinstance(selected, str):
            if len(default_titles) != 1 or len(default_titles[0]) != 1:
                raise ValueError(
                    f'{argument_name} for {plot_name} must provide one entry '
                    'per panel and per axis.'
                )
            return [[selected]]
        selected_panels = list(selected)
        if len(default_titles) == 1 and len(default_titles[0]) > 1:
            if (
                len(selected_panels) == 1 and
                isinstance(selected_panels[0], Sequence) and
                not isinstance(selected_panels[0], (str, bytes))
            ):
                selected_panels = list(selected_panels[0])
            return [self.__resolve_axis_title_group(
                selected_panels, default_titles[0], argument_name, plot_name
            )]
        if len(selected_panels) != len(default_titles):
            raise ValueError(
                f'{argument_name} for {plot_name} must contain '
                f'{len(default_titles)} panel entries.'
            )
        resolved = []
        for panel_titles, panel_defaults in zip(
            selected_panels, default_titles
        ):
            if len(panel_defaults) == 1:
                if panel_titles is None:
                    resolved.append(list(panel_defaults))
                elif isinstance(panel_titles, str):
                    resolved.append([panel_titles])
                else:
                    raise TypeError(
                        f'{argument_name} entries for single-axis panels '
                        'must be strings or None.'
                    )
            else:
                if panel_titles is None:
                    resolved.append(list(panel_defaults))
                elif isinstance(panel_titles, Sequence) and not isinstance(
                    panel_titles, (str, bytes)
                ):
                    resolved.append(self.__resolve_axis_title_group(
                        list(panel_titles), panel_defaults,
                        argument_name, plot_name
                    ))
                else:
                    raise TypeError(
                        f'{argument_name} for a twin-axis panel must contain '
                        'the left and right titles.'
                    )
        return resolved

    @staticmethod
    def __resolve_axis_title_group(
        selected: Sequence,
        defaults: Sequence[str],
        argument_name: str,
        plot_name: str,
    ) -> list[str]:
        if len(selected) != len(defaults):
            raise ValueError(
                f'{argument_name} for {plot_name} must provide '
                f'{len(defaults)} titles for its twin axes.'
            )
        resolved = []
        for title, default in zip(selected, defaults):
            if title is None:
                resolved.append(default)
            elif isinstance(title, str):
                resolved.append(title)
            else:
                raise TypeError(
                    f'{argument_name} entries must be strings or None.'
                )
        return resolved

    def __format_comparison_figure(
        self,
        figure: plt.Figure,
        parameters: Mapping[str, object],
        plot_name: str,
        primary_title_axis: plt.Axes | None,
    ) -> None:
        """Apply title and typography settings without changing global rcParams."""
        selected_title = self.__plot_specific_setting(
            parameters['title'], plot_name
        )
        primary_title_artist = None
        if selected_title is not None:
            if primary_title_axis is None:
                if figure._suptitle is None:
                    primary_title_artist = figure.suptitle(selected_title)
                else:
                    figure._suptitle.set_text(selected_title)
                    primary_title_artist = figure._suptitle
            else:
                primary_title_axis.set_title(selected_title)
                primary_title_artist = primary_title_axis.title
        elif primary_title_axis is not None:
            primary_title_artist = primary_title_axis.title
        elif figure._suptitle is not None:
            primary_title_artist = figure._suptitle

        title_fontsize = parameters['title_fontsize']
        if title_fontsize is not None:
            if primary_title_artist is not None:
                primary_title_artist.set_fontsize(float(title_fontsize))
            for axis in figure.axes:
                axis.title.set_fontsize(float(title_fontsize))
        for axis in figure.axes:
            if parameters['x_axis_fontsize'] is not None:
                axis.xaxis.label.set_fontsize(
                    float(parameters['x_axis_fontsize'])
                )
            if parameters['y_axis_fontsize'] is not None:
                axis.yaxis.label.set_fontsize(
                    float(parameters['y_axis_fontsize'])
                )
            if parameters['x_tick_fontsize'] is not None:
                axis.tick_params(
                    axis='x', labelsize=float(parameters['x_tick_fontsize'])
                )
            if parameters['y_tick_fontsize'] is not None:
                axis.tick_params(
                    axis='y', labelsize=float(parameters['y_tick_fontsize'])
                )
        if parameters['font_family'] is not None:
            font_family = str(parameters['font_family'])
            for text_item in figure.findobj(match=Text):
                text_item.set_fontfamily(font_family)

    @staticmethod
    def __center_categorical_strain_axis(
        axis: plt.Axes,
        strain_count: int,
    ) -> None:
        """Give each categorical strain an equal-width x-axis section."""
        if strain_count > 0:
            axis.set_xlim(-0.5, strain_count - 0.5)

    @staticmethod
    def __resolve_legend_position(position: str | None) -> str:
        """Map centered legend-position names to Matplotlib locations."""
        if position is None:
            return 'upper right'
        if not isinstance(position, str):
            raise TypeError('legend_position must be a string or None.')
        normalized = ' '.join(
            position.strip().lower().replace('_', ' ').split()
        )
        aliases = {
            'middle top': 'upper center',
            'top middle': 'upper center',
            'upper center': 'upper center',
            'middle center': 'center',
            'center': 'center',
            'middle bottom': 'lower center',
            'bottom middle': 'lower center',
            'lower center': 'lower center',
            'upper right': 'upper right',
        }
        if normalized not in aliases:
            raise ValueError(
                "legend_position must be 'middle top', 'middle center', "
                "'middle bottom', or None."
            )
        return aliases[normalized]

    @staticmethod
    def __resolve_panel_b_curve_parameters(
        curve: str | None,
        show_histogram: bool | None,
        resolution: int | None,
    ) -> tuple[str, str, int]:
        """Normalize optional Panel B run-speed display controls."""
        if curve is None:
            method = 'current'
        elif not isinstance(curve, str):
            raise TypeError('panel_b_curve must be a string or None.')
        else:
            aliases = {
                'current': 'current',
                'binned': 'current',
                'normal': 'normal',
                'norm': 'normal',
                'normal_fit': 'normal',
            }
            normalized = curve.strip().lower().replace('-', '_')
            if normalized not in aliases:
                raise ValueError(
                    "panel_b_curve must be 'normal', 'current', or None."
                )
            method = aliases[normalized]

        if show_histogram is not None and not isinstance(
            show_histogram, bool
        ):
            raise TypeError(
                'panel_b_show_histogram must be a boolean or None.'
            )
        if method == 'current':
            if show_histogram is False:
                raise ValueError(
                    'panel_b_show_histogram=False requires '
                    "panel_b_curve='normal'."
                )
            histogram_mode = (
                'current' if show_histogram is None else 'histogram'
            )
        else:
            histogram_mode = (
                'histogram' if show_histogram is not False else 'hidden'
            )

        if resolution is None:
            normalized_resolution = 300
        elif isinstance(resolution, (bool, np.bool_)) or not isinstance(
            resolution, (int, np.integer)
        ):
            raise TypeError(
                'panel_b_curve_resolution must be an integer or None.'
            )
        else:
            normalized_resolution = int(resolution)
            if normalized_resolution < 2:
                raise ValueError(
                    'panel_b_curve_resolution must be at least 2.'
                )
        return method, histogram_mode, normalized_resolution

    @staticmethod
    def __resolve_panel_e_curve_parameters(
        curve: str | None,
        factor: float | str | None,
        resolution: int | None,
    ) -> tuple[str, float | str | None, int]:
        """Normalize optional Panel E turn-angle curve controls."""
        if curve is None:
            method = 'binned'
        elif not isinstance(curve, str):
            raise TypeError('panel_e_curve must be a string or None.')
        else:
            aliases = {
                'binned': 'binned',
                'histogram': 'binned',
                'current': 'binned',
                'kde': 'kde',
                'gaussian_kde': 'kde',
                'normal': 'normal',
                'norm': 'normal',
                'normal_fit': 'normal',
            }
            normalized = curve.strip().lower().replace('-', '_')
            if normalized not in aliases:
                raise ValueError(
                    "panel_e_curve must be 'binned', 'kde', 'normal', or "
                    'None.'
                )
            method = aliases[normalized]

        if resolution is None:
            normalized_resolution = 300
        else:
            if isinstance(resolution, (bool, np.bool_)) or not isinstance(
                resolution, (int, np.integer)
            ):
                raise TypeError(
                    'panel_e_curve_resolution must be an integer or None.'
                )
            normalized_resolution = int(resolution)
            if normalized_resolution < 2:
                raise ValueError(
                    'panel_e_curve_resolution must be at least 2.'
                )

        normalized_factor: float | str | None = factor
        if factor is not None:
            if method != 'kde':
                raise ValueError(
                    'panel_e_curve_factor is only used with '
                    "panel_e_curve='kde'."
                )
            if isinstance(factor, str):
                normalized_factor = factor.strip().lower()
                if normalized_factor not in {'scott', 'silverman'}:
                    raise ValueError(
                        "panel_e_curve_factor must be 'scott', 'silverman', "
                        'a positive number, or None.'
                    )
            elif isinstance(factor, (bool, np.bool_)) or not isinstance(
                factor, (int, float, np.integer, np.floating)
            ):
                raise TypeError(
                    'panel_e_curve_factor must be numeric, a supported '
                    'bandwidth name, or None.'
                )
            else:
                normalized_factor = float(factor)
                if (
                    not np.isfinite(normalized_factor) or
                    normalized_factor <= 0
                ):
                    raise ValueError(
                        'panel_e_curve_factor must be finite and positive.'
                    )
        return method, normalized_factor, normalized_resolution

    @staticmethod
    def __panel_e_curve_values(
        angles: np.ndarray,
        bin_centers: np.ndarray,
        bin_density: np.ndarray,
        method: str,
        factor: float | str | None,
        resolution: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the selected Panel E curve without altering raw exports."""
        if method == 'binned':
            return bin_centers, bin_density
        if len(angles) < 2 or len(np.unique(angles)) < 2:
            raise ValueError(
                f"Panel E curve method '{method}' requires at least two "
                'distinct turning angles per strain.'
            )
        x_values = np.linspace(-180.0, 180.0, resolution)
        if method == 'kde':
            try:
                estimator = gaussian_kde(angles, bw_method=factor)
            except (ValueError, np.linalg.LinAlgError) as error:
                raise ValueError(
                    'Panel E KDE could not be fitted to the turning angles.'
                ) from error
            return x_values, estimator(x_values)

        fitted_location, fitted_scale = norm.fit(angles)
        if not np.isfinite(fitted_scale) or fitted_scale <= 0:
            raise ValueError(
                'Panel E normal curve requires nonzero angle variation.'
            )
        return x_values, norm.pdf(
            x_values, loc=fitted_location, scale=fitted_scale
        )

    @staticmethod
    def __split_figure_save_paths(
        save_path: str | os.PathLike | None,
        panel_labels: Sequence[str],
    ) -> list[Path | None]:
        """Add stable panel suffixes when saving split figures."""
        if save_path is None:
            return [None for _ in panel_labels]
        path = Path(save_path).expanduser()
        if path.exists() and path.is_dir():
            return [
                path / f'Run_and_tumble_comparison_panel_{label}.png'
                for label in panel_labels
            ]
        if not path.suffix:
            path = path.with_suffix('.png')
        return [
            path.with_name(f'{path.stem}_panel_{label}{path.suffix}')
            for label in panel_labels
        ]

    @staticmethod
    def __resolve_colors(
        strains: list[str],
        colors: Mapping[str, str] | Sequence[str] | str | None,
    ) -> dict[str, object]:
        colors = ComparativeStats.__validate_strain_colors(colors)
        if isinstance(colors, Mapping):
            missing = [strain for strain in strains if strain not in colors]
            if missing:
                raise ValueError(
                    f'colors mapping is missing strains: {missing}'
                )
            return {strain: colors[strain] for strain in strains}
        if isinstance(colors, str):
            if is_color_like(colors):
                return {strain: colors for strain in strains}
            try:
                colormap = plt.get_cmap(colors)
            except ValueError as error:
                raise ValueError(
                    f"colors '{colors}' is neither a valid color nor a "
                    'Matplotlib colormap.'
                ) from error
            denominator = max(len(strains) - 1, 1)
            return {
                strain: colormap(index / denominator)
                for index, strain in enumerate(strains)
            }
        if colors is not None and not isinstance(colors, (str, bytes)):
            color_values = list(colors)
            if not color_values:
                raise ValueError('colors sequence cannot be empty.')
            return {
                strain: color_values[index % len(color_values)]
                for index, strain in enumerate(strains)
            }
        colormap = plt.get_cmap('tab10')
        return {
            strain: colormap(index % colormap.N)
            for index, strain in enumerate(strains)
        }

    @staticmethod
    def __resolve_output_path(
        output_path: str | os.PathLike,
        default_filename: str,
    ) -> Path:
        path = Path(output_path).expanduser()
        if path.suffix:
            if path.suffix.lower() != '.csv':
                raise ValueError('Comparison data can only be exported as CSV.')
            resolved = path
        else:
            resolved = path / default_filename
        resolved.parent.mkdir(parents=True, exist_ok=True)
        return resolved

    def __export_dataframe(
        self,
        dataframe: pd.DataFrame,
        output_path: str | os.PathLike,
        default_filename: str,
    ) -> str:
        resolved_path = self.__resolve_output_path(
            output_path, default_filename
        )
        dataframe.to_csv(resolved_path, index=False)
        print(f'Comparison dataframe saved to {resolved_path}')
        return str(resolved_path)

    def __export_cached_dataframe(
        self,
        dataframe: pd.DataFrame,
        output_path: str | os.PathLike,
        default_filename: str,
        producing_method: str,
    ) -> str:
        if dataframe.empty:
            raise ValueError(
                f'Run {producing_method} before exporting its results.'
            )
        return self.__export_dataframe(
            dataframe, output_path, default_filename
        )

    @staticmethod
    def __finalize_figure(
        figure: plt.Figure,
        save_path: str | os.PathLike | None,
        dpi: int,
        show: bool,
    ) -> None:
        if isinstance(dpi, bool) or not isinstance(dpi, int) or dpi < 1:
            raise ValueError('dpi must be a positive integer.')
        if save_path is not None:
            path = Path(save_path).expanduser()
            if not path.suffix:
                path = path.with_suffix('.png')
            path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(path, dpi=dpi, bbox_inches='tight')
        if show:
            plt.show()

    def __reset_comparison_results(self) -> None:
        self._angle_comparison_dataframe = pd.DataFrame()
        self._velocity_comparison_dataframe = pd.DataFrame()
        self._fitted_speed_observations_dataframe = pd.DataFrame()
        self._fitted_speed_comparison_dataframe = pd.DataFrame()
        self._fitted_speed_export_dataframe = pd.DataFrame()
        self._turn_comparison_dataframe = pd.DataFrame()
        self._run_tumble_plot_data = {}
        self._angular_msd_comparison_dataframe = pd.DataFrame()
        self._msd_comparison_dataframe = pd.DataFrame()
