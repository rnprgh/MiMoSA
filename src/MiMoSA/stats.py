"""Tracker-aware morphology, motion, turn, run/tumble, and MSD analyses."""
import os
import re
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import trackpy as tp
from distfit import distfit
from matplotlib.collections import LineCollection
from matplotlib.colors import Colormap, Normalize, is_color_like
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
from matplotlib.text import Text
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.signal import find_peaks, savgol_filter
from scipy.stats import linregress, pearsonr
from tqdm import tqdm, trange

from .track import Tracker  # type: ignore


class _UseConfiguredPlotValue:
    """Represent an omitted per-call plotting option."""

    def __repr__(self) -> str:
        return 'USE_CONFIGURED_PLOT_VALUE'


_USE_CONFIGURED_PLOT_VALUE = _UseConfiguredPlotValue()


class Stats:
    """Analyze a selected linked-particle dataframe from a Tracker object."""

    DEFAULT_DISTRIBUTION = 'norm'
    DEFAULT_SPEED_UNIT = 'scale_units/s'
    SOURCE_DATAFRAMES = {'active', 'full', 'filtered', 'updated'}
    PARTICLE_METRIC_PLOT_DEFAULTS = {
        'title': None,
        'title_fontsize': None,
        'x_axis_fontsize': None,
        'y_axis_fontsize': None,
        'x_tick_fontsize': None,
        'y_tick_fontsize': None,
        'show_particle_ID': True,
        'show_particle_ID_only_with_value': False,
        'particle_ID_spacing': 1,
        'y_tick_spacing': None,
        'font_family': None,
        'dpi': 300,
        'save_plots': False,
        'save_plot_path': None,
        'file_extension': 'png',
    }
    ANALYSIS_PLOT_DEFAULTS = {
        'particle_marker': None,
        'marker_fill_mode': None,
        'track_thickness': 2.0,
        'smooth_trajectory_thickness': 1.5,
        'segment_marker_thickness': 1.0,
        'crop_to_track': False,
        'angle_mode': 'absolute',
        'show_turns': True,
        'turn_color': 'darkviolet',
        'turn_display_mode': 'markers',
        'turn_marker': 'o',
        'show_smoothed_trajectory': True,
        'smoothed_trajectory_color': 'lightblue',
        'show_tumbles': True,
        'tumble_color': '#D55E00',
        'tumble_display_mode': 'markers',
        'tumble_marker': None,
        'show_angle_smoothed_trajectory': False,
        'smoothed_angle_trajectory_color': 'goldenrod',
        'show_velocity_smoothed_trajectory': False,
        'smoothed_velocity_trajectory_color': 'goldenrod',
        'title': None,
        'title_fontsize': None,
        'x_axis_fontsize': None,
        'y_axis_fontsize': None,
        'x_tick_fontsize': None,
        'y_tick_fontsize': None,
        'font_family': None,
        'dpi': 300,
        'save_plots': False,
        'save_plot_path': None,
        'file_extension': 'png',
    }
    TRAJECTORY_SMOOTHING_METHODS = {
        'moving_average',
        'triangular_smoothing',
        'sg_filter_rdp',
    }
    TABLE_1_VERTICAL_COLUMNS = ['Parameter', 'Value']
    VELOCITY_TABLE_1_VERTICAL_COLUMNS = [
        'Parameter', 'Value', 'n', 'n_definition',
    ]
    ANGULAR_MSD_DR_PARAMETER = 'Dr (rad²/s)'
    STEP_METRIC_COLUMNS = [
        'source_dataframe',
        'particle',
        'start_frame',
        'end_frame',
        'frame_delta',
        'elapsed_time',
        'start_centroid_x_pixels',
        'start_centroid_y_pixels',
        'end_centroid_x_pixels',
        'end_centroid_y_pixels',
        'delta_centroid_x_pixels',
        'delta_centroid_y_pixels',
        'step_distance',
        'speed',
        'distance_unit',
        'time_unit',
        'speed_unit',
        'discard_initial_frames',
        'discard_final_frames',
    ]
    TURN_SUMMARY_COLUMNS = [
        'source_dataframe',
        'particle',
        'first_frame',
        'last_frame',
        'detection_count',
        'segment_count',
        'processed_point_count',
        'angle_count',
        'number_of_turns',
        'mean_absolute_turn_angle_degrees',
        'total_path_length',
        'observed_duration_seconds',
        'turns_per_distance',
        'turns_per_second',
        'distance_unit',
        'minimum_turn_angle_degrees',
        'smoothing_method',
        'moving_average_window',
        'triangular_smoothing_window',
        'sg_filter_window_length',
        'sg_filter_polyorder',
        'rdp_epsilon_pixels',
        'max_frame_gap',
        'discard_initial_frames',
        'discard_final_frames',
    ]
    TURN_ANGLE_COLUMNS = [
        'source_dataframe',
        'particle',
        'segment_id',
        'processed_vertex_index',
        'frame',
        'elapsed_time_seconds',
        'centroid_x_pixels',
        'centroid_y_pixels',
        'turn_angle_degrees',
        'absolute_turn_angle_degrees',
        'is_turn',
        'cumulative_distance',
        'distance_unit',
        'smoothing_method',
        'effective_smoothing_window',
    ]
    TUMBLE_POINT_COLUMNS = [
        'source_dataframe',
        'particle',
        'segment_id',
        'frame',
        'next_frame',
        'frame_delta_to_next',
        'elapsed_time_seconds',
        'raw_position_x_pixels',
        'raw_position_y_pixels',
        'smoothed_position_x_pixels',
        'smoothed_position_y_pixels',
        'velocity_x',
        'velocity_y',
        'instantaneous_speed',
        'turn_angle_degrees',
        'exceeds_tumble_threshold',
        'singleton_confirmation_angle_degrees',
        'state',
        'event_id',
        'distance_unit',
        'speed_unit',
        'smoothing_method',
        'effective_smoothing_window',
        'tumble_threshold_angle_degrees',
    ]
    TUMBLE_EVENT_COLUMNS = [
        'source_dataframe',
        'particle',
        'segment_id',
        'event_id',
        'event_type',
        'start_frame',
        'end_frame',
        'support_end_frame',
        'point_count',
        'interval_seconds',
        'sampling_support_seconds',
        'path_length',
        'mean_speed',
        'std_speed',
        'mean_angular_speed_degrees_per_point',
        'std_angular_speed_degrees_per_point',
        'direction_change_within_run_degrees',
        'run_to_run_direction_change_degrees',
        'preceding_run_event_id',
        'following_run_event_id',
        'distance_unit',
        'speed_unit',
        'smoothing_method',
        'effective_smoothing_window',
        'tumble_threshold_angle_degrees',
    ]
    TUMBLE_SUMMARY_COLUMNS = [
        'source_dataframe',
        'particle',
        'original_first_frame',
        'original_last_frame',
        'original_detection_count',
        'discarded_detection_count',
        'analyzed_first_frame',
        'analyzed_last_frame',
        'analyzed_detection_count',
        'segment_count',
        'analyzable_angle_count',
        'classified_angle_count',
        'unclassified_angle_count',
        'number_of_runs',
        'number_of_tumbles',
        'runs_per_tumble',
        'analyzed_tracking_time_seconds',
        'total_run_interval_seconds',
        'total_tumble_interval_seconds',
        'tumble_time_fraction',
        'tumble_frequency_per_second',
        'mean_run_speed',
        'std_run_mean_speed_between_events',
        'mean_run_point_speed',
        'std_run_point_speed',
        'run_speed_point_count',
        'mean_tumble_speed',
        'std_tumble_speed',
        'mean_run_interval_seconds',
        'std_run_interval_seconds',
        'mean_tumble_interval_seconds',
        'std_tumble_interval_seconds',
        'mean_time_between_tumble_starts_seconds',
        'std_time_between_tumble_starts_seconds',
        'mean_run_angular_speed_degrees_per_point',
        'std_run_angular_speed_degrees_per_point',
        'mean_tumble_angular_speed_degrees_per_point',
        'std_tumble_angular_speed_degrees_per_point',
        'mean_direction_change_within_runs_degrees',
        'std_direction_change_within_runs_degrees',
        'mean_run_to_run_direction_change_degrees',
        'std_run_to_run_direction_change_degrees',
        'distance_unit',
        'speed_unit',
        'smoothing_method',
        'moving_average_window',
        'triangular_smoothing_window',
        'sg_filter_window_length',
        'sg_filter_polyorder',
        'rdp_epsilon_pixels',
        'discard_initial_frames',
        'discard_final_frames',
        'tumble_threshold_angle_degrees',
        'max_frame_gap',
        'analysis_status',
    ]
    TUMBLE_POPULATION_COLUMNS = [
        'source_dataframe',
        'particle_count',
        'particles_with_analyzable_angles',
        'particles_with_tumbles',
        'total_number_of_runs',
        'total_number_of_tumbles',
        'total_analyzed_tracking_time_seconds',
        'mean_tumble_frequency_per_second',
        'std_tumble_frequency_per_second',
        'mean_number_of_runs_per_particle',
        'std_number_of_runs_per_particle',
        'mean_number_of_tumbles_per_particle',
        'std_number_of_tumbles_per_particle',
        'mean_run_speed',
        'std_run_speed_between_particles',
        'mean_tumble_speed',
        'std_tumble_speed_between_particles',
        'mean_run_interval_seconds',
        'std_run_interval_seconds_between_particles',
        'mean_tumble_interval_seconds',
        'std_tumble_interval_seconds_between_particles',
        'mean_time_between_tumble_starts_seconds',
        'std_time_between_tumble_starts_seconds_between_particles',
        'mean_run_angular_speed_degrees_per_point',
        'std_run_angular_speed_degrees_per_point_between_particles',
        'mean_tumble_angular_speed_degrees_per_point',
        'std_tumble_angular_speed_degrees_per_point_between_particles',
        'mean_direction_change_within_runs_degrees',
        'std_direction_change_within_runs_degrees_between_particles',
        'mean_run_to_run_direction_change_degrees',
        'std_run_to_run_direction_change_degrees_between_particles',
        'distance_unit',
        'speed_unit',
        'smoothing_method',
        'moving_average_window',
        'triangular_smoothing_window',
        'sg_filter_window_length',
        'sg_filter_polyorder',
        'rdp_epsilon_pixels',
        'discard_initial_frames',
        'discard_final_frames',
        'tumble_threshold_angle_degrees',
        'max_frame_gap',
    ]
    VELOCITY_TUMBLE_POINT_COLUMNS = [
        'source_dataframe',
        'particle',
        'segment_id',
        'frame',
        'previous_frame',
        'frame_delta_from_previous',
        'elapsed_time_from_previous_seconds',
        'elapsed_time_seconds',
        'raw_position_x_pixels',
        'raw_position_y_pixels',
        'smoothed_position_x_pixels',
        'smoothed_position_y_pixels',
        'velocity_x',
        'velocity_y',
        'speed',
        'heading_radians',
        'previous_valid_heading_frame',
        'angular_elapsed_time_seconds',
        'angular_change_radians',
        'angular_velocity_magnitude_radians_per_second',
        'persistence_target_elapsed_time_seconds',
        'persistence_target_frame',
        'persistence_direction_change_radians',
        'persistence_cosine',
        'is_speed_minimum',
        'speed_minimum_passes',
        'is_angular_velocity_maximum',
        'angular_velocity_maximum_passes',
        'state',
        'event_id',
        'distance_unit',
        'speed_unit',
        'angular_velocity_unit',
        'smoothing_method',
        'effective_smoothing_window',
    ]
    VELOCITY_TUMBLE_EVENT_COLUMNS = [
        'source_dataframe',
        'particle',
        'segment_id',
        'event_id',
        'event_type',
        'start_frame',
        'end_frame',
        'support_start_frame',
        'point_count',
        'interval_seconds',
        'sampling_support_seconds',
        'path_length',
        'mean_speed',
        'minimum_speed',
        'maximum_speed',
        'mean_angular_velocity_magnitude_radians_per_second',
        'maximum_angular_velocity_magnitude_radians_per_second',
        'speed_minimum_frame',
        'speed_minimum_value',
        'speed_minimum_depth',
        'speed_depth_ratio',
        'speed_left_maximum_frame',
        'speed_right_maximum_frame',
        'speed_period_start_frame',
        'speed_period_end_frame',
        'angular_velocity_maximum_frame',
        'angular_velocity_maximum_value',
        'angular_velocity_maximum_height',
        'angular_left_minimum_frame',
        'angular_right_minimum_frame',
        'angular_total_directional_change_radians',
        'angular_directional_change_threshold_radians',
        'angular_period_start_frame',
        'angular_period_end_frame',
        'matching_overlap_start_frame',
        'matching_overlap_end_frame',
        'merged_candidate_count',
        'evidence_scope',
        'preceding_run_fit_start_frame',
        'preceding_run_fit_end_frame',
        'following_run_fit_start_frame',
        'following_run_fit_end_frame',
        'preceding_run_direction_radians',
        'following_run_direction_radians',
        'run_to_run_turn_angle_radians',
        'run_to_run_turn_angle_degrees',
        'run_to_run_directional_cosine',
        'distance_unit',
        'speed_unit',
        'angular_velocity_unit',
        'smoothing_method',
        'effective_smoothing_window',
        'speed_drop_ratio_threshold',
        'speed_period_depth_fraction',
        'angular_change_coefficient',
        'minimum_heading_speed',
    ]
    VELOCITY_TUMBLE_SUMMARY_COLUMNS = [
        'source_dataframe',
        'particle',
        'original_first_frame',
        'original_last_frame',
        'original_detection_count',
        'discarded_detection_count',
        'analyzed_first_frame',
        'analyzed_last_frame',
        'analyzed_detection_count',
        'segment_count',
        'analyzable_segment_count',
        'kinematic_point_count',
        'unclassified_detection_count',
        'number_of_runs',
        'number_of_tumbles',
        'runs_per_tumble',
        'analyzed_tracking_time_seconds',
        'classified_tracking_time_seconds',
        'total_run_interval_seconds',
        'total_tumble_interval_seconds',
        'total_unclassified_interval_seconds',
        'tumble_time_fraction',
        'tumble_frequency_per_second',
        'point_mean_run_speed',
        'point_std_run_speed',
        'point_mean_tumble_speed',
        'point_std_tumble_speed',
        'mean_run_interval_seconds',
        'std_run_interval_seconds',
        'mean_tumble_interval_seconds',
        'std_tumble_interval_seconds',
        'mean_time_between_tumble_starts_seconds',
        'std_time_between_tumble_starts_seconds',
        'mean_run_angular_velocity_magnitude_radians_per_second',
        'std_run_angular_velocity_magnitude_radians_per_second',
        'mean_tumble_angular_velocity_magnitude_radians_per_second',
        'std_tumble_angular_velocity_magnitude_radians_per_second',
        'time_weighted_vR',
        'vR_interval_count',
        'vR_support_seconds',
        'time_weighted_vT',
        'vT_interval_count',
        'vT_support_seconds',
        't_R',
        'std_t_R',
        'sem_t_R',
        't_R_interval_count',
        't_R_censored_interval_count',
        't_T',
        'std_t_T',
        't_T_interval_count',
        't_T_censored_interval_count',
        'p',
        'std_p',
        'p_direction_change_count',
        'R',
        'std_R',
        'R_run_transition_count',
        'distance_unit',
        'speed_unit',
        'angular_velocity_unit',
        'smoothing_method',
        'moving_average_window',
        'triangular_smoothing_window',
        'sg_filter_window_length',
        'sg_filter_polyorder',
        'rdp_epsilon_pixels',
        'discard_initial_frames',
        'discard_final_frames',
        'speed_drop_ratio_threshold',
        'speed_period_depth_fraction',
        'angular_change_coefficient',
        'minimum_heading_speed',
        'speed_extrema_prominence',
        'angular_velocity_extrema_prominence',
        'extrema_min_distance',
        'persistence_interval_seconds',
        'run_direction_fit_points',
        'max_frame_gap',
        'analysis_status',
    ]
    VELOCITY_TUMBLE_POPULATION_COLUMNS = [
        'source_dataframe',
        'particle_count',
        'particles_with_kinematic_points',
        'particles_with_tumbles',
        'total_number_of_runs',
        'total_number_of_tumbles',
        'total_analyzed_tracking_time_seconds',
        'total_classified_tracking_time_seconds',
        'total_unclassified_interval_seconds',
        'pooled_tumble_frequency_per_second',
        'mean_particle_tumble_frequency_per_second',
        'std_particle_tumble_frequency_per_second',
        'mean_particle_tumble_time_fraction',
        'std_particle_tumble_time_fraction',
        'mean_number_of_runs_per_particle',
        'std_number_of_runs_per_particle',
        'mean_number_of_tumbles_per_particle',
        'std_number_of_tumbles_per_particle',
        'particle_mean_point_run_speed',
        'particle_std_point_run_speed',
        'particle_mean_point_tumble_speed',
        'particle_std_point_tumble_speed',
        'mean_run_interval_seconds',
        'std_run_interval_seconds_between_particles',
        'mean_tumble_interval_seconds',
        'std_tumble_interval_seconds_between_particles',
        'mean_time_between_tumble_starts_seconds',
        'std_time_between_tumble_starts_seconds_between_particles',
        'mean_run_angular_velocity_magnitude_radians_per_second',
        'std_run_angular_velocity_between_particles',
        'mean_tumble_angular_velocity_magnitude_radians_per_second',
        'std_tumble_angular_velocity_between_particles',
        'pooled_time_weighted_vR',
        'pooled_time_weighted_vR_ci_lower',
        'pooled_time_weighted_vR_ci_upper',
        'vR_interval_count',
        'vR_support_seconds',
        'vR_particle_count',
        'pooled_time_weighted_vT',
        'pooled_time_weighted_vT_ci_lower',
        'pooled_time_weighted_vT_ci_upper',
        'vT_interval_count',
        'vT_support_seconds',
        'vT_particle_count',
        't_R',
        'std_t_R',
        'sem_t_R',
        't_R_interval_count',
        't_R_censored_interval_count',
        't_T',
        'std_t_T',
        't_T_interval_count',
        't_T_censored_interval_count',
        'p',
        'std_p',
        'p_direction_change_count',
        'R',
        'std_R',
        'R_run_transition_count',
        'particle_mean_time_weighted_vR',
        'particle_std_time_weighted_vR',
        'particle_sem_time_weighted_vR',
        'particle_mean_time_weighted_vT',
        'particle_std_time_weighted_vT',
        'particle_sem_time_weighted_vT',
        'mean_particle_t_R',
        'std_particle_t_R',
        'mean_particle_t_T',
        'std_particle_t_T',
        'mean_particle_p',
        'std_particle_p',
        'mean_particle_R',
        'std_particle_R',
        'distance_unit',
        'speed_unit',
        'angular_velocity_unit',
        'smoothing_method',
        'moving_average_window',
        'triangular_smoothing_window',
        'sg_filter_window_length',
        'sg_filter_polyorder',
        'rdp_epsilon_pixels',
        'discard_initial_frames',
        'discard_final_frames',
        'speed_drop_ratio_threshold',
        'speed_period_depth_fraction',
        'angular_change_coefficient',
        'minimum_heading_speed',
        'speed_extrema_prominence',
        'angular_velocity_extrema_prominence',
        'extrema_min_distance',
        'persistence_interval_seconds',
        'run_direction_fit_points',
        'max_frame_gap',
        'pooled_speed_bootstrap_resamples',
        'pooled_speed_bootstrap_confidence_level',
        'pooled_speed_bootstrap_random_seed',
    ]
    FITTED_MEAN_SPEED_COLUMNS = [
        'source_dataframe',
        'particle',
        'mean_speed',
        'distribution_type',
        'fit_range_source',
        'requested_fit_range_start',
        'requested_fit_range_end',
        'ci_range_lower_percentile',
        'ci_range_upper_percentile',
        'capture_speed_in_fps',
        'pixel_scale_factor',
        'scale_units',
        'speed_unit',
        'discard_initial_frames',
        'discard_final_frames',
    ]
    INDIVIDUAL_MSD_COLUMNS = [
        'source_dataframe',
        'particle',
        'lag_time',
        'individual_msd',
        'capture_speed_in_fps',
        'pixel_scale_factor',
        'scale_units',
        'time_unit',
        'msd_unit',
    ]
    ENSEMBLE_MSD_COLUMNS = [
        'source_dataframe',
        'lag_time',
        'ensemble_msd',
        'capture_speed_in_fps',
        'pixel_scale_factor',
        'scale_units',
        'time_unit',
        'msd_unit',
    ]
    MSD_FIT_COLUMNS = [
        'source_dataframe',
        'particle_count',
        'selected_particle_count',
        'contributing_particle_count',
        'max_lag_time_frames',
        'discard_initial_frames',
        'discard_final_frames',
        'requested_alpha_fit_lag_start',
        'requested_alpha_fit_lag_end',
        'fit_lag_start',
        'fit_lag_end',
        'fit_point_count',
        'alpha',
        'log_intercept',
        'r_squared',
        'p_value',
        'alpha_standard_error',
        'fit_status',
        'fit_message',
        'capture_speed_in_fps',
        'pixel_scale_factor',
        'scale_units',
        'time_unit',
        'msd_unit',
    ]

    def __init__(
        self,
        tracker_object: Tracker,
        source_dataframe: str = 'active',
        strain: str = 'define cell strain'
    ) -> None:
        """
        Initialize particle statistics from a selected Tracker dataframe.

        Args:
            tracker_object (Tracker): Tracker containing linked-particle dataframes.
            source_dataframe (str): One of 'active', 'full', 'filtered', or
                'updated'. 'active' selects updated, then filtered, then full.
            strain (str): Cell-strain label included in every dataframe
                produced for export. Defaults to 'define cell strain'.
        """
        if not isinstance(tracker_object, Tracker):
            raise TypeError('tracker_object must be a Tracker instance.')
        if not isinstance(strain, str):
            raise TypeError('strain must be a string.')
        self._strain = strain.strip()
        if not self._strain:
            raise ValueError('strain must be a non-empty string.')

        self._parent = tracker_object
        self._directory: str = tracker_object.get_directory()
        capture_object = tracker_object._parent._parent
        frame_rate_information = capture_object.get_frame_rate()
        capture_speed_in_fps = (
            frame_rate_information.get('user_provided_fps') or
            frame_rate_information.get('default_fps')
        )
        self._capture_speed_in_fps = self.__coerce_optional_number(
            capture_speed_in_fps
        )
        self.pixel_scale_factor = self.__coerce_optional_number(
            capture_object.get_pixel_scale_factor()
        )
        capture_scale_units = capture_object.get_scale_units()
        self._scale_units = (
            str(capture_scale_units).strip()
            if capture_scale_units is not None and str(capture_scale_units).strip()
            else 'scale_units'
        )
        self._sorted_dataframe = pd.DataFrame()
        self._source_dataframe = 'active'
        self._resolved_source_dataframe = 'full'
        self._mean_array: list[float] = []
        self._mean_speeds_dataframe = pd.DataFrame()
        self._calculated_speed_unit: str | None = None
        self._step_metrics_dataframe = pd.DataFrame(
            columns=self.STEP_METRIC_COLUMNS
        )
        self._particle_characteristics_dataframe = pd.DataFrame()
        self._turn_summary_dataframe = pd.DataFrame(
            columns=self.TURN_SUMMARY_COLUMNS
        )
        self._turn_angles_dataframe = pd.DataFrame(
            columns=self.TURN_ANGLE_COLUMNS
        )
        self._turn_path_data: dict[int, list[dict]] = {}
        self._turn_analysis_metadata: dict = {}
        self._tumble_summary_dataframe = pd.DataFrame(
            columns=self.TUMBLE_SUMMARY_COLUMNS
        )
        self._tumble_events_dataframe = pd.DataFrame(
            columns=self.TUMBLE_EVENT_COLUMNS
        )
        self._tumble_points_dataframe = pd.DataFrame(
            columns=self.TUMBLE_POINT_COLUMNS
        )
        self._tumble_population_dataframe = pd.DataFrame(
            columns=self.TUMBLE_POPULATION_COLUMNS
        )
        self._angle_tumble_table_1_dataframe = pd.DataFrame(
            columns=self.TABLE_1_VERTICAL_COLUMNS
        )
        self._tumble_path_data: dict[int, list[dict]] = {}
        self._tumble_analysis_metadata: dict = {}
        self._velocity_tumble_summary_dataframe = pd.DataFrame(
            columns=self.VELOCITY_TUMBLE_SUMMARY_COLUMNS
        )
        self._velocity_tumble_events_dataframe = pd.DataFrame(
            columns=self.VELOCITY_TUMBLE_EVENT_COLUMNS
        )
        self._velocity_tumble_points_dataframe = pd.DataFrame(
            columns=self.VELOCITY_TUMBLE_POINT_COLUMNS
        )
        self._velocity_tumble_population_dataframe = pd.DataFrame(
            columns=self.VELOCITY_TUMBLE_POPULATION_COLUMNS
        )
        self._velocity_tumble_table_1_dataframe = pd.DataFrame(
            columns=self.VELOCITY_TABLE_1_VERTICAL_COLUMNS
        )
        self._velocity_tumble_analysis_metadata: dict = {}
        self._speed_feature_particles_dataframe = pd.DataFrame()
        self._speed_feature_correlation_dataframe = pd.DataFrame()
        self._individual_msd_dataframe = pd.DataFrame()
        self._ensemble_msd_dataframe = pd.DataFrame()
        self._msd_fit_dataframe = pd.DataFrame()
        self._msd_metadata: dict = {}
        self._particle_metric_plotting_parameters = dict(
            self.PARTICLE_METRIC_PLOT_DEFAULTS
        )
        self._analysis_plotting_parameters = dict(
            self.ANALYSIS_PLOT_DEFAULTS
        )
        self.set_source_dataframe(source_dataframe)

    def set_source_dataframe(
        self,
        source_dataframe: str = 'active'
    ) -> pd.DataFrame:
        """
        Select and snapshot a linked-particle dataframe from the Tracker.

        Selecting a new source clears every previously calculated result.

        Args:
            source_dataframe (str): 'active', 'full', 'filtered', or 'updated'.

        Returns:
            pd.DataFrame: Copy of the selected, frame-sorted dataframe.
        """
        if not isinstance(source_dataframe, str):
            raise TypeError('source_dataframe must be a string.')
        normalized_source = source_dataframe.strip().lower()
        if normalized_source not in self.SOURCE_DATAFRAMES:
            raise ValueError(
                'source_dataframe must be one of: active, full, filtered, or '
                'updated.'
            )
        canonical_source = normalized_source

        dataframe_getters = {
            'full': self._parent.get_full_linked_particles_dataframe,
            'filtered': self._parent.get_filtered_linked_particles_dataframe,
            'updated': self._parent.get_updated_linked_particles_dataframe
        }
        if canonical_source == 'active':
            updated_dataframe = dataframe_getters['updated']()
            filtered_dataframe = dataframe_getters['filtered']()
            if not updated_dataframe.empty:
                resolved_source = 'updated'
            elif not filtered_dataframe.empty:
                resolved_source = 'filtered'
            else:
                resolved_source = 'full'
        else:
            resolved_source = canonical_source

        selected_dataframe = dataframe_getters[resolved_source]().copy()
        if selected_dataframe.empty:
            raise ValueError(
                f"No rows are available in the '{resolved_source}' linked "
                'particle dataframe.'
            )

        required_columns = [
            'particle', 'frame', 'centroid_x', 'centroid_y'
        ]
        missing_columns = [
            column for column in required_columns
            if column not in selected_dataframe.columns
        ]
        if missing_columns:
            raise ValueError(
                'Selected linked-particle dataframe is missing required '
                f'columns: {missing_columns}'
            )

        numeric_columns = ['particle', 'frame', 'centroid_x', 'centroid_y']
        for column in numeric_columns:
            selected_dataframe[column] = pd.to_numeric(
                selected_dataframe[column], errors='coerce'
            )
        finite_numeric_rows = np.isfinite(
            selected_dataframe[numeric_columns].to_numpy(dtype=float)
        ).all(axis=1)
        if not finite_numeric_rows.all():
            invalid_row_count = int((~finite_numeric_rows).sum())
            raise ValueError(
                f'Selected linked-particle dataframe contains '
                f'{invalid_row_count} row(s) with invalid particle, frame, or '
                'centroid values.'
            )
        particle_values = selected_dataframe['particle'].to_numpy(dtype=float)
        if not np.equal(particle_values, np.floor(particle_values)).all():
            raise ValueError('Particle IDs must be whole numbers.')
        selected_dataframe['particle'] = particle_values.astype(np.int64)
        frame_values = selected_dataframe['frame'].to_numpy(dtype=float)
        if not np.equal(frame_values, np.floor(frame_values)).all():
            raise ValueError('Frame values must be whole numbers.')
        selected_dataframe['frame'] = frame_values.astype(np.int64)
        duplicate_frame_rows = selected_dataframe.duplicated(
            subset=['particle', 'frame'], keep=False
        )
        if duplicate_frame_rows.any():
            duplicate_count = int(duplicate_frame_rows.sum())
            raise ValueError(
                f'Selected linked-particle dataframe contains {duplicate_count} '
                'row(s) with duplicate particle/frame combinations.'
            )

        selected_dataframe = selected_dataframe.sort_values(
            by=['particle', 'frame'], kind='stable'
        ).reset_index(drop=True)
        self._sorted_dataframe = selected_dataframe
        self._source_dataframe = canonical_source
        self._resolved_source_dataframe = resolved_source
        self.__reset_derived_analysis_results()

        particle_count = selected_dataframe['particle'].nunique()
        print(
            f"Stats source selected: {resolved_source} "
            f'({particle_count} particles, {len(selected_dataframe)} rows).'
        )
        return selected_dataframe.copy()

    def get_source_dataframe(self) -> pd.DataFrame:
        """Return a copy of the selected linked-particle dataframe snapshot."""
        return self._sorted_dataframe.copy()

    def get_source_dataframe_name(self) -> str:
        """Return the resolved Tracker dataframe name used for statistics."""
        return self._resolved_source_dataframe

    def calculate_step_metrics(
        self,
        particle_ids: int | list[int] | tuple[int, ...] | None = None,
        speed_unit: str = 'pixels/frame',
        max_frame_gap: int | None = None,
        discard_initial_frames: int = 10,
        discard_final_frames: int = 10
    ) -> pd.DataFrame:
        """
        Calculate one motion row for every retained consecutive detection pair.

        Frame gaps are accounted for in elapsed time. Set max_frame_gap to omit
        intervals that bridge a longer disappearance.

        Args:
            particle_ids: Optional particle ID or collection of IDs.
            speed_unit: pixels/frame, pixels/s, scale_units/frame, or
                scale_units/s. Scale-unit choices apply the Capture pixel
                scale factor automatically.
            max_frame_gap: Optional largest frame difference to retain.
            discard_initial_frames: Number of source frames discarded from the
                start of each track. Only detections in that inclusive frame
                window are removed.
            discard_final_frames: Number of source frames discarded from the
                end of each track. Only detections in that inclusive frame
                window are removed.

        Returns:
            pd.DataFrame: Long-form interval distances and speeds.
        """
        self._step_metrics_dataframe = pd.DataFrame(
            columns=self.STEP_METRIC_COLUMNS
        )
        selected_dataframe = self.__select_particle_rows(
            particle_ids,
            discard_initial_frames=discard_initial_frames,
            discard_final_frames=discard_final_frames
        )
        normalized_max_gap = self.__normalize_max_frame_gap(max_frame_gap)
        speed_unit_key, speed_unit_label = self.__normalize_speed_unit(
            speed_unit
        )
        (
            distance_factor,
            distance_unit,
            time_factor,
            time_unit,
        ) = self.__get_speed_unit_components(speed_unit_key)

        step_dataframes = []
        for particle_id, particle_rows in selected_dataframe.groupby(
            'particle', sort=True
        ):
            particle_rows = particle_rows.sort_values(
                by='frame', kind='stable'
            ).reset_index(drop=True)
            if len(particle_rows) < 2:
                continue

            frames = particle_rows['frame'].to_numpy(dtype=float)
            centroid_x = particle_rows['centroid_x'].to_numpy(dtype=float)
            centroid_y = particle_rows['centroid_y'].to_numpy(dtype=float)
            frame_delta = np.diff(frames)
            delta_x = np.diff(centroid_x)
            delta_y = np.diff(centroid_y)
            distance_pixels = np.hypot(delta_x, delta_y)
            retain_intervals = np.ones(len(frame_delta), dtype=bool)
            if normalized_max_gap is not None:
                retain_intervals &= frame_delta <= normalized_max_gap
            if not retain_intervals.any():
                continue

            elapsed_time = frame_delta * time_factor
            step_distance = distance_pixels * distance_factor
            speed = step_distance / elapsed_time
            interval_indices = np.flatnonzero(retain_intervals)
            step_dataframes.append(pd.DataFrame({
                'source_dataframe': self._resolved_source_dataframe,
                'particle': int(particle_id),
                'start_frame': frames[:-1][interval_indices].astype(int),
                'end_frame': frames[1:][interval_indices].astype(int),
                'frame_delta': frame_delta[interval_indices].astype(int),
                'elapsed_time': elapsed_time[interval_indices],
                'start_centroid_x_pixels': centroid_x[:-1][interval_indices],
                'start_centroid_y_pixels': centroid_y[:-1][interval_indices],
                'end_centroid_x_pixels': centroid_x[1:][interval_indices],
                'end_centroid_y_pixels': centroid_y[1:][interval_indices],
                'delta_centroid_x_pixels': delta_x[interval_indices],
                'delta_centroid_y_pixels': delta_y[interval_indices],
                'step_distance': step_distance[interval_indices],
                'speed': speed[interval_indices],
                'distance_unit': distance_unit,
                'time_unit': time_unit,
                'speed_unit': speed_unit_label,
                'discard_initial_frames': int(discard_initial_frames),
                'discard_final_frames': int(discard_final_frames),
            }))

        if step_dataframes:
            step_metrics = pd.concat(step_dataframes, ignore_index=True)
            step_metrics = step_metrics[self.STEP_METRIC_COLUMNS]
        else:
            step_metrics = pd.DataFrame(columns=self.STEP_METRIC_COLUMNS)
        step_metrics = self.__add_strain_field(step_metrics)
        self._step_metrics_dataframe = step_metrics
        return step_metrics.copy()

    def calculate_particle_characteristics(
        self,
        particle_ids: int | list[int] | tuple[int, ...] | None = None,
        length_unit: str = 'scale_units',
        speed_unit: str = DEFAULT_SPEED_UNIT,
        max_frame_gap: int | None = None,
        discard_initial_frames: int = 10,
        discard_final_frames: int = 10
    ) -> pd.DataFrame:
        """
        Summarize morphology and motion for each selected particle track.

        These direct descriptive summaries avoid fitting a distribution to
        every particle and retain particle IDs, source information, and units.
        fitted_mean_speed is the distribution-fitted value from the latest
        compatible calculate_speed_and_plot_mean call. It is NaN when no
        cached value matches the selected source, speed unit, and discard
        windows. It is matched by particle ID rather than row position.
        mean_speed is total retained path length divided by total retained
        elapsed time; mean_interval_speed is the unweighted interval mean.

        Args:
            particle_ids: Optional particle ID or collection of IDs.
            length_unit: 'pixels', 'scale_units', or the configured scale unit.
                scale_units applies the Capture pixel scale automatically.
            speed_unit: Unit used for interval and mean speeds. Use
                scale_units/frame or scale_units/s for automatic scaling.
            max_frame_gap: Optional largest frame difference used for motion.
            discard_initial_frames: Number of source frames discarded from the
                start of each track before morphology and motion summaries.
            discard_final_frames: Number of source frames discarded from the
                end of each track before morphology and motion summaries.

        Returns:
            pd.DataFrame: One morphology and motion summary row per particle.
        """
        self._particle_characteristics_dataframe = pd.DataFrame()
        selected_dataframe = self.__select_particle_rows(
            particle_ids,
            required_columns=['area', 'major_axis_length', 'minor_axis_length'],
            discard_initial_frames=discard_initial_frames,
            discard_final_frames=discard_final_frames
        )
        _, length_factor, length_unit_label = self.__normalize_length_unit(
            length_unit
        )
        area_factor = length_factor**2
        area_unit_label = f'{length_unit_label}²'
        step_metrics = self.calculate_step_metrics(
            particle_ids=particle_ids,
            speed_unit=speed_unit,
            max_frame_gap=max_frame_gap,
            discard_initial_frames=discard_initial_frames,
            discard_final_frames=discard_final_frames
        )
        _, speed_unit_label = self.__normalize_speed_unit(speed_unit)
        fitted_mean_speed_lookup = self.__get_fitted_mean_speed_lookup(
            speed_unit_label=speed_unit_label,
            discard_initial_frames=discard_initial_frames,
            discard_final_frames=discard_final_frames
        )
        step_groups = {
            int(particle): rows
            for particle, rows in step_metrics.groupby('particle', sort=False)
        }

        summary_rows = []
        for particle_id, particle_rows in selected_dataframe.groupby(
            'particle', sort=True
        ):
            particle_rows = particle_rows.sort_values(
                by='frame', kind='stable'
            )
            major_lengths = self.__finite_numeric_values(
                particle_rows['major_axis_length']
            ) * length_factor
            minor_lengths = self.__finite_numeric_values(
                particle_rows['minor_axis_length']
            ) * length_factor
            areas = self.__finite_numeric_values(
                particle_rows['area']
            ) * area_factor
            paired_axes = particle_rows[
                ['major_axis_length', 'minor_axis_length']
            ].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)
            valid_axes = (
                np.isfinite(paired_axes).all(axis=1) &
                (paired_axes[:, 1] > 0)
            )
            aspect_ratios = (
                paired_axes[valid_axes, 0] / paired_axes[valid_axes, 1]
            )
            particle_steps = step_groups.get(
                int(particle_id), pd.DataFrame(columns=self.STEP_METRIC_COLUMNS)
            )
            speeds = (
                particle_steps['speed'].to_numpy(dtype=float)
                if not particle_steps.empty
                else np.array([], dtype=float)
            )
            overall_average_speed = (
                float(particle_steps['step_distance'].sum()) /
                float(particle_steps['elapsed_time'].sum())
                if (
                    not particle_steps.empty and
                    float(particle_steps['elapsed_time'].sum()) > 0
                )
                else np.nan
            )
            path_length = (
                float(np.hypot(
                    particle_steps['delta_centroid_x_pixels'],
                    particle_steps['delta_centroid_y_pixels']
                ).sum()) * length_factor
                if not particle_steps.empty
                else 0.0
            )
            first_frame = int(particle_rows['frame'].iloc[0])
            last_frame = int(particle_rows['frame'].iloc[-1])
            frame_span = last_frame - first_frame
            frame_span_duration_seconds = (
                frame_span / self._capture_speed_in_fps
                if self.__has_valid_frame_rate()
                else np.nan
            )
            observed_duration_seconds = (
                float(particle_steps['frame_delta'].sum()) /
                self._capture_speed_in_fps
                if not particle_steps.empty and self.__has_valid_frame_rate()
                else np.nan
            )
            summary_rows.append({
                'source_dataframe': self._resolved_source_dataframe,
                'particle': int(particle_id),
                'first_frame': first_frame,
                'last_frame': last_frame,
                'detection_count': int(len(particle_rows)),
                'frame_span': frame_span,
                'frame_span_duration_seconds': frame_span_duration_seconds,
                'observed_duration_seconds': observed_duration_seconds,
                'mean_major_axis_length': self.__safe_mean(major_lengths),
                'median_major_axis_length': self.__safe_median(major_lengths),
                'std_major_axis_length': self.__safe_std(major_lengths),
                'mean_minor_axis_length': self.__safe_mean(minor_lengths),
                'median_minor_axis_length': self.__safe_median(minor_lengths),
                'std_minor_axis_length': self.__safe_std(minor_lengths),
                'mean_aspect_ratio': self.__safe_mean(aspect_ratios),
                'median_aspect_ratio': self.__safe_median(aspect_ratios),
                'std_aspect_ratio': self.__safe_std(aspect_ratios),
                'mean_area': self.__safe_mean(areas),
                'median_area': self.__safe_median(areas),
                'std_area': self.__safe_std(areas),
                'speed_interval_count': int(len(speeds)),
                'fitted_mean_speed': fitted_mean_speed_lookup.get(
                    int(particle_id), np.nan
                ),
                'mean_speed': overall_average_speed,
                'mean_interval_speed': self.__safe_mean(speeds),
                'median_speed': self.__safe_median(speeds),
                'std_speed': self.__safe_std(speeds),
                'total_path_length': path_length,
                'length_unit': length_unit_label,
                'area_unit': area_unit_label,
                'speed_unit': speed_unit_label,
                'discard_initial_frames': int(discard_initial_frames),
                'discard_final_frames': int(discard_final_frames),
            })

        characteristics = self.__add_strain_field(
            pd.DataFrame(summary_rows)
        )
        self._particle_characteristics_dataframe = characteristics
        return characteristics.copy()

    def plot_particle_characteristic_distributions(
        self,
        bins: int = 20,
        figsize: tuple[float, float] = (12, 9),
        save_path: str | None = None,
        dpi: int = 300,
        show: bool = True
    ) -> tuple:
        """Plot population distributions from calculate_particle_characteristics."""
        self.__validate_positive_integer(bins, 'bins')
        characteristics = self._particle_characteristics_dataframe
        if characteristics.empty:
            raise ValueError(
                'Calculate particle characteristics before plotting them.'
            )

        length_unit = self.__plot_unit_label(
            characteristics['length_unit'].iloc[0]
        )
        area_unit = self.__plot_unit_label(
            characteristics['area_unit'].iloc[0]
        )
        speed_unit = self.__plot_unit_label(
            characteristics['speed_unit'].iloc[0]
        )
        fig, axes = plt.subplots(2, 2, figsize=figsize)

        major_values = self.__finite_numeric_values(
            characteristics['mean_major_axis_length']
        )
        minor_values = self.__finite_numeric_values(
            characteristics['mean_minor_axis_length']
        )
        combined_lengths = np.concatenate([major_values, minor_values])
        if combined_lengths.size:
            length_bins = np.histogram_bin_edges(combined_lengths, bins=bins)
            axes[0, 0].hist(
                major_values, bins=length_bins, alpha=0.65,
                label='Mean major axis', color='#D35400'
            )
            axes[0, 0].hist(
                minor_values, bins=length_bins, alpha=0.65,
                label='Mean minor axis', color='#3498DB'
            )
        axes[0, 0].set_xlabel(f'Length ({length_unit})')
        axes[0, 0].set_ylabel('Particle count')
        axes[0, 0].set_title('Mean axis lengths')
        axes[0, 0].legend()

        plot_definitions = [
            (axes[0, 1], 'mean_aspect_ratio', 'Mean aspect ratio', '#9B59B6'),
            (axes[1, 0], 'mean_area', f'Mean area ({area_unit})', '#2A9D8F'),
            (axes[1, 1], 'mean_speed', f'Mean speed ({speed_unit})', '#2E86C1'),
        ]
        for ax, column, label, color in plot_definitions:
            values = self.__finite_numeric_values(characteristics[column])
            if values.size:
                ax.hist(values, bins=bins, color=color, alpha=0.75)
            ax.set_xlabel(label)
            ax.set_ylabel('Particle count')
            ax.set_title(label)

        fig.suptitle(
            f'Particle characteristics - {self._resolved_source_dataframe}'
        )
        fig.tight_layout()
        self.__finalize_figure(fig, save_path, dpi, show)
        return fig, axes

    def plot_trajectories_from_origin(
        self,
        particle_ids: int | list[int] | tuple[int, ...] | None = None,
        distance_unit: str = 'scale_units',
        show_particle_ids: bool = False,
        invert_vertical_axis: bool = True,
        figsize: tuple[float, float] = (10, 8),
        save_path: str | None = None,
        dpi: int = 300,
        show: bool = True,
        discard_initial_frames: int = 10,
        discard_final_frames: int = 10
    ) -> tuple:
        """
        Plot trajectories from a shared origin in pixels or scale_units.

        scale_units applies the Capture pixel scale factor automatically and
        labels the axes with the configured physical unit.
        discard_initial_frames and discard_final_frames remove detections in
        inclusive source-frame windows at each track boundary before
        trajectories are moved to the shared origin.
        """
        selected_dataframe = self.__select_particle_rows(
            particle_ids,
            discard_initial_frames=discard_initial_frames,
            discard_final_frames=discard_final_frames
        )
        _, distance_factor, distance_unit_label = self.__normalize_length_unit(
            distance_unit,
            argument_name='distance_unit'
        )
        distance_unit_label = self.__plot_unit_label(distance_unit_label)
        fig, ax = plt.subplots(figsize=figsize)
        particle_values = selected_dataframe['particle'].unique()
        color_map = plt.get_cmap('tab20', max(len(particle_values), 1))

        for color_index, (particle_id, particle_rows) in enumerate(
            selected_dataframe.groupby('particle', sort=True)
        ):
            particle_rows = particle_rows.sort_values(
                by='frame', kind='stable'
            )
            horizontal = particle_rows['centroid_y'].to_numpy(dtype=float)
            vertical = particle_rows['centroid_x'].to_numpy(dtype=float)
            horizontal = (horizontal - horizontal[0]) * distance_factor
            vertical = (vertical - vertical[0]) * distance_factor
            color = color_map(color_index)
            ax.plot(
                horizontal, vertical, linewidth=1.5, alpha=0.85,
                color=color, label=str(int(particle_id))
            )
            if show_particle_ids:
                ax.text(
                    horizontal[-1], vertical[-1], str(int(particle_id)),
                    fontsize=8, color=color
                )

        ax.axhline(0, color='black', linewidth=0.8, alpha=0.5)
        ax.axvline(0, color='black', linewidth=0.8, alpha=0.5)
        ax.set_xlabel(f'X displacement ({distance_unit_label})')
        ax.set_ylabel(f'Y displacement ({distance_unit_label})')
        ax.set_title(
            f'Trajectories from origin - {self._resolved_source_dataframe}'
        )
        if invert_vertical_axis:
            ax.invert_yaxis()
        if len(particle_values) <= 20:
            ax.legend(title='Particle')
        fig.tight_layout()
        self.__finalize_figure(fig, save_path, dpi, show)
        return fig, ax

    def calculate_turn_statistics(
        self,
        particle_ids: int | list[int] | tuple[int, ...] | None = None,
        minimum_turn_angle_degrees: float = 15.0,
        smoothing_method: str = 'sg_filter_rdp',
        moving_average_window: int = 5,
        triangular_smoothing_window: int | None = 17,
        sg_filter_window_length: int | None = 5,
        sg_filter_polyorder: int = 2,
        rdp_epsilon_pixels: float = 5.0,
        distance_unit: str = 'scale_units',
        max_frame_gap: int | None = 1,
        discard_initial_frames: int = 10,
        discard_final_frames: int = 10
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Detect turns on selectively smoothed particle trajectories.

        Signed turn angles use Cartesian image coordinates: centroid_y is the
        horizontal axis and negative centroid_x is the vertical axis. Positive
        angles are counterclockwise. Tracks are split across gaps larger than
        max_frame_gap so missing observations do not create artificial turns.
        smoothing_method accepts moving_average, triangular_smoothing, or
        sg_filter_rdp. The default preserves the established SG-filtered,
        sparse RDP turn analysis. Moving-average and triangular modes calculate
        turns at their finite, frame-aligned processed points without RDP.
        distance_unit accepts pixels, scale_units, or the configured physical
        unit; scale_units applies the Capture pixel scale automatically.
        discard_initial_frames and discard_final_frames remove detections in
        inclusive source-frame windows at each track boundary before gap
        splitting, smoothing, and turn detection.
        ``number_of_turns`` counts processed angle points at or above the
        threshold; consecutive qualifying points are not merged into one
        event, so counts can differ across smoothing methods.

        Args:
            particle_ids: Optional particle ID or collection of IDs.
            minimum_turn_angle_degrees: Absolute processed-path angle required
                for a point to be classified as a turn.
            smoothing_method: 'moving_average', 'triangular_smoothing', or
                'sg_filter_rdp'.
            moving_average_window: Complete centered odd moving-average span
                of at least 3 points. Unsupported boundary points are omitted
                from turn-angle calculation.
            triangular_smoothing_window: Odd triangular-filter span of at least
                3 points, with edge-value padding.
            sg_filter_window_length: Odd SG-filter span of at least 3 points.
            sg_filter_polyorder: Nonnegative SG polynomial order smaller than
                sg_filter_window_length.
            rdp_epsilon_pixels: RDP perpendicular-distance tolerance in pixels,
                used after the SG filter.
            distance_unit: 'pixels', 'scale_units', or the configured Capture
                scale unit.
            max_frame_gap: Split a track when adjacent detections are farther
                apart than this many frames. None permits all gaps.
            discard_initial_frames: Width of the source-frame window removed
                from the start of each track.
            discard_final_frames: Width of the source-frame window removed
                from the end of each track.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: Per-particle summaries and one
                row for every measurable processed-path angle. Filter the
                second dataframe where is_turn is True for detected events.
        """
        self._turn_summary_dataframe = pd.DataFrame(
            columns=self.TURN_SUMMARY_COLUMNS
        )
        self._turn_angles_dataframe = pd.DataFrame(
            columns=self.TURN_ANGLE_COLUMNS
        )
        self._turn_path_data = {}
        self._turn_analysis_metadata = {}
        if not np.isfinite(minimum_turn_angle_degrees):
            raise ValueError('minimum_turn_angle_degrees must be finite.')
        if not 0 <= minimum_turn_angle_degrees <= 180:
            raise ValueError(
                'minimum_turn_angle_degrees must be between 0 and 180.'
            )
        smoothing_configuration = (
            self.__normalize_smoothing_configuration(
                smoothing_method=smoothing_method,
                moving_average_window=moving_average_window,
                triangular_smoothing_window=(
                    triangular_smoothing_window
                ),
                sg_filter_window_length=sg_filter_window_length,
                sg_filter_polyorder=sg_filter_polyorder,
                rdp_epsilon_pixels=rdp_epsilon_pixels
            )
        )
        smoothing_output = self.__smoothing_configuration_output(
            smoothing_configuration
        )
        normalized_max_gap = self.__normalize_max_frame_gap(max_frame_gap)
        if normalized_max_gap is None or normalized_max_gap > 1:
            warnings.warn(
                'Retained frame gaps make smoothing irregularly sampled: the '
                'selected window operates over detection order. Use '
                'max_frame_gap=1 for uniformly sampled turn analysis.',
                UserWarning,
                stacklevel=2
            )
        _, distance_factor, distance_unit_label = self.__normalize_length_unit(
            distance_unit,
            argument_name='distance_unit'
        )
        selected_dataframe = self.__select_particle_rows(
            particle_ids,
            discard_initial_frames=discard_initial_frames,
            discard_final_frames=discard_final_frames
        )

        summary_rows = []
        angle_rows = []
        path_data: dict[int, list[dict]] = {}
        grouped_particles = selected_dataframe.groupby('particle', sort=True)
        for particle_id, particle_rows in tqdm(
            grouped_particles,
            total=selected_dataframe['particle'].nunique(),
            desc='Analyzing turns'
        ):
            particle_id = int(particle_id)
            particle_rows = particle_rows.sort_values(
                by='frame', kind='stable'
            ).reset_index(drop=True)
            frames = particle_rows['frame'].to_numpy(dtype=int)
            first_frame = int(frames[0])
            last_frame = int(frames[-1])
            segments = self.__split_particle_track(
                particle_rows, normalized_max_gap
            )
            path_data[particle_id] = []
            particle_turn_angles = []
            particle_path_length_pixels = 0.0
            processed_distance_offset_pixels = 0.0
            observed_duration_seconds = 0.0
            processed_point_count = 0
            angle_count = 0

            for segment_id, segment_rows in enumerate(segments, start=1):
                segment_frames = segment_rows['frame'].to_numpy(dtype=int)
                raw_positions = np.column_stack([
                    segment_rows['centroid_y'].to_numpy(dtype=float),
                    -segment_rows['centroid_x'].to_numpy(dtype=float),
                ])
                smoothing_result = self.__smooth_analysis_positions(
                    raw_positions,
                    smoothing_configuration=smoothing_configuration,
                    rdp_output_mode='sparse'
                )
                smoothed_positions = smoothing_result[
                    'frame_aligned_positions'
                ]
                processed_positions = smoothing_result[
                    'analysis_positions'
                ]
                source_indices = smoothing_result['source_indices']
                effective_window = smoothing_result['effective_window']
                processed_frames = segment_frames[source_indices]
                segment_steps = np.linalg.norm(
                    np.diff(raw_positions, axis=0), axis=1
                )
                segment_path_length = float(segment_steps.sum())
                particle_path_length_pixels += segment_path_length
                processed_steps = np.linalg.norm(
                    np.diff(processed_positions, axis=0), axis=1
                )
                cumulative_processed_distance = np.concatenate([
                    [0.0], np.cumsum(processed_steps)
                ]) + processed_distance_offset_pixels
                processed_distance_offset_pixels += float(
                    processed_steps.sum()
                )
                signed_angles = self.__calculate_signed_turn_angles(
                    processed_positions
                )
                angle_vertex_indices = np.arange(
                    1, max(len(processed_positions) - 1, 1)
                )
                if len(signed_angles) != len(angle_vertex_indices):
                    angle_vertex_indices = np.arange(1, 1 + len(signed_angles))
                finite_angles = np.isfinite(signed_angles)
                turn_vertex_indices = []
                for angle_index in np.flatnonzero(finite_angles):
                    vertex_index = int(angle_vertex_indices[angle_index])
                    source_index = int(source_indices[vertex_index])
                    signed_angle = float(signed_angles[angle_index])
                    absolute_angle = abs(signed_angle)
                    is_turn = absolute_angle >= minimum_turn_angle_degrees
                    if is_turn:
                        turn_vertex_indices.append(vertex_index)
                        particle_turn_angles.append(absolute_angle)
                    elapsed_time_seconds = (
                        (int(segment_frames[source_index]) - first_frame) /
                        self._capture_speed_in_fps
                        if self.__has_valid_frame_rate()
                        else np.nan
                    )
                    original_row = segment_rows.iloc[source_index]
                    angle_rows.append({
                        'source_dataframe': self._resolved_source_dataframe,
                        'particle': particle_id,
                        'segment_id': segment_id,
                        'processed_vertex_index': vertex_index,
                        'frame': int(segment_frames[source_index]),
                        'elapsed_time_seconds': elapsed_time_seconds,
                        'centroid_x_pixels': float(original_row['centroid_x']),
                        'centroid_y_pixels': float(original_row['centroid_y']),
                        'turn_angle_degrees': signed_angle,
                        'absolute_turn_angle_degrees': absolute_angle,
                        'is_turn': bool(is_turn),
                        'cumulative_distance': (
                            cumulative_processed_distance[vertex_index] *
                            distance_factor
                        ),
                        'distance_unit': distance_unit_label,
                        'smoothing_method': smoothing_configuration[
                            'smoothing_method'
                        ],
                        'effective_smoothing_window': (
                            effective_window
                            if effective_window is not None
                            else np.nan
                        ),
                    })

                if self.__has_valid_frame_rate() and len(segment_frames) > 1:
                    observed_duration_seconds += (
                        int(segment_frames[-1]) - int(segment_frames[0])
                    ) / self._capture_speed_in_fps
                processed_point_count += len(processed_positions)
                angle_count += int(finite_angles.sum())
                path_data[particle_id].append({
                    'segment_id': segment_id,
                    'frames': segment_frames,
                    'raw_positions_pixels': raw_positions,
                    'smoothed_positions_pixels': smoothed_positions,
                    'processed_positions_pixels': processed_positions,
                    'source_indices': source_indices,
                    'rdp_source_indices': smoothing_result[
                        'rdp_source_indices'
                    ],
                    'processed_frames': processed_frames,
                    'cumulative_distance_pixels': cumulative_processed_distance,
                    'signed_angles_degrees': signed_angles,
                    'angle_vertex_indices': angle_vertex_indices,
                    'turn_vertex_indices': np.asarray(
                        turn_vertex_indices, dtype=int
                    ),
                    'effective_smoothing_window': effective_window,
                    'smoothing_method': smoothing_configuration[
                        'smoothing_method'
                    ],
                })

            turn_count = len(particle_turn_angles)
            total_path_length = (
                particle_path_length_pixels * distance_factor
            )
            summary_rows.append({
                'source_dataframe': self._resolved_source_dataframe,
                'particle': particle_id,
                'first_frame': first_frame,
                'last_frame': last_frame,
                'detection_count': int(len(particle_rows)),
                'segment_count': len(segments),
                'processed_point_count': processed_point_count,
                'angle_count': angle_count,
                'number_of_turns': turn_count,
                'mean_absolute_turn_angle_degrees': self.__safe_mean(
                    np.asarray(particle_turn_angles, dtype=float)
                ),
                'total_path_length': total_path_length,
                'observed_duration_seconds': (
                    observed_duration_seconds
                    if self.__has_valid_frame_rate()
                    else np.nan
                ),
                'turns_per_distance': (
                    turn_count / total_path_length
                    if total_path_length > 0
                    else np.nan
                ),
                'turns_per_second': (
                    turn_count / observed_duration_seconds
                    if observed_duration_seconds > 0
                    else np.nan
                ),
                'distance_unit': distance_unit_label,
                'minimum_turn_angle_degrees': minimum_turn_angle_degrees,
                **smoothing_output,
                'max_frame_gap': (
                    normalized_max_gap
                    if normalized_max_gap is not None
                    else np.nan
                ),
                'discard_initial_frames': int(discard_initial_frames),
                'discard_final_frames': int(discard_final_frames),
            })

        turn_summary = pd.DataFrame(
            summary_rows, columns=self.TURN_SUMMARY_COLUMNS
        )
        turn_angles = pd.DataFrame(
            angle_rows, columns=self.TURN_ANGLE_COLUMNS
        )
        turn_summary = self.__add_strain_field(turn_summary)
        turn_angles = self.__add_strain_field(turn_angles)
        self._turn_summary_dataframe = turn_summary
        self._turn_angles_dataframe = turn_angles
        self._turn_path_data = path_data
        self._turn_analysis_metadata = {
            'source_dataframe': self._resolved_source_dataframe,
            'strain': self._strain,
            'minimum_turn_angle_degrees': minimum_turn_angle_degrees,
            **smoothing_output,
            'rdp_output_mode': (
                'sparse'
                if smoothing_configuration[
                    'smoothing_method'
                ] == 'sg_filter_rdp'
                else 'not_applied'
            ),
            'distance_factor': distance_factor,
            'distance_unit': distance_unit_label,
            'max_frame_gap': normalized_max_gap,
            'discard_initial_frames': int(discard_initial_frames),
            'discard_final_frames': int(discard_final_frames),
            'turn_count_rule': (
                'Each qualifying processed angle point counts as one turn; '
                'consecutive qualifying points are not merged.'
            ),
        }
        return turn_summary.copy(), turn_angles.copy()

    def get_turn_events_dataframe(self) -> pd.DataFrame:
        """Return detected turn rows from the latest turn analysis."""
        if self._turn_angles_dataframe.empty:
            return self.__add_strain_field(
                pd.DataFrame(columns=self.TURN_ANGLE_COLUMNS)
            )
        return self._turn_angles_dataframe.loc[
            self._turn_angles_dataframe['is_turn']
        ].reset_index(drop=True).copy()

    def set_analysis_plotting_parameters(
        self,
        particle_marker: str | None = None,
        angle_mode: str = 'absolute',
        show_turns: bool = True,
        turn_color: str = 'darkviolet',
        turn_display_mode: str = 'markers',
        turn_marker: str = 'o',
        show_smoothed_trajectory: bool = True,
        smoothed_trajectory_color: str = 'lightblue',
        show_tumbles: bool = True,
        tumble_color: str = '#D55E00',
        tumble_display_mode: str = 'markers',
        tumble_marker: str | None = None,
        show_angle_smoothed_trajectory: bool = False,
        smoothed_angle_trajectory_color: str = 'goldenrod',
        show_velocity_smoothed_trajectory: bool = False,
        smoothed_velocity_trajectory_color: str = 'goldenrod',
        title: str | None = None,
        title_fontsize: float | None = None,
        x_axis_fontsize: float | None = None,
        y_axis_fontsize: float | None = None,
        x_tick_fontsize: float | None = None,
        y_tick_fontsize: float | None = None,
        font_family: str | None = None,
        dpi: int = 300,
        save_plots: bool = False,
        save_plot_path: str | os.PathLike | None = None,
        file_extension: str = 'png',
        track_thickness: float = 2.0,
        smooth_trajectory_thickness: float = 1.5,
        segment_marker_thickness: float = 1.0,
        crop_to_track: bool = False,
        marker_fill_mode: str | None = None
    ) -> None:
        """
        Set reusable defaults for the five particle-analysis plot functions.

        Omitted arguments in ``plot_particle_turn_analysis``,
        ``plot_particle_angle_tumble_analysis``,
        ``plot_particle_velocity_tumble_analysis``,
        ``plot_particle_turn_angle_tumble_analysis``, and
        ``plot_particle_turn_velocity_tumble_analysis`` use these values.
        Explicit per-call arguments take precedence. The generic smoothed-path
        settings also control the turn-analysis path in the combined plots.

        ``tumble_marker=None`` preserves each function's established default:
        a circle for the standalone angle-tumble plot and an x for velocity or
        combined tumble plots.

        Args:
            particle_marker: Optional observed-track position marker, such as
                'o', 's', '^', 'D', or 'x'. None draws no position markers.
            marker_fill_mode: 'solid' fills particle, turn, and tumble markers;
                'outline' draws only their colored outlines. None preserves
                each plot's established marker appearance. Inherently
                unfilled markers such as 'x' remain unfilled.
            track_thickness: Observed-track line width in points.
            smooth_trajectory_thickness: Smoothed-trajectory line width in
                points, including the optional independent smoothed paths in
                combined plots.
            segment_marker_thickness: Positive event-overlay scale factor. A
                value of 1 preserves the current turn/tumble segment widths,
                marker areas, and marker-edge widths; larger values enlarge
                them proportionally.
            crop_to_track: For the two combined plots, use the established
                tightly fitted trajectory limits. False uses a normal square
                plotting area with Matplotlib margins.
            angle_mode: 'absolute' or 'signed' turn-angle display.
            show_turns: Whether detected turns are displayed.
            turn_color: Matplotlib color for detected turns.
            turn_display_mode: 'markers' or observed-track 'segments'.
            turn_marker: Marker used for turn marker display mode, such as
                'o', 's', '^', 'D', or 'x'.
            show_smoothed_trajectory: Display the standalone smoothed path and
                the turn-analysis smoothed path in combined figures.
            smoothed_trajectory_color: Color of those smoothed paths.
            show_tumbles: Whether detected tumbles are displayed.
            tumble_color: Matplotlib color for detected tumbles.
            tumble_display_mode: 'markers' or observed-track 'segments'.
            tumble_marker: Tumble marker, such as 'o', 's', '^', 'D', or 'x',
                or None for each function's current method-specific default.
            show_angle_smoothed_trajectory: Display the independent
                angle-tumble smoothed path in the combined angle plot.
            smoothed_angle_trajectory_color: Color of that angle-tumble path.
            show_velocity_smoothed_trajectory: Display the independent
                velocity-tumble smoothed path in the combined velocity plot.
            smoothed_velocity_trajectory_color: Color of that velocity path.
            title: Custom primary title. None uses the established title; an
                empty string removes it.
            title_fontsize: Primary-title font size in points.
            x_axis_fontsize: X-axis label font size in points.
            y_axis_fontsize: Y-axis label font size in points.
            x_tick_fontsize: X-axis tick-label font size in points.
            y_tick_fontsize: Y-axis tick-label font size in points.
            font_family: Typeface for plot text. Portable families are
                'sans-serif', 'serif', 'monospace', 'cursive', and 'fantasy'.
                Bundled faces include 'DejaVu Sans', 'DejaVu Serif', and
                'DejaVu Sans Mono'. Installed system-font names, including
                'Aptos' when installed, are accepted.
            dpi: Resolution used when saving figures.
            save_plots: Automatically save plots without an explicit
                ``save_path``.
            save_plot_path: Automatic-save directory. Relative paths resolve
                from the Stats output directory; None uses that directory.
            file_extension: Automatic-save format: 'png' or 'tif'.
        """
        configured_parameters = {
            'particle_marker': particle_marker,
            'marker_fill_mode': marker_fill_mode,
            'track_thickness': track_thickness,
            'smooth_trajectory_thickness': smooth_trajectory_thickness,
            'segment_marker_thickness': segment_marker_thickness,
            'crop_to_track': crop_to_track,
            'angle_mode': angle_mode,
            'show_turns': show_turns,
            'turn_color': turn_color,
            'turn_display_mode': turn_display_mode,
            'turn_marker': turn_marker,
            'show_smoothed_trajectory': show_smoothed_trajectory,
            'smoothed_trajectory_color': smoothed_trajectory_color,
            'show_tumbles': show_tumbles,
            'tumble_color': tumble_color,
            'tumble_display_mode': tumble_display_mode,
            'tumble_marker': tumble_marker,
            'show_angle_smoothed_trajectory': (
                show_angle_smoothed_trajectory
            ),
            'smoothed_angle_trajectory_color': (
                smoothed_angle_trajectory_color
            ),
            'show_velocity_smoothed_trajectory': (
                show_velocity_smoothed_trajectory
            ),
            'smoothed_velocity_trajectory_color': (
                smoothed_velocity_trajectory_color
            ),
            'title': title,
            'title_fontsize': title_fontsize,
            'x_axis_fontsize': x_axis_fontsize,
            'y_axis_fontsize': y_axis_fontsize,
            'x_tick_fontsize': x_tick_fontsize,
            'y_tick_fontsize': y_tick_fontsize,
            'font_family': font_family,
            'dpi': dpi,
            'save_plots': save_plots,
            'save_plot_path': save_plot_path,
            'file_extension': file_extension,
        }
        self._analysis_plotting_parameters = (
            self.__validate_analysis_plotting_parameters(
                configured_parameters
            )
        )

    def get_analysis_plotting_parameters(self) -> dict:
        """Return a copy of the current particle-analysis plot defaults."""
        configured_parameters = getattr(
            self,
            '_analysis_plotting_parameters',
            self.ANALYSIS_PLOT_DEFAULTS
        )
        return dict(configured_parameters)

    def plot_particle_turn_analysis(
        self,
        particle_id: int,
        angle_mode: (
            str | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        figsize: tuple[float, float] = (14, 6),
        track_color: str | Colormap = 'viridis',
        save_path: str | os.PathLike | None = None,
        dpi: int | _UseConfiguredPlotValue = _USE_CONFIGURED_PLOT_VALUE,
        show: bool = True,
        show_turns: (
            bool | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        show_smoothed_trajectory: (
            bool | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        turn_display_mode: (
            str | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        turn_color: (
            str | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        smoothed_trajectory_color: (
            str | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        particle_marker: (
            str | None | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        turn_marker: (
            str | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        title: (
            str | None | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        title_fontsize: (
            float | None | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        x_axis_fontsize: (
            float | None | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        y_axis_fontsize: (
            float | None | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        x_tick_fontsize: (
            float | None | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        y_tick_fontsize: (
            float | None | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        font_family: (
            str | None | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        save_plots: (
            bool | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        save_plot_path: (
            str | os.PathLike | None | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        file_extension: (
            str | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        track_thickness: (
            float | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        smooth_trajectory_thickness: (
            float | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        segment_marker_thickness: (
            float | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        marker_fill_mode: (
            str | None | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE
    ) -> tuple:
        """
        Plot a colored trajectory and angle-versus-distance analysis.

        Turns can be drawn as the original vertex markers or by recoloring the
        observed-track edges spanned by the incoming and outgoing processed
        edges. The selected processed path and turn overlays can be hidden
        independently. ``show_smoothed_trajectory`` controls the processed
        trajectory produced by the selected smoothing method.

        Omitted configurable arguments use
        ``set_analysis_plotting_parameters``; explicit values override those
        settings for this call only.

        Args:
            track_color: Solid Matplotlib color, colormap name, or Colormap
                object for the observed track and angle points. A colormap
                colors values by elapsed time and adds a colorbar. Color-like
                names take precedence, so use ``plt.get_cmap('gray')`` for a
                gray elapsed-time gradient rather than a solid gray track.
            track_thickness: Observed-track line width in points.
            smooth_trajectory_thickness: Smoothed-path line width in points.
            segment_marker_thickness: Scale factor for turn segment widths,
                marker areas, and marker-edge widths.
            marker_fill_mode: 'solid' fills configured markers; 'outline'
                draws only their colored outlines. None preserves the
                established marker appearance.
            particle_marker: Optional Matplotlib marker for each observed
                particle position, such as 'o', 's', '^', 'D', or 'x'. None
                draws no observed-position markers.
            turn_marker: Matplotlib marker for detected turns, such as 'o',
                's', '^', 'D', or 'x'. The trajectory panel uses it in marker
                mode; the angle panel uses it in both marker and segment modes.
            title: Primary figure title. None restores the generated title;
                an empty string removes it.
            title_fontsize: Primary-title font size in points.
            x_axis_fontsize: X-axis label font size in points.
            y_axis_fontsize: Y-axis label font size in points.
            x_tick_fontsize: X-axis tick-label font size in points.
            y_tick_fontsize: Y-axis tick-label font size in points.
            font_family: Typeface for all figure text. Portable families are
                'sans-serif', 'serif', 'monospace', 'cursive', and 'fantasy'.
                Bundled faces include 'DejaVu Sans', 'DejaVu Serif', and
                'DejaVu Sans Mono'; installed system-font names are accepted.
            save_plots: Automatically save when ``save_path`` is None.
            save_plot_path: Directory for automatic numbered filenames.
            file_extension: Automatic-save format: 'png' or 'tif'.
        """
        plotting_parameters = self.__resolve_analysis_plotting_parameters(
            function_tumble_marker_default='x',
            angle_mode=angle_mode,
            track_thickness=track_thickness,
            smooth_trajectory_thickness=smooth_trajectory_thickness,
            segment_marker_thickness=segment_marker_thickness,
            marker_fill_mode=marker_fill_mode,
            show_turns=show_turns,
            show_smoothed_trajectory=show_smoothed_trajectory,
            turn_display_mode=turn_display_mode,
            turn_color=turn_color,
            smoothed_trajectory_color=smoothed_trajectory_color,
            particle_marker=particle_marker,
            turn_marker=turn_marker,
            title=title,
            title_fontsize=title_fontsize,
            x_axis_fontsize=x_axis_fontsize,
            y_axis_fontsize=y_axis_fontsize,
            x_tick_fontsize=x_tick_fontsize,
            y_tick_fontsize=y_tick_fontsize,
            font_family=font_family,
            dpi=dpi,
            save_plots=save_plots,
            save_plot_path=save_plot_path,
            file_extension=file_extension
        )
        angle_mode = plotting_parameters['angle_mode']
        show_turns = plotting_parameters['show_turns']
        show_smoothed_trajectory = plotting_parameters[
            'show_smoothed_trajectory'
        ]
        turn_display_mode = plotting_parameters['turn_display_mode']
        turn_color = plotting_parameters['turn_color']
        smoothed_trajectory_color = plotting_parameters[
            'smoothed_trajectory_color'
        ]
        particle_marker = plotting_parameters['particle_marker']
        turn_marker = plotting_parameters['turn_marker']
        track_thickness = plotting_parameters['track_thickness']
        smooth_trajectory_thickness = plotting_parameters[
            'smooth_trajectory_thickness'
        ]
        segment_marker_thickness = plotting_parameters[
            'segment_marker_thickness'
        ]
        marker_fill_mode = plotting_parameters['marker_fill_mode']
        dpi = plotting_parameters['dpi']
        particle_id = self.__normalize_single_particle_id(particle_id)
        if particle_id not in self._turn_path_data:
            raise ValueError(
                'Run calculate_turn_statistics for the requested particle '
                'before plotting its turn analysis.'
            )
        angle_mode = self.__normalize_angle_mode(angle_mode)
        self.__validate_boolean_argument(show_turns, 'show_turns')
        self.__validate_boolean_argument(
            show_smoothed_trajectory, 'show_smoothed_trajectory'
        )
        turn_display_mode = self.__normalize_event_display_mode(
            turn_display_mode, 'turn_display_mode'
        )
        self.__validate_plot_color(turn_color, 'turn_color')
        self.__validate_plot_color(
            smoothed_trajectory_color, 'smoothed_trajectory_color'
        )
        self.__validate_plot_marker(turn_marker, 'turn_marker')
        if particle_marker is not None:
            self.__validate_plot_marker(particle_marker, 'particle_marker')
        uniform_track_color, track_colormap = self.__resolve_track_color(
            track_color
        )
        path_segments = self._turn_path_data[particle_id]
        distance_factor = float(
            self._turn_analysis_metadata['distance_factor']
        )
        distance_unit = self.__plot_unit_label(
            self._turn_analysis_metadata['distance_unit']
        )
        processed_trajectory_label = self.__smoothing_trajectory_label(
            self._turn_analysis_metadata
        )
        all_frames = np.concatenate([
            segment['frames'] for segment in path_segments
        ])
        first_frame = int(all_frames.min())
        if self.__has_valid_frame_rate():
            color_values = (
                (all_frames - first_frame) / self._capture_speed_in_fps
            )
            colorbar_label = 'Elapsed time (s)'
        else:
            color_values = all_frames - first_frame
            colorbar_label = 'Elapsed frames'
        color_min = float(np.min(color_values))
        color_max = float(np.max(color_values))
        if color_max <= color_min:
            color_max = color_min + 1.0
        color_norm = Normalize(vmin=color_min, vmax=color_max)
        origin = path_segments[0]['raw_positions_pixels'][0]

        fig, (trajectory_ax, angle_ax) = plt.subplots(
            1, 2, figsize=figsize
        )
        colorbar_ax = (
            make_axes_locatable(angle_ax).append_axes(
                'right', size='4%', pad=0.05
            )
            if track_colormap is not None else None
        )
        raw_label_added = False
        processed_label_added = False
        turn_label_added = False
        for segment in path_segments:
            frames = segment['frames']
            raw_positions = (
                segment['raw_positions_pixels'] - origin
            ) * distance_factor
            for row_index in range(len(raw_positions) - 1):
                color_value = (
                    (frames[row_index] - first_frame) /
                    self._capture_speed_in_fps
                    if self.__has_valid_frame_rate()
                    else frames[row_index] - first_frame
                )
                trajectory_ax.plot(
                    raw_positions[row_index:row_index + 2, 0],
                    raw_positions[row_index:row_index + 2, 1],
                    color=self.__track_color_for_value(
                        uniform_track_color,
                        track_colormap,
                        color_norm,
                        color_value
                    ),
                    linewidth=track_thickness,
                    label='Observed trajectory' if not raw_label_added else None
                )
                raw_label_added = True
            marker_color_values = (
                (frames - first_frame) / self._capture_speed_in_fps
                if self.__has_valid_frame_rate()
                else frames - first_frame
            )
            self.__scatter_particle_positions(
                trajectory_ax,
                raw_positions,
                marker_color_values,
                uniform_track_color,
                track_colormap,
                color_norm,
                particle_marker,
                marker_fill_mode,
                label=(
                    'Observed trajectory'
                    if len(raw_positions) == 1 and not raw_label_added
                    else None
                )
            )
            if len(raw_positions) == 1 and particle_marker is not None:
                raw_label_added = True

            processed_positions = (
                segment['processed_positions_pixels'] - origin
            ) * distance_factor
            if show_smoothed_trajectory:
                trajectory_ax.plot(
                    processed_positions[:, 0], processed_positions[:, 1],
                    color=smoothed_trajectory_color,
                    linestyle='--',
                    linewidth=smooth_trajectory_thickness,
                    label=(
                        processed_trajectory_label
                        if not processed_label_added else None
                    )
                )
                processed_label_added = True
            turn_indices = segment['turn_vertex_indices']
            if (
                show_turns and len(turn_indices) and
                turn_display_mode == 'markers'
            ):
                trajectory_ax.scatter(
                    processed_positions[turn_indices, 0],
                    processed_positions[turn_indices, 1],
                    marker=turn_marker,
                    **self.__event_marker_color_arguments(
                        turn_marker,
                        turn_color,
                        marker_fill_mode,
                        default_outline=False
                    ),
                    s=35 * segment_marker_thickness,
                    zorder=5,
                    label='Detected turns' if not turn_label_added else None
                )
                turn_label_added = True
            elif show_turns and len(turn_indices):
                turn_edge_mask = self.__raw_turn_edge_mask(
                    raw_point_count=len(raw_positions),
                    source_indices=segment['source_indices'],
                    turn_vertex_indices=turn_indices
                )
                turn_label_added = (
                    self.__add_highlighted_track_segments(
                        trajectory_ax,
                        raw_positions,
                        turn_edge_mask,
                        color=turn_color,
                        label=(
                            'Detected turn segments'
                            if not turn_label_added else None
                        ),
                        linewidth=2.0 * segment_marker_thickness
                    ) or turn_label_added
                )

            finite_angles = np.isfinite(segment['signed_angles_degrees'])
            vertex_indices = segment['angle_vertex_indices'][finite_angles]
            angle_values = segment['signed_angles_degrees'][finite_angles]
            if angle_mode == 'absolute':
                angle_values = np.abs(angle_values)
            if len(angle_values):
                angle_distances = (
                    segment['cumulative_distance_pixels'][vertex_indices] *
                    distance_factor
                )
                angle_frames = segment['processed_frames'][vertex_indices]
                angle_colors = (
                    (angle_frames - first_frame) /
                    self._capture_speed_in_fps
                    if self.__has_valid_frame_rate()
                    else angle_frames - first_frame
                )
                angle_ax.plot(
                    angle_distances, angle_values,
                    color='gray', alpha=0.5, linewidth=1
                )
                if track_colormap is None:
                    angle_ax.scatter(
                        angle_distances, angle_values,
                        color=uniform_track_color, s=24
                    )
                else:
                    angle_ax.scatter(
                        angle_distances, angle_values,
                        c=angle_colors, cmap=track_colormap,
                        norm=color_norm, s=24
                    )
                turn_mask = np.isin(vertex_indices, turn_indices)
                if show_turns and turn_mask.any():
                    angle_ax.scatter(
                        angle_distances[turn_mask], angle_values[turn_mask],
                        marker=turn_marker,
                        **self.__event_marker_color_arguments(
                            turn_marker,
                            turn_color,
                            marker_fill_mode,
                            default_outline=True
                        ),
                        linewidths=1.5 * segment_marker_thickness,
                        s=70 * segment_marker_thickness,
                        label=(
                            'Detected turns'
                            if 'Detected turns' not in (
                                angle_ax.get_legend_handles_labels()[1]
                            )
                            else None
                        )
                    )

        if track_colormap is not None:
            scalar_map = plt.cm.ScalarMappable(
                norm=color_norm, cmap=track_colormap
            )
            scalar_map.set_array([])
            fig.colorbar(
                scalar_map, cax=colorbar_ax, label=colorbar_label
            )
        trajectory_ax.set_xlabel(f'X displacement ({distance_unit})')
        trajectory_ax.set_ylabel(f'Y displacement ({distance_unit})')
        trajectory_ax.set_title(f'Particle {particle_id} trajectory')
        trajectory_ax.set_aspect('equal', adjustable='datalim')
        trajectory_ax.legend()
        angle_ax.set_xlabel(f'Cumulative processed distance ({distance_unit})')
        angle_ax.set_ylabel(
            'Absolute turn angle (degrees)'
            if angle_mode == 'absolute'
            else 'Signed turn angle (degrees)'
        )
        angle_ax.set_ylim((0, 180) if angle_mode == 'absolute' else (-180, 180))
        angle_ax.set_title('Turning angle along trajectory')
        if angle_ax.get_legend_handles_labels()[0]:
            angle_ax.legend()
        default_title = (
            f'Turn analysis - {self._resolved_source_dataframe} tracks'
        )
        self.__format_analysis_figure(
            figure=fig,
            default_title=default_title,
            title=plotting_parameters['title'],
            primary_title_axis=None,
            title_fontsize=plotting_parameters['title_fontsize'],
            x_axis_fontsize=plotting_parameters['x_axis_fontsize'],
            y_axis_fontsize=plotting_parameters['y_axis_fontsize'],
            x_tick_fontsize=plotting_parameters['x_tick_fontsize'],
            y_tick_fontsize=plotting_parameters['y_tick_fontsize'],
            font_family=plotting_parameters['font_family']
        )
        save_path = self.__resolve_numbered_plot_save_path(
            save_path=save_path,
            save_plots=plotting_parameters['save_plots'],
            save_plot_path=plotting_parameters['save_plot_path'],
            file_extension=plotting_parameters['file_extension'],
            default_title=default_title,
            title=plotting_parameters['title']
        )
        fig.subplots_adjust(top=0.88, wspace=0.3)
        self.__finalize_figure(fig, save_path, dpi, show)
        return fig, (trajectory_ax, angle_ax)

    def plot_turn_angle_distribution(
        self,
        bins: int = 20,
        turns_only: bool = True,
        angle_mode: str = 'absolute',
        plot_style: str = 'polar',
        figsize: tuple[float, float] = (8, 8),
        save_path: str | None = None,
        dpi: int = 300,
        show: bool = True,
        color: str | Colormap | None = None,
        title: str | None = None,
        title_fontsize: float | None = None,
        x_axis_fontsize: float | None = None,
        y_axis_fontsize: float | None = None,
        x_tick_fontsize: float | None = None,
        y_tick_fontsize: float | None = None,
        font_family: str | None = 'Aptos',
        save_plots: bool = False,
        save_plot_path: str | os.PathLike | None = (
            './02_Outputs/06_Additional_Analysis/01_Plots/'
        ),
        file_extension: str = 'png'
    ) -> tuple:
        """
        Plot pooled processed-path angles as a histogram or polar chart.

        Args:
            bins: Number of equal-width angular bins.
            turns_only: Plot detected turns only when True; otherwise include
                every finite processed-path angle.
            angle_mode: 'absolute' for 0–180 degree magnitudes or 'signed' for
                -180–180 degree turn directions.
            plot_style: 'histogram' or 'polar'.
            figsize: Figure dimensions in inches.
            save_path: Optional explicit output filename. This takes precedence
                over automatic saving.
            dpi: Resolution used when saving the chart.
            show: Whether to display the chart interactively.
            color: A Matplotlib color, colormap name, or Colormap object. A
                fixed color is applied to every angular bin; colormap colors
                are cycled across bins. None preserves the established gray.
            title: Custom chart title. None uses the established title for the
                selected plot style; an empty string removes it.
            title_fontsize: Title font size in points.
            x_axis_fontsize: X-axis label font size in points. For polar plots,
                this applies to the angular-axis label when one is present.
            y_axis_fontsize: Y-axis label font size in points. For polar plots,
                this applies to the radial-axis label when one is present.
            x_tick_fontsize: X-axis or polar angular tick-label font size.
            y_tick_fontsize: Y-axis or polar radial tick-label font size.
            font_family: Typeface for all chart text. Portable families are
                'sans-serif', 'serif', 'monospace', 'cursive', and 'fantasy'.
                Bundled faces include 'DejaVu Sans', 'DejaVu Serif', and
                'DejaVu Sans Mono'. Installed system-font names, including
                'Aptos' when installed, are accepted.
            save_plots: Save the chart using an automatically numbered
                filename when ``save_path`` is None.
            save_plot_path: Automatic-save directory. Relative paths resolve
                from the Stats output directory.
            file_extension: Automatic-save format: 'png' or 'tif'.
        """
        self.__validate_positive_integer(bins, 'bins')
        if not isinstance(turns_only, bool):
            raise TypeError('turns_only must be a boolean.')
        angle_mode = self.__normalize_angle_mode(angle_mode)
        plot_style = str(plot_style).strip().lower()
        if plot_style not in {'histogram', 'polar'}:
            raise ValueError("plot_style must be 'histogram' or 'polar'.")
        plotting_parameters = self.__validate_basic_plotting_parameters(
            title=title,
            title_fontsize=title_fontsize,
            x_axis_fontsize=x_axis_fontsize,
            y_axis_fontsize=y_axis_fontsize,
            x_tick_fontsize=x_tick_fontsize,
            y_tick_fontsize=y_tick_fontsize,
            font_family=font_family,
            dpi=dpi,
            save_plots=save_plots,
            save_plot_path=save_plot_path,
            file_extension=file_extension,
            show=show
        )
        angle_dataframe = self._turn_angles_dataframe
        if turns_only and not angle_dataframe.empty:
            angle_dataframe = angle_dataframe.loc[angle_dataframe['is_turn']]
        if angle_dataframe.empty:
            raise ValueError('No turn-angle observations are available to plot.')

        if angle_mode == 'absolute':
            values = angle_dataframe[
                'absolute_turn_angle_degrees'
            ].to_numpy(dtype=float)
            value_range = (0.0, 180.0)
            angle_label = 'Absolute turn angle (degrees)'
        else:
            values = angle_dataframe['turn_angle_degrees'].to_numpy(dtype=float)
            value_range = (-180.0, 180.0)
            angle_label = 'Signed turn angle (degrees)'
        values = values[np.isfinite(values)]
        if not len(values):
            raise ValueError('No finite turn-angle observations are available.')

        counts, bin_edges = np.histogram(
            values, bins=bins, range=value_range
        )
        bar_colors = self.__resolve_bar_colors(
            color=color,
            bar_count=len(counts),
            default_color='#6C757D'
        )

        if plot_style == 'polar':
            center_degrees = (bin_edges[:-1] + bin_edges[1:]) / 2
            polar_centers = (
                np.mod(center_degrees, 360.0)
                if angle_mode == 'signed'
                else center_degrees
            )
            centers = np.deg2rad(polar_centers)
            widths = np.deg2rad(np.diff(bin_edges))
            fig, ax = plt.subplots(
                figsize=figsize, subplot_kw={'projection': 'polar'}
            )
            ax.bar(
                centers, counts, width=widths, color=bar_colors,
                alpha=0.7, edgecolor='black'
            )
            ax.set_theta_zero_location('E')
            ax.set_theta_direction(1)
            if angle_mode == 'absolute':
                ax.set_thetamin(0)
                ax.set_thetamax(180)
            default_title = angle_label
        else:
            fig, ax = plt.subplots(figsize=figsize)
            ax.bar(
                bin_edges[:-1], counts, width=np.diff(bin_edges),
                align='edge', color=bar_colors,
                alpha=0.75, edgecolor='black'
            )
            ax.set_xlabel(angle_label)
            ax.set_ylabel('Angle count')
            default_title = (
                'Detected turn-angle distribution'
                if turns_only else 'All processed-path angle distribution'
            )

        self.__format_particle_metric_axis(
            axis=ax,
            default_title=default_title,
            title=plotting_parameters['title'],
            title_fontsize=plotting_parameters['title_fontsize'],
            x_axis_fontsize=plotting_parameters['x_axis_fontsize'],
            y_axis_fontsize=plotting_parameters['y_axis_fontsize'],
            x_tick_fontsize=plotting_parameters['x_tick_fontsize'],
            y_tick_fontsize=plotting_parameters['y_tick_fontsize'],
            y_tick_spacing=None,
            font_family=plotting_parameters['font_family']
        )
        save_path = self.__resolve_numbered_plot_save_path(
            save_path=save_path,
            save_plots=plotting_parameters['save_plots'],
            save_plot_path=plotting_parameters['save_plot_path'],
            file_extension=plotting_parameters['file_extension'],
            default_title=default_title,
            title=plotting_parameters['title']
        )

        bin_summary = pd.DataFrame({
            'bin_start_degrees': bin_edges[:-1],
            'bin_end_degrees': bin_edges[1:],
            'count': counts,
            'angle_mode': angle_mode,
            'turns_only': turns_only,
        })
        fig.tight_layout()
        self.__finalize_figure(
            fig,
            save_path,
            plotting_parameters['dpi'],
            plotting_parameters['show']
        )
        return fig, ax, bin_summary

    def set_particle_metric_plotting_parameters(
        self,
        title: str | None = None,
        title_fontsize: float | None = None,
        x_axis_fontsize: float | None = None,
        y_axis_fontsize: float | None = None,
        x_tick_fontsize: float | None = None,
        y_tick_fontsize: float | None = None,
        show_particle_ID: bool = True,
        show_particle_ID_only_with_value: bool = False,
        particle_ID_spacing: int = 1,
        y_tick_spacing: float | None = None,
        font_family: str | None = None,
        dpi: int = 300,
        save_plots: bool = False,
        save_plot_path: str | os.PathLike | None = None,
        file_extension: str = 'png'
    ) -> None:
        """
        Set reusable defaults for the per-particle metric plot functions.

        These settings are used by ``plot_particle_turn_metric``,
        ``plot_particle_angle_tumble_metric``, and
        ``plot_particle_velocity_tumble_metric`` whenever the corresponding
        per-call argument is omitted. An explicitly supplied per-call value
        takes precedence, including ``title=None`` for the metric-based title
        and ``title=''`` for no displayed title.

        Args:
            title: Default custom title. None uses each metric-based title.
            title_fontsize: Default title font size in points.
            x_axis_fontsize: Default Cartesian x-axis label font size.
            y_axis_fontsize: Default Cartesian y-axis label font size.
            x_tick_fontsize: Default particle-ID tick-label font size.
            y_tick_fontsize: Default numeric or radial tick-label font size.
            show_particle_ID: Whether particle-ID tick labels are displayed.
            show_particle_ID_only_with_value: Whether to label only particles
                whose plotted metric is nonzero.
            particle_ID_spacing: X-tick spacing; label every nth eligible ID.
            y_tick_spacing: Numeric or radial major-tick interval. None lets
                Matplotlib select it automatically.
            font_family: Typeface for plot text. Portable families are
                'sans-serif', 'serif', 'monospace', 'cursive', and 'fantasy'.
                Bundled faces include 'DejaVu Sans', 'DejaVu Serif', and
                'DejaVu Sans Mono'. Installed system-font names are accepted.
            dpi: Resolution used when a plot is saved.
            save_plots: Automatically save each plot when no explicit
                ``save_path`` is supplied to the plotting function.
            save_plot_path: Automatic-save directory. Relative paths are
                resolved from the Stats output directory. None uses that
                output directory directly.
            file_extension: Automatic-save format: 'png' or 'tif'. A leading
                period and uppercase spellings are accepted and normalized.
        """
        configured_parameters = {
            'title': title,
            'title_fontsize': title_fontsize,
            'x_axis_fontsize': x_axis_fontsize,
            'y_axis_fontsize': y_axis_fontsize,
            'x_tick_fontsize': x_tick_fontsize,
            'y_tick_fontsize': y_tick_fontsize,
            'show_particle_ID': show_particle_ID,
            'show_particle_ID_only_with_value': (
                show_particle_ID_only_with_value
            ),
            'particle_ID_spacing': particle_ID_spacing,
            'y_tick_spacing': y_tick_spacing,
            'font_family': font_family,
            'dpi': dpi,
            'save_plots': save_plots,
            'save_plot_path': save_plot_path,
            'file_extension': file_extension,
        }
        self._particle_metric_plotting_parameters = (
            self.__validate_particle_metric_plotting_parameters(
                configured_parameters
            )
        )

    def get_particle_metric_plotting_parameters(self) -> dict:
        """Return a copy of the current per-particle metric plot defaults."""
        configured_parameters = getattr(
            self,
            '_particle_metric_plotting_parameters',
            self.PARTICLE_METRIC_PLOT_DEFAULTS
        )
        return dict(configured_parameters)

    def plot_particle_turn_metric(
        self,
        metric: str = 'number_of_turns',
        plot_style: str = 'bar',
        figsize: tuple[float, float] = (12, 6),
        save_path: str | os.PathLike | None = None,
        dpi: int | _UseConfiguredPlotValue = _USE_CONFIGURED_PLOT_VALUE,
        show: bool = True,
        show_particle_ID: (
            bool | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        color: str | Colormap | None = None,
        particle_ID_spacing: (
            int | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        show_particle_ID_only_with_value: (
            bool | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        title: (
            str | None | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        title_fontsize: (
            float | None | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        x_axis_fontsize: (
            float | None | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        y_axis_fontsize: (
            float | None | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        x_tick_fontsize: (
            float | None | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        y_tick_fontsize: (
            float | None | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        y_tick_spacing: (
            float | None | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        font_family: (
            str | None | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        save_plots: (
            bool | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        save_plot_path: (
            str | os.PathLike | None | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        file_extension: (
            str | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE
    ) -> tuple:
        """
        Plot one turn-count or normalized turn-rate value per particle.

        Omitted configurable arguments use the values established by
        ``set_particle_metric_plotting_parameters``. Explicit arguments
        override those defaults for this plot only.

        Args:
            show_particle_ID: Label every Cartesian or polar bar with its
                corresponding particle ID.
            color: A Matplotlib color, colormap name, or Colormap object. A
                fixed color is applied to every bar; colormap colors are
                cycled across particles. None preserves the established blue
                bar and viridis polar defaults.
            particle_ID_spacing: X-tick spacing: label every nth eligible
                particle ID. Use a value greater than 1 to reduce crowding.
            show_particle_ID_only_with_value: If True, label only particles
                whose plotted metric is nonzero. Their zero-height bars remain
                in the plot.
            title: Custom plot title. None keeps the metric-based title.
            title_fontsize: Title font size in points. None uses Matplotlib's
                current default.
            x_axis_fontsize: X-axis label font size in points. None uses the
                current default. Cartesian plots have an x-axis label.
            y_axis_fontsize: Y-axis label font size in points. None uses the
                current default. Cartesian plots have a y-axis label.
            x_tick_fontsize: Particle-ID tick-label font size in points.
            y_tick_fontsize: Numeric or radial tick-label font size in points.
            y_tick_spacing: Numeric or radial major-tick interval. None lets
                Matplotlib choose the interval automatically.
            font_family: Typeface for the title, axis labels, and tick labels.
                Portable Matplotlib families are 'sans-serif', 'serif',
                'monospace', 'cursive', and 'fantasy'. Bundled faces include
                'DejaVu Sans', 'DejaVu Serif', and 'DejaVu Sans Mono'. An
                installed system-font family name can also be used. None keeps
                Matplotlib's current default.
            save_plots: If True and save_path is None, automatically generate
                a numbered filename and save the plot.
            save_plot_path: Directory used for automatic filenames. None uses
                the Stats output directory.
            file_extension: Automatic-save format: 'png' or 'tif'.
        """
        valid_metrics = {
            'number_of_turns', 'turns_per_distance', 'turns_per_second'
        }
        if metric not in valid_metrics:
            raise ValueError(f'metric must be one of {sorted(valid_metrics)}.')
        plot_style = str(plot_style).strip().lower()
        if plot_style not in {'bar', 'polar'}:
            raise ValueError("plot_style must be 'bar' or 'polar'.")
        plotting_parameters = (
            self.__resolve_particle_metric_plotting_parameters(
                dpi=dpi,
                show_particle_ID=show_particle_ID,
                particle_ID_spacing=particle_ID_spacing,
                show_particle_ID_only_with_value=(
                    show_particle_ID_only_with_value
                ),
                title=title,
                title_fontsize=title_fontsize,
                x_axis_fontsize=x_axis_fontsize,
                y_axis_fontsize=y_axis_fontsize,
                x_tick_fontsize=x_tick_fontsize,
                y_tick_fontsize=y_tick_fontsize,
                y_tick_spacing=y_tick_spacing,
                font_family=font_family,
                save_plots=save_plots,
                save_plot_path=save_plot_path,
                file_extension=file_extension
            )
        )
        dpi = plotting_parameters['dpi']
        show_particle_ID = plotting_parameters['show_particle_ID']
        particle_ID_spacing = plotting_parameters['particle_ID_spacing']
        show_particle_ID_only_with_value = plotting_parameters[
            'show_particle_ID_only_with_value'
        ]
        title = plotting_parameters['title']
        title_fontsize = plotting_parameters['title_fontsize']
        x_axis_fontsize = plotting_parameters['x_axis_fontsize']
        y_axis_fontsize = plotting_parameters['y_axis_fontsize']
        x_tick_fontsize = plotting_parameters['x_tick_fontsize']
        y_tick_fontsize = plotting_parameters['y_tick_fontsize']
        y_tick_spacing = plotting_parameters['y_tick_spacing']
        font_family = plotting_parameters['font_family']
        turn_summary = self._turn_summary_dataframe
        if turn_summary.empty:
            raise ValueError('Calculate turn statistics before plotting them.')
        plot_dataframe = turn_summary[
            ['particle', metric]
        ].replace([np.inf, -np.inf], np.nan).dropna()
        if plot_dataframe.empty:
            raise ValueError(f'No finite values are available for {metric}.')

        values = plot_dataframe[metric].to_numpy(dtype=float)
        particle_labels = (
            plot_dataframe['particle'].astype(int).astype(str).to_numpy()
        )
        particle_label_indices = self.__particle_label_indices(
            values,
            particle_ID_spacing,
            show_particle_ID_only_with_value
        )
        if metric == 'number_of_turns':
            metric_label = 'Number of turns'
        elif metric == 'turns_per_second':
            metric_label = 'Turns per second'
        else:
            distance_unit = self.__plot_unit_label(
                turn_summary['distance_unit'].iloc[0]
            )
            metric_label = f'Turns per {distance_unit}'
        if plot_style == 'polar':
            bar_colors = self.__resolve_bar_colors(
                color=color,
                bar_count=len(values),
                default_color=plt.get_cmap('viridis')
            )
            width = np.pi / max(len(values), 1)
            angles = (
                np.linspace(0, np.pi, len(values), endpoint=False) + width / 2
            )
            fig, ax = plt.subplots(
                figsize=figsize, subplot_kw={'projection': 'polar'}
            )
            ax.bar(
                angles, values, width=width, color=bar_colors,
                edgecolor='black'
            )
            ax.set_thetamin(0)
            ax.set_thetamax(180)
            ax.set_theta_zero_location('W')
            ax.set_theta_direction(-1)
            if show_particle_ID:
                ax.set_xticks(angles[particle_label_indices])
                ax.set_xticklabels(particle_labels[particle_label_indices])
            else:
                ax.set_xticks([])
            if np.nanmax(values) <= 0:
                ax.set_ylim(0, 1)
        else:
            bar_colors = self.__resolve_bar_colors(
                color=color,
                bar_count=len(values),
                default_color='#457B9D'
            )
            fig, ax = plt.subplots(figsize=figsize)
            particle_positions = np.arange(len(values))
            ax.bar(
                particle_positions, values, color=bar_colors
            )
            if show_particle_ID:
                ax.set_xticks(particle_positions[particle_label_indices])
                ax.set_xticklabels(particle_labels[particle_label_indices])
            else:
                ax.set_xticks([])
            ax.set_xlabel('Particle')
            ax.tick_params(axis='x', labelrotation=90)
        self.__format_particle_metric_axis(
            axis=ax,
            default_title=metric_label,
            title=title,
            title_fontsize=title_fontsize,
            x_axis_fontsize=x_axis_fontsize,
            y_axis_fontsize=y_axis_fontsize,
            x_tick_fontsize=x_tick_fontsize,
            y_tick_fontsize=y_tick_fontsize,
            y_tick_spacing=y_tick_spacing,
            font_family=font_family
        )
        if plot_style == 'bar':
            ax.set_ylabel(metric_label)
        fig.tight_layout()
        save_path = self.__resolve_numbered_plot_save_path(
            save_path=save_path,
            save_plots=plotting_parameters['save_plots'],
            save_plot_path=plotting_parameters['save_plot_path'],
            file_extension=plotting_parameters['file_extension'],
            default_title=metric_label,
            title=title
        )
        self.__finalize_figure(fig, save_path, dpi, show)
        return fig, ax

    def calculate_angle_tumble_statistics(
        self,
        particle_ids: int | list[int] | tuple[int, ...] | None = None,
        discard_initial_frames: int = 10,
        discard_final_frames: int = 10,
        tumble_threshold_angle: float = 35.0,
        smoothing_method: str = 'moving_average',
        moving_average_window: int = 5,
        triangular_smoothing_window: int | None = 17,
        sg_filter_window_length: int | None = 5,
        sg_filter_polyorder: int = 2,
        rdp_epsilon_pixels: float = 5.0,
        distance_unit: str = 'scale_units',
        max_frame_gap: int | None = 1
    ) -> tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
    ]:
        """
        Classify 2D runs and tumbles using the Turner et al. method.

        The implementation follows the trajectory analysis in Turner et al.,
        Biophysical Journal 111 (2016), 630-639: velocity uses a five-point
        fourth-order central derivative, a run requires at least three
        successive angles at or below the threshold, and a tumble requires at
        least two successive angles above it. An isolated above-threshold point
        is retained only when the paper's summed-neighbor-vector test also
        exceeds the threshold. This implementation discards detections in the
        first and last 10 source-frame positions by default.

        This is a 2D adaptation. ``smoothing_method='moving_average'`` is the
        paper-aligned default. ``'triangular_smoothing'`` and
        ``'sg_filter_rdp'`` are alternative analyses. SG-filtered RDP vertices
        are interpolated back onto every original time point before temporal
        derivatives are calculated. Analysis uses Cartesian image coordinates
        (x = centroid_y and y = -centroid_x) without changing the stored
        centroid columns.
        Tumble-frequency time is the total post-discard observed trajectory
        time across retained segments; excluded frame gaps are not counted.
        Short retained segments still contribute to trajectory time even when
        they contain too few points for angle classification.

        Args:
            particle_ids: Optional particle ID or collection of IDs.
            discard_initial_frames: Number of source frames discarded from the
                start of each track. Only detections in that inclusive frame
                window are removed.
            discard_final_frames: Number of source frames discarded from the
                end of each track. Only detections in that inclusive frame
                window are removed.
            tumble_threshold_angle: Direction-change threshold in degrees.
                The paper used 35 degrees per point.
            smoothing_method: 'moving_average', 'triangular_smoothing', or
                'sg_filter_rdp'.
            moving_average_window: Complete centered odd moving-average span
                of at least 3 points. Turner et al. reported spans from 3
                through 9.
                Only positions with a complete centered window contribute to
                velocity and angle classification.
            triangular_smoothing_window: Odd triangular-filter span of at least
                3 points, with edge-value padding.
            sg_filter_window_length: Odd SG-filter span of at least 3 points.
            sg_filter_polyorder: Nonnegative SG polynomial order smaller than
                sg_filter_window_length.
            rdp_epsilon_pixels: RDP perpendicular-distance tolerance in pixels,
                used after the SG filter.
            distance_unit: 'pixels', 'scale_units', or the configured Capture
                scale unit. Scaling is applied automatically.
            max_frame_gap: Split a track when adjacent detections are farther
                apart than this many frames. None permits all gaps. Uneven
                retained sampling uses the corresponding five-point
                finite-difference weights instead of assuming unit frame steps.

        Returns:
            tuple[pd.DataFrame, ...]:
                Per-particle summaries, classified run/tumble events,
                point-level angle classifications, and an equal-particle-weight
                population summary, followed by a vertical Turner Table 1-style
                population dataframe. Event ``point_count`` is the number of
                classified angle points. ``support_end_frame`` is the following
                detection used by the event's last angle. Event intervals
                follow the paper and use end frame minus start frame; the
                additional sampling-support duration includes that following
                detection.
                In the per-particle summary, ``mean_run_speed`` and
                ``std_run_mean_speed_between_events`` summarize run-event mean
                speeds with equal event weight. ``mean_run_point_speed`` and
                ``std_run_point_speed`` instead pool all finite instantaneous
                speeds classified as run; ``run_speed_point_count`` reports
                the size of that point-level pool.
        """
        self._tumble_summary_dataframe = pd.DataFrame(
            columns=self.TUMBLE_SUMMARY_COLUMNS
        )
        self._tumble_events_dataframe = pd.DataFrame(
            columns=self.TUMBLE_EVENT_COLUMNS
        )
        self._tumble_points_dataframe = pd.DataFrame(
            columns=self.TUMBLE_POINT_COLUMNS
        )
        self._tumble_population_dataframe = pd.DataFrame(
            columns=self.TUMBLE_POPULATION_COLUMNS
        )
        self._angle_tumble_table_1_dataframe = pd.DataFrame(
            columns=self.TABLE_1_VERTICAL_COLUMNS
        )
        self._tumble_path_data = {}
        self._tumble_analysis_metadata = {}

        self.__validate_nonnegative_integer(
            discard_initial_frames, 'discard_initial_frames'
        )
        self.__validate_nonnegative_integer(
            discard_final_frames, 'discard_final_frames'
        )
        if not isinstance(
            tumble_threshold_angle, (int, float, np.integer, np.floating)
        ) or isinstance(tumble_threshold_angle, (bool, np.bool_)):
            raise TypeError('tumble_threshold_angle must be numeric.')
        tumble_threshold_angle = float(tumble_threshold_angle)
        if (
            not np.isfinite(tumble_threshold_angle) or
            not 0 <= tumble_threshold_angle <= 180
        ):
            raise ValueError(
                'tumble_threshold_angle must be finite and between 0 and 180.'
            )
        smoothing_configuration = (
            self.__normalize_smoothing_configuration(
                smoothing_method=smoothing_method,
                moving_average_window=moving_average_window,
                triangular_smoothing_window=(
                    triangular_smoothing_window
                ),
                sg_filter_window_length=sg_filter_window_length,
                sg_filter_polyorder=sg_filter_polyorder,
                rdp_epsilon_pixels=rdp_epsilon_pixels
            )
        )
        smoothing_output = self.__smoothing_configuration_output(
            smoothing_configuration
        )

        self.__validate_frame_rate()
        normalized_max_gap = self.__normalize_max_frame_gap(max_frame_gap)
        if normalized_max_gap is None or normalized_max_gap > 1:
            warnings.warn(
                'Retained frame gaps make smoothing irregularly sampled: the '
                'selected window operates over detection order, while temporal '
                'derivatives use real elapsed times. Use max_frame_gap=1 for '
                'the paper-aligned sampling mode.',
                UserWarning,
                stacklevel=2
            )
        _, distance_factor, distance_unit_label = self.__normalize_length_unit(
            distance_unit,
            argument_name='distance_unit'
        )
        speed_unit_label = f'{distance_unit_label}/s'
        selected_dataframe = self.__select_particle_rows(particle_ids)

        summary_rows = []
        event_rows = []
        point_rows = []
        tumble_path_data: dict[int, list[dict]] = {}
        grouped_particles = selected_dataframe.groupby('particle', sort=True)
        for particle_id, particle_rows in tqdm(
            grouped_particles,
            total=selected_dataframe['particle'].nunique(),
            desc='Classifying runs and tumbles'
        ):
            particle_id = int(particle_id)
            particle_rows = particle_rows.sort_values(
                by='frame', kind='stable'
            ).reset_index(drop=True)
            original_frames = particle_rows['frame'].to_numpy(dtype=int)
            analyzed_rows = self.__trim_particle_frame_windows(
                particle_rows,
                discard_initial_frames=discard_initial_frames,
                discard_final_frames=discard_final_frames
            ).reset_index(drop=True)
            discarded_count = len(particle_rows) - len(analyzed_rows)
            analyzed_frames = analyzed_rows['frame'].to_numpy(dtype=int)
            segments = (
                self.__split_particle_track(
                    analyzed_rows, normalized_max_gap
                )
                if not analyzed_rows.empty
                else []
            )
            particle_event_rows = []
            particle_point_rows = []
            tumble_path_data[particle_id] = []
            analyzed_tracking_time_seconds = 0.0
            elapsed_observed_time_seconds = 0.0
            next_event_id = 1

            for segment_id, segment_rows in enumerate(segments, start=1):
                segment_frames = segment_rows['frame'].to_numpy(dtype=int)
                segment_duration_seconds = 0.0
                if len(segment_frames) > 1:
                    segment_duration_seconds = (
                        int(segment_frames[-1]) - int(segment_frames[0])
                    ) / self._capture_speed_in_fps
                (
                    segment_point_rows,
                    segment_event_rows,
                    next_event_id,
                    segment_path_data,
                ) = self.__analyze_tumble_segment(
                    particle_id=particle_id,
                    segment_id=segment_id,
                    segment_rows=segment_rows,
                    elapsed_time_offset_seconds=(
                        elapsed_observed_time_seconds
                    ),
                    starting_event_id=next_event_id,
                    threshold_degrees=tumble_threshold_angle,
                    smoothing_configuration=smoothing_configuration,
                    distance_factor=distance_factor,
                    distance_unit=distance_unit_label,
                    speed_unit=speed_unit_label
                )
                particle_point_rows.extend(segment_point_rows)
                particle_event_rows.extend(segment_event_rows)
                tumble_path_data[particle_id].append(segment_path_data)
                analyzed_tracking_time_seconds += segment_duration_seconds
                elapsed_observed_time_seconds += segment_duration_seconds

            summary_rows.append(self.__summarize_tumble_particle(
                particle_id=particle_id,
                original_frames=original_frames,
                analyzed_frames=analyzed_frames,
                discarded_count=discarded_count,
                segment_count=len(segments),
                analyzed_tracking_time_seconds=(
                    analyzed_tracking_time_seconds
                ),
                particle_point_rows=particle_point_rows,
                particle_event_rows=particle_event_rows,
                distance_unit=distance_unit_label,
                speed_unit=speed_unit_label,
                smoothing_configuration=smoothing_configuration,
                discard_initial_frames=int(discard_initial_frames),
                discard_final_frames=int(discard_final_frames),
                tumble_threshold_angle=tumble_threshold_angle,
                max_frame_gap=normalized_max_gap
            ))
            point_rows.extend(particle_point_rows)
            event_rows.extend(particle_event_rows)

        tumble_summary = pd.DataFrame(
            summary_rows, columns=self.TUMBLE_SUMMARY_COLUMNS
        )
        tumble_events = pd.DataFrame(
            event_rows, columns=self.TUMBLE_EVENT_COLUMNS
        )
        tumble_points = pd.DataFrame(
            point_rows, columns=self.TUMBLE_POINT_COLUMNS
        )
        population_summary = self.__summarize_tumble_population(
            tumble_summary=tumble_summary,
            distance_unit=distance_unit_label,
            speed_unit=speed_unit_label,
            smoothing_configuration=smoothing_configuration,
            discard_initial_frames=int(discard_initial_frames),
            discard_final_frames=int(discard_final_frames),
            tumble_threshold_angle=tumble_threshold_angle,
            max_frame_gap=normalized_max_gap
        )
        table_1_summary = self.__build_angle_tumble_table_1(
            tumble_summary=tumble_summary,
            selected_dataframe=selected_dataframe,
            distance_factor=distance_factor,
            distance_unit=distance_unit_label,
            speed_unit=speed_unit_label,
            discard_initial_frames=int(discard_initial_frames),
            discard_final_frames=int(discard_final_frames)
        )

        tumble_summary = self.__add_strain_field(tumble_summary)
        tumble_events = self.__add_strain_field(tumble_events)
        tumble_points = self.__add_strain_field(tumble_points)
        population_summary = self.__add_strain_field(population_summary)
        table_1_summary = self.__add_strain_field(table_1_summary)

        self._tumble_summary_dataframe = tumble_summary
        self._tumble_events_dataframe = tumble_events
        self._tumble_points_dataframe = tumble_points
        self._tumble_population_dataframe = population_summary
        self._angle_tumble_table_1_dataframe = table_1_summary
        self._tumble_path_data = tumble_path_data
        self._tumble_analysis_metadata = {
            'source_dataframe': self._resolved_source_dataframe,
            'strain': self._strain,
            'discard_initial_frames': int(discard_initial_frames),
            'discard_final_frames': int(discard_final_frames),
            'tumble_threshold_angle_degrees': tumble_threshold_angle,
            **smoothing_output,
            'rdp_output_mode': (
                'linear interpolation on source-frame coordinates'
                if smoothing_configuration[
                    'smoothing_method'
                ] == 'sg_filter_rdp'
                else 'not_applied'
            ),
            'distance_factor': distance_factor,
            'distance_unit': distance_unit_label,
            'speed_unit': speed_unit_label,
            'max_frame_gap': normalized_max_gap,
            'analysis_dimensions': 2,
            'method_reference': (
                'Turner et al., Biophysical Journal 111 (2016), 630-639'
            ),
            'table_1_population_rule': (
                'Motion metrics weight finite per-cell means equally. Sample '
                'SD and SEM use the metric-specific finite cell count.'
            ),
            'table_1_body_size_rule': (
                'Body diameter and length are equal-cell means of each '
                'particle mean minor-axis and major-axis region-property '
                'measurements after the configured boundary discards.'
            ),
        }
        return (
            tumble_summary.copy(),
            tumble_events.copy(),
            tumble_points.copy(),
            population_summary.copy(),
            table_1_summary.copy()
        )

    def get_angle_tumble_statistics_dataframes(
        self
    ) -> tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
    ]:
        """Return copies of the latest angle-tumble result dataframes."""
        return (
            self._tumble_summary_dataframe.copy(),
            self._tumble_events_dataframe.copy(),
            self._tumble_points_dataframe.copy(),
            self._tumble_population_dataframe.copy(),
            self._angle_tumble_table_1_dataframe.copy()
        )

    def calculate_velocity_tumble_statistics(
        self,
        particle_ids: int | list[int] | tuple[int, ...] | None = None,
        discard_initial_frames: int = 10,
        discard_final_frames: int = 10,
        smoothing_method: str = 'triangular_smoothing',
        moving_average_window: int = 5,
        triangular_smoothing_window: int | None = 17,
        sg_filter_window_length: int | None = 5,
        sg_filter_polyorder: int = 2,
        rdp_epsilon_pixels: float = 5.0,
        speed_drop_ratio_threshold: float = 0.7,
        speed_period_depth_fraction: float = 0.2,
        angular_change_coefficient: float = 0.8,
        minimum_heading_speed: float | None = None,
        speed_extrema_prominence: float | None = None,
        angular_velocity_extrema_prominence: float | None = None,
        extrema_min_distance: int = 1,
        persistence_interval_seconds: float = 1 / 6,
        run_direction_fit_points: int = 4,
        distance_unit: str = 'scale_units',
        max_frame_gap: int | None = 1,
        bootstrap_resamples: int = 10_000,
        bootstrap_confidence_level: float = 0.95,
        bootstrap_random_seed: int | None = 0
    ) -> tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
    ]:
        """
        Detect tumbles from coincident speed dips and angular-speed peaks.

        This implements the method in Najafi et al., Science Advances 4
        (2018), eaar6425. By default, each gap-bounded trajectory is first
        smoothed with the paper implementation's triangular moving filter.
        Moving-average and SG-filtered RDP alternatives are also available.
        Backward finite differences then give velocity, speed, heading, and
        angular-velocity magnitude.

        A speed minimum is a candidate when its depth relative to the minimum
        is at least ``speed_drop_ratio_threshold``. Its candidate period is the
        connected region around that minimum satisfying
        ``v(t) - v_min <= speed_period_depth_fraction * delta_v``. An
        angular-velocity maximum is a candidate when the cumulative absolute
        directional change between its surrounding angular minima is at least
        ``sqrt(angular_change_coefficient * elapsed_seconds)``. A tumble is
        accepted only when qualifying speed and angular candidate periods
        overlap. As specified by the paper, the accepted tumble duration uses
        the angular period only. Overlapping or directly adjacent accepted
        angular periods are merged.

        The associated implementation used a 17-point triangular filter at
        60 frames per second. This implementation exposes that span and uses it
        by default, with edge-value padding and genuine two-sided local
        bracketing extrema. Optional prominence and minimum-distance arguments
        make visual calibration reproducible.
        Zero-speed headings are undefined; direction changes bridge from the
        last valid heading to the next valid heading so a pause cannot create
        an artificial turn. Segments with fewer than five angular-velocity
        samples remain unclassified rather than being assumed to be runs.
        Tumble frequency and time fraction use classified tracking time, while
        total observed and unclassified times are reported separately.
        Point positions use the RABiTPy-compatible Cartesian image convention
        (x = centroid_y, y = -centroid_x), and elapsed time is cumulative
        observed time with excluded inter-segment gaps omitted.
        Backward-difference velocity is assigned to its ending frame; the point
        table records the preceding frame and angular-heading support explicitly.
        The Table 1 quantities are also reported as
        ``pooled_time_weighted_vR``, ``pooled_time_weighted_vT``, ``t_R``,
        ``t_T``, ``p``, and ``R``. Directional persistence ``p`` is the mean
        cosine of within-run heading changes over
        ``persistence_interval_seconds``. Consecutive-run persistence ``R`` is
        the mean cosine of the direction change between fitted run directions.
        The run-only rotational mean square displacement is calculated from
        squared differences of unwrapped headings at each supported source-
        frame lag. A zero-intercept fit of ``RMSD(tau) = 2 * Dr * tau`` uses
        positive lags through the pooled mean of all reconstructed run
        intervals, including boundary-censored intervals. The fitted ``Dr``
        and its fit-support diagnostics are exported in the vertical Table 1
        dataframe.
        ``t_R`` and ``t_T`` use complete episodes; boundary-censored episode
        counts are retained in explicit diagnostic columns.
        ``point_mean_run_speed`` and ``point_mean_tumble_speed`` average all
        finite point speeds carrying the corresponding state label equally.
        ``time_weighted_vR`` and ``time_weighted_vT`` instead use only
        same-state intervals and weight their ending-point speeds by elapsed
        interval duration; state-transition intervals contribute to neither.
        Uncertainty for ``pooled_time_weighted_vR`` and
        ``pooled_time_weighted_vT`` is estimated with percentile
        particle-cluster bootstrap confidence intervals. Each resample draws
        contributing particles with replacement and recomputes the pooled,
        speed-time-weighted value. Equal-particle speed means are accompanied
        by their between-particle sample standard deviations and standard
        errors.

        Args:
            particle_ids: Optional particle ID or collection of IDs.
            discard_initial_frames: Number of source frames discarded from the
                start of each track. Only detections in that inclusive frame
                window are removed.
            discard_final_frames: Number of source frames discarded from the
                end of each track. Only detections in that inclusive frame
                window are removed.
            smoothing_method: 'moving_average', 'triangular_smoothing', or
                'sg_filter_rdp'. The triangular default is paper-aligned; the
                other choices are alternative analyses.
            moving_average_window: Complete centered odd moving-average span
                of at least 3 trajectory points. Boundary detections without
                a complete window are excluded from this velocity analysis.
            triangular_smoothing_window: Odd triangular-filter span of at
                least 3 points. The paper implementation used 17 points at
                60 frames per second. Short segments use their largest possible
                odd span.
            sg_filter_window_length: Odd SG-filter span of at least 3 points.
            sg_filter_polyorder: Nonnegative SG polynomial order smaller than
                sg_filter_window_length.
            rdp_epsilon_pixels: RDP perpendicular-distance tolerance in pixels,
                used after the SG filter. The retained path is linearly
                interpolated to the original frame support before derivatives.
            speed_drop_ratio_threshold: Required delta_v / v_min. The paper
                used 0.7.
            speed_period_depth_fraction: Fraction of delta_v defining the
                speed candidate period. The paper used 0.2.
            angular_change_coefficient: Coefficient inside the square-root
                angular threshold. The paper used 0.8 with seconds and radians.
            minimum_heading_speed: Optional noise-floor speed in the selected
                distance unit per second. Headings at or below this speed are
                undefined and bridged between the nearest valid directions.
                None rejects only numerically zero speeds.
            speed_extrema_prominence: Optional speed-dip prominence in the
                selected distance unit per second, applied to candidate speed
                minima. None retains all candidate minima.
            angular_velocity_extrema_prominence: Optional angular-speed peak
                prominence in radians per second, applied to candidate maxima.
                None retains all candidate maxima.
            extrema_min_distance: Minimum separation between candidate extrema
                in retained kinematic samples.
            persistence_interval_seconds: Time separation used to calculate
                within-run directional persistence ``p``. The paper used 1/6 s.
                Heading at the target time is circularly interpolated within
                the same contiguous run; tumble and frame-gap boundaries are
                never crossed.
            run_direction_fit_points: Number of smoothed trajectory points used
                in each linear fit for the directions before and after a tumble.
                The paper used four points.
            distance_unit: 'pixels', 'scale_units', or the configured Capture
                scale unit. Spatial scaling is applied automatically.
            max_frame_gap: Split tracks before smoothing and differentiation
                when adjacent detections are farther apart than this many
                frames. None permits all gaps while still using their real
                elapsed times. The default of 1 best matches the paper's
                uniformly sampled trajectories; larger values apply the
                selected smoother to retained detections.
            bootstrap_resamples: Number of whole-particle resamples used for
                percentile confidence intervals around pooled time-weighted
                ``vR`` and ``vT``. Must be at least 2.
            bootstrap_confidence_level: Central percentile confidence level for
                pooled time-weighted ``vR`` and ``vT``. Must be strictly
                between 0 and 1.
            bootstrap_random_seed: Nonnegative seed for reproducible particle-
                cluster resampling. None requests nondeterministic entropy.

        Returns:
            tuple[pd.DataFrame, ...]:
                Per-particle summaries, detected tumble events, point-level
                kinematics/classifications, a population summary, and a
                vertical Najafi Table 1-style dataframe. The particle and
                population tables include the Table 1 quantities
                ``time_weighted_vR``, ``time_weighted_vT``, ``t_R``, ``t_T``,
                ``p``, and ``R``. The population table uses
                ``pooled_time_weighted_vR`` and
                ``pooled_time_weighted_vT`` for the pooled speed quantities.
                The vertical table also includes ``Dr (rad²/s)`` from the
                run-only rotational mean-square-displacement fit.
                Population Table 1 values are pooled using their speed-time,
                complete-episode, direction-pair, or run-transition support;
                pooled speeds include particle-cluster bootstrap confidence
                intervals. Equal-particle speed means, sample standard
                deviations, and standard errors are separate.
                The ``t_R`` sample SD and SEM use all complete, uncensored run
                episodes. Sample SDs for ``t_T``, ``p``, and ``R`` use all
                contributing complete tumble episodes, within-run direction
                pairs, and fitted run transitions, respectively. Every vertical
                table row includes ``n`` and an explanation of its sample unit;
                low-sample uncertainty rows explain why their value is blank.
        """
        self._velocity_tumble_summary_dataframe = pd.DataFrame(
            columns=self.VELOCITY_TUMBLE_SUMMARY_COLUMNS
        )
        self._velocity_tumble_events_dataframe = pd.DataFrame(
            columns=self.VELOCITY_TUMBLE_EVENT_COLUMNS
        )
        self._velocity_tumble_points_dataframe = pd.DataFrame(
            columns=self.VELOCITY_TUMBLE_POINT_COLUMNS
        )
        self._velocity_tumble_population_dataframe = pd.DataFrame(
            columns=self.VELOCITY_TUMBLE_POPULATION_COLUMNS
        )
        self._velocity_tumble_table_1_dataframe = pd.DataFrame(
            columns=self.VELOCITY_TABLE_1_VERTICAL_COLUMNS
        )
        self._velocity_tumble_analysis_metadata = {}

        self.__validate_nonnegative_integer(
            discard_initial_frames, 'discard_initial_frames'
        )
        self.__validate_nonnegative_integer(
            discard_final_frames, 'discard_final_frames'
        )
        smoothing_configuration = (
            self.__normalize_smoothing_configuration(
                smoothing_method=smoothing_method,
                moving_average_window=moving_average_window,
                triangular_smoothing_window=(
                    triangular_smoothing_window
                ),
                sg_filter_window_length=sg_filter_window_length,
                sg_filter_polyorder=sg_filter_polyorder,
                rdp_epsilon_pixels=rdp_epsilon_pixels
            )
        )
        smoothing_output = self.__smoothing_configuration_output(
            smoothing_configuration
        )

        numeric_parameters = {
            'speed_drop_ratio_threshold': speed_drop_ratio_threshold,
            'speed_period_depth_fraction': speed_period_depth_fraction,
            'angular_change_coefficient': angular_change_coefficient,
        }
        normalized_numeric_parameters = {}
        for parameter_name, parameter_value in numeric_parameters.items():
            if isinstance(
                parameter_value, (bool, np.bool_)
            ) or not isinstance(
                parameter_value, (int, float, np.integer, np.floating)
            ):
                raise TypeError(f'{parameter_name} must be numeric.')
            normalized_value = float(parameter_value)
            if not np.isfinite(normalized_value) or normalized_value < 0:
                raise ValueError(
                    f'{parameter_name} must be finite and nonnegative.'
                )
            normalized_numeric_parameters[parameter_name] = normalized_value
        speed_drop_ratio_threshold = normalized_numeric_parameters[
            'speed_drop_ratio_threshold'
        ]
        speed_period_depth_fraction = normalized_numeric_parameters[
            'speed_period_depth_fraction'
        ]
        angular_change_coefficient = normalized_numeric_parameters[
            'angular_change_coefficient'
        ]
        if speed_period_depth_fraction > 1:
            raise ValueError(
                'speed_period_depth_fraction must be between 0 and 1.'
            )
        if minimum_heading_speed is not None:
            if isinstance(
                minimum_heading_speed, (bool, np.bool_)
            ) or not isinstance(
                minimum_heading_speed,
                (int, float, np.integer, np.floating)
            ):
                raise TypeError(
                    'minimum_heading_speed must be numeric or None.'
                )
            minimum_heading_speed = float(minimum_heading_speed)
            if (
                not np.isfinite(minimum_heading_speed) or
                minimum_heading_speed < 0
            ):
                raise ValueError(
                    'minimum_heading_speed must be finite and nonnegative.'
                )

        normalized_prominences = {}
        for parameter_name, parameter_value in {
            'speed_extrema_prominence': speed_extrema_prominence,
            'angular_velocity_extrema_prominence': (
                angular_velocity_extrema_prominence
            ),
        }.items():
            if parameter_value is None:
                normalized_prominences[parameter_name] = None
                continue
            if isinstance(
                parameter_value, (bool, np.bool_)
            ) or not isinstance(
                parameter_value, (int, float, np.integer, np.floating)
            ):
                raise TypeError(f'{parameter_name} must be numeric or None.')
            normalized_value = float(parameter_value)
            if not np.isfinite(normalized_value) or normalized_value < 0:
                raise ValueError(
                    f'{parameter_name} must be finite and nonnegative.'
                )
            normalized_prominences[parameter_name] = normalized_value
        speed_extrema_prominence = normalized_prominences[
            'speed_extrema_prominence'
        ]
        angular_velocity_extrema_prominence = normalized_prominences[
            'angular_velocity_extrema_prominence'
        ]
        self.__validate_positive_integer(
            extrema_min_distance, 'extrema_min_distance'
        )
        extrema_min_distance = int(extrema_min_distance)
        if isinstance(
            persistence_interval_seconds, (bool, np.bool_)
        ) or not isinstance(
            persistence_interval_seconds,
            (int, float, np.integer, np.floating)
        ):
            raise TypeError('persistence_interval_seconds must be numeric.')
        persistence_interval_seconds = float(
            persistence_interval_seconds
        )
        if (
            not np.isfinite(persistence_interval_seconds) or
            persistence_interval_seconds <= 0
        ):
            raise ValueError(
                'persistence_interval_seconds must be finite and positive.'
            )
        self.__validate_positive_integer(
            run_direction_fit_points, 'run_direction_fit_points'
        )
        run_direction_fit_points = int(run_direction_fit_points)
        if run_direction_fit_points < 2:
            raise ValueError(
                'run_direction_fit_points must be at least 2.'
            )
        self.__validate_positive_integer(
            bootstrap_resamples, 'bootstrap_resamples'
        )
        bootstrap_resamples = int(bootstrap_resamples)
        if bootstrap_resamples < 2:
            raise ValueError('bootstrap_resamples must be at least 2.')
        if isinstance(
            bootstrap_confidence_level, (bool, np.bool_)
        ) or not isinstance(
            bootstrap_confidence_level,
            (int, float, np.integer, np.floating)
        ):
            raise TypeError('bootstrap_confidence_level must be numeric.')
        bootstrap_confidence_level = float(bootstrap_confidence_level)
        if (
            not np.isfinite(bootstrap_confidence_level) or
            not 0 < bootstrap_confidence_level < 1
        ):
            raise ValueError(
                'bootstrap_confidence_level must be finite and strictly '
                'between 0 and 1.'
            )
        if bootstrap_random_seed is not None:
            self.__validate_nonnegative_integer(
                bootstrap_random_seed, 'bootstrap_random_seed'
            )
            bootstrap_random_seed = int(bootstrap_random_seed)

        self.__validate_frame_rate()
        normalized_max_gap = self.__normalize_max_frame_gap(max_frame_gap)
        if normalized_max_gap is None or normalized_max_gap > 1:
            warnings.warn(
                'Retained frame gaps make smoothing irregularly sampled: the '
                'selected window operates over detection order, while temporal '
                'derivatives use real elapsed times. Use max_frame_gap=1 for '
                'the paper-aligned sampling mode.',
                UserWarning,
                stacklevel=2
            )
        _, distance_factor, distance_unit_label = self.__normalize_length_unit(
            distance_unit,
            argument_name='distance_unit'
        )
        speed_unit_label = f'{distance_unit_label}/s'
        selected_dataframe = self.__select_particle_rows(particle_ids)

        summary_rows = []
        event_rows = []
        point_rows = []
        grouped_particles = selected_dataframe.groupby('particle', sort=True)
        for particle_id, particle_rows in tqdm(
            grouped_particles,
            total=selected_dataframe['particle'].nunique(),
            desc='Detecting velocity-based tumbles'
        ):
            particle_id = int(particle_id)
            particle_rows = particle_rows.sort_values(
                by='frame', kind='stable'
            ).reset_index(drop=True)
            original_frames = particle_rows['frame'].to_numpy(dtype=int)
            if (
                len(original_frames) > 1 and
                np.any(np.diff(original_frames) <= 0)
            ):
                raise ValueError(
                    f'Particle {particle_id} contains duplicate frame values; '
                    'velocity-based tumble analysis requires at most one '
                    'detection per particle and frame.'
                )
            analyzed_rows = self.__trim_particle_frame_windows(
                particle_rows,
                discard_initial_frames=discard_initial_frames,
                discard_final_frames=discard_final_frames
            ).reset_index(drop=True)
            discarded_count = len(particle_rows) - len(analyzed_rows)
            analyzed_frames = analyzed_rows['frame'].to_numpy(dtype=int)
            segments = (
                self.__split_particle_track(
                    analyzed_rows, normalized_max_gap
                )
                if not analyzed_rows.empty
                else []
            )

            particle_point_rows = []
            particle_event_rows = []
            segment_records = []
            next_event_id = 1
            elapsed_observed_time_seconds = 0.0
            for segment_id, segment_rows in enumerate(segments, start=1):
                (
                    segment_point_rows,
                    segment_event_rows,
                    next_event_id,
                ) = self.__analyze_velocity_tumble_segment(
                    particle_id=particle_id,
                    segment_id=segment_id,
                    segment_rows=segment_rows,
                    elapsed_time_offset_seconds=(
                        elapsed_observed_time_seconds
                    ),
                    starting_event_id=next_event_id,
                    smoothing_configuration=smoothing_configuration,
                    speed_drop_ratio_threshold=(
                        speed_drop_ratio_threshold
                    ),
                    speed_period_depth_fraction=(
                        speed_period_depth_fraction
                    ),
                    angular_change_coefficient=angular_change_coefficient,
                    minimum_heading_speed=minimum_heading_speed,
                    speed_extrema_prominence=speed_extrema_prominence,
                    angular_velocity_extrema_prominence=(
                        angular_velocity_extrema_prominence
                    ),
                    extrema_min_distance=extrema_min_distance,
                    distance_factor=distance_factor,
                    distance_unit=distance_unit_label,
                    speed_unit=speed_unit_label
                )
                original_segment_frames = segment_rows[
                    'frame'
                ].to_numpy(dtype=int)
                segment_frames = np.asarray([
                    row['frame']
                    for row in segment_point_rows
                    if (
                        np.isfinite(row['smoothed_position_x_pixels']) and
                        np.isfinite(row['smoothed_position_y_pixels'])
                    )
                ], dtype=int)
                observed_duration_seconds = (
                    (
                        int(original_segment_frames[-1]) -
                        int(original_segment_frames[0])
                    ) / self._capture_speed_in_fps
                    if len(original_segment_frames) > 1 else 0.0
                )
                classified_support_duration_seconds = (
                    (
                        int(segment_frames[-1]) -
                        int(segment_frames[0])
                    ) / self._capture_speed_in_fps
                    if len(segment_frames) > 1 else 0.0
                )
                finite_angular_count = sum(
                    np.isfinite(
                        row[
                            'angular_velocity_magnitude_radians_per_second'
                        ]
                    )
                    for row in segment_point_rows
                )
                segment_records.append({
                    'segment_id': segment_id,
                    'start_frame': (
                        int(segment_frames[0]) if len(segment_frames) else None
                    ),
                    'end_frame': (
                        int(segment_frames[-1]) if len(segment_frames) else None
                    ),
                    'duration_seconds': observed_duration_seconds,
                    'classified_support_duration_seconds': (
                        classified_support_duration_seconds
                    ),
                    'is_analyzable': finite_angular_count >= 5,
                })
                elapsed_observed_time_seconds += (
                    segment_records[-1]['duration_seconds']
                )
                particle_point_rows.extend(segment_point_rows)
                particle_event_rows.extend(segment_event_rows)

            summary_rows.append(
                self.__summarize_velocity_tumble_particle(
                    particle_id=particle_id,
                    original_frames=original_frames,
                    analyzed_frames=analyzed_frames,
                    discarded_count=discarded_count,
                    segment_records=segment_records,
                    point_rows=particle_point_rows,
                    event_rows=particle_event_rows,
                    distance_unit=distance_unit_label,
                    speed_unit=speed_unit_label,
                    smoothing_configuration=smoothing_configuration,
                    discard_initial_frames=int(discard_initial_frames),
                    discard_final_frames=int(discard_final_frames),
                    speed_drop_ratio_threshold=(
                        speed_drop_ratio_threshold
                    ),
                    speed_period_depth_fraction=(
                        speed_period_depth_fraction
                    ),
                    angular_change_coefficient=angular_change_coefficient,
                    minimum_heading_speed=minimum_heading_speed,
                    speed_extrema_prominence=speed_extrema_prominence,
                    angular_velocity_extrema_prominence=(
                        angular_velocity_extrema_prominence
                    ),
                    extrema_min_distance=extrema_min_distance,
                    persistence_interval_seconds=(
                        persistence_interval_seconds
                    ),
                    run_direction_fit_points=run_direction_fit_points,
                    max_frame_gap=normalized_max_gap
                )
            )
            point_rows.extend(particle_point_rows)
            event_rows.extend(particle_event_rows)

        velocity_tumble_summary = pd.DataFrame(
            summary_rows, columns=self.VELOCITY_TUMBLE_SUMMARY_COLUMNS
        )
        velocity_tumble_events = pd.DataFrame(
            event_rows, columns=self.VELOCITY_TUMBLE_EVENT_COLUMNS
        )
        velocity_tumble_points = pd.DataFrame(
            point_rows, columns=self.VELOCITY_TUMBLE_POINT_COLUMNS
        )
        velocity_tumble_population = (
            self.__summarize_velocity_tumble_population(
                tumble_summary=velocity_tumble_summary,
                distance_unit=distance_unit_label,
                speed_unit=speed_unit_label,
                smoothing_configuration=smoothing_configuration,
                discard_initial_frames=int(discard_initial_frames),
                discard_final_frames=int(discard_final_frames),
                speed_drop_ratio_threshold=speed_drop_ratio_threshold,
                speed_period_depth_fraction=speed_period_depth_fraction,
                angular_change_coefficient=angular_change_coefficient,
                minimum_heading_speed=minimum_heading_speed,
                speed_extrema_prominence=speed_extrema_prominence,
                angular_velocity_extrema_prominence=(
                    angular_velocity_extrema_prominence
                ),
                extrema_min_distance=extrema_min_distance,
                persistence_interval_seconds=(
                    persistence_interval_seconds
                ),
                run_direction_fit_points=run_direction_fit_points,
                max_frame_gap=normalized_max_gap,
                bootstrap_resamples=bootstrap_resamples,
                bootstrap_confidence_level=bootstrap_confidence_level,
                bootstrap_random_seed=bootstrap_random_seed
            )
        )
        angular_msd_fit = self.__calculate_velocity_run_angular_msd(
            points_dataframe=velocity_tumble_points,
            tumble_summary=velocity_tumble_summary
        )
        velocity_table_1_summary = self.__build_velocity_tumble_table_1(
            population_summary=velocity_tumble_population,
            angular_msd_fit=angular_msd_fit
        )

        velocity_tumble_summary = self.__add_strain_field(
            velocity_tumble_summary
        )
        velocity_tumble_events = self.__add_strain_field(
            velocity_tumble_events
        )
        velocity_tumble_points = self.__add_strain_field(
            velocity_tumble_points
        )
        velocity_tumble_population = self.__add_strain_field(
            velocity_tumble_population
        )
        velocity_table_1_summary = self.__add_strain_field(
            velocity_table_1_summary
        )

        self._velocity_tumble_summary_dataframe = velocity_tumble_summary
        self._velocity_tumble_events_dataframe = velocity_tumble_events
        self._velocity_tumble_points_dataframe = velocity_tumble_points
        self._velocity_tumble_population_dataframe = (
            velocity_tumble_population
        )
        self._velocity_tumble_table_1_dataframe = velocity_table_1_summary
        self._velocity_tumble_analysis_metadata = {
            'source_dataframe': self._resolved_source_dataframe,
            'strain': self._strain,
            'discard_initial_frames': int(discard_initial_frames),
            'discard_final_frames': int(discard_final_frames),
            **smoothing_output,
            'smoothing_boundary_rule': {
                'moving_average': (
                    'complete centered window; unsupported boundary '
                    'detections excluded'
                ),
                'triangular_smoothing': 'edge-value padding',
                'sg_filter_rdp': (
                    'SG interpolation followed by RDP reconstruction on '
                    'source-frame coordinates'
                ),
            }[smoothing_configuration['smoothing_method']],
            'rdp_output_mode': (
                'linear interpolation on source-frame coordinates'
                if smoothing_configuration[
                    'smoothing_method'
                ] == 'sg_filter_rdp'
                else 'not_applied'
            ),
            'speed_drop_ratio_threshold': speed_drop_ratio_threshold,
            'speed_period_depth_fraction': speed_period_depth_fraction,
            'angular_change_coefficient': angular_change_coefficient,
            'minimum_heading_speed': minimum_heading_speed,
            'speed_extrema_prominence': speed_extrema_prominence,
            'angular_velocity_extrema_prominence': (
                angular_velocity_extrema_prominence
            ),
            'extrema_min_distance': extrema_min_distance,
            'persistence_interval_seconds': persistence_interval_seconds,
            'run_direction_fit_points': run_direction_fit_points,
            'pooled_speed_bootstrap_resamples': bootstrap_resamples,
            'pooled_speed_bootstrap_confidence_level': (
                bootstrap_confidence_level
            ),
            'pooled_speed_bootstrap_random_seed': bootstrap_random_seed,
            'distance_factor': distance_factor,
            'distance_unit': distance_unit_label,
            'speed_unit': speed_unit_label,
            'angular_velocity_unit': 'rad/s',
            'max_frame_gap': normalized_max_gap,
            'analysis_dimensions': 2,
            'method_reference': (
                'Najafi et al., Science Advances 4 (2018), eaar6425'
            ),
            'duration_rule': (
                'Both speed and angular criteria confirm an event; the final '
                'duration is the angular-defined interval.'
            ),
            'zero_speed_heading_rule': (
                'Headings at or below the numerical/configured speed floor are '
                'skipped; angular change bridges between nearest valid headings.'
            ),
            'short_segment_rule': (
                'Segments with fewer than five finite angular-velocity samples '
                'remain unclassified.'
            ),
            'irregular_sampling_note': (
                'max_frame_gap=1 matches the uniformly sampled paper method; '
                'larger values use real elapsed times and apply the selected '
                'smoother over retained detections.'
            ),
            'kinematic_timestamp_rule': (
                'Backward-difference samples are assigned to their ending '
                'frames; support frames and elapsed times are included in the '
                'point and event tables.'
            ),
            'event_metric_support_rule': (
                'Event speed and path metrics use the angular event start and '
                'end indices under the paper endpoint-assigned convention; '
                'sampling-support columns are diagnostic only.'
            ),
            'frequency_denominator_rule': (
                'Tumble frequency and time fraction use kinematically '
                'classified support, excluding short segments and any '
                'smoother-unsupported boundary intervals.'
            ),
            'table_1_duration_rule': (
                't_R and t_T use complete episodes. First and last run '
                'periods and any event touching an analysis boundary are '
                'reported as censored and excluded from these means.'
            ),
            'table_1_population_rule': (
                'pooled_time_weighted_vR, pooled_time_weighted_vT, t_R, t_T, '
                'p, and R are pooled using their actual support; '
                'equal-particle means, sample standard deviations, and '
                'standard errors are reported separately.'
            ),
            'pooled_speed_bootstrap_rule': (
                'Pooled time-weighted vR and vT confidence intervals use a '
                'percentile particle-cluster bootstrap. Each replicate '
                'samples whole state-contributing particles with replacement '
                'and recomputes the duration-weighted pooled speed.'
            ),
            'table_1_t_R_uncertainty_rule': (
                'std_t_R is the sample SD across all complete uncensored run '
                'episodes; sem_t_R is std_t_R divided by the square root of '
                'their total count.'
            ),
            'table_1_uncertainty_rule': (
                'std_t_R and std_t_T are sample SDs across complete uncensored '
                'run and tumble episodes. std_p and std_R are sample SDs across '
                'supported within-run direction cosines and fitted run-to-run '
                'transition cosines. sem_t_R is std_t_R divided by the square '
                'root of the complete-run count. Sample SD and SEM require at '
                'least two contributing observations.'
            ),
            'table_1_sample_count_rule': (
                'Every vertical Table 1 row records n and n_definition. n is '
                'the metric-specific observation, particle-cluster, episode, '
                'direction-pair, transition, lag-bin, or dataset count; '
                'configuration and status rows mark n as not applicable.'
            ),
            'angular_msd_rule': (
                'Within each particle and segment, finite headings are split '
                'into contiguous run-only blocks and unwrapped. Squared '
                'heading differences are pair-pooled by exact source-frame '
                'lag without crossing tumbles, undefined headings, or segment '
                'boundaries.'
            ),
            'angular_msd_fit_rule': (
                'Within each exact-lag bin, squared direction changes are '
                'pair-pooled into one RMSD mean. Each lag-bin mean then '
                'receives equal regression weight in a least-squares fit '
                'constrained through the origin: RMSD(tau) = 2 * Dr * tau. '
                'The origin constraint and lag-bin weighting are explicit '
                'MiMoSA reproducibility conventions.'
            ),
            'angular_msd_fit_ceiling_rule': (
                'The maximum requested fit lag is the pooled mean of all '
                'reconstructed run intervals, including boundary-censored '
                'intervals: total_run_interval_seconds / number_of_runs.'
            ),
            **angular_msd_fit,
            'implementation_parameter_note': (
                'The associated implementation used a 17-point triangular '
                'filter at 60 frames per second. Other smoothing_method values '
                'are alternative analyses rather than paper reproductions. '
                'The edge rule, extrema prominence, and heading speed floor '
                'remain explicit MiMoSA arguments.'
            ),
        }
        return (
            velocity_tumble_summary.copy(),
            velocity_tumble_events.copy(),
            velocity_tumble_points.copy(),
            velocity_tumble_population.copy(),
            velocity_table_1_summary.copy()
        )

    def get_velocity_tumble_statistics_dataframes(
        self
    ) -> tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
    ]:
        """Return copies of the latest speed/angular tumble analysis."""
        return (
            self._velocity_tumble_summary_dataframe.copy(),
            self._velocity_tumble_events_dataframe.copy(),
            self._velocity_tumble_points_dataframe.copy(),
            self._velocity_tumble_population_dataframe.copy(),
            self._velocity_tumble_table_1_dataframe.copy()
        )

    def plot_particle_angle_tumble_metric(
        self,
        metric: str = 'number_of_tumbles',
        plot_style: str = 'bar',
        figsize: tuple[float, float] = (12, 6),
        save_path: str | os.PathLike | None = None,
        dpi: int | _UseConfiguredPlotValue = _USE_CONFIGURED_PLOT_VALUE,
        show: bool = True,
        show_particle_ID: (
            bool | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        color: str | Colormap | None = None,
        particle_ID_spacing: (
            int | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        show_particle_ID_only_with_value: (
            bool | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        title: (
            str | None | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        title_fontsize: (
            float | None | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        x_axis_fontsize: (
            float | None | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        y_axis_fontsize: (
            float | None | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        x_tick_fontsize: (
            float | None | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        y_tick_fontsize: (
            float | None | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        y_tick_spacing: (
            float | None | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        font_family: (
            str | None | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        save_plots: (
            bool | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        save_plot_path: (
            str | os.PathLike | None | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        file_extension: (
            str | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE
    ) -> tuple:
        """
        Plot one cached angle-tumble statistic per particle.

        Omitted configurable arguments use the values established by
        ``set_particle_metric_plotting_parameters``. Explicit arguments
        override those defaults for this plot only.

        Args:
            metric: Canonical per-particle result field to plot. Velocity
                choices include ``point_mean_run_speed``,
                ``point_mean_tumble_speed``, ``time_weighted_vR``, and
                ``time_weighted_vT``. The former ``mean_run_speed``,
                ``mean_tumble_speed``, ``v_R``, and ``v_T`` spellings remain
                accepted as plotting aliases.
            show_particle_ID: Label every Cartesian or polar bar with its
                corresponding particle ID.
            color: A Matplotlib color, colormap name, or Colormap object. A
                fixed color is applied to every bar; colormap colors are
                cycled across particles. None preserves the established
                orange bar and plasma polar defaults.
            particle_ID_spacing: X-tick spacing: label every nth eligible
                particle ID. Use a value greater than 1 to reduce crowding.
            show_particle_ID_only_with_value: If True, label only particles
                whose plotted metric is nonzero. Their zero-height bars remain
                in the plot.
            title: Custom plot title. None keeps the metric-based title.
            title_fontsize: Title font size in points. None uses Matplotlib's
                current default.
            x_axis_fontsize: X-axis label font size in points. None uses the
                current default. Cartesian plots have an x-axis label.
            y_axis_fontsize: Y-axis label font size in points. None uses the
                current default. Cartesian plots have a y-axis label.
            x_tick_fontsize: Particle-ID tick-label font size in points.
            y_tick_fontsize: Numeric or radial tick-label font size in points.
            y_tick_spacing: Numeric or radial major-tick interval. None lets
                Matplotlib choose the interval automatically.
            font_family: Typeface for the title, axis labels, and tick labels.
                Portable Matplotlib families are 'sans-serif', 'serif',
                'monospace', 'cursive', and 'fantasy'. Bundled faces include
                'DejaVu Sans', 'DejaVu Serif', and 'DejaVu Sans Mono'. An
                installed system-font family name can also be used. None keeps
                Matplotlib's current default.
            save_plots: If True and save_path is None, automatically generate
                a numbered filename and save the plot.
            save_plot_path: Directory used for automatic filenames. None uses
                the Stats output directory.
            file_extension: Automatic-save format: 'png' or 'tif'.
        """
        tumble_summary = self._tumble_summary_dataframe
        if tumble_summary.empty:
            raise ValueError(
                'Run calculate_angle_tumble_statistics before plotting '
                'angle-based tumble metrics.'
            )
        speed_unit = self.__plot_unit_label(
            tumble_summary['speed_unit'].iloc[0]
        )
        metric_labels = {
            'number_of_tumbles': 'Number of tumbles',
            'tumble_frequency_per_second': 'Tumble frequency (1/s)',
            'tumble_time_fraction': 'Fraction of analyzed time tumbling',
            'mean_tumble_interval_seconds': 'Mean tumble interval (s)',
            'mean_time_between_tumble_starts_seconds': (
                'Mean time between tumble starts (s)'
            ),
            'mean_tumble_speed': (
                f'Mean tumble speed ({speed_unit})'
            ),
            'mean_tumble_angular_speed_degrees_per_point': (
                'Mean tumble angular change (degrees/point)'
            ),
            'mean_run_to_run_direction_change_degrees': (
                'Mean run-to-run direction change (degrees)'
            ),
        }
        if metric not in metric_labels:
            raise ValueError(f'metric must be one of {sorted(metric_labels)}.')
        normalized_plot_style = str(plot_style).strip().lower()
        if normalized_plot_style not in {'bar', 'polar'}:
            raise ValueError("plot_style must be 'bar' or 'polar'.")
        plotting_parameters = (
            self.__resolve_particle_metric_plotting_parameters(
                dpi=dpi,
                show_particle_ID=show_particle_ID,
                particle_ID_spacing=particle_ID_spacing,
                show_particle_ID_only_with_value=(
                    show_particle_ID_only_with_value
                ),
                title=title,
                title_fontsize=title_fontsize,
                x_axis_fontsize=x_axis_fontsize,
                y_axis_fontsize=y_axis_fontsize,
                x_tick_fontsize=x_tick_fontsize,
                y_tick_fontsize=y_tick_fontsize,
                y_tick_spacing=y_tick_spacing,
                font_family=font_family,
                save_plots=save_plots,
                save_plot_path=save_plot_path,
                file_extension=file_extension
            )
        )
        dpi = plotting_parameters['dpi']
        show_particle_ID = plotting_parameters['show_particle_ID']
        particle_ID_spacing = plotting_parameters['particle_ID_spacing']
        show_particle_ID_only_with_value = plotting_parameters[
            'show_particle_ID_only_with_value'
        ]
        title = plotting_parameters['title']
        title_fontsize = plotting_parameters['title_fontsize']
        x_axis_fontsize = plotting_parameters['x_axis_fontsize']
        y_axis_fontsize = plotting_parameters['y_axis_fontsize']
        x_tick_fontsize = plotting_parameters['x_tick_fontsize']
        y_tick_fontsize = plotting_parameters['y_tick_fontsize']
        y_tick_spacing = plotting_parameters['y_tick_spacing']
        font_family = plotting_parameters['font_family']

        plot_dataframe = tumble_summary[
            ['particle', metric]
        ].replace([np.inf, -np.inf], np.nan).dropna()
        if plot_dataframe.empty:
            raise ValueError(f'No finite values are available for {metric}.')
        values = plot_dataframe[metric].to_numpy(dtype=float)
        particle_labels = (
            plot_dataframe['particle'].astype(int).astype(str).to_numpy()
        )
        particle_label_indices = self.__particle_label_indices(
            values,
            particle_ID_spacing,
            show_particle_ID_only_with_value
        )
        metric_label = metric_labels[metric]

        if normalized_plot_style == 'polar':
            if np.nanmin(values) < 0:
                raise ValueError(
                    'Polar plots require nonnegative metric values; use '
                    "plot_style='bar' for signed persistence values."
                )
            bar_colors = self.__resolve_bar_colors(
                color=color,
                bar_count=len(values),
                default_color=plt.get_cmap('plasma')
            )
            width = np.pi / max(len(values), 1)
            angles = (
                np.linspace(0, np.pi, len(values), endpoint=False) + width / 2
            )
            fig, ax = plt.subplots(
                figsize=figsize, subplot_kw={'projection': 'polar'}
            )
            ax.bar(
                angles, values, width=width, color=bar_colors,
                edgecolor='black'
            )
            ax.set_thetamin(0)
            ax.set_thetamax(180)
            ax.set_theta_zero_location('W')
            ax.set_theta_direction(-1)
            if show_particle_ID:
                ax.set_xticks(angles[particle_label_indices])
                ax.set_xticklabels(particle_labels[particle_label_indices])
            else:
                ax.set_xticks([])
            if np.nanmax(values) <= 0:
                ax.set_ylim(0, 1)
        else:
            bar_colors = self.__resolve_bar_colors(
                color=color,
                bar_count=len(values),
                default_color='#D55E00'
            )
            fig, ax = plt.subplots(figsize=figsize)
            particle_positions = np.arange(len(values))
            ax.bar(particle_positions, values, color=bar_colors)
            if show_particle_ID:
                ax.set_xticks(particle_positions[particle_label_indices])
                ax.set_xticklabels(particle_labels[particle_label_indices])
            else:
                ax.set_xticks([])
            ax.set_xlabel('Particle')
            ax.set_ylabel(metric_label)
            ax.tick_params(axis='x', labelrotation=90)
        self.__format_particle_metric_axis(
            axis=ax,
            default_title=metric_label,
            title=title,
            title_fontsize=title_fontsize,
            x_axis_fontsize=x_axis_fontsize,
            y_axis_fontsize=y_axis_fontsize,
            x_tick_fontsize=x_tick_fontsize,
            y_tick_fontsize=y_tick_fontsize,
            y_tick_spacing=y_tick_spacing,
            font_family=font_family
        )
        fig.tight_layout()
        save_path = self.__resolve_numbered_plot_save_path(
            save_path=save_path,
            save_plots=plotting_parameters['save_plots'],
            save_plot_path=plotting_parameters['save_plot_path'],
            file_extension=plotting_parameters['file_extension'],
            default_title=metric_label,
            title=title
        )
        self.__finalize_figure(fig, save_path, dpi, show)
        return fig, ax

    def plot_particle_velocity_tumble_metric(
        self,
        metric: str = 'number_of_tumbles',
        plot_style: str = 'bar',
        figsize: tuple[float, float] = (12, 6),
        save_path: str | os.PathLike | None = None,
        dpi: int | _UseConfiguredPlotValue = _USE_CONFIGURED_PLOT_VALUE,
        show: bool = True,
        show_particle_ID: (
            bool | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        color: str | Colormap | None = None,
        particle_ID_spacing: (
            int | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        show_particle_ID_only_with_value: (
            bool | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        title: (
            str | None | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        title_fontsize: (
            float | None | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        x_axis_fontsize: (
            float | None | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        y_axis_fontsize: (
            float | None | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        x_tick_fontsize: (
            float | None | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        y_tick_fontsize: (
            float | None | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        y_tick_spacing: (
            float | None | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        font_family: (
            str | None | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        save_plots: (
            bool | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        save_plot_path: (
            str | os.PathLike | None | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        file_extension: (
            str | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE
    ) -> tuple:
        """
        Plot one cached velocity-tumble statistic per particle.

        Omitted configurable arguments use the values established by
        ``set_particle_metric_plotting_parameters``. Explicit arguments
        override those defaults for this plot only.

        Args:
            show_particle_ID: Label every Cartesian or polar bar with its
                corresponding particle ID.
            color: A Matplotlib color, colormap name, or Colormap object. A
                fixed color is applied to every bar; colormap colors are
                cycled across particles. None preserves the established blue
                bar and viridis polar defaults.
            particle_ID_spacing: X-tick spacing: label every nth eligible
                particle ID. Use a value greater than 1 to reduce crowding.
            show_particle_ID_only_with_value: If True, label only particles
                whose plotted metric is nonzero. Their zero-height bars remain
                in the plot.
            title: Custom plot title. None keeps the metric-based title.
            title_fontsize: Title font size in points. None uses Matplotlib's
                current default.
            x_axis_fontsize: X-axis label font size in points. None uses the
                current default. Cartesian plots have an x-axis label.
            y_axis_fontsize: Y-axis label font size in points. None uses the
                current default. Cartesian plots have a y-axis label.
            x_tick_fontsize: Particle-ID tick-label font size in points.
            y_tick_fontsize: Numeric or radial tick-label font size in points.
            y_tick_spacing: Numeric or radial major-tick interval. None lets
                Matplotlib choose the interval automatically.
            font_family: Typeface for the title, axis labels, and tick labels.
                Portable Matplotlib families are 'sans-serif', 'serif',
                'monospace', 'cursive', and 'fantasy'. Bundled faces include
                'DejaVu Sans', 'DejaVu Serif', and 'DejaVu Sans Mono'. An
                installed system-font family name can also be used. None keeps
                Matplotlib's current default.
            save_plots: If True and save_path is None, automatically generate
                a numbered filename and save the plot.
            save_plot_path: Directory used for automatic filenames. None uses
                the Stats output directory.
            file_extension: Automatic-save format: 'png' or 'tif'.
        """
        tumble_summary = self._velocity_tumble_summary_dataframe
        if tumble_summary.empty:
            raise ValueError(
                'Run calculate_velocity_tumble_statistics before plotting '
                'velocity-based tumble metrics.'
            )
        metric = {
            'mean_run_speed': 'point_mean_run_speed',
            'mean_tumble_speed': 'point_mean_tumble_speed',
            'v_R': 'time_weighted_vR',
            'v_T': 'time_weighted_vT',
        }.get(metric, metric)
        speed_unit = self.__plot_unit_label(
            tumble_summary['speed_unit'].iloc[0]
        )
        metric_labels = {
            'number_of_tumbles': 'Number of velocity-based tumbles',
            'tumble_frequency_per_second': 'Tumble frequency (1/s)',
            'tumble_time_fraction': 'Fraction of classified time tumbling',
            'mean_time_between_tumble_starts_seconds': (
                'Mean time between tumble starts (s)'
            ),
            'point_mean_run_speed': (
                f'Point-mean run speed ({speed_unit})'
            ),
            'point_mean_tumble_speed': (
                f'Point-mean tumble speed ({speed_unit})'
            ),
            'mean_run_interval_seconds': (
                'Mean run interval including censored boundaries (s)'
            ),
            'mean_tumble_interval_seconds': (
                'Mean detected tumble interval (s)'
            ),
            'mean_run_angular_velocity_magnitude_radians_per_second': (
                'Mean run angular-velocity magnitude (rad/s)'
            ),
            'mean_tumble_angular_velocity_magnitude_radians_per_second': (
                'Mean tumble angular-velocity magnitude (rad/s)'
            ),
            'time_weighted_vR': f'Time-weighted vR ({speed_unit})',
            'time_weighted_vT': f'Time-weighted vT ({speed_unit})',
            't_R': 'Mean complete run time, t_R (s)',
            't_T': 'Mean complete tumble time, t_T (s)',
            'p': 'Run directional persistence, p',
            'R': 'Consecutive-run directional persistence, R',
        }
        if metric not in metric_labels:
            raise ValueError(f'metric must be one of {sorted(metric_labels)}.')
        normalized_plot_style = str(plot_style).strip().lower()
        if normalized_plot_style not in {'bar', 'polar'}:
            raise ValueError("plot_style must be 'bar' or 'polar'.")
        plotting_parameters = (
            self.__resolve_particle_metric_plotting_parameters(
                dpi=dpi,
                show_particle_ID=show_particle_ID,
                particle_ID_spacing=particle_ID_spacing,
                show_particle_ID_only_with_value=(
                    show_particle_ID_only_with_value
                ),
                title=title,
                title_fontsize=title_fontsize,
                x_axis_fontsize=x_axis_fontsize,
                y_axis_fontsize=y_axis_fontsize,
                x_tick_fontsize=x_tick_fontsize,
                y_tick_fontsize=y_tick_fontsize,
                y_tick_spacing=y_tick_spacing,
                font_family=font_family,
                save_plots=save_plots,
                save_plot_path=save_plot_path,
                file_extension=file_extension
            )
        )
        dpi = plotting_parameters['dpi']
        show_particle_ID = plotting_parameters['show_particle_ID']
        particle_ID_spacing = plotting_parameters['particle_ID_spacing']
        show_particle_ID_only_with_value = plotting_parameters[
            'show_particle_ID_only_with_value'
        ]
        title = plotting_parameters['title']
        title_fontsize = plotting_parameters['title_fontsize']
        x_axis_fontsize = plotting_parameters['x_axis_fontsize']
        y_axis_fontsize = plotting_parameters['y_axis_fontsize']
        x_tick_fontsize = plotting_parameters['x_tick_fontsize']
        y_tick_fontsize = plotting_parameters['y_tick_fontsize']
        y_tick_spacing = plotting_parameters['y_tick_spacing']
        font_family = plotting_parameters['font_family']

        plot_dataframe = tumble_summary[
            ['particle', metric]
        ].replace([np.inf, -np.inf], np.nan).dropna()
        if plot_dataframe.empty:
            raise ValueError(f'No finite values are available for {metric}.')
        values = plot_dataframe[metric].to_numpy(dtype=float)
        particle_labels = (
            plot_dataframe['particle'].astype(int).astype(str).to_numpy()
        )
        particle_label_indices = self.__particle_label_indices(
            values,
            particle_ID_spacing,
            show_particle_ID_only_with_value
        )
        metric_label = metric_labels[metric]

        if normalized_plot_style == 'polar':
            if np.nanmin(values) < 0:
                raise ValueError(
                    'Polar plots require nonnegative metric values; use '
                    "plot_style='bar' for signed persistence values."
                )
            bar_colors = self.__resolve_bar_colors(
                color=color,
                bar_count=len(values),
                default_color=plt.get_cmap('viridis')
            )
            width = np.pi / max(len(values), 1)
            angles = (
                np.linspace(0, np.pi, len(values), endpoint=False) + width / 2
            )
            fig, ax = plt.subplots(
                figsize=figsize, subplot_kw={'projection': 'polar'}
            )
            ax.bar(
                angles, values, width=width, color=bar_colors,
                edgecolor='black'
            )
            ax.set_thetamin(0)
            ax.set_thetamax(180)
            ax.set_theta_zero_location('W')
            ax.set_theta_direction(-1)
            if show_particle_ID:
                ax.set_xticks(angles[particle_label_indices])
                ax.set_xticklabels(particle_labels[particle_label_indices])
            else:
                ax.set_xticks([])
            if np.nanmax(values) <= 0:
                ax.set_ylim(0, 1)
        else:
            bar_colors = self.__resolve_bar_colors(
                color=color,
                bar_count=len(values),
                default_color='#0072B2'
            )
            fig, ax = plt.subplots(figsize=figsize)
            particle_positions = np.arange(len(values))
            ax.bar(particle_positions, values, color=bar_colors)
            if show_particle_ID:
                ax.set_xticks(particle_positions[particle_label_indices])
                ax.set_xticklabels(particle_labels[particle_label_indices])
            else:
                ax.set_xticks([])
            ax.set_xlabel('Particle')
            ax.set_ylabel(metric_label)
            ax.tick_params(axis='x', labelrotation=90)
        self.__format_particle_metric_axis(
            axis=ax,
            default_title=metric_label,
            title=title,
            title_fontsize=title_fontsize,
            x_axis_fontsize=x_axis_fontsize,
            y_axis_fontsize=y_axis_fontsize,
            x_tick_fontsize=x_tick_fontsize,
            y_tick_fontsize=y_tick_fontsize,
            y_tick_spacing=y_tick_spacing,
            font_family=font_family
        )
        fig.tight_layout()
        save_path = self.__resolve_numbered_plot_save_path(
            save_path=save_path,
            save_plots=plotting_parameters['save_plots'],
            save_plot_path=plotting_parameters['save_plot_path'],
            file_extension=plotting_parameters['file_extension'],
            default_title=metric_label,
            title=title
        )
        self.__finalize_figure(fig, save_path, dpi, show)
        return fig, ax

    def plot_particle_angle_tumble_analysis(
        self,
        particle_id: int,
        figsize: tuple[float, float] = (14, 6),
        track_color: str | Colormap = 'viridis',
        save_path: str | os.PathLike | None = None,
        dpi: int | _UseConfiguredPlotValue = _USE_CONFIGURED_PLOT_VALUE,
        show: bool = True,
        show_tumbles: (
            bool | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        show_smoothed_trajectory: (
            bool | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        tumble_display_mode: (
            str | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        tumble_color: (
            str | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        smoothed_trajectory_color: (
            str | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        particle_marker: (
            str | None | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        tumble_marker: (
            str | None | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        title: (
            str | None | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        title_fontsize: (
            float | None | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        x_axis_fontsize: (
            float | None | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        y_axis_fontsize: (
            float | None | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        x_tick_fontsize: (
            float | None | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        y_tick_fontsize: (
            float | None | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        font_family: (
            str | None | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        save_plots: (
            bool | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        save_plot_path: (
            str | os.PathLike | None | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        file_extension: (
            str | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        track_thickness: (
            float | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        smooth_trajectory_thickness: (
            float | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        segment_marker_thickness: (
            float | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        marker_fill_mode: (
            str | None | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE
    ) -> tuple:
        """
        Plot a colored trajectory and its cached tumble classifications.

        The trajectory panel uses every retained point from the full cached
        angle-tumble path, including points omitted from the finite-angle table.
        It can outline detected tumble points or recolor the exact observed
        track edges whose smoothed-path states are classified as tumbles. The
        angle panel uses the point-level table and shows the configured tumble
        threshold.

        Omitted configurable arguments use
        ``set_analysis_plotting_parameters``; explicit values override those
        settings for this call only.

        Args:
            track_color: Solid Matplotlib color, colormap name, or Colormap
                object for the observed track and angle points. A colormap
                colors values by elapsed time and adds a colorbar. Color-like
                names take precedence, so use ``plt.get_cmap('gray')`` for a
                gray elapsed-time gradient rather than a solid gray track.
            track_thickness: Observed-track line width in points.
            smooth_trajectory_thickness: Smoothed-path line width in points.
            segment_marker_thickness: Scale factor for tumble segment widths,
                marker areas, and marker-edge widths.
            marker_fill_mode: 'solid' fills configured markers; 'outline'
                draws only their colored outlines. None preserves the
                established marker appearance.
            particle_marker: Optional Matplotlib marker for each observed
                particle position, such as 'o', 's', '^', 'D', or 'x'. None
                draws no observed-position markers.
            tumble_marker: Matplotlib marker for detected tumbles when
                ``tumble_display_mode='markers'``, such as 'o', 's', '^',
                'D', or 'x'. None restores the circle default. It is unused in
                segment display mode.
            title: Primary figure title. None restores the generated title;
                an empty string removes it.
            title_fontsize: Primary-title font size in points.
            x_axis_fontsize: X-axis label font size in points.
            y_axis_fontsize: Y-axis label font size in points.
            x_tick_fontsize: X-axis tick-label font size in points.
            y_tick_fontsize: Y-axis tick-label font size in points.
            font_family: Typeface for all figure text. Portable families are
                'sans-serif', 'serif', 'monospace', 'cursive', and 'fantasy'.
                Bundled faces include 'DejaVu Sans', 'DejaVu Serif', and
                'DejaVu Sans Mono'; installed system-font names are accepted.
            save_plots: Automatically save when ``save_path`` is None.
            save_plot_path: Directory for automatic numbered filenames.
            file_extension: Automatic-save format: 'png' or 'tif'.
        """
        plotting_parameters = self.__resolve_analysis_plotting_parameters(
            function_tumble_marker_default='o',
            track_thickness=track_thickness,
            smooth_trajectory_thickness=smooth_trajectory_thickness,
            segment_marker_thickness=segment_marker_thickness,
            marker_fill_mode=marker_fill_mode,
            show_tumbles=show_tumbles,
            show_smoothed_trajectory=show_smoothed_trajectory,
            tumble_display_mode=tumble_display_mode,
            tumble_color=tumble_color,
            smoothed_trajectory_color=smoothed_trajectory_color,
            particle_marker=particle_marker,
            tumble_marker=tumble_marker,
            title=title,
            title_fontsize=title_fontsize,
            x_axis_fontsize=x_axis_fontsize,
            y_axis_fontsize=y_axis_fontsize,
            x_tick_fontsize=x_tick_fontsize,
            y_tick_fontsize=y_tick_fontsize,
            font_family=font_family,
            dpi=dpi,
            save_plots=save_plots,
            save_plot_path=save_plot_path,
            file_extension=file_extension
        )
        show_tumbles = plotting_parameters['show_tumbles']
        show_smoothed_trajectory = plotting_parameters[
            'show_smoothed_trajectory'
        ]
        tumble_display_mode = plotting_parameters['tumble_display_mode']
        tumble_color = plotting_parameters['tumble_color']
        smoothed_trajectory_color = plotting_parameters[
            'smoothed_trajectory_color'
        ]
        particle_marker = plotting_parameters['particle_marker']
        tumble_marker = plotting_parameters['tumble_marker']
        track_thickness = plotting_parameters['track_thickness']
        smooth_trajectory_thickness = plotting_parameters[
            'smooth_trajectory_thickness'
        ]
        segment_marker_thickness = plotting_parameters[
            'segment_marker_thickness'
        ]
        marker_fill_mode = plotting_parameters['marker_fill_mode']
        dpi = plotting_parameters['dpi']
        particle_id = self.__normalize_single_particle_id(particle_id)
        self.__validate_boolean_argument(show_tumbles, 'show_tumbles')
        self.__validate_boolean_argument(
            show_smoothed_trajectory, 'show_smoothed_trajectory'
        )
        tumble_display_mode = self.__normalize_event_display_mode(
            tumble_display_mode, 'tumble_display_mode'
        )
        self.__validate_plot_color(tumble_color, 'tumble_color')
        self.__validate_plot_color(
            smoothed_trajectory_color, 'smoothed_trajectory_color'
        )
        self.__validate_plot_marker(tumble_marker, 'tumble_marker')
        if particle_marker is not None:
            self.__validate_plot_marker(particle_marker, 'particle_marker')
        uniform_track_color, track_colormap = self.__resolve_track_color(
            track_color
        )
        if self._tumble_points_dataframe.empty:
            raise ValueError(
                'Run calculate_angle_tumble_statistics before plotting '
                'angle-based tumble analysis.'
            )
        particle_points = self._tumble_points_dataframe.loc[
            self._tumble_points_dataframe['particle'] == particle_id
        ].sort_values(
            by=['segment_id', 'frame'], kind='stable'
        ).reset_index(drop=True)
        if particle_points.empty:
            raise ValueError(
                f'No analyzable tumble points are available for particle '
                f'{particle_id}.'
            )
        path_segments = self._tumble_path_data.get(particle_id, [])
        if not path_segments:
            raise ValueError(
                'The complete angle-tumble path is unavailable; rerun '
                'calculate_angle_tumble_statistics before plotting this '
                'particle.'
            )

        distance_factor = float(
            self._tumble_analysis_metadata['distance_factor']
        )
        distance_unit = self.__plot_unit_label(
            self._tumble_analysis_metadata['distance_unit']
        )
        threshold_degrees = float(
            self._tumble_analysis_metadata[
                'tumble_threshold_angle_degrees'
            ]
        )
        smoothed_trajectory_label = self.__smoothing_trajectory_label(
            self._tumble_analysis_metadata
        )
        color_values = np.concatenate([
            np.asarray(
                path_segment['elapsed_time_seconds'], dtype=float
            )
            for path_segment in path_segments
        ])
        color_min = float(np.nanmin(color_values))
        color_max = float(np.nanmax(color_values))
        if color_max <= color_min:
            color_max = color_min + 1.0
        color_norm = Normalize(vmin=color_min, vmax=color_max)
        origin = np.asarray(
            path_segments[0]['raw_positions_pixels'][0], dtype=float
        )

        fig, (trajectory_ax, angle_ax) = plt.subplots(
            1, 2, figsize=figsize
        )
        colorbar_ax = (
            make_axes_locatable(angle_ax).append_axes(
                'right', size='4%', pad=0.05
            )
            if track_colormap is not None else None
        )
        raw_label_added = False
        smoothed_label_added = False
        tumble_trajectory_label_added = False
        tumble_angle_label_added = False
        for path_segment in path_segments:
            raw_positions = (
                path_segment['raw_positions_pixels'] - origin
            ) * distance_factor
            smoothed_positions = (
                path_segment['smoothed_positions_pixels'] - origin
            ) * distance_factor
            segment_times = np.asarray(
                path_segment['elapsed_time_seconds'], dtype=float
            )
            for row_index in range(len(raw_positions) - 1):
                trajectory_ax.plot(
                    raw_positions[row_index:row_index + 2, 0],
                    raw_positions[row_index:row_index + 2, 1],
                    color=self.__track_color_for_value(
                        uniform_track_color,
                        track_colormap,
                        color_norm,
                        segment_times[row_index]
                    ),
                    linewidth=track_thickness,
                    label=(
                        'Observed trajectory'
                        if not raw_label_added else None
                    )
                )
                raw_label_added = True
            self.__scatter_particle_positions(
                trajectory_ax,
                raw_positions,
                segment_times,
                uniform_track_color,
                track_colormap,
                color_norm,
                particle_marker,
                marker_fill_mode,
                label=(
                    'Observed trajectory'
                    if len(raw_positions) == 1 and not raw_label_added
                    else None
                )
            )
            if len(raw_positions) == 1 and particle_marker is not None:
                raw_label_added = True
            if show_smoothed_trajectory:
                trajectory_ax.plot(
                    smoothed_positions[:, 0],
                    smoothed_positions[:, 1],
                    color=smoothed_trajectory_color,
                    linestyle='--',
                    linewidth=smooth_trajectory_thickness,
                    label=(
                        smoothed_trajectory_label
                        if not smoothed_label_added else None
                    )
                )
                smoothed_label_added = True

        for _, segment_points in particle_points.groupby(
            'segment_id', sort=True
        ):
            smoothed_positions = segment_points[
                ['smoothed_position_x_pixels', 'smoothed_position_y_pixels']
            ].to_numpy(dtype=float)
            smoothed_positions = (
                smoothed_positions - origin
            ) * distance_factor
            segment_times = segment_points[
                'elapsed_time_seconds'
            ].to_numpy(dtype=float)

            tumble_mask = (
                segment_points['state'].astype(str).to_numpy() == 'tumble'
            )
            if (
                show_tumbles and tumble_display_mode == 'markers' and
                tumble_mask.any()
            ):
                trajectory_ax.scatter(
                    smoothed_positions[tumble_mask, 0],
                    smoothed_positions[tumble_mask, 1],
                    marker=tumble_marker,
                    **self.__event_marker_color_arguments(
                        tumble_marker,
                        tumble_color,
                        marker_fill_mode,
                        default_outline=True
                    ),
                    linewidths=1.8 * segment_marker_thickness,
                    s=80 * segment_marker_thickness,
                    zorder=6,
                    label=(
                        'Detected tumbles'
                        if not tumble_trajectory_label_added else None
                    )
                )
                tumble_trajectory_label_added = True

            angle_values = segment_points[
                'turn_angle_degrees'
            ].to_numpy(dtype=float)
            finite_angles = np.isfinite(angle_values)
            if finite_angles.any():
                angle_ax.plot(
                    segment_times[finite_angles],
                    angle_values[finite_angles],
                    color='gray', alpha=0.5, linewidth=1
                )
                if track_colormap is None:
                    angle_ax.scatter(
                        segment_times[finite_angles],
                        angle_values[finite_angles],
                        color=uniform_track_color, s=24
                    )
                else:
                    angle_ax.scatter(
                        segment_times[finite_angles],
                        angle_values[finite_angles],
                        c=segment_times[finite_angles],
                        cmap=track_colormap, norm=color_norm, s=24
                    )
                tumble_angle_mask = tumble_mask & finite_angles
                if (
                    show_tumbles and tumble_display_mode == 'markers' and
                    tumble_angle_mask.any()
                ):
                    angle_ax.scatter(
                        segment_times[tumble_angle_mask],
                        angle_values[tumble_angle_mask],
                        marker=tumble_marker,
                        **self.__event_marker_color_arguments(
                            tumble_marker,
                            tumble_color,
                            marker_fill_mode,
                            default_outline=True
                        ),
                        linewidths=1.8 * segment_marker_thickness,
                        s=80 * segment_marker_thickness,
                        zorder=6,
                        label=(
                            'Detected tumbles'
                            if not tumble_angle_label_added else None
                        )
                    )
                    tumble_angle_label_added = True

        if show_tumbles and tumble_display_mode == 'segments':
            for path_segment in path_segments:
                raw_positions = (
                    path_segment['raw_positions_pixels'] - origin
                ) * distance_factor
                states = np.asarray(path_segment['states'], dtype=str)
                outgoing_tumble_edges = states[:-1] == 'tumble'
                tumble_trajectory_label_added = (
                    self.__add_highlighted_track_segments(
                        trajectory_ax,
                        raw_positions,
                        outgoing_tumble_edges,
                        color=tumble_color,
                        label=(
                            'Detected tumble segments'
                            if not tumble_trajectory_label_added else None
                        ),
                        linewidth=2.0 * segment_marker_thickness
                    ) or tumble_trajectory_label_added
                )

        if track_colormap is not None:
            scalar_map = plt.cm.ScalarMappable(
                norm=color_norm, cmap=track_colormap
            )
            scalar_map.set_array([])
            fig.colorbar(
                scalar_map,
                cax=colorbar_ax,
                label='Observed elapsed time (s)'
            )
        trajectory_ax.set_xlabel(f'X displacement ({distance_unit})')
        trajectory_ax.set_ylabel(f'Y displacement ({distance_unit})')
        trajectory_ax.set_title(f'Particle {particle_id} trajectory')
        trajectory_ax.set_aspect('equal', adjustable='datalim')
        if trajectory_ax.get_legend_handles_labels()[0]:
            trajectory_ax.legend()
        angle_ax.axhline(
            threshold_degrees,
            color=tumble_color,
            linestyle='--',
            linewidth=1.5,
            label=f'Tumble threshold ({threshold_degrees:g}°)'
        )
        angle_ax.set_xlabel('Observed elapsed time (s)')
        angle_ax.set_ylabel('Direction change (degrees)')
        angle_ax.set_ylim(0, 180)
        angle_ax.set_title('Tumble classification along trajectory')
        angle_ax.legend()
        default_title = (
            'Angle-based tumble analysis - '
            f'{self._resolved_source_dataframe} tracks'
        )
        self.__format_analysis_figure(
            figure=fig,
            default_title=default_title,
            title=plotting_parameters['title'],
            primary_title_axis=None,
            title_fontsize=plotting_parameters['title_fontsize'],
            x_axis_fontsize=plotting_parameters['x_axis_fontsize'],
            y_axis_fontsize=plotting_parameters['y_axis_fontsize'],
            x_tick_fontsize=plotting_parameters['x_tick_fontsize'],
            y_tick_fontsize=plotting_parameters['y_tick_fontsize'],
            font_family=plotting_parameters['font_family']
        )
        save_path = self.__resolve_numbered_plot_save_path(
            save_path=save_path,
            save_plots=plotting_parameters['save_plots'],
            save_plot_path=plotting_parameters['save_plot_path'],
            file_extension=plotting_parameters['file_extension'],
            default_title=default_title,
            title=plotting_parameters['title']
        )
        fig.subplots_adjust(top=0.88, wspace=0.3)
        self.__finalize_figure(fig, save_path, dpi, show)
        return fig, (trajectory_ax, angle_ax)

    def plot_particle_velocity_tumble_analysis(
        self,
        particle_id: int,
        figsize: tuple[float, float] = (18, 6),
        track_color: str | Colormap = 'viridis',
        save_path: str | os.PathLike | None = None,
        dpi: int | _UseConfiguredPlotValue = _USE_CONFIGURED_PLOT_VALUE,
        show: bool = True,
        show_tumbles: (
            bool | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        show_smoothed_trajectory: (
            bool | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        tumble_display_mode: (
            str | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        tumble_color: (
            str | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        smoothed_trajectory_color: (
            str | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        particle_marker: (
            str | None | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        tumble_marker: (
            str | None | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        title: (
            str | None | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        title_fontsize: (
            float | None | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        x_axis_fontsize: (
            float | None | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        y_axis_fontsize: (
            float | None | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        x_tick_fontsize: (
            float | None | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        y_tick_fontsize: (
            float | None | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        font_family: (
            str | None | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        save_plots: (
            bool | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        save_plot_path: (
            str | os.PathLike | None | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        file_extension: (
            str | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        track_thickness: (
            float | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        smooth_trajectory_thickness: (
            float | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        segment_marker_thickness: (
            float | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        marker_fill_mode: (
            str | None | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE
    ) -> tuple:
        """
        Plot trajectory, speed, and angular speed for velocity tumbles.

        Tumbles can be drawn as the original x markers or by recoloring the
        observed-track edges whose ending points are classified as tumbles.

        Omitted configurable arguments use
        ``set_analysis_plotting_parameters``; explicit values override those
        settings for this call only.

        Args:
            track_color: Solid Matplotlib color, colormap name, or Colormap
                object for the observed track. A colormap colors positions by
                elapsed time and adds a colorbar. Color-like names take
                precedence, so use ``plt.get_cmap('gray')`` for a gray
                elapsed-time gradient rather than a solid gray track.
            track_thickness: Observed-track line width in points.
            smooth_trajectory_thickness: Smoothed-path line width in points.
            segment_marker_thickness: Scale factor for tumble segment widths,
                marker areas, and marker-edge widths.
            marker_fill_mode: 'solid' fills configured markers; 'outline'
                draws only their colored outlines. None preserves the
                established marker appearance.
            particle_marker: Optional Matplotlib marker for each observed
                particle position, such as 'o', 's', '^', 'D', or 'x'. None
                draws no observed-position markers.
            tumble_marker: Matplotlib marker for velocity-based tumbles when
                ``tumble_display_mode='markers'``, such as 'o', 's', '^',
                'D', or 'x'. None restores the x default. It is unused in
                segment display mode.
            title: Primary figure title. None restores the generated title;
                an empty string removes it.
            title_fontsize: Primary-title font size in points.
            x_axis_fontsize: X-axis label font size in points.
            y_axis_fontsize: Y-axis label font size in points.
            x_tick_fontsize: X-axis tick-label font size in points.
            y_tick_fontsize: Y-axis tick-label font size in points.
            font_family: Typeface for all figure text. Portable families are
                'sans-serif', 'serif', 'monospace', 'cursive', and 'fantasy'.
                Bundled faces include 'DejaVu Sans', 'DejaVu Serif', and
                'DejaVu Sans Mono'; installed system-font names are accepted.
            save_plots: Automatically save when ``save_path`` is None.
            save_plot_path: Directory for automatic numbered filenames.
            file_extension: Automatic-save format: 'png' or 'tif'.
        """
        plotting_parameters = self.__resolve_analysis_plotting_parameters(
            function_tumble_marker_default='x',
            track_thickness=track_thickness,
            smooth_trajectory_thickness=smooth_trajectory_thickness,
            segment_marker_thickness=segment_marker_thickness,
            marker_fill_mode=marker_fill_mode,
            show_tumbles=show_tumbles,
            show_smoothed_trajectory=show_smoothed_trajectory,
            tumble_display_mode=tumble_display_mode,
            tumble_color=tumble_color,
            smoothed_trajectory_color=smoothed_trajectory_color,
            particle_marker=particle_marker,
            tumble_marker=tumble_marker,
            title=title,
            title_fontsize=title_fontsize,
            x_axis_fontsize=x_axis_fontsize,
            y_axis_fontsize=y_axis_fontsize,
            x_tick_fontsize=x_tick_fontsize,
            y_tick_fontsize=y_tick_fontsize,
            font_family=font_family,
            dpi=dpi,
            save_plots=save_plots,
            save_plot_path=save_plot_path,
            file_extension=file_extension
        )
        show_tumbles = plotting_parameters['show_tumbles']
        show_smoothed_trajectory = plotting_parameters[
            'show_smoothed_trajectory'
        ]
        tumble_display_mode = plotting_parameters['tumble_display_mode']
        tumble_color = plotting_parameters['tumble_color']
        smoothed_trajectory_color = plotting_parameters[
            'smoothed_trajectory_color'
        ]
        particle_marker = plotting_parameters['particle_marker']
        tumble_marker = plotting_parameters['tumble_marker']
        track_thickness = plotting_parameters['track_thickness']
        smooth_trajectory_thickness = plotting_parameters[
            'smooth_trajectory_thickness'
        ]
        segment_marker_thickness = plotting_parameters[
            'segment_marker_thickness'
        ]
        marker_fill_mode = plotting_parameters['marker_fill_mode']
        dpi = plotting_parameters['dpi']
        particle_id = self.__normalize_single_particle_id(particle_id)
        self.__validate_boolean_argument(show_tumbles, 'show_tumbles')
        self.__validate_boolean_argument(
            show_smoothed_trajectory, 'show_smoothed_trajectory'
        )
        tumble_display_mode = self.__normalize_event_display_mode(
            tumble_display_mode, 'tumble_display_mode'
        )
        self.__validate_plot_color(tumble_color, 'tumble_color')
        self.__validate_plot_color(
            smoothed_trajectory_color, 'smoothed_trajectory_color'
        )
        self.__validate_plot_marker(tumble_marker, 'tumble_marker')
        if particle_marker is not None:
            self.__validate_plot_marker(particle_marker, 'particle_marker')
        uniform_track_color, track_colormap = self.__resolve_track_color(
            track_color
        )
        if self._velocity_tumble_points_dataframe.empty:
            raise ValueError(
                'Run calculate_velocity_tumble_statistics before plotting '
                'velocity-based tumble analysis.'
            )
        particle_points = self._velocity_tumble_points_dataframe.loc[
            self._velocity_tumble_points_dataframe['particle'] == particle_id
        ].sort_values(
            by=['segment_id', 'frame'], kind='stable'
        ).reset_index(drop=True)
        if particle_points.empty:
            raise ValueError(
                f'No velocity-tumble points are available for particle '
                f'{particle_id}.'
            )
        particle_events = self._velocity_tumble_events_dataframe.loc[
            self._velocity_tumble_events_dataframe['particle'] == particle_id
        ].sort_values(
            by=['segment_id', 'start_frame'], kind='stable'
        )

        distance_factor = float(
            self._velocity_tumble_analysis_metadata['distance_factor']
        )
        distance_unit = self.__plot_unit_label(
            self._velocity_tumble_analysis_metadata['distance_unit']
        )
        speed_unit = self.__plot_unit_label(
            self._velocity_tumble_analysis_metadata['speed_unit']
        )
        smoothed_trajectory_label = self.__smoothing_trajectory_label(
            self._velocity_tumble_analysis_metadata
        )
        color_values = pd.to_numeric(
            particle_points['elapsed_time_seconds'], errors='coerce'
        ).to_numpy(dtype=float)
        finite_color_values = color_values[np.isfinite(color_values)]
        if not len(finite_color_values):
            raise ValueError(
                f'No finite elapsed times are available for particle '
                f'{particle_id}.'
            )
        color_min = float(np.min(finite_color_values))
        color_max = float(np.max(finite_color_values))
        if color_max <= color_min:
            color_max = color_min + 1.0
        color_norm = Normalize(vmin=color_min, vmax=color_max)
        origin = particle_points[[
            'raw_position_x_pixels', 'raw_position_y_pixels'
        ]].iloc[0].to_numpy(dtype=float)

        fig, (
            trajectory_ax, speed_ax, angular_velocity_ax
        ) = plt.subplots(1, 3, figsize=figsize)
        colorbar_ax = (
            make_axes_locatable(angular_velocity_ax).append_axes(
                'right', size='4%', pad=0.05
            )
            if track_colormap is not None else None
        )
        raw_label_added = False
        smoothed_label_added = False
        tumble_trajectory_label_added = False
        speed_minimum_label_added = False
        angular_maximum_label_added = False
        tumble_interval_label_added = False

        for segment_id, segment_points in particle_points.groupby(
            'segment_id', sort=True
        ):
            raw_positions = segment_points[[
                'raw_position_x_pixels', 'raw_position_y_pixels'
            ]].to_numpy(dtype=float)
            raw_positions = (raw_positions - origin) * distance_factor
            smoothed_positions = segment_points[[
                'smoothed_position_x_pixels',
                'smoothed_position_y_pixels'
            ]].to_numpy(dtype=float)
            smoothed_positions = (
                smoothed_positions - origin
            ) * distance_factor
            segment_times = pd.to_numeric(
                segment_points['elapsed_time_seconds'], errors='coerce'
            ).to_numpy(dtype=float)
            for point_index in range(len(raw_positions) - 1):
                trajectory_ax.plot(
                    raw_positions[point_index:point_index + 2, 0],
                    raw_positions[point_index:point_index + 2, 1],
                    color=self.__track_color_for_value(
                        uniform_track_color,
                        track_colormap,
                        color_norm,
                        segment_times[point_index]
                    ),
                    linewidth=track_thickness,
                    label=(
                        'Observed trajectory'
                        if not raw_label_added else None
                    )
                )
                raw_label_added = True
            self.__scatter_particle_positions(
                trajectory_ax,
                raw_positions,
                segment_times,
                uniform_track_color,
                track_colormap,
                color_norm,
                particle_marker,
                marker_fill_mode,
                label=(
                    'Observed trajectory'
                    if len(raw_positions) == 1 and not raw_label_added
                    else None
                )
            )
            if len(raw_positions) == 1 and particle_marker is not None:
                raw_label_added = True
            if show_smoothed_trajectory:
                trajectory_ax.plot(
                    smoothed_positions[:, 0], smoothed_positions[:, 1],
                    color=smoothed_trajectory_color,
                    linestyle='--',
                    linewidth=smooth_trajectory_thickness,
                    label=(
                        smoothed_trajectory_label
                        if not smoothed_label_added else None
                    )
                )
                smoothed_label_added = True

            tumble_mask = (
                segment_points['state'].astype(str).to_numpy() == 'tumble'
            )
            if (
                show_tumbles and tumble_display_mode == 'markers' and
                tumble_mask.any()
            ):
                trajectory_ax.scatter(
                    smoothed_positions[tumble_mask, 0],
                    smoothed_positions[tumble_mask, 1],
                    marker=tumble_marker,
                    **self.__event_marker_color_arguments(
                        tumble_marker,
                        tumble_color,
                        marker_fill_mode,
                        default_outline=False
                    ),
                    linewidths=2.0 * segment_marker_thickness,
                    s=75 * segment_marker_thickness,
                    zorder=7,
                    label=(
                        'Velocity-based tumble points'
                        if not tumble_trajectory_label_added else None
                    )
                )
                tumble_trajectory_label_added = True
            elif show_tumbles and tumble_display_mode == 'segments':
                ending_tumble_edges = tumble_mask[1:]
                tumble_trajectory_label_added = (
                    self.__add_highlighted_track_segments(
                        trajectory_ax,
                        raw_positions,
                        ending_tumble_edges,
                        color=tumble_color,
                        label=(
                            'Velocity-based tumble segments'
                            if not tumble_trajectory_label_added else None
                        ),
                        linewidth=2.0 * segment_marker_thickness
                    ) or tumble_trajectory_label_added
                )

            speed_values = pd.to_numeric(
                segment_points['speed'], errors='coerce'
            ).to_numpy(dtype=float)
            finite_speeds = np.isfinite(speed_values)
            speed_ax.plot(
                segment_times[finite_speeds],
                speed_values[finite_speeds],
                color='#0072B2', linewidth=1.5
            )
            passing_speed_minima = (
                segment_points[
                    'speed_minimum_passes'
                ].fillna(False).to_numpy(dtype=bool) &
                finite_speeds
            )
            if passing_speed_minima.any():
                speed_ax.scatter(
                    segment_times[passing_speed_minima],
                    speed_values[passing_speed_minima],
                    color='#D55E00', marker='v', s=55, zorder=6,
                    label=(
                        'Qualifying speed minimum'
                        if not speed_minimum_label_added else None
                    )
                )
                speed_minimum_label_added = True

            angular_velocity_values = pd.to_numeric(
                segment_points[
                    'angular_velocity_magnitude_radians_per_second'
                ],
                errors='coerce'
            ).to_numpy(dtype=float)
            finite_angular_velocities = np.isfinite(
                angular_velocity_values
            )
            angular_velocity_ax.plot(
                segment_times[finite_angular_velocities],
                angular_velocity_values[finite_angular_velocities],
                color='#009E73', linewidth=1.5
            )
            passing_angular_maxima = (
                segment_points[
                    'angular_velocity_maximum_passes'
                ].fillna(False).to_numpy(dtype=bool) &
                finite_angular_velocities
            )
            if passing_angular_maxima.any():
                angular_velocity_ax.scatter(
                    segment_times[passing_angular_maxima],
                    angular_velocity_values[passing_angular_maxima],
                    color='#CC79A7', marker='^', s=55, zorder=6,
                    label=(
                        'Qualifying angular maximum'
                        if not angular_maximum_label_added else None
                    )
                )
                angular_maximum_label_added = True

            segment_events = particle_events.loc[
                particle_events['segment_id'] == segment_id
            ]
            segment_frames = pd.to_numeric(
                segment_points['frame'], errors='coerce'
            ).to_numpy(dtype=float)
            for _, event_row in (
                segment_events.iterrows() if show_tumbles else []
            ):
                start_time = float(np.interp(
                    float(event_row['start_frame']),
                    segment_frames, segment_times
                ))
                end_time = float(np.interp(
                    float(event_row['end_frame']),
                    segment_frames, segment_times
                ))
                interval_label = (
                    'Accepted tumble interval'
                    if not tumble_interval_label_added else None
                )
                speed_ax.axvspan(
                    start_time, end_time, color='#E69F00', alpha=0.18,
                    label=interval_label
                )
                angular_velocity_ax.axvspan(
                    start_time, end_time, color='#E69F00', alpha=0.18,
                    label=interval_label
                )
                tumble_interval_label_added = True

        if track_colormap is not None:
            scalar_map = plt.cm.ScalarMappable(
                norm=color_norm, cmap=track_colormap
            )
            scalar_map.set_array([])
            fig.colorbar(
                scalar_map, cax=colorbar_ax,
                label='Observed elapsed time (s)'
            )
        trajectory_ax.set_xlabel(f'X displacement ({distance_unit})')
        trajectory_ax.set_ylabel(f'Y displacement ({distance_unit})')
        trajectory_ax.set_title(f'Particle {particle_id} trajectory')
        trajectory_ax.set_aspect('equal', adjustable='datalim')
        speed_ax.set_xlabel('Observed elapsed time (s)')
        speed_ax.set_ylabel(f'Speed ({speed_unit})')
        speed_ax.set_title('Velocity signal, v(t)')
        angular_velocity_ax.set_xlabel('Observed elapsed time (s)')
        angular_velocity_ax.set_ylabel(
            'Angular-velocity magnitude (rad/s)'
        )
        angular_velocity_ax.set_title('Angular signal, |ω(t)|')
        for axis in (
            trajectory_ax, speed_ax, angular_velocity_ax
        ):
            if axis.get_legend_handles_labels()[0]:
                axis.legend()
        default_title = (
            'Velocity-based tumble analysis - '
            f'{self._resolved_source_dataframe} tracks'
        )
        self.__format_analysis_figure(
            figure=fig,
            default_title=default_title,
            title=plotting_parameters['title'],
            primary_title_axis=None,
            title_fontsize=plotting_parameters['title_fontsize'],
            x_axis_fontsize=plotting_parameters['x_axis_fontsize'],
            y_axis_fontsize=plotting_parameters['y_axis_fontsize'],
            x_tick_fontsize=plotting_parameters['x_tick_fontsize'],
            y_tick_fontsize=plotting_parameters['y_tick_fontsize'],
            font_family=plotting_parameters['font_family']
        )
        save_path = self.__resolve_numbered_plot_save_path(
            save_path=save_path,
            save_plots=plotting_parameters['save_plots'],
            save_plot_path=plotting_parameters['save_plot_path'],
            file_extension=plotting_parameters['file_extension'],
            default_title=default_title,
            title=plotting_parameters['title']
        )
        fig.subplots_adjust(top=0.84, wspace=0.35)
        self.__finalize_figure(fig, save_path, dpi, show)
        return fig, (
            trajectory_ax, speed_ax, angular_velocity_ax
        )

    def __plot_particle_turn_tumble_analysis(
        self,
        particle_id: int,
        tumble_method: str,
        plotting_parameters: dict,
        figsize: tuple[float, float] = (10, 8),
        track_color: str | Colormap = 'viridis',
        save_path: str | None = None,
        dpi: int = 300,
        show: bool = True,
        show_turns: bool = True,
        show_tumbles: bool = True,
        show_smoothed_turn_trajectory: bool = True,
        show_tumble_smoothed_trajectory: bool = False,
        turn_display_mode: str = 'markers',
        tumble_display_mode: str = 'markers',
        turn_color: str = 'darkviolet',
        tumble_color: str = '#D55E00',
        smoothed_turn_trajectory_color: str = 'lightblue',
        tumble_smoothed_trajectory_color: str = 'goldenrod',
        particle_marker: str | None = None,
        turn_marker: str = 'o',
        tumble_marker: str = 'x'
    ) -> tuple:
        """
        Overlay cached turn detections and tumble classifications on one track.

        Turn statistics and the selected tumble method must both be run for the
        same Stats dataframe when both overlays are shown. Marker mode preserves
        the original circles and x's; segment mode recolors only matching edges
        of the displayed observed trajectory.
        """
        particle_id = self.__normalize_single_particle_id(particle_id)
        self.__validate_boolean_argument(show_turns, 'show_turns')
        self.__validate_boolean_argument(show_tumbles, 'show_tumbles')
        self.__validate_boolean_argument(
            show_smoothed_turn_trajectory,
            'show_smoothed_turn_trajectory'
        )
        self.__validate_boolean_argument(
            show_tumble_smoothed_trajectory,
            'show_tumble_smoothed_trajectory'
        )
        turn_display_mode = self.__normalize_event_display_mode(
            turn_display_mode, 'turn_display_mode'
        )
        tumble_display_mode = self.__normalize_event_display_mode(
            tumble_display_mode, 'tumble_display_mode'
        )
        self.__validate_plot_color(turn_color, 'turn_color')
        self.__validate_plot_color(tumble_color, 'tumble_color')
        self.__validate_plot_color(
            smoothed_turn_trajectory_color,
            'smoothed_turn_trajectory_color'
        )
        self.__validate_plot_color(
            tumble_smoothed_trajectory_color,
            'tumble_smoothed_trajectory_color'
        )
        self.__validate_plot_marker(turn_marker, 'turn_marker')
        self.__validate_plot_marker(tumble_marker, 'tumble_marker')
        if particle_marker is not None:
            self.__validate_plot_marker(particle_marker, 'particle_marker')
        track_thickness = plotting_parameters['track_thickness']
        smooth_trajectory_thickness = plotting_parameters[
            'smooth_trajectory_thickness'
        ]
        segment_marker_thickness = plotting_parameters[
            'segment_marker_thickness'
        ]
        crop_to_track = plotting_parameters['crop_to_track']
        marker_fill_mode = plotting_parameters['marker_fill_mode']
        uniform_track_color, track_colormap = self.__resolve_track_color(
            track_color
        )
        if tumble_method == 'angle':
            tumble_points_dataframe = self._tumble_points_dataframe
            tumble_metadata = self._tumble_analysis_metadata
            calculation_name = 'calculate_angle_tumble_statistics'
            tumble_method_label = 'angle-based'
        elif tumble_method == 'velocity':
            tumble_points_dataframe = (
                self._velocity_tumble_points_dataframe
            )
            tumble_metadata = self._velocity_tumble_analysis_metadata
            calculation_name = 'calculate_velocity_tumble_statistics'
            tumble_method_label = 'velocity-based'
        else:
            raise ValueError("tumble_method must be 'angle' or 'velocity'.")
        if particle_id not in self._turn_path_data:
            raise ValueError(
                'Run calculate_turn_statistics for the requested particle '
                'before plotting turns and tumbles together.'
            )
        path_segments = self._turn_path_data[particle_id]
        if not path_segments:
            raise ValueError(
                f'No turn-analysis trajectory is available for particle '
                f'{particle_id}.'
            )
        particle_tumble_points = pd.DataFrame(
            columns=tumble_points_dataframe.columns
        )
        angle_tumble_path_segments: list[dict] = []
        needs_tumble_results = (
            show_tumbles or show_tumble_smoothed_trajectory
        )
        if needs_tumble_results:
            if not tumble_metadata:
                raise ValueError(
                    f'Run {calculation_name} before plotting its combined '
                    'trajectory analysis.'
                )
            turn_source = str(
                self._turn_analysis_metadata.get('source_dataframe', '')
            )
            tumble_source = str(
                tumble_metadata.get('source_dataframe', '')
            )
            if turn_source != tumble_source:
                raise ValueError(
                    'Turn and tumble results must come from the same source '
                    'dataframe before they can be plotted together.'
                )

        needs_tumble_points = (
            show_tumbles or
            (
                tumble_method == 'velocity' and
                show_tumble_smoothed_trajectory
            )
        )
        if needs_tumble_points:
            if tumble_points_dataframe.empty:
                raise ValueError(
                    f'Run {calculation_name} before plotting turns and tumbles '
                    'together.'
                )
            particle_tumble_points = tumble_points_dataframe.loc[
                tumble_points_dataframe['particle'] == particle_id
            ].sort_values(
                by=['segment_id', 'frame'], kind='stable'
            ).reset_index(drop=True)
            if particle_tumble_points.empty:
                raise ValueError(
                    f'No analyzable tumble points are available for particle '
                    f'{particle_id}.'
                )

        if tumble_method == 'angle' and show_tumble_smoothed_trajectory:
            angle_tumble_path_segments = self._tumble_path_data.get(
                particle_id, []
            )
            if not angle_tumble_path_segments:
                raise ValueError(
                    'No angle-tumble smoothed trajectory is available for '
                    f'particle {particle_id}; rerun '
                    'calculate_angle_tumble_statistics for that particle.'
                )
        distance_factor = float(
            self._turn_analysis_metadata['distance_factor']
        )
        distance_unit = self.__plot_unit_label(
            self._turn_analysis_metadata['distance_unit']
        )
        smoothed_turn_trajectory_label = (
            'Turn analysis: ' + self.__smoothing_trajectory_label(
                self._turn_analysis_metadata
            )
        )
        tumble_smoothed_trajectory_label = (
            f'{tumble_method.capitalize()}-tumble analysis: ' +
            self.__smoothing_trajectory_label(tumble_metadata)
        )
        all_frames = np.concatenate([
            segment['frames'] for segment in path_segments
        ])
        first_frame = int(all_frames.min())
        if self.__has_valid_frame_rate():
            color_values = (
                (all_frames - first_frame) / self._capture_speed_in_fps
            )
            colorbar_label = 'Elapsed time (s)'
        else:
            color_values = all_frames - first_frame
            colorbar_label = 'Elapsed frames'
        color_min = float(np.min(color_values))
        color_max = float(np.max(color_values))
        if color_max <= color_min:
            color_max = color_min + 1.0
        color_norm = Normalize(vmin=color_min, vmax=color_max)
        origin = path_segments[0]['raw_positions_pixels'][0]

        fig, trajectory_ax = plt.subplots(figsize=figsize)
        colorbar_ax = (
            make_axes_locatable(trajectory_ax).append_axes(
                'right', size='4%', pad=0.05
            )
            if track_colormap is not None else None
        )
        raw_label_added = False
        smoothed_turn_label_added = False
        turn_label_added = False
        for segment in path_segments:
            frames = segment['frames']
            raw_positions = (
                segment['raw_positions_pixels'] - origin
            ) * distance_factor
            for row_index in range(len(raw_positions) - 1):
                color_value = (
                    (frames[row_index] - first_frame) /
                    self._capture_speed_in_fps
                    if self.__has_valid_frame_rate()
                    else frames[row_index] - first_frame
                )
                trajectory_ax.plot(
                    raw_positions[row_index:row_index + 2, 0],
                    raw_positions[row_index:row_index + 2, 1],
                    color=self.__track_color_for_value(
                        uniform_track_color,
                        track_colormap,
                        color_norm,
                        color_value
                    ),
                    linewidth=track_thickness,
                    label='Observed trajectory' if not raw_label_added else None
                )
                raw_label_added = True
            marker_color_values = (
                (frames - first_frame) / self._capture_speed_in_fps
                if self.__has_valid_frame_rate()
                else frames - first_frame
            )
            self.__scatter_particle_positions(
                trajectory_ax,
                raw_positions,
                marker_color_values,
                uniform_track_color,
                track_colormap,
                color_norm,
                particle_marker,
                marker_fill_mode,
                label=(
                    'Observed trajectory'
                    if len(raw_positions) == 1 and not raw_label_added
                    else None
                )
            )
            if len(raw_positions) == 1 and particle_marker is not None:
                raw_label_added = True

            processed_positions = (
                segment['processed_positions_pixels'] - origin
            ) * distance_factor
            if show_smoothed_turn_trajectory:
                trajectory_ax.plot(
                    processed_positions[:, 0], processed_positions[:, 1],
                    color=smoothed_turn_trajectory_color,
                    linestyle='--',
                    linewidth=smooth_trajectory_thickness,
                    label=(
                        smoothed_turn_trajectory_label
                        if not smoothed_turn_label_added else None
                    )
                )
                smoothed_turn_label_added = True
            turn_indices = segment['turn_vertex_indices']
            if (
                show_turns and turn_display_mode == 'markers' and
                len(turn_indices)
            ):
                trajectory_ax.scatter(
                    processed_positions[turn_indices, 0],
                    processed_positions[turn_indices, 1],
                    marker=turn_marker,
                    **self.__event_marker_color_arguments(
                        turn_marker,
                        turn_color,
                        marker_fill_mode,
                        default_outline=True
                    ),
                    linewidths=1.8 * segment_marker_thickness,
                    s=75 * segment_marker_thickness,
                    zorder=6,
                    label='Detected turns' if not turn_label_added else None
                )
                turn_label_added = True
            elif show_turns and len(turn_indices):
                turn_edge_mask = self.__raw_turn_edge_mask(
                    raw_point_count=len(raw_positions),
                    source_indices=segment['source_indices'],
                    turn_vertex_indices=turn_indices
                )
                turn_label_added = (
                    self.__add_highlighted_track_segments(
                        trajectory_ax,
                        raw_positions,
                        turn_edge_mask,
                        color=turn_color,
                        label=(
                            'Detected turn segments'
                            if not turn_label_added else None
                        ),
                        linewidth=2.0 * segment_marker_thickness
                    ) or turn_label_added
                )

        tumble_smoothed_label_added = False
        if show_tumble_smoothed_trajectory:
            if tumble_method == 'angle':
                tumble_smoothed_segments = [
                    np.asarray(
                        segment['smoothed_positions_pixels'], dtype=float
                    )
                    for segment in angle_tumble_path_segments
                ]
            else:
                tumble_smoothed_segments = [
                    segment_points[[
                        'smoothed_position_x_pixels',
                        'smoothed_position_y_pixels'
                    ]].to_numpy(dtype=float)
                    for _, segment_points in (
                        particle_tumble_points.groupby(
                            'segment_id', sort=True
                        )
                    )
                ]
            for smoothed_positions_pixels in tumble_smoothed_segments:
                if not len(smoothed_positions_pixels):
                    continue
                smoothed_positions = (
                    smoothed_positions_pixels - origin
                ) * distance_factor
                trajectory_ax.plot(
                    smoothed_positions[:, 0], smoothed_positions[:, 1],
                    color=tumble_smoothed_trajectory_color,
                    linestyle='-.',
                    linewidth=smooth_trajectory_thickness,
                    zorder=4,
                    label=(
                        tumble_smoothed_trajectory_label
                        if not tumble_smoothed_label_added else None
                    )
                )
                tumble_smoothed_label_added = True

        tumble_points = (
            particle_tumble_points.loc[
                particle_tumble_points['state'].astype(str) == 'tumble'
            ]
            if show_tumbles else pd.DataFrame()
        )
        if (
            show_tumbles and tumble_display_mode == 'markers' and
            not tumble_points.empty
        ):
            tumble_positions = tumble_points[
                ['smoothed_position_x_pixels', 'smoothed_position_y_pixels']
            ].to_numpy(dtype=float)
            tumble_positions = (
                tumble_positions - origin
            ) * distance_factor
            trajectory_ax.scatter(
                tumble_positions[:, 0], tumble_positions[:, 1],
                marker=tumble_marker,
                **self.__event_marker_color_arguments(
                    tumble_marker,
                    tumble_color,
                    marker_fill_mode,
                    default_outline=False
                ),
                linewidths=2.0 * segment_marker_thickness,
                s=75 * segment_marker_thickness,
                zorder=7,
                label='Detected tumble points'
            )
        elif show_tumbles and tumble_display_mode == 'segments':
            tumble_segment_label_added = False
            tumble_edge_frame_pairs: set[tuple[int, int]] = set()
            if tumble_method == 'angle':
                tumble_path_segments = self._tumble_path_data.get(
                    particle_id, []
                )
                if not tumble_path_segments:
                    raise ValueError(
                        'The complete angle-tumble path is unavailable; rerun '
                        'calculate_angle_tumble_statistics before plotting '
                        'tumble segments.'
                    )
                for tumble_path_segment in tumble_path_segments:
                    tumble_frames = np.asarray(
                        tumble_path_segment['frames'], dtype=int
                    )
                    tumble_states = np.asarray(
                        tumble_path_segment['states'], dtype=str
                    )
                    edge_mask = tumble_states[:-1] == 'tumble'
                    tumble_edge_frame_pairs.update(
                        (
                            int(tumble_frames[edge_index]),
                            int(tumble_frames[edge_index + 1])
                        )
                        for edge_index in np.flatnonzero(edge_mask)
                    )
            else:
                for _, tumble_segment_points in (
                    particle_tumble_points.groupby(
                        'segment_id', sort=True
                    )
                ):
                    tumble_frames = tumble_segment_points[
                        'frame'
                    ].to_numpy(dtype=int)
                    tumble_states = (
                        tumble_segment_points['state']
                        .astype(str)
                        .to_numpy()
                    )
                    edge_mask = tumble_states[1:] == 'tumble'
                    tumble_edge_frame_pairs.update(
                        (
                            int(tumble_frames[edge_index]),
                            int(tumble_frames[edge_index + 1])
                        )
                        for edge_index in np.flatnonzero(edge_mask)
                    )

            for segment in path_segments:
                base_frames = np.asarray(segment['frames'], dtype=int)
                base_positions = (
                    segment['raw_positions_pixels'] - origin
                ) * distance_factor
                edge_mask = np.asarray([
                    (
                        int(start_frame),
                        int(end_frame)
                    ) in tumble_edge_frame_pairs
                    for start_frame, end_frame in zip(
                        base_frames[:-1], base_frames[1:]
                    )
                ], dtype=bool)
                tumble_segment_label_added = (
                    self.__add_highlighted_track_segments(
                        trajectory_ax,
                        base_positions,
                        edge_mask,
                        color=tumble_color,
                        label=(
                            'Detected tumble segments'
                            if not tumble_segment_label_added else None
                        ),
                        linewidth=2.0 * segment_marker_thickness
                    ) or tumble_segment_label_added
                )

        if track_colormap is not None:
            scalar_map = plt.cm.ScalarMappable(
                norm=color_norm, cmap=track_colormap
            )
            scalar_map.set_array([])
            fig.colorbar(
                scalar_map, cax=colorbar_ax, label=colorbar_label
            )
        trajectory_ax.set_xlabel(f'X displacement ({distance_unit})')
        trajectory_ax.set_ylabel(f'Y displacement ({distance_unit})')
        default_title = self.__combined_analysis_default_title(
            particle_id=particle_id,
            tumble_method_label=tumble_method_label,
            show_turns=show_turns,
            show_tumbles=show_tumbles
        )
        self.__format_combined_trajectory_axis(
            trajectory_ax, crop_to_track=crop_to_track
        )
        if trajectory_ax.get_legend_handles_labels()[0]:
            trajectory_ax.legend()
        self.__format_analysis_figure(
            figure=fig,
            default_title=default_title,
            title=plotting_parameters['title'],
            primary_title_axis=trajectory_ax,
            title_fontsize=plotting_parameters['title_fontsize'],
            x_axis_fontsize=plotting_parameters['x_axis_fontsize'],
            y_axis_fontsize=plotting_parameters['y_axis_fontsize'],
            x_tick_fontsize=plotting_parameters['x_tick_fontsize'],
            y_tick_fontsize=plotting_parameters['y_tick_fontsize'],
            font_family=plotting_parameters['font_family']
        )
        fig.tight_layout()
        self.__finalize_figure(fig, save_path, dpi, show)
        return fig, trajectory_ax

    def __plot_particle_velocity_tumble_comparison(
        self,
        particle_id: int,
        plotting_parameters: dict,
        figsize: tuple[float, float],
        speed_colormap_name: str,
        track_color: str | Colormap,
        tumble_color: str,
        direction_arrow_color: str,
        show_tumbles: bool,
        show_direction_arrow: bool,
        show_axes: bool,
        save_path: str | None,
        dpi: int,
        show: bool,
        particle_marker: str | None = None
    ) -> tuple:
        """
        Create speed-gradient and tumble-segment trajectory panels.

        A uniform ``track_color`` preserves the paper-style gray trajectory.
        A colormap name or Colormap object instead colors that trajectory by
        observed elapsed time. Color-like names take precedence, so ``'gray'``
        remains a uniform color; use ``plt.get_cmap('gray')`` for a gray
        elapsed-time gradient.
        """
        particle_id = self.__normalize_single_particle_id(particle_id)
        figsize = self.__normalize_figsize(
            figsize, 'velocity_comparison_figsize'
        )
        self.__validate_boolean_argument(show_tumbles, 'show_tumbles')
        self.__validate_boolean_argument(
            show_direction_arrow, 'show_direction_arrow'
        )
        self.__validate_boolean_argument(
            show_axes, 'show_comparison_axes'
        )
        if particle_marker is not None:
            self.__validate_plot_marker(particle_marker, 'particle_marker')
        track_thickness = plotting_parameters['track_thickness']
        segment_marker_thickness = plotting_parameters[
            'segment_marker_thickness'
        ]
        marker_fill_mode = plotting_parameters['marker_fill_mode']
        uniform_track_color = None
        elapsed_time_colormap = None
        if isinstance(track_color, Colormap):
            elapsed_time_colormap = track_color
        elif is_color_like(track_color):
            uniform_track_color = track_color
        elif isinstance(track_color, str):
            try:
                elapsed_time_colormap = plt.get_cmap(track_color)
            except ValueError as error:
                raise ValueError(
                    'comparison_track_color must be a '
                    'Matplotlib-compatible color or colormap.'
                ) from error
        else:
            raise ValueError(
                'comparison_track_color must be a Matplotlib-compatible '
                'color or colormap.'
            )
        self.__validate_plot_color(
            tumble_color, 'comparison_tumble_color'
        )
        self.__validate_plot_color(
            direction_arrow_color, 'direction_arrow_color'
        )
        speed_color_map = plt.get_cmap(speed_colormap_name)
        if self._velocity_tumble_points_dataframe.empty:
            raise ValueError(
                'Run calculate_velocity_tumble_statistics before plotting '
                'the velocity-tumble trajectory comparison.'
            )
        particle_points = self._velocity_tumble_points_dataframe.loc[
            self._velocity_tumble_points_dataframe['particle'] == particle_id
        ].sort_values(
            by=['segment_id', 'frame'], kind='stable'
        ).reset_index(drop=True)
        if particle_points.empty:
            raise ValueError(
                f'No velocity-tumble points are available for particle '
                f'{particle_id}.'
            )

        distance_factor = float(
            self._velocity_tumble_analysis_metadata['distance_factor']
        )
        distance_unit = self.__plot_unit_label(
            self._velocity_tumble_analysis_metadata['distance_unit']
        )
        speed_unit = self.__plot_unit_label(
            self._velocity_tumble_analysis_metadata['speed_unit']
        )
        origin = particle_points[[
            'raw_position_x_pixels', 'raw_position_y_pixels'
        ]].iloc[0].to_numpy(dtype=float)

        segment_records = []
        finite_edge_speeds = []
        finite_edge_elapsed_times = []
        all_positions = []
        edge_count = 0
        for _, segment_points in particle_points.groupby(
            'segment_id', sort=True
        ):
            positions = segment_points[[
                'smoothed_position_x_pixels',
                'smoothed_position_y_pixels'
            ]].to_numpy(dtype=float)
            positions = (positions - origin) * distance_factor
            speeds = pd.to_numeric(
                segment_points['speed'], errors='coerce'
            ).to_numpy(dtype=float)
            elapsed_times = pd.to_numeric(
                segment_points['elapsed_time_seconds'], errors='coerce'
            ).to_numpy(dtype=float)
            states = segment_points['state'].astype(str).to_numpy()
            all_positions.append(positions)
            if len(positions) < 2:
                continue
            edges = np.stack([positions[:-1], positions[1:]], axis=1)
            ending_speeds = speeds[1:]
            ending_elapsed_times = elapsed_times[1:]
            finite_position_edges = np.isfinite(edges).all(axis=(1, 2))
            finite_edges = (
                finite_position_edges & np.isfinite(ending_speeds)
            )
            finite_elapsed_time_edges = (
                finite_position_edges &
                np.isfinite(ending_elapsed_times)
            )
            edge_count += len(edges)
            finite_edge_speeds.extend(ending_speeds[finite_edges])
            finite_edge_elapsed_times.extend(
                ending_elapsed_times[finite_elapsed_time_edges]
            )
            segment_records.append({
                'positions': positions,
                'edges': edges,
                'ending_speeds': ending_speeds,
                'ending_elapsed_times': ending_elapsed_times,
                'speeds': speeds,
                'elapsed_times': elapsed_times,
                'finite_position_edges': finite_position_edges,
                'finite_edges': finite_edges,
                'finite_elapsed_time_edges': finite_elapsed_time_edges,
                'states': states,
            })

        if edge_count == 0:
            raise ValueError(
                'The requested particle has no consecutive trajectory points '
                'for the velocity comparison.'
            )
        finite_edge_speeds = np.asarray(
            finite_edge_speeds, dtype=float
        )
        if not len(finite_edge_speeds):
            raise ValueError(
                'The requested particle has no finite interval speeds for the '
                'velocity comparison.'
            )
        speed_min = float(np.min(finite_edge_speeds))
        speed_max = float(np.max(finite_edge_speeds))
        if speed_max <= speed_min:
            normalization_padding = max(
                abs(speed_min) * 0.01, 1e-12
            )
            speed_min -= normalization_padding
            speed_max += normalization_padding
        speed_norm = Normalize(vmin=speed_min, vmax=speed_max)
        elapsed_time_norm = None
        if elapsed_time_colormap is not None:
            finite_edge_elapsed_times = np.asarray(
                finite_edge_elapsed_times, dtype=float
            )
            if not len(finite_edge_elapsed_times):
                raise ValueError(
                    'The requested particle has no finite elapsed times for '
                    'the comparison-track colormap.'
                )
            elapsed_time_min = float(np.min(finite_edge_elapsed_times))
            elapsed_time_max = float(np.max(finite_edge_elapsed_times))
            if elapsed_time_max <= elapsed_time_min:
                normalization_padding = max(
                    abs(elapsed_time_min) * 0.01, 1e-12
                )
                elapsed_time_min -= normalization_padding
                elapsed_time_max += normalization_padding
            elapsed_time_norm = Normalize(
                vmin=elapsed_time_min, vmax=elapsed_time_max
            )

        finite_positions = np.concatenate(all_positions, axis=0)
        finite_positions = finite_positions[
            np.isfinite(finite_positions).all(axis=1)
        ]
        if not len(finite_positions):
            raise ValueError(
                'The requested particle has no finite positions for the '
                'velocity comparison.'
            )
        x_min, y_min = np.min(finite_positions, axis=0)
        x_max, y_max = np.max(finite_positions, axis=0)
        plot_extent = float(max(x_max - x_min, y_max - y_min))

        comparison_fig = plt.figure(figsize=figsize)
        if elapsed_time_colormap is None:
            comparison_grid = comparison_fig.add_gridspec(
                1, 3, width_ratios=(0.045, 1, 1), wspace=0.08
            )
            elapsed_time_colorbar_ax = None
        else:
            comparison_grid = comparison_fig.add_gridspec(
                1, 4, width_ratios=(0.045, 1, 1, 0.045), wspace=0.08
            )
            elapsed_time_colorbar_ax = comparison_fig.add_subplot(
                comparison_grid[0, 3]
            )
        colorbar_ax = comparison_fig.add_subplot(comparison_grid[0, 0])
        speed_trajectory_ax = comparison_fig.add_subplot(
            comparison_grid[0, 1]
        )
        tumble_trajectory_ax = comparison_fig.add_subplot(
            comparison_grid[0, 2],
            sharex=speed_trajectory_ax,
            sharey=speed_trajectory_ax
        )
        base_track_label_added = False
        tumble_segment_label_added = False
        comparison_track_linewidth = track_thickness
        direction_paths = []
        for segment_record in segment_records:
            positions = segment_record['positions']
            edges = segment_record['edges']
            finite_edges = segment_record['finite_edges']
            if finite_edges.any():
                speed_collection = LineCollection(
                    edges[finite_edges],
                    cmap=speed_color_map,
                    norm=speed_norm,
                    linewidths=1.25 * track_thickness,
                    zorder=3
                )
                speed_collection.set_array(
                    segment_record['ending_speeds'][finite_edges]
                )
                speed_trajectory_ax.add_collection(speed_collection)

            if particle_marker is not None:
                finite_speed_points = (
                    np.isfinite(positions).all(axis=1) &
                    np.isfinite(segment_record['speeds'])
                )
                if finite_speed_points.any():
                    self.__scatter_particle_positions(
                        speed_trajectory_ax,
                        positions[finite_speed_points],
                        segment_record['speeds'][finite_speed_points],
                        None,
                        speed_color_map,
                        speed_norm,
                        particle_marker,
                        marker_fill_mode,
                        zorder=4
                    )

            if elapsed_time_colormap is None:
                finite_position_edges = segment_record[
                    'finite_position_edges'
                ]
                if finite_position_edges.any():
                    tumble_trajectory_ax.add_collection(LineCollection(
                        edges[finite_position_edges],
                        colors=[uniform_track_color],
                        linewidths=comparison_track_linewidth,
                        zorder=2
                    ))
                    if not base_track_label_added:
                        tumble_trajectory_ax.add_line(Line2D(
                            [], [],
                            color=uniform_track_color,
                            linewidth=comparison_track_linewidth,
                            label='Trajectory'
                        ))
                    base_track_label_added = True
                if particle_marker is not None:
                    finite_marker_points = np.isfinite(positions).all(axis=1)
                    if finite_marker_points.any():
                        self.__scatter_particle_positions(
                            tumble_trajectory_ax,
                            positions[finite_marker_points],
                            segment_record['elapsed_times'][
                                finite_marker_points
                            ],
                            uniform_track_color,
                            None,
                            speed_norm,
                            particle_marker,
                            marker_fill_mode,
                            zorder=4
                        )
            else:
                finite_elapsed_time_edges = segment_record[
                    'finite_elapsed_time_edges'
                ]
                if finite_elapsed_time_edges.any():
                    elapsed_time_collection = LineCollection(
                        edges[finite_elapsed_time_edges],
                        cmap=elapsed_time_colormap,
                        norm=elapsed_time_norm,
                        linewidths=comparison_track_linewidth,
                        zorder=2
                    )
                    elapsed_time_collection.set_array(
                        segment_record[
                            'ending_elapsed_times'
                        ][finite_elapsed_time_edges]
                    )
                    tumble_trajectory_ax.add_collection(
                        elapsed_time_collection
                    )
                    if not base_track_label_added:
                        tumble_trajectory_ax.add_line(Line2D(
                            [], [],
                            color=elapsed_time_colormap(0.5),
                            linewidth=comparison_track_linewidth,
                            label='Trajectory (elapsed time)'
                        ))
                    base_track_label_added = True
                if particle_marker is not None:
                    finite_marker_points = (
                        np.isfinite(positions).all(axis=1) &
                        np.isfinite(segment_record['elapsed_times'])
                    )
                    if finite_marker_points.any():
                        self.__scatter_particle_positions(
                            tumble_trajectory_ax,
                            positions[finite_marker_points],
                            segment_record['elapsed_times'][
                                finite_marker_points
                            ],
                            None,
                            elapsed_time_colormap,
                            elapsed_time_norm,
                            particle_marker,
                            marker_fill_mode,
                            zorder=4
                        )
            if show_tumbles:
                ending_tumble_edges = (
                    segment_record['states'][1:] == 'tumble'
                )
                displayed_base_edges = (
                    segment_record['finite_position_edges']
                    if elapsed_time_colormap is None
                    else segment_record['finite_elapsed_time_edges']
                )
                ending_tumble_edges &= displayed_base_edges
                tumble_segment_label_added = (
                    self.__add_highlighted_track_segments(
                        tumble_trajectory_ax,
                        positions,
                        ending_tumble_edges,
                        color=tumble_color,
                        label=(
                            'Velocity-based tumble segments'
                            if not tumble_segment_label_added else None
                        ),
                        linewidth=2.0 * segment_marker_thickness,
                        zorder=6
                    ) or tumble_segment_label_added
                )

            finite_position_mask = np.isfinite(positions).all(axis=1)
            for start_index, end_index in self.__contiguous_true_ranges(
                finite_position_mask
            ):
                direction_path = positions[start_index:end_index + 1]
                if len(direction_path) < 2:
                    continue
                direction_path_length = float(np.linalg.norm(
                    np.diff(direction_path, axis=0), axis=1
                ).sum())
                if direction_path_length > np.finfo(float).eps:
                    direction_paths.append(direction_path)

        if show_direction_arrow:
            if direction_paths:
                direction_path = direction_paths[0]
                direction_edge_lengths = np.linalg.norm(
                    np.diff(direction_path, axis=0), axis=1
                )
                cumulative_lengths = np.concatenate([
                    [0.0], np.cumsum(direction_edge_lengths)
                ])
                total_direction_path_length = float(cumulative_lengths[-1])
                target_arrow_length = min(
                    max(plot_extent * 0.08, np.finfo(float).eps),
                    total_direction_path_length * 0.25
                )
                arrow_start_distance = 0.0
                arrow_end_distance = target_arrow_length

                def interpolate_path_position(distance: float) -> np.ndarray:
                    edge_index = int(np.searchsorted(
                        cumulative_lengths, distance, side='right'
                    ) - 1)
                    edge_index = min(
                        max(edge_index, 0), len(direction_path) - 2
                    )
                    edge_length = direction_edge_lengths[edge_index]
                    if edge_length <= np.finfo(float).eps:
                        return direction_path[edge_index].copy()
                    edge_fraction = (
                        distance - cumulative_lengths[edge_index]
                    ) / edge_length
                    return (
                        direction_path[edge_index] +
                        edge_fraction * (
                            direction_path[edge_index + 1] -
                            direction_path[edge_index]
                        )
                    )

                arrow_start = interpolate_path_position(
                    arrow_start_distance
                )
                arrow_end = interpolate_path_position(arrow_end_distance)
                arrow_vector = arrow_end - arrow_start
                arrow_length = float(np.linalg.norm(arrow_vector))
                if arrow_length > np.finfo(float).eps:
                    arrow_offset = np.array(
                        [-arrow_vector[1], arrow_vector[0]], dtype=float
                    ) / arrow_length
                    arrow_offset *= max(
                        plot_extent * 0.03, arrow_length * 0.2
                    )
                    tumble_trajectory_ax.annotate(
                        '',
                        xy=arrow_end + arrow_offset,
                        xytext=arrow_start + arrow_offset,
                        arrowprops={
                            'arrowstyle': '-|>',
                            'color': direction_arrow_color,
                            'linewidth': 1.8,
                            'mutation_scale': 14,
                        },
                        annotation_clip=False,
                        zorder=8
                    )
            else:
                warnings.warn(
                    'No nonzero trajectory edge is available for the '
                    'direction arrow.',
                    UserWarning,
                    stacklevel=2
                )

        scalar_map = plt.cm.ScalarMappable(
            norm=speed_norm, cmap=speed_color_map
        )
        scalar_map.set_array([])
        speed_colorbar = comparison_fig.colorbar(
            scalar_map,
            cax=colorbar_ax,
            label=f'Speed ({speed_unit})',
        )
        speed_colorbar.ax.yaxis.set_ticks_position('left')
        speed_colorbar.ax.yaxis.set_label_position('left')
        if elapsed_time_colormap is not None:
            elapsed_time_scalar_map = plt.cm.ScalarMappable(
                norm=elapsed_time_norm, cmap=elapsed_time_colormap
            )
            elapsed_time_scalar_map.set_array([])
            comparison_fig.colorbar(
                elapsed_time_scalar_map,
                cax=elapsed_time_colorbar_ax,
                label='Observed elapsed time (s)'
            )

        x_padding = max((x_max - x_min) * 0.05, 1e-9)
        y_padding = max((y_max - y_min) * 0.05, 1e-9)
        speed_trajectory_ax.set_xlim(
            float(x_min - x_padding), float(x_max + x_padding)
        )
        speed_trajectory_ax.set_ylim(
            float(y_min - y_padding), float(y_max + y_padding)
        )
        for axis in (speed_trajectory_ax, tumble_trajectory_ax):
            axis.set_aspect('equal', adjustable='box')
            if show_axes:
                axis.set_xlabel(f'X displacement ({distance_unit})')
                axis.set_ylabel(f'Y displacement ({distance_unit})')
            else:
                axis.set_axis_off()
        speed_trajectory_ax.text(
            0.02, 0.98, 'A',
            transform=speed_trajectory_ax.transAxes,
            ha='left', va='top', fontsize=14, fontweight='bold'
        )
        tumble_trajectory_ax.text(
            0.02, 0.98, 'B',
            transform=tumble_trajectory_ax.transAxes,
            ha='left', va='top', fontsize=14, fontweight='bold'
        )
        if show_axes:
            speed_trajectory_ax.set_title('Trajectory colored by speed')
            tumble_trajectory_ax.set_title(
                'Trajectory with velocity-based tumbles'
            )
            if tumble_trajectory_ax.get_legend_handles_labels()[0]:
                tumble_trajectory_ax.legend()
            comparison_default_title = (
                f'Particle {particle_id} '
                'velocity-tumble trajectory comparison'
            )
            self.__format_analysis_figure(
                figure=comparison_fig,
                default_title=comparison_default_title,
                title=None,
                primary_title_axis=None,
                title_fontsize=plotting_parameters['title_fontsize'],
                x_axis_fontsize=plotting_parameters['x_axis_fontsize'],
                y_axis_fontsize=plotting_parameters['y_axis_fontsize'],
                x_tick_fontsize=plotting_parameters['x_tick_fontsize'],
                y_tick_fontsize=plotting_parameters['y_tick_fontsize'],
                font_family=plotting_parameters['font_family']
            )
            comparison_fig.tight_layout(rect=(0, 0, 1, 0.95))
        else:
            self.__format_analysis_figure(
                figure=comparison_fig,
                default_title='',
                title='',
                primary_title_axis=None,
                title_fontsize=plotting_parameters['title_fontsize'],
                x_axis_fontsize=plotting_parameters['x_axis_fontsize'],
                y_axis_fontsize=plotting_parameters['y_axis_fontsize'],
                x_tick_fontsize=plotting_parameters['x_tick_fontsize'],
                y_tick_fontsize=plotting_parameters['y_tick_fontsize'],
                font_family=plotting_parameters['font_family']
            )
            comparison_fig.tight_layout(pad=0.4)
        self.__finalize_figure(
            comparison_fig, save_path, dpi, show
        )
        return comparison_fig, (
            speed_trajectory_ax, tumble_trajectory_ax
        )

    def plot_particle_turn_angle_tumble_analysis(
        self,
        particle_id: int,
        figsize: tuple[float, float] = (10, 8),
        track_color: str | Colormap = 'viridis',
        save_path: str | os.PathLike | None = None,
        dpi: int | _UseConfiguredPlotValue = _USE_CONFIGURED_PLOT_VALUE,
        show: bool = True,
        show_turns: (
            bool | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        show_tumbles: (
            bool | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        show_smoothed_turn_trajectory: (
            bool | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        turn_display_mode: (
            str | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        tumble_display_mode: (
            str | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        turn_color: (
            str | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        tumble_color: (
            str | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        smoothed_turn_trajectory_color: (
            str | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        show_angle_smoothed_trajectory: (
            bool | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        smoothed_angle_trajectory_color: (
            str | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        particle_marker: (
            str | None | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        turn_marker: (
            str | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        tumble_marker: (
            str | None | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        title: (
            str | None | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        title_fontsize: (
            float | None | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        x_axis_fontsize: (
            float | None | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        y_axis_fontsize: (
            float | None | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        x_tick_fontsize: (
            float | None | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        y_tick_fontsize: (
            float | None | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        font_family: (
            str | None | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        save_plots: (
            bool | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        save_plot_path: (
            str | os.PathLike | None | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        file_extension: (
            str | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        track_thickness: (
            float | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        smooth_trajectory_thickness: (
            float | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        segment_marker_thickness: (
            float | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        crop_to_track: (
            bool | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        marker_fill_mode: (
            str | None | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE
    ) -> tuple:
        """
        Overlay independently styled turns and angle-based tumbles.

        The processed turn path and frame-aligned angle-tumble smoothed path
        can be displayed and colored independently when their analyses used
        different smoothing algorithms.

        Omitted configurable arguments use
        ``set_analysis_plotting_parameters``; explicit values override those
        settings for this call only.

        Args:
            track_color: Solid Matplotlib color, colormap name, or Colormap
                object for the observed track. A colormap colors positions by
                elapsed time and adds a colorbar. Color-like names take
                precedence, so use ``plt.get_cmap('gray')`` for a gray
                elapsed-time gradient rather than a solid gray track.
            track_thickness: Observed-track line width in points.
            smooth_trajectory_thickness: Line width in points for both optional
                smoothed trajectories.
            segment_marker_thickness: Scale factor for turn/tumble segment
                widths, marker areas, and marker-edge widths.
            marker_fill_mode: 'solid' fills configured markers; 'outline'
                draws only their colored outlines. None preserves the
                established marker appearance.
            crop_to_track: If True, tightly fit the axes around the track using
                the established behavior. False uses a normal square plotting
                area with Matplotlib margins.
            show_smoothed_turn_trajectory: Display the processed trajectory
                cached by calculate_turn_statistics.
            smoothed_turn_trajectory_color: Color of the turn-analysis path.
            show_angle_smoothed_trajectory: Display the independent smoothed
                path cached by calculate_angle_tumble_statistics.
            smoothed_angle_trajectory_color: Color of the angle-tumble path.
            particle_marker: Optional Matplotlib marker for each observed
                particle position, such as 'o', 's', '^', 'D', or 'x'. None
                draws no observed-position markers.
            turn_marker: Matplotlib marker for turns when
                ``turn_display_mode='markers'``, such as 'o', 's', '^', 'D',
                or 'x'. It is unused in segment mode.
            tumble_marker: Matplotlib marker for angle-based tumbles when
                ``tumble_display_mode='markers'``, such as 'o', 's', '^', 'D',
                or 'x'. It is unused in segment mode; None restores the x
                default.
            title: Primary axis title. None restores the generated title; an
                empty string removes it.
            title_fontsize: Primary-title font size in points.
            x_axis_fontsize: X-axis label font size in points.
            y_axis_fontsize: Y-axis label font size in points.
            x_tick_fontsize: X-axis tick-label font size in points.
            y_tick_fontsize: Y-axis tick-label font size in points.
            font_family: Typeface for all figure text. Portable families are
                'sans-serif', 'serif', 'monospace', 'cursive', and 'fantasy'.
                Bundled faces include 'DejaVu Sans', 'DejaVu Serif', and
                'DejaVu Sans Mono'; installed system-font names are accepted.
            save_plots: Automatically save when ``save_path`` is None.
            save_plot_path: Directory for automatic numbered filenames.
            file_extension: Automatic-save format: 'png' or 'tif'.
        """
        plotting_parameters = self.__resolve_analysis_plotting_parameters(
            function_tumble_marker_default='x',
            track_thickness=track_thickness,
            smooth_trajectory_thickness=smooth_trajectory_thickness,
            segment_marker_thickness=segment_marker_thickness,
            crop_to_track=crop_to_track,
            marker_fill_mode=marker_fill_mode,
            show_turns=show_turns,
            show_tumbles=show_tumbles,
            show_smoothed_trajectory=show_smoothed_turn_trajectory,
            turn_display_mode=turn_display_mode,
            tumble_display_mode=tumble_display_mode,
            turn_color=turn_color,
            tumble_color=tumble_color,
            smoothed_trajectory_color=smoothed_turn_trajectory_color,
            show_angle_smoothed_trajectory=(
                show_angle_smoothed_trajectory
            ),
            smoothed_angle_trajectory_color=(
                smoothed_angle_trajectory_color
            ),
            particle_marker=particle_marker,
            turn_marker=turn_marker,
            tumble_marker=tumble_marker,
            title=title,
            title_fontsize=title_fontsize,
            x_axis_fontsize=x_axis_fontsize,
            y_axis_fontsize=y_axis_fontsize,
            x_tick_fontsize=x_tick_fontsize,
            y_tick_fontsize=y_tick_fontsize,
            font_family=font_family,
            dpi=dpi,
            save_plots=save_plots,
            save_plot_path=save_plot_path,
            file_extension=file_extension
        )
        particle_id = self.__normalize_single_particle_id(particle_id)
        default_title = self.__combined_analysis_default_title(
            particle_id=particle_id,
            tumble_method_label='angle-based',
            show_turns=plotting_parameters['show_turns'],
            show_tumbles=plotting_parameters['show_tumbles']
        )
        save_path = self.__resolve_numbered_plot_save_path(
            save_path=save_path,
            save_plots=plotting_parameters['save_plots'],
            save_plot_path=plotting_parameters['save_plot_path'],
            file_extension=plotting_parameters['file_extension'],
            default_title=default_title,
            title=plotting_parameters['title']
        )
        return self.__plot_particle_turn_tumble_analysis(
            particle_id=particle_id,
            tumble_method='angle',
            plotting_parameters=plotting_parameters,
            figsize=figsize,
            track_color=track_color,
            save_path=save_path,
            dpi=plotting_parameters['dpi'],
            show=show,
            show_turns=plotting_parameters['show_turns'],
            show_tumbles=plotting_parameters['show_tumbles'],
            show_smoothed_turn_trajectory=(
                plotting_parameters['show_smoothed_trajectory']
            ),
            show_tumble_smoothed_trajectory=(
                plotting_parameters['show_angle_smoothed_trajectory']
            ),
            turn_display_mode=plotting_parameters['turn_display_mode'],
            tumble_display_mode=plotting_parameters['tumble_display_mode'],
            turn_color=plotting_parameters['turn_color'],
            tumble_color=plotting_parameters['tumble_color'],
            smoothed_turn_trajectory_color=(
                plotting_parameters['smoothed_trajectory_color']
            ),
            tumble_smoothed_trajectory_color=(
                plotting_parameters['smoothed_angle_trajectory_color']
            ),
            particle_marker=plotting_parameters['particle_marker'],
            turn_marker=plotting_parameters['turn_marker'],
            tumble_marker=plotting_parameters['tumble_marker']
        )

    def plot_particle_turn_velocity_tumble_analysis(
        self,
        particle_id: int,
        figsize: tuple[float, float] = (10, 8),
        track_color: str | Colormap = 'viridis',
        save_path: str | os.PathLike | None = None,
        dpi: int | _UseConfiguredPlotValue = _USE_CONFIGURED_PLOT_VALUE,
        show: bool = True,
        show_turns: (
            bool | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        show_tumbles: (
            bool | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        show_smoothed_turn_trajectory: (
            bool | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        turn_display_mode: (
            str | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        tumble_display_mode: (
            str | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        turn_color: (
            str | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        tumble_color: (
            str | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        smoothed_turn_trajectory_color: (
            str | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        show_velocity_comparison: bool = True,
        velocity_comparison_figsize: tuple[float, float] = (14, 6),
        speed_colormap_name: str = 'turbo',
        comparison_track_color: str | Colormap = '0.55',
        comparison_tumble_color: str = 'red',
        direction_arrow_color: str = 'black',
        show_direction_arrow: bool = True,
        show_comparison_axes: bool = False,
        comparison_save_path: str | None = None,
        show_velocity_smoothed_trajectory: (
            bool | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        smoothed_velocity_trajectory_color: (
            str | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        particle_marker: (
            str | None | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        turn_marker: (
            str | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        tumble_marker: (
            str | None | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        title: (
            str | None | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        title_fontsize: (
            float | None | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        x_axis_fontsize: (
            float | None | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        y_axis_fontsize: (
            float | None | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        x_tick_fontsize: (
            float | None | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        y_tick_fontsize: (
            float | None | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        font_family: (
            str | None | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        save_plots: (
            bool | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        save_plot_path: (
            str | os.PathLike | None | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        file_extension: (
            str | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        track_thickness: (
            float | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        smooth_trajectory_thickness: (
            float | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        segment_marker_thickness: (
            float | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        crop_to_track: (
            bool | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE,
        marker_fill_mode: (
            str | None | _UseConfiguredPlotValue
        ) = _USE_CONFIGURED_PLOT_VALUE
    ) -> tuple:
        """
        Plot turns and velocity tumbles plus a Figure-5-style comparison.

        With ``show_velocity_comparison=True``, the return value contains the
        original overlay result and a second two-axis figure. The comparison's
        left track is colored by speed without event overlays; its right track
        uses either a uniform Matplotlib color or an elapsed-time colormap name
        or object, with velocity-tumble segments and a direction arrow beside
        the beginning of the track. Color-like names take precedence, so pass
        ``plt.get_cmap('gray')`` rather than ``'gray'`` for a gray elapsed-time
        gradient.
        The processed turn path and frame-aligned velocity-tumble smoothed path
        can be displayed and colored independently when their analyses used
        different smoothing algorithms.

        Omitted configurable arguments use
        ``set_analysis_plotting_parameters``; explicit values override those
        settings for this call only.

        Args:
            track_color: Solid Matplotlib color, colormap name, or Colormap
                object for the observed track in the combined overlay. A
                colormap colors positions by elapsed time and adds a colorbar.
                Color-like names take precedence, so use
                ``plt.get_cmap('gray')`` for a gray elapsed-time gradient.
            track_thickness: Observed-track line width in points. The separate
                comparison figure preserves its relative speed-track emphasis.
            smooth_trajectory_thickness: Line width in points for both optional
                smoothed trajectories in the combined overlay.
            segment_marker_thickness: Scale factor for turn/tumble segment
                widths, marker areas, and marker-edge widths, including tumble
                segments in the comparison figure.
            marker_fill_mode: 'solid' fills configured markers; 'outline'
                draws only their colored outlines. None preserves the
                established marker appearance.
            crop_to_track: If True, tightly fit the combined overlay axes around
                the track using the established behavior. False uses a normal
                square plotting area with Matplotlib margins.
            show_smoothed_turn_trajectory: Display the processed trajectory
                cached by calculate_turn_statistics.
            smoothed_turn_trajectory_color: Color of the turn-analysis path.
            show_velocity_smoothed_trajectory: Display the independent
                smoothed path cached by calculate_velocity_tumble_statistics.
            smoothed_velocity_trajectory_color: Color of the velocity-tumble
                path in the combined overlay. This does not change the
                separate Figure-5-style comparison.
            particle_marker: Optional Matplotlib marker for each observed
                particle position in the combined and comparison trajectory
                panels, such as 'o', 's', '^', 'D', or 'x'. None draws no
                observed-position markers.
            turn_marker: Matplotlib marker for turns in the combined overlay
                when ``turn_display_mode='markers'``, such as 'o', 's', '^',
                'D', or 'x'. It is unused in segment mode.
            tumble_marker: Matplotlib marker for velocity-based tumbles in the
                combined overlay when ``tumble_display_mode='markers'``, such
                as 'o', 's', '^', 'D', or 'x'. It does not change the
                segment-based comparison figure; None restores the x default.
            title: Primary overlay title. None restores the generated title;
                an empty string removes it. The comparison keeps its generated
                title when its axes are displayed.
            title_fontsize: Primary-title font size in points for both figures.
            x_axis_fontsize: X-axis label font size in points.
            y_axis_fontsize: Y-axis label font size in points.
            x_tick_fontsize: X-axis tick-label font size in points.
            y_tick_fontsize: Y-axis tick-label font size in points.
            font_family: Typeface for all text in both figures. Portable
                families are 'sans-serif', 'serif', 'monospace', 'cursive',
                and 'fantasy'. Bundled faces include 'DejaVu Sans',
                'DejaVu Serif', and 'DejaVu Sans Mono'; installed system-font
                names are accepted.
            save_plots: Automatically save when ``save_path`` is None.
            save_plot_path: Directory for automatic numbered filenames.
            file_extension: Automatic-save format: 'png' or 'tif'.
        """
        self.__validate_boolean_argument(
            show_velocity_comparison, 'show_velocity_comparison'
        )
        if not isinstance(show, bool):
            raise TypeError('show must be a boolean.')
        plotting_parameters = self.__resolve_analysis_plotting_parameters(
            function_tumble_marker_default='x',
            track_thickness=track_thickness,
            smooth_trajectory_thickness=smooth_trajectory_thickness,
            segment_marker_thickness=segment_marker_thickness,
            crop_to_track=crop_to_track,
            marker_fill_mode=marker_fill_mode,
            show_turns=show_turns,
            show_tumbles=show_tumbles,
            show_smoothed_trajectory=show_smoothed_turn_trajectory,
            turn_display_mode=turn_display_mode,
            tumble_display_mode=tumble_display_mode,
            turn_color=turn_color,
            tumble_color=tumble_color,
            smoothed_trajectory_color=smoothed_turn_trajectory_color,
            show_velocity_smoothed_trajectory=(
                show_velocity_smoothed_trajectory
            ),
            smoothed_velocity_trajectory_color=(
                smoothed_velocity_trajectory_color
            ),
            particle_marker=particle_marker,
            turn_marker=turn_marker,
            tumble_marker=tumble_marker,
            title=title,
            title_fontsize=title_fontsize,
            x_axis_fontsize=x_axis_fontsize,
            y_axis_fontsize=y_axis_fontsize,
            x_tick_fontsize=x_tick_fontsize,
            y_tick_fontsize=y_tick_fontsize,
            font_family=font_family,
            dpi=dpi,
            save_plots=save_plots,
            save_plot_path=save_plot_path,
            file_extension=file_extension
        )
        particle_id = self.__normalize_single_particle_id(particle_id)
        default_title = self.__combined_analysis_default_title(
            particle_id=particle_id,
            tumble_method_label='velocity-based',
            show_turns=plotting_parameters['show_turns'],
            show_tumbles=plotting_parameters['show_tumbles']
        )
        save_path = self.__resolve_numbered_plot_save_path(
            save_path=save_path,
            save_plots=plotting_parameters['save_plots'],
            save_plot_path=plotting_parameters['save_plot_path'],
            file_extension=plotting_parameters['file_extension'],
            default_title=default_title,
            title=plotting_parameters['title']
        )
        overlay_result = self.__plot_particle_turn_tumble_analysis(
            particle_id=particle_id,
            tumble_method='velocity',
            plotting_parameters=plotting_parameters,
            figsize=figsize,
            track_color=track_color,
            save_path=save_path,
            dpi=plotting_parameters['dpi'],
            show=(show if not show_velocity_comparison else False),
            show_turns=plotting_parameters['show_turns'],
            show_tumbles=plotting_parameters['show_tumbles'],
            show_smoothed_turn_trajectory=(
                plotting_parameters['show_smoothed_trajectory']
            ),
            show_tumble_smoothed_trajectory=(
                plotting_parameters['show_velocity_smoothed_trajectory']
            ),
            turn_display_mode=plotting_parameters['turn_display_mode'],
            tumble_display_mode=plotting_parameters['tumble_display_mode'],
            turn_color=plotting_parameters['turn_color'],
            tumble_color=plotting_parameters['tumble_color'],
            smoothed_turn_trajectory_color=(
                plotting_parameters['smoothed_trajectory_color']
            ),
            tumble_smoothed_trajectory_color=(
                plotting_parameters['smoothed_velocity_trajectory_color']
            ),
            particle_marker=plotting_parameters['particle_marker'],
            turn_marker=plotting_parameters['turn_marker'],
            tumble_marker=plotting_parameters['tumble_marker']
        )
        if not show_velocity_comparison:
            return overlay_result

        resolved_comparison_save_path = comparison_save_path
        if resolved_comparison_save_path is None:
            resolved_comparison_save_path = self.__derive_figure_save_path(
                save_path, '_velocity_comparison'
            )
        comparison_result = (
            self.__plot_particle_velocity_tumble_comparison(
                particle_id=particle_id,
                plotting_parameters=plotting_parameters,
                figsize=velocity_comparison_figsize,
                speed_colormap_name=speed_colormap_name,
                track_color=comparison_track_color,
                tumble_color=comparison_tumble_color,
                direction_arrow_color=direction_arrow_color,
                show_tumbles=plotting_parameters['show_tumbles'],
                show_direction_arrow=show_direction_arrow,
                show_axes=show_comparison_axes,
                save_path=resolved_comparison_save_path,
                dpi=plotting_parameters['dpi'],
                show=False,
                particle_marker=plotting_parameters['particle_marker']
            )
        )
        if show:
            plt.show()
        return overlay_result, comparison_result

    def calculate_speed_feature_correlation(
        self,
        feature_column: str = 'area',
        feature_unit: str = 'auto',
        speed_unit: str = DEFAULT_SPEED_UNIT,
        particle_ids: int | list[int] | tuple[int, ...] | None = None,
        max_frame_gap: int | None = None,
        discard_initial_frames: int = 10,
        discard_final_frames: int = 10
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Calculate a per-particle feature-versus-speed correlation.

        Area is scaled by the square of the capture pixel scale; axis lengths
        are scaled once. Other feature columns remain in their stored units.
        For area or axis lengths, feature_unit='auto' or 'scale_units' applies
        the appropriate Capture scaling; raw/stored retains pixel values.
        mean_speed is total retained path length divided by total retained
        elapsed time; mean_interval_speed is the unweighted interval mean.
        discard_initial_frames and discard_final_frames remove detections in
        inclusive source-frame windows at each track boundary before feature
        and speed means are calculated.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: Per-particle means and a one-row
                Pearson-correlation/linear-regression summary.
        """
        self._speed_feature_particles_dataframe = pd.DataFrame()
        self._speed_feature_correlation_dataframe = pd.DataFrame()
        if not isinstance(feature_column, str) or not feature_column.strip():
            raise ValueError('feature_column must be a non-empty string.')
        feature_column = feature_column.strip()
        selected_dataframe = self.__select_particle_rows(
            particle_ids,
            required_columns=[feature_column],
            discard_initial_frames=discard_initial_frames,
            discard_final_frames=discard_final_frames
        )
        feature_factor, feature_unit_label = self.__get_feature_conversion(
            feature_column, feature_unit
        )
        step_metrics = self.calculate_step_metrics(
            particle_ids=particle_ids,
            speed_unit=speed_unit,
            max_frame_gap=max_frame_gap,
            discard_initial_frames=discard_initial_frames,
            discard_final_frames=discard_final_frames
        )
        _, speed_unit_label = self.__normalize_speed_unit(speed_unit)
        step_groups = {
            int(particle): rows
            for particle, rows in step_metrics.groupby('particle', sort=False)
        }

        particle_rows = []
        for particle_id, feature_rows in selected_dataframe.groupby(
            'particle', sort=True
        ):
            feature_values = self.__finite_numeric_values(
                feature_rows[feature_column]
            ) * feature_factor
            particle_steps = step_groups.get(int(particle_id))
            speeds = (
                self.__finite_numeric_values(particle_steps['speed'])
                if particle_steps is not None and not particle_steps.empty
                else np.array([], dtype=float)
            )
            overall_average_speed = (
                float(particle_steps['step_distance'].sum()) /
                float(particle_steps['elapsed_time'].sum())
                if (
                    particle_steps is not None and
                    not particle_steps.empty and
                    float(particle_steps['elapsed_time'].sum()) > 0
                )
                else np.nan
            )
            particle_rows.append({
                'source_dataframe': self._resolved_source_dataframe,
                'particle': int(particle_id),
                'detection_count': int(len(feature_rows)),
                'speed_interval_count': int(len(speeds)),
                'mean_feature': self.__safe_mean(feature_values),
                'mean_speed': overall_average_speed,
                'mean_interval_speed': self.__safe_mean(speeds),
                'feature_column': feature_column,
                'feature_unit': feature_unit_label,
                'speed_unit': speed_unit_label,
                'discard_initial_frames': int(discard_initial_frames),
                'discard_final_frames': int(discard_final_frames),
            })
        particle_statistics = pd.DataFrame(particle_rows)
        finite_rows = particle_statistics[
            ['mean_feature', 'mean_speed']
        ].apply(pd.to_numeric, errors='coerce').replace(
            [np.inf, -np.inf], np.nan
        ).dropna()
        if len(finite_rows) < 2:
            raise ValueError(
                'At least two particles with finite feature and speed means '
                'are required for correlation.'
            )
        if (
            finite_rows['mean_feature'].nunique() < 2 or
            finite_rows['mean_speed'].nunique() < 2
        ):
            raise ValueError(
                'Feature and speed means must each contain at least two '
                'distinct values for correlation.'
            )
        correlation, p_value = pearsonr(
            finite_rows['mean_feature'], finite_rows['mean_speed']
        )
        regression = linregress(
            finite_rows['mean_feature'], finite_rows['mean_speed']
        )
        correlation_summary = pd.DataFrame([{
            'source_dataframe': self._resolved_source_dataframe,
            'feature_column': feature_column,
            'feature_unit': feature_unit_label,
            'speed_unit': speed_unit_label,
            'discard_initial_frames': int(discard_initial_frames),
            'discard_final_frames': int(discard_final_frames),
            'particle_count': int(len(finite_rows)),
            'pearson_r': float(correlation),
            'p_value': float(p_value),
            'slope': float(regression.slope),
            'intercept': float(regression.intercept),
            'r_squared': float(regression.rvalue**2),
            'slope_standard_error': float(regression.stderr),
        }])
        particle_statistics = self.__add_strain_field(particle_statistics)
        correlation_summary = self.__add_strain_field(correlation_summary)
        self._speed_feature_particles_dataframe = particle_statistics
        self._speed_feature_correlation_dataframe = correlation_summary
        return particle_statistics.copy(), correlation_summary.copy()

    def plot_speed_feature_correlation(
        self,
        figsize: tuple[float, float] = (8, 6),
        save_path: str | None = None,
        dpi: int = 300,
        show: bool = True,
        particle_color: str = 'C0',
        linear_fit_color: str = 'red',
        particle_marker: str = 'o'
    ) -> tuple:
        """
        Plot the latest per-particle speed-feature correlation and fit line.

        Args:
            particle_color: Matplotlib color for the per-particle points.
            linear_fit_color: Matplotlib color for the linear-regression line.
            particle_marker: Matplotlib marker shape for particle points, such
                as 'o', 's', '^', 'D', or 'x'.
        """
        self.__validate_plot_color(particle_color, 'particle_color')
        self.__validate_plot_color(linear_fit_color, 'linear_fit_color')
        self.__validate_plot_marker(particle_marker, 'particle_marker')
        particle_statistics = self._speed_feature_particles_dataframe
        correlation_summary = self._speed_feature_correlation_dataframe
        if particle_statistics.empty or correlation_summary.empty:
            raise ValueError(
                'Calculate a speed-feature correlation before plotting it.'
            )
        finite_rows = particle_statistics.replace(
            [np.inf, -np.inf], np.nan
        ).dropna(subset=['mean_feature', 'mean_speed'])
        result = correlation_summary.iloc[0]
        ordered_rows = finite_rows.sort_values('mean_feature')

        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(
            finite_rows['mean_feature'], finite_rows['mean_speed'],
            color=particle_color, marker=particle_marker,
            alpha=0.75, label='Particles'
        )
        fitted_speed = (
            float(result['slope']) * ordered_rows['mean_feature'] +
            float(result['intercept'])
        )
        ax.plot(
            ordered_rows['mean_feature'], fitted_speed,
            color=linear_fit_color, label='Linear fit'
        )
        feature_label = str(result['feature_column']).replace('_', ' ').title()
        feature_unit = self.__plot_unit_label(result['feature_unit'])
        speed_unit = self.__plot_unit_label(result['speed_unit'])
        ax.set_xlabel(f'Mean {feature_label} ({feature_unit})')
        ax.set_ylabel(f'Mean speed ({speed_unit})')
        ax.set_title(
            f"{feature_label} vs. speed "
            f"(r={float(result['pearson_r']):.3f}, "
            f"p={float(result['p_value']):.3g})"
        )
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        self.__finalize_figure(fig, save_path, dpi, show)
        return fig, ax

    def calculate_mean_squared_displacement(
        self,
        particle_ids: int | list[int] | tuple[int, ...] | None = None,
        max_lag_time_frames: int = 100,
        distance_unit: str = 'scale_units',
        time_unit: str = 'seconds',
        alpha_fit_lag_range: tuple[float, float] | None = None,
        discard_initial_frames: int = 10,
        discard_final_frames: int = 10
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Calculate individual and ensemble MSD plus a log-log slope alpha.

        max_lag_time_frames is always expressed in source frames. The returned
        long-form tables identify their source and units. Alpha fitting is
        reported as unavailable, rather than blocking valid MSD results, when
        fewer than two usable points fall inside alpha_fit_lag_range.
        distance_unit accepts pixels, scale_units, or the configured physical
        unit; scale_units applies the Capture pixel scale automatically.
        discard_initial_frames and discard_final_frames remove detections in
        inclusive source-frame windows at each track boundary before individual
        and ensemble MSD are calculated. Every returned table records the
        capture FPS and pixel calibration; the fit table also distinguishes the
        requested alpha-fit bounds from the data-supported bounds actually fit.
        """
        self._individual_msd_dataframe = pd.DataFrame()
        self._ensemble_msd_dataframe = pd.DataFrame()
        self._msd_fit_dataframe = pd.DataFrame()
        self._msd_metadata = {}
        self.__validate_positive_integer(
            max_lag_time_frames, 'max_lag_time_frames'
        )
        selected_dataframe = self.__select_particle_rows(
            particle_ids,
            discard_initial_frames=discard_initial_frames,
            discard_final_frames=discard_final_frames
        )
        _, distance_factor, distance_unit_label = self.__normalize_length_unit(
            distance_unit,
            argument_name='distance_unit'
        )
        normalized_time_unit = str(time_unit).strip().lower()
        if normalized_time_unit in {'second', 'seconds', 's'}:
            self.__validate_frame_rate()
            trackpy_fps = self._capture_speed_in_fps
            time_unit_label = 's'
        elif normalized_time_unit in {'frame', 'frames'}:
            trackpy_fps = 1.0
            time_unit_label = 'frames'
        else:
            raise ValueError("time_unit must be 'seconds' or 'frames'.")
        self.__validate_range(
            alpha_fit_lag_range,
            'alpha_fit_lag_range',
            allow_none=True
        )
        normalized_fit_range = (
            (
                float(alpha_fit_lag_range[0]),
                float(alpha_fit_lag_range[1])
            )
            if alpha_fit_lag_range is not None
            else None
        )

        msd_source = selected_dataframe[
            ['particle', 'frame', 'centroid_x', 'centroid_y']
        ].copy()
        individual_msd_wide = tp.imsd(
            msd_source,
            mpp=distance_factor,
            fps=trackpy_fps,
            max_lagtime=int(max_lag_time_frames),
            pos_columns=['centroid_y', 'centroid_x']
        )
        ensemble_series = tp.emsd(
            msd_source,
            mpp=distance_factor,
            fps=trackpy_fps,
            max_lagtime=int(max_lag_time_frames),
            detail=False,
            pos_columns=['centroid_y', 'centroid_x']
        )
        msd_unit = f'{distance_unit_label}²'
        individual_msd_wide.index.name = 'lag_time'
        individual_msd_wide.columns.name = 'particle'
        individual_msd = (
            individual_msd_wide.stack(dropna=True)
            .rename('individual_msd')
            .reset_index()
        )
        if not individual_msd.empty:
            individual_msd['particle'] = individual_msd[
                'particle'
            ].astype(int)
        individual_msd.insert(
            0, 'source_dataframe', self._resolved_source_dataframe
        )
        individual_msd['capture_speed_in_fps'] = self._capture_speed_in_fps
        individual_msd['pixel_scale_factor'] = self.pixel_scale_factor
        individual_msd['scale_units'] = self._scale_units
        individual_msd['time_unit'] = time_unit_label
        individual_msd['msd_unit'] = msd_unit
        individual_msd = individual_msd[self.INDIVIDUAL_MSD_COLUMNS]
        ensemble_msd = pd.DataFrame({
            'source_dataframe': self._resolved_source_dataframe,
            'lag_time': ensemble_series.index.to_numpy(dtype=float),
            'ensemble_msd': ensemble_series.to_numpy(dtype=float),
            'capture_speed_in_fps': self._capture_speed_in_fps,
            'pixel_scale_factor': self.pixel_scale_factor,
            'scale_units': self._scale_units,
            'time_unit': time_unit_label,
            'msd_unit': msd_unit,
        })
        ensemble_msd = ensemble_msd[self.ENSEMBLE_MSD_COLUMNS]
        fit_mask = (
            np.isfinite(ensemble_msd['lag_time']) &
            np.isfinite(ensemble_msd['ensemble_msd']) &
            (ensemble_msd['lag_time'] > 0) &
            (ensemble_msd['ensemble_msd'] > 0)
        )
        if normalized_fit_range is not None:
            fit_mask &= ensemble_msd['lag_time'].between(
                normalized_fit_range[0], normalized_fit_range[1]
            )
        fit_rows = ensemble_msd.loc[fit_mask]
        if len(fit_rows) >= 2:
            regression = linregress(
                np.log(fit_rows['lag_time']),
                np.log(fit_rows['ensemble_msd'])
            )
            fit_status = 'fitted'
            fit_message = ''
            fit_values = {
                'alpha': float(regression.slope),
                'log_intercept': float(regression.intercept),
                'r_squared': float(regression.rvalue**2),
                'p_value': float(regression.pvalue),
                'alpha_standard_error': float(regression.stderr),
            }
        else:
            fit_status = 'insufficient_data'
            fit_message = (
                'At least two positive finite ensemble-MSD points are required '
                'within alpha_fit_lag_range.'
            )
            fit_values = {
                'alpha': np.nan,
                'log_intercept': np.nan,
                'r_squared': np.nan,
                'p_value': np.nan,
                'alpha_standard_error': np.nan,
            }
        fit_summary = pd.DataFrame([{
            'source_dataframe': self._resolved_source_dataframe,
            'particle_count': int(individual_msd['particle'].nunique()),
            'selected_particle_count': int(
                selected_dataframe['particle'].nunique()
            ),
            'contributing_particle_count': int(
                individual_msd['particle'].nunique()
            ),
            'max_lag_time_frames': int(max_lag_time_frames),
            'discard_initial_frames': int(discard_initial_frames),
            'discard_final_frames': int(discard_final_frames),
            'requested_alpha_fit_lag_start': (
                normalized_fit_range[0]
                if normalized_fit_range is not None else np.nan
            ),
            'requested_alpha_fit_lag_end': (
                normalized_fit_range[1]
                if normalized_fit_range is not None else np.nan
            ),
            'fit_lag_start': (
                float(fit_rows['lag_time'].min())
                if len(fit_rows) else np.nan
            ),
            'fit_lag_end': (
                float(fit_rows['lag_time'].max())
                if len(fit_rows) else np.nan
            ),
            'fit_point_count': int(len(fit_rows)),
            **fit_values,
            'fit_status': fit_status,
            'fit_message': fit_message,
            'capture_speed_in_fps': self._capture_speed_in_fps,
            'pixel_scale_factor': self.pixel_scale_factor,
            'scale_units': self._scale_units,
            'time_unit': time_unit_label,
            'msd_unit': msd_unit,
        }], columns=self.MSD_FIT_COLUMNS)
        individual_msd = self.__add_strain_field(individual_msd)
        ensemble_msd = self.__add_strain_field(ensemble_msd)
        fit_summary = self.__add_strain_field(fit_summary)
        self._individual_msd_dataframe = individual_msd
        self._ensemble_msd_dataframe = ensemble_msd
        self._msd_fit_dataframe = fit_summary
        self._msd_metadata = {
            'distance_unit': distance_unit_label,
            'time_unit': time_unit_label,
            'msd_unit': msd_unit,
            'fit_lag_range': normalized_fit_range,
            'requested_alpha_fit_lag_range': normalized_fit_range,
            'capture_speed_in_fps': self._capture_speed_in_fps,
            'pixel_scale_factor': self.pixel_scale_factor,
            'scale_units': self._scale_units,
            'discard_initial_frames': int(discard_initial_frames),
            'discard_final_frames': int(discard_final_frames),
        }
        return (
            individual_msd.copy(),
            ensemble_msd.copy(),
            fit_summary.copy()
        )

    def plot_mean_squared_displacement(
        self,
        show_individual: bool = True,
        figsize: tuple[float, float] = (9, 7),
        save_path: str | None = None,
        dpi: int = 300,
        show: bool = True,
        ensemble_msd_color: str = 'blue',
        log_log_fit_color: str = 'red'
    ) -> tuple:
        """
        Plot cached individual/ensemble MSD and the fitted alpha slope.

        Args:
            ensemble_msd_color: Matplotlib color for the ensemble-MSD line.
            log_log_fit_color: Matplotlib color for the log-log fitted line.
        """
        if not isinstance(show_individual, bool):
            raise TypeError('show_individual must be a boolean.')
        self.__validate_plot_color(
            ensemble_msd_color, 'ensemble_msd_color'
        )
        self.__validate_plot_color(
            log_log_fit_color, 'log_log_fit_color'
        )
        if (
            self._individual_msd_dataframe.empty or
            self._ensemble_msd_dataframe.empty or
            self._msd_fit_dataframe.empty
        ):
            raise ValueError(
                'Calculate mean squared displacement before plotting it.'
            )
        individual_msd = self._individual_msd_dataframe
        ensemble_msd = self._ensemble_msd_dataframe
        fit_result = self._msd_fit_dataframe.iloc[0]

        fig, ax = plt.subplots(figsize=figsize)
        if show_individual:
            for _, particle_rows in individual_msd.groupby(
                'particle', sort=True
            ):
                particle_values = particle_rows['individual_msd'].to_numpy(
                    dtype=float
                )
                valid_values = (
                    np.isfinite(particle_values) & (particle_values > 0)
                )
                ax.plot(
                    particle_rows['lag_time'].to_numpy(dtype=float)[
                        valid_values
                    ],
                    particle_values[valid_values],
                    color='black', alpha=0.08, linewidth=0.8
                )
        ensemble_valid = (
            np.isfinite(ensemble_msd['lag_time']) &
            np.isfinite(ensemble_msd['ensemble_msd']) &
            (ensemble_msd['lag_time'] > 0) &
            (ensemble_msd['ensemble_msd'] > 0)
        )
        ax.plot(
            ensemble_msd.loc[ensemble_valid, 'lag_time'],
            ensemble_msd.loc[ensemble_valid, 'ensemble_msd'],
            color=ensemble_msd_color, linewidth=3, label='Ensemble MSD'
        )
        if np.isfinite(float(fit_result['alpha'])):
            fit_lags = ensemble_msd.loc[
                ensemble_msd['lag_time'].between(
                    float(fit_result['fit_lag_start']),
                    float(fit_result['fit_lag_end'])
                ) & ensemble_valid,
                'lag_time'
            ].to_numpy(dtype=float)
            fitted_msd = np.exp(float(fit_result['log_intercept'])) * (
                fit_lags ** float(fit_result['alpha'])
            )
            ax.plot(
                fit_lags, fitted_msd, color=log_log_fit_color,
                linestyle='--', linewidth=2,
                label=f"Log-log fit (α={float(fit_result['alpha']):.3f})"
            )
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(f"Lag time ({fit_result['time_unit']})")
        msd_unit = self.__plot_unit_label(fit_result['msd_unit'])
        ax.set_ylabel(f'Mean squared displacement ({msd_unit})')
        ax.set_title(
            f'Mean squared displacement - {self._resolved_source_dataframe}'
        )
        ax.legend()
        fig.tight_layout()
        self.__finalize_figure(fig, save_path, dpi, show)
        return fig, ax

    def calculate_speed_and_plot_mean(
        self,
        distribution_type: str = DEFAULT_DISTRIBUTION,
        fit_range: tuple[float, float] | None = None,
        ci_range: tuple[float, float] = (5, 95),
        bin_size: int = 30,
        speed_unit: str = DEFAULT_SPEED_UNIT,
        plot_results: bool = True,
        discard_initial_frames: int = 10,
        discard_final_frames: int = 10
    ) -> np.ndarray:
        """
        Calculate each particle's fitted mean speed and optionally plot it.

        Args:
            distribution_type (str): Distribution type for fitting (default: 'norm').
            fit_range (tuple): User-defined speed range for fitting. By default,
                the confidence interval determines the range.
            ci_range (tuple): Confidence interval range for the default speed
                limits. Defaults to (5, 95).
            bin_size (int): Number of bins for histogram (default: 30).
            speed_unit (str): Unit used for both the speed calculation and plot
                labels. Supported dimensions are pixels/frame, pixels/s. Use
                scale_units/frame or scale_units/s to resolve the configured
                Capture scale unit automatically.
            plot_results (bool): Plot each particle's speed histogram and
                fitted distribution. Defaults to True.
            discard_initial_frames (int): Number of source frames discarded
                from the start of each track before speed is calculated.
                Defaults to 10.
            discard_final_frames (int): Number of source frames discarded from
                the end of each track before speed is calculated. Defaults to
                10.

        Returns:
            np.ndarray: Array of mean speeds for each particle.

        The detailed export records the requested fitting settings and Capture
        calibration. Effective percentile-derived bounds remain particle-
        specific and are not treated as a shared run setting.
        """
        self._mean_array = []
        self._mean_speeds_dataframe = pd.DataFrame()
        self._calculated_speed_unit = None
        if not isinstance(plot_results, bool):
            raise TypeError('plot_results must be a boolean.')
        if not isinstance(distribution_type, str) or not distribution_type.strip():
            raise ValueError('distribution_type must be a non-empty string.')
        if isinstance(bin_size, bool) or not isinstance(
            bin_size, (int, np.integer)
        ):
            raise TypeError('bin_size must be an integer.')
        if bin_size < 1:
            raise ValueError('bin_size must be at least 1.')
        self.__validate_range(fit_range, 'fit_range', allow_none=True)
        self.__validate_range(ci_range, 'ci_range')
        fit_range = (
            tuple(float(value) for value in fit_range)
            if fit_range is not None
            else None
        )
        ci_range = tuple(float(value) for value in ci_range)
        if ci_range[0] < 0 or ci_range[1] > 100:
            raise ValueError('ci_range values must be between 0 and 100.')
        distribution_type = distribution_type.strip()

        speed_unit_key, speed_unit_label = self.__normalize_speed_unit(
            speed_unit
        )
        selected_dataframe = self.__select_particle_rows(
            None,
            discard_initial_frames=discard_initial_frames,
            discard_final_frames=discard_final_frames
        )
        unique_particles = selected_dataframe['particle'].unique()
        print(f'Total unique particles: {len(unique_particles)}')
        mean_array: list[float] = []
        mean_speed_rows = []

        # For each particle, compute speed and plot its distribution individually
        for idx in trange(len(unique_particles), desc='Calculating Speed'):
            each_particle = unique_particles[idx]
            current_particle = self.__get_particle_data(
                each_particle,
                source_dataframe=selected_dataframe
            )
            speed = self.__calculate_speed(
                current_particle,
                speed_unit_key=speed_unit_key
            )
            mean_speed = self.__fit_and_plot_speed_distribution(
                speed, each_particle, distribution_type=distribution_type,
                fit_range=fit_range, ci_range=ci_range,
                bin_size=bin_size, speed_unit=speed_unit_label,
                plot_results=plot_results
            )
            mean_array.append(mean_speed)
            mean_speed_rows.append({
                'source_dataframe': self._resolved_source_dataframe,
                'particle': each_particle,
                'mean_speed': mean_speed,
                'distribution_type': distribution_type,
                'fit_range_source': (
                    'requested' if fit_range is not None
                    else 'ci_percentiles'
                ),
                'requested_fit_range_start': (
                    fit_range[0] if fit_range is not None else np.nan
                ),
                'requested_fit_range_end': (
                    fit_range[1] if fit_range is not None else np.nan
                ),
                'ci_range_lower_percentile': ci_range[0],
                'ci_range_upper_percentile': ci_range[1],
                'capture_speed_in_fps': self._capture_speed_in_fps,
                'pixel_scale_factor': self.pixel_scale_factor,
                'scale_units': self._scale_units,
                'speed_unit': speed_unit_label,
                'discard_initial_frames': int(discard_initial_frames),
                'discard_final_frames': int(discard_final_frames),
            })

        self._mean_array = mean_array
        self._mean_speeds_dataframe = self.__add_strain_field(
            pd.DataFrame(
                mean_speed_rows,
                columns=self.FITTED_MEAN_SPEED_COLUMNS
            )
        )
        self._calculated_speed_unit = speed_unit_label
        return np.array(mean_array)

    def __reset_derived_analysis_results(self) -> None:
        """Clear results that belong to a previously selected source snapshot."""
        self._mean_array = []
        self._mean_speeds_dataframe = pd.DataFrame()
        self._calculated_speed_unit = None
        self._step_metrics_dataframe = pd.DataFrame(
            columns=self.STEP_METRIC_COLUMNS
        )
        self._particle_characteristics_dataframe = pd.DataFrame()
        self._turn_summary_dataframe = pd.DataFrame(
            columns=self.TURN_SUMMARY_COLUMNS
        )
        self._turn_angles_dataframe = pd.DataFrame(
            columns=self.TURN_ANGLE_COLUMNS
        )
        self._turn_path_data = {}
        self._turn_analysis_metadata = {}
        self._tumble_summary_dataframe = pd.DataFrame(
            columns=self.TUMBLE_SUMMARY_COLUMNS
        )
        self._tumble_events_dataframe = pd.DataFrame(
            columns=self.TUMBLE_EVENT_COLUMNS
        )
        self._tumble_points_dataframe = pd.DataFrame(
            columns=self.TUMBLE_POINT_COLUMNS
        )
        self._tumble_population_dataframe = pd.DataFrame(
            columns=self.TUMBLE_POPULATION_COLUMNS
        )
        self._angle_tumble_table_1_dataframe = pd.DataFrame(
            columns=self.TABLE_1_VERTICAL_COLUMNS
        )
        self._tumble_path_data = {}
        self._tumble_analysis_metadata = {}
        self._velocity_tumble_summary_dataframe = pd.DataFrame(
            columns=self.VELOCITY_TUMBLE_SUMMARY_COLUMNS
        )
        self._velocity_tumble_events_dataframe = pd.DataFrame(
            columns=self.VELOCITY_TUMBLE_EVENT_COLUMNS
        )
        self._velocity_tumble_points_dataframe = pd.DataFrame(
            columns=self.VELOCITY_TUMBLE_POINT_COLUMNS
        )
        self._velocity_tumble_population_dataframe = pd.DataFrame(
            columns=self.VELOCITY_TUMBLE_POPULATION_COLUMNS
        )
        self._velocity_tumble_table_1_dataframe = pd.DataFrame(
            columns=self.VELOCITY_TABLE_1_VERTICAL_COLUMNS
        )
        self._velocity_tumble_analysis_metadata = {}
        self._speed_feature_particles_dataframe = pd.DataFrame()
        self._speed_feature_correlation_dataframe = pd.DataFrame()
        self._individual_msd_dataframe = pd.DataFrame()
        self._ensemble_msd_dataframe = pd.DataFrame()
        self._msd_fit_dataframe = pd.DataFrame()
        self._msd_metadata = {}

    def __select_particle_rows(
        self,
        particle_ids: int | list[int] | tuple[int, ...] | None,
        required_columns: list[str] | None = None,
        discard_initial_frames: int = 0,
        discard_final_frames: int = 0
    ) -> pd.DataFrame:
        """
        Return validated particle rows after optional track-boundary trimming.

        The discard values describe inclusive source-frame windows relative to
        each particle's own first and last frame, not detection counts.
        """
        required_columns = required_columns or []
        missing_columns = [
            column for column in required_columns
            if column not in self._sorted_dataframe.columns
        ]
        if missing_columns:
            raise ValueError(
                'Selected linked-particle dataframe is missing columns '
                f'required for this analysis: {missing_columns}'
            )
        if particle_ids is None:
            selected_dataframe = self._sorted_dataframe.copy()
        elif isinstance(particle_ids, (int, float, np.integer, np.floating)):
            normalized_particle_ids = [
                self.__normalize_single_particle_id(particle_ids)
            ]
        elif isinstance(particle_ids, (list, tuple)) and particle_ids:
            normalized_particle_ids = [
                self.__normalize_single_particle_id(particle_id)
                for particle_id in particle_ids
            ]
        elif isinstance(particle_ids, (list, tuple)):
            raise ValueError('particle_ids cannot be empty.')
        else:
            raise TypeError(
                'particle_ids must be an integer, a list/tuple of integers, '
                'or None.'
            )
        if particle_ids is not None:
            normalized_particle_ids = list(dict.fromkeys(normalized_particle_ids))
            available_particles = set(
                self._sorted_dataframe['particle'].astype(int).unique()
            )
            missing_particles = [
                particle_id for particle_id in normalized_particle_ids
                if particle_id not in available_particles
            ]
            if missing_particles:
                raise ValueError(
                    f'Particle IDs are not available in the selected '
                    f'{self._resolved_source_dataframe} dataframe: '
                    f'{missing_particles}'
                )
            selected_dataframe = self._sorted_dataframe.loc[
                self._sorted_dataframe['particle'].isin(normalized_particle_ids)
            ].copy()

        selected_dataframe = selected_dataframe.sort_values(
            by=['particle', 'frame'], kind='stable'
        )
        selected_dataframe = self.__trim_particle_frame_windows(
            selected_dataframe,
            discard_initial_frames=discard_initial_frames,
            discard_final_frames=discard_final_frames
        )
        if selected_dataframe.empty:
            raise ValueError(
                'No detections remain after discarding the requested initial '
                'and final source-frame windows from each selected particle.'
            )
        return selected_dataframe

    def __trim_particle_frame_windows(
        self,
        particle_rows: pd.DataFrame,
        discard_initial_frames: int = 0,
        discard_final_frames: int = 0
    ) -> pd.DataFrame:
        """
        Remove observed rows in inclusive per-particle boundary-frame windows.

        A value N removes rows from first_frame through first_frame + N - 1,
        or from last_frame - N + 1 through last_frame. Missing detections inside
        either frame window therefore do not increase the number of rows
        discarded.
        """
        self.__validate_nonnegative_integer(
            discard_initial_frames, 'discard_initial_frames'
        )
        self.__validate_nonnegative_integer(
            discard_final_frames, 'discard_final_frames'
        )
        if particle_rows.empty:
            return particle_rows.copy()

        frames = particle_rows['frame'].astype(np.int64)
        grouped_frames = particle_rows.groupby(
            'particle', sort=False
        )['frame']
        first_frames = grouped_frames.transform('min').astype(np.int64)
        last_frames = grouped_frames.transform('max').astype(np.int64)
        retain_rows = pd.Series(True, index=particle_rows.index, dtype=bool)
        if discard_initial_frames:
            initial_window_end = (
                first_frames + int(discard_initial_frames) - 1
            )
            retain_rows &= frames > initial_window_end
        if discard_final_frames:
            final_window_start = (
                last_frames - int(discard_final_frames) + 1
            )
            retain_rows &= frames < final_window_start
        return particle_rows.loc[retain_rows].copy()

    @staticmethod
    def __normalize_single_particle_id(particle_id: object) -> int:
        """Validate and normalize one integer-valued particle ID."""
        if isinstance(particle_id, (bool, np.bool_)) or not isinstance(
            particle_id, (int, float, np.integer, np.floating)
        ):
            raise TypeError('Particle IDs must be integer-valued numbers.')
        numeric_particle = float(particle_id)
        if not np.isfinite(numeric_particle) or not numeric_particle.is_integer():
            raise ValueError('Particle IDs must be finite whole numbers.')
        return int(numeric_particle)

    @staticmethod
    def __triangular_smooth_positions(
        positions: np.ndarray,
        window_length: int | None
    ) -> tuple[np.ndarray, int | None]:
        """Apply a centered triangular filter with edge-value padding."""
        positions = np.asarray(positions, dtype=float)
        if window_length is None or len(positions) < 3:
            return positions.copy(), None
        largest_odd_window = (
            len(positions) if len(positions) % 2 == 1 else len(positions) - 1
        )
        effective_window = min(int(window_length), largest_odd_window)
        if effective_window < 3:
            return positions.copy(), None
        half_window = effective_window // 2
        ascending_weights = np.arange(1, half_window + 2, dtype=float)
        weights = np.concatenate([
            ascending_weights,
            ascending_weights[-2::-1]
        ])
        weights /= np.sum(weights)
        padded_positions = np.pad(
            positions,
            ((half_window, half_window), (0, 0)),
            mode='edge'
        )
        smoothed_positions = np.vstack([
            np.sum(
                padded_positions[
                    point_index:point_index + effective_window
                ] * weights[:, None],
                axis=0
            )
            for point_index in range(len(positions))
        ])
        return smoothed_positions, effective_window

    @staticmethod
    def __surrounding_extrema(
        center_index: int,
        opposite_extrema: np.ndarray
    ) -> tuple[int, int] | None:
        """Return the nearest opposite extrema bracketing one extremum."""
        opposite_extrema = np.asarray(opposite_extrema, dtype=int)
        left_extrema = opposite_extrema[opposite_extrema < center_index]
        right_extrema = opposite_extrema[opposite_extrema > center_index]
        if not len(left_extrema) or not len(right_extrema):
            return None
        return int(left_extrema[-1]), int(right_extrema[0])

    @staticmethod
    def __condition_bounds_around_index(
        condition: np.ndarray,
        center_index: int
    ) -> tuple[int, int]:
        """Expand from a qualifying center through its connected True region."""
        condition = np.asarray(condition, dtype=bool)
        start_index = int(center_index)
        end_index = int(center_index)
        while start_index > 0 and condition[start_index - 1]:
            start_index -= 1
        while end_index + 1 < len(condition) and condition[end_index + 1]:
            end_index += 1
        return start_index, end_index

    def __analyze_velocity_tumble_segment(
        self,
        particle_id: int,
        segment_id: int,
        segment_rows: pd.DataFrame,
        elapsed_time_offset_seconds: float,
        starting_event_id: int,
        smoothing_configuration: dict,
        speed_drop_ratio_threshold: float,
        speed_period_depth_fraction: float,
        angular_change_coefficient: float,
        minimum_heading_speed: float | None,
        speed_extrema_prominence: float | None,
        angular_velocity_extrema_prominence: float | None,
        extrema_min_distance: int,
        distance_factor: float,
        distance_unit: str,
        speed_unit: str
    ) -> tuple[list[dict], list[dict], int]:
        """Analyze one uninterrupted trajectory segment using Najafi et al."""
        frames = segment_rows['frame'].to_numpy(dtype=int)
        centroid_positions = segment_rows[
            ['centroid_x', 'centroid_y']
        ].to_numpy(dtype=float)
        raw_positions = np.column_stack([
            centroid_positions[:, 1],
            -centroid_positions[:, 0]
        ])
        smoothing_result = self.__smooth_analysis_positions(
            raw_positions,
            smoothing_configuration=smoothing_configuration,
            rdp_output_mode='interpolated',
            interpolation_coordinates=frames
        )
        smoothed_positions = smoothing_result['frame_aligned_positions']
        effective_window = smoothing_result['effective_window']
        smoothing_method = smoothing_configuration['smoothing_method']
        point_count = len(frames)
        position_is_supported = np.all(
            np.isfinite(smoothed_positions), axis=1
        )
        absolute_times = frames.astype(float) / self._capture_speed_in_fps
        elapsed_times = (
            float(elapsed_time_offset_seconds) +
            (
                frames.astype(float) - float(frames[0])
            ) / self._capture_speed_in_fps
        ) if point_count else np.array([], dtype=float)

        velocities = np.full((point_count, 2), np.nan, dtype=float)
        speeds = np.full(point_count, np.nan, dtype=float)
        headings = np.full(point_count, np.nan, dtype=float)
        previous_valid_heading_frames = np.full(
            point_count, np.nan, dtype=float
        )
        angular_elapsed_times = np.full(point_count, np.nan, dtype=float)
        angular_changes = np.full(point_count, np.nan, dtype=float)
        angular_velocity_magnitudes = np.full(
            point_count, np.nan, dtype=float
        )
        valid_heading_indices = np.array([], dtype=int)
        if point_count > 1:
            elapsed_steps = np.diff(absolute_times)
            displacements = (
                np.diff(smoothed_positions, axis=0) * distance_factor
            )
            velocities[1:] = displacements / elapsed_steps[:, None]
            speeds[1:] = np.linalg.norm(velocities[1:], axis=1)
            finite_speeds = speeds[np.isfinite(speeds)]
            finite_speed_scale = (
                float(np.max(np.abs(finite_speeds)))
                if len(finite_speeds)
                else 0.0
            )
            heading_speed_tolerance = max(
                np.finfo(float).eps,
                finite_speed_scale * 1e-12,
                (
                    float(minimum_heading_speed)
                    if minimum_heading_speed is not None else 0.0
                )
            )
            valid_heading_indices = np.flatnonzero(
                speeds > heading_speed_tolerance
            )
            headings[valid_heading_indices] = np.arctan2(
                velocities[valid_heading_indices, 1],
                velocities[valid_heading_indices, 0]
            )
            for previous_index, current_index in zip(
                valid_heading_indices[:-1],
                valid_heading_indices[1:]
            ):
                signed_heading_change = float(np.arctan2(
                    np.sin(
                        headings[current_index] -
                        headings[previous_index]
                    ),
                    np.cos(
                        headings[current_index] -
                        headings[previous_index]
                    )
                ))
                angular_changes[current_index] = abs(
                    signed_heading_change
                )
                previous_valid_heading_frames[current_index] = frames[
                    previous_index
                ]
                angular_elapsed_times[current_index] = (
                    absolute_times[current_index] -
                    absolute_times[previous_index]
                )
                angular_velocity_magnitudes[current_index] = (
                    angular_changes[current_index] /
                    angular_elapsed_times[current_index]
                )

        speed_minimum_flags = np.zeros(point_count, dtype=bool)
        speed_minimum_pass_flags = np.zeros(point_count, dtype=bool)
        angular_maximum_flags = np.zeros(point_count, dtype=bool)
        angular_maximum_pass_flags = np.zeros(point_count, dtype=bool)
        speed_candidates = []
        angular_candidates = []

        speed_peak_arguments = {'distance': extrema_min_distance}
        if speed_extrema_prominence is not None:
            speed_peak_arguments['prominence'] = speed_extrema_prominence
        if point_count > 3 and np.isfinite(speeds).any():
            valid_speed_indices = np.flatnonzero(np.isfinite(speeds))
            finite_speed_values = speeds[valid_speed_indices]
            speed_maxima = (
                valid_speed_indices[find_peaks(finite_speed_values)[0]]
            )
            speed_minima = (
                valid_speed_indices[
                    find_peaks(
                        -finite_speed_values, **speed_peak_arguments
                    )[0]
                ]
            )
            speed_minimum_flags[speed_minima] = True
            finite_speed_scale = np.nanmax(np.abs(finite_speed_values))
            zero_speed_tolerance = max(
                np.finfo(float).eps,
                float(finite_speed_scale) * 1e-12
            )
            for minimum_index in speed_minima:
                surrounding_maxima = self.__surrounding_extrema(
                    int(minimum_index), speed_maxima
                )
                if surrounding_maxima is None:
                    continue
                left_maximum, right_maximum = surrounding_maxima
                minimum_speed = float(speeds[minimum_index])
                depth = max(
                    float(speeds[left_maximum] - minimum_speed),
                    float(speeds[right_maximum] - minimum_speed)
                )
                depth = max(depth, 0.0)
                if minimum_speed <= zero_speed_tolerance:
                    depth_ratio = np.inf if depth > zero_speed_tolerance else 0.0
                else:
                    depth_ratio = depth / minimum_speed
                period_condition = (
                    np.isfinite(speeds) &
                    (
                        speeds - minimum_speed <=
                        speed_period_depth_fraction * depth +
                        zero_speed_tolerance
                    )
                )
                period_condition[:left_maximum] = False
                period_condition[right_maximum + 1:] = False
                period_start, period_end = (
                    self.__condition_bounds_around_index(
                        period_condition, int(minimum_index)
                    )
                )
                passes = bool(
                    depth_ratio >= speed_drop_ratio_threshold
                )
                speed_minimum_pass_flags[minimum_index] = passes
                speed_candidates.append({
                    'index': int(minimum_index),
                    'left_maximum': left_maximum,
                    'right_maximum': right_maximum,
                    'minimum_speed': minimum_speed,
                    'depth': depth,
                    'depth_ratio': float(depth_ratio),
                    'period_start': period_start,
                    'period_end': period_end,
                    'passes': passes,
                })

        angular_peak_arguments = {'distance': extrema_min_distance}
        if angular_velocity_extrema_prominence is not None:
            angular_peak_arguments['prominence'] = (
                angular_velocity_extrema_prominence
            )
        if point_count > 4:
            valid_angular_indices = np.flatnonzero(
                np.isfinite(angular_velocity_magnitudes)
            )
            finite_angular_values = angular_velocity_magnitudes[
                valid_angular_indices
            ]
            angular_maxima_local = find_peaks(
                finite_angular_values, **angular_peak_arguments
            )[0]
            angular_minima_local = find_peaks(
                -finite_angular_values
            )[0]
            angular_maxima = valid_angular_indices[angular_maxima_local]
            angular_maximum_flags[angular_maxima] = True
            angular_tolerance = (
                max(
                    np.finfo(float).eps,
                    float(np.nanmax(np.abs(finite_angular_values))) * 1e-12
                )
                if len(finite_angular_values)
                else np.finfo(float).eps
            )
            for maximum_local_index in angular_maxima_local:
                maximum_index = int(
                    valid_angular_indices[maximum_local_index]
                )
                surrounding_minima = self.__surrounding_extrema(
                    int(maximum_local_index), angular_minima_local
                )
                if surrounding_minima is None:
                    continue
                left_minimum_local, right_minimum_local = (
                    surrounding_minima
                )
                left_minimum = int(
                    valid_angular_indices[left_minimum_local]
                )
                right_minimum = int(
                    valid_angular_indices[right_minimum_local]
                )
                maximum_angular_velocity = float(
                    angular_velocity_magnitudes[maximum_index]
                )
                height = max(
                    maximum_angular_velocity -
                    float(angular_velocity_magnitudes[left_minimum]),
                    maximum_angular_velocity -
                    float(angular_velocity_magnitudes[right_minimum])
                )
                height = max(height, 0.0)
                surrounding_duration = float(
                    absolute_times[right_minimum] -
                    absolute_times[left_minimum]
                )
                total_directional_change = float(np.nansum(
                    angular_changes[left_minimum + 1:right_minimum + 1]
                ))
                directional_change_threshold = float(np.sqrt(
                    angular_change_coefficient * surrounding_duration
                ))
                period_condition = (
                    maximum_angular_velocity - finite_angular_values <=
                    height + angular_tolerance
                )
                period_condition[:left_minimum_local] = False
                period_condition[right_minimum_local + 1:] = False
                period_start_local, period_end_local = (
                    self.__condition_bounds_around_index(
                        period_condition, int(maximum_local_index)
                    )
                )
                period_start = int(
                    valid_angular_indices[period_start_local]
                )
                period_end = int(
                    valid_angular_indices[period_end_local]
                )
                passes = bool(
                    total_directional_change >
                    directional_change_threshold
                )
                angular_maximum_pass_flags[maximum_index] = passes
                angular_candidates.append({
                    'index': int(maximum_index),
                    'left_minimum': left_minimum,
                    'right_minimum': right_minimum,
                    'maximum_value': maximum_angular_velocity,
                    'height': height,
                    'total_directional_change': total_directional_change,
                    'directional_change_threshold': (
                        directional_change_threshold
                    ),
                    'period_start': period_start,
                    'period_end': period_end,
                    'passes': passes,
                })

        matched_candidates = []
        passing_speed_candidates = [
            candidate for candidate in speed_candidates
            if candidate['passes']
        ]
        for angular_candidate in angular_candidates:
            if not angular_candidate['passes']:
                continue
            overlapping_speed_candidates = []
            for speed_candidate in passing_speed_candidates:
                overlap_start = max(
                    angular_candidate['period_start'],
                    speed_candidate['period_start']
                )
                overlap_end = min(
                    angular_candidate['period_end'],
                    speed_candidate['period_end']
                )
                if overlap_start <= overlap_end:
                    overlapping_speed_candidates.append((
                        speed_candidate,
                        overlap_start,
                        overlap_end
                    ))
            if not overlapping_speed_candidates:
                continue
            speed_candidate, overlap_start, overlap_end = max(
                overlapping_speed_candidates,
                key=lambda match: (
                    match[2] - match[1] + 1,
                    -abs(
                        match[0]['index'] - angular_candidate['index']
                    ),
                    match[0]['depth_ratio']
                )
            )
            matched_candidates.append({
                'angular': angular_candidate,
                'speed': speed_candidate,
                'overlap_start': overlap_start,
                'overlap_end': overlap_end,
            })

        merged_candidate_groups = []
        for matched_candidate in sorted(
            matched_candidates,
            key=lambda candidate: (
                candidate['angular']['period_start'],
                candidate['angular']['period_end']
            )
        ):
            candidate_start = matched_candidate['angular']['period_start']
            candidate_end = matched_candidate['angular']['period_end']
            if (
                not merged_candidate_groups or
                candidate_start >
                merged_candidate_groups[-1]['end_index'] + 1 or
                (
                    candidate_start >
                    merged_candidate_groups[-1]['end_index'] and
                    frames[candidate_start] -
                    frames[
                        merged_candidate_groups[-1]['end_index']
                    ] > 1
                )
            ):
                merged_candidate_groups.append({
                    'start_index': candidate_start,
                    'end_index': candidate_end,
                    'matches': [matched_candidate],
                })
            else:
                merged_candidate_groups[-1]['end_index'] = max(
                    merged_candidate_groups[-1]['end_index'],
                    candidate_end
                )
                merged_candidate_groups[-1]['matches'].append(
                    matched_candidate
                )

        segment_is_analyzable = bool(
            np.sum(np.isfinite(angular_velocity_magnitudes)) >= 5
        )
        states = np.full(point_count, 'unclassified', dtype=object)
        if segment_is_analyzable:
            states[position_is_supported] = 'run'
        event_ids: list[int | None] = [None] * point_count
        event_rows = []
        next_event_id = int(starting_event_id)
        for candidate_group in merged_candidate_groups:
            start_index = int(candidate_group['start_index'])
            end_index = int(candidate_group['end_index'])
            representative_match = max(
                candidate_group['matches'],
                key=lambda match: (
                    (
                        match['angular']['total_directional_change'] /
                        max(
                            match['angular'][
                                'directional_change_threshold'
                            ],
                            np.finfo(float).eps
                        )
                    ),
                    match['speed']['depth_ratio']
                )
            )
            angular_candidate = representative_match['angular']
            speed_candidate = representative_match['speed']
            states[start_index:end_index + 1] = 'tumble'
            for point_index in range(start_index, end_index + 1):
                event_ids[point_index] = next_event_id

            preceding_heading_indices = valid_heading_indices[
                valid_heading_indices < start_index
            ]
            support_start_index = (
                int(preceding_heading_indices[-1])
                if len(preceding_heading_indices)
                else max(start_index - 1, 0)
            )
            event_speeds = speeds[start_index:end_index + 1]
            event_speeds = event_speeds[np.isfinite(event_speeds)]
            event_angular_velocities = angular_velocity_magnitudes[
                start_index:end_index + 1
            ]
            event_angular_velocities = event_angular_velocities[
                np.isfinite(event_angular_velocities)
            ]
            event_path_length = float(np.sum(np.linalg.norm(
                np.diff(
                    smoothed_positions[start_index:end_index + 1],
                    axis=0
                ) * distance_factor,
                axis=1
            ))) if end_index > start_index else 0.0
            event_rows.append({
                'source_dataframe': self._resolved_source_dataframe,
                'particle': particle_id,
                'segment_id': segment_id,
                'event_id': next_event_id,
                'event_type': 'tumble',
                'start_frame': int(frames[start_index]),
                'end_frame': int(frames[end_index]),
                'support_start_frame': int(frames[support_start_index]),
                'point_count': int(end_index - start_index + 1),
                'interval_seconds': float(
                    absolute_times[end_index] - absolute_times[start_index]
                ),
                'sampling_support_seconds': float(
                    absolute_times[end_index] -
                    absolute_times[support_start_index]
                ),
                'path_length': event_path_length,
                'mean_speed': self.__safe_mean(event_speeds),
                'minimum_speed': (
                    float(np.min(event_speeds))
                    if len(event_speeds) else np.nan
                ),
                'maximum_speed': (
                    float(np.max(event_speeds))
                    if len(event_speeds) else np.nan
                ),
                'mean_angular_velocity_magnitude_radians_per_second': (
                    self.__safe_mean(event_angular_velocities)
                ),
                'maximum_angular_velocity_magnitude_radians_per_second': (
                    float(np.max(event_angular_velocities))
                    if len(event_angular_velocities) else np.nan
                ),
                'speed_minimum_frame': int(
                    frames[speed_candidate['index']]
                ),
                'speed_minimum_value': speed_candidate['minimum_speed'],
                'speed_minimum_depth': speed_candidate['depth'],
                'speed_depth_ratio': speed_candidate['depth_ratio'],
                'speed_left_maximum_frame': int(
                    frames[speed_candidate['left_maximum']]
                ),
                'speed_right_maximum_frame': int(
                    frames[speed_candidate['right_maximum']]
                ),
                'speed_period_start_frame': int(
                    frames[speed_candidate['period_start']]
                ),
                'speed_period_end_frame': int(
                    frames[speed_candidate['period_end']]
                ),
                'angular_velocity_maximum_frame': int(
                    frames[angular_candidate['index']]
                ),
                'angular_velocity_maximum_value': (
                    angular_candidate['maximum_value']
                ),
                'angular_velocity_maximum_height': (
                    angular_candidate['height']
                ),
                'angular_left_minimum_frame': int(
                    frames[angular_candidate['left_minimum']]
                ),
                'angular_right_minimum_frame': int(
                    frames[angular_candidate['right_minimum']]
                ),
                'angular_total_directional_change_radians': (
                    angular_candidate['total_directional_change']
                ),
                'angular_directional_change_threshold_radians': (
                    angular_candidate['directional_change_threshold']
                ),
                'angular_period_start_frame': int(frames[start_index]),
                'angular_period_end_frame': int(frames[end_index]),
                'matching_overlap_start_frame': int(
                    frames[representative_match['overlap_start']]
                ),
                'matching_overlap_end_frame': int(
                    frames[representative_match['overlap_end']]
                ),
                'merged_candidate_count': int(
                    len(candidate_group['matches'])
                ),
                'evidence_scope': 'representative_matched_candidate',
                'preceding_run_fit_start_frame': np.nan,
                'preceding_run_fit_end_frame': np.nan,
                'following_run_fit_start_frame': np.nan,
                'following_run_fit_end_frame': np.nan,
                'preceding_run_direction_radians': np.nan,
                'following_run_direction_radians': np.nan,
                'run_to_run_turn_angle_radians': np.nan,
                'run_to_run_turn_angle_degrees': np.nan,
                'run_to_run_directional_cosine': np.nan,
                'distance_unit': distance_unit,
                'speed_unit': speed_unit,
                'angular_velocity_unit': 'rad/s',
                'smoothing_method': smoothing_method,
                'effective_smoothing_window': (
                    effective_window
                    if effective_window is not None
                    else np.nan
                ),
                'speed_drop_ratio_threshold': (
                    speed_drop_ratio_threshold
                ),
                'speed_period_depth_fraction': (
                    speed_period_depth_fraction
                ),
                'angular_change_coefficient': angular_change_coefficient,
                'minimum_heading_speed': minimum_heading_speed,
            })
            next_event_id += 1

        point_rows = []
        for point_index in range(point_count):
            point_rows.append({
                'source_dataframe': self._resolved_source_dataframe,
                'particle': particle_id,
                'segment_id': segment_id,
                'frame': int(frames[point_index]),
                'previous_frame': (
                    int(frames[point_index - 1])
                    if point_index > 0 else None
                ),
                'frame_delta_from_previous': (
                    int(frames[point_index] - frames[point_index - 1])
                    if point_index > 0 else np.nan
                ),
                'elapsed_time_from_previous_seconds': (
                    float(
                        absolute_times[point_index] -
                        absolute_times[point_index - 1]
                    )
                    if point_index > 0 else np.nan
                ),
                'elapsed_time_seconds': float(elapsed_times[point_index]),
                'raw_position_x_pixels': float(
                    raw_positions[point_index, 0]
                ),
                'raw_position_y_pixels': float(
                    raw_positions[point_index, 1]
                ),
                'smoothed_position_x_pixels': float(
                    smoothed_positions[point_index, 0]
                ),
                'smoothed_position_y_pixels': float(
                    smoothed_positions[point_index, 1]
                ),
                'velocity_x': float(velocities[point_index, 0]),
                'velocity_y': float(velocities[point_index, 1]),
                'speed': float(speeds[point_index]),
                'heading_radians': float(headings[point_index]),
                'previous_valid_heading_frame': float(
                    previous_valid_heading_frames[point_index]
                ),
                'angular_elapsed_time_seconds': float(
                    angular_elapsed_times[point_index]
                ),
                'angular_change_radians': float(
                    angular_changes[point_index]
                ),
                'angular_velocity_magnitude_radians_per_second': float(
                    angular_velocity_magnitudes[point_index]
                ),
                'persistence_target_elapsed_time_seconds': np.nan,
                'persistence_target_frame': np.nan,
                'persistence_direction_change_radians': np.nan,
                'persistence_cosine': np.nan,
                'is_speed_minimum': bool(
                    speed_minimum_flags[point_index]
                ),
                'speed_minimum_passes': bool(
                    speed_minimum_pass_flags[point_index]
                ),
                'is_angular_velocity_maximum': bool(
                    angular_maximum_flags[point_index]
                ),
                'angular_velocity_maximum_passes': bool(
                    angular_maximum_pass_flags[point_index]
                ),
                'state': str(states[point_index]),
                'event_id': event_ids[point_index],
                'distance_unit': distance_unit,
                'speed_unit': speed_unit,
                'angular_velocity_unit': 'rad/s',
                'smoothing_method': smoothing_method,
                'effective_smoothing_window': (
                    effective_window
                    if effective_window is not None
                    else np.nan
                ),
            })
        return point_rows, event_rows, next_event_id

    def __calculate_velocity_tumble_table_metrics(
        self,
        points_dataframe: pd.DataFrame,
        events_dataframe: pd.DataFrame,
        persistence_interval_seconds: float,
        run_direction_fit_points: int
    ) -> dict:
        """Calculate the Najafi Table 1 speed and direction quantities."""
        point_annotation_columns = [
            'persistence_target_elapsed_time_seconds',
            'persistence_target_frame',
            'persistence_direction_change_radians',
            'persistence_cosine',
        ]
        event_annotation_columns = [
            'preceding_run_fit_start_frame',
            'preceding_run_fit_end_frame',
            'following_run_fit_start_frame',
            'following_run_fit_end_frame',
            'preceding_run_direction_radians',
            'following_run_direction_radians',
            'run_to_run_turn_angle_radians',
            'run_to_run_turn_angle_degrees',
            'run_to_run_directional_cosine',
        ]
        for column_name in point_annotation_columns:
            points_dataframe[column_name] = np.nan
        for column_name in event_annotation_columns:
            events_dataframe[column_name] = np.nan

        run_speed_values = np.array([], dtype=float)
        run_speed_durations = np.array([], dtype=float)
        tumble_speed_values = np.array([], dtype=float)
        tumble_speed_durations = np.array([], dtype=float)
        if not points_dataframe.empty:
            speeds = pd.to_numeric(
                points_dataframe['speed'], errors='coerce'
            ).to_numpy(dtype=float)
            step_durations = pd.to_numeric(
                points_dataframe[
                    'elapsed_time_from_previous_seconds'
                ],
                errors='coerce'
            ).to_numpy(dtype=float)
            states = points_dataframe['state'].astype(str).to_numpy()
            previous_states = np.empty(len(states), dtype=object)
            previous_states[0] = ''
            previous_states[1:] = states[:-1]
            segment_ids = pd.to_numeric(
                points_dataframe['segment_id'], errors='coerce'
            ).to_numpy(dtype=float)
            previous_segment_ids = np.full(len(segment_ids), np.nan)
            previous_segment_ids[1:] = segment_ids[:-1]
            valid_step = (
                np.isfinite(speeds) &
                np.isfinite(step_durations) &
                (step_durations > 0) &
                (states == previous_states) &
                (segment_ids == previous_segment_ids)
            )
            run_speed_mask = valid_step & (states == 'run')
            tumble_speed_mask = valid_step & (states == 'tumble')
            run_speed_values = speeds[run_speed_mask]
            run_speed_durations = step_durations[run_speed_mask]
            tumble_speed_values = speeds[tumble_speed_mask]
            tumble_speed_durations = step_durations[tumble_speed_mask]

        def time_weighted_mean(
            values: np.ndarray,
            durations: np.ndarray
        ) -> float:
            total_duration = float(np.sum(durations))
            if total_duration <= 0:
                return np.nan
            return float(np.sum(values * durations) / total_duration)

        persistence_cosines = []
        if not points_dataframe.empty:
            grouped_segments = points_dataframe.groupby(
                'segment_id', sort=True
            )
            for _, segment_points in grouped_segments:
                segment_points = segment_points.sort_values(
                    by='frame', kind='stable'
                )
                segment_indices = segment_points.index.to_numpy(dtype=int)
                segment_states = (
                    segment_points['state'].astype(str).to_numpy()
                )
                segment_previous_states = np.empty(
                    len(segment_states), dtype=object
                )
                segment_previous_states[0] = ''
                segment_previous_states[1:] = segment_states[:-1]
                headings = pd.to_numeric(
                    segment_points['heading_radians'], errors='coerce'
                ).to_numpy(dtype=float)
                finite_run_heading = (
                    (segment_states == 'run') &
                    (segment_previous_states == 'run') &
                    np.isfinite(headings)
                )
                finite_indices = np.flatnonzero(finite_run_heading)
                if not len(finite_indices):
                    continue
                block_starts = np.r_[
                    0, np.flatnonzero(np.diff(finite_indices) > 1) + 1
                ]
                block_ends = np.r_[
                    block_starts[1:], len(finite_indices)
                ]
                for block_start, block_end in zip(
                    block_starts, block_ends
                ):
                    local_indices = finite_indices[
                        block_start:block_end
                    ]
                    if len(local_indices) < 2:
                        continue
                    block_points = segment_points.iloc[local_indices]
                    block_times = pd.to_numeric(
                        block_points['elapsed_time_seconds'],
                        errors='coerce'
                    ).to_numpy(dtype=float)
                    block_frames = pd.to_numeric(
                        block_points['frame'], errors='coerce'
                    ).to_numpy(dtype=float)
                    block_headings = np.unwrap(
                        headings[local_indices]
                    )
                    if (
                        not np.all(np.isfinite(block_times)) or
                        np.any(np.diff(block_times) <= 0)
                    ):
                        continue
                    time_tolerance = max(
                        np.finfo(float).eps,
                        float(np.max(np.abs(block_times))) * 1e-12
                    )
                    for source_index, source_time in enumerate(block_times):
                        target_time = (
                            float(source_time) +
                            persistence_interval_seconds
                        )
                        if (
                            target_time >
                            float(block_times[-1]) + time_tolerance
                        ):
                            continue
                        target_heading = float(np.interp(
                            target_time, block_times, block_headings
                        ))
                        signed_change = float(np.arctan2(
                            np.sin(
                                target_heading -
                                block_headings[source_index]
                            ),
                            np.cos(
                                target_heading -
                                block_headings[source_index]
                            )
                        ))
                        persistence_cosine = float(
                            np.cos(signed_change)
                        )
                        dataframe_index = int(
                            segment_indices[local_indices[source_index]]
                        )
                        points_dataframe.at[
                            dataframe_index,
                            'persistence_target_elapsed_time_seconds'
                        ] = target_time
                        points_dataframe.at[
                            dataframe_index,
                            'persistence_target_frame'
                        ] = float(np.interp(
                            target_time, block_times, block_frames
                        ))
                        points_dataframe.at[
                            dataframe_index,
                            'persistence_direction_change_radians'
                        ] = signed_change
                        points_dataframe.at[
                            dataframe_index,
                            'persistence_cosine'
                        ] = persistence_cosine
                        persistence_cosines.append(persistence_cosine)

        run_transition_cosines = []
        if not points_dataframe.empty and not events_dataframe.empty:
            for event_index, event_row in events_dataframe.iterrows():
                segment_points = points_dataframe.loc[
                    points_dataframe['segment_id'] ==
                    event_row['segment_id']
                ].sort_values(by='frame', kind='stable')
                if segment_points.empty:
                    continue
                segment_frames = pd.to_numeric(
                    segment_points['frame'], errors='coerce'
                ).to_numpy(dtype=float)
                segment_states = (
                    segment_points['state'].astype(str).to_numpy()
                )
                start_positions = np.flatnonzero(
                    segment_frames == float(event_row['start_frame'])
                )
                end_positions = np.flatnonzero(
                    segment_frames == float(event_row['end_frame'])
                )
                if not len(start_positions) or not len(end_positions):
                    continue
                preceding_end = int(start_positions[0]) - 1
                following_start = int(end_positions[-1]) + 1
                if (
                    preceding_end < 0 or
                    following_start >= len(segment_points)
                ):
                    continue
                preceding_start = preceding_end
                while (
                    preceding_start > 0 and
                    segment_states[preceding_start - 1] == 'run'
                ):
                    preceding_start -= 1
                following_end = following_start
                while (
                    following_end + 1 < len(segment_points) and
                    segment_states[following_end + 1] == 'run'
                ):
                    following_end += 1
                if (
                    segment_states[preceding_end] != 'run' or
                    segment_states[following_start] != 'run'
                ):
                    continue
                preceding_points = segment_points.iloc[
                    max(
                        preceding_start,
                        preceding_end - run_direction_fit_points + 1
                    ):preceding_end + 1
                ]
                following_points = segment_points.iloc[
                    following_start:min(
                        following_end + 1,
                        following_start + run_direction_fit_points
                    )
                ]
                if (
                    len(preceding_points) < run_direction_fit_points or
                    len(following_points) < run_direction_fit_points
                ):
                    continue

                fitted_directions = []
                fit_is_valid = True
                for fit_points in (
                    preceding_points, following_points
                ):
                    fit_times = pd.to_numeric(
                        fit_points['elapsed_time_seconds'],
                        errors='coerce'
                    ).to_numpy(dtype=float)
                    fit_positions = fit_points[[
                        'smoothed_position_x_pixels',
                        'smoothed_position_y_pixels'
                    ]].to_numpy(dtype=float)
                    if (
                        not np.all(np.isfinite(fit_times)) or
                        not np.all(np.isfinite(fit_positions)) or
                        np.ptp(fit_times) <= 0
                    ):
                        fit_is_valid = False
                        break
                    centered_times = fit_times - np.mean(fit_times)
                    denominator = float(np.sum(centered_times ** 2))
                    if denominator <= 0:
                        fit_is_valid = False
                        break
                    slopes = (
                        centered_times[:, None] *
                        (
                            fit_positions -
                            np.mean(fit_positions, axis=0)
                        )
                    ).sum(axis=0) / denominator
                    slope_magnitude = float(np.linalg.norm(slopes))
                    position_scale = max(
                        1.0,
                        float(np.nanmax(np.abs(fit_positions)))
                    )
                    if (
                        not np.isfinite(slope_magnitude) or
                        slope_magnitude <=
                        np.finfo(float).eps * position_scale
                    ):
                        fit_is_valid = False
                        break
                    fitted_directions.append(float(np.arctan2(
                        slopes[1], slopes[0]
                    )))
                if not fit_is_valid:
                    continue
                turn_angle = float(np.arctan2(
                    np.sin(fitted_directions[1] - fitted_directions[0]),
                    np.cos(fitted_directions[1] - fitted_directions[0])
                ))
                directional_cosine = float(np.cos(turn_angle))
                events_dataframe.at[
                    event_index, 'preceding_run_fit_start_frame'
                ] = float(preceding_points['frame'].iloc[0])
                events_dataframe.at[
                    event_index, 'preceding_run_fit_end_frame'
                ] = float(preceding_points['frame'].iloc[-1])
                events_dataframe.at[
                    event_index, 'following_run_fit_start_frame'
                ] = float(following_points['frame'].iloc[0])
                events_dataframe.at[
                    event_index, 'following_run_fit_end_frame'
                ] = float(following_points['frame'].iloc[-1])
                events_dataframe.at[
                    event_index, 'preceding_run_direction_radians'
                ] = fitted_directions[0]
                events_dataframe.at[
                    event_index, 'following_run_direction_radians'
                ] = fitted_directions[1]
                events_dataframe.at[
                    event_index, 'run_to_run_turn_angle_radians'
                ] = turn_angle
                events_dataframe.at[
                    event_index, 'run_to_run_turn_angle_degrees'
                ] = float(np.degrees(turn_angle))
                events_dataframe.at[
                    event_index, 'run_to_run_directional_cosine'
                ] = directional_cosine
                run_transition_cosines.append(directional_cosine)

        return {
            'time_weighted_vR': time_weighted_mean(
                run_speed_values, run_speed_durations
            ),
            'vR_interval_count': int(len(run_speed_values)),
            'vR_support_seconds': float(np.sum(run_speed_durations)),
            'time_weighted_vT': time_weighted_mean(
                tumble_speed_values, tumble_speed_durations
            ),
            'vT_interval_count': int(len(tumble_speed_values)),
            'vT_support_seconds': float(
                np.sum(tumble_speed_durations)
            ),
            'p': self.__safe_mean(
                np.asarray(persistence_cosines, dtype=float)
            ),
            'std_p': self.__safe_std(
                np.asarray(persistence_cosines, dtype=float)
            ),
            'p_direction_change_count': int(
                len(persistence_cosines)
            ),
            'R': self.__safe_mean(
                np.asarray(run_transition_cosines, dtype=float)
            ),
            'std_R': self.__safe_std(
                np.asarray(run_transition_cosines, dtype=float)
            ),
            'R_run_transition_count': int(
                len(run_transition_cosines)
            ),
        }

    def __summarize_velocity_tumble_particle(
        self,
        particle_id: int,
        original_frames: np.ndarray,
        analyzed_frames: np.ndarray,
        discarded_count: int,
        segment_records: list[dict],
        point_rows: list[dict],
        event_rows: list[dict],
        distance_unit: str,
        speed_unit: str,
        smoothing_configuration: dict,
        discard_initial_frames: int,
        discard_final_frames: int,
        speed_drop_ratio_threshold: float,
        speed_period_depth_fraction: float,
        angular_change_coefficient: float,
        minimum_heading_speed: float | None,
        speed_extrema_prominence: float | None,
        angular_velocity_extrema_prominence: float | None,
        extrema_min_distance: int,
        persistence_interval_seconds: float,
        run_direction_fit_points: int,
        max_frame_gap: int | None
    ) -> dict:
        """Summarize one particle's speed/angular tumble classification."""
        points_dataframe = pd.DataFrame(
            point_rows, columns=self.VELOCITY_TUMBLE_POINT_COLUMNS
        )
        events_dataframe = pd.DataFrame(
            event_rows, columns=self.VELOCITY_TUMBLE_EVENT_COLUMNS
        )
        table_metrics = self.__calculate_velocity_tumble_table_metrics(
            points_dataframe=points_dataframe,
            events_dataframe=events_dataframe,
            persistence_interval_seconds=persistence_interval_seconds,
            run_direction_fit_points=run_direction_fit_points
        )
        point_annotation_columns = [
            'persistence_target_elapsed_time_seconds',
            'persistence_target_frame',
            'persistence_direction_change_radians',
            'persistence_cosine',
        ]
        for original_row, annotations in zip(
            point_rows,
            points_dataframe[
                point_annotation_columns
            ].to_dict(orient='records')
        ):
            original_row.update(annotations)
        event_annotation_columns = [
            'preceding_run_fit_start_frame',
            'preceding_run_fit_end_frame',
            'following_run_fit_start_frame',
            'following_run_fit_end_frame',
            'preceding_run_direction_radians',
            'following_run_direction_radians',
            'run_to_run_turn_angle_radians',
            'run_to_run_turn_angle_degrees',
            'run_to_run_directional_cosine',
        ]
        for original_row, annotations in zip(
            event_rows,
            events_dataframe[
                event_annotation_columns
            ].to_dict(orient='records')
        ):
            original_row.update(annotations)
        analyzed_tracking_time = float(sum(
            record['duration_seconds'] for record in segment_records
        ))
        classified_tracking_time = float(sum(
            record['classified_support_duration_seconds']
            for record in segment_records
            if record['is_analyzable']
        ))
        analyzable_segment_count = int(sum(
            bool(record['is_analyzable']) for record in segment_records
        ))
        total_unclassified_interval = max(
            analyzed_tracking_time - classified_tracking_time,
            0.0
        )
        all_tumble_intervals = (
            events_dataframe['interval_seconds'].to_numpy(dtype=float)
            if not events_dataframe.empty else np.array([], dtype=float)
        )
        total_tumble_interval = float(np.sum(all_tumble_intervals))

        run_intervals = []
        complete_run_intervals = []
        censored_run_interval_count = 0
        complete_tumble_intervals = []
        censored_tumble_interval_count = 0
        time_between_tumble_starts = []
        for segment_record in segment_records:
            if not segment_record['is_analyzable']:
                continue
            segment_start = (
                float(segment_record['start_frame']) /
                self._capture_speed_in_fps
            )
            segment_end = (
                float(segment_record['end_frame']) /
                self._capture_speed_in_fps
            )
            segment_events = (
                events_dataframe.loc[
                    events_dataframe['segment_id'] ==
                    segment_record['segment_id']
                ].sort_values(by='start_frame', kind='stable')
                if not events_dataframe.empty
                else events_dataframe
            )
            cursor = segment_start
            previous_event_was_complete = False
            for event_position, (_, event_row) in enumerate(
                segment_events.iterrows()
            ):
                event_start = (
                    float(event_row['start_frame']) /
                    self._capture_speed_in_fps
                )
                event_end = (
                    float(event_row['end_frame']) /
                    self._capture_speed_in_fps
                )
                event_is_complete = bool(
                    event_start > segment_start and
                    event_end < segment_end
                )
                if event_start > cursor:
                    run_interval = event_start - cursor
                    run_intervals.append(run_interval)
                    if (
                        event_position > 0 and
                        previous_event_was_complete and
                        event_is_complete
                    ):
                        complete_run_intervals.append(run_interval)
                    else:
                        censored_run_interval_count += 1
                if event_is_complete:
                    complete_tumble_intervals.append(
                        float(event_row['interval_seconds'])
                    )
                else:
                    censored_tumble_interval_count += 1
                cursor = max(cursor, event_end)
                previous_event_was_complete = event_is_complete
            if segment_end > cursor:
                run_intervals.append(segment_end - cursor)
                censored_run_interval_count += 1
            elif not len(segment_events) and segment_end == cursor:
                censored_run_interval_count += 1
            if len(segment_events) > 1:
                event_start_times = (
                    segment_events['start_frame'].to_numpy(dtype=float) /
                    self._capture_speed_in_fps
                )
                time_between_tumble_starts.extend(
                    np.diff(event_start_times).tolist()
                )

        run_intervals_array = np.asarray(run_intervals, dtype=float)
        complete_run_intervals_array = np.asarray(
            complete_run_intervals, dtype=float
        )
        complete_tumble_intervals_array = np.asarray(
            complete_tumble_intervals, dtype=float
        )
        time_between_tumble_starts_array = np.asarray(
            time_between_tumble_starts, dtype=float
        )
        total_run_interval = float(np.sum(run_intervals_array))
        if not points_dataframe.empty:
            speeds = pd.to_numeric(
                points_dataframe['speed'], errors='coerce'
            ).to_numpy(dtype=float)
            angular_velocities = pd.to_numeric(
                points_dataframe[
                    'angular_velocity_magnitude_radians_per_second'
                ],
                errors='coerce'
            ).to_numpy(dtype=float)
            states = points_dataframe['state'].to_numpy(dtype=str)
            run_speeds = speeds[
                (states == 'run') & np.isfinite(speeds)
            ]
            tumble_speeds = speeds[
                (states == 'tumble') & np.isfinite(speeds)
            ]
            run_angular_velocities = angular_velocities[
                (states == 'run') & np.isfinite(angular_velocities)
            ]
            tumble_angular_velocities = angular_velocities[
                (states == 'tumble') & np.isfinite(angular_velocities)
            ]
            kinematic_point_count = int(
                np.sum(np.isfinite(angular_velocities))
            )
            unclassified_detection_count = int(
                np.sum(states == 'unclassified')
            )
        else:
            run_speeds = np.array([], dtype=float)
            tumble_speeds = np.array([], dtype=float)
            run_angular_velocities = np.array([], dtype=float)
            tumble_angular_velocities = np.array([], dtype=float)
            kinematic_point_count = 0
            unclassified_detection_count = 0

        number_of_tumbles = int(len(events_dataframe))
        number_of_runs = int(len(run_intervals_array))
        if not len(analyzed_frames):
            analysis_status = 'no_detections_after_discard'
        elif not analyzable_segment_count:
            analysis_status = 'insufficient_kinematic_points_for_extrema'
        elif number_of_tumbles:
            analysis_status = 'analyzed_with_tumbles'
        else:
            analysis_status = 'analyzed_no_tumbles'
        smoothing_output = self.__smoothing_configuration_output(
            smoothing_configuration
        )
        return {
            'source_dataframe': self._resolved_source_dataframe,
            'particle': particle_id,
            'original_first_frame': (
                int(original_frames[0]) if len(original_frames) else np.nan
            ),
            'original_last_frame': (
                int(original_frames[-1]) if len(original_frames) else np.nan
            ),
            'original_detection_count': int(len(original_frames)),
            'discarded_detection_count': int(discarded_count),
            'analyzed_first_frame': (
                int(analyzed_frames[0]) if len(analyzed_frames) else np.nan
            ),
            'analyzed_last_frame': (
                int(analyzed_frames[-1]) if len(analyzed_frames) else np.nan
            ),
            'analyzed_detection_count': int(len(analyzed_frames)),
            'segment_count': int(len(segment_records)),
            'analyzable_segment_count': analyzable_segment_count,
            'kinematic_point_count': kinematic_point_count,
            'unclassified_detection_count': (
                unclassified_detection_count
            ),
            'number_of_runs': number_of_runs,
            'number_of_tumbles': number_of_tumbles,
            'runs_per_tumble': (
                number_of_runs / number_of_tumbles
                if number_of_tumbles else np.nan
            ),
            'analyzed_tracking_time_seconds': analyzed_tracking_time,
            'classified_tracking_time_seconds': classified_tracking_time,
            'total_run_interval_seconds': total_run_interval,
            'total_tumble_interval_seconds': total_tumble_interval,
            'total_unclassified_interval_seconds': (
                total_unclassified_interval
            ),
            'tumble_time_fraction': (
                total_tumble_interval / classified_tracking_time
                if classified_tracking_time > 0 else np.nan
            ),
            'tumble_frequency_per_second': (
                number_of_tumbles / classified_tracking_time
                if classified_tracking_time > 0 else np.nan
            ),
            'point_mean_run_speed': self.__safe_mean(run_speeds),
            'point_std_run_speed': self.__safe_std(run_speeds),
            'point_mean_tumble_speed': self.__safe_mean(tumble_speeds),
            'point_std_tumble_speed': self.__safe_std(tumble_speeds),
            'mean_run_interval_seconds': self.__safe_mean(
                run_intervals_array
            ),
            'std_run_interval_seconds': self.__safe_std(
                run_intervals_array
            ),
            'mean_tumble_interval_seconds': self.__safe_mean(
                all_tumble_intervals
            ),
            'std_tumble_interval_seconds': self.__safe_std(
                all_tumble_intervals
            ),
            'mean_time_between_tumble_starts_seconds': self.__safe_mean(
                time_between_tumble_starts_array
            ),
            'std_time_between_tumble_starts_seconds': self.__safe_std(
                time_between_tumble_starts_array
            ),
            'mean_run_angular_velocity_magnitude_radians_per_second': (
                self.__safe_mean(run_angular_velocities)
            ),
            'std_run_angular_velocity_magnitude_radians_per_second': (
                self.__safe_std(run_angular_velocities)
            ),
            'mean_tumble_angular_velocity_magnitude_radians_per_second': (
                self.__safe_mean(tumble_angular_velocities)
            ),
            'std_tumble_angular_velocity_magnitude_radians_per_second': (
                self.__safe_std(tumble_angular_velocities)
            ),
            'time_weighted_vR': table_metrics['time_weighted_vR'],
            'vR_interval_count': table_metrics['vR_interval_count'],
            'vR_support_seconds': table_metrics['vR_support_seconds'],
            'time_weighted_vT': table_metrics['time_weighted_vT'],
            'vT_interval_count': table_metrics['vT_interval_count'],
            'vT_support_seconds': table_metrics['vT_support_seconds'],
            't_R': self.__safe_mean(complete_run_intervals_array),
            'std_t_R': self.__safe_std(complete_run_intervals_array),
            'sem_t_R': self.__safe_sem(complete_run_intervals_array),
            't_R_interval_count': int(
                len(complete_run_intervals_array)
            ),
            't_R_censored_interval_count': int(
                censored_run_interval_count
            ),
            't_T': self.__safe_mean(complete_tumble_intervals_array),
            'std_t_T': self.__safe_std(complete_tumble_intervals_array),
            't_T_interval_count': int(
                len(complete_tumble_intervals_array)
            ),
            't_T_censored_interval_count': int(
                censored_tumble_interval_count
            ),
            'p': table_metrics['p'],
            'std_p': table_metrics['std_p'],
            'p_direction_change_count': table_metrics[
                'p_direction_change_count'
            ],
            'R': table_metrics['R'],
            'std_R': table_metrics['std_R'],
            'R_run_transition_count': table_metrics[
                'R_run_transition_count'
            ],
            'distance_unit': distance_unit,
            'speed_unit': speed_unit,
            'angular_velocity_unit': 'rad/s',
            **smoothing_output,
            'discard_initial_frames': discard_initial_frames,
            'discard_final_frames': discard_final_frames,
            'speed_drop_ratio_threshold': speed_drop_ratio_threshold,
            'speed_period_depth_fraction': speed_period_depth_fraction,
            'angular_change_coefficient': angular_change_coefficient,
            'minimum_heading_speed': minimum_heading_speed,
            'speed_extrema_prominence': speed_extrema_prominence,
            'angular_velocity_extrema_prominence': (
                angular_velocity_extrema_prominence
            ),
            'extrema_min_distance': extrema_min_distance,
            'persistence_interval_seconds': (
                persistence_interval_seconds
            ),
            'run_direction_fit_points': run_direction_fit_points,
            'max_frame_gap': max_frame_gap,
            'analysis_status': analysis_status,
        }

    def __summarize_velocity_tumble_population(
        self,
        tumble_summary: pd.DataFrame,
        distance_unit: str,
        speed_unit: str,
        smoothing_configuration: dict,
        discard_initial_frames: int,
        discard_final_frames: int,
        speed_drop_ratio_threshold: float,
        speed_period_depth_fraction: float,
        angular_change_coefficient: float,
        minimum_heading_speed: float | None,
        speed_extrema_prominence: float | None,
        angular_velocity_extrema_prominence: float | None,
        extrema_min_distance: int,
        persistence_interval_seconds: float,
        run_direction_fit_points: int,
        max_frame_gap: int | None,
        bootstrap_resamples: int = 10_000,
        bootstrap_confidence_level: float = 0.95,
        bootstrap_random_seed: int | None = 0
    ) -> pd.DataFrame:
        """Build pooled Table 1 and equal-particle velocity summaries."""
        def metric_values(column_name: str) -> np.ndarray:
            return self.__finite_numeric_values(tumble_summary[column_name])

        def pooled_metric(
            value_column: str,
            weight_column: str
        ) -> float:
            values = pd.to_numeric(
                tumble_summary[value_column], errors='coerce'
            ).to_numpy(dtype=float)
            weights = pd.to_numeric(
                tumble_summary[weight_column], errors='coerce'
            ).to_numpy(dtype=float)
            valid = (
                np.isfinite(values) &
                np.isfinite(weights) &
                (weights > 0)
            )
            if not valid.any():
                return np.nan
            return float(
                np.sum(values[valid] * weights[valid]) /
                np.sum(weights[valid])
            )

        def total_support(column_name: str) -> float:
            return float(np.sum(metric_values(column_name)))

        def contributing_particle_count(
            value_column: str,
            weight_column: str
        ) -> int:
            values = pd.to_numeric(
                tumble_summary[value_column], errors='coerce'
            ).to_numpy(dtype=float)
            weights = pd.to_numeric(
                tumble_summary[weight_column], errors='coerce'
            ).to_numpy(dtype=float)
            return int(np.sum(
                np.isfinite(values) & np.isfinite(weights) & (weights > 0)
            ))

        bootstrap_seed_sequence = np.random.SeedSequence(
            bootstrap_random_seed
        )
        run_bootstrap_seed, tumble_bootstrap_seed = (
            bootstrap_seed_sequence.spawn(2)
        )
        v_R_ci_lower, v_R_ci_upper = (
            self.__particle_cluster_bootstrap_weighted_mean_ci(
                values=pd.to_numeric(
                    tumble_summary['time_weighted_vR'], errors='coerce'
                ).to_numpy(dtype=float),
                weights=pd.to_numeric(
                    tumble_summary['vR_support_seconds'], errors='coerce'
                ).to_numpy(dtype=float),
                resamples=bootstrap_resamples,
                confidence_level=bootstrap_confidence_level,
                random_generator=np.random.default_rng(run_bootstrap_seed)
            )
        )
        v_T_ci_lower, v_T_ci_upper = (
            self.__particle_cluster_bootstrap_weighted_mean_ci(
                values=pd.to_numeric(
                    tumble_summary['time_weighted_vT'], errors='coerce'
                ).to_numpy(dtype=float),
                weights=pd.to_numeric(
                    tumble_summary['vT_support_seconds'], errors='coerce'
                ).to_numpy(dtype=float),
                resamples=bootstrap_resamples,
                confidence_level=bootstrap_confidence_level,
                random_generator=np.random.default_rng(tumble_bootstrap_seed)
            )
        )

        particle_count = int(len(tumble_summary))
        total_runs = int(
            pd.to_numeric(
                tumble_summary['number_of_runs'], errors='coerce'
            ).fillna(0).sum()
        )
        total_tumbles = int(
            pd.to_numeric(
                tumble_summary['number_of_tumbles'], errors='coerce'
            ).fillna(0).sum()
        )
        tracking_times = metric_values(
            'analyzed_tracking_time_seconds'
        )
        total_tracking_time = float(np.sum(tracking_times))
        total_classified_tracking_time = float(np.sum(metric_values(
            'classified_tracking_time_seconds'
        )))
        total_unclassified_interval = float(np.sum(metric_values(
            'total_unclassified_interval_seconds'
        )))
        (
            pooled_t_R,
            pooled_t_R_std,
            pooled_t_R_sem,
            pooled_t_R_count,
        ) = self.__combine_group_mean_std_sem(
            pd.to_numeric(
                tumble_summary['t_R'], errors='coerce'
            ).to_numpy(dtype=float),
            pd.to_numeric(
                tumble_summary['std_t_R'], errors='coerce'
            ).to_numpy(dtype=float),
            pd.to_numeric(
                tumble_summary['t_R_interval_count'], errors='coerce'
            ).to_numpy(dtype=float)
        )
        (
            pooled_t_T,
            pooled_t_T_std,
            _,
            pooled_t_T_count,
        ) = self.__combine_group_mean_std_sem(
            pd.to_numeric(
                tumble_summary['t_T'], errors='coerce'
            ).to_numpy(dtype=float),
            pd.to_numeric(
                tumble_summary['std_t_T'], errors='coerce'
            ).to_numpy(dtype=float),
            pd.to_numeric(
                tumble_summary['t_T_interval_count'], errors='coerce'
            ).to_numpy(dtype=float)
        )
        (
            pooled_p,
            pooled_p_std,
            _,
            pooled_p_count,
        ) = self.__combine_group_mean_std_sem(
            pd.to_numeric(
                tumble_summary['p'], errors='coerce'
            ).to_numpy(dtype=float),
            pd.to_numeric(
                tumble_summary['std_p'], errors='coerce'
            ).to_numpy(dtype=float),
            pd.to_numeric(
                tumble_summary['p_direction_change_count'], errors='coerce'
            ).to_numpy(dtype=float)
        )
        (
            pooled_R,
            pooled_R_std,
            _,
            pooled_R_count,
        ) = self.__combine_group_mean_std_sem(
            pd.to_numeric(
                tumble_summary['R'], errors='coerce'
            ).to_numpy(dtype=float),
            pd.to_numeric(
                tumble_summary['std_R'], errors='coerce'
            ).to_numpy(dtype=float),
            pd.to_numeric(
                tumble_summary['R_run_transition_count'], errors='coerce'
            ).to_numpy(dtype=float)
        )
        smoothing_output = self.__smoothing_configuration_output(
            smoothing_configuration
        )
        population_row = {
            'source_dataframe': self._resolved_source_dataframe,
            'particle_count': particle_count,
            'particles_with_kinematic_points': int(
                (
                    pd.to_numeric(
                        tumble_summary['kinematic_point_count'],
                        errors='coerce'
                    ).fillna(0) > 0
                ).sum()
            ),
            'particles_with_tumbles': int(
                (
                    pd.to_numeric(
                        tumble_summary['number_of_tumbles'],
                        errors='coerce'
                    ).fillna(0) > 0
                ).sum()
            ),
            'total_number_of_runs': total_runs,
            'total_number_of_tumbles': total_tumbles,
            'total_analyzed_tracking_time_seconds': total_tracking_time,
            'total_classified_tracking_time_seconds': (
                total_classified_tracking_time
            ),
            'total_unclassified_interval_seconds': (
                total_unclassified_interval
            ),
            'pooled_tumble_frequency_per_second': (
                total_tumbles / total_classified_tracking_time
                if total_classified_tracking_time > 0 else np.nan
            ),
            'mean_particle_tumble_frequency_per_second': self.__safe_mean(
                metric_values('tumble_frequency_per_second')
            ),
            'std_particle_tumble_frequency_per_second': self.__safe_std(
                metric_values('tumble_frequency_per_second')
            ),
            'mean_particle_tumble_time_fraction': self.__safe_mean(
                metric_values('tumble_time_fraction')
            ),
            'std_particle_tumble_time_fraction': self.__safe_std(
                metric_values('tumble_time_fraction')
            ),
            'mean_number_of_runs_per_particle': self.__safe_mean(
                metric_values('number_of_runs')
            ),
            'std_number_of_runs_per_particle': self.__safe_std(
                metric_values('number_of_runs')
            ),
            'mean_number_of_tumbles_per_particle': self.__safe_mean(
                metric_values('number_of_tumbles')
            ),
            'std_number_of_tumbles_per_particle': self.__safe_std(
                metric_values('number_of_tumbles')
            ),
            'particle_mean_point_run_speed': self.__safe_mean(
                metric_values('point_mean_run_speed')
            ),
            'particle_std_point_run_speed': self.__safe_std(
                metric_values('point_mean_run_speed')
            ),
            'particle_mean_point_tumble_speed': self.__safe_mean(
                metric_values('point_mean_tumble_speed')
            ),
            'particle_std_point_tumble_speed': self.__safe_std(
                metric_values('point_mean_tumble_speed')
            ),
            'mean_run_interval_seconds': self.__safe_mean(
                metric_values('mean_run_interval_seconds')
            ),
            'std_run_interval_seconds_between_particles': self.__safe_std(
                metric_values('mean_run_interval_seconds')
            ),
            'mean_tumble_interval_seconds': self.__safe_mean(
                metric_values('mean_tumble_interval_seconds')
            ),
            'std_tumble_interval_seconds_between_particles': self.__safe_std(
                metric_values('mean_tumble_interval_seconds')
            ),
            'mean_time_between_tumble_starts_seconds': self.__safe_mean(
                metric_values('mean_time_between_tumble_starts_seconds')
            ),
            'std_time_between_tumble_starts_seconds_between_particles': (
                self.__safe_std(
                    metric_values(
                        'mean_time_between_tumble_starts_seconds'
                    )
                )
            ),
            'mean_run_angular_velocity_magnitude_radians_per_second': (
                self.__safe_mean(metric_values(
                    'mean_run_angular_velocity_magnitude_radians_per_second'
                ))
            ),
            'std_run_angular_velocity_between_particles': self.__safe_std(
                metric_values(
                    'mean_run_angular_velocity_magnitude_radians_per_second'
                )
            ),
            'mean_tumble_angular_velocity_magnitude_radians_per_second': (
                self.__safe_mean(metric_values(
                    'mean_tumble_angular_velocity_magnitude_radians_per_second'
                ))
            ),
            'std_tumble_angular_velocity_between_particles': self.__safe_std(
                metric_values(
                    'mean_tumble_angular_velocity_magnitude_radians_per_second'
                )
            ),
            'pooled_time_weighted_vR': pooled_metric(
                'time_weighted_vR', 'vR_support_seconds'
            ),
            'pooled_time_weighted_vR_ci_lower': v_R_ci_lower,
            'pooled_time_weighted_vR_ci_upper': v_R_ci_upper,
            'vR_interval_count': int(total_support(
                'vR_interval_count'
            )),
            'vR_support_seconds': total_support(
                'vR_support_seconds'
            ),
            'vR_particle_count': contributing_particle_count(
                'time_weighted_vR', 'vR_support_seconds'
            ),
            'pooled_time_weighted_vT': pooled_metric(
                'time_weighted_vT', 'vT_support_seconds'
            ),
            'pooled_time_weighted_vT_ci_lower': v_T_ci_lower,
            'pooled_time_weighted_vT_ci_upper': v_T_ci_upper,
            'vT_interval_count': int(total_support(
                'vT_interval_count'
            )),
            'vT_support_seconds': total_support(
                'vT_support_seconds'
            ),
            'vT_particle_count': contributing_particle_count(
                'time_weighted_vT', 'vT_support_seconds'
            ),
            't_R': pooled_t_R,
            'std_t_R': pooled_t_R_std,
            'sem_t_R': pooled_t_R_sem,
            't_R_interval_count': pooled_t_R_count,
            't_R_censored_interval_count': int(total_support(
                't_R_censored_interval_count'
            )),
            't_T': pooled_t_T,
            'std_t_T': pooled_t_T_std,
            't_T_interval_count': pooled_t_T_count,
            't_T_censored_interval_count': int(total_support(
                't_T_censored_interval_count'
            )),
            'p': pooled_p,
            'std_p': pooled_p_std,
            'p_direction_change_count': pooled_p_count,
            'R': pooled_R,
            'std_R': pooled_R_std,
            'R_run_transition_count': pooled_R_count,
            'particle_mean_time_weighted_vR': self.__safe_mean(
                metric_values('time_weighted_vR')
            ),
            'particle_std_time_weighted_vR': self.__safe_std(
                metric_values('time_weighted_vR')
            ),
            'particle_sem_time_weighted_vR': self.__safe_sem(
                metric_values('time_weighted_vR')
            ),
            'particle_mean_time_weighted_vT': self.__safe_mean(
                metric_values('time_weighted_vT')
            ),
            'particle_std_time_weighted_vT': self.__safe_std(
                metric_values('time_weighted_vT')
            ),
            'particle_sem_time_weighted_vT': self.__safe_sem(
                metric_values('time_weighted_vT')
            ),
            'mean_particle_t_R': self.__safe_mean(
                metric_values('t_R')
            ),
            'std_particle_t_R': self.__safe_std(
                metric_values('t_R')
            ),
            'mean_particle_t_T': self.__safe_mean(
                metric_values('t_T')
            ),
            'std_particle_t_T': self.__safe_std(
                metric_values('t_T')
            ),
            'mean_particle_p': self.__safe_mean(
                metric_values('p')
            ),
            'std_particle_p': self.__safe_std(
                metric_values('p')
            ),
            'mean_particle_R': self.__safe_mean(
                metric_values('R')
            ),
            'std_particle_R': self.__safe_std(
                metric_values('R')
            ),
            'distance_unit': distance_unit,
            'speed_unit': speed_unit,
            'angular_velocity_unit': 'rad/s',
            **smoothing_output,
            'discard_initial_frames': discard_initial_frames,
            'discard_final_frames': discard_final_frames,
            'speed_drop_ratio_threshold': speed_drop_ratio_threshold,
            'speed_period_depth_fraction': speed_period_depth_fraction,
            'angular_change_coefficient': angular_change_coefficient,
            'minimum_heading_speed': minimum_heading_speed,
            'speed_extrema_prominence': speed_extrema_prominence,
            'angular_velocity_extrema_prominence': (
                angular_velocity_extrema_prominence
            ),
            'extrema_min_distance': extrema_min_distance,
            'persistence_interval_seconds': (
                persistence_interval_seconds
            ),
            'run_direction_fit_points': run_direction_fit_points,
            'max_frame_gap': max_frame_gap,
            'pooled_speed_bootstrap_resamples': bootstrap_resamples,
            'pooled_speed_bootstrap_confidence_level': (
                bootstrap_confidence_level
            ),
            'pooled_speed_bootstrap_random_seed': bootstrap_random_seed,
        }
        return pd.DataFrame(
            [population_row],
            columns=self.VELOCITY_TUMBLE_POPULATION_COLUMNS
        )

    def __calculate_velocity_run_angular_msd(
        self,
        points_dataframe: pd.DataFrame,
        tumble_summary: pd.DataFrame
    ) -> dict:
        """Fit the run-only angular MSD and return auditable fit results."""
        result = {
            'rotational_diffusion_coefficient_radians_squared_per_second': (
                np.nan
            ),
            'angular_msd_fit_max_lag_seconds': np.nan,
            'angular_msd_largest_fitted_lag_seconds': np.nan,
            'angular_msd_fit_lag_count': 0,
            'angular_msd_fit_direction_pair_count': 0,
            'angular_msd_fit_r_squared': np.nan,
            'angular_msd_fit_status': 'mean_run_time_unavailable',
        }
        if tumble_summary.empty:
            return result

        run_counts = pd.to_numeric(
            tumble_summary['number_of_runs'], errors='coerce'
        ).to_numpy(dtype=float)
        run_durations = pd.to_numeric(
            tumble_summary['total_run_interval_seconds'], errors='coerce'
        ).to_numpy(dtype=float)
        valid_run_support = (
            np.isfinite(run_counts) &
            np.isfinite(run_durations) &
            (run_counts > 0) &
            (run_durations >= 0)
        )
        total_run_count = float(np.sum(run_counts[valid_run_support]))
        total_run_duration = float(np.sum(
            run_durations[valid_run_support]
        ))
        if total_run_count <= 0 or total_run_duration <= 0:
            return result
        fit_max_lag_seconds = total_run_duration / total_run_count
        result['angular_msd_fit_max_lag_seconds'] = fit_max_lag_seconds
        result['angular_msd_fit_status'] = 'run_heading_pairs_unavailable'
        if points_dataframe.empty:
            return result

        lag_squared_change_sums: dict[int, float] = {}
        lag_pair_counts: dict[int, int] = {}
        lag_tolerance = max(
            np.finfo(float).eps,
            abs(fit_max_lag_seconds) * 1e-12
        )
        max_fit_frame_lag = int(np.floor(
            (fit_max_lag_seconds + lag_tolerance) *
            self._capture_speed_in_fps
        ))
        if max_fit_frame_lag < 1:
            result['angular_msd_fit_status'] = 'insufficient_lag_support'
            return result
        grouped_segments = points_dataframe.groupby(
            ['particle', 'segment_id'], sort=False
        )
        for _, segment_points in grouped_segments:
            segment_points = segment_points.sort_values(
                by='frame', kind='stable'
            )
            if segment_points.empty:
                continue
            states = segment_points['state'].astype(str).to_numpy()
            previous_states = np.empty(len(states), dtype=object)
            previous_states[0] = ''
            previous_states[1:] = states[:-1]
            headings = pd.to_numeric(
                segment_points['heading_radians'], errors='coerce'
            ).to_numpy(dtype=float)
            frames = pd.to_numeric(
                segment_points['frame'], errors='coerce'
            ).to_numpy(dtype=float)
            finite_run_heading = (
                (states == 'run') &
                (previous_states == 'run') &
                np.isfinite(headings) &
                np.isfinite(frames)
            )
            finite_indices = np.flatnonzero(finite_run_heading)
            if len(finite_indices) < 2:
                continue
            block_starts = np.r_[
                0, np.flatnonzero(np.diff(finite_indices) > 1) + 1
            ]
            block_ends = np.r_[block_starts[1:], len(finite_indices)]
            for block_start, block_end in zip(block_starts, block_ends):
                local_indices = finite_indices[block_start:block_end]
                if len(local_indices) < 2:
                    continue
                block_frames = frames[local_indices].astype(int)
                if np.any(np.diff(block_frames) <= 0):
                    continue
                block_headings = np.unwrap(headings[local_indices])
                for offset in range(1, len(local_indices)):
                    frame_lags = (
                        block_frames[offset:] - block_frames[:-offset]
                    )
                    if int(np.min(frame_lags)) > max_fit_frame_lag:
                        break
                    heading_changes = (
                        block_headings[offset:] - block_headings[:-offset]
                    )
                    valid_pairs = (
                        (frame_lags > 0) &
                        np.isfinite(heading_changes) &
                        (frame_lags <= max_fit_frame_lag)
                    )
                    for frame_lag in np.unique(frame_lags[valid_pairs]):
                        lag_mask = valid_pairs & (frame_lags == frame_lag)
                        squared_changes = heading_changes[lag_mask] ** 2
                        normalized_lag = int(frame_lag)
                        lag_squared_change_sums[normalized_lag] = (
                            lag_squared_change_sums.get(normalized_lag, 0.0) +
                            float(np.sum(squared_changes))
                        )
                        lag_pair_counts[normalized_lag] = (
                            lag_pair_counts.get(normalized_lag, 0) +
                            int(len(squared_changes))
                        )

        supported_lags = sorted(
            frame_lag for frame_lag, pair_count in lag_pair_counts.items()
            if pair_count > 0
        )
        if len(supported_lags) < 2:
            result['angular_msd_fit_status'] = 'insufficient_lag_support'
            return result
        lag_times = np.asarray(
            supported_lags, dtype=float
        ) / self._capture_speed_in_fps
        angular_msd_values = np.asarray([
            lag_squared_change_sums[frame_lag] /
            lag_pair_counts[frame_lag]
            for frame_lag in supported_lags
        ], dtype=float)
        finite_fit = (
            np.isfinite(lag_times) &
            np.isfinite(angular_msd_values) &
            (lag_times > 0)
        )
        lag_times = lag_times[finite_fit]
        angular_msd_values = angular_msd_values[finite_fit]
        fitted_lags = np.asarray(supported_lags, dtype=int)[finite_fit]
        if len(lag_times) < 2:
            result['angular_msd_fit_status'] = 'insufficient_lag_support'
            return result
        slope_denominator = float(np.sum(lag_times ** 2))
        if not np.isfinite(slope_denominator) or slope_denominator <= 0:
            result['angular_msd_fit_status'] = 'invalid_fit_support'
            return result
        slope = float(
            np.sum(lag_times * angular_msd_values) / slope_denominator
        )
        if not np.isfinite(slope) or slope < 0:
            result['angular_msd_fit_status'] = 'invalid_fit_slope'
            return result
        fitted_values = slope * lag_times
        residual_sum_squares = float(np.sum(
            (angular_msd_values - fitted_values) ** 2
        ))
        uncentered_total_sum_squares = float(np.sum(
            angular_msd_values ** 2
        ))
        if uncentered_total_sum_squares > np.finfo(float).eps:
            fit_r_squared = (
                1 - residual_sum_squares / uncentered_total_sum_squares
            )
        elif residual_sum_squares <= np.finfo(float).eps:
            fit_r_squared = 1.0
        else:
            fit_r_squared = np.nan
        result.update({
            'rotational_diffusion_coefficient_radians_squared_per_second': (
                slope / 2
            ),
            'angular_msd_largest_fitted_lag_seconds': float(
                np.max(lag_times)
            ),
            'angular_msd_fit_lag_count': int(len(lag_times)),
            'angular_msd_fit_direction_pair_count': int(sum(
                lag_pair_counts[int(frame_lag)]
                for frame_lag in fitted_lags
            )),
            'angular_msd_fit_r_squared': fit_r_squared,
            'angular_msd_fit_status': 'fitted',
        })
        return result

    def __build_velocity_tumble_table_1(
        self,
        population_summary: pd.DataFrame,
        angular_msd_fit: dict
    ) -> pd.DataFrame:
        """Build a vertical Najafi-style population parameter table."""
        if population_summary.empty:
            population_row = pd.Series(dtype=object)
            speed_unit = f'{self._scale_units}/s'
        else:
            population_row = population_summary.iloc[0]
            speed_unit = str(population_row['speed_unit'])

        def population_value(column_name: str) -> float:
            numeric_value = pd.to_numeric(
                population_row.get(column_name, np.nan), errors='coerce'
            )
            if pd.isna(numeric_value):
                return np.nan
            numeric_value = float(numeric_value)
            return numeric_value if np.isfinite(numeric_value) else np.nan

        def population_count(column_name: str) -> int:
            value = population_value(column_name)
            return int(value) if np.isfinite(value) else 0

        def sample_sd_definition(description: str, sample_count: int) -> str:
            if sample_count < 2:
                return (
                    f'{description}; sample SD not calculated because n='
                    f'{sample_count}, but at least 2 observations are required'
                )
            return f'{description}; sample SD calculated with ddof=1'

        def sem_definition(description: str, sample_count: int) -> str:
            if sample_count < 2:
                return (
                    f'{description}; SEM not calculated because n='
                    f'{sample_count}, but at least 2 observations are required'
                )
            return f'{description}; SEM is sample SD / sqrt(n)'

        rows: list[dict] = []

        def append_row(
            parameter: str,
            value: object,
            sample_count: int | float = np.nan,
            n_definition: str = 'not applicable'
        ) -> None:
            rows.append({
                'Parameter': parameter,
                'Value': value,
                'n': sample_count,
                'n_definition': n_definition,
            })

        particle_count = population_count('particle_count')
        run_speed_count = population_count('vR_interval_count')
        run_particle_count = population_count('vR_particle_count')
        tumble_speed_count = population_count('vT_interval_count')
        tumble_particle_count = population_count('vT_particle_count')
        run_interval_count = population_count('t_R_interval_count')
        tumble_interval_count = population_count('t_T_interval_count')
        persistence_count = population_count('p_direction_change_count')
        run_transition_count = population_count('R_run_transition_count')
        fit_lag_count = int(angular_msd_fit['angular_msd_fit_lag_count'])

        append_row(
            'Strain', self._strain, particle_count,
            'particles represented in the population summary'
        )
        append_row(
            f'pooled_time_weighted_vR ({speed_unit})',
            population_value('pooled_time_weighted_vR'), run_speed_count,
            'finite run-state speed intervals; pooled mean is weighted by '
            'their elapsed-time support'
        )
        speed_ci_definition = (
            'particles with finite time_weighted_vR and positive run support; '
            'whole '
            'particles are resampled as bootstrap clusters'
        )
        append_row(
            f'pooled_time_weighted_vR_ci_lower ({speed_unit})',
            population_value('pooled_time_weighted_vR_ci_lower'),
            run_particle_count,
            speed_ci_definition
        )
        append_row(
            f'pooled_time_weighted_vR_ci_upper ({speed_unit})',
            population_value('pooled_time_weighted_vR_ci_upper'),
            run_particle_count,
            speed_ci_definition
        )
        run_particle_definition = (
            'particles with finite time_weighted_vR; each particle '
            'contributes one equal-weight value'
        )
        append_row(
            f'particle_mean_time_weighted_vR ({speed_unit})',
            population_value('particle_mean_time_weighted_vR'),
            run_particle_count,
            run_particle_definition
        )
        append_row(
            f'particle_std_time_weighted_vR ({speed_unit})',
            population_value('particle_std_time_weighted_vR'),
            run_particle_count,
            sample_sd_definition(
                run_particle_definition, int(run_particle_count)
            )
        )
        append_row(
            f'particle_sem_time_weighted_vR ({speed_unit})',
            population_value('particle_sem_time_weighted_vR'),
            run_particle_count,
            sem_definition(run_particle_definition, int(run_particle_count))
        )
        append_row(
            f'pooled_time_weighted_vT ({speed_unit})',
            population_value('pooled_time_weighted_vT'), tumble_speed_count,
            'finite tumble-state speed intervals; pooled mean is weighted by '
            'their elapsed-time support'
        )
        tumble_speed_ci_definition = (
            'particles with finite time_weighted_vT and positive tumble '
            'support; whole particles are resampled as bootstrap clusters'
        )
        append_row(
            f'pooled_time_weighted_vT_ci_lower ({speed_unit})',
            population_value('pooled_time_weighted_vT_ci_lower'),
            tumble_particle_count,
            tumble_speed_ci_definition
        )
        append_row(
            f'pooled_time_weighted_vT_ci_upper ({speed_unit})',
            population_value('pooled_time_weighted_vT_ci_upper'),
            tumble_particle_count,
            tumble_speed_ci_definition
        )
        tumble_particle_definition = (
            'particles with finite time_weighted_vT; each particle '
            'contributes one equal-weight value'
        )
        append_row(
            f'particle_mean_time_weighted_vT ({speed_unit})',
            population_value('particle_mean_time_weighted_vT'),
            tumble_particle_count,
            tumble_particle_definition
        )
        append_row(
            f'particle_std_time_weighted_vT ({speed_unit})',
            population_value('particle_std_time_weighted_vT'),
            tumble_particle_count,
            sample_sd_definition(
                tumble_particle_definition, int(tumble_particle_count)
            )
        )
        append_row(
            f'particle_sem_time_weighted_vT ({speed_unit})',
            population_value('particle_sem_time_weighted_vT'),
            tumble_particle_count,
            sem_definition(
                tumble_particle_definition, int(tumble_particle_count)
            )
        )
        for parameter in (
            'pooled_speed_bootstrap_resamples',
            'pooled_speed_bootstrap_confidence_level',
            'pooled_speed_bootstrap_random_seed',
        ):
            append_row(
                parameter, population_value(parameter), np.nan,
                'not applicable; this row is a bootstrap configuration value'
            )
        run_interval_definition = 'complete uncensored run episodes'
        append_row(
            'Mean tR (s)', population_value('t_R'), run_interval_count,
            run_interval_definition
        )
        append_row(
            '±std', population_value('std_t_R'), run_interval_count,
            sample_sd_definition(
                run_interval_definition, int(run_interval_count)
            )
        )
        append_row(
            '±sem', population_value('sem_t_R'), run_interval_count,
            sem_definition(run_interval_definition, int(run_interval_count))
        )
        tumble_interval_definition = 'complete uncensored tumble episodes'
        append_row(
            'Mean tT (s)', population_value('t_T'), tumble_interval_count,
            tumble_interval_definition
        )
        append_row(
            'std_t_T (s)', population_value('std_t_T'),
            tumble_interval_count,
            sample_sd_definition(
                tumble_interval_definition, int(tumble_interval_count)
            )
        )
        persistence_definition = (
            'supported within-run direction-change cosine observations'
        )
        append_row(
            'p', population_value('p'), persistence_count,
            persistence_definition
        )
        append_row(
            'std_p', population_value('std_p'), persistence_count,
            sample_sd_definition(
                persistence_definition, int(persistence_count)
            )
        )
        transition_definition = (
            'supported fitted run-to-run transition cosine observations'
        )
        append_row(
            'R', population_value('R'), run_transition_count,
            transition_definition
        )
        append_row(
            'std_R', population_value('std_R'), run_transition_count,
            sample_sd_definition(
                transition_definition, int(run_transition_count)
            )
        )
        append_row(
            self.ANGULAR_MSD_DR_PARAMETER,
            angular_msd_fit[
                'rotational_diffusion_coefficient_radians_squared_per_second'
            ],
            fit_lag_count,
            'run-only angular-MSD lag-bin means in the origin-constrained fit'
        )
        append_row(
            'Dr fit max lag (s)',
            angular_msd_fit['angular_msd_fit_max_lag_seconds'],
            population_count('total_number_of_runs'),
            'reconstructed run intervals used to calculate the requested fit '
            'ceiling, including boundary-censored runs'
        )
        append_row(
            'Dr largest fitted lag (s)',
            angular_msd_fit['angular_msd_largest_fitted_lag_seconds'],
            fit_lag_count,
            'run-only angular-MSD lag-bin means retained in the fit'
        )
        append_row(
            'Dr fit lag count', fit_lag_count, np.nan,
            'not applicable; the Value is the number of fitted lag bins'
        )
        append_row(
            'Dr fit direction-pair count',
            angular_msd_fit['angular_msd_fit_direction_pair_count'], np.nan,
            'not applicable; the Value is the total direction-pair support '
            'summed over fitted lag bins'
        )
        append_row(
            'Dr fit uncentered R^2',
            angular_msd_fit['angular_msd_fit_r_squared'], fit_lag_count,
            'run-only angular-MSD lag-bin means retained in the fit'
        )
        append_row(
            'Dr fit status', angular_msd_fit['angular_msd_fit_status'], np.nan,
            'not applicable; this row reports fit status or failure reason'
        )
        return pd.DataFrame(
            rows, columns=self.VELOCITY_TABLE_1_VERTICAL_COLUMNS
        )

    @staticmethod
    def __normalize_max_frame_gap(max_frame_gap: int | None) -> int | None:
        """Validate an optional largest retained frame difference."""
        if max_frame_gap is None:
            return None
        if isinstance(max_frame_gap, bool) or not isinstance(
            max_frame_gap, (int, np.integer)
        ):
            raise TypeError('max_frame_gap must be an integer or None.')
        if max_frame_gap < 1:
            raise ValueError('max_frame_gap must be at least 1.')
        return int(max_frame_gap)

    def __get_speed_unit_components(
        self,
        speed_unit_key: str
    ) -> tuple[float, str, float, str]:
        """Return distance/time conversion factors and labels for a speed key."""
        if speed_unit_key.startswith('scale_units'):
            self.__validate_pixel_scale_factor()
            distance_factor = self.pixel_scale_factor
            distance_unit = self._scale_units
        else:
            distance_factor = 1.0
            distance_unit = 'pixels'
        if speed_unit_key.endswith('per_second'):
            self.__validate_frame_rate()
            time_factor = 1.0 / self._capture_speed_in_fps
            time_unit = 'seconds'
        else:
            time_factor = 1.0
            time_unit = 'frames'
        return distance_factor, distance_unit, time_factor, time_unit

    def __normalize_length_unit(
        self,
        length_unit: str,
        argument_name: str = 'length_unit'
    ) -> tuple[str, float, str]:
        """Normalize a spatial unit and return its pixel conversion factor."""
        if not isinstance(length_unit, str):
            raise TypeError(f'{argument_name} must be a string.')
        normalized_unit = length_unit.strip().lower().replace('μ', 'µ')
        pixel_aliases = {'pixel', 'pixels', 'px'}
        scale_aliases = {
            'scale', 'scale_unit', 'scale_units', 'scale unit', 'scale units',
            self._scale_units.lower().replace('μ', 'µ')
        }
        if self._scale_units.lower().replace('μ', 'µ') in {
            'µm', 'um', 'micrometer', 'micrometers',
            'micrometre', 'micrometres'
        }:
            scale_aliases.update({
                'µm', 'um', 'micrometer', 'micrometers',
                'micrometre', 'micrometres'
            })
        if normalized_unit in pixel_aliases:
            return 'pixels', 1.0, 'pixels'
        if normalized_unit in scale_aliases:
            self.__validate_pixel_scale_factor()
            return 'scale_units', self.pixel_scale_factor, self._scale_units
        raise ValueError(
            f"{argument_name} must be 'pixels', 'scale_units', or the configured "
            f"scale unit '{self._scale_units}'."
        )

    def __get_feature_conversion(
        self,
        feature_column: str,
        feature_unit: str
    ) -> tuple[float, str]:
        """Return a conversion factor and label for a tracked feature column."""
        if not isinstance(feature_unit, str):
            raise TypeError('feature_unit must be a string.')
        normalized_feature_unit = feature_unit.strip().lower().replace('μ', 'µ')
        length_features = {'major_axis_length', 'minor_axis_length'}
        if feature_column == 'area':
            if normalized_feature_unit == 'auto':
                normalized_feature_unit = 'scale_units'
            if normalized_feature_unit in {'raw', 'stored'}:
                return 1.0, 'pixels²'
            base_unit = normalized_feature_unit.replace('^2', '').replace('²', '')
            _, factor, label = self.__normalize_length_unit(
                base_unit,
                argument_name='feature_unit'
            )
            return factor**2, f'{label}²'
        if feature_column in length_features:
            if normalized_feature_unit == 'auto':
                normalized_feature_unit = 'scale_units'
            if normalized_feature_unit in {'raw', 'stored'}:
                return 1.0, 'pixels'
            _, factor, label = self.__normalize_length_unit(
                normalized_feature_unit,
                argument_name='feature_unit'
            )
            return factor, label
        if normalized_feature_unit not in {'auto', 'raw', 'stored'}:
            raise ValueError(
                'feature_unit must be auto/raw/stored for non-area, '
                'non-length features.'
            )
        return 1.0, 'stored units'

    @staticmethod
    def __finite_numeric_values(values: pd.Series | np.ndarray) -> np.ndarray:
        """Return finite numeric values without mutating the source."""
        numeric_values = pd.to_numeric(
            pd.Series(values), errors='coerce'
        ).to_numpy(dtype=float)
        return numeric_values[np.isfinite(numeric_values)]

    @staticmethod
    def __particle_cluster_bootstrap_weighted_mean_ci(
        values: np.ndarray,
        weights: np.ndarray,
        resamples: int,
        confidence_level: float,
        random_generator: np.random.Generator
    ) -> tuple[float, float]:
        """Return a percentile CI after resampling contributing particles."""
        values = np.asarray(values, dtype=float)
        weights = np.asarray(weights, dtype=float)
        valid = (
            np.isfinite(values) &
            np.isfinite(weights) &
            (weights > 0)
        )
        values = values[valid]
        weights = weights[valid]
        particle_count = len(values)
        if particle_count < 2:
            return np.nan, np.nan

        bootstrap_means = np.empty(resamples, dtype=float)
        for resample_index in range(resamples):
            sampled_indices = random_generator.integers(
                0, particle_count, size=particle_count
            )
            sampled_weights = weights[sampled_indices]
            bootstrap_means[resample_index] = float(
                np.sum(values[sampled_indices] * sampled_weights) /
                np.sum(sampled_weights)
            )

        tail_probability = (1.0 - confidence_level) / 2.0
        lower, upper = np.quantile(
            bootstrap_means,
            [tail_probability, 1.0 - tail_probability]
        )
        return float(lower), float(upper)

    @staticmethod
    def __safe_mean(values: np.ndarray) -> float:
        """Return a finite-array arithmetic mean or NaN."""
        finite_values = np.asarray(values, dtype=float)
        finite_values = finite_values[np.isfinite(finite_values)]
        return float(np.mean(finite_values)) if len(finite_values) else np.nan

    @staticmethod
    def __safe_median(values: np.ndarray) -> float:
        """Return a finite-array median or NaN."""
        finite_values = np.asarray(values, dtype=float)
        finite_values = finite_values[np.isfinite(finite_values)]
        return float(np.median(finite_values)) if len(finite_values) else np.nan

    @staticmethod
    def __safe_std(values: np.ndarray) -> float:
        """Return sample standard deviation or NaN when fewer than two values."""
        finite_values = np.asarray(values, dtype=float)
        finite_values = finite_values[np.isfinite(finite_values)]
        return (
            float(np.std(finite_values, ddof=1))
            if len(finite_values) > 1
            else np.nan
        )

    @classmethod
    def __safe_sem(cls, values: np.ndarray) -> float:
        """Return sample standard error or NaN with fewer than two values."""
        finite_values = np.asarray(values, dtype=float)
        finite_values = finite_values[np.isfinite(finite_values)]
        sample_std = cls.__safe_std(finite_values)
        return (
            float(sample_std / np.sqrt(len(finite_values)))
            if np.isfinite(sample_std)
            else np.nan
        )

    @staticmethod
    def __combine_group_mean_std_sem(
        group_means: np.ndarray,
        group_stds: np.ndarray,
        group_counts: np.ndarray
    ) -> tuple[float, float, float, int]:
        """Combine group moments into exact pooled mean, sample SD, and SEM."""
        means = np.asarray(group_means, dtype=float)
        stds = np.asarray(group_stds, dtype=float)
        counts = np.asarray(group_counts, dtype=float)
        valid_groups = (
            np.isfinite(means) &
            np.isfinite(counts) &
            (counts > 0)
        )
        means = means[valid_groups]
        stds = stds[valid_groups]
        counts = counts[valid_groups]
        if not len(counts):
            return np.nan, np.nan, np.nan, 0
        total_count = int(np.sum(counts))
        pooled_mean = float(np.sum(means * counts) / total_count)
        if total_count < 2:
            return pooled_mean, np.nan, np.nan, total_count
        if np.any((counts > 1) & ~np.isfinite(stds)):
            return pooled_mean, np.nan, np.nan, total_count
        within_group_sum_squares = np.sum(
            np.where(counts > 1, (counts - 1) * stds**2, 0.0)
        )
        between_group_sum_squares = np.sum(
            counts * (means - pooled_mean)**2
        )
        pooled_variance = (
            (within_group_sum_squares + between_group_sum_squares) /
            (total_count - 1)
        )
        pooled_std = float(np.sqrt(max(float(pooled_variance), 0.0)))
        pooled_sem = float(pooled_std / np.sqrt(total_count))
        return pooled_mean, pooled_std, pooled_sem, total_count

    def __has_valid_frame_rate(self) -> bool:
        """Return whether capture FPS is finite and positive."""
        return bool(
            np.isfinite(self._capture_speed_in_fps) and
            self._capture_speed_in_fps > 0
        )

    def __get_fitted_mean_speed_lookup(
        self,
        speed_unit_label: str,
        discard_initial_frames: int,
        discard_final_frames: int
    ) -> dict[int, float]:
        """Return cached fitted means compatible with this analysis request."""
        cached_means = self._mean_speeds_dataframe
        required_columns = {
            'source_dataframe',
            'particle',
            'mean_speed',
            'speed_unit',
            'discard_initial_frames',
            'discard_final_frames',
        }
        if (
            cached_means.empty or
            self._calculated_speed_unit != speed_unit_label or
            not required_columns.issubset(cached_means.columns)
        ):
            return {}

        compatible_rows = cached_means.loc[
            cached_means['source_dataframe'].eq(
                self._resolved_source_dataframe
            ) &
            cached_means['speed_unit'].eq(speed_unit_label) &
            pd.to_numeric(
                cached_means['discard_initial_frames'], errors='coerce'
            ).eq(int(discard_initial_frames)) &
            pd.to_numeric(
                cached_means['discard_final_frames'], errors='coerce'
            ).eq(int(discard_final_frames)),
            ['particle', 'mean_speed']
        ]

        fitted_mean_speeds: dict[int, float] = {}
        for particle, mean_speed in compatible_rows.itertuples(
            index=False, name=None
        ):
            numeric_particle = pd.to_numeric(particle, errors='coerce')
            numeric_mean = pd.to_numeric(mean_speed, errors='coerce')
            if pd.isna(numeric_particle):
                continue
            fitted_mean_speeds[int(numeric_particle)] = float(numeric_mean)
        return fitted_mean_speeds

    @staticmethod
    def __validate_positive_integer(value: object, argument_name: str) -> None:
        """Validate a positive integer plotting or analysis parameter."""
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f'{argument_name} must be an integer.')
        if value < 1:
            raise ValueError(f'{argument_name} must be at least 1.')

    @staticmethod
    def __validate_nonnegative_integer(
        value: object,
        argument_name: str
    ) -> None:
        """Validate an integer analysis parameter that may be zero."""
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f'{argument_name} must be an integer.')
        if value < 0:
            raise ValueError(f'{argument_name} must be nonnegative.')

    @staticmethod
    def __validate_boolean_argument(
        value: object,
        argument_name: str
    ) -> None:
        """Validate a public plotting toggle."""
        if not isinstance(value, (bool, np.bool_)):
            raise TypeError(f'{argument_name} must be a boolean.')

    @staticmethod
    def __particle_label_indices(
        values: np.ndarray,
        particle_ID_spacing: int,
        show_particle_ID_only_with_value: bool
    ) -> np.ndarray:
        """Select regularly spaced particle labels, optionally excluding zero."""
        eligible_indices = np.arange(len(values))
        if show_particle_ID_only_with_value:
            eligible_indices = eligible_indices[values != 0]
        return eligible_indices[::particle_ID_spacing]

    def __resolve_particle_metric_plotting_parameters(
        self,
        **per_call_parameters: object
    ) -> dict:
        """Merge explicitly supplied plot options with configured defaults."""
        configured_parameters = self.get_particle_metric_plotting_parameters()
        resolved_parameters = {}
        for argument_name, per_call_value in per_call_parameters.items():
            if per_call_value is _USE_CONFIGURED_PLOT_VALUE:
                resolved_parameters[argument_name] = (
                    configured_parameters.get(
                        argument_name,
                        self.PARTICLE_METRIC_PLOT_DEFAULTS[argument_name]
                    )
                )
            else:
                resolved_parameters[argument_name] = per_call_value
        return self.__validate_particle_metric_plotting_parameters(
            resolved_parameters
        )

    def __validate_particle_metric_plotting_parameters(
        self,
        parameters: dict
    ) -> dict:
        """Validate and normalize reusable particle-metric plot settings."""
        validated_parameters = dict(parameters)
        self.__validate_boolean_argument(
            validated_parameters['show_particle_ID'],
            'show_particle_ID'
        )
        self.__validate_boolean_argument(
            validated_parameters['show_particle_ID_only_with_value'],
            'show_particle_ID_only_with_value'
        )
        self.__validate_positive_integer(
            validated_parameters['particle_ID_spacing'],
            'particle_ID_spacing'
        )
        self.__validate_positive_integer(
            validated_parameters['dpi'],
            'dpi'
        )
        self.__validate_boolean_argument(
            validated_parameters['save_plots'],
            'save_plots'
        )
        self.__validate_plot_text_options(
            title=validated_parameters['title'],
            title_fontsize=validated_parameters['title_fontsize'],
            x_axis_fontsize=validated_parameters['x_axis_fontsize'],
            y_axis_fontsize=validated_parameters['y_axis_fontsize'],
            x_tick_fontsize=validated_parameters['x_tick_fontsize'],
            y_tick_fontsize=validated_parameters['y_tick_fontsize'],
            y_tick_spacing=validated_parameters['y_tick_spacing'],
            font_family=validated_parameters['font_family']
        )
        save_plot_path = validated_parameters['save_plot_path']
        if save_plot_path is not None:
            if not isinstance(save_plot_path, (str, os.PathLike)):
                raise TypeError(
                    'save_plot_path must be a path string or None.'
                )
            if not os.fspath(save_plot_path).strip():
                raise ValueError('save_plot_path must not be empty.')
        file_extension = validated_parameters['file_extension']
        if not isinstance(file_extension, str):
            raise TypeError('file_extension must be a string.')
        normalized_extension = file_extension.strip().lower().lstrip('.')
        if normalized_extension not in {'png', 'tif'}:
            raise ValueError("file_extension must be 'png' or 'tif'.")
        validated_parameters['file_extension'] = normalized_extension
        return validated_parameters

    def __validate_basic_plotting_parameters(
        self,
        **parameters: object
    ) -> dict:
        """Validate typography and automatic-save options for a basic plot."""
        validated_parameters = dict(parameters)
        self.__validate_positive_integer(
            validated_parameters['dpi'], 'dpi'
        )
        for argument_name in ('save_plots', 'show'):
            self.__validate_boolean_argument(
                validated_parameters[argument_name], argument_name
            )
        self.__validate_plot_text_options(
            title=validated_parameters['title'],
            title_fontsize=validated_parameters['title_fontsize'],
            x_axis_fontsize=validated_parameters['x_axis_fontsize'],
            y_axis_fontsize=validated_parameters['y_axis_fontsize'],
            x_tick_fontsize=validated_parameters['x_tick_fontsize'],
            y_tick_fontsize=validated_parameters['y_tick_fontsize'],
            y_tick_spacing=None,
            font_family=validated_parameters['font_family']
        )
        save_plot_path = validated_parameters['save_plot_path']
        if save_plot_path is not None:
            if not isinstance(save_plot_path, (str, os.PathLike)):
                raise TypeError(
                    'save_plot_path must be a path string or None.'
                )
            if not os.fspath(save_plot_path).strip():
                raise ValueError('save_plot_path must not be empty.')
        file_extension = validated_parameters['file_extension']
        if not isinstance(file_extension, str):
            raise TypeError('file_extension must be a string.')
        normalized_extension = file_extension.strip().lower().lstrip('.')
        if normalized_extension not in {'png', 'tif'}:
            raise ValueError("file_extension must be 'png' or 'tif'.")
        validated_parameters['file_extension'] = normalized_extension
        return validated_parameters

    def __resolve_analysis_plotting_parameters(
        self,
        function_tumble_marker_default: str,
        **per_call_parameters: object
    ) -> dict:
        """Merge analysis plot overrides with the configured defaults."""
        resolved_parameters = dict(self.ANALYSIS_PLOT_DEFAULTS)
        resolved_parameters.update(
            self.get_analysis_plotting_parameters()
        )
        for argument_name, per_call_value in per_call_parameters.items():
            if per_call_value is not _USE_CONFIGURED_PLOT_VALUE:
                resolved_parameters[argument_name] = per_call_value
        resolved_parameters = (
            self.__validate_analysis_plotting_parameters(
                resolved_parameters
            )
        )
        if resolved_parameters['tumble_marker'] is None:
            resolved_parameters['tumble_marker'] = (
                function_tumble_marker_default
            )
        self.__validate_plot_marker(
            resolved_parameters['tumble_marker'], 'tumble_marker'
        )
        return resolved_parameters

    def __validate_analysis_plotting_parameters(
        self,
        parameters: dict
    ) -> dict:
        """Validate and normalize reusable particle-analysis plot settings."""
        validated_parameters = dict(parameters)
        validated_parameters['angle_mode'] = self.__normalize_angle_mode(
            validated_parameters['angle_mode']
        )
        for argument_name in (
            'show_turns',
            'show_smoothed_trajectory',
            'show_tumbles',
            'show_angle_smoothed_trajectory',
            'show_velocity_smoothed_trajectory',
            'crop_to_track',
            'save_plots',
        ):
            self.__validate_boolean_argument(
                validated_parameters[argument_name], argument_name
            )
        for argument_name in ('turn_display_mode', 'tumble_display_mode'):
            validated_parameters[argument_name] = (
                self.__normalize_event_display_mode(
                    validated_parameters[argument_name], argument_name
                )
            )
        validated_parameters['marker_fill_mode'] = (
            self.__normalize_marker_fill_mode(
                validated_parameters['marker_fill_mode']
            )
        )
        for argument_name in (
            'turn_color',
            'smoothed_trajectory_color',
            'tumble_color',
            'smoothed_angle_trajectory_color',
            'smoothed_velocity_trajectory_color',
        ):
            self.__validate_plot_color(
                validated_parameters[argument_name], argument_name
            )
        for argument_name in (
            'track_thickness',
            'smooth_trajectory_thickness',
            'segment_marker_thickness',
        ):
            value = validated_parameters[argument_name]
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, float, np.integer, np.floating)
            ):
                raise TypeError(f'{argument_name} must be numeric.')
            normalized_value = float(value)
            if not np.isfinite(normalized_value) or normalized_value <= 0:
                raise ValueError(
                    f'{argument_name} must be finite and positive.'
                )
            validated_parameters[argument_name] = normalized_value
        particle_marker = validated_parameters['particle_marker']
        if particle_marker is not None:
            self.__validate_plot_marker(
                particle_marker, 'particle_marker'
            )
        self.__validate_plot_marker(
            validated_parameters['turn_marker'], 'turn_marker'
        )
        tumble_marker = validated_parameters['tumble_marker']
        if tumble_marker is not None:
            self.__validate_plot_marker(tumble_marker, 'tumble_marker')
        self.__validate_positive_integer(
            validated_parameters['dpi'], 'dpi'
        )
        self.__validate_plot_text_options(
            title=validated_parameters['title'],
            title_fontsize=validated_parameters['title_fontsize'],
            x_axis_fontsize=validated_parameters['x_axis_fontsize'],
            y_axis_fontsize=validated_parameters['y_axis_fontsize'],
            x_tick_fontsize=validated_parameters['x_tick_fontsize'],
            y_tick_fontsize=validated_parameters['y_tick_fontsize'],
            y_tick_spacing=None,
            font_family=validated_parameters['font_family']
        )
        save_plot_path = validated_parameters['save_plot_path']
        if save_plot_path is not None:
            if not isinstance(save_plot_path, (str, os.PathLike)):
                raise TypeError(
                    'save_plot_path must be a path string or None.'
                )
            if not os.fspath(save_plot_path).strip():
                raise ValueError('save_plot_path must not be empty.')
        file_extension = validated_parameters['file_extension']
        if not isinstance(file_extension, str):
            raise TypeError('file_extension must be a string.')
        normalized_extension = file_extension.strip().lower().lstrip('.')
        if normalized_extension not in {'png', 'tif'}:
            raise ValueError("file_extension must be 'png' or 'tif'.")
        validated_parameters['file_extension'] = normalized_extension
        return validated_parameters

    def __resolve_numbered_plot_save_path(
        self,
        save_path: str | os.PathLike | None,
        save_plots: bool,
        save_plot_path: str | os.PathLike | None,
        file_extension: str,
        default_title: str,
        title: str | None
    ) -> str | os.PathLike | None:
        """Generate the first available numbered automatic plot filename."""
        if save_path is not None or not save_plots:
            return save_path

        requested_directory = (
            '' if save_plot_path is None else os.fspath(save_plot_path)
        )
        output_root = getattr(self, '_directory', os.getcwd())
        resolved_directory = (
            requested_directory
            if os.path.isabs(requested_directory)
            else os.path.join(output_root, requested_directory)
        )
        used_numbers = set()
        number_width = 2
        if os.path.exists(resolved_directory):
            if not os.path.isdir(resolved_directory):
                raise NotADirectoryError(
                    f'Automatic plot path is not a directory: '
                    f'{resolved_directory}'
                )
            for directory_entry in os.scandir(resolved_directory):
                number_match = re.match(
                    r'^(\d+)(?:[_ -]|$)', directory_entry.name
                )
                if number_match is None:
                    continue
                number_text = number_match.group(1)
                used_numbers.add(int(number_text))
                number_width = max(number_width, len(number_text))
        next_number = 1
        while next_number in used_numbers:
            next_number += 1

        filename_title = (
            default_title
            if title is None or not title.strip()
            else title.strip()
        )
        safe_title = re.sub(
            r'[^\w .()=-]+', '_', filename_title
        ).strip(' ._')
        safe_title = re.sub(r'\s+', ' ', safe_title) or 'Plot'
        filename = (
            f'{next_number:0{number_width}d}_{safe_title}.{file_extension}'
        )
        return os.path.join(requested_directory, filename)

    @staticmethod
    def __validate_plot_text_options(
        title: str | None,
        title_fontsize: float | None,
        x_axis_fontsize: float | None,
        y_axis_fontsize: float | None,
        x_tick_fontsize: float | None,
        y_tick_fontsize: float | None,
        y_tick_spacing: float | None,
        font_family: str | None
    ) -> None:
        """Validate reusable plot text and numeric tick options."""
        if title is not None and not isinstance(title, str):
            raise TypeError('title must be a string or None.')
        if font_family is not None:
            if not isinstance(font_family, str):
                raise TypeError('font_family must be a string or None.')
            if not font_family.strip():
                raise ValueError('font_family must not be empty.')
        optional_positive_numbers = {
            'title_fontsize': title_fontsize,
            'x_axis_fontsize': x_axis_fontsize,
            'y_axis_fontsize': y_axis_fontsize,
            'x_tick_fontsize': x_tick_fontsize,
            'y_tick_fontsize': y_tick_fontsize,
            'y_tick_spacing': y_tick_spacing,
        }
        for argument_name, value in optional_positive_numbers.items():
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

    @staticmethod
    def __format_particle_metric_axis(
        axis,
        default_title: str,
        title: str | None,
        title_fontsize: float | None,
        x_axis_fontsize: float | None,
        y_axis_fontsize: float | None,
        x_tick_fontsize: float | None,
        y_tick_fontsize: float | None,
        y_tick_spacing: float | None,
        font_family: str | None
    ) -> None:
        """Apply local typography and tick formatting to a metric axis."""
        axis.set_title(default_title if title is None else title)
        if title_fontsize is not None:
            axis.title.set_fontsize(float(title_fontsize))
        if x_axis_fontsize is not None:
            axis.xaxis.label.set_fontsize(float(x_axis_fontsize))
        if y_axis_fontsize is not None:
            axis.yaxis.label.set_fontsize(float(y_axis_fontsize))
        if y_tick_spacing is not None:
            axis.yaxis.set_major_locator(MultipleLocator(float(y_tick_spacing)))
        if x_tick_fontsize is not None:
            axis.tick_params(axis='x', labelsize=float(x_tick_fontsize))
        if y_tick_fontsize is not None:
            axis.tick_params(axis='y', labelsize=float(y_tick_fontsize))
        if font_family is not None:
            selected_font_family = font_family.strip()
            text_items = [
                axis.title,
                axis.xaxis.label,
                axis.yaxis.label,
                *axis.get_xticklabels(),
                *axis.get_yticklabels(),
            ]
            for text_item in text_items:
                text_item.set_fontfamily(selected_font_family)

    @staticmethod
    def __format_analysis_figure(
        figure,
        default_title: str,
        title: str | None,
        primary_title_axis,
        title_fontsize: float | None,
        x_axis_fontsize: float | None,
        y_axis_fontsize: float | None,
        x_tick_fontsize: float | None,
        y_tick_fontsize: float | None,
        font_family: str | None
    ) -> None:
        """Apply a primary title and local typography to an analysis figure."""
        resolved_title = default_title if title is None else title
        if primary_title_axis is None:
            primary_title_artist = (
                figure.suptitle(resolved_title) if resolved_title else None
            )
        else:
            primary_title_axis.set_title(resolved_title)
            primary_title_artist = primary_title_axis.title
        if title_fontsize is not None and primary_title_artist is not None:
            primary_title_artist.set_fontsize(float(title_fontsize))

        for axis in figure.axes:
            if x_axis_fontsize is not None:
                axis.xaxis.label.set_fontsize(float(x_axis_fontsize))
            if y_axis_fontsize is not None:
                axis.yaxis.label.set_fontsize(float(y_axis_fontsize))
            if x_tick_fontsize is not None:
                axis.tick_params(axis='x', labelsize=float(x_tick_fontsize))
            if y_tick_fontsize is not None:
                axis.tick_params(axis='y', labelsize=float(y_tick_fontsize))
        if font_family is not None:
            selected_font_family = font_family.strip()
            for text_item in figure.findobj(match=Text):
                text_item.set_fontfamily(selected_font_family)

    @staticmethod
    def __plot_unit_label(unit: object) -> str:
        """Use the Greek mu in plot-only micrometre unit labels."""
        return re.sub(
            r'(?<![A-Za-z])(?:u|\N{MICRO SIGN}|\N{GREEK SMALL LETTER MU})m'
            r'(?![A-Za-z])',
            '\N{GREEK SMALL LETTER MU}m',
            str(unit),
            flags=re.IGNORECASE
        )

    @staticmethod
    def __combined_analysis_default_title(
        particle_id: int,
        tumble_method_label: str,
        show_turns: bool,
        show_tumbles: bool
    ) -> str:
        """Build the established title for a combined analysis plot."""
        displayed_overlays = []
        if show_turns:
            displayed_overlays.append('detected turns')
        if show_tumbles:
            displayed_overlays.append(f'{tumble_method_label} tumbles')
        title = f'Particle {particle_id} trajectory'
        if displayed_overlays:
            title += f": {' and '.join(displayed_overlays)}"
        return title

    @staticmethod
    def __format_combined_trajectory_axis(
        axis,
        crop_to_track: bool
    ) -> None:
        """Apply cropped legacy limits or an uncropped square plotting area."""
        if crop_to_track:
            axis.set_aspect('equal', adjustable='datalim')
            return
        axis.margins(x=0.05, y=0.05)
        axis.set_aspect('equal', adjustable='box')
        axis.set_box_aspect(1)

    @staticmethod
    def __normalize_event_display_mode(
        display_mode: str,
        argument_name: str
    ) -> str:
        """Normalize marker-versus-segment event rendering."""
        if not isinstance(display_mode, str):
            raise TypeError(f'{argument_name} must be a string.')
        normalized_mode = display_mode.strip().lower()
        if normalized_mode not in {'markers', 'segments'}:
            raise ValueError(
                f"{argument_name} must be 'markers' or 'segments'."
            )
        return normalized_mode

    @staticmethod
    def __normalize_marker_fill_mode(
        marker_fill_mode: str | None
    ) -> str | None:
        """Normalize solid-versus-outline marker rendering."""
        if marker_fill_mode is None:
            return None
        if not isinstance(marker_fill_mode, str):
            raise TypeError(
                'marker_fill_mode must be a string or None.'
            )
        normalized_mode = marker_fill_mode.strip().lower()
        if normalized_mode not in {'solid', 'outline'}:
            raise ValueError(
                "marker_fill_mode must be 'solid', 'outline', or None."
            )
        return normalized_mode

    @staticmethod
    def __validate_plot_color(color: object, argument_name: str) -> None:
        """Validate a Matplotlib-compatible color argument."""
        if not is_color_like(color):
            raise ValueError(
                f'{argument_name} must be a Matplotlib-compatible color.'
            )

    @staticmethod
    def __validate_plot_marker(marker: object, argument_name: str) -> None:
        """Validate a visible Matplotlib marker string."""
        if not isinstance(marker, str) or not marker.strip():
            raise ValueError(
                f'{argument_name} must be a valid Matplotlib marker.'
            )
        try:
            MarkerStyle(marker)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f'{argument_name} must be a valid Matplotlib marker.'
            ) from error

    @staticmethod
    def __event_marker_color_arguments(
        marker: str,
        color: object,
        marker_fill_mode: str | None,
        default_outline: bool
    ) -> dict:
        """Style filled and inherently unfilled event markers visibly."""
        marker_style = MarkerStyle(marker)
        use_outline = (
            marker_fill_mode == 'outline' or
            (marker_fill_mode is None and default_outline)
        )
        if use_outline and marker_style.is_filled():
            return {'facecolors': 'none', 'edgecolors': color}
        return {'color': color}

    @staticmethod
    def __resolve_track_color(
        track_color: object,
        argument_name: str = 'track_color'
    ) -> tuple[object | None, Colormap | None]:
        """Resolve a uniform track color or an elapsed-time colormap."""
        if isinstance(track_color, Colormap):
            return None, track_color
        if is_color_like(track_color):
            return track_color, None
        if isinstance(track_color, str):
            try:
                return None, plt.get_cmap(track_color)
            except ValueError as error:
                raise ValueError(
                    f'{argument_name} must be a Matplotlib-compatible solid '
                    'color, colormap name, or Colormap object.'
                ) from error
        raise TypeError(
            f'{argument_name} must be a Matplotlib-compatible solid color, '
            'colormap name, or Colormap object.'
        )

    @staticmethod
    def __track_color_for_value(
        uniform_track_color: object | None,
        track_colormap: Colormap | None,
        color_norm: Normalize,
        color_value: float
    ) -> object:
        """Return the uniform or colormap-derived color for one track edge."""
        if track_colormap is None:
            return uniform_track_color
        return track_colormap(color_norm(color_value))

    @staticmethod
    def __scatter_particle_positions(
        axis,
        positions: np.ndarray,
        color_values: np.ndarray,
        uniform_track_color: object | None,
        track_colormap: Colormap | None,
        color_norm: Normalize,
        particle_marker: str | None,
        marker_fill_mode: str | None,
        label: str | None = None,
        zorder: float = 3
    ) -> None:
        """Draw observed particle positions with uniform or mapped colors."""
        if particle_marker is None or not len(positions):
            return
        scatter_arguments = {
            'marker': particle_marker,
            's': 20,
            'zorder': zorder,
            'label': label,
        }
        marker_style = MarkerStyle(particle_marker)
        use_outline = (
            marker_fill_mode == 'outline' and marker_style.is_filled()
        )
        if use_outline:
            scatter_arguments['facecolors'] = 'none'
            scatter_arguments['edgecolors'] = (
                uniform_track_color
                if track_colormap is None else
                track_colormap(color_norm(color_values))
            )
        elif track_colormap is None:
            scatter_arguments['color'] = uniform_track_color
        else:
            scatter_arguments.update({
                'c': color_values,
                'cmap': track_colormap,
                'norm': color_norm,
            })
        axis.scatter(
            positions[:, 0],
            positions[:, 1],
            **scatter_arguments
        )

    @staticmethod
    def __resolve_bar_colors(
        color: str | Colormap | None,
        bar_count: int,
        default_color: str | Colormap
    ) -> str | list:
        """Resolve a fixed color or a repeatable colormap palette."""
        selected_color = default_color if color is None else color
        if is_color_like(selected_color):
            return selected_color
        if isinstance(selected_color, Colormap):
            color_map = selected_color
        elif isinstance(selected_color, str):
            try:
                color_map = plt.get_cmap(selected_color)
            except ValueError as error:
                raise ValueError(
                    'color must be a Matplotlib-compatible color or '
                    'colormap.'
                ) from error
        else:
            raise TypeError(
                'color must be a Matplotlib-compatible color, colormap name, '
                'Colormap object, or None.'
            )

        listed_colors = getattr(color_map, 'colors', None)
        if (
            listed_colors is not None and
            len(listed_colors) and
            int(color_map.N) <= 20
        ):
            palette = list(listed_colors)
        else:
            palette_size = min(max(int(bar_count), 1), int(color_map.N))
            palette = list(color_map(np.linspace(0, 1, palette_size)))
        return [
            palette[index % len(palette)]
            for index in range(int(bar_count))
        ]

    @staticmethod
    def __normalize_figsize(
        figsize: object,
        argument_name: str
    ) -> tuple[float, float]:
        """Validate and normalize a two-value positive figure size."""
        if (
            not isinstance(figsize, (tuple, list)) or
            len(figsize) != 2
        ):
            raise TypeError(
                f'{argument_name} must contain exactly two numeric values.'
            )
        normalized_values = []
        for value in figsize:
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, float, np.integer, np.floating)
            ):
                raise TypeError(
                    f'{argument_name} must contain exactly two numeric values.'
                )
            normalized_value = float(value)
            if not np.isfinite(normalized_value) or normalized_value <= 0:
                raise ValueError(
                    f'{argument_name} values must be finite and positive.'
                )
            normalized_values.append(normalized_value)
        return normalized_values[0], normalized_values[1]

    @staticmethod
    def __raw_turn_edge_mask(
        raw_point_count: int,
        source_indices: np.ndarray,
        turn_vertex_indices: np.ndarray
    ) -> np.ndarray:
        """Map incoming and outgoing processed turn edges onto raw edges."""
        edge_mask = np.zeros(max(raw_point_count - 1, 0), dtype=bool)
        source_indices = np.asarray(source_indices, dtype=int)
        turn_vertex_indices = np.asarray(turn_vertex_indices, dtype=int)
        for vertex_index in turn_vertex_indices:
            if vertex_index <= 0 or vertex_index >= len(source_indices) - 1:
                continue
            raw_start = int(source_indices[vertex_index - 1])
            raw_stop = int(source_indices[vertex_index + 1])
            raw_start = max(raw_start, 0)
            raw_stop = min(raw_stop, len(edge_mask))
            if raw_stop > raw_start:
                edge_mask[raw_start:raw_stop] = True
        return edge_mask

    @staticmethod
    def __add_highlighted_track_segments(
        axis,
        positions: np.ndarray,
        edge_mask: np.ndarray,
        color: str,
        label: str | None,
        linewidth: float = 2.0,
        zorder: float = 7
    ) -> bool:
        """Overlay selected consecutive trajectory edges on an axis."""
        positions = np.asarray(positions, dtype=float)
        edge_mask = np.asarray(edge_mask, dtype=bool)
        if len(positions) < 2:
            return False
        if edge_mask.shape != (len(positions) - 1,):
            raise ValueError(
                'Internal trajectory edge mask does not match the path.'
            )
        edge_indices = np.flatnonzero(edge_mask)
        if not len(edge_indices):
            return False
        segments = np.stack([
            positions[edge_indices],
            positions[edge_indices + 1],
        ], axis=1)
        finite_segments = np.isfinite(segments).all(axis=(1, 2))
        segments = segments[finite_segments]
        if not len(segments):
            return False
        axis.add_collection(LineCollection(
            segments,
            colors=[color],
            linewidths=linewidth,
            zorder=zorder
        ))
        if label is not None:
            axis.add_line(Line2D(
                [], [], color=color, linewidth=linewidth, label=label
            ))
        return True

    @staticmethod
    def __derive_figure_save_path(
        save_path: str | os.PathLike | None,
        suffix: str
    ) -> str | None:
        """Insert a suffix before a figure filename extension."""
        if save_path is None:
            return None
        requested_path = os.fspath(save_path)
        path_root, path_extension = os.path.splitext(requested_path)
        if path_extension:
            return f'{path_root}{suffix}{path_extension}'
        return f'{requested_path}{suffix}'

    def __finalize_figure(
        self,
        figure,
        save_path: str | None,
        dpi: int,
        show: bool
    ) -> None:
        """Optionally save and show a figure without replacing its directory."""
        self.__validate_positive_integer(dpi, 'dpi')
        if not isinstance(show, bool):
            raise TypeError('show must be a boolean.')
        if save_path is not None:
            if not isinstance(save_path, (str, os.PathLike)):
                raise TypeError('save_path must be a path string or None.')
            requested_path = os.fspath(save_path)
            complete_path = (
                requested_path
                if os.path.isabs(requested_path)
                else os.path.join(self._directory, requested_path)
            )
            output_directory = os.path.dirname(complete_path)
            if output_directory:
                os.makedirs(output_directory, exist_ok=True)
            figure.savefig(complete_path, dpi=dpi, bbox_inches='tight')
            print(f'Figure saved to {complete_path}')
        if show:
            plt.show()

    @staticmethod
    def __validate_sg_filter_parameters(
        window_length: int | None,
        polyorder: int
    ) -> None:
        """Validate Savitzky-Golay parameters before per-segment adaptation."""
        if isinstance(polyorder, bool) or not isinstance(
            polyorder, (int, np.integer)
        ):
            raise TypeError('sg_filter_polyorder must be an integer.')
        if polyorder < 0:
            raise ValueError('sg_filter_polyorder must be nonnegative.')
        if window_length is None:
            raise ValueError(
                'sg_filter_window_length must be provided when '
                "smoothing_method='sg_filter_rdp'."
            )
        if isinstance(window_length, bool) or not isinstance(
            window_length, (int, np.integer)
        ):
            raise TypeError(
                'sg_filter_window_length must be an odd integer.'
            )
        if window_length < 3 or window_length % 2 == 0:
            raise ValueError(
                'sg_filter_window_length must be an odd integer of at least 3.'
            )
        if polyorder >= window_length:
            raise ValueError(
                'sg_filter_polyorder must be smaller than '
                'sg_filter_window_length.'
            )

    @classmethod
    def __normalize_smoothing_configuration(
        cls,
        smoothing_method: str,
        moving_average_window: int,
        triangular_smoothing_window: int | None,
        sg_filter_window_length: int | None,
        sg_filter_polyorder: int,
        rdp_epsilon_pixels: float
    ) -> dict:
        """Validate and retain only settings used by one smoothing method."""
        if not isinstance(smoothing_method, str):
            raise TypeError('smoothing_method must be a string.')
        normalized_method = smoothing_method.strip().lower()
        for separator in ('-', ' ', '+'):
            normalized_method = normalized_method.replace(separator, '_')
        normalized_method = '_'.join(
            part for part in normalized_method.split('_') if part
        )
        if normalized_method not in cls.TRAJECTORY_SMOOTHING_METHODS:
            raise ValueError(
                "smoothing_method must be 'moving_average', "
                "'triangular_smoothing', or 'sg_filter_rdp'."
            )

        configuration = {
            'smoothing_method': normalized_method,
            'moving_average_window': None,
            'triangular_smoothing_window': None,
            'sg_filter_window_length': None,
            'sg_filter_polyorder': None,
            'rdp_epsilon_pixels': None,
        }
        if normalized_method == 'moving_average':
            if isinstance(
                moving_average_window, (bool, np.bool_)
            ) or not isinstance(
                moving_average_window, (int, np.integer)
            ):
                raise TypeError('moving_average_window must be an integer.')
            moving_average_window = int(moving_average_window)
            if moving_average_window < 3:
                raise ValueError(
                    'moving_average_window must be an odd integer of at least '
                    '3 trajectory points.'
                )
            if moving_average_window % 2 == 0:
                raise ValueError(
                    'moving_average_window must be an odd integer.'
                )
            configuration['moving_average_window'] = moving_average_window
        elif normalized_method == 'triangular_smoothing':
            if isinstance(
                triangular_smoothing_window, (bool, np.bool_)
            ) or not isinstance(
                triangular_smoothing_window, (int, np.integer)
            ):
                raise TypeError(
                    'triangular_smoothing_window must be an odd integer.'
                )
            triangular_smoothing_window = int(
                triangular_smoothing_window
            )
            if (
                triangular_smoothing_window < 3 or
                triangular_smoothing_window % 2 == 0
            ):
                raise ValueError(
                    'triangular_smoothing_window must be an odd integer of at '
                    'least 3.'
                )
            configuration['triangular_smoothing_window'] = (
                triangular_smoothing_window
            )
        else:
            cls.__validate_sg_filter_parameters(
                sg_filter_window_length, sg_filter_polyorder
            )
            if isinstance(
                rdp_epsilon_pixels, (bool, np.bool_)
            ) or not isinstance(
                rdp_epsilon_pixels,
                (int, float, np.integer, np.floating)
            ):
                raise TypeError('rdp_epsilon_pixels must be numeric.')
            rdp_epsilon_pixels = float(rdp_epsilon_pixels)
            if (
                not np.isfinite(rdp_epsilon_pixels) or
                rdp_epsilon_pixels < 0
            ):
                raise ValueError(
                    'rdp_epsilon_pixels must be finite and nonnegative.'
                )
            configuration.update({
                'sg_filter_window_length': int(sg_filter_window_length),
                'sg_filter_polyorder': int(sg_filter_polyorder),
                'rdp_epsilon_pixels': rdp_epsilon_pixels,
            })
        return configuration

    @staticmethod
    def __smoothing_configuration_output(
        smoothing_configuration: dict
    ) -> dict:
        """Format selected smoothing settings for dataframe result rows."""
        return {
            key: (
                value if value is not None else np.nan
            )
            for key, value in smoothing_configuration.items()
        }

    @staticmethod
    def __smoothing_trajectory_label(metadata: dict) -> str:
        """Return a plot label for the smoothing method cached in metadata."""
        smoothing_method = metadata.get('smoothing_method')
        return {
            'moving_average': 'Moving-average trajectory',
            'triangular_smoothing': 'Triangular-smoothed trajectory',
            'sg_filter_rdp': 'SG-filtered RDP trajectory',
        }.get(smoothing_method, 'Processed trajectory')

    @staticmethod
    def __split_particle_track(
        particle_rows: pd.DataFrame,
        max_frame_gap: int | None
    ) -> list[pd.DataFrame]:
        """Split a sorted track wherever its frame difference is too large."""
        if max_frame_gap is None or len(particle_rows) < 2:
            return [particle_rows.reset_index(drop=True)]
        frames = particle_rows['frame'].to_numpy(dtype=int)
        segment_ids = np.concatenate([
            [0], np.cumsum(np.diff(frames) > max_frame_gap)
        ])
        return [
            segment_rows.reset_index(drop=True)
            for _, segment_rows in particle_rows.groupby(segment_ids, sort=True)
        ]

    @staticmethod
    def __smooth_positions(
        positions: np.ndarray,
        window_length: int | None,
        polyorder: int
    ) -> tuple[np.ndarray, int | None]:
        """Apply an SG filter, adapting its odd window to short trajectories."""
        if window_length is None or len(positions) < 3:
            return positions.copy(), None
        largest_odd_window = (
            len(positions) if len(positions) % 2 == 1 else len(positions) - 1
        )
        effective_window = min(int(window_length), largest_odd_window)
        if effective_window <= polyorder:
            return positions.copy(), None
        smoothed_positions = savgol_filter(
            positions,
            window_length=effective_window,
            polyorder=polyorder,
            axis=0,
            mode='interp'
        )
        return np.asarray(smoothed_positions, dtype=float), effective_window

    @staticmethod
    def __rdp_keep_mask(points: np.ndarray, epsilon: float) -> np.ndarray:
        """Return an iterative Ramer-Douglas-Peucker point-retention mask."""
        point_count = len(points)
        if point_count <= 2:
            return np.ones(point_count, dtype=bool)
        keep_mask = np.zeros(point_count, dtype=bool)
        keep_mask[[0, -1]] = True
        segment_stack = [(0, point_count - 1)]
        while segment_stack:
            start_index, end_index = segment_stack.pop()
            if end_index <= start_index + 1:
                continue
            segment_start = points[start_index]
            segment_end = points[end_index]
            interior_points = points[start_index + 1:end_index]
            segment_vector = segment_end - segment_start
            segment_length = float(np.linalg.norm(segment_vector))
            if segment_length <= np.finfo(float).eps:
                distances = np.linalg.norm(
                    interior_points - segment_start, axis=1
                )
            else:
                relative_points = interior_points - segment_start
                distances = np.abs(
                    segment_vector[0] * relative_points[:, 1] -
                    segment_vector[1] * relative_points[:, 0]
                ) / segment_length
            furthest_offset = int(np.argmax(distances))
            furthest_distance = float(distances[furthest_offset])
            if furthest_distance > epsilon:
                furthest_index = start_index + 1 + furthest_offset
                keep_mask[furthest_index] = True
                segment_stack.append((start_index, furthest_index))
                segment_stack.append((furthest_index, end_index))
        return keep_mask

    @staticmethod
    def __calculate_signed_turn_angles(positions: np.ndarray) -> np.ndarray:
        """Calculate robust signed angles at processed-path interior vertices."""
        if len(positions) < 3:
            return np.array([], dtype=float)
        directions = np.diff(positions, axis=0)
        first_vectors = directions[:-1]
        second_vectors = directions[1:]
        vector_products = (
            np.linalg.norm(first_vectors, axis=1) *
            np.linalg.norm(second_vectors, axis=1)
        )
        dot_products = np.einsum('ij,ij->i', first_vectors, second_vectors)
        cross_products = (
            first_vectors[:, 0] * second_vectors[:, 1] -
            first_vectors[:, 1] * second_vectors[:, 0]
        )
        angles = np.degrees(np.arctan2(cross_products, dot_products))
        angles[vector_products <= np.finfo(float).eps] = np.nan
        return angles

    @staticmethod
    def __moving_average_smooth_positions(
        positions: np.ndarray,
        window_length: int
    ) -> tuple[np.ndarray, int | None]:
        """Apply a complete centered moving average without edge padding."""
        positions = np.asarray(positions, dtype=float)
        if len(positions) < 3:
            return positions.copy(), None
        largest_odd_window = (
            len(positions) if len(positions) % 2 == 1 else len(positions) - 1
        )
        effective_window = min(int(window_length), largest_odd_window)
        if effective_window < 3:
            return positions.copy(), None
        smoothed_positions = (
            pd.DataFrame(positions)
            .rolling(
                window=effective_window,
                center=True,
                min_periods=effective_window
            )
            .mean()
            .to_numpy(dtype=float)
        )
        return smoothed_positions, effective_window

    def __smooth_analysis_positions(
        self,
        positions: np.ndarray,
        smoothing_configuration: dict,
        rdp_output_mode: str,
        interpolation_coordinates: np.ndarray | None = None
    ) -> dict:
        """Apply a selected smoother with sparse or frame-aligned RDP output."""
        positions = np.asarray(positions, dtype=float)
        if rdp_output_mode not in {'sparse', 'interpolated'}:
            raise ValueError(
                "rdp_output_mode must be 'sparse' or 'interpolated'."
            )
        smoothing_method = smoothing_configuration['smoothing_method']
        if smoothing_method == 'moving_average':
            frame_aligned_positions, effective_window = (
                self.__moving_average_smooth_positions(
                    positions,
                    smoothing_configuration['moving_average_window']
                )
            )
            finite_mask = np.all(
                np.isfinite(frame_aligned_positions), axis=1
            )
            source_indices = np.flatnonzero(finite_mask)
            return {
                'frame_aligned_positions': frame_aligned_positions,
                'analysis_positions': frame_aligned_positions[source_indices],
                'source_indices': source_indices,
                'rdp_source_indices': np.array([], dtype=int),
                'effective_window': effective_window,
            }
        if smoothing_method == 'triangular_smoothing':
            frame_aligned_positions, effective_window = (
                self.__triangular_smooth_positions(
                    positions,
                    smoothing_configuration['triangular_smoothing_window']
                )
            )
            source_indices = np.arange(len(positions), dtype=int)
            return {
                'frame_aligned_positions': frame_aligned_positions,
                'analysis_positions': frame_aligned_positions.copy(),
                'source_indices': source_indices,
                'rdp_source_indices': np.array([], dtype=int),
                'effective_window': effective_window,
            }

        sg_positions, effective_window = self.__smooth_positions(
            positions,
            smoothing_configuration['sg_filter_window_length'],
            smoothing_configuration['sg_filter_polyorder']
        )
        keep_mask = self.__rdp_keep_mask(
            sg_positions,
            epsilon=smoothing_configuration['rdp_epsilon_pixels']
        )
        retained_indices = np.flatnonzero(keep_mask)
        retained_positions = sg_positions[retained_indices]
        if rdp_output_mode == 'sparse':
            return {
                'frame_aligned_positions': sg_positions,
                'analysis_positions': retained_positions,
                'source_indices': retained_indices,
                'rdp_source_indices': retained_indices.copy(),
                'effective_window': effective_window,
            }

        source_indices = np.arange(len(positions), dtype=int)
        if len(retained_indices) < 2:
            reconstructed_positions = sg_positions.copy()
        else:
            if interpolation_coordinates is None:
                interpolation_coordinates = source_indices.astype(float)
            interpolation_coordinates = np.asarray(
                interpolation_coordinates, dtype=float
            )
            if interpolation_coordinates.shape != (len(positions),):
                raise ValueError(
                    'interpolation_coordinates must have one value per '
                    'trajectory point.'
                )
            if (
                not np.all(np.isfinite(interpolation_coordinates)) or
                np.any(np.diff(interpolation_coordinates) <= 0)
            ):
                raise ValueError(
                    'interpolation_coordinates must be finite and increase '
                    'strictly.'
                )
            reconstructed_positions = np.column_stack([
                np.interp(
                    interpolation_coordinates,
                    interpolation_coordinates[retained_indices],
                    retained_positions[:, dimension]
                )
                for dimension in range(positions.shape[1])
            ])
        return {
            'frame_aligned_positions': reconstructed_positions,
            'analysis_positions': reconstructed_positions.copy(),
            'source_indices': source_indices,
            'rdp_source_indices': retained_indices,
            'effective_window': effective_window,
        }

    def __calculate_five_point_velocities(
        self,
        positions: np.ndarray,
        frames: np.ndarray
    ) -> np.ndarray:
        """
        Calculate five-point velocities for uniform or uneven frame spacing.

        For consecutive frames this is exactly the fourth-order central
        difference used by Turner et al. For retained uneven spacing, local
        finite-difference weights are solved from the actual frame offsets.
        """
        positions = np.asarray(positions, dtype=float)
        frames = np.asarray(frames, dtype=float)
        velocities = np.full_like(positions, np.nan, dtype=float)
        if len(positions) < 5:
            return velocities

        derivative_target = np.array([0, 1, 0, 0, 0], dtype=float)
        weights_by_offsets: dict[tuple[float, ...], np.ndarray] = {}
        for center_index in range(2, len(positions) - 2):
            local_slice = slice(center_index - 2, center_index + 3)
            frame_offsets = frames[local_slice] - frames[center_index]
            offset_key = tuple(float(value) for value in frame_offsets)
            weights = weights_by_offsets.get(offset_key)
            if weights is None:
                coefficient_matrix = np.vstack([
                    frame_offsets**power for power in range(5)
                ])
                try:
                    weights = np.linalg.solve(
                        coefficient_matrix, derivative_target
                    )
                except np.linalg.LinAlgError:
                    continue
                weights_by_offsets[offset_key] = weights
            velocities[center_index] = (
                weights @ positions[local_slice]
            ) * self._capture_speed_in_fps
        return velocities

    @staticmethod
    def __angle_between_vectors(
        first_vector: np.ndarray,
        second_vector: np.ndarray
    ) -> float:
        """Return the unsigned angle between two vectors in degrees."""
        first_vector = np.asarray(first_vector, dtype=float)
        second_vector = np.asarray(second_vector, dtype=float)
        magnitude_product = (
            float(np.linalg.norm(first_vector)) *
            float(np.linalg.norm(second_vector))
        )
        if (
            not np.isfinite(magnitude_product) or
            magnitude_product <= np.finfo(float).eps
        ):
            return np.nan
        cosine = float(np.dot(first_vector, second_vector)) / magnitude_product
        return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))

    @staticmethod
    def __contiguous_true_ranges(
        mask: np.ndarray
    ) -> list[tuple[int, int]]:
        """Return inclusive index bounds for each contiguous True region."""
        mask = np.asarray(mask, dtype=bool)
        true_indices = np.flatnonzero(mask)
        if not len(true_indices):
            return []
        split_locations = np.flatnonzero(np.diff(true_indices) > 1) + 1
        groups = np.split(true_indices, split_locations)
        return [
            (int(group[0]), int(group[-1]))
            for group in groups if len(group)
        ]

    def __classify_tumble_states(
        self,
        velocities: np.ndarray,
        threshold_degrees: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Apply the paper's run, tumble, and singleton sequence rules."""
        point_count = len(velocities)
        angle_values = np.full(point_count, np.nan, dtype=float)
        for point_index in range(2, point_count - 3):
            angle_values[point_index] = self.__angle_between_vectors(
                velocities[point_index],
                velocities[point_index + 1]
            )
        valid_angles = np.isfinite(angle_values)
        threshold_tolerance = max(
            1e-10, abs(float(threshold_degrees)) * 1e-12
        )
        exceeds_threshold = valid_angles & (
            angle_values > threshold_degrees + threshold_tolerance
        )
        states = np.full(point_count, 'unclassified', dtype=object)
        singleton_confirmation_angles = np.full(
            point_count, np.nan, dtype=float
        )

        for start_index, end_index in self.__contiguous_true_ranges(
            exceeds_threshold
        ):
            if end_index - start_index + 1 >= 2:
                states[start_index:end_index + 1] = 'tumble'
                continue
            point_index = start_index
            if not 1 <= point_index < point_count - 2:
                continue
            preceding_vector = (
                velocities[point_index - 1] +
                velocities[point_index]
            )
            following_vector = (
                velocities[point_index + 1] +
                velocities[point_index + 2]
            )
            confirmation_angle = self.__angle_between_vectors(
                preceding_vector, following_vector
            )
            singleton_confirmation_angles[point_index] = confirmation_angle
            if (
                np.isfinite(confirmation_angle) and
                confirmation_angle > threshold_degrees + threshold_tolerance
            ):
                states[point_index] = 'tumble'

        run_candidates = valid_angles & ~exceeds_threshold
        for start_index, end_index in self.__contiguous_true_ranges(
            run_candidates
        ):
            if end_index - start_index + 1 >= 3:
                states[start_index:end_index + 1] = 'run'
        return (
            angle_values,
            exceeds_threshold,
            states,
            singleton_confirmation_angles
        )

    def __analyze_tumble_segment(
        self,
        particle_id: int,
        segment_id: int,
        segment_rows: pd.DataFrame,
        elapsed_time_offset_seconds: float,
        starting_event_id: int,
        threshold_degrees: float,
        smoothing_configuration: dict,
        distance_factor: float,
        distance_unit: str,
        speed_unit: str
    ) -> tuple[list[dict], list[dict], int, dict]:
        """Classify runs and tumbles within one uninterrupted track segment."""
        frames = segment_rows['frame'].to_numpy(dtype=int)
        raw_positions = np.column_stack([
            segment_rows['centroid_y'].to_numpy(dtype=float),
            -segment_rows['centroid_x'].to_numpy(dtype=float),
        ])
        smoothing_result = self.__smooth_analysis_positions(
            raw_positions,
            smoothing_configuration=smoothing_configuration,
            rdp_output_mode='interpolated',
            interpolation_coordinates=frames
        )
        smoothed_positions = smoothing_result['frame_aligned_positions']
        effective_window = smoothing_result['effective_window']
        smoothing_method = smoothing_configuration['smoothing_method']
        velocities_pixels_per_second = (
            self.__calculate_five_point_velocities(
                smoothed_positions, frames
            )
        )
        velocities = velocities_pixels_per_second * distance_factor
        speeds = np.linalg.norm(velocities, axis=1)

        (
            angle_values,
            exceeds_threshold,
            states,
            singleton_confirmation_angles,
        ) = self.__classify_tumble_states(
            velocities, threshold_degrees=threshold_degrees
        )
        valid_angles = np.isfinite(angle_values)

        event_id_by_point: list[int | None] = [None] * len(frames)
        internal_events = []
        next_event_id = starting_event_id
        point_index = 0
        while point_index < len(frames):
            event_type = str(states[point_index])
            if event_type not in {'run', 'tumble'}:
                point_index += 1
                continue
            event_end_index = point_index
            while (
                event_end_index + 1 < len(frames) and
                states[event_end_index + 1] == event_type
            ):
                event_end_index += 1
            event_indices = np.arange(
                point_index, event_end_index + 1, dtype=int
            )
            for event_point_index in event_indices:
                event_id_by_point[event_point_index] = next_event_id
            event_speeds = speeds[event_indices]
            event_angles = angle_values[event_indices]
            event_positions = smoothed_positions[
                point_index:event_end_index + 1
            ]
            event_path_length = float(np.linalg.norm(
                np.diff(event_positions, axis=0), axis=1
            ).sum()) * distance_factor
            direction_change_within_run = np.nan
            if event_type == 'run' and len(event_indices) >= 3:
                direction_change_within_run = (
                    self.__angle_between_vectors(
                        velocities[event_indices[:3]].sum(axis=0),
                        velocities[event_indices[-3:]].sum(axis=0)
                    )
                )
            internal_events.append({
                'source_dataframe': self._resolved_source_dataframe,
                'particle': particle_id,
                'segment_id': segment_id,
                'event_id': next_event_id,
                'event_type': event_type,
                'start_frame': int(frames[point_index]),
                'end_frame': int(frames[event_end_index]),
                'support_end_frame': int(frames[event_end_index + 1]),
                'point_count': int(len(event_indices)),
                'interval_seconds': (
                    int(frames[event_end_index]) -
                    int(frames[point_index])
                ) / self._capture_speed_in_fps,
                'sampling_support_seconds': (
                    int(frames[event_end_index + 1]) -
                    int(frames[point_index])
                ) / self._capture_speed_in_fps,
                'path_length': event_path_length,
                'mean_speed': self.__safe_mean(event_speeds),
                'std_speed': self.__safe_std(event_speeds),
                'mean_angular_speed_degrees_per_point': self.__safe_mean(
                    event_angles
                ),
                'std_angular_speed_degrees_per_point': self.__safe_std(
                    event_angles
                ),
                'direction_change_within_run_degrees': (
                    direction_change_within_run
                ),
                'run_to_run_direction_change_degrees': np.nan,
                'preceding_run_event_id': pd.NA,
                'following_run_event_id': pd.NA,
                'distance_unit': distance_unit,
                'speed_unit': speed_unit,
                'smoothing_method': smoothing_method,
                'effective_smoothing_window': (
                    effective_window
                    if effective_window is not None
                    else np.nan
                ),
                'tumble_threshold_angle_degrees': threshold_degrees,
                '_indices': event_indices,
            })
            next_event_id += 1
            point_index = event_end_index + 1

        for event_index, event in enumerate(internal_events):
            if event['event_type'] != 'tumble':
                continue
            preceding_run = (
                internal_events[event_index - 1]
                if event_index > 0
                else None
            )
            following_run = (
                internal_events[event_index + 1]
                if event_index + 1 < len(internal_events)
                else None
            )
            if preceding_run is None or following_run is None:
                continue
            if (
                preceding_run['event_type'] != 'run' or
                following_run['event_type'] != 'run'
            ):
                continue
            preceding_indices = preceding_run['_indices']
            tumble_indices = event['_indices']
            following_indices = following_run['_indices']
            if (
                len(preceding_indices) < 3 or
                len(following_indices) < 3 or
                int(preceding_indices[-1]) + 1 != int(tumble_indices[0]) or
                int(tumble_indices[-1]) + 1 != int(following_indices[0])
            ):
                continue
            event['preceding_run_event_id'] = int(
                preceding_run['event_id']
            )
            event['following_run_event_id'] = int(
                following_run['event_id']
            )
            event['run_to_run_direction_change_degrees'] = (
                self.__angle_between_vectors(
                    velocities[preceding_indices[-3:]].sum(axis=0),
                    velocities[following_indices[:3]].sum(axis=0)
                )
            )

        point_rows = []
        for point_index in np.flatnonzero(valid_angles):
            point_rows.append({
                'source_dataframe': self._resolved_source_dataframe,
                'particle': particle_id,
                'segment_id': segment_id,
                'frame': int(frames[point_index]),
                'next_frame': int(frames[point_index + 1]),
                'frame_delta_to_next': int(
                    frames[point_index + 1] - frames[point_index]
                ),
                'elapsed_time_seconds': (
                    elapsed_time_offset_seconds +
                    (
                        int(frames[point_index]) -
                        int(frames[0])
                    ) / self._capture_speed_in_fps
                ),
                'raw_position_x_pixels': float(
                    raw_positions[point_index, 0]
                ),
                'raw_position_y_pixels': float(
                    raw_positions[point_index, 1]
                ),
                'smoothed_position_x_pixels': float(
                    smoothed_positions[point_index, 0]
                ),
                'smoothed_position_y_pixels': float(
                    smoothed_positions[point_index, 1]
                ),
                'velocity_x': float(velocities[point_index, 0]),
                'velocity_y': float(velocities[point_index, 1]),
                'instantaneous_speed': float(speeds[point_index]),
                'turn_angle_degrees': float(angle_values[point_index]),
                'exceeds_tumble_threshold': bool(
                    exceeds_threshold[point_index]
                ),
                'singleton_confirmation_angle_degrees': float(
                    singleton_confirmation_angles[point_index]
                ),
                'state': str(states[point_index]),
                'event_id': (
                    int(event_id_by_point[point_index])
                    if event_id_by_point[point_index] is not None
                    else pd.NA
                ),
                'distance_unit': distance_unit,
                'speed_unit': speed_unit,
                'smoothing_method': smoothing_method,
                'effective_smoothing_window': (
                    effective_window
                    if effective_window is not None
                    else np.nan
                ),
                'tumble_threshold_angle_degrees': threshold_degrees,
            })
        event_rows = [
            {
                key: value
                for key, value in event.items()
                if not key.startswith('_')
            }
            for event in internal_events
        ]
        path_data = {
            'segment_id': segment_id,
            'frames': frames.copy(),
            'elapsed_time_seconds': (
                elapsed_time_offset_seconds +
                (frames - frames[0]) / self._capture_speed_in_fps
            ),
            'raw_positions_pixels': raw_positions.copy(),
            'smoothed_positions_pixels': smoothed_positions.copy(),
            'states': states.copy(),
            'effective_smoothing_window': effective_window,
            'smoothing_method': smoothing_method,
            'rdp_source_indices': smoothing_result['rdp_source_indices'],
        }
        return point_rows, event_rows, next_event_id, path_data

    def __summarize_tumble_particle(
        self,
        particle_id: int,
        original_frames: np.ndarray,
        analyzed_frames: np.ndarray,
        discarded_count: int,
        segment_count: int,
        analyzed_tracking_time_seconds: float,
        particle_point_rows: list[dict],
        particle_event_rows: list[dict],
        distance_unit: str,
        speed_unit: str,
        smoothing_configuration: dict,
        discard_initial_frames: int,
        discard_final_frames: int,
        tumble_threshold_angle: float,
        max_frame_gap: int | None
    ) -> dict:
        """Summarize one particle's classified run and tumble events."""
        event_dataframe = pd.DataFrame(particle_event_rows)
        if event_dataframe.empty:
            run_events = pd.DataFrame()
            tumble_events = pd.DataFrame()
        else:
            run_events = event_dataframe.loc[
                event_dataframe['event_type'] == 'run'
            ]
            tumble_events = event_dataframe.loc[
                event_dataframe['event_type'] == 'tumble'
            ]
        point_states = [
            row['state'] for row in particle_point_rows
        ]
        classified_angle_count = sum(
            state in {'run', 'tumble'} for state in point_states
        )
        number_of_runs = len(run_events)
        number_of_tumbles = len(tumble_events)
        total_run_interval = (
            float(run_events['interval_seconds'].sum())
            if not run_events.empty else 0.0
        )
        total_tumble_interval = (
            float(tumble_events['interval_seconds'].sum())
            if not tumble_events.empty else 0.0
        )

        def event_values(
            dataframe: pd.DataFrame,
            column: str
        ) -> np.ndarray:
            if dataframe.empty:
                return np.array([], dtype=float)
            return pd.to_numeric(
                dataframe[column], errors='coerce'
            ).to_numpy(dtype=float)

        run_speeds = event_values(run_events, 'mean_speed')
        run_point_speeds = self.__finite_numeric_values(np.asarray([
            row.get('instantaneous_speed', np.nan)
            for row in particle_point_rows
            if row.get('state') == 'run'
        ], dtype=object))
        tumble_speeds = event_values(tumble_events, 'mean_speed')
        run_intervals = event_values(run_events, 'interval_seconds')
        tumble_intervals = event_values(tumble_events, 'interval_seconds')
        run_angular_speeds = event_values(
            run_events, 'mean_angular_speed_degrees_per_point'
        )
        tumble_angular_speeds = event_values(
            tumble_events, 'mean_angular_speed_degrees_per_point'
        )
        within_run_changes = event_values(
            run_events, 'direction_change_within_run_degrees'
        )
        run_to_run_changes = event_values(
            tumble_events, 'run_to_run_direction_change_degrees'
        )
        tumble_start_intervals = []
        if not tumble_events.empty:
            for _, segment_tumbles in tumble_events.groupby(
                'segment_id', sort=True
            ):
                start_frames = (
                    segment_tumbles.sort_values(
                        'start_frame', kind='stable'
                    )['start_frame'].to_numpy(dtype=float)
                )
                if len(start_frames) > 1:
                    tumble_start_intervals.extend(
                        np.diff(start_frames) /
                        self._capture_speed_in_fps
                    )
        tumble_start_intervals = np.asarray(
            tumble_start_intervals, dtype=float
        )

        if not len(analyzed_frames):
            analysis_status = 'no_detections_after_discard'
        elif not particle_point_rows:
            analysis_status = 'insufficient_points_for_five_point_velocity'
        elif not classified_angle_count:
            analysis_status = 'no_runs_or_tumbles_classified'
        else:
            analysis_status = 'complete'

        smoothing_output = self.__smoothing_configuration_output(
            smoothing_configuration
        )
        return {
            'source_dataframe': self._resolved_source_dataframe,
            'particle': particle_id,
            'original_first_frame': int(original_frames[0]),
            'original_last_frame': int(original_frames[-1]),
            'original_detection_count': int(len(original_frames)),
            'discarded_detection_count': int(discarded_count),
            'analyzed_first_frame': (
                int(analyzed_frames[0]) if len(analyzed_frames) else np.nan
            ),
            'analyzed_last_frame': (
                int(analyzed_frames[-1]) if len(analyzed_frames) else np.nan
            ),
            'analyzed_detection_count': int(len(analyzed_frames)),
            'segment_count': int(segment_count),
            'analyzable_angle_count': int(len(particle_point_rows)),
            'classified_angle_count': int(classified_angle_count),
            'unclassified_angle_count': int(
                len(particle_point_rows) - classified_angle_count
            ),
            'number_of_runs': int(number_of_runs),
            'number_of_tumbles': int(number_of_tumbles),
            'runs_per_tumble': (
                number_of_runs / number_of_tumbles
                if number_of_tumbles else np.nan
            ),
            'analyzed_tracking_time_seconds': float(
                analyzed_tracking_time_seconds
            ),
            'total_run_interval_seconds': total_run_interval,
            'total_tumble_interval_seconds': total_tumble_interval,
            'tumble_time_fraction': (
                total_tumble_interval / analyzed_tracking_time_seconds
                if analyzed_tracking_time_seconds > 0 else np.nan
            ),
            'tumble_frequency_per_second': (
                number_of_tumbles / analyzed_tracking_time_seconds
                if analyzed_tracking_time_seconds > 0 else np.nan
            ),
            'mean_run_speed': self.__safe_mean(run_speeds),
            'std_run_mean_speed_between_events': self.__safe_std(run_speeds),
            'mean_run_point_speed': self.__safe_mean(run_point_speeds),
            'std_run_point_speed': self.__safe_std(run_point_speeds),
            'run_speed_point_count': int(len(run_point_speeds)),
            'mean_tumble_speed': self.__safe_mean(tumble_speeds),
            'std_tumble_speed': self.__safe_std(tumble_speeds),
            'mean_run_interval_seconds': self.__safe_mean(run_intervals),
            'std_run_interval_seconds': self.__safe_std(run_intervals),
            'mean_tumble_interval_seconds': self.__safe_mean(
                tumble_intervals
            ),
            'std_tumble_interval_seconds': self.__safe_std(
                tumble_intervals
            ),
            'mean_time_between_tumble_starts_seconds': self.__safe_mean(
                tumble_start_intervals
            ),
            'std_time_between_tumble_starts_seconds': self.__safe_std(
                tumble_start_intervals
            ),
            'mean_run_angular_speed_degrees_per_point': self.__safe_mean(
                run_angular_speeds
            ),
            'std_run_angular_speed_degrees_per_point': self.__safe_std(
                run_angular_speeds
            ),
            'mean_tumble_angular_speed_degrees_per_point': self.__safe_mean(
                tumble_angular_speeds
            ),
            'std_tumble_angular_speed_degrees_per_point': self.__safe_std(
                tumble_angular_speeds
            ),
            'mean_direction_change_within_runs_degrees': self.__safe_mean(
                within_run_changes
            ),
            'std_direction_change_within_runs_degrees': self.__safe_std(
                within_run_changes
            ),
            'mean_run_to_run_direction_change_degrees': self.__safe_mean(
                run_to_run_changes
            ),
            'std_run_to_run_direction_change_degrees': self.__safe_std(
                run_to_run_changes
            ),
            'distance_unit': distance_unit,
            'speed_unit': speed_unit,
            **smoothing_output,
            'discard_initial_frames': discard_initial_frames,
            'discard_final_frames': discard_final_frames,
            'tumble_threshold_angle_degrees': tumble_threshold_angle,
            'max_frame_gap': (
                max_frame_gap if max_frame_gap is not None else np.nan
            ),
            'analysis_status': analysis_status,
        }

    def __summarize_tumble_population(
        self,
        tumble_summary: pd.DataFrame,
        distance_unit: str,
        speed_unit: str,
        smoothing_configuration: dict,
        discard_initial_frames: int,
        discard_final_frames: int,
        tumble_threshold_angle: float,
        max_frame_gap: int | None
    ) -> pd.DataFrame:
        """Summarize particle-level metrics with equal weight per particle."""
        def summary_values(column: str) -> np.ndarray:
            if tumble_summary.empty:
                return np.array([], dtype=float)
            return pd.to_numeric(
                tumble_summary[column], errors='coerce'
            ).to_numpy(dtype=float)

        smoothing_output = self.__smoothing_configuration_output(
            smoothing_configuration
        )
        population_row = {
            'source_dataframe': self._resolved_source_dataframe,
            'particle_count': int(len(tumble_summary)),
            'particles_with_analyzable_angles': int(
                (tumble_summary['analyzable_angle_count'] > 0).sum()
            ) if not tumble_summary.empty else 0,
            'particles_with_tumbles': int(
                (tumble_summary['number_of_tumbles'] > 0).sum()
            ) if not tumble_summary.empty else 0,
            'total_number_of_runs': int(
                tumble_summary['number_of_runs'].sum()
            ) if not tumble_summary.empty else 0,
            'total_number_of_tumbles': int(
                tumble_summary['number_of_tumbles'].sum()
            ) if not tumble_summary.empty else 0,
            'total_analyzed_tracking_time_seconds': float(
                tumble_summary['analyzed_tracking_time_seconds'].sum()
            ) if not tumble_summary.empty else 0.0,
            'mean_tumble_frequency_per_second': self.__safe_mean(
                summary_values('tumble_frequency_per_second')
            ),
            'std_tumble_frequency_per_second': self.__safe_std(
                summary_values('tumble_frequency_per_second')
            ),
            'mean_number_of_runs_per_particle': self.__safe_mean(
                summary_values('number_of_runs')
            ),
            'std_number_of_runs_per_particle': self.__safe_std(
                summary_values('number_of_runs')
            ),
            'mean_number_of_tumbles_per_particle': self.__safe_mean(
                summary_values('number_of_tumbles')
            ),
            'std_number_of_tumbles_per_particle': self.__safe_std(
                summary_values('number_of_tumbles')
            ),
            'mean_run_speed': self.__safe_mean(
                summary_values('mean_run_speed')
            ),
            'std_run_speed_between_particles': self.__safe_std(
                summary_values('mean_run_speed')
            ),
            'mean_tumble_speed': self.__safe_mean(
                summary_values('mean_tumble_speed')
            ),
            'std_tumble_speed_between_particles': self.__safe_std(
                summary_values('mean_tumble_speed')
            ),
            'mean_run_interval_seconds': self.__safe_mean(
                summary_values('mean_run_interval_seconds')
            ),
            'std_run_interval_seconds_between_particles': self.__safe_std(
                summary_values('mean_run_interval_seconds')
            ),
            'mean_tumble_interval_seconds': self.__safe_mean(
                summary_values('mean_tumble_interval_seconds')
            ),
            'std_tumble_interval_seconds_between_particles': self.__safe_std(
                summary_values('mean_tumble_interval_seconds')
            ),
            'mean_time_between_tumble_starts_seconds': self.__safe_mean(
                summary_values('mean_time_between_tumble_starts_seconds')
            ),
            'std_time_between_tumble_starts_seconds_between_particles': (
                self.__safe_std(
                    summary_values(
                        'mean_time_between_tumble_starts_seconds'
                    )
                )
            ),
            'mean_run_angular_speed_degrees_per_point': self.__safe_mean(
                summary_values(
                    'mean_run_angular_speed_degrees_per_point'
                )
            ),
            'std_run_angular_speed_degrees_per_point_between_particles': (
                self.__safe_std(
                    summary_values(
                        'mean_run_angular_speed_degrees_per_point'
                    )
                )
            ),
            'mean_tumble_angular_speed_degrees_per_point': self.__safe_mean(
                summary_values(
                    'mean_tumble_angular_speed_degrees_per_point'
                )
            ),
            'std_tumble_angular_speed_degrees_per_point_between_particles': (
                self.__safe_std(
                    summary_values(
                        'mean_tumble_angular_speed_degrees_per_point'
                    )
                )
            ),
            'mean_direction_change_within_runs_degrees': self.__safe_mean(
                summary_values(
                    'mean_direction_change_within_runs_degrees'
                )
            ),
            'std_direction_change_within_runs_degrees_between_particles': (
                self.__safe_std(
                    summary_values(
                        'mean_direction_change_within_runs_degrees'
                    )
                )
            ),
            'mean_run_to_run_direction_change_degrees': self.__safe_mean(
                summary_values(
                    'mean_run_to_run_direction_change_degrees'
                )
            ),
            'std_run_to_run_direction_change_degrees_between_particles': (
                self.__safe_std(
                    summary_values(
                        'mean_run_to_run_direction_change_degrees'
                    )
                )
            ),
            'distance_unit': distance_unit,
            'speed_unit': speed_unit,
            **smoothing_output,
            'discard_initial_frames': discard_initial_frames,
            'discard_final_frames': discard_final_frames,
            'tumble_threshold_angle_degrees': tumble_threshold_angle,
            'max_frame_gap': (
                max_frame_gap if max_frame_gap is not None else np.nan
            ),
        }
        return pd.DataFrame(
            [population_row], columns=self.TUMBLE_POPULATION_COLUMNS
        )

    def __build_angle_tumble_table_1(
        self,
        tumble_summary: pd.DataFrame,
        selected_dataframe: pd.DataFrame,
        distance_factor: float,
        distance_unit: str,
        speed_unit: str,
        discard_initial_frames: int,
        discard_final_frames: int
    ) -> pd.DataFrame:
        """Build a vertical Turner-style equal-cell population table."""
        def summary_values(column_name: str) -> np.ndarray:
            if tumble_summary.empty:
                return np.array([], dtype=float)
            return self.__finite_numeric_values(
                tumble_summary[column_name]
            )

        def append_distribution_rows(
            rows: list[dict],
            label: str,
            values: np.ndarray
        ) -> None:
            rows.extend([
                {
                    'Parameter': label,
                    'Value': self.__safe_mean(values),
                },
                {
                    'Parameter': '±std',
                    'Value': self.__safe_std(values),
                },
                {
                    'Parameter': '±sem',
                    'Value': self.__safe_sem(values),
                },
            ])

        trimmed_dataframe = self.__trim_particle_frame_windows(
            selected_dataframe,
            discard_initial_frames=discard_initial_frames,
            discard_final_frames=discard_final_frames
        )
        morphology_columns = ['minor_axis_length', 'major_axis_length']
        missing_morphology_columns = [
            column_name
            for column_name in morphology_columns
            if column_name not in trimmed_dataframe.columns
        ]
        if missing_morphology_columns:
            warnings.warn(
                'Turner-style body size values are unavailable because the '
                'selected dataframe is missing morphology columns: '
                f'{missing_morphology_columns}. Values are reported as NaN.',
                UserWarning,
                stacklevel=3
            )

        def equal_cell_axis_mean(column_name: str) -> float:
            if column_name not in trimmed_dataframe.columns:
                return np.nan
            particle_means = []
            for _, particle_rows in trimmed_dataframe.groupby(
                'particle', sort=True
            ):
                axis_values = self.__finite_numeric_values(
                    particle_rows[column_name]
                ) * distance_factor
                particle_means.append(self.__safe_mean(axis_values))
            return self.__safe_mean(np.asarray(particle_means, dtype=float))

        number_of_cells = int(len(tumble_summary))
        number_of_cells_that_tumbled = (
            int((pd.to_numeric(
                tumble_summary['number_of_tumbles'], errors='coerce'
            ).fillna(0) > 0).sum())
            if not tumble_summary.empty else 0
        )
        total_number_of_runs = (
            int(pd.to_numeric(
                tumble_summary['number_of_runs'], errors='coerce'
            ).fillna(0).sum())
            if not tumble_summary.empty else 0
        )
        total_number_of_tumbles = (
            int(pd.to_numeric(
                tumble_summary['number_of_tumbles'], errors='coerce'
            ).fillna(0).sum())
            if not tumble_summary.empty else 0
        )
        total_tracking_time = float(np.sum(summary_values(
            'analyzed_tracking_time_seconds'
        )))
        rows: list[dict] = [
            {'Parameter': 'Number of cells', 'Value': number_of_cells},
            {
                'Parameter': 'Number of cells that tumbled',
                'Value': number_of_cells_that_tumbled,
            },
        ]
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
            (
                'Tumble frequency (s^-1)',
                'tumble_frequency_per_second',
            ),
        ]
        for label, column_name in table_metrics:
            append_distribution_rows(
                rows, label, summary_values(column_name)
            )
        rows.extend([
            {
                'Parameter': 'Number of events (runs, tumbles)',
                'Value': f'{total_number_of_runs}, {total_number_of_tumbles}',
            },
            {
                'Parameter': 'Total tracking time (s)',
                'Value': total_tracking_time,
            },
            {
                'Parameter': f'Body diameter ({distance_unit})',
                'Value': equal_cell_axis_mean('minor_axis_length'),
            },
            {
                'Parameter': f'Body length ({distance_unit})',
                'Value': equal_cell_axis_mean('major_axis_length'),
            },
        ])
        return pd.DataFrame(rows, columns=self.TABLE_1_VERTICAL_COLUMNS)

    def __add_strain_field(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Return an export-ready copy with strain as its first column."""
        output = dataframe.copy()
        if 'strain' in output.columns:
            output['strain'] = self._strain
            ordered_columns = [
                'strain',
                *[column for column in output.columns if column != 'strain'],
            ]
            return output[ordered_columns]
        output.insert(0, 'strain', self._strain)
        return output

    @staticmethod
    def __normalize_angle_mode(angle_mode: str) -> str:
        """Normalize absolute/signed turn-angle display modes."""
        if not isinstance(angle_mode, str):
            raise TypeError('angle_mode must be a string.')
        normalized_mode = angle_mode.strip().lower()
        if normalized_mode not in {'absolute', 'signed'}:
            raise ValueError("angle_mode must be 'absolute' or 'signed'.")
        return normalized_mode

    def __get_particle_data(
        self,
        particle: int,
        source_dataframe: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        """
        Get data for a specific particle.

        Args:
            particle (int): The particle ID.
            source_dataframe (pd.DataFrame | None): Optional preselected source
                rows. None uses the complete Stats source snapshot.

        Returns:
            pd.DataFrame: DataFrame containing data for the specified particle.
        """
        dataframe = (
            self._sorted_dataframe
            if source_dataframe is None
            else source_dataframe
        )
        return dataframe.loc[
            dataframe['particle'] == particle,
            ['centroid_x', 'centroid_y', 'frame', 'particle']
        ].sort_values(by='frame', kind='stable').copy()

    def __calculate_speed(
        self,
        particle_data: pd.DataFrame,
        speed_unit_key: str
    ) -> np.ndarray:
        """
        Calculate the speed of a particle.

        Args:
            particle_data (pd.DataFrame): DataFrame containing data for a particle.
            speed_unit_key (str): Normalized calculation mode for the requested
                speed unit.

        Returns:
            np.ndarray: Array of speeds for the particle.
        """
        # A speed requires an interval between two detections.
        if len(particle_data) < 2:
            return np.array([], dtype=float)

        x = particle_data['centroid_x'].to_numpy()
        y = particle_data['centroid_y'].to_numpy()
        x_diff = np.diff(x)
        y_diff = np.diff(y)
        distance = np.sqrt(x_diff**2 + y_diff**2)
        frame_difference = np.diff(
            particle_data['frame'].to_numpy(dtype=float)
        )
        if np.any(frame_difference <= 0):
            raise ValueError(
                'Particle frames must increase strictly when calculating speed.'
            )

        if speed_unit_key == 'pixels_per_frame':
            speed = distance / frame_difference
        elif speed_unit_key == 'pixels_per_second':
            self.__validate_frame_rate()
            speed = distance / (
                frame_difference / self._capture_speed_in_fps
            )
        elif speed_unit_key == 'scale_units_per_frame':
            self.__validate_pixel_scale_factor()
            speed = distance * self.pixel_scale_factor / frame_difference
        else:
            self.__validate_frame_rate()
            self.__validate_pixel_scale_factor()
            speed = (
                distance * self.pixel_scale_factor /
                (frame_difference / self._capture_speed_in_fps)
            )

        return speed

    def __normalize_speed_unit(self, speed_unit: str) -> tuple[str, str]:
        """Normalize a requested speed unit into a calculation mode and label."""
        if not isinstance(speed_unit, str):
            raise TypeError('speed_unit must be a string.')
        normalized_unit = speed_unit.strip().lower()
        if normalized_unit.startswith('(') and normalized_unit.endswith(')'):
            normalized_unit = normalized_unit[1:-1].strip()
        normalized_unit = normalized_unit.replace('μ', 'µ')

        pixel_frame_aliases = {
            'pixels/frame', 'pixel/frame', 'px/frame',
            'pixels per frame', 'pixel per frame', 'px per frame'
        }
        pixel_second_aliases = {
            'pixels/s', 'pixel/s', 'px/s',
            'pixels/second', 'pixel/second', 'px/second',
            'pixels per second', 'pixel per second', 'px per second'
        }
        configured_scale_unit = self._scale_units.lower().replace('μ', 'µ')
        scale_frame_aliases = {
            f'{configured_scale_unit}/frame',
            f'{configured_scale_unit} per frame',
            'scale_unit/frame', 'scale_units/frame',
            'scale_unit per frame', 'scale_units per frame'
        }
        scale_second_aliases = {
            f'{configured_scale_unit}/s',
            f'{configured_scale_unit}/second',
            f'{configured_scale_unit} per second',
            'scale_unit/s', 'scale_units/s',
            'scale_unit/second', 'scale_units/second',
            'scale_unit per second', 'scale_units per second'
        }
        micrometer_units = {
            'µm', 'um', 'micrometer', 'micrometers',
            'micrometre', 'micrometres'
        }
        if configured_scale_unit in micrometer_units:
            scale_frame_aliases.update({
                'µm/frame', 'um/frame', 'µm per frame', 'um per frame'
            })
            scale_second_aliases.update({
                'µm/s', 'um/s', 'µm/second', 'um/second',
                'µm per second', 'um per second'
            })

        if normalized_unit in pixel_frame_aliases:
            return 'pixels_per_frame', 'pixels/frame'
        if normalized_unit in pixel_second_aliases:
            return 'pixels_per_second', 'pixels/s'
        if normalized_unit in scale_frame_aliases:
            return 'scale_units_per_frame', f'{self._scale_units}/frame'
        if normalized_unit in scale_second_aliases:
            return 'scale_units_per_second', f'{self._scale_units}/s'
        raise ValueError(
            'speed_unit must describe pixels/frame, pixels/s, '
            'scale_units/frame, or scale_units/s. The generic scale_units '
            f'aliases resolve to {self._scale_units}.'
        )

    @staticmethod
    def __coerce_optional_number(value: object) -> float:
        """Convert capture metadata to a float, retaining missing values as NaN."""
        if value is None or (isinstance(value, str) and not value.strip()):
            return float('nan')
        try:
            return float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f'Capture metadata value must be numeric; received {value!r}.'
            ) from error

    @staticmethod
    def __validate_range(
        range_value: tuple[float, float] | None,
        argument_name: str,
        allow_none: bool = False
    ) -> None:
        """Validate a two-value increasing numeric range."""
        if range_value is None:
            if allow_none:
                return
            raise TypeError(f'{argument_name} must be a two-value tuple.')
        if not isinstance(range_value, tuple) or len(range_value) != 2:
            raise TypeError(f'{argument_name} must be a two-value tuple.')
        try:
            lower_value, upper_value = (float(value) for value in range_value)
        except (TypeError, ValueError) as error:
            raise TypeError(
                f'{argument_name} must contain numeric values.'
            ) from error
        if not np.isfinite([lower_value, upper_value]).all():
            raise ValueError(f'{argument_name} values must be finite.')
        if lower_value >= upper_value:
            raise ValueError(
                f'{argument_name} must increase from lower to upper value.'
            )

    def __validate_frame_rate(self) -> None:
        """Validate the frame rate before a per-second calculation."""
        if (
            not np.isfinite(self._capture_speed_in_fps) or
            self._capture_speed_in_fps <= 0
        ):
            raise ValueError(
                'A finite capture frame rate greater than 0 is required for '
                'per-second speeds.'
            )

    def __validate_pixel_scale_factor(self) -> None:
        """Validate the pixel scale before a scaled-distance calculation."""
        if (
            not np.isfinite(self.pixel_scale_factor) or
            self.pixel_scale_factor <= 0
        ):
            raise ValueError(
                'A finite pixel scale factor greater than 0 is required for '
                f'{self._scale_units} speeds.'
            )

    def __fit_and_plot_speed_distribution(
        self,
        speed: np.ndarray,
        particle: int,
        distribution_type: str = 'norm',
        fit_range: tuple[float, float] | None = None,
        ci_range: tuple[float, float] = (5, 95),
        bin_size: int = 30,
        speed_unit: str = DEFAULT_SPEED_UNIT,
        plot_results: bool = True
    ) -> float:
        """
        Fit speed within a selected range and plot its histogram and fit.

        Args:
            speed (np.ndarray): Array of speeds.
            particle (int): Particle ID.
            distribution_type (str): Distribution type for fitting, such as
                "norm", "expon", or "gamma". Defaults to "norm".
            fit_range (tuple): User-defined speed range for fitting. By default,
                the confidence interval determines the range.
            ci_range (tuple): Confidence interval used for the default speed
                limits. Defaults to (5, 95).
            bin_size (int): Number of bins for histogram (default: 30).
            speed_unit (str): Unit of speed displayed on the plots.
            plot_results (bool): Whether to create the per-particle plots.

        Returns:
            float: Mean speed of the particle.
        """
        if len(speed) == 0:
            print(f"Particle {particle}: No speed data available.")
            return float('nan')

        # Use the requested range or derive one from the confidence interval.
        if fit_range:
            lower_bound, upper_bound = fit_range
        else:
            lower_bound, upper_bound = np.percentile(speed, ci_range)

        # Filter speed values within the selected range
        speed_filtered = speed[(speed >= lower_bound) & (speed <= upper_bound)]

        if len(speed_filtered) < 2:
            print(
                f'Particle {particle}: Not enough data points within selected '
                f'range ({lower_bound}-{upper_bound} {speed_unit}).'
            )
            return float('nan')

        # Fit distribution only to the filtered speed values
        try:
            speed_distribution = distfit(distr=distribution_type, verbose=0)
            speed_distribution.fit_transform(speed_filtered, verbose=False)
            mean_speed = float(speed_distribution.model['model'].mean())
        except Exception as e:  # pylint: disable=broad-except
            print(
                f"Error fitting distribution '{distribution_type}' for particle {particle}: {e}")
            return float('nan')

        if not plot_results:
            return mean_speed

        # Create figure with two subplots: left for histogram, right for fitted distribution
        display_speed_unit = self.__plot_unit_label(speed_unit)
        _, axes_local = plt.subplots(1, 2, figsize=(12, 5))

        # Left Plot: Histogram with full data and fit range highlighted
        ax_hist = axes_local[0]
        ax_hist.hist(speed, bins=bin_size, alpha=0.7,
                     color='black', label='Speed (Full Data)')
        ax_hist.axvline(lower_bound, color='gray', linestyle='dashed',
                        label=(
                            f'Lower Bound ({lower_bound} '
                            f'{display_speed_unit})'
                        ))
        ax_hist.axvline(upper_bound, color='gray', linestyle='dashed',
                        label=(
                            f'Upper Bound ({upper_bound} '
                            f'{display_speed_unit})'
                        ))
        ax_hist.set_title(
            f'Particle: {particle} - Speed Histogram (Full Data)')
        ax_hist.set_xlabel(f'Speed ({display_speed_unit})')
        ax_hist.set_ylabel('Frequency')
        ax_hist.legend()

        # Right Plot: Distribution fit for selected range and its fitted mean.
        ax_dist = axes_local[1]
        speed_distribution.plot(ax=ax_dist)
        ax_dist.axvline(mean_speed, color='blue', linestyle='dashed', linewidth=2,
                        label=(
                            f'Mean Speed: {mean_speed:.2f} '
                            f'{display_speed_unit}'
                        ))
        ax_dist.set_xlim(lower_bound, upper_bound)
        ax_dist.set_title(
            f'Particle: {particle} - {distribution_type.capitalize()} Fit')

        # Remove unwanted markers (e.g., "CII low/high") from the legend
        handles, labels = ax_dist.get_legend_handles_labels()
        new_handles_labels = [(h, l) for h, l in zip(
            handles, labels) if 'CII' not in l]
        if new_handles_labels:
            new_handles, new_labels = zip(*new_handles_labels)
            ax_dist.legend(new_handles, new_labels)

        plt.tight_layout()
        plt.show()

        return mean_speed

    @staticmethod
    # type: ignore
    def __hide_unused_subplots(fig: plt.Figure, axes: np.ndarray, start_idx: int) -> None:
        """
        Hide unused subplots.

        Args:
            fig (plt.Figure): Matplotlib figure object.
            axes (np.ndarray): Array of Matplotlib axes objects.
            start_idx (int): Starting index to hide subplots.

        Returns:
            None
        """
        for j in range(start_idx, len(axes)):
            fig.delaxes(axes[j])

    def plot_overall_mean_speed_distribution(
        self,
        bins: int = 10,
        speed_unit: str | None = None,
        title: str | None = None,
        title_fontsize: float | None = None,
        x_axis_fontsize: float | None = None,
        y_axis_fontsize: float | None = None,
        x_tick_fontsize: float | None = None,
        y_tick_fontsize: float | None = None,
        font_family: str | None = 'Aptos',
        dpi: int = 300,
        save_plots: bool = False,
        save_plot_path: str | os.PathLike | None = (
            './02_Outputs/05_Basic_Stats/'
        ),
        file_extension: str = 'png',
        show: bool = True,
        show_n: bool = False
    ) -> tuple:
        """
        Plot the overall mean speed distribution.

        Args:
            bins (int): Number of bins for the histogram.
            speed_unit (str | None): Optional unit label to verify against the
                preceding speed calculation. Use scale_units/frame or
                scale_units/s to resolve the configured Capture scale unit.
                None uses the calculated unit.
            title: Custom histogram title. None uses the established metric
                title; an empty string removes the displayed title.
            title_fontsize: Title font size in points.
            x_axis_fontsize: X-axis label font size in points.
            y_axis_fontsize: Y-axis label font size in points.
            x_tick_fontsize: X-axis tick-label font size in points.
            y_tick_fontsize: Y-axis tick-label font size in points.
            font_family: Typeface for all histogram text. Portable families
                are 'sans-serif', 'serif', 'monospace', 'cursive', and
                'fantasy'. Bundled faces include 'DejaVu Sans',
                'DejaVu Serif', and 'DejaVu Sans Mono'. Installed system-font
                names, including 'Aptos' when installed, are accepted.
            dpi: Resolution used when saving the histogram.
            save_plots: Save the histogram using an automatically numbered
                filename.
            save_plot_path: Automatic-save directory. Relative paths resolve
                from the Stats output directory.
            file_extension: Automatic-save format: 'png' or 'tif'.
            show: Whether to display the histogram interactively.
            show_n: Whether to show the number of cells in a legend within
                the plot area.
        Returns:
            tuple: ``(figure, axis)`` for the histogram.
        """
        if not self._mean_array or self._calculated_speed_unit is None:
            raise ValueError(
                'Calculate mean speeds before plotting their overall distribution.'
            )
        if isinstance(bins, bool) or not isinstance(bins, (int, np.integer)):
            raise TypeError('bins must be an integer.')
        if bins < 1:
            raise ValueError('bins must be at least 1.')
        self.__validate_boolean_argument(show_n, 'show_n')
        plotting_parameters = self.__validate_basic_plotting_parameters(
            title=title,
            title_fontsize=title_fontsize,
            x_axis_fontsize=x_axis_fontsize,
            y_axis_fontsize=y_axis_fontsize,
            x_tick_fontsize=x_tick_fontsize,
            y_tick_fontsize=y_tick_fontsize,
            font_family=font_family,
            dpi=dpi,
            save_plots=save_plots,
            save_plot_path=save_plot_path,
            file_extension=file_extension,
            show=show
        )
        if speed_unit is None:
            speed_unit_label = self._calculated_speed_unit
        else:
            _, speed_unit_label = self.__normalize_speed_unit(speed_unit)
            if speed_unit_label != self._calculated_speed_unit:
                raise ValueError(
                    f'The calculated mean speeds use '
                    f'{self._calculated_speed_unit}, not {speed_unit_label}.'
                )

        fig, ax = plt.subplots(figsize=(10, 6))
        mean_array = np.asarray(self._mean_array, dtype=float)
        mean_array = mean_array[np.isfinite(mean_array)]
        if mean_array.size == 0:
            raise ValueError('No finite mean speeds are available to plot.')
        _, _, histogram_patches = ax.hist(
            mean_array, bins=bins, density=False,
            alpha=0.7, label='Mean Speeds'
        )
        if show_n:
            legend_font = plotting_parameters['font_family']
            ax.legend(
                handles=[histogram_patches[0]],
                labels=[f'n = {mean_array.size}'],
                prop=(
                    {'family': legend_font.strip()}
                    if legend_font is not None else None
                )
            )
        default_title = 'Overall Fitted Mean Speed Distribution'
        display_speed_unit = self.__plot_unit_label(speed_unit_label)
        ax.set_xlabel(f'Fitted Mean Speed ({display_speed_unit})')
        ax.set_ylabel('Frequency')
        self.__format_particle_metric_axis(
            axis=ax,
            default_title=default_title,
            title=plotting_parameters['title'],
            title_fontsize=plotting_parameters['title_fontsize'],
            x_axis_fontsize=plotting_parameters['x_axis_fontsize'],
            y_axis_fontsize=plotting_parameters['y_axis_fontsize'],
            x_tick_fontsize=plotting_parameters['x_tick_fontsize'],
            y_tick_fontsize=plotting_parameters['y_tick_fontsize'],
            y_tick_spacing=None,
            font_family=plotting_parameters['font_family']
        )
        save_path = self.__resolve_numbered_plot_save_path(
            save_path=None,
            save_plots=plotting_parameters['save_plots'],
            save_plot_path=plotting_parameters['save_plot_path'],
            file_extension=plotting_parameters['file_extension'],
            default_title=default_title,
            title=plotting_parameters['title']
        )
        fig.tight_layout()
        self.__finalize_figure(
            fig,
            save_path,
            plotting_parameters['dpi'],
            plotting_parameters['show']
        )
        return fig, ax

    def save_mean_speeds(
        self,
        filename: str,
        include_details: bool = False
    ) -> None:
        """
        Save the mean speeds to a CSV file.

        Args:
            filename (str): The filename to save the CSV file.
            include_details (bool): Include particle/source provenance,
                requested fit settings, Capture calibration, units, and frame
                discard settings. When False, the export contains strain and
                mean_speed only.

        Returns:
            None
        """
        if self._mean_speeds_dataframe.empty:
            raise ValueError('Calculate mean speeds before saving them.')
        if not isinstance(include_details, bool):
            raise TypeError('include_details must be a boolean.')
        root_name, extension = os.path.splitext(filename)
        if extension and extension.lower() != '.csv':
            raise ValueError('Fitted mean speeds can only be saved as a CSV file.')
        output_filename = filename if extension else f'{root_name}.csv'
        save_file_path = os.path.join(self._directory, output_filename)
        output_directory = os.path.dirname(save_file_path)
        if output_directory:
            os.makedirs(output_directory, exist_ok=True)
        output_dataframe = (
            self._mean_speeds_dataframe
            if include_details
            else self._mean_speeds_dataframe[['strain', 'mean_speed']]
        )
        output_dataframe.to_csv(save_file_path, index=False)
        print(f'Fitted mean speeds saved to {save_file_path}')

    def get_mean_speeds_dataframe(self) -> pd.DataFrame:
        """Return fitted means with settings, calibration, and provenance."""
        return self._mean_speeds_dataframe.copy()
