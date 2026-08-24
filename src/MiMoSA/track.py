"""
Module for tracking particles in 2D using trackpy.
"""

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
import trackpy as tp
from matplotlib import cm
from tqdm import tqdm

from .identify import Identify

FrameRange = tuple[int, int]
DeleteFrameRange = tuple[int | None] | tuple[int | None, int | None]
DeleteFrameRangeSpec = list[DeleteFrameRange] | DeleteFrameRange | None
ParticleDeleteSpec = int | tuple[int | DeleteFrameRangeSpec, ...]
ParticleSplitSpec = int | tuple[int, int]


class Tracker:
    """
    Class for tracking particles in 2D using trackpy.
    """
    DEFAULT_POSITION_COLUMNS: list[str] = [
        'centroid_x', 'centroid_y']  # Default position columns
    VIDEO_QUALITY_SCALES: dict[str, float] = {
        'original': 1.0,
        'high': 1.0,
        'medium': 0.75,
        'low': 0.5,
        'preview': 0.25
    }
    TRACK_DETAILS_COLUMNS: list[str] = [
        'particle',
        'first_frame',
        'last_frame',
        'first_centroid_x',
        'first_centroid_y',
        'last_centroid_x',
        'last_centroid_y',
        'endpoint_net_displacement_pixels',
        'cumulative_path_length_pixels'
    ]
    POSSIBLE_COLLISION_COLUMNS: list[str] = [
        'source_dataframe',
        'collision_event_id',
        'particle_1',
        'particle_2',
        'first_frame',
        'last_frame',
        'collision_frame_count',
        'closest_frame',
        'minimum_centroid_distance_pixels',
        'mean_centroid_distance_pixels',
        'collision_threshold_pixels_at_closest_frame',
        'particle_1_centroid_x_at_closest_frame',
        'particle_1_centroid_y_at_closest_frame',
        'particle_2_centroid_x_at_closest_frame',
        'particle_2_centroid_y_at_closest_frame',
        'particle_1_major_axis_length_at_closest_frame',
        'particle_2_major_axis_length_at_closest_frame',
        'threshold_basis',
        'review_status',
        'review_notes'
    ]

    def __init__(self, identify_object: Identify) -> None:
        """
        Initialize the Tracker object.
        """
        if not isinstance(identify_object, Identify):
            raise TypeError(
                "Identify_object must be provided and should be an instance of Identify.")

        self._parent = identify_object
        self._region_props_dataframe: pd.DataFrame = identify_object.get_region_props_dataframe(prefer_filtered=True)
        self._directory: str = identify_object.get_directory()
        self._capture_speed_in_fps = identify_object._parent.get_frame_rate()
        self._pixel_scale_factor: float = identify_object._parent.get_pixel_scale_factor()
        self._linked_particles_dataframes: pd.DataFrame = pd.DataFrame()
        self._filtered_particles_dataframes: pd.DataFrame = pd.DataFrame()
        self._updated_particles_dataframes: pd.DataFrame = pd.DataFrame()
        self._eliminated_particles_dataframe: pd.DataFrame = pd.DataFrame()
        self._linked_particles_metadata: dict = {}
        self._position_columns: list[str] = self.DEFAULT_POSITION_COLUMNS

    def link_particles(self, max_distance: float, max_memory: int, position_columns: list[str]) -> pd.DataFrame:
        """
        Link particles in a DataFrame.

        Args:
            max_distance (float): Maximum distance features can move between frames.
            max_memory (int): Maximum number of frames during which a feature can vanish.
            position_columns (List[str]): List containing the column names for the x and y positions. Default is ['centroid_x', 'centroid_y'].

        Returns:
            pd.DataFrame: DataFrame containing the linked particles.
        """
        if position_columns:
            self._position_columns = position_columns

        linked_dataframe = tp.link_df(self._region_props_dataframe, search_range=max_distance,
                                      memory=max_memory, pos_columns=self._position_columns)
        self._linked_particles_dataframes = linked_dataframe
        self._filtered_particles_dataframes = pd.DataFrame()
        self._updated_particles_dataframes = pd.DataFrame()
        self._eliminated_particles_dataframe = pd.DataFrame()
        self._linked_particles_metadata = {
            'tracker_type': 'basic',
            'max_distance': max_distance,
            'max_memory': max_memory,
            'position_columns': self._position_columns
        }
        particle_count = linked_dataframe['particle'].nunique()
        print(f'Successfully linked {particle_count} particles.')
        return linked_dataframe

    def link_particles_with_features(
        self,
        max_distance: float,
        max_memory: int = 0,
        max_step_distance: float | None = None,
        position_columns: list[str] | None = None,
        feature_columns: list[str] | None = None,
        feature_weights: dict[str, float] | None = None,
        momentum_weight: float = 0.0,
        direction_weight: float = 0.0,
        momentum_window: int = 0,
        is_update_particles: bool = True,
        feature_averaging_window: int = 1,
        future_feature_averaging_window: int = 1,
        edge_max_memory: int | None = None,
        edge_margin: float = 10.0
    ) -> pd.DataFrame:
        """
        Link particles using centroid position, with optional morphology and recent direction of travel.

        By default, this method delegates to trackpy.link_df and should match link_particles().
        Advanced feature/momentum/direction costs are only enabled when one or more advanced
        options are explicitly provided.

        Future feature averaging changes the relative morphology cost between candidates; it is
        not a hard morphology-rejection threshold. Future candidate chains are matched by position
        across consecutive frames only, so missing frames are never bridged during lookahead.

        Args:
            max_distance (float): Maximum allowed centroid distance between a predicted particle
                position and a candidate detection in the next frame.
            max_memory (int): Maximum number of frames during which a particle can be missing.
            max_step_distance (float | None): Optional hard maximum allowed distance between the last
                observed particle centroid and the candidate detection centroid. This prevents
                unrealistic frame-to-frame jumps even when the predicted position is nearby.
                If None, no extra hard step-distance gate is added in default trackpy-compatible mode.
                In advanced mode, None defaults to max_distance.
            position_columns (list[str] | None): Exactly two centroid columns. Defaults to
                self._position_columns. Edge detection interprets them in image-axis order:
                row/height first and column/width second. The default MiMoSA centroid columns,
                retained from RABiTPy, already use this order.
            feature_columns (list[str] | None): Optional morphological columns to include in the cost.
                Common examples are ['area', 'major_axis_length', 'minor_axis_length'].
                Defaults to None, which disables morphology costs and preserves link_particles() behavior.
            feature_weights (dict[str, float] | None): Optional per-feature weights.
                Features not provided in this dictionary use weight 1.0.
            momentum_weight (float): Additive weight applied to the predicted-position distance cost.
                Default 0.0 disables this advanced cost and preserves link_particles() behavior.
            direction_weight (float): Additive weight applied to the direction-change cost.
                Default 0.0 disables this advanced cost and preserves link_particles() behavior.
            momentum_window (int): Number of recent steps used to estimate particle velocity.
                Default 0 disables momentum prediction and preserves link_particles() behavior.
            is_update_particles (bool): Whether to update self._linked_particles_dataframes.
            feature_averaging_window (int): Number of recent detections used to calculate
                each track's mean feature values. Candidate features are compared against
                these means. Defaults to 1, which preserves the previous behavior of
                comparing against only the most recent detection.
            future_feature_averaging_window (int): Number of consecutive candidate detections
                used to calculate future mean feature values. The current candidate is included,
                followed by up to window - 1 position-matched detections from consecutive frames.
                Defaults to 1, which compares against only the current candidate.
            edge_max_memory (int | None): Maximum number of missing frames retained for a track
                whose last centroid is within edge_margin pixels of an image boundary. Must be
                less than or equal to max_memory. None disables edge-specific expiration.
            edge_margin (float): Distance from an image boundary, in pixels, used to identify
                an edge track. Defaults to 10.0 pixels.

        Returns:
            pd.DataFrame: DataFrame containing linked particles.
        """
        if self._region_props_dataframe.empty:
            raise ValueError(
                "No region properties dataframe available. Please generate region properties first."
            )

        if position_columns:
            self._position_columns = position_columns

        feature_columns = [] if feature_columns is None else feature_columns
        feature_weights = feature_weights or {}

        if len(self._position_columns) != 2:
            raise ValueError("position_columns must contain exactly two column names.")

        if (
            isinstance(max_distance, bool) or
            not isinstance(max_distance, (int, float, np.integer, np.floating))
        ):
            raise TypeError("max_distance must be a number.")
        if not np.isfinite(max_distance) or max_distance <= 0:
            raise ValueError("max_distance must be a finite number greater than zero.")

        if max_step_distance is not None:
            if (
                isinstance(max_step_distance, bool) or
                not isinstance(max_step_distance, (int, float, np.integer, np.floating))
            ):
                raise TypeError("max_step_distance must be a number or None.")
            if not np.isfinite(max_step_distance) or max_step_distance <= 0:
                raise ValueError(
                    "max_step_distance must be a finite number greater than zero."
                )

        if isinstance(max_memory, bool) or not isinstance(max_memory, (int, np.integer)):
            raise TypeError("max_memory must be an integer.")
        if max_memory < 0:
            raise ValueError("max_memory cannot be negative.")
        max_memory = int(max_memory)

        if (
            isinstance(feature_averaging_window, bool) or
            not isinstance(feature_averaging_window, (int, np.integer))
        ):
            raise TypeError("feature_averaging_window must be an integer.")
        if feature_averaging_window < 1:
            raise ValueError("feature_averaging_window must be at least 1.")
        feature_averaging_window = int(feature_averaging_window)

        if (
            isinstance(future_feature_averaging_window, bool) or
            not isinstance(future_feature_averaging_window, (int, np.integer))
        ):
            raise TypeError("future_feature_averaging_window must be an integer.")
        if future_feature_averaging_window < 1:
            raise ValueError("future_feature_averaging_window must be at least 1.")
        future_feature_averaging_window = int(future_feature_averaging_window)

        tracking_frame_shape: tuple[int, int] | None = None
        if edge_max_memory is not None:
            if isinstance(edge_max_memory, bool) or not isinstance(edge_max_memory, (int, np.integer)):
                raise TypeError("edge_max_memory must be an integer or None.")
            if edge_max_memory < 0:
                raise ValueError("edge_max_memory cannot be negative.")
            if edge_max_memory > max_memory:
                raise ValueError("edge_max_memory cannot be greater than max_memory.")
            edge_max_memory = int(edge_max_memory)

            if (
                isinstance(edge_margin, bool) or
                not isinstance(edge_margin, (int, float, np.integer, np.floating))
            ):
                raise TypeError("edge_margin must be a number.")
            edge_margin = float(edge_margin)
            if not np.isfinite(edge_margin) or edge_margin < 0:
                raise ValueError("edge_margin must be a finite, non-negative number.")

            tracking_frame_shape = self.__get_tracking_frame_shape()
            if edge_margin * 2 >= min(tracking_frame_shape):
                raise ValueError(
                    "edge_margin must be less than half of the smallest image dimension."
                )

        required_columns = ['frame'] + self._position_columns
        missing_required_columns = [
            column for column in required_columns
            if column not in self._region_props_dataframe.columns
        ]
        if missing_required_columns:
            raise ValueError(
                f"Missing required columns for linking: {missing_required_columns}"
            )

        advanced_tracking_enabled = (
            len(feature_columns) > 0 or
            len(feature_weights) > 0 or
            max_step_distance is not None or
            momentum_weight != 0.0 or
            direction_weight != 0.0 or
            momentum_window > 0 or
            edge_max_memory is not None
        )

        if not advanced_tracking_enabled:
            linked_dataframe = tp.link_df(
                self._region_props_dataframe,
                search_range=max_distance,
                memory=max_memory,
                pos_columns=self._position_columns
            )
            if is_update_particles:
                self._linked_particles_dataframes = linked_dataframe
                self._filtered_particles_dataframes = pd.DataFrame()
                self._updated_particles_dataframes = pd.DataFrame()
                self._eliminated_particles_dataframe = pd.DataFrame()
                self._linked_particles_metadata = {
                    'tracker_type': 'advanced_trackpy_compatible',
                    'max_distance': max_distance,
                    'max_memory': max_memory,
                    'max_step_distance': max_step_distance,
                    'position_columns': self._position_columns,
                    'feature_columns': feature_columns,
                    'feature_weights': feature_weights,
                    'feature_averaging_window': feature_averaging_window,
                    'future_feature_averaging_window': future_feature_averaging_window,
                    'momentum_weight': momentum_weight,
                    'direction_weight': direction_weight,
                    'momentum_window': momentum_window,
                    'edge_max_memory': edge_max_memory,
                    'edge_margin': edge_margin,
                    'advanced_tracking_enabled': False
                }
            particle_count = linked_dataframe['particle'].nunique()
            print(f'Successfully linked {particle_count} particles using trackpy-compatible default linking.')
            return linked_dataframe

        if max_step_distance is None:
            max_step_distance = max_distance

        missing_feature_columns = [
            column for column in feature_columns
            if column not in self._region_props_dataframe.columns
        ]
        if missing_feature_columns:
            raise ValueError(
                f"Requested feature columns are missing from dataframe: {missing_feature_columns}"
            )

        unknown_feature_weights = sorted(set(feature_weights) - set(feature_columns))
        if unknown_feature_weights:
            raise ValueError(
                "feature_weights contains columns not listed in feature_columns: "
                f"{unknown_feature_weights}"
            )

        invalid_feature_weights = []
        for column, weight in feature_weights.items():
            if (
                isinstance(weight, bool) or
                not isinstance(weight, (int, float, np.integer, np.floating)) or
                not np.isfinite(weight) or
                weight < 0
            ):
                invalid_feature_weights.append(column)
        if invalid_feature_weights:
            raise ValueError(
                "Feature weights must be finite, non-negative numbers. Invalid weights for: "
                f"{invalid_feature_weights}"
            )

        non_finite_feature_columns = []
        for column in feature_columns:
            numeric_values = pd.to_numeric(
                self._region_props_dataframe[column], errors='coerce'
            ).to_numpy(dtype=float)
            if not np.isfinite(numeric_values).all():
                non_finite_feature_columns.append(column)
        if non_finite_feature_columns:
            raise ValueError(
                "Feature columns must contain only finite numeric values. Invalid columns: "
                f"{non_finite_feature_columns}"
            )

        sorted_dataframe = self._region_props_dataframe.sort_values(
            by=['frame'] + self._position_columns
        ).reset_index(drop=True)

        future_feature_means = self.__calculate_future_feature_means(
            dataframe=sorted_dataframe,
            feature_columns=feature_columns,
            future_feature_averaging_window=future_feature_averaging_window,
            max_step_distance=max_step_distance
        )

        effective_momentum_weight = momentum_weight
        if not feature_columns and direction_weight == 0.0 and momentum_weight == 0.0:
            # Position distance remains a meaningful tie-breaker when edge memory or a
            # hard step-distance gate is the only enabled advanced option.
            effective_momentum_weight = 1.0

        active_tracks: dict[int, list[pd.Series]] = {}
        completed_tracks: dict[int, list[pd.Series]] = {}
        next_particle_id = 0

        frame_values = sorted(sorted_dataframe['frame'].unique())
        for frame_value in tqdm(frame_values, desc='Linking particles with features'):
            current_frame_dataframe = sorted_dataframe[
                sorted_dataframe['frame'] == frame_value
            ]

            # Edge tracks use a shorter memory so an entering object cannot inherit an
            # identifier from a different object that already left the field of view.
            tracks_to_complete = []
            for particle_id, track_rows in active_tracks.items():
                last_frame = track_rows[-1]['frame']
                effective_max_memory = max_memory
                if (
                    edge_max_memory is not None and
                    tracking_frame_shape is not None and
                    self.__is_detection_near_edge(
                        detection=track_rows[-1],
                        frame_shape=tracking_frame_shape,
                        edge_margin=edge_margin
                    )
                ):
                    effective_max_memory = edge_max_memory

                if frame_value - last_frame > effective_max_memory + 1:
                    tracks_to_complete.append(particle_id)

            for particle_id in tracks_to_complete:
                completed_tracks[particle_id] = active_tracks.pop(particle_id)

            if len(active_tracks) == 0:
                for _, detection in current_frame_dataframe.iterrows():
                    active_tracks[next_particle_id] = [detection.copy()]
                    next_particle_id += 1
                continue

            active_particle_ids = list(active_tracks.keys())
            cost_matrix = np.full(
                (len(active_particle_ids), len(current_frame_dataframe)),
                fill_value=np.inf,
                dtype=float
            )

            for track_index, particle_id in enumerate(active_particle_ids):
                track_rows = active_tracks[particle_id]
                for detection_position, (detection_index, detection) in enumerate(
                    current_frame_dataframe.iterrows()
                ):
                    cost_matrix[track_index, detection_position] = self.__calculate_linking_cost(
                        track_rows=track_rows,
                        detection=detection,
                        max_distance=max_distance,
                        max_step_distance=max_step_distance,
                        feature_columns=feature_columns,
                        feature_weights=feature_weights,
                        feature_averaging_window=feature_averaging_window,
                        future_feature_values=future_feature_means.get(int(detection_index)),
                        momentum_weight=effective_momentum_weight,
                        direction_weight=direction_weight,
                        momentum_window=momentum_window
                    )

            assigned_track_indices: set[int] = set()
            assigned_detection_indices: set[int] = set()

            finite_cost_matrix = np.where(np.isfinite(cost_matrix), cost_matrix, 1e12)
            row_indices, column_indices = linear_sum_assignment(finite_cost_matrix)

            for row_index, column_index in zip(row_indices, column_indices):
                if not np.isfinite(cost_matrix[row_index, column_index]):
                    continue

                particle_id = active_particle_ids[row_index]
                detection = current_frame_dataframe.iloc[column_index].copy()
                active_tracks[particle_id].append(detection)
                assigned_track_indices.add(row_index)
                assigned_detection_indices.add(column_index)

            # New detections not assigned to any existing track start new particles.
            for detection_position, (_, detection) in enumerate(current_frame_dataframe.iterrows()):
                if detection_position not in assigned_detection_indices:
                    active_tracks[next_particle_id] = [detection.copy()]
                    next_particle_id += 1

        completed_tracks.update(active_tracks)

        linked_rows = []
        for particle_id, track_rows in completed_tracks.items():
            for row in track_rows:
                row_copy = row.copy()
                row_copy['particle'] = particle_id
                linked_rows.append(row_copy)

        linked_dataframe = pd.DataFrame(linked_rows)
        linked_dataframe = linked_dataframe.sort_values(
            by=['particle', 'frame']
        ).reset_index(drop=True)

        if is_update_particles:
            self._linked_particles_dataframes = linked_dataframe
            self._filtered_particles_dataframes = pd.DataFrame()
            self._updated_particles_dataframes = pd.DataFrame()
            self._eliminated_particles_dataframe = pd.DataFrame()
            self._linked_particles_metadata = {
                'tracker_type': 'advanced',
                'max_distance': max_distance,
                'max_memory': max_memory,
                'max_step_distance': max_step_distance,
                'position_columns': self._position_columns,
                'feature_columns': feature_columns,
                'feature_weights': feature_weights,
                'feature_averaging_window': feature_averaging_window,
                'future_feature_averaging_window': future_feature_averaging_window,
                'momentum_weight': momentum_weight,
                'direction_weight': direction_weight,
                'momentum_window': momentum_window,
                'edge_max_memory': edge_max_memory,
                'edge_margin': edge_margin,
                'advanced_tracking_enabled': True
            }

        particle_count = linked_dataframe['particle'].nunique()
        print(f'Successfully linked {particle_count} particles using position, morphology, and momentum.')
        return linked_dataframe

    def filter_particles(
        self,
        min_frames: int,
        min_total_path: float,
        is_update_particles: bool = True,
        display_eliminated_particles: bool = True,
        *,
        min_displacement: float | None = None
    ) -> pd.DataFrame:
        """
        Filter particles by detection count, total path, and displacement.

        Filtering is always applied to the full linked tracks stored in
        self._linked_particles_dataframes. The full linked tracks are not overwritten.
        If is_update_particles is True, the filtered result is stored separately in
        self._filtered_particles_dataframes and an eliminated-particle audit is stored in
        self._eliminated_particles_dataframe.

        min_total_path is the sum of Euclidean distances between consecutive detected
        centroids after sorting by frame. Missed frames are not interpolated.
        min_displacement is the Euclidean distance between the initial and final
        detected centroids. A particle must be strictly above each enabled distance
        threshold to be retained.

        Args:
            min_frames (int): Minimum number of detections a particle must have to be kept.
            min_total_path (float): Cumulative observed path length that a particle
                must strictly exceed, in centroid-coordinate pixels.
            is_update_particles (bool): Whether to update self._filtered_particles_dataframes.
            display_eliminated_particles (bool): Whether to print the eliminated-particle
                audit dataframe. Defaults to True.
            min_displacement (float | None): Initial-to-final Euclidean displacement
                that a particle must strictly exceed, in centroid-coordinate pixels.
                This keyword-only filter is disabled when None. Defaults to None.
        Returns:
            pd.DataFrame: DataFrame containing the filtered particles.
        """
        if self._linked_particles_dataframes.empty:
            raise ValueError(
                "No linked dataframes available. Please link particles first.")

        if isinstance(min_frames, bool) or not isinstance(min_frames, (int, np.integer)):
            raise TypeError('min_frames must be an integer.')
        if min_frames < 1:
            raise ValueError('min_frames must be at least 1.')
        min_frames = int(min_frames)

        if (
            isinstance(min_total_path, bool) or
            not isinstance(min_total_path, (int, float, np.integer, np.floating))
        ):
            raise TypeError('min_total_path must be a number.')
        if not np.isfinite(min_total_path) or min_total_path < 0:
            raise ValueError('min_total_path must be a finite, non-negative number.')
        min_total_path = float(min_total_path)

        if min_displacement is not None:
            if (
                isinstance(min_displacement, bool) or
                not isinstance(
                    min_displacement,
                    (int, float, np.integer, np.floating)
                )
            ):
                raise TypeError('min_displacement must be a number or None.')
            if not np.isfinite(min_displacement) or min_displacement < 0:
                raise ValueError(
                    'min_displacement must be a finite, non-negative number or None.'
                )
            min_displacement = float(min_displacement)

        if not isinstance(is_update_particles, bool):
            raise TypeError('is_update_particles must be a boolean.')
        if not isinstance(display_eliminated_particles, bool):
            raise TypeError('display_eliminated_particles must be a boolean.')

        source_dataframe = self._linked_particles_dataframes
        if (
            not isinstance(self._position_columns, (list, tuple)) or
            len(self._position_columns) != 2
        ):
            raise ValueError(
                'Filtering requires exactly two position columns.'
            )
        position_columns = list(self._position_columns)

        required_columns = ['particle', 'frame'] + position_columns
        missing_columns = [
            column for column in required_columns
            if column not in source_dataframe.columns
        ]
        if missing_columns:
            raise ValueError(
                f'Linked dataframe is missing required filter columns: {missing_columns}'
            )

        first_position_column = position_columns[0]
        second_position_column = position_columns[1]
        audit_columns = [
            'particle',
            'elimination_reason',
            'detection_count',
            'frame_count',
            'duplicate_frame_count',
            'first_frame',
            'last_frame',
            f'first_{first_position_column}',
            f'first_{second_position_column}',
            f'last_{first_position_column}',
            f'last_{second_position_column}',
            'endpoint_net_displacement_pixels',
            'cumulative_path_length_pixels',
            'max_displacement_from_start_pixels',
            'minimum_detections_required',
            'minimum_cumulative_path_length_required_pixels',
            'minimum_endpoint_net_displacement_required_pixels',
            'frame_filter_passed',
            'cumulative_path_length_filter_passed',
            'endpoint_net_displacement_filter_passed'
        ]

        eliminated_rows = []
        retained_particle_ids = []
        particles_passing_frame_filter = 0
        particles_passing_total_path_filter = 0
        for particle_id, particle_rows in source_dataframe.groupby(
            'particle', sort=True
        ):
            sorted_particle_rows = particle_rows.sort_values(
                by='frame', kind='stable'
            )
            detection_count = len(sorted_particle_rows)
            frame_count = int(sorted_particle_rows['frame'].nunique())
            duplicate_frame_count = detection_count - frame_count
            first_row = sorted_particle_rows.iloc[0]
            last_row = sorted_particle_rows.iloc[-1]

            positions = sorted_particle_rows[
                position_columns
            ].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)
            (
                endpoint_net_displacement,
                cumulative_path_length,
                max_displacement_from_start
            ) = self.__calculate_track_motion_metrics(positions)

            frame_filter_passed = detection_count >= min_frames
            if frame_filter_passed:
                particles_passing_frame_filter += 1
            cumulative_path_length_filter_passed = (
                np.isfinite(cumulative_path_length) and
                cumulative_path_length > min_total_path
            )
            endpoint_net_displacement_filter_passed = (
                min_displacement is None or
                (
                    np.isfinite(endpoint_net_displacement) and
                    endpoint_net_displacement > min_displacement
                )
            )

            if frame_filter_passed and cumulative_path_length_filter_passed:
                particles_passing_total_path_filter += 1

            if (
                frame_filter_passed and
                cumulative_path_length_filter_passed and
                endpoint_net_displacement_filter_passed
            ):
                retained_particle_ids.append(particle_id)
                continue

            if not frame_filter_passed:
                elimination_reason = 'detection_count_below_minimum'
            elif not np.isfinite(cumulative_path_length):
                elimination_reason = 'cumulative_path_length_unavailable'
            else:
                elimination_reason = (
                    'cumulative_path_length_not_above_minimum'
                )
            if (
                frame_filter_passed and
                cumulative_path_length_filter_passed
            ):
                if not np.isfinite(endpoint_net_displacement):
                    elimination_reason = 'endpoint_net_displacement_unavailable'
                else:
                    elimination_reason = (
                        'endpoint_net_displacement_not_above_minimum'
                    )

            eliminated_rows.append({
                'particle': particle_id,
                'elimination_reason': elimination_reason,
                'detection_count': detection_count,
                'frame_count': frame_count,
                'duplicate_frame_count': duplicate_frame_count,
                'first_frame': first_row['frame'],
                'last_frame': last_row['frame'],
                f'first_{first_position_column}': first_row[first_position_column],
                f'first_{second_position_column}': first_row[second_position_column],
                f'last_{first_position_column}': last_row[first_position_column],
                f'last_{second_position_column}': last_row[second_position_column],
                'endpoint_net_displacement_pixels': endpoint_net_displacement,
                'cumulative_path_length_pixels': cumulative_path_length,
                'max_displacement_from_start_pixels': max_displacement_from_start,
                'minimum_detections_required': min_frames,
                'minimum_cumulative_path_length_required_pixels': min_total_path,
                'minimum_endpoint_net_displacement_required_pixels': (
                    min_displacement
                ),
                'frame_filter_passed': frame_filter_passed,
                'cumulative_path_length_filter_passed': (
                    cumulative_path_length_filter_passed
                ),
                'endpoint_net_displacement_filter_passed': (
                    endpoint_net_displacement_filter_passed
                )
            })

        eliminated_particles_dataframe = pd.DataFrame(
            eliminated_rows, columns=audit_columns
        )
        result_dataframe = source_dataframe[
            source_dataframe['particle'].isin(retained_particle_ids)
        ].copy()

        if is_update_particles:
            self._filtered_particles_dataframes = result_dataframe
            self._updated_particles_dataframes = pd.DataFrame()
            self._eliminated_particles_dataframe = eliminated_particles_dataframe
            self._linked_particles_metadata.update({
                'filter_min_detections': min_frames,
                'filter_min_cumulative_path_length_pixels': min_total_path,
                'filter_min_net_displacement_pixels': min_displacement,
                'filter_path_length_definition': (
                    'Sum of Euclidean distances between consecutive observed '
                    'centroids after sorting by frame; missed frames are not '
                    'interpolated and the particle must be strictly above the '
                    'threshold.'
                ),
                'filter_displacement_definition': (
                    'Euclidean distance between the initial and final observed '
                    'centroids after sorting by frame; when a threshold is '
                    'provided, the particle must be strictly above it.'
                )
            })

        print(
            f'After filtering for at least {min_frames} detections: '
            f'{particles_passing_frame_filter} unique particles'
        )
        print(
            f'After requiring cumulative path length > {min_total_path:g} pixels: '
            f'{particles_passing_total_path_filter} unique particles'
        )
        if min_displacement is not None:
            print(
                'After requiring initial-to-final displacement > '
                f'{min_displacement:g} pixels: '
                f'{len(retained_particle_ids)} unique particles'
            )

        if eliminated_particles_dataframe.empty:
            print('No particles were eliminated by the requested filters.')
        else:
            reason_counts = eliminated_particles_dataframe[
                'elimination_reason'
            ].value_counts()
            print('Eliminated particle counts by reason:')
            print(reason_counts.to_string())
            if display_eliminated_particles:
                display_columns = [
                    'particle',
                    'elimination_reason',
                    'detection_count',
                    'frame_count',
                    'endpoint_net_displacement_pixels',
                    'cumulative_path_length_pixels',
                    'max_displacement_from_start_pixels'
                ]
                print('Eliminated particle audit:')
                print(
                    eliminated_particles_dataframe[
                        display_columns
                    ].to_string(index=False)
                )

        return result_dataframe

    def compute_plot_save_MSD(self, max_lag_time: int = 100, is_save: bool = False, output_file_name: str = 'Mean_Squared_Difference') -> pd.DataFrame:
        """
        Calculate the Mean Squared Displacement (MSD) of the particles.
        Args:
            max_lag_time (int): Maximum lag time to calculate the MSD. Default is 100.
            is_save (bool): Whether to save the MSD values to a CSV file. Default is False.
            output_file_name (str): Name of the output file to save the MSD values. Default is 'Mean_Squared_Difference'.
        Returns:
            pd.DataFrame: DataFrame containing the MSD values.
        """
        if self._linked_particles_dataframes.empty:
            raise ValueError(
                "No linked dataframes available. Please link particles first.")

        # Calculate the MSD
        msd_dataframe = tp.imsd(
            self._linked_particles_dataframes, mpp=self._pixel_scale_factor, fps=self._capture_speed_in_fps, max_lagtime=max_lag_time, pos_columns=self._position_columns[::-1])

        # Calculate the EMSD
        emsd_dataframe = tp.emsd(
            self._linked_particles_dataframes, mpp=self._pixel_scale_factor, fps=self._capture_speed_in_fps, max_lagtime=max_lag_time, pos_columns=self._position_columns[::-1])

        # Create subplots
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=300)

        # Plot MSD
        ax1.plot(msd_dataframe.index, msd_dataframe, 'k-',
                 alpha=0.1)  # black lines, semitransparent
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set(
            ylabel=r'$\langle \Delta r^2 \rangle$ [$\mu$m$^2$]', xlabel='lag time $t$')
        ax1.set_title('Mean Squared Displacement (MSD)')

        # Plot EMSD
        ax2.plot(emsd_dataframe.index, emsd_dataframe, 'bo')
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.set(
            ylabel=r'$\langle \Delta r^2 \rangle$ [$\mu$m$^2$]', xlabel='lag time $t$')
        ax2.set_title('Ensemble Mean Squared Displacement (EMSD)')

        plt.show()

        # Save the MSD values to a CSV file
        if is_save:
            output_path = os.path.join(
                self._directory, f'{output_file_name}.csv')
            msd_dataframe.to_csv(output_path, index=False)
            print(f'MSD values saved to {output_path}')

        return msd_dataframe

    def save_linked_dataframes(
        self,
        output_file_name: str,
        save_filtered: bool = True,
        save_updated: bool = True,
        save_eliminated: bool = True,
        save_track_details: bool = True,
        save_possible_collisions: bool = True,
        collision_source: str = 'active',
        collision_distance_threshold: float | None = 20.0
    ) -> None:
        """
        Save linked tracks and optional audit dataframes.

        If output_file_name ends with '.xlsx', available dataframes are saved into
        separate sheets of one Excel workbook. Each exported linked-track sheet can
        include a per-particle details sheet. A possible-collision sheet can summarize
        consecutive close approaches for manual review. For other formats, filtered,
        updated, and eliminated-particle audit dataframes are saved to companion files
        when available. Applied particle-filter settings are exported to *_filter_settings
        companion files and a filter_settings workbook sheet. Tracking metadata is exported
        to *_metadata companion files, including NumPy.
        If no extension is provided, CSV, XLSX, pickle, and NumPy exports are created.

        Missing output folders are created automatically. Existing folders and
        unrelated files are preserved; same-named export files are overwritten.

        Args:
            output_file_name (str): Output filename or path. Extension can be '.csv', '.xlsx',
                '.pkl', '.pickle', '.npy', or omitted.
            save_filtered (bool): Whether to include the filtered tracks when available.
            save_updated (bool): Whether to include the updated tracks when available.
            save_eliminated (bool): Whether to include the eliminated-particle filter
                audit when filter_particles has been run with is_update_particles=True.
            save_track_details (bool): Whether XLSX exports include one per-particle
                details sheet for every exported linked-track dataframe.
            save_possible_collisions (bool): Whether XLSX exports include a
                possible_collisions review sheet.
            collision_source (str): Linked-track dataframe used for collision review:
                'active', 'full', 'filtered', 'updated', or 'all'. 'active' selects
                updated, then filtered, then full, based on the dataframes being exported.
            collision_distance_threshold (float | None): Fixed maximum centroid distance
                in pixels for a possible collision. Defaults to 20.0. None uses the
                size-aware threshold (major_axis_length_1 + major_axis_length_2) / 2.
        """
        if self._linked_particles_dataframes.empty:
            raise ValueError(
                "No linked dataframes available. Please link particles first.")

        for parameter_name, parameter_value in (
            ('save_filtered', save_filtered),
            ('save_updated', save_updated),
            ('save_eliminated', save_eliminated),
            ('save_track_details', save_track_details),
            ('save_possible_collisions', save_possible_collisions)
        ):
            if not isinstance(parameter_value, bool):
                raise TypeError(f'{parameter_name} must be a boolean.')

        if not isinstance(collision_source, str):
            raise TypeError('collision_source must be a string.')
        normalized_collision_source = collision_source.strip().lower()
        allowed_collision_sources = {
            'active', 'full', 'filtered', 'updated', 'all'
        }
        if normalized_collision_source not in allowed_collision_sources:
            raise ValueError(
                "collision_source must be one of: 'active', 'full', "
                "'filtered', 'updated', or 'all'."
            )

        if collision_distance_threshold is not None:
            if (
                isinstance(collision_distance_threshold, bool) or
                not isinstance(
                    collision_distance_threshold,
                    (int, float, np.integer, np.floating)
                )
            ):
                raise TypeError(
                    'collision_distance_threshold must be a number or None.'
                )
            if (
                not np.isfinite(collision_distance_threshold) or
                collision_distance_threshold <= 0
            ):
                raise ValueError(
                    'collision_distance_threshold must be finite and greater than 0.'
                )
            collision_distance_threshold = float(
                collision_distance_threshold
            )

        root_name, extension = os.path.splitext(output_file_name)
        output_base_path = os.path.join(self._directory, root_name)
        output_directory = os.path.dirname(output_base_path)
        if output_directory:
            os.makedirs(output_directory, exist_ok=True)

        linked_dataframe = self.__prepare_dataframe_for_export(
            self._linked_particles_dataframes)
        filtered_dataframe = self.__prepare_dataframe_for_export(
            self._filtered_particles_dataframes) if not self._filtered_particles_dataframes.empty else pd.DataFrame()
        updated_dataframe = self.__prepare_dataframe_for_export(
            self._updated_particles_dataframes) if not self._updated_particles_dataframes.empty else pd.DataFrame()
        eliminated_dataframe = self._eliminated_particles_dataframe.copy()
        has_elimination_report = len(eliminated_dataframe.columns) > 0
        exported_track_dataframes = {
            'full': self._linked_particles_dataframes
        }
        if save_filtered and not self._filtered_particles_dataframes.empty:
            exported_track_dataframes['filtered'] = (
                self._filtered_particles_dataframes
            )
        if save_updated and not self._updated_particles_dataframes.empty:
            exported_track_dataframes['updated'] = (
                self._updated_particles_dataframes
            )

        export_metadata = self._linked_particles_metadata.copy()
        if not export_metadata:
            export_metadata = {
                'tracker_type': 'unknown',
                'position_columns': self._position_columns
            }
        for collision_metadata_key in (
            'possible_collision_source',
            'possible_collision_dataframes_evaluated',
            'possible_collision_position_columns',
            'possible_collision_distance_threshold_pixels',
            'possible_collision_definition'
        ):
            export_metadata.pop(collision_metadata_key, None)

        should_save_csv = extension.lower() in ('', '.csv')
        should_save_xlsx = extension.lower() in ('', '.xlsx')
        should_save_pickle = extension.lower() in ('', '.pkl', '.pickle')
        should_save_npy = extension.lower() in ('', '.npy')

        if not should_save_csv and not should_save_xlsx and not should_save_pickle and not should_save_npy:
            raise ValueError(
                "Unsupported output extension. Use '.csv', '.xlsx', '.pkl', '.pickle', '.npy', or omit the extension to save all formats."
            )

        track_details_dataframes: dict[str, pd.DataFrame] = {}
        possible_collisions_dataframe = pd.DataFrame()
        if should_save_xlsx:
            if save_track_details:
                track_details_dataframes = {
                    dataframe_name: self.__build_track_details_dataframe(
                        dataframe
                    )
                    for dataframe_name, dataframe in (
                        exported_track_dataframes.items()
                    )
                }

            if save_possible_collisions:
                collision_dataframes = self.__select_collision_dataframes(
                    exported_track_dataframes=exported_track_dataframes,
                    collision_source=normalized_collision_source
                )
                possible_collisions_dataframe = (
                    self.__build_possible_collisions_dataframe(
                        collision_dataframes=collision_dataframes,
                        collision_distance_threshold=(
                            collision_distance_threshold
                        )
                    )
                )
                if collision_distance_threshold is None:
                    collision_definition = (
                        'Centroid distance is no greater than half the sum of '
                        'the two major-axis lengths. Consecutive qualifying '
                        'frames for each particle pair form one review event. '
                        'Encounters involving three or more tracks appear as '
                        'multiple pairwise events.'
                    )
                else:
                    collision_definition = (
                        'Centroid distance is no greater than the fixed '
                        f'{collision_distance_threshold:g}-pixel threshold. '
                        'Consecutive qualifying frames for each particle pair '
                        'form one review event. Encounters involving three or '
                        'more tracks appear as multiple pairwise events.'
                    )
                export_metadata.update({
                    'possible_collision_source': normalized_collision_source,
                    'possible_collision_dataframes_evaluated': list(
                        collision_dataframes.keys()
                    ),
                    'possible_collision_position_columns': {
                        dataframe_name: self.__get_export_position_columns(
                            dataframe
                        )
                        for dataframe_name, dataframe in (
                            collision_dataframes.items()
                        )
                    },
                    'possible_collision_distance_threshold_pixels': (
                        collision_distance_threshold
                    ),
                    'possible_collision_definition': collision_definition
                })

        metadata_dataframe = pd.DataFrame([
            {'parameter': key, 'value': repr(value)}
            for key, value in export_metadata.items()
        ])
        filter_settings_dataframe = pd.DataFrame(
            [
                {
                    'min_frames': export_metadata['filter_min_detections'],
                    'min_total_path': export_metadata[
                        'filter_min_cumulative_path_length_pixels'
                    ],
                    'min_displacement': export_metadata.get(
                        'filter_min_net_displacement_pixels'
                    )
                }
            ] if (
                'filter_min_detections' in export_metadata and
                'filter_min_cumulative_path_length_pixels' in export_metadata
            ) else [],
            columns=['min_frames', 'min_total_path', 'min_displacement']
        )

        if should_save_csv:
            linked_csv_path = f'{output_base_path}.csv'
            linked_dataframe.to_csv(linked_csv_path, index=False)
            print(f'Full linked dataframe saved to {linked_csv_path}')

            if save_filtered and not filtered_dataframe.empty:
                filtered_csv_path = f'{output_base_path}_filtered.csv'
                filtered_dataframe.to_csv(filtered_csv_path, index=False)
                print(f'Filtered linked dataframe saved to {filtered_csv_path}')

            if save_updated and not updated_dataframe.empty:
                updated_csv_path = f'{output_base_path}_updated.csv'
                updated_dataframe.to_csv(updated_csv_path, index=False)
                print(f'Updated linked dataframe saved to {updated_csv_path}')

            if save_eliminated and has_elimination_report:
                eliminated_csv_path = f'{output_base_path}_eliminated.csv'
                eliminated_dataframe.to_csv(eliminated_csv_path, index=False)
                print(f'Eliminated-particle audit saved to {eliminated_csv_path}')

            metadata_csv_path = f'{output_base_path}_metadata.csv'
            metadata_dataframe.to_csv(metadata_csv_path, index=False)
            print(f'Linked dataframe metadata csv saved to {metadata_csv_path}')

            filter_settings_csv_path = f'{output_base_path}_filter_settings.csv'
            filter_settings_dataframe.to_csv(
                filter_settings_csv_path, index=False)
            print(
                'Particle filter settings csv saved to '
                f'{filter_settings_csv_path}'
            )

        if should_save_xlsx:
            xlsx_path = f'{output_base_path}.xlsx'
            with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
                linked_dataframe.to_excel(
                    writer, sheet_name='linked_tracks_full', index=False)
                if save_track_details:
                    track_details_dataframes['full'].to_excel(
                        writer,
                        sheet_name='linked_tracks_full_details',
                        index=False
                    )
                if save_filtered and not filtered_dataframe.empty:
                    filtered_dataframe.to_excel(
                        writer, sheet_name='linked_tracks_filtered', index=False)
                    if save_track_details:
                        track_details_dataframes['filtered'].to_excel(
                            writer,
                            sheet_name='linked_tracks_filtered_details',
                            index=False
                        )
                if save_updated and not updated_dataframe.empty:
                    updated_dataframe.to_excel(
                        writer, sheet_name='linked_tracks_updated', index=False)
                    if save_track_details:
                        track_details_dataframes['updated'].to_excel(
                            writer,
                            sheet_name='linked_tracks_updated_details',
                            index=False
                        )
                if save_eliminated and has_elimination_report:
                    eliminated_dataframe.to_excel(
                        writer, sheet_name='particles_eliminated', index=False)
                if save_possible_collisions:
                    possible_collisions_dataframe.to_excel(
                        writer, sheet_name='possible_collisions', index=False)
                metadata_dataframe.to_excel(
                    writer, sheet_name='tracking_metadata', index=False)
                filter_settings_dataframe.to_excel(
                    writer, sheet_name='filter_settings', index=False)
            print(f'Linked dataframe workbook saved to {xlsx_path}')
            if save_possible_collisions:
                evaluated_dataframes = ', '.join(
                    collision_dataframes.keys()
                )
                print(
                    'Possible-collision review exported: '
                    f'{len(possible_collisions_dataframe)} events '
                    f'from {evaluated_dataframes} tracks.'
                )

        if should_save_pickle:
            pickle_path = f'{output_base_path}.pkl'
            linked_dataframe.to_pickle(pickle_path)
            print(f'Full linked dataframe pickle saved to {pickle_path}')

            if save_filtered and not filtered_dataframe.empty:
                filtered_pickle_path = f'{output_base_path}_filtered.pkl'
                filtered_dataframe.to_pickle(filtered_pickle_path)
                print(f'Filtered linked dataframe pickle saved to {filtered_pickle_path}')

            if save_updated and not updated_dataframe.empty:
                updated_pickle_path = f'{output_base_path}_updated.pkl'
                updated_dataframe.to_pickle(updated_pickle_path)
                print(f'Updated linked dataframe pickle saved to {updated_pickle_path}')

            if save_eliminated and has_elimination_report:
                eliminated_pickle_path = f'{output_base_path}_eliminated.pkl'
                eliminated_dataframe.to_pickle(eliminated_pickle_path)
                print(
                    f'Eliminated-particle audit pickle saved to '
                    f'{eliminated_pickle_path}'
                )

            metadata_pickle_path = f'{output_base_path}_metadata.pkl'
            pd.to_pickle(export_metadata, metadata_pickle_path)
            print(f'Linked dataframe metadata pickle saved to {metadata_pickle_path}')

            filter_settings_pickle_path = f'{output_base_path}_filter_settings.pkl'
            filter_settings_dataframe.to_pickle(filter_settings_pickle_path)
            print(
                'Particle filter settings pickle saved to '
                f'{filter_settings_pickle_path}'
            )

        if should_save_npy:
            npy_path = f'{output_base_path}.npy'
            np.save(
                npy_path,
                {
                    'columns': linked_dataframe.columns.to_list(),
                    'data': linked_dataframe.to_numpy(dtype=object),
                    'metadata': export_metadata
                },
                allow_pickle=True
            )
            print(f'Full linked dataframe NumPy file saved to {npy_path}')

            if save_filtered and not filtered_dataframe.empty:
                filtered_npy_path = f'{output_base_path}_filtered.npy'
                np.save(
                    filtered_npy_path,
                    {
                        'columns': filtered_dataframe.columns.to_list(),
                        'data': filtered_dataframe.to_numpy(dtype=object),
                        'metadata': export_metadata
                    },
                    allow_pickle=True
                )
                print(f'Filtered linked dataframe NumPy file saved to {filtered_npy_path}')

            if save_updated and not updated_dataframe.empty:
                updated_npy_path = f'{output_base_path}_updated.npy'
                np.save(
                    updated_npy_path,
                    {
                        'columns': updated_dataframe.columns.to_list(),
                        'data': updated_dataframe.to_numpy(dtype=object),
                        'metadata': export_metadata
                    },
                    allow_pickle=True
                )
                print(f'Updated linked dataframe NumPy file saved to {updated_npy_path}')

            if save_eliminated and has_elimination_report:
                eliminated_npy_path = f'{output_base_path}_eliminated.npy'
                np.save(
                    eliminated_npy_path,
                    {
                        'columns': eliminated_dataframe.columns.to_list(),
                        'data': eliminated_dataframe.to_numpy(dtype=object),
                        'metadata': export_metadata
                    },
                    allow_pickle=True
                )
                print(
                    f'Eliminated-particle audit NumPy file saved to '
                    f'{eliminated_npy_path}'
                )

            metadata_npy_path = f'{output_base_path}_metadata.npy'
            np.save(
                metadata_npy_path,
                {
                    'columns': metadata_dataframe.columns.to_list(),
                    'data': metadata_dataframe.to_numpy(dtype=object),
                    'metadata': export_metadata
                },
                allow_pickle=True
            )
            print(
                'Linked dataframe metadata NumPy file saved to '
                f'{metadata_npy_path}'
            )

            filter_settings_npy_path = f'{output_base_path}_filter_settings.npy'
            np.save(
                filter_settings_npy_path,
                {
                    'columns': filter_settings_dataframe.columns.to_list(),
                    'data': filter_settings_dataframe.to_numpy(dtype=object)
                },
                allow_pickle=True
            )
            print(
                'Particle filter settings NumPy file saved to '
                f'{filter_settings_npy_path}'
            )

    def load_linked_dataframes(
        self,
        input_file_name: str,
        filtered_input_file_name: str | None = None,
        updated_input_file_name: str | None = None,
        metadata_input_file_name: str | None = None,
        eliminated_input_file_name: str | None = None
    ) -> pd.DataFrame:
        """
        Load full and optionally filtered linked particle dataframes from disk.

        Supported formats are '.csv', '.xlsx', '.pkl', '.pickle', and '.npy'. For XLSX files,
        this method looks for sheets named 'linked_tracks_full', 'linked_tracks_filtered',
        'linked_tracks_updated', 'particles_eliminated', and 'tracking_metadata'. For
        non-XLSX files, companion files named *_filtered, *_updated, *_eliminated, and
        *_metadata are loaded automatically when present. Per-track details and possible
        collisions are derived export sheets; they are ignored during loading and
        recalculated from the linked tracks on the next XLSX export.
        """
        input_path = os.path.join(self._directory, input_file_name)
        if not os.path.exists(input_path):
            raise FileNotFoundError(f'Linked particles file does not exist: {input_path}')

        input_extension = os.path.splitext(input_path)[1].lower()
        self._filtered_particles_dataframes = pd.DataFrame()
        self._updated_particles_dataframes = pd.DataFrame()
        self._eliminated_particles_dataframe = pd.DataFrame()
        self._linked_particles_metadata = {}

        if input_extension == '.xlsx':
            workbook = pd.ExcelFile(input_path)
            if 'linked_tracks_full' in workbook.sheet_names:
                self._linked_particles_dataframes = pd.read_excel(input_path, sheet_name='linked_tracks_full')
            else:
                self._linked_particles_dataframes = pd.read_excel(input_path)

            if 'linked_tracks_filtered' in workbook.sheet_names:
                self._filtered_particles_dataframes = pd.read_excel(input_path, sheet_name='linked_tracks_filtered')

            if 'linked_tracks_updated' in workbook.sheet_names:
                self._updated_particles_dataframes = pd.read_excel(input_path, sheet_name='linked_tracks_updated')

            if 'particles_eliminated' in workbook.sheet_names:
                self._eliminated_particles_dataframe = pd.read_excel(
                    input_path, sheet_name='particles_eliminated'
                )

            if 'tracking_metadata' in workbook.sheet_names:
                metadata_dataframe = pd.read_excel(input_path, sheet_name='tracking_metadata')
                self._linked_particles_metadata = self.__metadata_dataframe_to_dict(metadata_dataframe)
        else:
            loaded_full_dataframe, loaded_metadata = self.__load_linked_dataframe_from_path(input_path)
            self._linked_particles_dataframes = loaded_full_dataframe
            if loaded_metadata:
                self._linked_particles_metadata = loaded_metadata

            if filtered_input_file_name is not None:
                filtered_path = os.path.join(self._directory, filtered_input_file_name)
                if not os.path.exists(filtered_path):
                    raise FileNotFoundError(f'Filtered linked particles file does not exist: {filtered_path}')
                self._filtered_particles_dataframes, filtered_metadata = self.__load_linked_dataframe_from_path(filtered_path)
                if filtered_metadata and not self._linked_particles_metadata:
                    self._linked_particles_metadata = filtered_metadata
            else:
                root_name, extension = os.path.splitext(input_path)
                companion_filtered_path = f'{root_name}_filtered{extension}'
                if os.path.exists(companion_filtered_path):
                    self._filtered_particles_dataframes, filtered_metadata = self.__load_linked_dataframe_from_path(companion_filtered_path)
                    if filtered_metadata and not self._linked_particles_metadata:
                        self._linked_particles_metadata = filtered_metadata

            if updated_input_file_name is not None:
                updated_path = os.path.join(self._directory, updated_input_file_name)
                if not os.path.exists(updated_path):
                    raise FileNotFoundError(f'Updated linked particles file does not exist: {updated_path}')
                self._updated_particles_dataframes, updated_metadata = self.__load_linked_dataframe_from_path(updated_path)
                if updated_metadata and not self._linked_particles_metadata:
                    self._linked_particles_metadata = updated_metadata
            else:
                root_name, extension = os.path.splitext(input_path)
                companion_updated_path = f'{root_name}_updated{extension}'
                if os.path.exists(companion_updated_path):
                    self._updated_particles_dataframes, updated_metadata = self.__load_linked_dataframe_from_path(companion_updated_path)
                    if updated_metadata and not self._linked_particles_metadata:
                        self._linked_particles_metadata = updated_metadata

            if eliminated_input_file_name is not None:
                eliminated_path = os.path.join(
                    self._directory, eliminated_input_file_name
                )
                if not os.path.exists(eliminated_path):
                    raise FileNotFoundError(
                        'Eliminated-particle audit file does not exist: '
                        f'{eliminated_path}'
                    )
                self._eliminated_particles_dataframe, eliminated_metadata = (
                    self.__load_linked_dataframe_from_path(eliminated_path)
                )
                if eliminated_metadata and not self._linked_particles_metadata:
                    self._linked_particles_metadata = eliminated_metadata
            else:
                root_name, extension = os.path.splitext(input_path)
                companion_eliminated_path = f'{root_name}_eliminated{extension}'
                if os.path.exists(companion_eliminated_path):
                    self._eliminated_particles_dataframe, eliminated_metadata = (
                        self.__load_linked_dataframe_from_path(
                            companion_eliminated_path
                        )
                    )
                    if eliminated_metadata and not self._linked_particles_metadata:
                        self._linked_particles_metadata = eliminated_metadata

            if metadata_input_file_name is not None:
                metadata_path = os.path.join(self._directory, metadata_input_file_name)
                if not os.path.exists(metadata_path):
                    raise FileNotFoundError(f'Linked particles metadata file does not exist: {metadata_path}')
                self._linked_particles_metadata = self.__load_linked_metadata_from_path(metadata_path)
            elif not self._linked_particles_metadata:
                root_name, _ = os.path.splitext(input_path)
                companion_metadata_path = f'{root_name}_metadata.pkl'
                if os.path.exists(companion_metadata_path):
                    self._linked_particles_metadata = self.__load_linked_metadata_from_path(companion_metadata_path)

        if 'particle' not in self._linked_particles_dataframes.columns:
            raise ValueError("Loaded linked particles dataframe is missing required 'particle' column.")
        if 'frame' not in self._linked_particles_dataframes.columns:
            raise ValueError("Loaded linked particles dataframe is missing required 'frame' column.")

        self._linked_particles_dataframes = (
            self.__remove_legacy_coordinate_columns(
                self._linked_particles_dataframes
            )
        )
        self._filtered_particles_dataframes = (
            self.__remove_legacy_coordinate_columns(
                self._filtered_particles_dataframes
            )
        )
        self._updated_particles_dataframes = (
            self.__remove_legacy_coordinate_columns(
                self._updated_particles_dataframes
            )
        )
        self.__restore_position_columns_from_loaded_dataframes()

        print(f'Full linked particles loaded: {len(self._linked_particles_dataframes)} rows')
        if not self._filtered_particles_dataframes.empty:
            print(f'Filtered linked particles loaded: {len(self._filtered_particles_dataframes)} rows')
        else:
            print('No filtered linked particles loaded.')
        if not self._updated_particles_dataframes.empty:
            print(f'Updated linked particles loaded: {len(self._updated_particles_dataframes)} rows')
        else:
            print('No updated linked particles loaded.')
        if len(self._eliminated_particles_dataframe.columns) > 0:
            print(
                'Eliminated-particle audit loaded: '
                f'{len(self._eliminated_particles_dataframe)} rows'
            )
        else:
            print('No eliminated-particle audit loaded.')
        if self._linked_particles_metadata:
            print('Loaded tracking metadata:')
            for key, value in self._linked_particles_metadata.items():
                print(f'  {key}: {value}')
        else:
            print('No tracking metadata loaded.')

        return self.get_linked_particles_dataframe(prefer_filtered=True)

    def get_linked_particles_dataframe(self, prefer_filtered: bool = True, prefer_updated: bool = True) -> pd.DataFrame:
        """
        Get linked particle dataframe.

        Args:
            prefer_filtered: If True and filtered tracks are available, prefer filtered over full.
            prefer_updated: If True and manually updated tracks are available, prefer updated over all others.
        """
        if prefer_updated and not self._updated_particles_dataframes.empty:
            return self._updated_particles_dataframes

        if prefer_filtered and not self._filtered_particles_dataframes.empty:
            return self._filtered_particles_dataframes

        return self._linked_particles_dataframes

    def get_full_linked_particles_dataframe(self) -> pd.DataFrame:
        """
        Get full, unfiltered linked particle dataframe.
        """
        return self._linked_particles_dataframes

    def get_filtered_linked_particles_dataframe(self) -> pd.DataFrame:
        """
        Get filtered linked particle dataframe.
        """
        return self._filtered_particles_dataframes

    def get_eliminated_particles_dataframe(self) -> pd.DataFrame:
        """
        Get the audit dataframe describing particles removed by filter_particles.
        """
        return self._eliminated_particles_dataframe

    def get_linked_particles_metadata(self) -> dict:
        """
        Get metadata describing the linked-particle generation parameters.
        """
        return self._linked_particles_metadata

    def get_updated_linked_particles_dataframe(self) -> pd.DataFrame:
        """
        Get manually updated linked particle dataframe.
        """
        return self._updated_particles_dataframes

    def lookup_particle(
        self,
        particle_ids: int | list[int] | tuple[int, ...],
        source_dataframe: str = "active"
    ) -> pd.DataFrame:
        """
        Display the frame range for one or more particle IDs.

        Args:
            particle_ids: One particle ID or a list/tuple of particle IDs.
            source_dataframe: Source tracks to inspect. Use "active", "full",
                "filtered", or "updated".

        Returns:
            pd.DataFrame: Frame-range summary for each particle found.
        """
        if isinstance(particle_ids, int):
            requested_particle_ids = [particle_ids]
        elif isinstance(particle_ids, (list, tuple)):
            requested_particle_ids = list(particle_ids)
        else:
            raise ValueError('particle_ids must be an integer, list, or tuple.')
        if not requested_particle_ids:
            raise ValueError('particle_ids cannot be empty.')
        if any(
            isinstance(particle_id, bool) or not isinstance(particle_id, int)
            for particle_id in requested_particle_ids
        ):
            raise ValueError('particle_ids must contain only integers.')
        if len(set(requested_particle_ids)) != len(requested_particle_ids):
            raise ValueError('particle_ids cannot contain duplicate values.')

        source = self.__get_tracking_dataframe_by_name(source_dataframe)
        if source.empty:
            raise ValueError(
                f"No linked dataframe available for source_dataframe='{source_dataframe}'."
            )

        required_columns = ('particle', 'frame')
        missing_columns = [
            column for column in required_columns if column not in source.columns
        ]
        if missing_columns:
            raise ValueError(
                f'Linked dataframe is missing required columns: {missing_columns}'
            )

        lookup_rows = []
        missing_particle_ids = []
        for particle_id in requested_particle_ids:
            particle_rows = source[source['particle'] == particle_id]
            if particle_rows.empty:
                missing_particle_ids.append(particle_id)
                continue

            unique_frame_count = particle_rows['frame'].nunique()
            row_count = len(particle_rows)
            lookup_rows.append({
                'particle': particle_id,
                'first_frame': particle_rows['frame'].min(),
                'last_frame': particle_rows['frame'].max(),
                'frame_count': unique_frame_count,
                'row_count': row_count,
                'duplicate_frame_count': row_count - unique_frame_count
            })

        lookup_dataframe = pd.DataFrame(
            lookup_rows,
            columns=[
                'particle',
                'first_frame',
                'last_frame',
                'frame_count',
                'row_count',
                'duplicate_frame_count'
            ]
        )

        if missing_particle_ids:
            print(
                f'Particle IDs not found in {source_dataframe} tracks: '
                f'{missing_particle_ids}'
            )

        if lookup_dataframe.empty:
            print(f'No requested particles found in {source_dataframe} tracks.')
        else:
            print(f'Particle frame ranges from {source_dataframe} tracks:')
            print(lookup_dataframe.to_string(index=False))

        return lookup_dataframe

    def manually_update_particle_labels(
        self,
        particles_to_link: list[tuple[int, ...]] | None = None,
        particles_to_delete: list[ParticleDeleteSpec] | tuple[ParticleDeleteSpec, ...] | None = None,
        link_frame_range: list[FrameRange | None] | tuple[FrameRange | None, ...] | None = None,
        delete_frame_range: list[DeleteFrameRangeSpec] | tuple[DeleteFrameRangeSpec, ...] | None = None,
        source_dataframe: str = "active",
        is_update_particles: bool = True,
        sort_result: bool = True,
        particles_to_split: ParticleSplitSpec | list[ParticleSplitSpec] | tuple[ParticleSplitSpec, ...] | None = None,
        split_from_frame: int | list[int] | tuple[int, ...] | None = None,
        particles_to_add: int | list[int] | tuple[int, ...] | None = None,
        df_to_add_from: str | list[str] | tuple[str, ...] | None = None
    ) -> pd.DataFrame:
        """
        Manually restore, merge, delete, and/or split particle labels.

        For each tuple in particles_to_link, the first value is kept and all following values
        are reassigned to that first value. Automatic links with up to three overlapping
        frames retain the rows from the lower-numbered particle. Links with four or more
        overlapping frames are skipped unless an inclusive link_frame_range is provided.
        Each particle requested in particles_to_add is restored exactly from its aligned
        full, filtered, or updated donor dataframe: any current rows for that particle are
        replaced by all donor rows. Restorations are applied before deletions, links, and
        splits, while full and filtered tracker dataframes remain unchanged.
        Each requested split retains rows before split_from_frame under the original
        particle label and assigns rows from that frame onward to a new particle label.
        New split labels are greater than every label in both the full linked dataframe
        and the dataframe being updated. Splits are applied after requested deletions
        and links.

        Args:
            particles_to_link: Groups of particle labels to merge.
            particles_to_delete: Particle labels to remove. When delete_frame_range
                is omitted, entries can instead contain a particle followed by one
                or more frame ranges. For example, (7, (143, 151), (262,)) deletes
                frames 143-151 and frame 262 through the end for particle 7. Plain
                particle labels and (particle, frame_range) pairs remain supported.
            particles_to_add: Particle labels to restore into the selected source.
            df_to_add_from: Donor dataframe names aligned with particles_to_add.
                Each entry must be "full", "filtered", or "updated". A single
                string applies the same donor dataframe to every requested particle.
                When particles_to_add is empty, the donor selection is ignored and
                no particles are restored.
            particles_to_split: One particle label or a list/tuple of labels to
                split. When split_from_frame is omitted, entries can instead be
                (particle, split_frame) pairs, for example [(38, 110), (45, 32)].
            link_frame_range: Inclusive frame ranges aligned with particles_to_link.
                Only rows from replacement particles inside a specified range are
                relabeled. Use None or omit trailing entries for automatic linking.
            delete_frame_range: Per-particle inclusive frame ranges aligned with
                particles_to_delete. Use None or omit a trailing entry to delete the whole
                particle. Each particle can receive one range or a list of ranges. Use
                (None, end_frame) for the beginning through end_frame, and either
                (start_frame,) or (start_frame, None) for start_frame through the last
                available frame.
            split_from_frame: Inclusive split frames aligned with particles_to_split.
                Rows at or after each frame receive a new particle label. A scalar is
                accepted when splitting one particle. Omit this argument when using
                (particle, split_frame) pairs in particles_to_split.
            source_dataframe: Source tracks to curate. Use "active", "full", "filtered", or "updated".
            is_update_particles: Whether to store the curated result in self._updated_particles_dataframes.
            sort_result: Whether to sort by particle and frame when possible.

        Returns:
            pd.DataFrame: Curated linked-particle dataframe.
        """
        particles_to_link = [] if particles_to_link is None else particles_to_link

        def is_integer_value(value: object) -> bool:
            return (
                not isinstance(value, (bool, np.bool_)) and
                isinstance(value, (int, np.integer))
            )

        def is_particle_delete_spec(value: object) -> bool:
            if (
                not isinstance(value, tuple) or
                len(value) < 2 or
                not is_integer_value(value[0])
            ):
                return False
            if len(value) == 2:
                return value[1] is None or isinstance(value[1], (list, tuple))
            return all(isinstance(frame_range, tuple) for frame_range in value[1:])

        def is_particle_split_pair(value: object) -> bool:
            return (
                isinstance(value, tuple) and
                len(value) == 2 and
                is_integer_value(value[0]) and
                is_integer_value(value[1])
            )

        def looks_like_delete_frame_range(value: object) -> bool:
            return (
                isinstance(value, tuple) and
                len(value) in (1, 2) and
                all(not isinstance(boundary, (list, tuple)) for boundary in value)
            )

        if particles_to_delete is None:
            particle_delete_entries = []
        elif is_particle_delete_spec(particles_to_delete):
            particle_delete_entries = [particles_to_delete]
        elif isinstance(particles_to_delete, (list, tuple)):
            particle_delete_entries = list(particles_to_delete)
        else:
            raise ValueError(
                'particles_to_delete must be a list, tuple, or None.'
            )

        normalized_particles_to_delete = []
        embedded_delete_range_specs: list[DeleteFrameRangeSpec] = []
        has_embedded_delete_ranges = False
        for delete_entry in particle_delete_entries:
            if is_integer_value(delete_entry):
                normalized_particles_to_delete.append(int(delete_entry))
                embedded_delete_range_specs.append(None)
                continue

            if is_particle_delete_spec(delete_entry):
                particle_label = delete_entry[0]
                range_spec = (
                    delete_entry[1]
                    if len(delete_entry) == 2
                    else list(delete_entry[1:])
                )
                normalized_particles_to_delete.append(int(particle_label))
                embedded_delete_range_specs.append(range_spec)
                has_embedded_delete_ranges = True
                continue

            raise ValueError(
                'particles_to_delete must contain integer particle labels or '
                '(particle, frame_range, ...) entries such as '
                '(7, (143, 151), (262,)).'
            )

        if has_embedded_delete_ranges and delete_frame_range is not None:
            raise ValueError(
                'delete_frame_range must be omitted when particles_to_delete '
                'contains embedded particle frame ranges.'
            )

        particles_to_delete = normalized_particles_to_delete
        has_frame_limited_deletions = (
            delete_frame_range is not None or
            any(bool(range_spec) for range_spec in embedded_delete_range_specs)
        )
        if particles_to_add is None:
            particles_to_add = []
        elif (
            isinstance(particles_to_add, (int, np.integer)) and
            not isinstance(particles_to_add, (bool, np.bool_))
        ):
            particles_to_add = [int(particles_to_add)]
        elif isinstance(particles_to_add, (list, tuple)):
            particles_to_add = list(particles_to_add)
        else:
            raise ValueError(
                'particles_to_add must be an integer, list, tuple, or None.'
            )
        if any(
            isinstance(particle_label, (bool, np.bool_)) or
            not isinstance(particle_label, (int, np.integer))
            for particle_label in particles_to_add
        ):
            raise ValueError('particles_to_add must contain only integers.')
        particles_to_add = [
            int(particle_label) for particle_label in particles_to_add
        ]
        if len(set(particles_to_add)) != len(particles_to_add):
            raise ValueError('particles_to_add cannot contain duplicate values.')

        if (
            df_to_add_from is not None and
            not isinstance(df_to_add_from, (str, list, tuple))
        ):
            raise ValueError(
                'df_to_add_from must be a string, list, tuple, or None.'
            )

        if not particles_to_add:
            add_source_specs = []
        elif df_to_add_from is None:
            add_source_specs = []
        elif isinstance(df_to_add_from, str):
            add_source_specs = [df_to_add_from] * len(particles_to_add)
        else:
            add_source_specs = list(df_to_add_from)
        if particles_to_add and not add_source_specs:
            raise ValueError(
                'df_to_add_from is required when particles_to_add is provided.'
            )
        if len(particles_to_add) != len(add_source_specs):
            raise ValueError(
                'particles_to_add and df_to_add_from must contain the same '
                'number of entries.'
            )
        if any(not isinstance(source_name, str) for source_name in add_source_specs):
            raise ValueError(
                'df_to_add_from must contain only dataframe-name strings.'
            )
        add_source_specs = [
            source_name.strip().lower() for source_name in add_source_specs
        ]
        valid_add_sources = ('full', 'filtered', 'updated')
        invalid_add_sources = sorted({
            source_name
            for source_name in add_source_specs
            if source_name not in valid_add_sources
        })
        if invalid_add_sources:
            raise ValueError(
                f'df_to_add_from must contain only {valid_add_sources}. '
                f'Received invalid values: {invalid_add_sources}'
            )

        add_delete_conflicts = sorted(
            set(particles_to_add).intersection(particles_to_delete)
        )
        if add_delete_conflicts:
            raise ValueError(
                'Particle labels cannot be both added and deleted: '
                f'{add_delete_conflicts}'
            )

        if particles_to_split is None:
            particle_split_entries = []
        elif is_integer_value(particles_to_split):
            particle_split_entries = [particles_to_split]
        elif (
            split_from_frame is None and
            is_particle_split_pair(particles_to_split)
        ):
            particle_split_entries = [particles_to_split]
        elif isinstance(particles_to_split, (list, tuple)):
            particle_split_entries = list(particles_to_split)
        else:
            raise ValueError(
                'particles_to_split must be an integer, list, tuple, or None.'
            )

        normalized_particles_to_split = []
        embedded_split_frame_specs = []
        has_embedded_split_frames = False
        has_split_labels_without_frames = False
        for split_entry in particle_split_entries:
            if is_integer_value(split_entry):
                normalized_particles_to_split.append(int(split_entry))
                has_split_labels_without_frames = True
                continue

            if is_particle_split_pair(split_entry):
                particle_label, split_frame = split_entry
                normalized_particles_to_split.append(int(particle_label))
                embedded_split_frame_specs.append(int(split_frame))
                has_embedded_split_frames = True
                continue

            raise ValueError(
                'particles_to_split must contain integer particle labels or '
                '(particle, split_frame) integer pairs such as (38, 110).'
            )

        if has_embedded_split_frames:
            if split_from_frame is not None:
                raise ValueError(
                    'split_from_frame must be omitted when particles_to_split '
                    'contains (particle, split_frame) pairs.'
                )
            if has_split_labels_without_frames:
                raise ValueError(
                    'When particles_to_split contains paired split frames, every '
                    'entry must be a (particle, split_frame) pair.'
                )
            split_frame_specs = embedded_split_frame_specs
        elif split_from_frame is None:
            split_frame_specs = []
        elif is_integer_value(split_from_frame):
            split_frame_specs = [int(split_from_frame)]
        elif isinstance(split_from_frame, (list, tuple)):
            split_frame_specs = list(split_from_frame)
        else:
            raise ValueError(
                'split_from_frame must be an integer, list, tuple, or None.'
            )

        particles_to_split = normalized_particles_to_split

        if len(particles_to_split) != len(split_frame_specs):
            raise ValueError(
                'particles_to_split and split_from_frame must contain the same '
                'number of entries.'
            )
        if any(
            not is_integer_value(particle_label)
            for particle_label in particles_to_split
        ):
            raise ValueError('particles_to_split must contain only integers.')
        if len(set(particles_to_split)) != len(particles_to_split):
            raise ValueError('particles_to_split cannot contain duplicate values.')
        if any(
            not is_integer_value(frame)
            for frame in split_frame_specs
        ):
            raise ValueError('split_from_frame must contain only integers.')
        split_frame_specs = [int(frame) for frame in split_frame_specs]

        source = self.__get_tracking_dataframe_by_name(source_dataframe).copy()

        if source.empty:
            raise ValueError(f"No linked dataframe available for source_dataframe='{source_dataframe}'.")

        if "particle" not in source.columns:
            raise ValueError("Linked dataframe must contain a 'particle' column.")

        if (
            particles_to_add or
            particles_to_link or
            has_frame_limited_deletions or
            particles_to_split
        ) and "frame" not in source.columns:
            raise ValueError("Linked dataframe must contain a 'frame' column for frame-aware updates.")

        validated_additions = []
        for particle_label, add_source_name in zip(
            particles_to_add, add_source_specs
        ):
            addition_dataframe = self.__get_tracking_dataframe_by_name(
                add_source_name
            )
            if addition_dataframe.empty:
                raise ValueError(
                    f"No linked dataframe available for "
                    f"df_to_add_from='{add_source_name}'."
                )
            missing_addition_columns = sorted(
                set(source.columns).difference(addition_dataframe.columns)
            )
            extra_addition_columns = sorted(
                set(addition_dataframe.columns).difference(source.columns)
            )
            if (
                len(source.columns) != len(addition_dataframe.columns) or
                missing_addition_columns or
                extra_addition_columns
            ):
                raise ValueError(
                    f"Cannot add particle {particle_label} from "
                    f"'{add_source_name}': donor and target dataframe columns "
                    f'must match. Missing donor columns: '
                    f'{missing_addition_columns}; extra donor columns: '
                    f'{extra_addition_columns}.'
                )
            donor_rows = addition_dataframe.loc[
                addition_dataframe['particle'] == particle_label
            ].copy()
            if donor_rows.empty:
                raise ValueError(
                    f"Particle {particle_label} was not found in the "
                    f"'{add_source_name}' donor dataframe."
                )
            duplicate_donor_frames = donor_rows.duplicated(
                subset=['particle', 'frame'], keep=False
            )
            if duplicate_donor_frames.any():
                duplicate_frames = sorted(
                    donor_rows.loc[duplicate_donor_frames, 'frame']
                    .drop_duplicates()
                    .tolist()
                )
                raise ValueError(
                    f"Cannot add particle {particle_label} from "
                    f"'{add_source_name}': duplicate particle/frame rows were "
                    f'found for frames {duplicate_frames}.'
                )
            validated_additions.append({
                'particle': particle_label,
                'source_dataframe': add_source_name,
                'rows': donor_rows.reindex(columns=source.columns)
            })

        addition_results = []
        for addition in validated_additions:
            particle_label = addition['particle']
            previous_row_count = int(
                (source['particle'] == particle_label).sum()
            )
            source = source.loc[
                source['particle'] != particle_label
            ].copy()
            donor_rows = addition['rows']
            source = pd.concat(
                [source, donor_rows], ignore_index=True
            )
            addition_result = {
                'particle': particle_label,
                'source_dataframe': addition['source_dataframe'],
                'replaced_target_row_count': previous_row_count,
                'restored_row_count': int(len(donor_rows))
            }
            addition_results.append(addition_result)
            print(
                f"Restored particle {particle_label} from "
                f"'{addition['source_dataframe']}' tracks: replaced "
                f'{previous_row_count} target row(s) with '
                f'{len(donor_rows)} donor row(s).'
            )

        next_particle_label = 0
        if particles_to_split:
            full_dataframe = self._linked_particles_dataframes
            if full_dataframe.empty or 'particle' not in full_dataframe.columns:
                raise ValueError(
                    'Full linked particle labels are required before tracks can '
                    'be split.'
                )

            numeric_particle_labels = pd.concat(
                [
                    pd.to_numeric(source['particle'], errors='coerce'),
                    pd.to_numeric(
                        full_dataframe['particle'], errors='coerce'
                    )
                ],
                ignore_index=True
            )
            if (
                numeric_particle_labels.isna().any() or
                not np.equal(
                    numeric_particle_labels,
                    np.floor(numeric_particle_labels)
                ).all()
            ):
                raise ValueError(
                    'Particle labels in the full and selected dataframes must be '
                    'integers before tracks can be split.'
                )
            next_particle_label = int(numeric_particle_labels.max()) + 1

        replacement_labels = []

        for label_group in particles_to_link:
            if len(label_group) < 2:
                continue

            for label_to_replace in label_group[1:]:
                if label_to_replace in replacement_labels:
                    raise ValueError(
                        f"Particle label {label_to_replace} appears in more than one merge group."
                    )
                replacement_labels.append(label_to_replace)

        particles_to_delete_set = set(particles_to_delete)
        labels_to_replace_set = set(replacement_labels)

        conflicting_labels = particles_to_delete_set.intersection(labels_to_replace_set)

        if conflicting_labels:
            raise ValueError(
                f"Particle labels cannot be both deleted and merged: {sorted(conflicting_labels)}"
            )

        existing_labels = set(source["particle"].unique())

        requested_labels = (
            set(replacement_labels)
            .union({label_group[0] for label_group in particles_to_link if label_group})
            .union(particles_to_delete_set)
            .union(particles_to_split)
        )

        missing_labels = sorted(requested_labels.difference(existing_labels))

        if missing_labels:
            print(f"Warning: requested particle labels not found in source dataframe: {missing_labels}")

        if has_embedded_delete_ranges:
            delete_range_specs = embedded_delete_range_specs
        elif delete_frame_range is None:
            delete_range_specs: list[DeleteFrameRangeSpec] = [None] * len(particles_to_delete)
        else:
            if (
                len(particles_to_delete) == 1 and
                looks_like_delete_frame_range(delete_frame_range)
            ):
                delete_range_specs = [delete_frame_range]
            else:
                delete_range_specs = list(delete_frame_range)
                if (
                    len(particles_to_delete) == 1 and
                    delete_range_specs and
                    all(
                        looks_like_delete_frame_range(frame_range)
                        for frame_range in delete_range_specs
                    )
                ):
                    delete_range_specs = [delete_range_specs]

            if len(delete_range_specs) > len(particles_to_delete):
                raise ValueError(
                    'delete_frame_range cannot contain more entries than particles_to_delete.'
                )
            delete_range_specs.extend(
                [None] * (len(particles_to_delete) - len(delete_range_specs))
            )

        fully_deleted_labels = []
        ranged_deletions: dict[
            int, list[tuple[int | None, int | None]]
        ] = {}
        for particle_label, range_spec in zip(particles_to_delete, delete_range_specs):
            if particle_label not in existing_labels:
                continue

            if not range_spec:
                source = source[source['particle'] != particle_label].copy()
                fully_deleted_labels.append(particle_label)
                continue

            if isinstance(range_spec, tuple):
                frame_ranges = [range_spec]
            elif isinstance(range_spec, list):
                frame_ranges = range_spec
            else:
                raise ValueError(
                    'Each delete frame range must be a tuple or list of tuples.'
                )

            validated_ranges = []
            rows_to_delete = pd.Series(False, index=source.index)
            for frame_range in frame_ranges:
                if (
                    not isinstance(frame_range, tuple) or
                    len(frame_range) not in (1, 2) or
                    any(
                        value is not None and not is_integer_value(value)
                        for value in frame_range
                    )
                ):
                    raise ValueError(
                        'Each delete frame range must be (start_frame, end_frame), '
                        '(None, end_frame), or (start_frame,), using integer frame '
                        'values.'
                    )

                start_frame = frame_range[0]
                end_frame = frame_range[1] if len(frame_range) == 2 else None
                start_frame = (
                    int(start_frame) if start_frame is not None else None
                )
                end_frame = int(end_frame) if end_frame is not None else None
                if (
                    start_frame is not None and
                    end_frame is not None and
                    start_frame > end_frame
                ):
                    raise ValueError(
                        f'Delete frame range start ({start_frame}) cannot exceed end ({end_frame}).'
                    )

                validated_ranges.append((start_frame, end_frame))
                range_rows_to_delete = source['particle'] == particle_label
                if start_frame is not None:
                    range_rows_to_delete &= source['frame'] >= start_frame
                if end_frame is not None:
                    range_rows_to_delete &= source['frame'] <= end_frame
                rows_to_delete |= range_rows_to_delete

            source = source[~rows_to_delete].copy()
            ranged_deletions[particle_label] = validated_ranges

        if link_frame_range is None:
            link_range_specs: list[FrameRange | None] = [None] * len(particles_to_link)
        else:
            link_range_specs = list(link_frame_range)
            if len(link_range_specs) > len(particles_to_link):
                raise ValueError(
                    'link_frame_range cannot contain more entries than particles_to_link.'
                )

            for frame_range in link_range_specs:
                if frame_range is None:
                    continue
                if (
                    not isinstance(frame_range, tuple) or
                    len(frame_range) != 2 or
                    any(isinstance(value, bool) or not isinstance(value, int) for value in frame_range)
                ):
                    raise ValueError(
                        'Each link frame range must be a (start_frame, end_frame) integer tuple or None.'
                    )
                start_frame, end_frame = frame_range
                if start_frame > end_frame:
                    raise ValueError(
                        f'Link frame range start ({start_frame}) cannot exceed end ({end_frame}).'
                    )

            link_range_specs.extend(
                [None] * (len(particles_to_link) - len(link_range_specs))
            )

        successful_replacement_map = {}
        successful_range_links = []
        skipped_links = []
        for label_group, frame_range in zip(particles_to_link, link_range_specs):
            if len(label_group) < 2:
                continue
            if frame_range is not None and len(label_group) != 2:
                raise ValueError(
                    'link_frame_range can only be used with two-particle link groups.'
                )

            label_to_keep = label_group[0]
            for label_to_replace in label_group[1:]:
                group_labels = (label_to_keep, label_to_replace)
                if any(label not in set(source['particle'].unique()) for label in group_labels):
                    continue

                if frame_range is not None:
                    start_frame, end_frame = frame_range
                    replacement_rows_in_range = (
                        (source['particle'] == label_to_replace) &
                        source['frame'].between(start_frame, end_frame, inclusive='both')
                    )
                    if not replacement_rows_in_range.any():
                        print(
                            f'No rows for particle {label_to_replace} were found in link '
                            f'frame range {frame_range}; no ranged link was applied.'
                        )
                        continue

                    replacement_frames = set(
                        source.loc[replacement_rows_in_range, 'frame']
                    )
                    keep_frames = set(
                        source.loc[source['particle'] == label_to_keep, 'frame']
                    )
                    overlapping_frames = sorted(
                        keep_frames.intersection(replacement_frames)
                    )
                    if overlapping_frames:
                        lower_label = min(group_labels)
                        higher_label = max(group_labels)
                        rows_to_remove = (
                            (source['particle'] == higher_label) &
                            source['frame'].isin(overlapping_frames)
                        )
                        source = source[~rows_to_remove].copy()
                        print(
                            f'Resolved {len(overlapping_frames)} overlapping frame(s) in '
                            f'link range {frame_range}; retained rows from lower-numbered '
                            f'particle {lower_label}.'
                        )

                    replacement_rows_in_range = (
                        (source['particle'] == label_to_replace) &
                        source['frame'].between(start_frame, end_frame, inclusive='both')
                    )
                    updated_row_count = int(replacement_rows_in_range.sum())
                    source.loc[
                        replacement_rows_in_range, 'particle'
                    ] = label_to_keep
                    successful_range_links.append({
                        'particle_to_replace': label_to_replace,
                        'particle_to_keep': label_to_keep,
                        'frame_range': frame_range,
                        'updated_row_count': updated_row_count
                    })
                    print(
                        f'Linked particle {label_to_replace} -> {label_to_keep} for frames '
                        f'{start_frame}-{end_frame}; updated {updated_row_count} row(s).'
                    )
                    continue

                keep_frames = set(
                    source.loc[source['particle'] == label_to_keep, 'frame']
                )
                replace_frames = set(
                    source.loc[source['particle'] == label_to_replace, 'frame']
                )
                overlapping_frames = sorted(keep_frames.intersection(replace_frames))

                if len(overlapping_frames) >= 4:
                    skipped_links.append((label_to_keep, label_to_replace))
                    overlap_rows = source[
                        source['particle'].isin(group_labels) &
                        source['frame'].isin(overlapping_frames)
                    ].sort_values(by=['frame', 'particle'])
                    print(
                        f'Cannot link particles {label_to_keep} and {label_to_replace}: '
                        f'{len(overlapping_frames)} overlapping frames '
                        f'{overlapping_frames}. The link was not applied.'
                    )
                    print('Overlapping particle rows:')
                    print(overlap_rows.to_string(index=False))
                    print('Use link_frame_range to apply a frame-limited link.')
                    continue

                if overlapping_frames:
                    lower_label = min(group_labels)
                    higher_label = max(group_labels)
                    rows_to_remove = (
                        (source['particle'] == higher_label) &
                        source['frame'].isin(overlapping_frames)
                    )
                    source = source[~rows_to_remove].copy()
                    print(
                        f'Resolved {len(overlapping_frames)} overlapping frame(s) for '
                        f'particles {label_to_keep} and {label_to_replace}; retained rows '
                        f'from lower-numbered particle {lower_label}.'
                    )

                source.loc[
                    source['particle'] == label_to_replace, 'particle'
                ] = label_to_keep
                successful_replacement_map[label_to_replace] = label_to_keep

        successful_splits = []
        for particle_label, split_frame in zip(
            particles_to_split, split_frame_specs
        ):
            particle_rows = source['particle'] == particle_label
            if not particle_rows.any():
                print(
                    f'Cannot split particle {particle_label}: no rows remain in the '
                    'updated source dataframe.'
                )
                continue

            rows_before_split = particle_rows & (source['frame'] < split_frame)
            rows_from_split = particle_rows & (source['frame'] >= split_frame)
            if not rows_before_split.any():
                first_frame = source.loc[particle_rows, 'frame'].min()
                print(
                    f'Cannot split particle {particle_label} at frame {split_frame}: '
                    f'the particle has no rows before that frame (first frame is '
                    f'{first_frame}).'
                )
                continue
            if not rows_from_split.any():
                last_frame = source.loc[particle_rows, 'frame'].max()
                print(
                    f'Cannot split particle {particle_label} at frame {split_frame}: '
                    f'the particle has no rows at or after that frame (last frame is '
                    f'{last_frame}).'
                )
                continue

            new_particle_label = next_particle_label
            next_particle_label += 1
            source.loc[rows_from_split, 'particle'] = new_particle_label
            split_row_count = int(rows_from_split.sum())
            successful_splits.append({
                'original_particle': particle_label,
                'new_particle': new_particle_label,
                'split_from_frame': split_frame,
                'updated_row_count': split_row_count
            })
            print(
                f'Split particle {particle_label} at frame {split_frame}; assigned '
                f'{split_row_count} row(s) from frame {split_frame} onward to new '
                f'particle {new_particle_label}.'
            )

        if sort_result:
            sort_columns = ["particle", "frame"] if "frame" in source.columns else ["particle"]
            source = source.reset_index(drop=True).sort_values(by=sort_columns).reset_index(drop=True)

        if is_update_particles:
            self._updated_particles_dataframes = source.copy()

        print(
            f"Manual particle label update complete. "
            f"Updated dataframe has {source['particle'].nunique()} unique particles and {len(source)} rows."
        )

        if addition_results:
            print(f'Restored particle labels: {addition_results}')

        if successful_replacement_map:
            print(f"Merged labels using replacement map: {successful_replacement_map}")

        if successful_range_links:
            print(f'Applied frame-limited particle links: {successful_range_links}')

        if fully_deleted_labels:
            print(f"Deleted complete particle labels: {sorted(fully_deleted_labels)}")

        if ranged_deletions:
            print(f"Deleted particle frame ranges: {ranged_deletions}")

        if successful_splits:
            print(f'Split particle labels: {successful_splits}')

        if skipped_links:
            print(f'Links skipped because of frame overlap: {skipped_links}')

        return source

    def clear_manually_updated_particle_labels(self) -> None:
        """
        Clear manually updated linked particle dataframe.
        """
        self._updated_particles_dataframes = pd.DataFrame()

    def plot_trajectories_using_trackpy(self, dataframe_to_plot: str | list[str] = 'active') -> None:
        """
        Plot particle trajectories using Trackpy's built-in plotting function.

        This preserves Trackpy's original plotting style and colormap.

        Args:
            dataframe_to_plot (str | list[str]): One or more of 'full', 'filtered',
                or 'updated'. Each selected dataframe is plotted in a separate figure.
                The legacy single-string values 'active' and 'both' remain supported.
        """
        dataframe_items = self.__get_tracking_dataframes_for_plot(dataframe_to_plot)

        for plot_label, plot_dataframe in dataframe_items:
            if plot_dataframe.empty:
                print(f'No {plot_label} linked dataframe available to plot.')
                continue

            plot_dataframe = plot_dataframe.reset_index(drop=True)
            plt.figure(figsize=(12, 6))
            tp.plot_traj(
                plot_dataframe,
                pos_columns=self._position_columns[::-1]
            )
            print(f'Trackpy trajectories plotted successfully for {plot_label} tracks.')

    def plot_trajectories_using_matplotlib(
        self,
        dataframe_to_plot: str | list[str] = 'active',
        figsize: tuple[float, float] = (12, 6),
        show_particle_ids: bool = False,
        particle_id_fontsize: int = 8,
        colormap_name: str = 'tab20',
        line_width: float = 1.5,
        alpha: float = 0.9
    ) -> None:
        """
        Plot particle trajectories directly with Matplotlib.

        This method provides more control than Trackpy's built-in plotting function,
        including figure size, a limited legend, and optional particle ID labels.

        Args:
            dataframe_to_plot (str | list[str]): One or more of 'full', 'filtered',
                or 'updated'. Each selected dataframe is plotted in a separate figure.
                The legacy single-string values 'active' and 'both' remain supported.
            figsize (tuple[float, float]): Matplotlib figure size in inches.
            show_particle_ids (bool): Whether to label each track with its particle ID at the final point.
            particle_id_fontsize (int): Font size for overlaid particle ID labels.
            colormap_name (str): Matplotlib colormap used to color particles.
            line_width (float): Width of each trajectory line.
            alpha (float): Transparency of each trajectory line.
        """
        dataframe_items = self.__get_tracking_dataframes_for_plot(dataframe_to_plot)

        for plot_label, plot_dataframe in dataframe_items:
            if plot_dataframe.empty:
                print(f'No {plot_label} linked dataframe available to plot.')
                continue

            required_columns = ['particle'] + self._position_columns
            missing_columns = [column for column in required_columns if column not in plot_dataframe.columns]
            if missing_columns:
                raise ValueError(f'Missing required columns for trajectory plotting: {missing_columns}')

            sort_columns = ['particle', 'frame'] if 'frame' in plot_dataframe.columns else ['particle']
            sorted_dataframe = plot_dataframe.reset_index(drop=True).sort_values(
                by=sort_columns
            )
            particle_ids = sorted_dataframe['particle'].unique()
            cmap = cm.get_cmap(colormap_name, len(particle_ids))

            _, ax = plt.subplots(figsize=figsize)

            for particle_index, particle_id in enumerate(particle_ids):
                particle_data = sorted_dataframe[sorted_dataframe['particle'] == particle_id]
                color = cmap(particle_index)
                x_values = particle_data[self._position_columns[1]]
                y_values = particle_data[self._position_columns[0]]

                ax.plot(
                    x_values,
                    y_values,
                    label=str(particle_id),
                    color=color,
                    linewidth=line_width,
                    alpha=alpha
                )

                if show_particle_ids and not particle_data.empty:
                    ax.text(
                        x_values.iloc[-1],
                        y_values.iloc[-1],
                        str(particle_id),
                        fontsize=particle_id_fontsize,
                        color=color
                    )

            ax.set_xlabel('Centroid X')
            ax.set_ylabel('Centroid Y')
            ax.invert_yaxis()
            ax.set_title(f'Trajectories - {plot_label}')
            self.__limit_particle_legend_entries(
                ax=ax,
                max_first_items=10,
                max_last_items=10,
                title='Particle ID'
            )
            plt.show()
            print(f'Matplotlib trajectories plotted successfully for {plot_label} tracks.')

    def sort_and_plot_scatter_of_trajectories(self, is_update_particles: bool = False, dataframe_to_plot: str | list[str] = 'active') -> None:
        """
        Plot scatter plots of particle trajectories.

        Args:
            is_update_particles (bool): Whether to replace the stored full linked dataframe
                with the sorted full dataframe. Default is False to keep plotting non-mutating.
            dataframe_to_plot (str | list[str]): One or more of 'full', 'filtered',
                or 'updated'. Each selected dataframe is plotted in a separate figure.
                The legacy single-string values 'active' and 'both' remain supported.
        """
        dataframe_items = self.__get_tracking_dataframes_for_plot(dataframe_to_plot)

        for plot_label, plot_dataframe in dataframe_items:
            if plot_dataframe.empty:
                print(f'No {plot_label} linked dataframe available to plot.')
                continue

            plt.figure(figsize=(12, 6))
            cols = ['centroid_x', 'centroid_y', 'frame', 'particle']
            sort_by = ['particle', 'frame']
            sorted_dataframe = self.__shape_and_sort_dataframe(
                plot_dataframe, cols, sort_by)

            if is_update_particles and plot_label == 'full':
                self._linked_particles_dataframes = sorted_dataframe

            sns.scatterplot(data=sorted_dataframe, x='centroid_y',
                            y='centroid_x', hue='particle', palette='bright', s=8)
            plt.xlabel('Centroid X')
            plt.ylabel('Centroid Y')
            plt.gca().invert_yaxis()
            plt.gca().set_title(f'Scatter plot of trajectories - {plot_label}')

            n_cols = len(
                sorted_dataframe) // 20000 if len(sorted_dataframe) > 20000 else 1

            self.__limit_particle_legend_entries(
                ax=plt.gca(),
                max_first_items=10,
                max_last_items=10,
                title='Particle ID'
            )
            plt.show()

    def visualize_particle_trajectories_from_origin(self, show_axes: bool = True, dataframe_to_plot: str | list[str] = 'active'):
        """
        Visualize particle trajectories shifted so each particle starts from the origin.

        Args:
            show_axes (bool): Whether to show axes lines at the origin.
            dataframe_to_plot (str | list[str]): One or more of 'full', 'filtered',
                or 'updated'. Each selected dataframe is plotted in a separate figure.
                The legacy single-string values 'active' and 'both' remain supported.
        Returns:
            None
        """
        dataframe_items = self.__get_tracking_dataframes_for_plot(dataframe_to_plot)

        for plot_label, df in dataframe_items:
            if df.empty:
                print(f'No {plot_label} linked dataframe available to plot.')
                continue

            shifted_df = pd.DataFrame()

            for particle_id in df['particle'].unique():
                particle_data = df[df['particle'] == particle_id].copy()

                particle_data['shifted_centroid_x'] = particle_data['centroid_x'] - \
                    particle_data['centroid_x'].iloc[0]
                particle_data['shifted_centroid_y'] = particle_data['centroid_y'] - \
                    particle_data['centroid_y'].iloc[0]

                shifted_df = pd.concat(
                    [shifted_df, particle_data], ignore_index=True)

            plt.figure(figsize=(12, 6))
            sns.scatterplot(data=shifted_df, x='shifted_centroid_y',
                            y='shifted_centroid_x', hue='particle', palette='bright', s=50)

            plt.xlabel('Centroid X')
            plt.ylabel('Centroid Y')

            for particle_id in shifted_df['particle'].unique():
                particle_data = shifted_df[shifted_df['particle'] == particle_id]
                plt.plot(particle_data['shifted_centroid_y'],
                         particle_data['shifted_centroid_x'])

            plt.gca().invert_yaxis()
            plt.gca().set_title(f'Trajectories of particles initiated from origin - {plot_label}')

            if show_axes:
                plt.axhline(0, color='black', linewidth=1)
                plt.axvline(0, color='black', linewidth=1)

            n_cols = len(
                shifted_df) // 20000 if len(shifted_df) > 20000 else 1

            self.__limit_particle_legend_entries(
                ax=plt.gca(),
                max_first_items=10,
                max_last_items=10,
                title='Particle ID'
            )

            plt.show()

    def visualize_particle_heatmap(self, dataframe_to_plot: str | list[str] = 'active'):
        """
        Create heatmaps of particle densities based on original centroids.

        Args:
            dataframe_to_plot (str | list[str]): One or more of 'full', 'filtered',
                or 'updated'. Each selected dataframe is plotted in a separate figure.
                The legacy single-string values 'active' and 'both' remain supported.
        """
        dataframe_items = self.__get_tracking_dataframes_for_plot(dataframe_to_plot)

        for plot_label, df in dataframe_items:
            if df.empty:
                print(f'No {plot_label} linked dataframe available to plot.')
                continue

            plt.figure(figsize=(10, 8))
            sns.kdeplot(
                x=df['centroid_y'],
                y=df['centroid_x'],
                fill=True,
                cmap='viridis',
                cbar=True
            )
            plt.title(f'Heatmap of Particle Densities - {plot_label}')
            plt.xlabel('Centroid X')
            plt.ylabel('Centroid Y')
            plt.gca().invert_yaxis()
            plt.show()

    def get_directory(self):
        """
        Retrieves the working directory.
        Returns:
        str: The working directory.
        """
        return self._directory

    def set_linked_particles_dataframes(self, linked_particles_dataframes: pd.DataFrame) -> None:
        """
        Set the linked particles dataframes.
        Args:
            linked_particles_dataframes (pd.DataFrame): The linked particles dataframes.
        Returns:
            None
        """
        self._linked_particles_dataframes = linked_particles_dataframes
        self._filtered_particles_dataframes = pd.DataFrame()
        self._updated_particles_dataframes = pd.DataFrame()
        self._eliminated_particles_dataframe = pd.DataFrame()
        self._linked_particles_metadata = {}

    @staticmethod
    def __calculate_track_motion_metrics(
        positions: np.ndarray
    ) -> tuple[float, float, float]:
        """
        Calculate endpoint displacement, cumulative path, and maximum excursion.
        """
        if positions.ndim != 2 or positions.shape[1] != 2 or len(positions) == 0:
            return np.nan, np.nan, np.nan

        endpoints_are_finite = np.isfinite(positions[[0, -1]]).all()
        endpoint_net_displacement = (
            float(np.linalg.norm(positions[-1] - positions[0]))
            if endpoints_are_finite else np.nan
        )

        if not np.isfinite(positions).all():
            return endpoint_net_displacement, np.nan, np.nan

        position_steps = np.diff(positions, axis=0)
        cumulative_path_length = (
            float(np.linalg.norm(position_steps, axis=1).sum())
            if len(position_steps) > 0 else 0.0
        )
        max_displacement_from_start = float(
            np.linalg.norm(positions - positions[0], axis=1).max()
        )
        return (
            endpoint_net_displacement,
            cumulative_path_length,
            max_displacement_from_start
        )

    def __get_export_position_columns(
        self,
        dataframe: pd.DataFrame
    ) -> list[str]:
        """
        Resolve the two coordinate columns used by derived XLSX export sheets.
        """
        centroid_columns = ['centroid_x', 'centroid_y']
        if all(column in dataframe.columns for column in centroid_columns):
            return centroid_columns

        if (
            isinstance(self._position_columns, (list, tuple)) and
            len(self._position_columns) == 2 and
            all(column in dataframe.columns for column in self._position_columns)
        ):
            return list(self._position_columns)

        raise ValueError(
            'Derived linked-track export sheets require centroid_x and '
            'centroid_y, or two available tracker position columns.'
        )

    def __build_track_details_dataframe(
        self,
        dataframe: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Build one motion-summary row for every particle in a linked dataframe.
        """
        position_columns = self.__get_export_position_columns(dataframe)
        required_columns = ['particle', 'frame'] + position_columns
        missing_columns = [
            column for column in required_columns
            if column not in dataframe.columns
        ]
        if missing_columns:
            raise ValueError(
                'Cannot build linked-track details because the dataframe is '
                f'missing required columns: {missing_columns}'
            )

        detail_rows = []
        for particle_id, particle_rows in dataframe.groupby(
            'particle', sort=True
        ):
            sorted_rows = particle_rows.copy()
            sorted_rows['_frame_numeric'] = pd.to_numeric(
                sorted_rows['frame'], errors='coerce'
            )
            if not np.isfinite(
                sorted_rows['_frame_numeric'].to_numpy(dtype=float)
            ).all():
                raise ValueError(
                    f'Particle {particle_id} contains a non-numeric frame value; '
                    'track details require numeric frames.'
                )
            sorted_rows = sorted_rows.sort_values(
                by='_frame_numeric', kind='stable'
            )
            positions = sorted_rows[
                position_columns
            ].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)
            (
                endpoint_net_displacement,
                cumulative_path_length,
                _
            ) = self.__calculate_track_motion_metrics(positions)

            detail_rows.append({
                'particle': particle_id,
                'first_frame': sorted_rows.iloc[0]['_frame_numeric'],
                'last_frame': sorted_rows.iloc[-1]['_frame_numeric'],
                'first_centroid_x': positions[0, 0],
                'first_centroid_y': positions[0, 1],
                'last_centroid_x': positions[-1, 0],
                'last_centroid_y': positions[-1, 1],
                'endpoint_net_displacement_pixels': (
                    endpoint_net_displacement
                ),
                'cumulative_path_length_pixels': cumulative_path_length
            })

        return pd.DataFrame(
            detail_rows,
            columns=self.TRACK_DETAILS_COLUMNS
        )

    @staticmethod
    def __select_collision_dataframes(
        exported_track_dataframes: dict[str, pd.DataFrame],
        collision_source: str
    ) -> dict[str, pd.DataFrame]:
        """
        Select exported linked-track dataframes for possible-collision review.
        """
        if collision_source == 'all':
            return exported_track_dataframes.copy()

        if collision_source == 'active':
            for dataframe_name in ('updated', 'filtered', 'full'):
                if dataframe_name in exported_track_dataframes:
                    return {
                        dataframe_name: exported_track_dataframes[dataframe_name]
                    }

        if collision_source not in exported_track_dataframes:
            raise ValueError(
                f"collision_source='{collision_source}' is not available among "
                'the linked-track dataframes selected for export.'
            )

        return {
            collision_source: exported_track_dataframes[collision_source]
        }

    def __build_possible_collisions_dataframe(
        self,
        collision_dataframes: dict[str, pd.DataFrame],
        collision_distance_threshold: float | None
    ) -> pd.DataFrame:
        """
        Build pairwise, consecutive-frame possible-collision review events.
        """
        collision_events = []
        for dataframe_name, dataframe in collision_dataframes.items():
            dataframe_events = self.__find_possible_collision_events(
                dataframe=dataframe,
                dataframe_name=dataframe_name,
                collision_distance_threshold=collision_distance_threshold
            )
            collision_events.extend(dataframe_events)

        collision_events.sort(key=lambda event: (
            list(collision_dataframes).index(event['source_dataframe']),
            event['first_frame'],
            event['minimum_centroid_distance_pixels'],
            event['particle_1'],
            event['particle_2']
        ))
        for event_number, collision_event in enumerate(
            collision_events, start=1
        ):
            collision_event['collision_event_id'] = f'C{event_number:05d}'

        return pd.DataFrame(
            collision_events,
            columns=self.POSSIBLE_COLLISION_COLUMNS
        )

    def __find_possible_collision_events(
        self,
        dataframe: pd.DataFrame,
        dataframe_name: str,
        collision_distance_threshold: float | None
    ) -> list[dict]:
        """
        Find consecutive close-approach events between distinct particle pairs.
        """
        position_columns = self.__get_export_position_columns(dataframe)
        required_columns = ['particle', 'frame'] + position_columns
        if collision_distance_threshold is None:
            required_columns.append('major_axis_length')
        missing_columns = [
            column for column in required_columns
            if column not in dataframe.columns
        ]
        if missing_columns:
            if (
                collision_distance_threshold is None and
                missing_columns == ['major_axis_length']
            ):
                raise ValueError(
                    'Size-aware collision review requires major_axis_length. '
                    'Provide collision_distance_threshold to use a fixed pixel '
                    'distance instead.'
                )
            raise ValueError(
                'Cannot evaluate possible collisions because the dataframe is '
                f'missing required columns: {missing_columns}'
            )

        working_columns = ['particle', 'frame'] + position_columns
        if 'major_axis_length' in dataframe.columns:
            working_columns.append('major_axis_length')
        working_dataframe = dataframe[working_columns].copy()
        for column in working_columns:
            working_dataframe[column] = pd.to_numeric(
                working_dataframe[column], errors='coerce'
            )
        finite_required_columns = ['particle', 'frame'] + position_columns
        finite_rows = np.isfinite(
            working_dataframe[finite_required_columns].to_numpy(dtype=float)
        ).all(axis=1)
        working_dataframe = working_dataframe.loc[finite_rows]

        candidate_rows = []
        threshold_basis = (
            'fixed_centroid_distance'
            if collision_distance_threshold is not None
            else 'half_sum_major_axis_lengths'
        )
        for frame_value, frame_dataframe in working_dataframe.groupby(
            'frame', sort=True
        ):
            frame_dataframe = frame_dataframe.sort_values(
                by='particle', kind='stable'
            )
            if len(frame_dataframe) < 2:
                continue

            positions = frame_dataframe[
                position_columns
            ].to_numpy(dtype=float)
            particles = frame_dataframe['particle'].to_numpy(dtype=float)

            if collision_distance_threshold is None:
                major_axis_lengths = frame_dataframe[
                    'major_axis_length'
                ].to_numpy(dtype=float)
                finite_positive_lengths = major_axis_lengths[
                    np.isfinite(major_axis_lengths) &
                    (major_axis_lengths > 0)
                ]
                if len(finite_positive_lengths) < 2:
                    continue
                maximum_search_distance = float(
                    finite_positive_lengths.max()
                )
            else:
                major_axis_lengths = np.full(
                    len(frame_dataframe), np.nan, dtype=float
                )
                if 'major_axis_length' in frame_dataframe.columns:
                    major_axis_lengths = frame_dataframe[
                        'major_axis_length'
                    ].to_numpy(dtype=float)
                maximum_search_distance = collision_distance_threshold

            potential_pairs = sorted(
                cKDTree(positions).query_pairs(
                    r=maximum_search_distance
                )
            )
            if not potential_pairs:
                continue
            pair_indices = np.asarray(potential_pairs, dtype=int)
            row_indices = pair_indices[:, 0]
            column_indices = pair_indices[:, 1]
            pair_distances = np.linalg.norm(
                positions[row_indices] - positions[column_indices], axis=1
            )

            if collision_distance_threshold is None:
                pair_thresholds = (
                    major_axis_lengths[row_indices] +
                    major_axis_lengths[column_indices]
                ) / 2
            else:
                pair_thresholds = np.full(
                    len(row_indices),
                    collision_distance_threshold,
                    dtype=float
                )

            qualifying_pairs = (
                np.isfinite(pair_distances) &
                np.isfinite(pair_thresholds) &
                (pair_thresholds > 0) &
                (particles[row_indices] != particles[column_indices]) &
                (pair_distances <= pair_thresholds)
            )
            for pair_index in np.flatnonzero(qualifying_pairs):
                first_index = row_indices[pair_index]
                second_index = column_indices[pair_index]
                candidate_rows.append({
                    'particle_1': particles[first_index],
                    'particle_2': particles[second_index],
                    'frame': frame_value,
                    'centroid_distance_pixels': pair_distances[pair_index],
                    'collision_threshold_pixels': pair_thresholds[pair_index],
                    'particle_1_centroid_x': positions[first_index, 0],
                    'particle_1_centroid_y': positions[first_index, 1],
                    'particle_2_centroid_x': positions[second_index, 0],
                    'particle_2_centroid_y': positions[second_index, 1],
                    'particle_1_major_axis_length': (
                        major_axis_lengths[first_index]
                    ),
                    'particle_2_major_axis_length': (
                        major_axis_lengths[second_index]
                    )
                })

        if not candidate_rows:
            return []

        candidate_dataframe = pd.DataFrame(candidate_rows).sort_values(
            by=['particle_1', 'particle_2', 'frame'], kind='stable'
        )
        collision_events = []
        for _, pair_rows in candidate_dataframe.groupby(
            ['particle_1', 'particle_2'], sort=True
        ):
            pair_rows = pair_rows.sort_values(by='frame', kind='stable')
            event_groups = pair_rows['frame'].diff().gt(1).cumsum()
            for _, event_rows in pair_rows.groupby(event_groups, sort=True):
                closest_row = event_rows.loc[
                    event_rows['centroid_distance_pixels'].idxmin()
                ]
                collision_events.append({
                    'source_dataframe': dataframe_name,
                    'collision_event_id': '',
                    'particle_1': closest_row['particle_1'],
                    'particle_2': closest_row['particle_2'],
                    'first_frame': event_rows['frame'].min(),
                    'last_frame': event_rows['frame'].max(),
                    'collision_frame_count': int(
                        event_rows['frame'].nunique()
                    ),
                    'closest_frame': closest_row['frame'],
                    'minimum_centroid_distance_pixels': closest_row[
                        'centroid_distance_pixels'
                    ],
                    'mean_centroid_distance_pixels': event_rows[
                        'centroid_distance_pixels'
                    ].mean(),
                    'collision_threshold_pixels_at_closest_frame': closest_row[
                        'collision_threshold_pixels'
                    ],
                    'particle_1_centroid_x_at_closest_frame': closest_row[
                        'particle_1_centroid_x'
                    ],
                    'particle_1_centroid_y_at_closest_frame': closest_row[
                        'particle_1_centroid_y'
                    ],
                    'particle_2_centroid_x_at_closest_frame': closest_row[
                        'particle_2_centroid_x'
                    ],
                    'particle_2_centroid_y_at_closest_frame': closest_row[
                        'particle_2_centroid_y'
                    ],
                    'particle_1_major_axis_length_at_closest_frame': closest_row[
                        'particle_1_major_axis_length'
                    ],
                    'particle_2_major_axis_length_at_closest_frame': closest_row[
                        'particle_2_major_axis_length'
                    ],
                    'threshold_basis': threshold_basis,
                    'review_status': 'Pending',
                    'review_notes': ''
                })

        return collision_events

    def __prepare_dataframe_for_export(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare a linked/filtered particle dataframe for export without modifying the original.
        """
        if dataframe.empty:
            return pd.DataFrame()

        return self.__remove_legacy_coordinate_columns(dataframe)

    @staticmethod
    def __remove_legacy_coordinate_columns(
        dataframe: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Return a copy without obsolete swapped-coordinate export columns.
        """
        return dataframe.drop(
            columns=['new_x', 'new_y'],
            errors='ignore'
        ).copy()

    def overlay_tracks_on_video(
        self,
        output_video_filename: str,
        colormap_name: str = "viridis",
        frame_index_offset: int = -1,
        show_labels: bool = True,
        font_scale: float = 0.5,
        font_thickness: int = 1,
        label_offset_x: int = 5,
        label_offset_y: int = -5,
        dataframe: pd.DataFrame | None = None,
        tracks_to_overlay: int | tuple[int | float, int | float] | list[int | float] | str | None = 'all',
        specific_tracks_to_overlay: list[int | float] | None = None,
        video_quality: str = 'high',
        video_codec: str | None = None
    ) -> None:
        """
        Overlays tracked particle trajectories onto the original captured frames and saves the output video.
        Uses internal parent objects for video information.

        Args:
            output_video_filename (str): Name of the output video file to save with overlaid tracks.
                Missing parent folders are created automatically. Existing folders
                and unrelated files are preserved; a same-named video is overwritten.
            colormap_name (str, optional): Name of the matplotlib colormap to assign unique colors to particles.
                Default is "viridis".
            frame_index_offset (int, optional): Offset to adjust frame indexing differences between
                the tracking data and video frames. Default is -1.
            show_labels (bool, optional): Whether to overlay particle ID labels on the video frames. Default is True.
            font_scale (float, optional): Font size for particle ID labels. Default is 0.5.
            font_thickness (int, optional): Thickness of the label text. Default is 1.
            label_offset_x (int, optional): Horizontal offset for label placement. Default is 5.
            label_offset_y (int, optional): Vertical offset for label placement. Default is -5.
            dataframe (pd.DataFrame | None): Optional linked-particle dataframe to
                overlay. When None, manually updated tracks are preferred, followed by
                filtered and then full linked tracks. The supplied dataframe is not
                modified.
            tracks_to_overlay (int | tuple[int | float, int | float] | list[int | float] | str | None):
                Selects the first N available particle IDs when an integer is supplied.
                A two-value tuple selects every available particle ID within that
                inclusive range; for example, (10, 30) selects available IDs 10 through
                30. A list selects those exact particle IDs, while an empty list or None
                selects no base tracks. Use 'all' (case-insensitive) to select every
                particle. Defaults to 'all'.
            specific_tracks_to_overlay (list[int | float] | None): Particle IDs to add
                explicitly. The result is the union of the base selection and these
                specific IDs. None or an empty list adds no IDs. An empty combined
                selection exports the video without track or label overlays. Unavailable
                IDs raise an error instead of being silently ignored.
            video_quality (str): Output-resolution preset. Options are 'original' or
                'high' (100%), 'medium' (75%), 'low' (50%), and 'preview' (25%). Lower
                resolutions reduce processing time and file size. Defaults to 'high'.
            video_codec (str | None): Optional four-character OpenCV codec. None selects
                'mp4v' for MP4/MOV/M4V output and 'MJPG' for AVI output. Codec availability
                depends on the local OpenCV/FFmpeg installation.
            NOTE: The overlay is drawn on Capture.get_captured_working_frames(), not on Identify._working_masks.
                This keeps tracks overlaid on the original microscopy frames rather than on masks.

        Raises:
            ValueError: If tracking data is not available.
            ValueError: If no captured frames are available from the Capture class.
        """
        if dataframe is not None and not isinstance(dataframe, pd.DataFrame):
            raise TypeError('dataframe must be a pandas DataFrame or None.')

        tracking_data = (
            dataframe.copy()
            if dataframe is not None
            else self.get_linked_particles_dataframe(
                prefer_updated=True,
                prefer_filtered=True
            ).copy()
        )
        if tracking_data.empty:
            raise ValueError(
                "Tracking data is empty. Ensure particles are linked before overlaying tracks.")

        required_tracking_columns = {
            'frame', 'particle', 'centroid_x', 'centroid_y'
        }
        missing_tracking_columns = sorted(
            required_tracking_columns.difference(tracking_data.columns)
        )
        if missing_tracking_columns:
            raise ValueError(
                'Tracking dataframe is missing required columns: '
                f'{missing_tracking_columns}'
            )

        tracking_data = self.__select_tracking_data_for_overlay(
            tracking_data=tracking_data,
            tracks_to_overlay=tracks_to_overlay,
            specific_tracks_to_overlay=specific_tracks_to_overlay
        )

        # Retrieve the working directory from the Capture class via the Identify object
        working_directory = self._parent.get_directory()

        # Construct the full output path. exist_ok preserves an existing folder and
        # its contents; OpenCV replaces only a same-named output video file.
        output_video_path = os.path.join(
            working_directory, output_video_filename)
        output_video_directory = os.path.dirname(output_video_path)
        if output_video_directory:
            os.makedirs(output_video_directory, exist_ok=True)

        # Retrieve the original captured frames from the Capture object and validate.
        # Track overlays should be drawn on the raw captured frames, not on the
        # segmented masks stored in Identify._working_masks.
        frames = self._parent._parent.get_captured_working_frames()
        if not frames:
            raise ValueError(
                "No captured frames available. Ensure the video is loaded and processed correctly.")

        # Retrieve frame rate information
        frame_rate_info = self._parent._parent.get_frame_rate()
        fps = frame_rate_info.get(
            'user_provided_fps') or frame_rate_info.get('default_fps') or 15

        output_frame_size = self.__get_video_output_size(
            frame=frames[0],
            video_quality=video_quality
        )
        original_height, original_width = np.asarray(frames[0]).shape[:2]
        output_width, output_height = output_frame_size
        coordinate_scale_x = output_width / original_width
        coordinate_scale_y = output_height / original_height

        # Track coordinates and label styling must use the selected output resolution.
        tracking_data['centroid_y'] = (
            tracking_data['centroid_y'].astype(float) * coordinate_scale_x
        )
        tracking_data['centroid_x'] = (
            tracking_data['centroid_x'].astype(float) * coordinate_scale_y
        )
        display_scale = min(coordinate_scale_x, coordinate_scale_y)
        output_font_scale = max(0.3, font_scale * display_scale)
        output_font_thickness = max(1, int(round(font_thickness * display_scale)))
        output_line_thickness = max(1, int(round(2 * display_scale)))
        output_label_offset_x = int(round(label_offset_x * coordinate_scale_x))
        output_label_offset_y = int(round(label_offset_y * coordinate_scale_y))

        # Sort tracking data by frame and particle for consistency
        tracking_data = tracking_data.reset_index(drop=True)
        tracking_data = tracking_data.sort_values(by=["frame", "particle"])

        # Generate unique colors for particles
        particle_colors = self._generate_particle_colors(
            tracking_data, colormap_name)

        # Initialize video writer using a prepared copy of the first captured frame.
        first_overlay_frame = self._prepare_frame_for_overlay(
            frames[0], output_frame_size=output_frame_size
        )
        video_writer = self._initialize_video_writer(
            first_overlay_frame, fps, output_video_path, video_codec=video_codec)

        # Dictionary to store cumulative particle tracks
        particle_tracks = {particle: []
                           for particle in tracking_data['particle'].unique()}

        # Total number of frames
        total_frames = len(frames)

        # Initialize progress bar
        try:
            with tqdm(total=total_frames, desc="Overlaying Tracks on Video") as progress_bar:
                for current_frame_index, raw_frame in enumerate(frames):
                    # Work on a video-display copy so drawing the overlay does not mutate
                    # the original captured frames stored in the Capture object.
                    frame = self._prepare_frame_for_overlay(
                        raw_frame, output_frame_size=output_frame_size
                    )

                    # Adjust frame index based on the offset
                    adjusted_frame_index = current_frame_index + 1 + frame_index_offset

                    # Extract particle data for the current frame
                    frame_data = tracking_data[tracking_data['frame']
                                               == adjusted_frame_index]

                    # Update particle tracks
                    self._update_particle_tracks(frame_data, particle_tracks)

                    # Draw particle trajectories on the frame
                    self._draw_particle_tracks(
                        frame, particle_tracks, particle_colors,
                        line_thickness=output_line_thickness)

                    # Optionally, draw particle ID labels at the last tracked position
                    if show_labels:
                        for particle_id, track_points in particle_tracks.items():
                            if track_points:
                                last_position = track_points[-1]
                                label_position = (
                                    last_position[0] + output_label_offset_x,
                                    last_position[1] + output_label_offset_y
                                )
                                cv2.putText(
                                    frame, str(particle_id), label_position,
                                    cv2.FONT_HERSHEY_SIMPLEX, output_font_scale,
                                    particle_colors.get(
                                        particle_id, (255, 255, 255)),
                                    output_font_thickness, cv2.LINE_AA
                                )

                    # Write the processed frame to the output video
                    video_writer.write(frame)
                    progress_bar.update(1)
        finally:
            # Release video writer resources even if frame processing raises an error.
            video_writer.release()

        overlay_status = (
            'with overlaid tracks'
            if particle_tracks
            else 'without track overlays'
        )
        print(
            f'Processed {video_quality.lower()} quality video {overlay_status} '
            f'saved to {output_video_path}')

    # Private methods
    def __select_tracking_data_for_overlay(
        self,
        tracking_data: pd.DataFrame,
        tracks_to_overlay: int | tuple[int | float, int | float] | list[int | float] | str | None,
        specific_tracks_to_overlay: list[int | float] | None
    ) -> pd.DataFrame:
        """
        Return tracking rows for the requested particle IDs.

        An empty combined selection returns an empty dataframe with the original
        columns so the video can be exported without track or label overlays.
        """
        if tracking_data['particle'].isna().any():
            raise ValueError(
                'Tracking dataframe contains missing particle IDs and cannot be overlaid.'
            )

        available_particle_ids = list(pd.unique(tracking_data['particle']))
        try:
            available_particle_ids = sorted(available_particle_ids)
        except TypeError:
            available_particle_ids = sorted(
                available_particle_ids, key=lambda particle_id: str(particle_id)
            )

        if specific_tracks_to_overlay is None:
            specific_particle_ids = []
        elif not isinstance(specific_tracks_to_overlay, (list, tuple, set, np.ndarray)):
            raise TypeError(
                'specific_tracks_to_overlay must be a list-like collection or None.'
            )
        else:
            specific_particle_ids = list(specific_tracks_to_overlay)

        invalid_specific_particle_ids = [
            particle_id for particle_id in specific_particle_ids
            if (
                isinstance(particle_id, bool) or
                not isinstance(
                    particle_id,
                    (int, float, np.integer, np.floating)
                ) or
                not np.isfinite(particle_id)
            )
        ]
        if invalid_specific_particle_ids:
            raise ValueError(
                'specific_tracks_to_overlay must contain finite numeric particle IDs. '
                f'Invalid IDs: {invalid_specific_particle_ids}'
            )

        unavailable_particle_ids = [
            particle_id for particle_id in specific_particle_ids
            if particle_id not in available_particle_ids
        ]
        if unavailable_particle_ids:
            raise ValueError(
                'Requested specific particle IDs are not available: '
                f'{unavailable_particle_ids}. Available IDs: {available_particle_ids}'
            )

        if isinstance(tracks_to_overlay, bool):
            raise TypeError(
                "tracks_to_overlay must be a positive integer, inclusive range tuple, "
                "exact-ID list, 'all', or None."
            )

        if isinstance(tracks_to_overlay, (int, np.integer)):
            if tracks_to_overlay < 1:
                raise ValueError('tracks_to_overlay must be at least 1 when using a count.')
            selected_particle_ids = available_particle_ids[:int(tracks_to_overlay)]
        elif isinstance(tracks_to_overlay, tuple):
            if len(tracks_to_overlay) != 2:
                raise ValueError(
                    'A tracks_to_overlay range must be a (start_id, end_id) tuple.'
                )

            start_particle_id, end_particle_id = tracks_to_overlay
            invalid_range_values = [
                particle_id for particle_id in tracks_to_overlay
                if (
                    isinstance(particle_id, bool) or
                    not isinstance(
                        particle_id,
                        (int, float, np.integer, np.floating)
                    ) or
                    not np.isfinite(particle_id)
                )
            ]
            if invalid_range_values:
                raise ValueError(
                    'tracks_to_overlay range values must be finite numeric particle IDs. '
                    f'Invalid values: {invalid_range_values}'
                )
            if start_particle_id > end_particle_id:
                raise ValueError(
                    'tracks_to_overlay range start cannot be greater than its end.'
                )

            non_numeric_available_ids = [
                particle_id for particle_id in available_particle_ids
                if (
                    isinstance(particle_id, bool) or
                    not isinstance(
                        particle_id,
                        (int, float, np.integer, np.floating)
                    ) or
                    not np.isfinite(particle_id)
                )
            ]
            if non_numeric_available_ids:
                raise ValueError(
                    'Range selection requires finite numeric particle IDs. Invalid '
                    f'available IDs: {non_numeric_available_ids}'
                )

            selected_particle_ids = [
                particle_id for particle_id in available_particle_ids
                if start_particle_id <= particle_id <= end_particle_id
            ]
            if not selected_particle_ids:
                raise ValueError(
                    'No available particle IDs were found in inclusive range '
                    f'{tracks_to_overlay}.'
                )
        elif isinstance(tracks_to_overlay, list):
            invalid_track_particle_ids = [
                particle_id for particle_id in tracks_to_overlay
                if (
                    isinstance(particle_id, bool) or
                    not isinstance(
                        particle_id,
                        (int, float, np.integer, np.floating)
                    ) or
                    not np.isfinite(particle_id)
                )
            ]
            if invalid_track_particle_ids:
                raise ValueError(
                    'tracks_to_overlay must contain finite numeric particle IDs. '
                    f'Invalid IDs: {invalid_track_particle_ids}'
                )

            unavailable_track_particle_ids = [
                particle_id for particle_id in tracks_to_overlay
                if particle_id not in available_particle_ids
            ]
            if unavailable_track_particle_ids:
                raise ValueError(
                    'Requested tracks_to_overlay particle IDs are not available: '
                    f'{unavailable_track_particle_ids}. '
                    f'Available IDs: {available_particle_ids}'
                )

            selected_particle_ids = []
            for particle_id in tracks_to_overlay:
                if particle_id not in selected_particle_ids:
                    selected_particle_ids.append(particle_id)
        elif isinstance(tracks_to_overlay, str):
            if tracks_to_overlay.lower() != 'all':
                raise ValueError("The only supported string for tracks_to_overlay is 'all'.")
            selected_particle_ids = available_particle_ids.copy()
        elif tracks_to_overlay is None:
            selected_particle_ids = []
        else:
            raise TypeError(
                "tracks_to_overlay must be a positive integer, inclusive range tuple, "
                "exact-ID list, 'all', or None."
            )

        for particle_id in specific_particle_ids:
            if particle_id not in selected_particle_ids:
                selected_particle_ids.append(particle_id)

        selected_tracking_data = tracking_data[
            tracking_data['particle'].isin(selected_particle_ids)
        ].copy()
        print(
            f'Overlaying {len(selected_particle_ids)} of '
            f'{len(available_particle_ids)} available particle tracks.'
        )
        return selected_tracking_data

    def __get_video_output_size(
        self,
        frame: np.ndarray,
        video_quality: str
    ) -> tuple[int, int]:
        """
        Return an even output width and height for a video-quality preset.
        """
        if not isinstance(video_quality, str):
            raise TypeError('video_quality must be a string.')

        normalized_quality = video_quality.lower()
        if normalized_quality not in self.VIDEO_QUALITY_SCALES:
            raise ValueError(
                'video_quality must be one of: original, high, medium, low, preview.'
            )

        frame_shape = np.asarray(frame).shape
        if len(frame_shape) < 2:
            raise ValueError(
                f'Unsupported frame shape for video export: {frame_shape}'
            )

        frame_height, frame_width = frame_shape[:2]
        output_scale = self.VIDEO_QUALITY_SCALES[normalized_quality]

        def make_even(dimension: int) -> int:
            scaled_dimension = max(2, int(round(dimension * output_scale)))
            return scaled_dimension if scaled_dimension % 2 == 0 else scaled_dimension - 1

        return make_even(frame_width), make_even(frame_height)

    def __load_linked_dataframe_from_path(self, file_path: str) -> tuple[pd.DataFrame, dict]:
        """
        Load a linked dataframe and optional metadata from a supported file path.
        """
        extension = os.path.splitext(file_path)[1].lower()

        if extension == '.csv':
            return pd.read_csv(file_path), {}

        if extension in ('.pkl', '.pickle'):
            loaded_object = pd.read_pickle(file_path)
            if isinstance(loaded_object, pd.DataFrame):
                return loaded_object, {}
            if isinstance(loaded_object, dict) and 'dataframe' in loaded_object:
                return loaded_object['dataframe'], loaded_object.get('metadata', {})
            raise ValueError(f'Unsupported pickle contents in linked particles file: {file_path}')

        if extension == '.npy':
            loaded_object = np.load(file_path, allow_pickle=True).item()
            dataframe = pd.DataFrame(
                loaded_object['data'],
                columns=loaded_object['columns']
            )
            metadata = loaded_object.get('metadata', {})
            return dataframe, metadata

        if extension == '.xlsx':
            return pd.read_excel(file_path), {}

        raise ValueError(
            "Unsupported linked particles file extension. Use '.csv', '.xlsx', '.pkl', '.pickle', or '.npy'."
        )

    def __load_linked_metadata_from_path(self, file_path: str) -> dict:
        """
        Load linked-particle metadata from a supported metadata file.
        """
        extension = os.path.splitext(file_path)[1].lower()

        if extension in ('.pkl', '.pickle'):
            metadata = pd.read_pickle(file_path)
            return metadata if isinstance(metadata, dict) else {}

        if extension == '.csv':
            return self.__metadata_dataframe_to_dict(pd.read_csv(file_path))

        if extension == '.xlsx':
            return self.__metadata_dataframe_to_dict(pd.read_excel(file_path))

        if extension == '.npy':
            loaded_object = np.load(file_path, allow_pickle=True).item()
            return loaded_object.get('metadata', loaded_object if isinstance(loaded_object, dict) else {})

        raise ValueError(
            "Unsupported metadata file extension. Use '.csv', '.xlsx', '.pkl', '.pickle', or '.npy'."
        )

    def __metadata_dataframe_to_dict(self, metadata_dataframe: pd.DataFrame) -> dict:
        """
        Convert a two-column metadata dataframe into a dictionary.
        """
        if metadata_dataframe.empty or 'parameter' not in metadata_dataframe.columns or 'value' not in metadata_dataframe.columns:
            return {}

        metadata = {}
        for _, row in metadata_dataframe.iterrows():
            key = row['parameter']
            value = row['value']
            if pd.isna(value):
                metadata[key] = None
                continue
            try:
                metadata[key] = eval(value, {"__builtins__": {}}, {})
            except Exception:
                metadata[key] = value
        return metadata

    def __restore_position_columns_from_loaded_dataframes(self) -> None:
        """
        Restore position columns after loading linked tracks.
        """
        if self._linked_particles_metadata and 'position_columns' in self._linked_particles_metadata:
            self._position_columns = self._linked_particles_metadata['position_columns']
            return

        if all(column in self._linked_particles_dataframes.columns for column in self.DEFAULT_POSITION_COLUMNS):
            self._position_columns = self.DEFAULT_POSITION_COLUMNS

    def __get_tracking_frame_shape(self) -> tuple[int, int]:
        """
        Return the image height and width used to classify edge detections.
        """
        image_collections = []
        if hasattr(self._parent, 'get_working_masks'):
            image_collections.append(self._parent.get_working_masks())

        capture_object = getattr(self._parent, '_parent', None)
        if capture_object is not None and hasattr(capture_object, 'get_captured_working_frames'):
            image_collections.append(capture_object.get_captured_working_frames())

        for images in image_collections:
            if images is None or len(images) == 0:
                continue
            image_shape = np.asarray(images[0]).shape
            if len(image_shape) >= 2:
                return int(image_shape[0]), int(image_shape[1])

        raise ValueError(
            "edge_max_memory requires at least one working mask or captured frame "
            "to determine the image boundaries."
        )

    def __is_detection_near_edge(
        self,
        detection: pd.Series,
        frame_shape: tuple[int, int],
        edge_margin: float
    ) -> bool:
        """
        Return whether a detection centroid lies within edge_margin of any image edge.

        Region-property centroids are stored in image-axis order: the first position
        column maps to image height and the second maps to image width.
        """
        first_position = float(detection[self._position_columns[0]])
        second_position = float(detection[self._position_columns[1]])
        height, width = frame_shape

        return (
            first_position <= edge_margin or
            first_position >= height - 1 - edge_margin or
            second_position <= edge_margin or
            second_position >= width - 1 - edge_margin
        )

    def __calculate_future_feature_means(
        self,
        dataframe: pd.DataFrame,
        feature_columns: list[str],
        future_feature_averaging_window: int,
        max_step_distance: float
    ) -> dict[int, dict[str, float]]:
        """
        Build position-only future tracklets and cache their mean feature values.

        Successors are matched one-to-one only across consecutive frames. Missing
        frames are never bridged, so this lookahead cannot join an exited particle
        to a later particle merely because both are near the same boundary.
        """
        if not feature_columns or future_feature_averaging_window <= 1:
            return {}

        successor_indices: dict[int, int] = {}
        frame_values = sorted(dataframe['frame'].unique())

        for current_frame, next_frame in zip(frame_values[:-1], frame_values[1:]):
            if next_frame - current_frame != 1:
                continue

            current_frame_dataframe = dataframe[dataframe['frame'] == current_frame]
            next_frame_dataframe = dataframe[dataframe['frame'] == next_frame]
            if current_frame_dataframe.empty or next_frame_dataframe.empty:
                continue

            current_positions = current_frame_dataframe[
                self._position_columns
            ].to_numpy(dtype=float)
            next_positions = next_frame_dataframe[
                self._position_columns
            ].to_numpy(dtype=float)
            distance_matrix = np.linalg.norm(
                current_positions[:, np.newaxis, :] - next_positions[np.newaxis, :, :],
                axis=2
            )
            allowed_links = np.isfinite(distance_matrix) & (
                distance_matrix <= max_step_distance
            )
            if not allowed_links.any():
                continue

            assignment_costs = np.where(allowed_links, distance_matrix, 1e12)
            row_indices, column_indices = linear_sum_assignment(assignment_costs)
            current_indices = current_frame_dataframe.index.to_numpy()
            next_indices = next_frame_dataframe.index.to_numpy()

            for row_index, column_index in zip(row_indices, column_indices):
                if not allowed_links[row_index, column_index]:
                    continue
                successor_indices[int(current_indices[row_index])] = int(
                    next_indices[column_index]
                )

        feature_arrays = {
            column: pd.to_numeric(dataframe[column], errors='coerce').to_numpy(dtype=float)
            for column in feature_columns
        }
        future_feature_means: dict[int, dict[str, float]] = {}
        for start_index in dataframe.index:
            tracklet_indices = [int(start_index)]
            while len(tracklet_indices) < future_feature_averaging_window:
                next_index = successor_indices.get(tracklet_indices[-1])
                if next_index is None:
                    break
                tracklet_indices.append(next_index)

            feature_means: dict[str, float] = {}
            for column in feature_columns:
                feature_values = feature_arrays[column][tracklet_indices]
                feature_values = feature_values[np.isfinite(feature_values)]
                if len(feature_values) > 0:
                    feature_means[column] = float(np.mean(feature_values))
            future_feature_means[int(start_index)] = feature_means

        return future_feature_means

    def __calculate_linking_cost(
        self,
        track_rows: list[pd.Series],
        detection: pd.Series,
        max_distance: float,
        max_step_distance: float,
        feature_columns: list[str],
        feature_weights: dict[str, float],
        feature_averaging_window: int,
        future_feature_values: dict[str, float] | None,
        momentum_weight: float,
        direction_weight: float,
        momentum_window: int
    ) -> float:
        """
        Calculate weighted cost between an existing track and a candidate detection.
        """
        last_row = track_rows[-1]
        predicted_position = self.__predict_next_position(track_rows, momentum_window)
        detection_position = np.array([
            detection[self._position_columns[0]],
            detection[self._position_columns[1]]
        ], dtype=float)

        last_position = np.array([
            last_row[self._position_columns[0]],
            last_row[self._position_columns[1]]
        ], dtype=float)

        step_distance = np.linalg.norm(detection_position - last_position)
        if step_distance > max_step_distance:
            return np.inf

        position_distance = np.linalg.norm(detection_position - predicted_position)
        if position_distance > max_distance:
            return np.inf

        position_cost = position_distance / max(max_distance, 1e-9)
        feature_cost = self.__calculate_feature_cost(
            track_rows=track_rows,
            detection=detection,
            feature_columns=feature_columns,
            feature_weights=feature_weights,
            feature_averaging_window=feature_averaging_window,
            future_feature_values=future_feature_values
        )
        direction_cost = self.__calculate_direction_cost(
            track_rows=track_rows,
            detection_position=detection_position
        )

        return (
            momentum_weight * position_cost +
            feature_cost +
            direction_weight * direction_cost
        )

    def __predict_next_position(self, track_rows: list[pd.Series], momentum_window: int) -> np.ndarray:
        """
        Predict the next particle position from recent velocity.
        """
        last_position = np.array([
            track_rows[-1][self._position_columns[0]],
            track_rows[-1][self._position_columns[1]]
        ], dtype=float)

        if len(track_rows) < 2 or momentum_window <= 0:
            return last_position

        recent_rows = track_rows[-(momentum_window + 1):]
        recent_positions = np.array([
            [row[self._position_columns[0]], row[self._position_columns[1]]]
            for row in recent_rows
        ], dtype=float)
        recent_displacements = np.diff(recent_positions, axis=0)

        if len(recent_displacements) == 0:
            return last_position

        mean_velocity = recent_displacements.mean(axis=0)
        return last_position + mean_velocity

    def __calculate_feature_cost(
        self,
        track_rows: list[pd.Series],
        detection: pd.Series,
        feature_columns: list[str],
        feature_weights: dict[str, float],
        feature_averaging_window: int,
        future_feature_values: dict[str, float] | None
    ) -> float:
        """
        Calculate morphology mismatch cost using recent mean track features.
        """
        if not feature_columns:
            return 0.0

        recent_rows = track_rows[-feature_averaging_window:]
        total_weight = 0.0
        weighted_cost = 0.0
        for column in feature_columns:
            mean_track_value = float(np.mean([
                float(row[column]) for row in recent_rows
            ]))
            if future_feature_values is None:
                detection_value = float(detection[column])
            else:
                detection_value = float(
                    future_feature_values.get(column, detection[column])
                )
            scale = max(abs(mean_track_value), abs(detection_value), 1.0)
            weight = float(feature_weights.get(column, 1.0))
            weighted_cost += weight * abs(detection_value - mean_track_value) / scale
            total_weight += weight

        if total_weight == 0:
            return 0.0
        return weighted_cost / total_weight

    def __calculate_direction_cost(
        self,
        track_rows: list[pd.Series],
        detection_position: np.ndarray
    ) -> float:
        """
        Calculate direction-change cost between recent particle movement and candidate movement.
        """
        if len(track_rows) < 2:
            return 0.0

        previous_position = np.array([
            track_rows[-2][self._position_columns[0]],
            track_rows[-2][self._position_columns[1]]
        ], dtype=float)
        last_position = np.array([
            track_rows[-1][self._position_columns[0]],
            track_rows[-1][self._position_columns[1]]
        ], dtype=float)

        previous_vector = last_position - previous_position
        candidate_vector = detection_position - last_position

        previous_norm = np.linalg.norm(previous_vector)
        candidate_norm = np.linalg.norm(candidate_vector)
        if previous_norm == 0 or candidate_norm == 0:
            return 0.0

        cosine_similarity = np.dot(previous_vector, candidate_vector) / (previous_norm * candidate_norm)
        cosine_similarity = np.clip(cosine_similarity, -1.0, 1.0)

        # Cost is 0 for same direction and 1 for exact opposite direction.
        return (1.0 - cosine_similarity) / 2.0

    def __shape_and_sort_dataframe(self, dataframe: pd.DataFrame, cols: list[str], sort_by: list[str]) -> pd.DataFrame:
        """
        Shape and sort the dataframe.
        Args:
            dataframe (pd.DataFrame): The dataframe to shape and sort.
            cols (List[str]): List of column names to set for the dataframe.
            sort_by (List[str]): List of column names to sort the dataframe by.
            Returns:
            pd.DataFrame: The shaped and sorted dataframe.
        """
        temp_dataframe = pd.DataFrame(data=dataframe, columns=cols)
        temp_dataframe.columns = cols
        temp_dataframe.index.name = None
        return temp_dataframe.sort_values(by=sort_by, ascending=True)

    def _generate_particle_colors(self, tracking_data: pd.DataFrame, colormap_name: str) -> dict:
        """
        Generates unique colors for each particle using the specified colormap.

        Args:
            tracking_data (pd.DataFrame): DataFrame containing tracking data.
            colormap_name (str): Name of the matplotlib colormap to use.

        Returns:
            dict: A dictionary mapping each particle ID to its assigned color in BGR format.
        """

        unique_particles = tracking_data['particle'].unique()
        num_particles = len(unique_particles)
        if num_particles == 0:
            return {}

        cmap = cm.get_cmap(colormap_name, num_particles)
        particle_colors = {particle: cmap(
            i)[:3] for i, particle in enumerate(unique_particles)}

        # Convert colors from 0-1 range to 0-255 range and from RGB to BGR for OpenCV
        particle_colors_bgr = {
            p: (int(c[2] * 255), int(c[1] * 255),
                int(c[0] * 255))  # Convert RGB to BGR
            for p, c in particle_colors.items()
        }

        return particle_colors_bgr

    def _initialize_video_writer(
        self,
        frame: np.ndarray,
        fps: float,
        output_video_path: str,
        video_codec: str | None = None
    ) -> cv2.VideoWriter:
        """
        Initializes the OpenCV VideoWriter object.

        Args:
            frame (np.ndarray): A single frame from the video to determine frame size.
            fps (float): Frames per second for the output video.
            output_video_path (str): Full path to save the output video.
            video_codec (str | None): Optional four-character codec. None chooses a
                compatible default from the output extension.

        Returns:
            cv2.VideoWriter: Initialized VideoWriter object.
        """
        frame_height, frame_width = frame.shape[:2]
        if video_codec is None:
            output_extension = os.path.splitext(output_video_path)[1].lower()
            video_codec = 'MJPG' if output_extension == '.avi' else 'mp4v'
        if not isinstance(video_codec, str) or len(video_codec) != 4:
            raise ValueError('video_codec must be a four-character codec string or None.')

        fourcc = cv2.VideoWriter_fourcc(*video_codec)
        video_writer = cv2.VideoWriter(
            output_video_path, fourcc, fps, (frame_width, frame_height))
        if not video_writer.isOpened():
            video_writer.release()
            raise ValueError(
                f"Unable to initialize video writer with codec '{video_codec}' for "
                f"output path: {output_video_path}"
            )
        return video_writer

    def _update_particle_tracks(self, frame_data: pd.DataFrame, particle_tracks: dict) -> None:
        """
        Updates the particle tracks dictionary with new positions from the current frame.

        Args:
            frame_data (pd.DataFrame): DataFrame containing particle data for the current frame.
            particle_tracks (dict): Dictionary storing cumulative tracks for each particle.
        """
        for _, row in frame_data.iterrows():
            particle_id = row['particle']
            # Swap x and y coordinates for correct alignment (assuming 'centroid_y' is x and 'centroid_x' is y)
            particle_position = (
                int(row['centroid_y']), int(row['centroid_x']))
            particle_tracks[particle_id].append(particle_position)

    def _draw_particle_tracks(
        self,
        frame: np.ndarray,
        particle_tracks: dict,
        particle_colors: dict,
        line_thickness: int = 2
    ) -> None:
        """
        Draws the particle tracks on the given frame.

        Args:
            frame (np.ndarray): The video frame to draw on.
            particle_tracks (dict): Dictionary storing cumulative tracks for each particle.
            particle_colors (dict): Dictionary mapping each particle ID to its color.
            line_thickness (int): Width of trajectory lines in output pixels.
        """
        for particle_id, track_points in particle_tracks.items():
            # Default to white if not found
            track_color = particle_colors.get(particle_id, (255, 255, 255))
            for i in range(1, len(track_points)):
                cv2.line(
                    frame, track_points[i - 1], track_points[i], track_color,
                    thickness=line_thickness)

    def _prepare_frame_for_overlay(
        self,
        frame: np.ndarray,
        output_frame_size: tuple[int, int] | None = None
    ) -> np.ndarray:
        """
        Prepares a captured frame for OpenCV video overlay/export.

        The tracker should draw overlays on the original captured frames rather than
        Identify._working_masks, but captured microscopy frames may be 16-bit grayscale.
        OpenCV video writing and colored overlays are most reliable on 8-bit BGR frames.

        Args:
            frame (np.ndarray): Raw captured frame from the Capture object.
            output_frame_size (tuple[int, int] | None): Optional output width and
                height. Resizing before normalization speeds up preview exports.

        Returns:
            np.ndarray: 8-bit BGR copy suitable for drawing and video writing.
        """
        frame_array = np.asarray(frame)
        if output_frame_size is not None:
            if (
                not isinstance(output_frame_size, tuple) or
                len(output_frame_size) != 2 or
                any(
                    isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1
                    for value in output_frame_size
                )
            ):
                raise ValueError(
                    'output_frame_size must be a (width, height) tuple of positive integers.'
                )

            output_width, output_height = output_frame_size
            if frame_array.shape[1] != output_width or frame_array.shape[0] != output_height:
                frame_array = cv2.resize(
                    frame_array,
                    (int(output_width), int(output_height)),
                    interpolation=cv2.INTER_AREA
                )

        if frame_array.ndim == 2:
            display_frame = frame_array.astype(np.float32)
            frame_min = float(np.min(display_frame))
            frame_max = float(np.max(display_frame))
            frame_range = frame_max - frame_min

            if frame_range > 0:
                display_frame = ((display_frame - frame_min) / frame_range * 255).astype(np.uint8)
            else:
                display_frame = np.zeros_like(display_frame, dtype=np.uint8)

            return cv2.cvtColor(display_frame, cv2.COLOR_GRAY2BGR)

        if frame_array.ndim == 3:
            if frame_array.dtype != np.uint8:
                display_frame = frame_array.astype(np.float32)
                frame_min = float(np.min(display_frame))
                frame_max = float(np.max(display_frame))
                frame_range = frame_max - frame_min

                if frame_range > 0:
                    display_frame = ((display_frame - frame_min) / frame_range * 255).astype(np.uint8)
                else:
                    display_frame = np.zeros_like(frame_array, dtype=np.uint8)
            else:
                display_frame = frame_array.copy()

            if display_frame.shape[-1] == 1:
                return cv2.cvtColor(display_frame[:, :, 0], cv2.COLOR_GRAY2BGR)
            if display_frame.shape[-1] == 3:
                return display_frame.copy()
            if display_frame.shape[-1] == 4:
                return cv2.cvtColor(display_frame, cv2.COLOR_BGRA2BGR)

        raise ValueError(f'Unsupported frame shape for overlay export: {frame_array.shape}')

    def __limit_particle_legend_entries(self, ax, max_first_items: int = 10, max_last_items: int = 10, title: str = 'Particle ID') -> None:
        """
        Limit large particle legends to the first and last particle labels.
        """
        handles, labels = ax.get_legend_handles_labels()
        if not handles or not labels:
            return

        # Remove seaborn/matplotlib legend-title pseudo entries when present.
        legend_entries = [
            (handle, label)
            for handle, label in zip(handles, labels)
            if label != title and not label.startswith('_')
        ]

        if len(legend_entries) <= max_first_items + max_last_items:
            selected_entries = legend_entries
        else:
            first_entries = legend_entries[:max_first_items]
            last_entries = legend_entries[-max_last_items:]
            spacer_handle = plt.Line2D([], [], linestyle='none', marker=None)
            selected_entries = first_entries + [(spacer_handle, '...')] + last_entries

        selected_handles = [entry[0] for entry in selected_entries]
        selected_labels = [entry[1] for entry in selected_entries]
        n_cols = 1
        ax.legend(
            selected_handles,
            selected_labels,
            title=f'{title} (first/last 10)' if len(legend_entries) > max_first_items + max_last_items else title,
            bbox_to_anchor=(1.05, 1),
            loc='upper left',
            borderaxespad=0.,
            ncols=n_cols
        )

    def __get_tracking_dataframe_by_name(self, source_dataframe: str) -> pd.DataFrame:
        """
        Return a single linked-particle dataframe by source name.
        """
        valid_options = ("active", "full", "filtered", "updated")

        if source_dataframe not in valid_options:
            raise ValueError(
                f"source_dataframe must be one of {valid_options}. Received: {source_dataframe}"
            )

        if source_dataframe == "active":
            return self.get_linked_particles_dataframe(prefer_updated=True, prefer_filtered=True)

        if source_dataframe == "full":
            return self._linked_particles_dataframes

        if source_dataframe == "filtered":
            return self._filtered_particles_dataframes

        return self._updated_particles_dataframes

    def __get_tracking_dataframes_for_plot(self, dataframe_to_plot: str | list[str] = 'active') -> list[tuple[str, pd.DataFrame]]:
        """
        Select one or more linked-particle dataframes for plotting.
        """
        valid_options = ('full', 'filtered', 'updated')

        if dataframe_to_plot == 'active':
            if not self._updated_particles_dataframes.empty:
                return [('updated', self._updated_particles_dataframes)]
            if not self._filtered_particles_dataframes.empty:
                return [('filtered', self._filtered_particles_dataframes)]
            return [('full', self._linked_particles_dataframes)]

        if dataframe_to_plot == 'both':
            return [
                ('full', self._linked_particles_dataframes),
                ('filtered', self._filtered_particles_dataframes),
                ('updated', self._updated_particles_dataframes)
            ]

        requested_dataframes = (
            [dataframe_to_plot]
            if isinstance(dataframe_to_plot, str)
            else dataframe_to_plot
        )

        if not isinstance(requested_dataframes, list) or not requested_dataframes:
            raise ValueError(
                f'dataframe_to_plot must be one or more of {valid_options}. '
                f'Received: {dataframe_to_plot}'
            )

        invalid_options = [
            option for option in requested_dataframes
            if option not in valid_options
        ]
        if invalid_options:
            raise ValueError(
                f'dataframe_to_plot must contain only {valid_options}. '
                f'Received invalid values: {invalid_options}'
            )

        if len(set(requested_dataframes)) != len(requested_dataframes):
            raise ValueError('dataframe_to_plot cannot contain duplicate values.')

        dataframe_map = {
            'full': self._linked_particles_dataframes,
            'filtered': self._filtered_particles_dataframes,
            'updated': self._updated_particles_dataframes
        }
        return [
            (dataframe_name, dataframe_map[dataframe_name])
            for dataframe_name in requested_dataframes
        ]
