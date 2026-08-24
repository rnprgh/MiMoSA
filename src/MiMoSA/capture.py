"""
The capture module provides the Capture class to retrieve file paths, store images
from a video, and perform related operations.
"""

import math
import os
import re
import cv2
from tqdm import tqdm, trange
import tifffile


class Capture:
    """
    The Capture class provides methods to retrieve file paths, store images from a video,
    and perform related operations.
    """

    DEFAULT_FILE_DIRECTORY = 'input_files'
    DEFAULT_STORE_IMAGE_FILE_DIRECTORY = 'frames_from_video'
    SUPPORTED_INPUT_VIDEO_FILE_TYPES = ['avi', 'mp4', 'mpg', 'mpeg']
    DEFAULT_PIXEL_SCALE_FACTOR = 1
    DEFAULT_SCALE_UNITS = 'units'
    DEFAULT_CAPTURE_SPEED_IN_FPS = 15

    def __init__(self, working_directory=DEFAULT_FILE_DIRECTORY):
        """
        Initializes a new instance of the Capture class.

        Args:
          working_directory (str): The directory where the input files are located.
          Defaults to DEFAULT_FILE_DIRECTORY.
        """
        self._directory = self.__handle_working_directory_preprocess(working_directory)
        self._supported_video_file_types: list[str] = Capture.SUPPORTED_INPUT_VIDEO_FILE_TYPES
        self._video_file_path: str = ''
        self._video_file_name: str = ''
        self._default_fps: int = 0
        self._actual_fps: int = 0
        self._pixel_scale_factor: float = 0.0
        self._scale_units: str = ''
        self._video_frames_store_path: str = ''
        self._all_captured_frames: list = []
        self._working_captured_frames: list = []
        self._working_frame_range: dict = {}

    def load_video(self, file_name=''):
        """
        Retrieves and validates the file path based on user input and prints the video information.

        Args:
          file_name (str, optional): The name of the video file. Defaults to ''.

        Returns:
          str: The validated file path.

        Raises:
          FileNotFoundError: If the specified file is not found.
        """
        try:
            self._video_file_name = file_name

            self.__validate_the_file_name_and_type(file_name)
            file_path = self.__generate_and_validate_file_path(file_name)
            self._video_file_path = file_path

            video = cv2.VideoCapture(self._video_file_path)
            try:
                default_video_fps = self.__print_video_info_and_get_fps(video)
                self._default_fps = default_video_fps
            finally:
                video.release()

            print(f'Video file loaded successfully: {file_path}')
            return file_path

        except FileNotFoundError as e:
            print(e)

    def process_video_into_frames(self, pixel_scale_factor: float = DEFAULT_PIXEL_SCALE_FACTOR,
                                  scale_units: str = DEFAULT_SCALE_UNITS,
                                  capture_speed_in_fps=None,
                                  is_store_video_frames=False,
                                  store_images_path=DEFAULT_STORE_IMAGE_FILE_DIRECTORY) -> list:
        """
        Processes the already loaded video into frames based on the given pixel to micrometer conversion factor and capture speed.
        The pixel_scale_factor is mandatory to process the video into frames else will give an error.

        Args:
          pixel_scale_factor (float, optional): The pixel scale factor. Defaults to DEFAULT_PIXEL_SCALE_FACTOR.
          scale_units (str, optional): The scale units. Defaults to DEFAULT_SCALE_UNITS.
          capture_speed_in_fps (int, optional): Capture speed in frames/sec. Defaults to given video FPS or 15. Note: If the user provides the capture speed in FPS, The time of the video might be different from the actual video time, although the total frames will be constant.
          is_store_video_frames (bool, optional): Flag to store video frames. Defaults to False.
          store_images_path (str, optional): Path to store images. Existing folders
            and unrelated files are preserved; same-named frame files are overwritten.
            Defaults to DEFAULT_STORE_IMAGE_FILE_DIRECTORY.

        Returns:
          list: The captured frames.

        Raises:
          ValueError: If the pixel to micrometer conversion factor is not provided.
        """
        try:
            if not pixel_scale_factor:
                raise ValueError('Error: A valid Pixel Scale factor is mandatory to process the video and calculate the stats')

            # If the user provides the capture speed in FPS, then use that value else use the video FPS
            if capture_speed_in_fps:
                self._actual_fps = capture_speed_in_fps
                print(f'User provided a FPS: {self._actual_fps}, So processing the video with the given FPS')
            else:
                self._actual_fps = self._default_fps

            captured_frames = []
            self._pixel_scale_factor = pixel_scale_factor
            self._scale_units = scale_units

            captured_frames = self.__capture_images_from_video(is_store_video_frames,
                                                               store_images_path)
            self.__set_captured_frames(captured_frames)
            print(
                f'Processed video into frames successfully with pixel scale factor: {self._pixel_scale_factor} {self._scale_units}'
            )
            return captured_frames
        except ValueError as e:
            print(e)

    def get_captured_working_frames(self):
        """
        Retrieves the captured frames currently selected for downstream work.

        Returns:
          list: The working captured frames.
        """
        return self._working_captured_frames

    def get_all_captured_frames(self):
        """
        Retrieves the complete, unmodified list of imported captured frames.

        Returns:
          list: All captured frames imported from the source.
        """
        return self._all_captured_frames

    def select_working_captured_frames(
        self,
        start_frame: int = 0,
        end_frame: int | None = None,
        frame_indexing: str = 'zero_based',
        end_inclusive: bool = True,
    ) -> list:
        """
        Selects a range of imported frames for downstream processing.

        The complete imported frame list remains unchanged in
        ``_all_captured_frames``. ``get_captured_working_frames()`` returns the selected
        range after this method is called.

        Args:
          start_frame (int): First frame in the requested range.
          end_frame (int | None): Last frame in the requested range. If None,
            frames are selected through the end of the imported sequence.
          frame_indexing (str): Either 'zero_based' or 'one_based'.
          end_inclusive (bool): If True, include end_frame in the selection.

        Returns:
          list: The selected working captured frames.

        Raises:
          ValueError: If no frames have been imported or the range is invalid.
          TypeError: If the frame bounds are not integers.
        """
        if not self._all_captured_frames:
            raise ValueError(
                'No captured frames are available. Load or create frames before selecting a working range.'
            )

        if isinstance(start_frame, bool) or not isinstance(start_frame, int):
            raise TypeError('start_frame must be an integer.')

        if end_frame is not None and (
            isinstance(end_frame, bool) or not isinstance(end_frame, int)
        ):
            raise TypeError('end_frame must be an integer or None.')

        if frame_indexing not in ('zero_based', 'one_based'):
            raise ValueError(
                "frame_indexing must be either 'zero_based' or 'one_based'."
            )

        index_offset = 0 if frame_indexing == 'zero_based' else 1
        start_index = start_frame - index_offset
        total_frame_count = len(self._all_captured_frames)

        if end_frame is None:
            end_index_exclusive = total_frame_count
        else:
            end_index = end_frame - index_offset
            end_index_exclusive = end_index + 1 if end_inclusive else end_index

        if start_index < 0 or start_index >= total_frame_count:
            raise ValueError(
                f'start_frame={start_frame} is outside the available frame range.'
            )

        if end_index_exclusive <= start_index:
            raise ValueError('The selected frame range must contain at least one frame.')

        if end_index_exclusive > total_frame_count:
            raise ValueError(
                f'end_frame={end_frame} is outside the available frame range.'
            )

        self._working_captured_frames = self._all_captured_frames[
            start_index:end_index_exclusive
        ]
        self._working_frame_range = {
            'start_index': start_index,
            'end_index': end_index_exclusive - 1,
            'start_frame': start_index + index_offset,
            'end_frame': end_index_exclusive - 1 + index_offset,
            'requested_start_frame': start_frame,
            'requested_end_frame': end_frame,
            'frame_indexing': frame_indexing,
            'end_inclusive': end_inclusive,
            'selected_frame_count': len(self._working_captured_frames),
            'total_frame_count': total_frame_count,
        }

        print(
            f'Selected {len(self._working_captured_frames)} working captured frames '
            f'from {start_frame} through '
            f'{self._working_frame_range["end_frame"]} ({frame_indexing}).'
        )
        return self._working_captured_frames

    def get_working_frame_range(self) -> dict:
        """
        Retrieves metadata describing the active working captured frame range.

        Returns:
          dict: A copy of the active frame-range metadata.
        """
        return self._working_frame_range.copy()

    def reset_working_captured_frames(self) -> list:
        """
        Restores all imported frames as the active working captured frames.

        Returns:
          list: All imported frames, restored as the working selection.
        """
        self._working_captured_frames = list(self._all_captured_frames)
        total_frame_count = len(self._all_captured_frames)

        if total_frame_count:
            self._working_frame_range = {
                'start_index': 0,
                'end_index': total_frame_count - 1,
                'start_frame': 0,
                'end_frame': total_frame_count - 1,
                'requested_start_frame': 0,
                'requested_end_frame': total_frame_count - 1,
                'frame_indexing': 'zero_based',
                'end_inclusive': True,
                'selected_frame_count': total_frame_count,
                'total_frame_count': total_frame_count,
            }
        else:
            self._working_frame_range = {}

        return self._working_captured_frames

    def get_directory(self):
        """
        Retrieves the working directory.
        Returns:
          str: The working directory.
        """
        return self._directory

    def get_frame_rate(self) -> dict:
        """
        Retrieves the frame rate.
        Returns:
          dict: The frame rates (user provided and default).
        """
        return { 'user_provided_fps': self._actual_fps, 'default_fps': self._default_fps }

    def get_pixel_scale_factor(self):
        """
        Retrieves the pixel to micrometer conversion factor.
        Returns:
          float: The pixel to micrometer conversion factor.
        """
        return self._pixel_scale_factor

    def get_scale_units(self) -> str:
        """
        Retrieves the spatial scale units.

        Returns:
          str: The units associated with the pixel scale factor.
        """
        return self._scale_units

    def load_capture_properties_from_file(
        self,
        file_path: str | os.PathLike,
        sheet_name: str | int = 0,
        is_update_properties: bool = True
    ) -> dict[str, float | int | str]:
        """
        Load frame rate and spatial-scale properties from an Excel key/value sheet.

        Recognized labels include FPS, Pixel Scale Factor, Scale Units, Frames,
        Resolution, Timestamp #First, and Timestamp #Last. Label matching is
        case-insensitive and ignores punctuation and whitespace. Relative paths are
        resolved from the Capture working directory.

        Args:
          file_path (str | os.PathLike): Path to an .xlsx or .xlsm workbook.
          sheet_name (str | int): Worksheet name or zero-based index. Defaults to 0.
          is_update_properties (bool): Whether to update this Capture object's
            frame rate and scale properties. Defaults to True.

        Returns:
          dict: Validated capture and source metadata keyed by
            capture_speed_in_fps, pixel_scale_factor, scale_units, frames,
            resolution, timestamp_first_frame, and timestamp_last_frame.

        Raises:
          FileNotFoundError: If the workbook does not exist.
          ValueError: If required properties are missing or invalid.
          TypeError: If arguments have invalid types.
        """
        if not isinstance(file_path, (str, os.PathLike)):
            raise TypeError('file_path must be a string or path-like object.')
        if isinstance(sheet_name, bool) or not isinstance(sheet_name, (str, int)):
            raise TypeError('sheet_name must be a worksheet name or integer index.')
        if not isinstance(is_update_properties, bool):
            raise TypeError('is_update_properties must be a boolean.')

        file_path_string = os.fspath(file_path)
        if not file_path_string:
            raise ValueError('file_path cannot be empty.')

        complete_file_path = os.path.normpath(
            file_path_string
            if os.path.isabs(file_path_string)
            else os.path.join(self._directory, file_path_string)
        )
        if not os.path.isfile(complete_file_path):
            raise FileNotFoundError(
                f'Capture properties workbook not found: {complete_file_path}'
            )

        workbook_extension = os.path.splitext(complete_file_path)[1].lower()
        if workbook_extension not in ('.xlsx', '.xlsm'):
            raise ValueError(
                'Capture properties file must be an .xlsx or .xlsm workbook.'
            )

        try:
            import pandas as pd
        except ImportError as error:
            raise ImportError(
                'pandas and an Excel reader are required to load capture properties.'
            ) from error

        details_table = pd.read_excel(
            complete_file_path,
            sheet_name=sheet_name,
            header=None
        )
        if details_table.empty or details_table.shape[1] < 2:
            raise ValueError(
                'Capture properties workbook must contain labels and values in '
                'separate columns.'
            )

        label_aliases = {
            'fps': 'capture_speed_in_fps',
            'capture_speed': 'capture_speed_in_fps',
            'capture_speed_in_fps': 'capture_speed_in_fps',
            'frame_rate': 'capture_speed_in_fps',
            'frames_per_second': 'capture_speed_in_fps',
            'frames': 'frames',
            'frame_count': 'frames',
            'total_frames': 'frames',
            'resolution': 'resolution',
            'image_resolution': 'resolution',
            'pixel_scale_factor': 'pixel_scale_factor',
            'pixel_size': 'pixel_scale_factor',
            'scale_units': 'scale_units',
            'scale_unit': 'scale_units',
            'units': 'scale_units',
            'timestamp_first': 'timestamp_first_frame',
            'timestamp_first_frame': 'timestamp_first_frame',
            'first_timestamp': 'timestamp_first_frame',
            'timestamp_last': 'timestamp_last_frame',
            'timestamp_last_frame': 'timestamp_last_frame',
            'last_timestamp': 'timestamp_last_frame'
        }
        extracted_properties = {}
        for _, row in details_table.iterrows():
            for column_index in range(len(row) - 1):
                label = row.iloc[column_index]
                if pd.isna(label):
                    continue

                normalized_label = re.sub(
                    r'[^a-z0-9]+',
                    '_',
                    str(label).strip().lower()
                ).strip('_')
                property_name = label_aliases.get(normalized_label)
                if property_name is None:
                    continue
                if property_name in extracted_properties:
                    raise ValueError(
                        f'Capture property {property_name!r} appears more than once '
                        f'in worksheet {sheet_name!r}.'
                    )

                following_values = row.iloc[column_index + 1:].dropna()
                if following_values.empty:
                    raise ValueError(
                        f'Capture property {property_name!r} does not have a value.'
                    )
                extracted_properties[property_name] = following_values.iloc[0]
                break

        required_properties = {
            'capture_speed_in_fps',
            'pixel_scale_factor',
            'scale_units',
            'frames',
            'resolution',
            'timestamp_first_frame',
            'timestamp_last_frame'
        }
        missing_properties = sorted(
            required_properties.difference(extracted_properties)
        )
        if missing_properties:
            raise ValueError(
                'Capture properties workbook is missing required values: '
                f'{missing_properties}'
            )

        try:
            capture_speed_in_fps = float(
                extracted_properties['capture_speed_in_fps']
            )
            pixel_scale_factor = float(
                extracted_properties['pixel_scale_factor']
            )
            frames_value = float(extracted_properties['frames'])
            timestamp_first_frame = float(
                extracted_properties['timestamp_first_frame']
            )
            timestamp_last_frame = float(
                extracted_properties['timestamp_last_frame']
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                'FPS, Pixel Scale Factor, Frames, and timestamps must contain '
                'numeric values.'
            ) from error

        if not math.isfinite(capture_speed_in_fps) or capture_speed_in_fps <= 0:
            raise ValueError('FPS must be a positive finite number.')
        if not math.isfinite(pixel_scale_factor) or pixel_scale_factor <= 0:
            raise ValueError('Pixel Scale Factor must be a positive finite number.')
        if (
            not math.isfinite(frames_value) or
            frames_value <= 0 or
            not frames_value.is_integer()
        ):
            raise ValueError('Frames must be a positive integer.')
        if not math.isfinite(timestamp_first_frame):
            raise ValueError('Timestamp #First must be a finite number.')
        if not math.isfinite(timestamp_last_frame):
            raise ValueError('Timestamp #Last must be a finite number.')
        if timestamp_last_frame < timestamp_first_frame:
            raise ValueError(
                'Timestamp #Last cannot be earlier than Timestamp #First.'
            )
        frames = int(frames_value)

        scale_units_value = extracted_properties['scale_units']
        if pd.isna(scale_units_value) or not str(scale_units_value).strip():
            raise ValueError('Scale Units must contain a non-empty value.')
        scale_units = str(scale_units_value).strip()

        resolution_value = extracted_properties['resolution']
        if pd.isna(resolution_value) or not str(resolution_value).strip():
            raise ValueError('Resolution must contain a non-empty value.')
        resolution = str(resolution_value).strip()

        capture_properties = {
            'capture_speed_in_fps': capture_speed_in_fps,
            'pixel_scale_factor': pixel_scale_factor,
            'scale_units': scale_units,
            'frames': frames,
            'resolution': resolution,
            'timestamp_first_frame': timestamp_first_frame,
            'timestamp_last_frame': timestamp_last_frame
        }

        if is_update_properties:
            self.set_properties(
                capture_speed_in_fps=capture_speed_in_fps,
                pixel_scale_factor=pixel_scale_factor,
                scale_units=scale_units
            )
            self._default_fps = capture_speed_in_fps

        print(
            f'Loaded capture properties from {complete_file_path}: '
            f'{frames} frames at {resolution}, {capture_speed_in_fps} FPS, '
            f'pixel scale factor {pixel_scale_factor} {scale_units}, timestamps '
            f'{timestamp_first_frame}-{timestamp_last_frame}.'
        )
        return capture_properties

    def load_images_as_frames(self, folder_path, capture_speed_in_fps=DEFAULT_CAPTURE_SPEED_IN_FPS, pixel_scale_factor=DEFAULT_PIXEL_SCALE_FACTOR, scale_units=DEFAULT_SCALE_UNITS):
        """
        Loads all images from the given folder as frames in alphabetical order of the filenames.

        Args:
          folder_path (str): The path of the folder containing the images.
          capture_speed_in_fps (int, optional): The capture speed in frames per second. Defaults to DEFAULT_CAPTURE_SPEED_IN_FPS.
          pixel_scale_factor (float, optional): The pixel scale factor. Defaults to DEFAULT_PIXEL_SCALE_FACTOR.
          scale_units (str, optional): The scale units. Defaults to DEFAULT_SCALE_UNITS.

        Raises:
          FileNotFoundError: If the specified folder is not found.
        """
        self._default_fps = capture_speed_in_fps
        self._actual_fps = capture_speed_in_fps
        self._pixel_scale_factor = pixel_scale_factor
        self._scale_units = scale_units

        complete_folder_path = os.path.join(self._directory, folder_path)
        if not os.path.isdir(complete_folder_path):
            raise FileNotFoundError(f'Folder not found: {complete_folder_path}')

        image_files = self.__list_files(complete_folder_path)
        frames = []
        for index in trange(len(image_files), desc='Loading frames'):
            image_path = os.path.join(complete_folder_path, image_files[index])
            frame = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if frame is not None:
                frames.append(frame)

        self.__set_captured_frames(frames)
        print(f'{len(frames)} frames loaded from folder: {complete_folder_path}')
        return frames

    def set_properties(self, pixel_scale_factor: float = DEFAULT_PIXEL_SCALE_FACTOR, scale_units: str = DEFAULT_SCALE_UNITS, capture_speed_in_fps=None):
        """
        Sets the properties of the Capture object.

        Args:
          pixel_scale_factor (float, optional): The pixel scale factor. Defaults to 1.
          scale_units (str, optional): The scale units. Defaults to 'units'.
          capture_speed_in_fps (int, optional): Capture speed in frames/sec. Defaults to 15.
        """
        self._pixel_scale_factor = pixel_scale_factor
        self._scale_units = scale_units
        self._actual_fps = capture_speed_in_fps

    def load_tiff_images_as_frames(
        self,
        file_name="",
        capture_speed_in_fps=DEFAULT_CAPTURE_SPEED_IN_FPS,
        pixel_scale_factor=DEFAULT_PIXEL_SCALE_FACTOR,
        scale_units=DEFAULT_SCALE_UNITS,
        is_store_video_frames=True,
        store_images_path=DEFAULT_STORE_IMAGE_FILE_DIRECTORY,
    ):
        """
        Loads TIFF images as frames from the specified file.
        file_name: The name of the TIFF file.
        capture_speed_in_fps: The capture speed in frames per second.
        pixel_scale_factor: The pixel scale factor.
        scale_units: The scale units.
        is_store_video_frames: Flag to store video frames.
        store_images_path: Path to store images.
        """
        self._default_fps = capture_speed_in_fps
        self._actual_fps = capture_speed_in_fps
        self._pixel_scale_factor = pixel_scale_factor
        self._scale_units = scale_units

        file_path = os.path.join(self._directory, file_name)
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        if is_store_video_frames:
            self._video_frames_store_path = self.__handle_folder_preprocess(
                store_images_path
            )

        with tifffile.TiffFile(file_path) as tiff:
            frames = [page.asarray() for page in tiff.pages]

            if is_store_video_frames:
                for index in trange(len(frames), desc="Saving frames"):
                    frame_number = str(index).zfill(len(str(len(frames))))
                    tifffile.imwrite(
                        os.path.join(
                            self._video_frames_store_path, f"frame_{frame_number}.tiff"
                        ),
                        frames[index],
                    )

        self.__set_captured_frames(frames)
        print(f"{len(frames)} frames loaded from TIFF file: {file_path}")
        return frames

    # Private Methods
    def __set_captured_frames(self, frames: list) -> None:
        """
        Stores a newly imported frame sequence and resets the working selection.

        Args:
          frames (list): The complete frame sequence imported from the source.
        """
        self._all_captured_frames = list(frames)
        self.reset_working_captured_frames()

    def __capture_images_from_video(self,
                                    is_store_video_frames=False,
                                    store_images_path=DEFAULT_STORE_IMAGE_FILE_DIRECTORY):
        """
        Captures images from a video file.

        Args:
          is_store_video_frames (bool, optional): Flag to store video frames. Defaults to False.
          store_images_path (str, optional): Path to store images. Defaults to DEFAULT_STORE_IMAGE_FILE_DIRECTORY.

        Returns:
          list: The captured frames.
        """
        if is_store_video_frames:
            complete_store_path = self.__handle_folder_preprocess(store_images_path)
            self._video_frames_store_path = complete_store_path

        captured_frames = []
        video = cv2.VideoCapture(self._video_file_path)
        try:
            captured_frames = self.__capture_and_store_frames(video, is_store_video_frames)
        finally:
            video.release()

        print(f'{len(captured_frames)} frame(s) captured successfully '
              f'for the video FPS: {self._actual_fps} to the '
              f'folder: {self._video_frames_store_path}'
              )
        return captured_frames

    def __validate_the_file_name_and_type(self, file_name=''):
        """
        Validates the file name and type.

        Args:
          file_name (str, optional): The name of the file. Defaults to ''.

        Raises:
          ValueError: If the file name is empty or the file type is not supported.
        """

        self.__validate_empty_file_name(file_name)
        self.__validate_file_type(file_name)

    def __validate_empty_file_name(self, filename=''):
        """
        Checks if the file name is empty.

        Args:
          filename (str): The file name to check.

        Raises:
          ValueError: If the file name is empty.
        """
        if not filename:
            raise ValueError('Empty file name. Please provide a valid filename')

    def __validate_file_type(self, filename=''):
        """
        Checks if the file type is supported.

        Args:
          filename (str): The file name to check.

        Raises:
          ValueError: If the file type is not supported.
        """
        file_type = self.__get_file_type(filename)
        self.__is_supported_file_type(file_type)

    def __get_file_type(self, filename=''):
        """
        Retrieves the file type from the given file name.

        Args:
          filename (str): The file name.

        Returns:
          str: The file type.
        """
        return filename.split('.')[-1]

    def __is_supported_file_type(self, file_type=''):
        """
        Checks if the file type is supported.

        Args:
          file_type (str): The file type to check.

        Raises:
          ValueError: If the file type is not supported.
        """
        if file_type not in self._supported_video_file_types:
            raise ValueError(
                'Invalid file type. Please provide a valid file type')

    def __generate_and_validate_file_path(self, file_name=''):
        """
        Generates and validates the file path based on the given file name.

        Args:
          file_name (str): The file name.

        Returns:
          str: The validated file path.

        Raises:
          FileNotFoundError: If the file path does not exist.
        """
        file_path = self.__generate_file_path(file_name)
        self.__check_file_path(file_path)
        return file_path

    def __generate_file_path(self, file_name=''):
        """
        Generates the file path based on the given file name.

        Args:
          file_name (str): The file name.

        Returns:
          str: The generated file path.
        """
        return os.path.join(self._directory, file_name)

    def __check_file_path(self, file_path=''):
        """
        Checks if the file path exists.

        Args:
          file_path (str): The file path to check.

        Raises:
          FileNotFoundError: If the file path does not exist.
        """
        if not os.path.isfile(file_path):
            raise FileNotFoundError(
                'File does not exist. Please provide a valid filename')

    def __convert_fps_to_ms(self):
        """
        Converts frames per second (FPS) to milliseconds (ms).

        Returns:
          float: The converted value in milliseconds.
        """
        milliseconds_in_a_second = 1000
        round_off_decimals = 2
        return round(milliseconds_in_a_second / self._actual_fps, round_off_decimals)

    def __handle_working_directory_preprocess(self, working_directory):
        """
        Handles the preprocessing of the working directory.

        Args:
          working_directory (str): The working directory.

        Returns:
          str: The complete path of the working directory.
        """
        if not working_directory:
            user_working_dir = os.getcwd()
            working_directory = os.path.join(user_working_dir, Capture.DEFAULT_FILE_DIRECTORY)
        if not os.path.exists(working_directory):
            os.makedirs(working_directory)
        return working_directory

    def __handle_folder_preprocess(self, image_store_path):
        """
        Resolve the image store folder and create it when missing.

        Existing folders and their contents are preserved. Individual export files
        with the same names can still be overwritten by their respective writers.

        Args:
          image_store_path (str): The path of the image store folder.

        Returns:
          str: The complete path of the image store folder.
        """
        complete_path = os.path.join(self._directory, image_store_path)
        os.makedirs(complete_path, exist_ok=True)
        return complete_path

    def __print_video_info_and_get_fps(self, video):
        """
        Prints the information about the video and retrieves the frames per second (FPS).

        Args:
          video: The video object.
        """
        round_off_decimals = 2
        video_fps = video.get(cv2.CAP_PROP_FPS)
        print('---------- Video Stats ----------')
        print(f'Video Frame Width: {int(video.get(cv2.CAP_PROP_FRAME_WIDTH))}')
        print(
            f'Video Frame Height: {int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))}')
        print(f'Frame Rate: {video_fps} FPS')
        print(f'Total Frames: {video.get(cv2.CAP_PROP_FRAME_COUNT)} frames')
        print(f'Video Duration (s): {round(video.get(cv2.CAP_PROP_FRAME_COUNT) / video.get(cv2.CAP_PROP_FPS), round_off_decimals)}')
        print('---------------------------------')
        return video_fps

    def __get_total_frames(self, video):
        """
        Retrieves the total frames of the video.

        Args:
          video: The video object.

        Returns:
          int: The total frames of the video.
        """
        return int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    def __capture_and_store_frames(self, video, is_store_video_frames=False):
        """
        Captures and stores frames from the video.

        Args:
          video: The video object.
          is_store_video_frames (bool, optional): Flag to store video frames. Defaults to False.

        Returns:
          list: The captured frames.
        """
        frame_to_capture_in_ms = 0
        frame_counter = 0
        image_path_prefix = 'frame'
        captured_frames = []
        capture_speed_in_ms = self.__convert_fps_to_ms()

        total_frames = self.__get_total_frames(video)
        with tqdm(total=total_frames, desc='Frame capture progress') as progress_bar:

            while True:
                video.set(cv2.CAP_PROP_POS_MSEC, frame_to_capture_in_ms)
                has_frame, frame = video.read()

                if not has_frame:
                    break

                if is_store_video_frames:
                    frame_counter += 1
                    frame_number = str(frame_counter).zfill(len(str(total_frames)))
                    cv2.imwrite(f'{self._video_frames_store_path}/{image_path_prefix}_{frame_number}.tiff', frame)

                captured_frames.append(frame)
                progress_bar.update(1)
                frame_to_capture_in_ms += capture_speed_in_ms

        return captured_frames

    def __extract_number_from_file_name(self, filename=''):
        """
        Extracts the last number from the filename for natural frame sorting.
        Args:
          filename (str): The filename.
        Returns:
          int: The extracted number, or -1 if no number is present.
        """
        stem = os.path.splitext(filename)[0]
        numbers = re.findall(r'\d+', stem)
        if not numbers:
            return -1
        return int(numbers[-1])

    def __remove_folders_from_file_list(self, files=None, file_path=''):
        """
        Removes folders from the list of files.
        Args:
          files (list, optional): The list of files. Defaults to None.
          file_path (str): The file path.
        Returns:
          list: The filtered list of files.
        """
        if files is None:
            files = []
        return [f for f in files if os.path.isfile(os.path.join(file_path, f))]

    def __list_files(self, file_path=''):
        """
        Lists supported image files in the given folder.
        Args:
          file_path (str): The path of the folder.
        Returns:
          list: The sorted list of image files.
        """
        supported_image_extensions = ('.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp')
        files = os.listdir(file_path)
        filtered_files = self.__remove_folders_from_file_list(files, file_path)
        image_files = [
            f for f in filtered_files
            if f.lower().endswith(supported_image_extensions) and not f.startswith('.')
        ]
        sorted_files = sorted(image_files, key=self.__extract_number_from_file_name)
        return sorted_files
