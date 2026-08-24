"""
The identify module provides the Identify class to perform object identification
on frames using different methods.
"""
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cellpose_omni import core, models  # type: ignore
from cellpose_omni.models import MODEL_NAMES
from matplotlib.lines import Line2D  # type: ignore
from omnipose.utils import normalize99  # type: ignore
from scipy.optimize import curve_fit  # type: ignore
from skimage import measure
from skimage.color import rgb2gray
from skimage.filters import threshold_isodata  # pylint: disable=E0611
from skimage.filters import threshold_li  # pylint: disable=E0611
from skimage.filters import threshold_mean  # pylint: disable=E0611
from skimage.filters import threshold_minimum  # pylint: disable=E0611
from skimage.filters import threshold_otsu  # pylint: disable=E0611
from skimage.filters import threshold_triangle  # pylint: disable=E0611
from skimage.filters import threshold_yen  # pylint: disable=E0611
from skimage.filters import \
    try_all_threshold  # pylint: disable=E0611; pylint: disable=E0611
from tqdm import tqdm, trange

from .capture import Capture
from .constants import (OMNIPOSE_DEFAULT_PARAMS, AvailableOperations,
                        AvailableProps, PropsThreshold)


class Identify:
    """
    The Identify class provides methods to perform object identification on frames
    using nominal thresholding and omnipose models.
    """

    def __init__(self, capture_frame_object: Capture):
        """
        Initializes a new instance of the Identify class.
        Args:
            capture_frame_object (Capture | None): The Capture object to use for identification.
        Raises:
            TypeError: If the capture_frame_object is not an instance of Capture.
        """
        if not isinstance(capture_frame_object, Capture):
            raise TypeError(
                "capture_frame_object must be an instance of Capture")

        self._parent = capture_frame_object
        self._all_captured_frames: List = capture_frame_object.get_all_captured_frames()
        self._captured_working_frames: List = capture_frame_object.get_captured_working_frames()
        self._all_masks: List = []
        self._working_masks: List = []
        self._directory: str = capture_frame_object.get_directory()

        self._region_props_dataframe: pd.DataFrame = pd.DataFrame()
        self._filtered_region_props_dataframe: pd.DataFrame = pd.DataFrame()
        self._applied_filter_settings_dataframe: pd.DataFrame = pd.DataFrame(
            columns=['property', 'operation', 'value']
        )

        self._normalized_frames: List = []
        self._omnipose_model: models.CellposeModel | None = None
        self._omnipose_params: dict = {}
        self._mask_store_path: str = ''

    def show_working_frames(self, images_to_show_count: int = 5, images_per_row: int = 5, use_gray_cmap: bool = False, image_size: tuple = (5, 5)) -> None:
        """
        Displays the working captured frames.
        Args:
            images_to_show_count (int): The number of (equidistant) images to show. Default is 5.
            images_per_row (int): The number of images to show per row. Default is 5.
            use_gray_cmap (bool): Whether to use a grayscale colormap.
            image_size (tuple): The size of the image to display. Default is (5, 5).
        Returns:
            None
        """
        self._show_images(
            images=self._captured_working_frames,
            images_to_show_count=images_to_show_count,
            images_per_row=images_per_row,
            use_gray_cmap=use_gray_cmap,
            image_size=image_size,
            image_label='Image'
        )

    def show_all_frames(self, images_to_show_count: int = 5, images_per_row: int = 5, use_gray_cmap: bool = False, image_size: tuple = (5, 5)) -> None:
        """Displays images from the complete captured frame sequence."""
        self._show_images(
            images=self._all_captured_frames,
            images_to_show_count=images_to_show_count,
            images_per_row=images_per_row,
            use_gray_cmap=use_gray_cmap,
            image_size=image_size,
            image_label='Image'
        )

    def show_working_masks(self, images_to_show_count: int = 5, images_per_row: int = 5, use_gray_cmap: bool = True, image_size: tuple = (5, 5)) -> None:
        """
        Displays the working generated or imported masks.
        Args:
            images_to_show_count (int): The number of (equidistant) masks to show. Default is 5.
            images_per_row (int): The number of masks to show per row. Default is 5.
            use_gray_cmap (bool): Whether to use a grayscale colormap.
            image_size (tuple): The size of each mask to display. Default is (5, 5).
        Returns:
            None
        """
        if not self._working_masks:
            raise ValueError('No masks are available. Generate or import masks first.')

        self._show_images(
            images=self._working_masks,
            images_to_show_count=images_to_show_count,
            images_per_row=images_per_row,
            use_gray_cmap=use_gray_cmap,
            image_size=image_size,
            image_label='Mask'
        )

    def show_all_masks(self, images_to_show_count: int = 5, images_per_row: int = 5, use_gray_cmap: bool = True, image_size: tuple = (5, 5)) -> None:
        """Displays images from the complete generated or imported mask sequence."""
        if not self._all_masks:
            raise ValueError('No masks are available. Generate or import masks first.')

        self._show_images(
            images=self._all_masks,
            images_to_show_count=images_to_show_count,
            images_per_row=images_per_row,
            use_gray_cmap=use_gray_cmap,
            image_size=image_size,
            image_label='Mask'
        )

    def get_all_masks(self) -> List:
        """Returns all generated or imported masks."""
        return self._all_masks

    def get_working_masks(self) -> List:
        """Returns the masks currently used for downstream analysis."""
        return self._working_masks

    def _show_images(self, images: List, images_to_show_count: int, images_per_row: int, use_gray_cmap: bool, image_size: tuple, image_label: str) -> None:
        """Displays equidistant images from a frame or mask collection."""
        total_images = len(images)
        if images_to_show_count > total_images:
            raise ValueError(
                'The number of images to show cannot be greater than the total number of images.')

        if images_per_row <= 0:
            raise ValueError(
                'The number of images per row cannot be negative.')

        jump = total_images // images_to_show_count
        selected_images_indices = range(0, total_images, jump)[
            :images_to_show_count]

        # Determine the number of rows based on the user input
        n_cols = images_per_row
        n_rows = int(np.ceil(images_to_show_count / n_cols))

        # Calculate dynamic figure size
        fig_width = n_cols * image_size[0]
        fig_height = n_rows * image_size[1]

        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(fig_width, fig_height))
        axes = np.atleast_1d(axes).ravel()

        for idx, img_idx in enumerate(selected_images_indices):
            ax = axes[idx]
            image = images[img_idx]

            if use_gray_cmap:
                image_array = np.asarray(image)
                unique_values = np.unique(image_array)
                is_binary_image = (
                    image_array.dtype == bool or
                    np.array_equal(unique_values, np.array([0])) or
                    np.array_equal(unique_values, np.array([1])) or
                    np.array_equal(unique_values, np.array([False])) or
                    np.array_equal(unique_values, np.array([True])) or
                    np.array_equal(unique_values, np.array([0, 1])) or
                    np.array_equal(unique_values, np.array([False, True]))
                )

                if is_binary_image:
                    ax.imshow(image_array.astype(np.uint8), cmap='gray', vmin=0, vmax=1)
                else:
                    ax.imshow(image_array, cmap='gray')
            else:
                ax.imshow(image)
            ax.set_title(f'{image_label} {img_idx + 1}')
            ax.axis('off')  # Hide axes for better visualization

        # Hide any remaining subplots if there are fewer images than subplot slots
        for i in range(images_to_show_count, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()

    def _frame_to_grayscale_float(self, frame: np.ndarray) -> np.ndarray:
        """
        Converts a frame to a normalized grayscale float image in the range 0-1.
        Supports already-grayscale 2D frames and RGB/RGBA/BGR/BGRA 3D frames.
        Args:
            frame (np.ndarray): The input frame.
        Returns:
            np.ndarray: The normalized grayscale float image.
        """
        frame_array = np.asarray(frame)

        if frame_array.ndim == 2:
            gray_scale = frame_array.astype(np.float32)
        elif frame_array.ndim == 3:
            if frame_array.shape[-1] == 3:
                gray_scale = rgb2gray(frame_array)
            elif frame_array.shape[-1] == 4:
                gray_scale = rgb2gray(frame_array[..., :3])
            else:
                raise ValueError(f'Unsupported 3D frame shape for grayscale conversion: {frame_array.shape}')
        else:
            raise ValueError(f'Unsupported frame shape for grayscale conversion: {frame_array.shape}')

        gray_scale = gray_scale.astype(np.float32)

        if gray_scale.size == 0:
            return gray_scale

        gray_min = float(np.min(gray_scale))
        gray_max = float(np.max(gray_scale))

        if gray_max > 1.0 or gray_min < 0.0:
            gray_range = gray_max - gray_min
            if gray_range > 0:
                gray_scale = (gray_scale - gray_min) / gray_range
            else:
                gray_scale = np.zeros_like(gray_scale, dtype=np.float32)

        return gray_scale

    # Nominal Methods
    def apply_grayscale_thresholding(self, threshold: float = 0.5, object_polarity: str = 'dark_on_bright', is_update_masks: bool = True, save_masks: bool = False, masks_store_path: str = 'masks') -> List:
        """
        Applies manual grayscale thresholding to the captured frames.
        Args:
            threshold (float): The threshold value to use for thresholding, after grayscale normalization to 0-1.
            object_polarity (str): Whether objects are darker or brighter than the background.
                Use 'dark_on_bright' for dark objects on a bright background.
                Use 'bright_on_dark' for bright objects on a dark background.
            is_update_masks (bool): Whether to retain the generated masks on this object.
            save_masks (bool): Whether to save the thresholded masks.
            masks_store_path (str): The path to store the masks.
        Returns:
            List: The generated masks after applying grayscale thresholding.
        """
        generated_masks: List = []
        if threshold < 0 or threshold > 1:
            raise ValueError(
                'The threshold value should be between 0 and 1.')

        valid_object_polarities = ('dark_on_bright', 'bright_on_dark')
        if object_polarity not in valid_object_polarities:
            raise ValueError(
                f"object_polarity should be one of {valid_object_polarities}.")

        for frame_index in trange(len(self._captured_working_frames), desc='Applying grayscale thresholding'):
            gray_scale = self._frame_to_grayscale_float(
                self._captured_working_frames[frame_index])

            if object_polarity == 'dark_on_bright':
                binary_image = gray_scale < threshold
            else:
                binary_image = gray_scale > threshold

            generated_masks.append(binary_image)

        if save_masks:
            self.save_masks(
                masks=generated_masks,
                masks_store_path=masks_store_path,
                file_prefix='manual_grayscale_mask'
            )

        if is_update_masks:
            self.__set_masks(generated_masks)

        print('Threshold applied successfully.')
        return generated_masks

    def try_all_algorithm_based_thresholding(self, frame_index: int = 0) -> None:
        """
        Applies thresholding algorithms from the scikit-image library to the captured frames.
        Read more about it here: https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.try_all_threshold
        NOTE: The function might show dark objects over a light background. But for actual implementation, opposite is required.
        Args:
            frame_index (int): The index of the frame to apply the thresholding to. Default is 0.
        Returns:
            None
        """
        gray_image = self._frame_to_grayscale_float(self._captured_working_frames[frame_index])
        fig, ax = try_all_threshold(gray_image, figsize=(10, 8), verbose=False)
        print("Following thresholding algorithms are applied: 'isodata', 'li', 'mean', 'minimum', 'otsu', 'triangle', 'yen'")
        plt.show()

    def apply_algorithm_based_thresholding(self, algorithm: str = 'otsu', is_color_inverse: bool = False, is_update_masks: bool = True, save_masks: bool = False, masks_store_path: str = 'masks', **kwargs) -> List:
        """
        Applies algorithm-based thresholding to the captured frames.
        NOTE: If the dark objects over a light background are getting displayed, Put 'is_color_inverse' to True to correct it before moving to the next step.
        Args:
            algorithm (str): The algorithm to use for thresholding.
                'otsu', 'isodata', 'li', 'mean', 'minimum', 'otsu', 'triangle', 'yen' are the available options.
                Default is 'otsu'.
            is_update_masks (bool): Whether to retain the generated masks on this object.
            is_color_inverse (bool): Whether to invert the colors. Default is False.
            save_masks (bool): Whether to save the thresholded masks.
            masks_store_path (str): The path to store the masks.
            **kwargs: Additional keyword arguments for the thresholding algorithms.
                Check the skimage documentation for more information on other passable args
                Link: https://scikit-image.org/docs/stable/api/skimage.filters.html
        Returns:
            List: The generated masks after applying algorithm-based thresholding.
        """

        # Mapping of available algorithms to their corresponding functions
        algorithm_function_map = {
            'otsu': threshold_otsu,
            'isodata': threshold_isodata,
            'li': threshold_li,
            'mean': threshold_mean,
            'minimum': threshold_minimum,
            'triangle': threshold_triangle,
            'yen': threshold_yen
        }

        if algorithm not in algorithm_function_map:
            raise ValueError(
                f"Algorithm '{algorithm}' is not recognized. Available algorithms: {list(algorithm_function_map.keys())}")

        # Retrieve the threshold function based on the selected algorithm
        threshold_function = algorithm_function_map[algorithm]
        generated_masks: List = []

        for frame_index in trange(len(self._captured_working_frames), desc='Applying algorithm-based thresholding'):
            gray_scale = self._frame_to_grayscale_float(self._captured_working_frames[frame_index])

            # Invert the colors if required
            gray_scale = 1 - gray_scale if is_color_inverse else gray_scale

            threshold_value = threshold_function(gray_scale, **kwargs)
            binary_image = gray_scale > threshold_value
            generated_masks.append(binary_image)

        if save_masks:
            self.save_masks(
                masks=generated_masks,
                masks_store_path=masks_store_path,
                file_prefix=f'{algorithm}_mask'
            )

        if is_update_masks:
            self.__set_masks(generated_masks)

        print(
            f'Selected {algorithm} Algorithm-based thresholding applied successfully.')
        print(
            "NOTE: If dark objects are displayed over a light background, set 'is_color_inverse' to True "
            "and redo the thresholding to correct it before proceeding to the next step."
        )

        return generated_masks

    def apply_gaussian_adaptive_thresholding(self, block_size: int = 11, c: int = 2, is_color_inverse: bool = False, is_update_masks: bool = True, save_masks: bool = False, masks_store_path: str = 'masks') -> List:
        """
        Applies Gaussian adaptive thresholding to the captured frames.
        NOTE: If the dark objects over a light background are getting displayed, Put 'is_color_inverse' to True to correct it before moving to the next step.
        Args:
            block_size (int): The size of the local block for adaptive thresholding. Must be an odd integer greater than 1. Default is 11.
            c (int): The constant to subtract from the mean. Default is 2.
            is_color_inverse (bool): Whether to invert the colors. Default is False.
            is_update_masks (bool): Whether to retain the generated masks on this object.
            save_masks (bool): Whether to save the thresholded masks.
            masks_store_path (str): The path to store the masks.
        Returns:
            List: The generated masks after applying Gaussian adaptive thresholding.
        """
        if block_size <= 1:
            raise ValueError('block_size must be an odd integer greater than 1 for OpenCV adaptiveThreshold.')

        if block_size % 2 == 0:
            raise ValueError(
                f'block_size must be odd for OpenCV adaptiveThreshold. Received block_size={block_size}. '
                f'Try block_size={block_size - 1} or block_size={block_size + 1} instead.'
            )
        generated_masks: List = []
        for frame_index in trange(len(self._captured_working_frames), desc='Applying Gaussian adaptive thresholding'):
            gray_scale = self._frame_to_grayscale_float(self._captured_working_frames[frame_index])
            gray_scale = (gray_scale * 255).astype('uint8')

            # Apply Gaussian adaptive thresholding
            thresholded_image = cv2.adaptiveThreshold(
                gray_scale, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c
            )

            # Convert thresholded image to binary format (True/False)
            # This creates a boolean array (True/False)
            binary_image = thresholded_image > 0

            # Convert to 0/1 representation
            binary_image = binary_image.astype(int)

            # Invert the binary image if is_color_inverse is True
            if is_color_inverse:
                binary_image = 1 - binary_image  # Inverts the binary image

            generated_masks.append(binary_image)

        if save_masks:
            self.save_masks(
                masks=generated_masks,
                masks_store_path=masks_store_path,
                file_prefix='gaussian_adaptive_mask'
            )

        if is_update_masks:
            self.__set_masks(generated_masks)

        print('Gaussian adaptive thresholding applied successfully.')
        print(
            "NOTE: If dark objects are displayed over a light background, set 'is_color_inverse' to True "
            "and redo the thresholding to correct it before proceeding to the next step."
        )
        return generated_masks

    def apply_color_inverse(self, is_update_masks: bool = True) -> List:
        """
        Applies color inverse to the captured frames.
        Args:
            is_update_masks (bool): Whether to retain the generated inverse masks on this object.
        Returns:
            List: The generated inverse masks.
        """
        generated_masks: List = []
        for frame_index in trange(len(self._captured_working_frames), desc='Applying color inverse'):
            inverted_frame = cv2.bitwise_not(
                self._captured_working_frames[frame_index])
            generated_masks.append(inverted_frame)

        if is_update_masks:
            self.__set_masks(generated_masks)

        print('Color inverse applied successfully.')
        return generated_masks

    def generate_region_props_to_dataframe(self, view_props: List[AvailableProps]) -> pd.DataFrame:
        """
        Generates region properties for the captured frames.
        Args:
            view_props (List[AvailableProps]): The list of view properties to generate region properties for.
        Returns:
            pd.DataFrame: The region properties dataframe.
        """
        if not view_props or len(view_props) == 0:
            raise ValueError('The view properties cannot be None or empty.')

        region_props_dataframe = pd.DataFrame()
        if not self._working_masks:
            raise ValueError('No masks are available. Generate or import masks first.')

        for frame_index in trange(len(self._working_masks), desc='Generating region properties'):
            labelled_frame = measure.label(self._working_masks[frame_index])
            properties = tuple(prop.value for prop in view_props)
            region_props = measure.regionprops_table(
                labelled_frame, properties=properties)

            frame_dataframe = pd.DataFrame(region_props)
            frame_dataframe.columns = self.__get_custom_column_names(
                view_props)
            frame_dataframe['frame'] = frame_index + 1

            region_props_dataframe = pd.concat(
                [region_props_dataframe, frame_dataframe], ignore_index=True)
        self._region_props_dataframe = region_props_dataframe
        self._filtered_region_props_dataframe = pd.DataFrame()
        self._applied_filter_settings_dataframe = pd.DataFrame(
            columns=['property', 'operation', 'value']
        )

        print('Region properties generated successfully.')
        return region_props_dataframe

    def apply_filters_on_region_props(self, props_threshold: List[PropsThreshold], is_update_dataframes: bool = True) -> pd.DataFrame:
        """
        Applies filters on the region properties dataframe.
        Args:
            props_threshold (List[PropsThreshold]): The list of property thresholds to apply.
            is_update_dataframes (bool): Whether to update the region properties dataframe.
        Returns:
            pd.DataFrame: The filtered region properties dataframe.
        """
        if self._region_props_dataframe.empty:
            raise ValueError(
                'Region properties dataframe is empty. Please generate region properties first.'
            )
        filtered_df = self._region_props_dataframe.copy()
        for threshold_condition in props_threshold:
            prop = threshold_condition['property']
            operation = threshold_condition['operation']
            value = threshold_condition['value']

            # Applying the filter based on the operation
            if operation == AvailableOperations.GREATER_THAN:
                filtered_df = filtered_df[filtered_df[prop.value] > value]
            elif operation == AvailableOperations.LESS_THAN:
                filtered_df = filtered_df[filtered_df[prop.value] < value]
            elif operation == AvailableOperations.EQUALS:
                filtered_df = filtered_df[filtered_df[prop.value] == value]
            else:
                print(f'Invalid operation in props_threshold: {operation}')

        if is_update_dataframes:
            self._filtered_region_props_dataframe = filtered_df.copy()
            self._applied_filter_settings_dataframe = pd.DataFrame(
                [
                    {
                        'property': threshold_condition['property'].value,
                        'operation': threshold_condition['operation'].value,
                        'value': threshold_condition['value']
                    }
                    for threshold_condition in props_threshold
                ],
                columns=['property', 'operation', 'value']
            )

        print('Filters applied successfully.')
        print(
            f'Full region count: {len(self._region_props_dataframe)}; filtered region count: {len(filtered_df)}'
        )
        return filtered_df

    # Omnipose Methods
    def get_possible_omnipose_model_names(self) -> List[str]:
        """
        Gets the possible omnipose model names.
        Returns:
            List[str]: The list of possible omnipose model names.
        """
        return MODEL_NAMES

    def initialize_omnipose_model(self, model_name: str = 'bact_phase_omni', use_gpu: bool = False, params: dict = OMNIPOSE_DEFAULT_PARAMS) -> None:
        """
        Initializes the omnipose model.
        Args:
            model_name (str): The name of the omnipose model to use.
            params (dict): The parameters to use for the omnipose model.
            use_gpu (bool): Whether to use the GPU for the omnipose model.
        Returns:
            None
        """
        is_gpu_activated = self.__activate_gpu() if use_gpu else False
        omnipose_model = models.CellposeModel(
            gpu=is_gpu_activated, model_type=model_name)
        self.__prepare_frames_for_omnipose_model()
        self._omnipose_model = omnipose_model
        self._omnipose_params = params
        print('Omnipose model initialized successfully.')

    def apply_omnipose_masking(self, batch_size: int = 50, save_masks: bool = False, masks_store_path: str = 'masks', is_update_masks: bool = True) -> List:
        """
        Segments the objects using the omnipose model.
        Args:
            save_masks (bool): Whether to save the masks.
            masks_store_path (str): The path to store the masks.
            is_update_masks (bool): Whether to retain the generated masks on this object.
        Returns:
            List: The segmented masks.
        """
        if not self._omnipose_model:
            raise ValueError('The omnipose model is not initialized.')

        if masks_store_path:
            complete_store_path = self.__handle_folder_preprocess(
                masks_store_path)
            self._mask_store_path = complete_store_path

        masks = self.__get_masks_from_batch_wise_segmented_images(
            batch_size=batch_size,
            save_masks=save_masks)
        print('Objects segmented successfully using the omnipose model.')

        if is_update_masks:
            self.__set_masks(masks)
        return masks

    # Utility Methods
    def save_masks(self, masks: List | None = None, masks_store_path: str = 'masks', file_prefix: str = 'mask') -> None:
        """
        Saves mask frames to disk.
        Args:
            masks (List | None): The masks to save. If None, saves the retained masks.
            masks_store_path (str): The path to store the masks. Existing folders
                and unrelated files are preserved; same-named masks are overwritten.
            file_prefix (str): Prefix to use for saved mask filenames.
        Returns:
            None
        """
        masks_to_save = self._working_masks if masks is None else masks
        if not masks_to_save:
            raise ValueError('No masks were provided and no masks are retained on this object.')

        complete_store_path = self.__handle_folder_preprocess(masks_store_path)
        self._mask_store_path = complete_store_path
        self.__save_masks_to_folder(
            masks=masks_to_save,
            masks_store_path=complete_store_path,
            file_prefix=file_prefix
        )
        print('Masks saved successfully to path: ', complete_store_path)

    def import_saved_masks(self, masks_store_path: str = 'masks', is_update_masks: bool = True, import_as_binary: bool = False) -> List:
        """
        Imports a complete saved mask sequence from disk. The full sequence is
        retained in ``_all_masks`` and the active captured-frame range is retained
        in ``_working_masks``.
        Args:
            masks_store_path (str): The path containing saved masks.
            is_update_masks (bool): Whether to retain the imported masks on this object.
            import_as_binary (bool): Whether to convert imported masks to boolean foreground/background masks.
                If False, integer label values are preserved.
        Returns:
            List: The imported masks.
        """
        complete_masks_path = os.path.join(self._directory, masks_store_path)
        if not os.path.exists(complete_masks_path):
            raise FileNotFoundError(f'Mask folder does not exist: {complete_masks_path}')

        valid_extensions = ('.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp')
        mask_files = sorted(
            file for file in os.listdir(complete_masks_path)
            if file.lower().endswith(valid_extensions)
        )
        if not mask_files:
            raise FileNotFoundError(f'No mask files found in folder: {complete_masks_path}')

        imported_masks: List = []
        for file in mask_files:
            mask_path = os.path.join(complete_masks_path, file)
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            if mask is None:
                raise IOError(f'Failed to read mask from path: {mask_path}')

            if mask.ndim == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

            if import_as_binary:
                mask = mask > 0

            imported_masks.append(mask)

        if len(imported_masks) != len(self._all_captured_frames):
            raise ValueError(
                f'Imported mask count ({len(imported_masks)}) must match the total captured frame count '
                f'({len(self._all_captured_frames)}).'
            )

        if is_update_masks:
            self.__set_masks(imported_masks)

        print('Masks imported successfully from path: ', complete_masks_path)
        return imported_masks

    def plot_centroids(self, show_time=False, prefer_filtered: bool = True) -> None:
        """
        Plots the centroids of the objects.
        Args:
            show_time (bool): Whether to show the time on the plot.
            prefer_filtered (bool): Whether to plot filtered region properties when available.
        Returns:
            None
        """
        dataframe_to_plot = self.get_region_props_dataframe(
            prefer_filtered=prefer_filtered)
        if dataframe_to_plot.empty:
            raise ValueError(
                'Region properties dataframe is empty. Please generate region properties first.'
            )

        if show_time:
            plt.scatter(dataframe_to_plot['centroid_y'], dataframe_to_plot['centroid_x'],
                        s=5, c=dataframe_to_plot['frame'], cmap="jet_r")

            # Add color bar with label
            color_bar = plt.colorbar()
            color_bar.set_label('Frame Number')
        else:
            plt.scatter(dataframe_to_plot['centroid_y'],
                        dataframe_to_plot['centroid_x'], s=5, color='black')
        # Add title
        plt.title('Scatter Plot of Centroids Over Frames')

        # Add axis labels
        plt.xlabel('Pixel X')
        plt.ylabel('Pixel Y')

        plt.gca().invert_yaxis()
        plt.gca().set_aspect('equal', adjustable='box')
        plt.grid(False)
        plt.show()

    def save_identified_objects(self, output_file_name='identified_objects', save_filtered: bool = True) -> None:
        """
        Saves identified region properties to disk.

        If output_file_name ends with '.xlsx', the full region properties, filtered region
        properties, and applied filter settings are saved into separate sheets of one Excel
        workbook. If output_file_name ends with '.csv', the full region properties are saved
        to that CSV path, filtered region properties are saved to a companion *_Filtered.csv
        file when available, and filter settings are saved to Applied_Filter_Settings.csv in
        the same directory.
        if no extension is provided, CSV, XLSX, pickle, and NumPy exports are created.

        Pickle is the preferred reload format because it preserves dataframe dtypes and column
        metadata more faithfully than CSV. NumPy export is also written as a compact reloadable
        dictionary containing dataframe columns and values.

        Missing output folders are created automatically. Existing folders and
        unrelated files are preserved; same-named export files are overwritten.

        Args:
            output_file_name (str): Output filename or path. Extension can be '.csv', '.xlsx', or omitted.
            save_filtered (bool): Whether to include filtered region properties when available.
        Returns:
            None
        """
        if self._region_props_dataframe.empty:
            raise ValueError(
                'Region properties dataframe is empty. Please generate region properties first.'
            )

        root_name, extension = os.path.splitext(output_file_name)
        output_base_path = os.path.join(self._directory, root_name)
        output_directory = os.path.dirname(output_base_path)
        if output_directory:
            os.makedirs(output_directory, exist_ok=True)
        filter_settings_base_path = os.path.join(
            output_directory, 'Applied_Filter_Settings')

        full_dataframe = self._region_props_dataframe.drop(
            columns=['new_x', 'new_y'],
            errors='ignore'
        ).copy()

        filtered_dataframe = pd.DataFrame()
        if not self._filtered_region_props_dataframe.empty:
            filtered_dataframe = self._filtered_region_props_dataframe.drop(
                columns=['new_x', 'new_y'],
                errors='ignore'
            ).copy()

        filter_settings_dataframe = self._applied_filter_settings_dataframe.copy()

        should_save_csv = extension.lower() in ('', '.csv')
        should_save_xlsx = extension.lower() in ('', '.xlsx')
        should_save_pickle = extension.lower() in ('', '.pkl', '.pickle')
        should_save_npy = extension.lower() in ('', '.npy')

        if not should_save_csv and not should_save_xlsx and not should_save_pickle and not should_save_npy:
            raise ValueError(
                "Unsupported output extension. Use '.csv', '.xlsx', '.pkl', '.pickle', '.npy', or omit the extension to save all formats."
            )

        if should_save_csv:
            full_csv_path = f'{output_base_path}.csv'
            full_dataframe.to_csv(full_csv_path, index=False)
            print('Full identified objects saved successfully to path: ', full_csv_path)

            if save_filtered and not filtered_dataframe.empty:
                filtered_csv_path = f'{output_base_path}_Filtered.csv'
                filtered_dataframe.to_csv(filtered_csv_path, index=False)
                print('Filtered identified objects saved successfully to path: ', filtered_csv_path)

            filter_settings_csv_path = f'{filter_settings_base_path}.csv'
            filter_settings_dataframe.to_csv(filter_settings_csv_path, index=False)
            print('Applied filter settings saved successfully to path: ', filter_settings_csv_path)

        if should_save_xlsx:
            xlsx_path = f'{output_base_path}_Complete.xlsx'
            with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
                full_dataframe.to_excel(
                    writer, sheet_name='region_props_full', index=False)
                if save_filtered and not filtered_dataframe.empty:
                    filtered_dataframe.to_excel(
                        writer, sheet_name='region_props_filtered', index=False)
                filter_settings_dataframe.to_excel(
                    writer, sheet_name='Applied_Filter_Settings', index=False)
            print('Identified objects workbook saved successfully to path: ', xlsx_path)

        if should_save_pickle:
            pickle_path = f'{output_base_path}.pkl'
            full_dataframe.to_pickle(pickle_path)
            print('Full identified objects pickle saved successfully to path: ', pickle_path)

            if save_filtered and not filtered_dataframe.empty:
                filtered_pickle_path = f'{output_base_path}_Filtered.pkl'
                filtered_dataframe.to_pickle(filtered_pickle_path)
                print('Filtered identified objects pickle saved successfully to path: ', filtered_pickle_path)

            filter_settings_pickle_path = f'{filter_settings_base_path}.pkl'
            filter_settings_dataframe.to_pickle(filter_settings_pickle_path)
            print('Applied filter settings pickle saved successfully to path: ', filter_settings_pickle_path)

        if should_save_npy:
            npy_path = f'{output_base_path}.npy'
            np.save(
                npy_path,
                {
                    'columns': full_dataframe.columns.to_list(),
                    'data': full_dataframe.to_numpy(dtype=object)
                },
                allow_pickle=True
            )
            print('Full identified objects NumPy file saved successfully to path: ', npy_path)

            if save_filtered and not filtered_dataframe.empty:
                filtered_npy_path = f'{output_base_path}_Filtered.npy'
                np.save(
                    filtered_npy_path,
                    {
                        'columns': filtered_dataframe.columns.to_list(),
                        'data': filtered_dataframe.to_numpy(dtype=object)
                    },
                    allow_pickle=True
                )
                print('Filtered identified objects NumPy file saved successfully to path: ', filtered_npy_path)

            filter_settings_npy_path = f'{filter_settings_base_path}.npy'
            np.save(
                filter_settings_npy_path,
                {
                    'columns': filter_settings_dataframe.columns.to_list(),
                    'data': filter_settings_dataframe.to_numpy(dtype=object)
                },
                allow_pickle=True
            )
            print('Applied filter settings NumPy file saved successfully to path: ', filter_settings_npy_path)

    def load_region_props_from_file(self, full_region_props_path: str, filtered_region_props_path: str | None = None) -> pd.DataFrame:
        """
        Loads full and optionally filtered region properties dataframes from disk.

        Supported formats are '.csv', '.xlsx', '.pkl', '.pickle', and '.npy'. For XLSX files,
        this method looks for sheets named 'region_props_full' and 'region_props_filtered'.
        If a separate filtered_region_props_path is provided, that file is loaded as the
        filtered dataframe.

        Args:
            full_region_props_path (str): Path to the full region properties file.
            filtered_region_props_path (str | None): Optional path to the filtered region properties file.
        Returns:
            pd.DataFrame: The active region properties dataframe, preferring filtered props when available.
        """
        full_path = os.path.join(self._directory, full_region_props_path)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f'Full region properties file does not exist: {full_path}')

        full_extension = os.path.splitext(full_path)[1].lower()

        if full_extension == '.xlsx':
            workbook = pd.ExcelFile(full_path)
            if 'region_props_full' in workbook.sheet_names:
                self._region_props_dataframe = pd.read_excel(
                    full_path, sheet_name='region_props_full')
            else:
                self._region_props_dataframe = pd.read_excel(full_path)

            if 'region_props_filtered' in workbook.sheet_names:
                self._filtered_region_props_dataframe = pd.read_excel(
                    full_path, sheet_name='region_props_filtered')
            else:
                self._filtered_region_props_dataframe = pd.DataFrame()
        else:
            self._region_props_dataframe = self.__load_region_props_dataframe_from_path(
                full_path)
            self._filtered_region_props_dataframe = pd.DataFrame()

        if filtered_region_props_path is not None:
            filtered_path = os.path.join(self._directory, filtered_region_props_path)
            if not os.path.exists(filtered_path):
                raise FileNotFoundError(f'Filtered region properties file does not exist: {filtered_path}')
            self._filtered_region_props_dataframe = self.__load_region_props_dataframe_from_path(
                filtered_path)
        elif full_extension != '.xlsx':
            root_name, extension = os.path.splitext(full_path)
            companion_filtered_path = f'{root_name}_filtered{extension}'
            if os.path.exists(companion_filtered_path):
                self._filtered_region_props_dataframe = self.__load_region_props_dataframe_from_path(
                    companion_filtered_path)

        self._region_props_dataframe = self._region_props_dataframe.drop(
            columns=['new_x', 'new_y'],
            errors='ignore'
        )
        self._filtered_region_props_dataframe = self._filtered_region_props_dataframe.drop(
            columns=['new_x', 'new_y'],
            errors='ignore'
        )

        print(f'Full region properties loaded: {len(self._region_props_dataframe)} rows')
        if not self._filtered_region_props_dataframe.empty:
            print(f'Filtered region properties loaded: {len(self._filtered_region_props_dataframe)} rows')
        else:
            print('No filtered region properties loaded.')

        return self.get_region_props_dataframe(prefer_filtered=True)

    def get_region_props_dataframe(self, prefer_filtered: bool = True) -> pd.DataFrame:
        """
        Gets the region properties dataframe.
        Args:
            prefer_filtered (bool): If True and filtered region properties are available,
                return self._filtered_region_props_dataframe. Otherwise return the full
                self._region_props_dataframe.
        Returns:
            pd.DataFrame: The requested region properties dataframe.
        """
        if prefer_filtered and not self._filtered_region_props_dataframe.empty:
            return self._filtered_region_props_dataframe
        return self._region_props_dataframe

    def get_full_region_props_dataframe(self) -> pd.DataFrame:
        """
        Gets the full, unfiltered region properties dataframe.
        Returns:
            pd.DataFrame: The full region properties dataframe.
        """
        return self._region_props_dataframe

    def get_filtered_region_props_dataframe(self) -> pd.DataFrame:
        """
        Gets the filtered region properties dataframe.
        Returns:
            pd.DataFrame: The filtered region properties dataframe.
        """
        return self._filtered_region_props_dataframe

    def analyze_major_axis_length_distribution(
        self,
        dataframe: pd.DataFrame | None = None,
        prefer_filtered: bool = True,
        aggregate_by_particle: bool = False,
        particle_column: str = 'particle',
        component_count: int | None = None,
        max_components: int = 6,
        min_component_fraction: float = 0.01,
        show_plot: bool = True
    ) -> dict:
        """
        Analyze major-axis-length modes and identify supported morphology ranges.

        The source can be this object's full or filtered region properties, or any
        supplied dataframe such as finalized linked particle tracks. When
        aggregate_by_particle is True, the distribution contains one mean
        observation per particle instead of one observation per detection.

        A Gaussian mixture model is selected by Bayesian information criterion
        unless component_count is provided. The biological labels single_cells,
        long_single_cells, candidate_linked_pairs, and longer_chains are assigned
        when three consecutive modes have compatible length and area ratios. A
        longer-chain range is added only when the following mode also has compatible
        scaling.

        Args:
            dataframe (pd.DataFrame | None): Optional region-properties or linked-particle
                dataframe. If None, uses this Identify object's region properties.
            prefer_filtered (bool): Prefer retained filtered region properties when
                dataframe is None.
            aggregate_by_particle (bool): Average morphology per particle before fitting.
            particle_column (str): Particle identifier used for aggregation.
            component_count (int | None): Optional fixed number of mixture components.
            max_components (int): Maximum components considered during automatic selection.
            min_component_fraction (float): Minimum fraction required for each biologically
                labeled component.
            show_plot (bool): Whether to display the fitted distribution and ranges.

        Returns:
            dict: Analysis metadata, component_summary, classification_ranges,
                classified_dataframe, support diagnostics, and the optional figure.
        """
        try:
            from sklearn.mixture import GaussianMixture  # type: ignore
        except ImportError as error:
            raise ImportError(
                'scikit-learn is required to analyze major-axis-length distributions.'
            ) from error

        if isinstance(max_components, bool) or not isinstance(max_components, (int, np.integer)):
            raise TypeError('max_components must be an integer.')
        if max_components < 1:
            raise ValueError('max_components must be at least 1.')
        max_components = int(max_components)

        if component_count is not None:
            if isinstance(component_count, bool) or not isinstance(component_count, (int, np.integer)):
                raise TypeError('component_count must be an integer or None.')
            component_count = int(component_count)
            if component_count < 1 or component_count > max_components:
                raise ValueError(
                    'component_count must be between 1 and max_components.'
                )

        if not 0 < min_component_fraction <= 1:
            raise ValueError('min_component_fraction must be greater than 0 and at most 1.')

        if dataframe is None:
            source_dataframe = self.get_region_props_dataframe(
                prefer_filtered=prefer_filtered).copy()
            source_name = (
                'filtered_region_properties'
                if prefer_filtered and not self._filtered_region_props_dataframe.empty
                else 'full_region_properties'
            )
        else:
            if not isinstance(dataframe, pd.DataFrame):
                raise TypeError('dataframe must be a pandas DataFrame or None.')
            source_dataframe = dataframe.copy()
            source_name = 'provided_dataframe'

        if source_dataframe.empty:
            raise ValueError('No rows are available for major-axis-length analysis.')

        length_column = 'major_axis_length'
        if length_column not in source_dataframe.columns:
            raise ValueError(
                "The dataframe must contain a 'major_axis_length' column."
            )

        morphology_columns = [length_column]
        for optional_column in ('area', 'minor_axis_length'):
            if optional_column in source_dataframe.columns:
                morphology_columns.append(optional_column)

        for column in morphology_columns:
            source_dataframe[column] = pd.to_numeric(
                source_dataframe[column], errors='coerce')

        valid_length_mask = (
            np.isfinite(source_dataframe[length_column]) &
            (source_dataframe[length_column] > 0)
        )
        source_dataframe = source_dataframe.loc[valid_length_mask].copy()

        if aggregate_by_particle:
            if particle_column not in source_dataframe.columns:
                raise ValueError(
                    f"aggregate_by_particle=True requires a '{particle_column}' column."
                )
            analysis_dataframe = source_dataframe.groupby(
                particle_column, as_index=False, dropna=False
            )[morphology_columns].mean()
            analysis_unit = 'particle_mean'
        else:
            analysis_dataframe = source_dataframe
            analysis_unit = 'detection'

        analysis_dataframe = analysis_dataframe[
            np.isfinite(analysis_dataframe[length_column]) &
            (analysis_dataframe[length_column] > 0)
        ].copy()

        values = analysis_dataframe[length_column].to_numpy(dtype=float)
        unique_value_count = len(np.unique(values))
        maximum_fittable_components = min(
            max_components,
            unique_value_count,
            max(1, len(values) // 10)
        )

        if len(values) < 20:
            raise ValueError(
                'At least 20 valid major-axis-length observations are required.'
            )
        if component_count is not None and component_count > maximum_fittable_components:
            raise ValueError(
                'component_count is too large for the number of available observations.'
            )

        fit_values = values.reshape(-1, 1)
        fitted_models = {}
        bic_scores = {}
        for candidate_count in range(1, maximum_fittable_components + 1):
            model = GaussianMixture(
                n_components=candidate_count,
                covariance_type='full',
                n_init=5,
                max_iter=1000,
                random_state=0,
                reg_covar=1e-6
            )
            model.fit(fit_values)
            fitted_models[candidate_count] = model
            bic_scores[candidate_count] = float(model.bic(fit_values))

        selected_component_count = (
            component_count
            if component_count is not None
            else min(bic_scores, key=bic_scores.get)
        )
        selected_model = fitted_models[selected_component_count]

        model_means = selected_model.means_.reshape(-1)
        model_standard_deviations = np.sqrt(
            selected_model.covariances_.reshape(-1))
        model_weights = selected_model.weights_.reshape(-1)
        component_order = np.argsort(model_means)

        ordered_means = model_means[component_order]
        ordered_standard_deviations = model_standard_deviations[component_order]
        ordered_weights = model_weights[component_order]

        component_boundaries = []
        for component_index in range(selected_component_count - 1):
            left_mean = ordered_means[component_index]
            right_mean = ordered_means[component_index + 1]
            boundary_grid = np.linspace(left_mean, right_mean, 20001)

            left_log_density = (
                np.log(max(ordered_weights[component_index], 1e-12)) -
                np.log(max(ordered_standard_deviations[component_index], 1e-12)) -
                0.5 * (
                    (boundary_grid - left_mean) /
                    max(ordered_standard_deviations[component_index], 1e-12)
                ) ** 2
            )
            right_log_density = (
                np.log(max(ordered_weights[component_index + 1], 1e-12)) -
                np.log(max(ordered_standard_deviations[component_index + 1], 1e-12)) -
                0.5 * (
                    (boundary_grid - right_mean) /
                    max(ordered_standard_deviations[component_index + 1], 1e-12)
                ) ** 2
            )
            closest_index = int(np.argmin(np.abs(
                left_log_density - right_log_density)))
            component_boundaries.append(float(boundary_grid[closest_index]))

        component_assignments = np.searchsorted(
            component_boundaries, values, side='right')
        component_column = f'{length_column}_component'
        class_column = f'{length_column}_class'
        analysis_dataframe[component_column] = component_assignments

        pixel_scale_factor = float(self._parent.get_pixel_scale_factor())
        has_valid_scale = np.isfinite(pixel_scale_factor) and pixel_scale_factor > 0
        scale_units = self._parent.get_scale_units() or 'scale_units'

        component_rows = []
        for component_index in range(selected_component_count):
            component_mask = component_assignments == component_index
            component_dataframe = analysis_dataframe.loc[component_mask]
            lower_bound = (
                float('-inf')
                if component_index == 0
                else component_boundaries[component_index - 1]
            )
            upper_bound = (
                float('inf')
                if component_index == selected_component_count - 1
                else component_boundaries[component_index]
            )

            row = {
                'component': component_index,
                'lower_bound_pixels': lower_bound,
                'upper_bound_pixels': upper_bound,
                'model_mean_pixels': float(ordered_means[component_index]),
                'mean_pixels': float(component_dataframe[length_column].mean()),
                'median_pixels': float(component_dataframe[length_column].median()),
                'standard_deviation_pixels': float(component_dataframe[length_column].std()),
                'count': int(len(component_dataframe)),
                'fraction': float(len(component_dataframe) / len(analysis_dataframe)),
            }
            if has_valid_scale:
                row.update({
                    'lower_bound_scale_units': lower_bound * pixel_scale_factor,
                    'upper_bound_scale_units': upper_bound * pixel_scale_factor,
                    'mean_scale_units': row['mean_pixels'] * pixel_scale_factor,
                    'standard_deviation_scale_units': (
                        row['standard_deviation_pixels'] * pixel_scale_factor
                    ),
                })
            if 'area' in component_dataframe.columns:
                row['mean_area'] = float(component_dataframe['area'].mean())
            if 'minor_axis_length' in component_dataframe.columns:
                row['mean_minor_axis_length'] = float(
                    component_dataframe['minor_axis_length'].mean())
            component_rows.append(row)

        component_summary = pd.DataFrame(component_rows).set_index('component')
        classification_supported = False
        longer_chains_supported = False
        support_diagnostics = {}
        support_messages = []
        selected_sequence_start = None

        if selected_component_count < 3:
            support_messages.append(
                'Fewer than three distribution modes were selected.'
            )
        elif 'mean_area' not in component_summary.columns:
            support_messages.append(
                "An 'area' column is required to support biological range labels."
            )
        else:
            sequence_candidates = []
            for sequence_start in range(selected_component_count - 2):
                single_row = component_summary.iloc[sequence_start]
                long_single_row = component_summary.iloc[sequence_start + 1]
                pair_row = component_summary.iloc[sequence_start + 2]

                single_length = single_row['mean_pixels']
                single_area = single_row['mean_area']
                length_ratios = {
                    'long_single_to_single': long_single_row['mean_pixels'] / single_length,
                    'pair_to_single': pair_row['mean_pixels'] / single_length,
                }
                area_ratios = {
                    'long_single_to_single': long_single_row['mean_area'] / single_area,
                    'pair_to_single': pair_row['mean_area'] / single_area,
                }
                component_fractions = [
                    single_row['fraction'],
                    long_single_row['fraction'],
                    pair_row['fraction'],
                ]

                sequence_is_supported = (
                    1.15 <= length_ratios['long_single_to_single'] <= 1.80 and
                    1.65 <= length_ratios['pair_to_single'] <= 2.40 and
                    1.20 <= area_ratios['long_single_to_single'] <= 1.90 and
                    1.60 <= area_ratios['pair_to_single'] <= 2.60 and
                    min(component_fractions) >= min_component_fraction
                )

                minor_axis_ratio = None
                if 'mean_minor_axis_length' in component_summary.columns:
                    minor_axis_ratio = (
                        pair_row['mean_minor_axis_length'] /
                        single_row['mean_minor_axis_length']
                    )
                    sequence_is_supported = (
                        sequence_is_supported and
                        0.75 <= minor_axis_ratio <= 1.35
                    )

                ratio_score = (
                    abs(np.log(length_ratios['long_single_to_single'] / 1.5)) +
                    abs(np.log(length_ratios['pair_to_single'] / 2.0)) +
                    abs(np.log(area_ratios['long_single_to_single'] / 1.5)) +
                    abs(np.log(area_ratios['pair_to_single'] / 2.0))
                )
                if minor_axis_ratio is not None:
                    ratio_score += abs(np.log(minor_axis_ratio))

                if sequence_is_supported:
                    sequence_candidates.append((
                        float(ratio_score),
                        sequence_start,
                        length_ratios,
                        area_ratios,
                        minor_axis_ratio
                    ))

            if sequence_candidates:
                (
                    _,
                    selected_sequence_start,
                    length_ratios,
                    area_ratios,
                    minor_axis_ratio
                ) = min(sequence_candidates, key=lambda candidate: candidate[0])
                classification_supported = True
                support_diagnostics = {
                    'length_ratios': length_ratios,
                    'area_ratios': area_ratios,
                    'pair_to_single_minor_axis_ratio': minor_axis_ratio,
                }
                support_messages.append(
                    'Single-cell, long-single-cell, and candidate-linked-pair modes '
                    'have compatible length and area scaling.'
                )

                chain_component = selected_sequence_start + 3
                if chain_component < selected_component_count:
                    single_row = component_summary.iloc[selected_sequence_start]
                    chain_row = component_summary.iloc[chain_component]
                    chain_length_ratio = (
                        chain_row['mean_pixels'] / single_row['mean_pixels']
                    )
                    chain_area_ratio = (
                        chain_row['mean_area'] / single_row['mean_area']
                    )
                    longer_chains_supported = (
                        2.50 <= chain_length_ratio <= 4.00 and
                        2.20 <= chain_area_ratio <= 4.20 and
                        chain_row['fraction'] >= min_component_fraction
                    )
                    support_diagnostics['length_ratios'][
                        'chain_to_single'
                    ] = chain_length_ratio
                    support_diagnostics['area_ratios'][
                        'chain_to_single'
                    ] = chain_area_ratio

                if longer_chains_supported:
                    support_messages.append(
                        'The following mode also supports a longer-chain range.'
                    )
                else:
                    support_messages.append(
                        'No following mode satisfied the longer-chain scaling checks.'
                    )
            else:
                support_messages.append(
                    'No three consecutive modes satisfied the single-cell, long-single-cell, '
                    'and candidate-pair scaling checks.'
                )

        biological_labels = [
            'single_cells',
            'long_single_cells',
            'candidate_linked_pairs',
            'longer_chains',
        ]
        component_to_class = {}
        if classification_supported and selected_sequence_start is not None:
            component_to_class = {
                selected_sequence_start + offset: label
                for offset, label in enumerate(biological_labels[:3])
            }
            if longer_chains_supported:
                component_to_class[
                    selected_sequence_start + 3
                ] = biological_labels[3]

        analysis_dataframe[class_column] = (
            analysis_dataframe[component_column]
            .map(component_to_class)
            .fillna('outside_supported_ranges')
        )

        if classification_supported:
            selected_components = list(component_to_class.keys())
            classification_ranges = component_summary.loc[selected_components].copy()
            classification_ranges.insert(
                0, 'classification', list(component_to_class.values()))
            classification_ranges = classification_ranges.set_index('classification')
        else:
            classification_ranges = pd.DataFrame()

        analysis_warnings = []
        if classification_supported:
            analysis_warnings.append(
                'Morphology labels are heuristic and should be validated against '
                'representative images and masks.'
            )
        if (
            component_count is None and
            selected_component_count == maximum_fittable_components
        ):
            analysis_warnings.append(
                'The lowest BIC occurred at max_components; additional distribution structure may exist.'
            )
        if not aggregate_by_particle and particle_column in source_dataframe.columns:
            analysis_warnings.append(
                'Linked-track rows are detection-weighted. Use aggregate_by_particle=True '
                'to give each particle equal weight.'
            )

        figure = None
        if show_plot:
            figure, axis = plt.subplots(figsize=(12, 6))
            plot_lower = max(0.0, float(np.quantile(values, 0.001)))
            plot_upper = float(np.quantile(values, 0.995))
            histogram_bin_count = int(np.clip(
                np.sqrt(len(values)), 25, 200))
            axis.hist(
                values,
                bins=histogram_bin_count,
                range=(plot_lower, plot_upper),
                density=True,
                alpha=0.35,
                color='#4C78A8',
                label='Observations'
            )

            plot_grid = np.linspace(plot_lower, plot_upper, 2000)
            mixture_density = np.zeros_like(plot_grid)
            for mean_value, standard_deviation, weight in zip(
                ordered_means,
                ordered_standard_deviations,
                ordered_weights
            ):
                standard_deviation = max(float(standard_deviation), 1e-12)
                component_density = (
                    weight /
                    (standard_deviation * np.sqrt(2 * np.pi)) *
                    np.exp(-0.5 * ((plot_grid - mean_value) / standard_deviation) ** 2)
                )
                mixture_density += component_density
            axis.plot(
                plot_grid,
                mixture_density,
                color='#1F4E79',
                linewidth=2.5,
                label=f'{selected_component_count}-component mixture'
            )

            class_colors = {
                'single_cells': '#2E8B57',
                'long_single_cells': '#D99C24',
                'candidate_linked_pairs': '#6A5ACD',
                'longer_chains': '#C44E52',
            }
            if classification_supported:
                for label, range_row in classification_ranges.iterrows():
                    lower_bound = max(
                        plot_lower, range_row['lower_bound_pixels'])
                    upper_bound = min(
                        plot_upper, range_row['upper_bound_pixels'])
                    axis.axvspan(
                        lower_bound,
                        upper_bound,
                        color=class_colors[label],
                        alpha=0.10,
                        label=label.replace('_', ' ')
                    )
                    axis.axvline(
                        range_row['mean_pixels'],
                        color=class_colors[label],
                        linestyle='--',
                        linewidth=1.5
                    )

            axis.set_xlim(plot_lower, plot_upper)
            axis.set_xlabel('Major-axis length (pixels)')
            axis.set_ylabel('Density')
            support_label = (
                'supported morphology ranges'
                if classification_supported
                else 'biological ranges not supported'
            )
            axis.set_title(
                f'Major-axis-length distribution ({len(values):,} {analysis_unit}s)\n'
                f'{support_label}'
            )
            axis.legend(frameon=False)

            if has_valid_scale:
                secondary_axis = axis.secondary_xaxis(
                    'top',
                    functions=(
                        lambda pixel_value: pixel_value * pixel_scale_factor,
                        lambda scale_value: scale_value / pixel_scale_factor
                    )
                )
                secondary_axis.set_xlabel(
                    f'Major-axis length ({scale_units})')

            plt.tight_layout()
            plt.show()

        print(
            f'Analyzed {len(values):,} {analysis_unit}s using a '
            f'{selected_component_count}-component major-axis-length model.'
        )
        if classification_supported:
            print('Supported morphology ranges were identified successfully.')
            for message in support_messages:
                print(f'- {message}')
        else:
            print('Biological morphology ranges were not assigned automatically.')
            for message in support_messages:
                print(f'- {message}')

        return {
            'source': source_name,
            'analysis_unit': analysis_unit,
            'observation_count': len(values),
            'pixel_scale_factor': pixel_scale_factor,
            'scale_units': scale_units,
            'selected_component_count': selected_component_count,
            'bic_scores': bic_scores,
            'classification_supported': classification_supported,
            'longer_chains_supported': longer_chains_supported,
            'support_messages': support_messages,
            'support_diagnostics': support_diagnostics,
            'warnings': analysis_warnings,
            'component_summary': component_summary,
            'classification_ranges': classification_ranges,
            'classified_dataframe': analysis_dataframe,
            'component_column': component_column,
            'class_column': class_column,
            'figure': figure,
        }

    def get_directory(self):
        """
        Retrieves the working directory.
        Returns:
          str: The working directory.
        """
        return self._directory

    # Private methods
    def __load_region_props_dataframe_from_path(self, file_path: str) -> pd.DataFrame:
        """
        Loads a region properties dataframe from a supported file path.
        """
        extension = os.path.splitext(file_path)[1].lower()

        if extension == '.csv':
            return pd.read_csv(file_path)

        if extension in ('.pkl', '.pickle'):
            return pd.read_pickle(file_path)

        if extension == '.npy':
            loaded_object = np.load(file_path, allow_pickle=True).item()
            return pd.DataFrame(
                loaded_object['data'],
                columns=loaded_object['columns']
            )

        if extension == '.xlsx':
            return pd.read_excel(file_path)

        raise ValueError(
            "Unsupported region properties file extension. Use '.csv', '.xlsx', '.pkl', '.pickle', or '.npy'."
        )

    def __get_custom_column_names(self, view_props: List[AvailableProps]) -> List[str]:
        """
        Function to get the column names.
        Args:
            view_props (List[AvailableProps]): The list of view properties to get the column names for.
        Returns:
            List[str]: The list of column names for the view properties.
        Raises:
            ValueError: If the view properties are None.
        """
        column_names: List[str] = []
        if view_props is None:
            raise ValueError('The view properties cannot be None.')
        for view_prop in view_props:
            if view_prop == AvailableProps.CENTROID:
                column_names.append('centroid_x')
                column_names.append('centroid_y')
            else:
                column_names.append(view_prop.value)
        return column_names

    def __prepare_frames_for_omnipose_model(self) -> None:
        """
        Prepares the captured frames for the omnipose model - Normalizing the frames.
        Returns:
            None
        """
        if self._normalized_frames:
            print('Frames are already prepared for the omnipose model.')
            return
        frames_to_prepare = self._captured_working_frames
        is_binary_frames = self.__are_frames_binary(frames_to_prepare)
        if is_binary_frames:
            frames_to_prepare = self.__convert_to_uint8(frames_to_prepare)

        normalized_frames = []
        for frame_index in trange(len(frames_to_prepare), desc='Preparing frames for the omnipose model'):
            gray_image = frames_to_prepare[frame_index] if is_binary_frames else self._frame_to_grayscale_float(
                frames_to_prepare[frame_index])
            normalized_frame = normalize99(gray_image)
            normalized_frames.append(normalized_frame)
        self._normalized_frames = normalized_frames
        print('Frames prepared successfully for the omnipose model.')

    def __are_frames_binary(self, frames: List) -> bool:
        """
        Checks if all frames are binary (contain only True or False values).
        Args:
            frames (List): The frames to be checked.
        Returns:
            bool: True if all frames are binary, False otherwise.
        """
        return all(np.array_equal(frame, frame.astype(bool)) for frame in frames)

    def __convert_to_uint8(self, binary_frames: List) -> List:
        """
        Converts binary frames to uint8 format.
        Args:
            binary_frames (List): The binary frames to be converted.
        Returns:
            List: The converted frames in uint8 format.
        """
        uint8_frames = [(frame.astype(np.uint8) * 255)
                        for frame in binary_frames]
        return uint8_frames

    def __activate_gpu(self) -> bool:
        """
        Activates the GPU for the omnipose model.
        Returns:
            bool: Whether the GPU is activated.
        """
        use_gpu = core.use_gpu()
        print(f'>>> GPU activated? {use_gpu}')
        return use_gpu

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

    def __get_segmented_masks(self, batch_images: List) -> List:
        """
        Gets the segmented masks for the batch images.
        Args:
            batch_images (List): The list of batch images to get the segmented masks for.
        Returns:
            List: The segmented masks.
        """
        tic = time.time()
        masks, _, _ = self._omnipose_model.eval(
            batch_images, **self._omnipose_params)  # type: ignore
        net_time = time.time() - tic
        print(f'total segmentation time: {net_time}s')
        return masks

    def __process_batch_images_to_get_masks(self, batch_images: List, save_masks: bool = True, batch_start_index: int = 0) -> List:
        """
        Processes the batch images to get the binary masks.
        Args:
            batch_images (List): The list of batch images to process.
            save_masks (bool): Whether to save the masks.
        Returns:
            List: The binary masks.
        """
        masks = self.__get_segmented_masks(batch_images)
        if save_masks:
            self.__save_masks_to_folder(
                masks=masks,
                masks_store_path=self._mask_store_path,
                file_prefix='omnipose_mask',
                start_index=batch_start_index
            )
        return masks

    def __save_masks_to_folder(self, masks: List, masks_store_path: str, file_prefix: str = 'mask', start_index: int = 0) -> None:
        """
        Saves a list of masks to a folder.
        Args:
            masks (List): The list of masks to save.
            masks_store_path (str): The folder path where masks should be saved.
            file_prefix (str): Prefix to use for saved mask filenames.
            start_index (int): Zero-based offset within the working captured frames.
        Returns:
            None
        """
        mask_count = len(masks)
        frame_range = self._parent.get_working_frame_range()
        working_start_index = frame_range.get('start_index', 0)
        total_frame_count = frame_range.get(
            'total_frame_count', len(self._captured_working_frames))
        first_frame_number = working_start_index + start_index + 1
        last_frame_number = first_frame_number + mask_count - 1
        mask_index_width = len(str(max(total_frame_count, last_frame_number)))
        for i, mask in enumerate(masks):
            frame_number = first_frame_number + i
            mask_number = str(frame_number).zfill(mask_index_width)
            mask_path = os.path.join(
                masks_store_path, f'{file_prefix}_{mask_number}.tiff')
            self.__write_mask_to_path(mask, mask_path)

    def __set_masks(self, masks: List) -> None:
        """Stores masks and aligns the working masks with the working frames."""
        mask_count = len(masks)
        all_frame_count = len(self._all_captured_frames)
        working_frame_count = len(self._captured_working_frames)

        if mask_count == all_frame_count:
            all_masks = list(masks)
            frame_range = self._parent.get_working_frame_range()
            start_index = frame_range.get('start_index', 0)
            end_index = frame_range.get(
                'end_index', start_index + working_frame_count - 1)
            working_masks = all_masks[start_index:end_index + 1]
        elif mask_count == working_frame_count:
            all_masks = list(masks)
            working_masks = list(all_masks)
        else:
            raise ValueError(
                f'Mask count ({mask_count}) must match either the total captured frame count '
                f'({all_frame_count}) or the working captured frame count '
                f'({working_frame_count}).'
            )

        if len(working_masks) != working_frame_count:
            raise ValueError(
                f'Working mask count ({len(working_masks)}) must match the working '
                f'captured frame count ({working_frame_count}).'
            )

        self._all_masks = all_masks
        self._working_masks = working_masks

    def __write_mask_to_path(self, mask: np.ndarray, mask_path: str) -> None:
        """
        Writes one mask to disk using a TIFF-safe dtype.
        Args:
            mask (np.ndarray): The mask to write.
            mask_path (str): The destination file path.
        Returns:
            None
        """
        # OpenCV can fail silently when writing unsupported dtypes.
        # Label masks should be saved as integer TIFFs so label IDs are preserved.
        mask_to_save = np.asarray(mask)
        if mask_to_save.dtype == bool:
            mask_to_save = mask_to_save.astype(np.uint8) * 255
        elif np.issubdtype(mask_to_save.dtype, np.integer):
            if mask_to_save.max(initial=0) <= np.iinfo(np.uint16).max:
                mask_to_save = mask_to_save.astype(np.uint16)
            else:
                mask_to_save = mask_to_save.astype(np.uint32)
        else:
            mask_to_save = mask_to_save.astype(np.float32)

        is_written = cv2.imwrite(mask_path, mask_to_save)
        if not is_written:
            raise IOError(f'Failed to write mask to path: {mask_path}')

    def __get_masks_from_batch_wise_segmented_images(self, batch_size: int = 5, save_masks: bool = True) -> List:
        """
        Gets the masks from the batch-wise segmented images.
        Args:
            batch_size (int): The batch size to use for segmentation.
            save_masks (bool): Whether to save the masks.
        Returns:
            List: The masks.
        """
        resultant_masks = []
        for each in trange(0, len(self._normalized_frames), batch_size, desc='Segmenting images'):
            batch_images = self._normalized_frames[each: each + batch_size]
            binary_masks = self.__process_batch_images_to_get_masks(
                batch_images, save_masks, each)
            print(f'Batch {each // batch_size + 1} segmentation is complete')
            resultant_masks += binary_masks
        return resultant_masks

    # Gaussian Fit Methods
    def __2d_gaussian(self, coords, amplitude, xo, yo, sigma_x, sigma_y, offset):
        """
        A static 2D Gaussian model.
        """
        x, y = coords
        return (amplitude * np.exp(-(((x - xo) ** 2) / (2 * sigma_x ** 2) +
                                     ((y - yo) ** 2) / (2 * sigma_y ** 2))) + offset).ravel()

    def __extract_subimage(self, gray_frame: np.ndarray, center: tuple, fit_window: int):
        """
        Extracts a sub-image centered at `center` using a window of size fit_window.

        Args:
            gray_frame (np.ndarray): Grayscale image.
            center (tuple): Center in (y, x) order.
            fit_window (int): Half-size of the window.

        Returns:
            sub_image (np.ndarray): Extracted sub-image.
            xmin, ymin (int): Top-left coordinates of the sub-image.
            If the window exceeds image boundaries, returns (None, None, None).
        """
        x_center = center[1]  # center is (y, x)
        y_center = center[0]
        xmin = int(x_center - fit_window)
        xmax = int(x_center + fit_window)
        ymin = int(y_center - fit_window)
        ymax = int(y_center + fit_window)

        if xmin < 0 or ymin < 0 or xmax >= gray_frame.shape[1] or ymax >= gray_frame.shape[0]:
            return None, None, None
        sub_image = gray_frame[ymin:ymax, xmin:xmax]
        return sub_image, xmin, ymin

    def __fit_gaussian_to_subimage(self, sub_image: np.ndarray, fit_window: int):
        """
        Performs a 2D Gaussian fit on the given sub-image.

        Args:
            sub_image (np.ndarray): The extracted sub-image.
            fit_window (int): Half-size of the window.

        Returns:
            opt_param (np.ndarray): Optimized parameters.
            x_vals, y_vals (np.ndarray): Meshgrid arrays corresponding to the sub-image.
        """
        x_vals, y_vals = np.meshgrid(
            np.arange(sub_image.shape[1]), np.arange(sub_image.shape[0]))
        x_data = x_vals.ravel()
        y_data = y_vals.ravel()
        intensity_data = sub_image.ravel()

        amplitude_guess = np.max(sub_image)
        sigma_guess = max(fit_window / 2, 1)
        offset_guess = np.min(sub_image)
        initial_guess = (amplitude_guess, fit_window / 2,
                         fit_window / 2, sigma_guess, sigma_guess, offset_guess)
        bounds = ((0, 0, 0, 0.1, 0.1, 0), (np.inf, fit_window,
                  fit_window, fit_window, fit_window, np.inf))

        opt_param, _ = curve_fit(self.__2d_gaussian, (x_data, y_data), intensity_data,
                                 p0=initial_guess, bounds=bounds)
        return opt_param, x_vals, y_vals

    def __compute_refined_centroid(self, opt_param: np.ndarray, xmin: int, ymin: int, initial_center: tuple, fit_window: int):
        """
        Computes the refined centroid (in full-image coordinates) from the fitted parameters.

        Args:
            opt_param (np.ndarray): Optimized Gaussian parameters.
            xmin, ymin (int): Top-left coordinates of the sub-image.
            initial_center (tuple): Original centroid in (y, x) order.
            fit_window (int): Half-size of the window.

        Returns:
            (refined_x, refined_y): The refined centroid.
            If the shift exceeds fit_window, returns the initial centroid.
        """
        refined_x = xmin + opt_param[1]
        refined_y = ymin + opt_param[2]
        x_init = initial_center[1]
        y_init = initial_center[0]
        if abs(refined_x - x_init) > fit_window or abs(refined_y - y_init) > fit_window:
            return x_init, y_init
        return refined_x, refined_y

    def __process_gaussian_fit_on_a_frame(self, gray_frame: np.ndarray, frame_index: pd.DataFrame, fit_window: int = 7) -> tuple[dict, dict]:
        """
        Processes the Gaussian fit on a single frame.

        Args:
            gray_frame (np.ndarray): Grayscale image.
            frame_index (int): The index of the frame to process.
            fit_window (int): Half-size of the window for the Gaussian fit.
        Returns:
            initial_centroids (dict): Initial centroids in (y, x) order.
            gaussian_centroids (dict): Refined centroids in (y, x) order.
        """
        frame_region_props = self._region_props_dataframe[
            self._region_props_dataframe['frame'] == frame_index]

        # Skip empty frames
        if frame_region_props.empty:
            return None  # Skip empty frames

        initial_centroids = {}
        gaussian_centroids = {}

        for _, row in frame_region_props.iterrows():
            initial_center = (row['centroid_y'], row['centroid_x'])
            label = row['label']

            initial_centroids[label] = initial_center

            sub_image, xmin, ymin = self.__extract_subimage(
                gray_frame, initial_center, fit_window)
            if sub_image is None:
                gaussian_centroids[label] = initial_center
                continue

            try:
                opt_param, _, _ = self.__fit_gaussian_to_subimage(
                    sub_image, fit_window)
                refined_x, refined_y = self.__compute_refined_centroid(
                    opt_param, xmin, ymin, initial_center, fit_window)
                gaussian_centroids[label] = (refined_y, refined_x)
            except Exception:  # pylint: disable=W0703
                gaussian_centroids[label] = initial_center
        return initial_centroids, gaussian_centroids

    def visualize_gaussian_fit_on_a_frame(self, frame_index: int, fit_window: int = 7) -> None:
        """
        Visualizes the Gaussian fit on all centroids in a frame.

        Args:
            frame_index (int): The index of the frame to visualize.
            fit_window (int): Half-size of the window for the Gaussian fit.

        Returns:
            None
        """
        if self._region_props_dataframe.empty:
            print(
                "Region properties dataframe is empty. Please generate region properties first.")
            return

        if frame_index < 0 or frame_index > len(self._captured_working_frames):
            print("Invalid frame index.")
            return

        gray_frame = self._captured_working_frames[frame_index - 1]

        initial_centroids, refined_centroids = self.__process_gaussian_fit_on_a_frame(
            gray_frame, frame_index, fit_window)

        plt.figure(figsize=(10, 8))
        plt.imshow(gray_frame, cmap='gray')
        plt.plot([center[0] for center in initial_centroids.values()],
                 [center[1] for center in initial_centroids.values()], 'bo', markersize=5)
        plt.plot([center[0] for center in refined_centroids.values()],
                 [center[1] for center in refined_centroids.values()], 'g*', markersize=5)
        plt.title(f"Gaussian Fit on All Centroids in Frame {frame_index}")
        plt.xlabel("Y (pixels)")
        plt.ylabel("X (pixels)")
        legend_elements = [Line2D([0], [0], marker='o', color='w', label='Initial centroid',
                                  markerfacecolor='b', markersize=8),
                           Line2D([0], [0], marker='*', color='w', label='Refined centroid',
                                  markerfacecolor='g', markersize=12)]
        plt.legend(handles=legend_elements, loc='upper right')
        plt.show()

    def optimize_centroids_using_gaussian_fit(
        self,
        fit_window: int = 7,
        max_workers: int = None,
        output_file_name: str = 'Gaussian_Fit_on_Centroids_Fit_Window_Applied'
    ) -> None:
        """
        Optimizes the centroid coordinates in the internal region properties dataframe using
        parallel processing with threads. Each frame is processed independently: for each region
        in a frame, a sub-image is extracted and a 2D Gaussian fit is performed. If the refined
        centroid is within acceptable bounds, the internal dataframe is updated accordingly.

        This version uses ThreadPoolExecutor for parallel processing.

        Args:
            fit_window (int): The search window size around the detected centroid (default is 7).
            max_workers (int): Maximum number of worker threads. If None, ThreadPoolExecutor
                            will choose a default.
            output_file_name (str): Output basename or path for the automatically exported
                CSV and NumPy fit-window files. Relative paths use the capture working
                directory. The '.csv' or '.npy' extension can be included or omitted.
        Returns:
            None
        """
        if self._region_props_dataframe.empty:
            print(
                "Region properties dataframe is empty. Please generate region properties first.")
            return

        output_base_name, output_extension = os.path.splitext(
            output_file_name)
        if output_extension.lower() not in ('', '.csv', '.npy'):
            raise ValueError(
                "Unsupported output extension. Use '.csv', '.npy', or omit "
                'the extension.'
            )
        output_base_path = os.path.join(self._directory, output_base_name)
        output_directory = os.path.dirname(output_base_path)

        num_frames = len(self._captured_working_frames)
        # Assuming frames are numbered starting at 1
        frame_numbers = range(1, num_frames + 1)

        futures = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for frame_num in frame_numbers:
                # Submit processing task for each captured frame.
                futures[executor.submit(self.__process_gaussian_fit_on_a_frame,
                                        self._captured_working_frames[frame_num - 1],
                                        frame_num, fit_window)] = frame_num

            results = {}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing frames (Parallel)"):
                frame_num = futures[future]
                try:
                    # Expecting each future to return a tuple: (frame_num, (initial_centroids, refined_centroids))
                    result = future.result()
                    results[frame_num] = result[1]
                except Exception as e:  # pylint: disable=W0703
                    print(f"Error processing frame {frame_num}: {e}")
                    results[frame_num] = None

        # Update the internal dataframe using the refined centroids
        for frame_num, result in tqdm(results.items(), desc="Updating centroids"):
            if result is None:
                continue
            for label, refined_center in result.items():
                self._region_props_dataframe.loc[
                    (self._region_props_dataframe['frame'] == frame_num) &
                    (self._region_props_dataframe['label'] == label),
                    ['centroid_y', 'centroid_x']
                ] = refined_center

        if output_directory:
            os.makedirs(output_directory, exist_ok=True)

        fit_window_dataframe = pd.DataFrame({'fit_window': [fit_window]})
        csv_path = f'{output_base_path}.csv'
        fit_window_dataframe.to_csv(csv_path, index=False)

        npy_path = f'{output_base_path}.npy'
        np.save(npy_path, {'fit_window': fit_window}, allow_pickle=True)

        print("Centroids optimized successfully using Gaussian fit.")
        print('Gaussian fit window saved successfully to path: ', csv_path)
        print('Gaussian fit window saved successfully to path: ', npy_path)
