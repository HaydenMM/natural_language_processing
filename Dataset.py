"""
 Copyright 2020 Carnegie Mellon University. All rights reserved.

 NREC Confidential and Proprietary

 This notice must appear in all copies of this file and its derivatives

"""

import logging
import os
import re
import shutil
import subprocess
from tempfile import TemporaryFile

from more_itertools import windowed
import numpy as np

from eurystheus.utils import helperfuns, CameraIntrinsics, CameraPosition, Timestamps, WorkDirectory
from eurystheus.datastructures import Label

from ouroboros.db.schemas import LabelType, Role, Mission, Robot


DOCKER_CONTAINER_NAME = os.getenv('RYS_CONTAINER_NAME', 'unincorn_3')
# Default max length for datasets is 60 seconds.
DATASET_MAX_LEN = os.getenv('RYS_MAX_LENGTH', 60)

DEFAULT_LOG_PARENT_DIRECTORY = os.path.join(os.sep, 'mnt', 'aidtr', 'Staging', '')
OVERRIDE_LOG_PARENT_DIRECTORY = os.getenv('RYS_LOG_PATH', None)


class Dataset:
    """ Parent class for the 4 types of datasets handled by the Labeling
        Manager. It allows external scripts to have a single unified API that
        works for each different dataset type.

        This is the only class that should be called by external scripts. The
        factory() method will create the appropriate dataset child class based
        on the passed dataset_type.
    """
    @staticmethod
    def factory(dataset_type, **kwargs):
        """ Create a new Dataset of type dataset_type.

            Args:
                dataset_type (ouroboros.db.schemas.LabelType): An
                    Enum defining the type of dataset to create.

            Returns: dataset (Dataset): A new dataset class with populated data
        """
        if dataset_type == LabelType.BOXLABEL:
            dataset = _Dataset2dBbox
        elif dataset_type == LabelType.SEGLABEL:
            dataset = _Dataset2dSegmentation
        elif dataset_type in (LabelType.POINTCLOUDBOXLABEL, LabelType.POINTCLOUDSEGLABEL):
            dataset = _Dataset3d
        else:
            raise NotImplementedError(
                'Dataset type {} is not currently in '
                'supported set [{}, {}].'.format(
                    dataset_type, LabelType.BOXLABEL, LabelType.SEGLABEL,
                    LabelType.POINTCLOUDBOXLABEL, LabelType.POINTCLOUDSEGLABEL
                )
            )

        return dataset(dataset_type, **kwargs)


class _Dataset:
    """ A class containing all the shared attributes of all dataset types
    """
    def __init__(self, dataset_type, **kwargs):
        """
            Args:
                dataset_type (ouroboros.db.schemas.LabelType): An
                    Enum defining the type of dataset to create.

            Optional Args:
                ID (int): The ID of the dataset (also the name of the dataset
                    on the labeling tool)
                project (str): The ID of the project (also the name of the
                    project on the labeling tool). Deepen API V1.
                profile (str): The id of the labeling profile to assign to the
                    dataset. Deepen API V2.
                module (int): The module (on the AIDTR robot) the dataset is
                    generated from
                lens (int): The camera within the module (on the AIDTR robot)
                    the dataset is generated from
                start (int): The start timestamp (in microseconds) of the dataset
                stop (int): The end timestamp (in microseconds) of the dataset
                directions (str): Directions for the labelers of the dataset
                comments (str): Comments created during sawmilling of the log
                log_path (str): The path to the log which was collected on the
                    AIDTR robot and stored on the AIDTR FreeNAS
                metadata (dict): Data collection (log level) metadata of the
                    dataset
                mission (str): The mission the robot was driven when collecting
                    this data.
                labels (ouroboros.db.schemas.BoxLabel): Any existing labels
                    (if relabeling datasets)
                intrinsics_path (str): The path to camera intrinsics for this
                    datasets; stored on the AIDTR FreeNAS
                urdf_path (str): The path to the calibration URDF for this
                    dataset; stored on the AIDTR FreeNAS
        """
        self.type = dataset_type

        self.name = kwargs.get('ID')
        self.project_name = kwargs.get('project')
        self.profile = kwargs.get('profile', None)
        self.module = kwargs.get('module')
        self.lens = kwargs.get('lens')
        self.start = kwargs.get('start')
        self.stop = kwargs.get('stop')
        self.directions = kwargs.get('directions')
        self.comments = kwargs.get('comments')
        self.log_path = kwargs.get('log_path')
        self.metadata = kwargs.get('metadata')
        self.mission = kwargs.get('mission')
        self.deepen_id = None
        self.image_files = None

        robot = kwargs.get('robot')
        if robot:
            self.robot = robot

        self.labels = [Label.fromRow(label) for label in kwargs.get('labels', [])]

        if OVERRIDE_LOG_PARENT_DIRECTORY is not None:
            self.log_path = self.log_path.replace(
                DEFAULT_LOG_PARENT_DIRECTORY,
                ''
            )
            self.log_path = os.path.join(
                OVERRIDE_LOG_PARENT_DIRECTORY,
                self.log_path
            )

        # The name of the camera as represented in the calibration / pose files
        self.camera_name = WorkDirectory.getCameraName(self.module, self.lens, self.robot)
        # the primary camera is the middle modules top camera
        if self.robot == "LHEX2":
            self.primary_cam_name = WorkDirectory.getCameraName(0, 0, self.robot)
        else:
            self.primary_cam_name = WorkDirectory.getCameraName(2, 0, self.robot)

        # Construct the path to the calibration files
        self.intrinsics_path = os.path.join(
            helperfuns.getFreenasMountPoint(),
            kwargs.get('intrinsics_path')
        )
        self.urdf_path = os.path.join(
            helperfuns.getFreenasMountPoint(),
            kwargs.get('urdf_path')
        )

        # If the start and end timestamps have more than 18 significant digits,
        # they are in nanoseconds and need to be converted to micro.
        if self.start > 1e17 and self.stop > 1e17:
            self.start = self.start // 1000
            self.stop = self.stop // 1000

        # Tests to keep invalid or very large datasets from being uploaded
        if self.start > self.stop:
            raise ValueError('Invalid Start and Stop values ({} and {}). '
                             'Start must be less than Stop.'.format(
                                 self.start, self.stop
                             ))

        max_length = float(DATASET_MAX_LEN) * 1e+6
        dataset_length = self.stop - self.start
        if dataset_length > max_length:
            raise ValueError('Dataset is too long ({} seconds).'
                             'Max length set to {} seconds.'.format(
                                 float(dataset_length) // float(1e+6),
                                 float(max_length) // float(1e+6)
                             ))

    def findImageFiles(self, work_directory, filter_pattern, full_path=False):
        """Find the path to images to be included in this dataset

        Args:
            work_directory (eurystheus.utils.WorkDirectory): Wrapper util for
               handling temporary directories and the contents used for
               datasets.
            filter_pattern (str, optional): A regex pattern, matches pass the
                filter. Defaults to None.
            full_path (bool, optional): If True will collect the absolute path
                to the image files. Defaults to False.

        Returns:
            None
        """
        
        image_files = os.listdir(work_directory.getRectifiedImagePath(''))
        
        if filter_pattern:
            reg_pattern = re.compile(filter_pattern)
            image_files = list(filter(reg_pattern.match, image_files))
       
        if not full_path:
            image_files = [os.path.basename(image_file) for image_file in image_files]
        else:
            image_files = [os.path.abspath(image_file) for image_file in image_files]

        self.image_files = image_files

    def prepareForExport(self, work_directory, filter_pattern):
        """ Gets camera params as well as cuts and extracts lumbers for upload.

            Args:
                work_directory (Eurystheus.utils.WorkDirectory): A helper class
                    that helps navigate the extracted lumbers and their files.
            filter_pattern (str): A regex pattern, matches pass the  filter.

            Returns:
                None
        """
        logging.info('Preparing dataset for export to labeling tool.')

        if not helperfuns.workdirOverride():
            lumber_path = self.cutLumber(
                work_directory=work_directory
            )

            self.extractLumber(
                work_directory=work_directory,
                lumber_path=lumber_path,
            )

        self.intrinsics = CameraIntrinsics(self.intrinsics_path)
        self.positions = CameraPosition(work_directory.getWorkDirectoryPath())
        print("\n" + "Lumber path:" + lumber_path)
        print("\n" + str(self.extractLumber))
        print("\n" + "intrinsics:" + str(self.intrinsics))
        print("\n" + "positions:" + str(self.positions))
        self.findImageFiles(work_directory, filter_pattern)

    def cutLumber(self, work_directory):
        """ Calls the LogSplitter to cut a lumber from a log.

            Args:
                work_directory (Eurystheus.utils.WorkDirectory): A helper class
                    that helps navigate the extracted lumbers and their files.

            Returns:
                lumber_path (str): the path to the extracted lumber.
        """
        lumber_path = os.path.join(
            work_directory.getWorkDirectoryPath(),
            '{}_{}'.format(self.name, os.path.basename(self.log_path))
        )
        logging.info("Cutting Lumber to '{}'.".format(lumber_path))

        cmd = [
            '/opt/unicorn-deploy/unicorn/bin/LogSplitter',
            '-i', self.log_path,
            '-o', lumber_path,
            '-s', str(self.start),
            '-t', str(self.stop)
        ]

        containerized = os.getenv("RYS_DOCKER", False)
        if not containerized:
            cmd = ['docker', 'exec', '--user', '{}'.format(os.getuid()), DOCKER_CONTAINER_NAME] + cmd

        logging.info('\t{}'.format(cmd))

        subprocess.run(cmd, check=True)

        return lumber_path

    def extractLumber(self, work_directory, lumber_path):
        """ Calls the URFExtractor extract the images / pointclouds from a
                lumber.

            Must be implemented by subclasses.

            Args:
                work_directory (Eurystheus.utils.WorkDirectory): A helper class
                    that helps navigate the extracted lumbers and their files.
                lumber_path (str): the path to the extracted lumber

            Returns:
                None
        """
        raise NotImplementedError('Abstract method not implemented.')

    def __repr__(self):
        return '\n{} (ID: {}; Project: {}; Module: {}; Lens: {}; Log Path: {})'.format(
            self.__class__,
            self.name, self.project_name, self.module, self.lens, self.log_path
        )


class _Dataset2d(_Dataset):
    def __init__(self, dataset_type, **kwargs):
        super().__init__(dataset_type, **kwargs)

        # TODO: what even is this?
        self.dims = 'images'

    def extractLumber(self, work_directory, lumber_path):
        """ Calls the URFExtractor extract the images (either RBG or thermal)
            from a lumber.

            Args:
                work_directory (Eurystheus.utils.WorkDirectory): A helper class
                    that helps navigate the extracted lumbers and their files.
                lumber_path (str): the path to the extracted lumber.

            Returns:
                None
        """
        logging.info('Extracting Lumber.')

        cmd = [
            '/opt/unicorn-deploy/aidtr/bin/URFExtractor',
            '--intrinsicsPath', self.intrinsics_path,
            '--urdf', self.urdf_path,
            '-c', '/opt/unicorn-deploy/config/sensorModuleLayout.cfg',
            '-r', lumber_path,
            '-o', work_directory.getWorkDirectoryPath(),
            '--Poses',
            '--Rectify',
            '--Timestamps',
            '--Original',
            '--module {}'.format(self.module),
            '--source {}'.format(self.lens)
        ]

        containerized = os.getenv("RYS_DOCKER", False)
        if not containerized:
            cmd = ['docker', 'exec', '--user', '{}'.format(os.getuid()), DOCKER_CONTAINER_NAME] + cmd

        logging.info('\t{}'.format(cmd))

        subprocess.run(cmd, check=True)


class _Dataset2dBbox(_Dataset2d):
    """ A class that defines the specific attributes of a 2d bounding box
        dataset
    """
    def __init__(self, dataset_type, **kwargs):
        super().__init__(dataset_type, **kwargs)
        # The roles must a user posses in order to label this dataset.
        # Note: This isn't used right now, since labeling tasks are claimed not
        # assigned
        self.labeler_roles = [Role.BOX2D_LABELER]

    def prepareForExport(self, work_directory):
        """ Extends _Dataset's prepareForExport to include the sensor data per
                frame.

            Args:
                work_directory (Eurystheus.utils.WorkDirectory): A helper class
                    that helps navigate the extracted lumbers and their files.

            Returns:
                None
        """
        filter_pattern = 'p0m{}l{}.*'.format(self.module, self.lens)
        super().prepareForExport(work_directory, filter_pattern)

        self.timestamps = Timestamps(str(work_directory))

        # Groups rectified images by the frame (one image per frame in this case)
        frames, self.rectified_image_groups = helperfuns.groupby(self.image_files, helperfuns.findFrame)
        self.frames = [int(frame) for frame in frames]
        print(self.frames, self.rectified_image_groups)
        # No frame points or frames necessary for 2d BBox labels. Needed
        # for consistent iteration across all dataset types.
        self.frame_points = []

    def iter(self, work_directory, no_imu, **kwargs):
        """ A custom iterator for 2D datasets.

            Args:
                work_directory (Eurystheus.utils.WorkDirectory): A helper class
                    that helps navigate the extracted lumbers and their files.
                no_imu (bool): If True, position data will not be included
                    or used in the extracted dataset.

            Returns (yields for each frame):
                dictionary {
                    frame (int): The id of the frame
                    rectified_image_group (list[str]): A list of rectified image names
                    camera_params (dict): a list of camera parameters for
                        each camera
                }

        """
        self.prepareForExport(work_directory)
        idat = zip(
            self.frames,
            self.rectified_image_groups
        )
        print(idat)
        if len(self.rectified_image_groups) == 0:
            raise Exception('No images for module {} lens {} exist for '
                            ' dataset with ID {}. Aborting upload.'.format(
                                self.module, self.lens, self.name
                            ))

        idat = zip(
            self.frames,
            self.rectified_image_groups
        )
        print(idat)
        cur_frame = 0
        total_frames = len(self.frames)
        for frame, rectified_group in idat:
            logging.info('[{} of {}] Processing Frame: {} '.format(
                cur_frame + 1,
                total_frames,
                frame
            ))
            cur_frame += 1

            # In cases where there is no IMU data, the no_imu flag can be set
            # and the position data passed to deepen will be dummy position data.
            if no_imu:
                device_pos = self.positions.fudge(frame)
                camera_pos = self.positions.fudge(frame)
            else:
                device_pos = self.positions.get(frame, self.primary_cam_name)
                camera_pos = self.positions.get(frame, self.camera_name)

            yield {
                'frame': frame,
                'robot': self.robot,
                'rectified_image_group': rectified_group,
                'device_params': device_pos,
                'camera_params': {
                    self.camera_name: {
                        **self.intrinsics.get(self.camera_name),
                        **camera_pos
                    }
                }
            }


class _Dataset2dSegmentation(_Dataset2dBbox):
    """ A class that defines the specific attributes of a 2d segmentation
        dataset

        Note: this is the same as 2d bounding box datasets except for
            labeler roles.
    """
    def __init__(self, dataset_type, **kwargs):
        super().__init__(dataset_type, **kwargs)
        # The roles must a user posses in order to label this dataset.
        self.labeler_roles = [Role.SEG2D_LABELER]

    def iter(self, work_directory, no_imu, **kwargs):
        """ A custom iterator for 2D Segmentation datasets.

            Note: The difference is minimal. Segmentation datasets with a mission
                "SceneSegmenation" will only return a single frame for labeling.
                All other segmentation datasets will return all the frames in the
                dataset.

            Args:
                work_directory (Eurystheus.utils.WorkDirectory): A helper class
                    that helps navigate the extracted lumbers and their files.
                no_imu (bool): If True, position data will not be included
                    or used in the extracted dataset.

            Returns (a list of length 1):
                dictionary {
                    frame (int): The id of the frame
                    rectified_image_group (list[str]): A list of rectified image names
                    camera_params (dict): a list of camera parameters for
                        each camera
                }

        """
        print("ITER")
        for cur_frame_idx, data in enumerate(super().iter(work_directory, no_imu, **kwargs)):
            # We only want to label a single frame for SceneSegmenation datasets (they have 2 to 5 frames).
            # To do so, we call the super class iterator (a generator for all frames of data) but only yield
            # the 3rd frame (or the last frame of the dataset if it has less than three frames)-- as the
            # later frames are more likely to be captured when the robot is stationary. Returning immediately
            # after the yield, stops iteration after one frame is processed.
            # Otherwise we yield all the result in sequence, maintaining the default functionality.
            print(1)
            if self.mission == Mission.SCENESEGMENTATION:
                # Get the frame to return for SceneSegmentation datasets.
                # TODO: This seems really dumb to do inside the loop...
                num_frames = len(self.frames)
                return_frame = min(2, num_frames - 1)  # Return the third frame, or the last frame (if there are less than 3)

                # Skip the first |return_frame| frames.
                if cur_frame_idx < return_frame:
                    continue

                self.frames = [self.frames[return_frame]]
                self.rectified_image_groups = [self.rectified_image_groups[return_frame]]

                logging.info('SCENESEGMENTATION log, only processing frame {} of {}.'.format(cur_frame_idx, num_frames))

                yield data
                print(data)
                return  # after processing the return_frame, return so no other frames are processed.
            else:
                print(data)
                yield data


class _Dataset3d(_Dataset):
    """ A class that defines the specific attributes of a 3d dataset

    Note: this hasn't been used in a year or so. With the number of Deepen
    changes I expect it to need rework.
    """
    def __init__(self, dataset_type, **kwargs):
        super().__init__(dataset_type, **kwargs)

        # TODO: What even is this?
        self.dims = '3d'

        # The roles must a user posses in order to label this dataset.
        self.labeler_roles = [Role.BOX3D_LABELER]
        # For 3d datasets we always upload all modules images
        self.modules = [0, 1, 2, 3, 4]
        # For 3d datasets we always upload both the RGB and Thermal images
        self.lenses = [1, 2]

    def extractLumber(self, work_directory, lumber_path):
        """ Calls the URFExtractor extract the images & pointclouds form a
                lumber.

            Args:
                work_directory (Eurystheus.utils.WorkDirectory): A helper class
                    that helps navigate the extracted lumbers and their files.
                lumber_path (str): the path to the extracted lumber.

            Returns:
                None
        """
        logging.info('Extracting Lumber.')

        cmd = [
            '/opt/unicorn-deploy/aidtr/bin/URFExtractor',
            '--intrinsicsPath', self.intrinsics_path,
            '-c', '/opt/unicorn-deploy/config/sensorModuleLayout.cfg'
            '--urdf', self.urdf_path,
            '-r', lumber_path,
            '-o', work_directory.getWorkDirectoryPath(),
            '--Poses',
            '--Rectify',
            '--Lidar',
            *['--module {}'.format(mod) for mod in self.modules],
            *['--source {}'.format(lens) for lens in self.lenses]
        ]

        containerized = os.getenv("RYS_DOCKER", False)
        if not containerized:
            cmd = ['docker', 'exec', '--user', '{}'.format(os.getuid()), DOCKER_CONTAINER_NAME] + cmd

        logging.info('\t{}'.format(cmd))

        subprocess.run(cmd, check=True)

    def prepareForExport(self, work_directory, export_stereo_pointcloud):
        """ Extends _Dataset's prepareForExport to include the sensor data per
                frame.

            Args:
                work_directory (Eurystheus.utils.WorkDirectory): A helper class
                    that helps navigate the extracted lumbers and their files.
                export_stereo_pointcloud (bool): Option, if present also export
                    and upload the stereo point cloud (along with the LiDAR
                    point cloud) to the labeling tool.

            Returns:
                None
        """
        filter_pattern = 'p0m\dl[{}].*'.format(','.join([str(lens) for lens in self.lenses]))
        super().prepareForExport(work_directory, filter_pattern)

        frames, self.rectified_image_groups = helperfuns.groupby(self.image_files, helperfuns.findFrame)
        self.frames = [int(frame) for frame in frames]

        os.mkdir(work_directory.merged_pointcloud_dir)

        self.lidar_pointcloud_groups = []
        for start, stop in windowed([0] + self.frames, 2):
            point_cloud_positions = self.positions.get_between(
                start, stop, 'velodyne'
            )
            pointcloud_data = {
                'filepaths': [],
                'transform_matrices': []
            }
            for point_cloud_position in point_cloud_positions:
                pointcloud_data['filepaths'].append(work_directory.getLidarPointCloudPath(
                    '{}.pcd'.format(point_cloud_position['timestamp'])
                ))

                R = helperfuns.rotationFromQuaternion(
                    **point_cloud_position['heading']
                )
                t = np.array([
                    point_cloud_position['position']['x'],
                    point_cloud_position['position']['y'],
                    point_cloud_position['position']['z']
                ])

                pointcloud_data['transform_matrices'].append(helperfuns.composeH(
                    R, t
                ))

            outfile = work_directory.getMergedPointCloudPath(
                '{}.pcd'.format(stop)
            )
            helperfuns.accumulatePointClouds(
                filepaths=pointcloud_data['filepaths'],
                outfile=outfile,
                transform_matrices=pointcloud_data['transform_matrices']
            )
            self.lidar_pointcloud_groups.append(outfile)

    def getFramePoints(self, work_directory, pointcloud_groups):
        """ Counts the number of points within each frames point cloud(s)

            Args:
                work_directory (Eurystheus.utils.WorkDirectory): A helper class
                    that helps navigate the extracted lumbers and their files.
                pointcloud_groups (list[list[str]]): a list (number of frames)
                    of list (point clouds in frame) of strings (paths to point
                    clouds on disk)

            Returns:
                frame_points (list[int]): the number of points in each frame's
                    point cloud(s).
        """
        frame_points = []
        for pointcloud_group in pointcloud_groups:
            group_points = 0
            for pointcloud in pointcloud_group:
                with open(pointcloud, 'r') as infile:
                    # Parse the header, extracting the total number of points
                    for header in infile:
                        header = header.strip()
                        # Keep track of the total number of points in the group
                        if header.startswith(work_directory.NUM_POINTS_KEY):
                            group_points += int(header.split(' ')[-1])
                            break
            frame_points.append(group_points)

        return frame_points

    def mergePointClouds(self, work_directory):
        """ Merges point clouds from the 5 modules into a single cloud.

            Note: this is a brute force method. Overlapping points
            are not combined.

            Args:
                work_directory (Eurystheus.utils.WorkDirectory): A helper class
                    that helps navigate the extracted lumbers and their files.

            Returns:
                frame_points (list[int]): the number of points in each frame's
                    merged point cloud.
        """
        logging.info('Extracting Lumber Merging Point Clouds')
        outdir = work_directory.merged_pointcloud_dir
        if os.path.exists(outdir):
            shutil.rmtree(outdir)
        os.mkdir(outdir)

        # Get the Lumbers pointcloud files and group them by frame id
        # TODO: this should be global point cloud
        pointcloud_dir = work_directory.local_pointcloud_dir

        group_ids, pointcloud_files_groups = helperfuns.groupby(
            os.listdir(pointcloud_dir), helperfuns.findFrame
        )

        # Merge each group putting the merged files into the outdir
        frame_points = []
        for group_id, pointcloud_group in zip(group_ids, pointcloud_files_groups):

            # Each frame will be merged into a single temporary file
            with TemporaryFile(mode='w+t') as tmp_outfile:
                # the header needs to define the total number of points in the
                # cloud, which we wont know until we have finished merging
                total_points = 0
                counted_points = False
                for num_merged, pointcloud_file in enumerate(pointcloud_group):
                    # Open each point cloud file and
                    pointcloud_path = os.path.join(
                        pointcloud_dir, pointcloud_file
                    )
                    with open(pointcloud_path, 'r') as infile:
                        # Parse the header, extracting the total number of points
                        # And writing a single header to the merged file
                        for header in infile:
                            if num_merged == 0:
                                tmp_outfile.write(header)

                            header = header.strip()
                            # Keep track of the total number of points
                            if not counted_points and header.startswith(work_directory.NUM_POINTS_KEY):
                                total_points += int(header.split(' ')[-1])
                            # Stop when header is reached
                            if header == work_directory.END_PLY_HEADER:
                                break

                        # After the header is parsed, start copying the points to
                        # the merged file.
                        for point in infile:
                            tmp_outfile.write(point)

                # Now the merged file has been created.
                # Copy all its data from the temp file to the final output with the
                # correct number of points in the header
                tmp_outfile.seek(0)
                outfile = os.path.join(outdir, 'p0_%s.ply' % group_id)
                with open(outfile, 'w') as merged_pointcloud:
                    for line in tmp_outfile:
                        if line.startswith(work_directory.NUM_POINTS_KEY):
                            line = '%s %i\n' % (
                                work_directory.END_PLY_HEADER, total_points
                            )
                        else:
                            line = line

                        merged_pointcloud.write(line)

            # Keep track of each merged point cloud's total number of points
            frame_points.append(total_points)

        return frame_points

    def iter(self, work_directory, export_stereo_pointcloud, **kwargs):
        """ A custom iterator for 3D datasets.

            Args:
                work_directory (Eurystheus.utils.WorkDirectory): A helper class
                    that helps navigate the extracted lumbers and their files.
                export_stereo_pointcloud (bool): Option, if present also export
                    and upload the stereo point cloud (along with the LiDAR
                    point cloud) to the labeling tool.

            Returns (yields for each frame):
                dictionary {
                    frame (int): The id of the frame
                    rectified_image_group (list[str]): A list of rectified
                        image names
                    lidar_pointcloud_group (list[str]): A list of lidar point
                        cloud file names
                    stereo_pointcloud_group (list[str]): A list of stereo point
                        cloud file names
                    camera_params (dict): a list of camera parameters for
                        each camera
                }

        """
        self.prepareForExport(work_directory, export_stereo_pointcloud)

        idat = zip(
            self.frames,
            self.rectified_image_groups,
            self.lidar_pointcloud_groups,
        )

        cur_frame = 0
        total_frames = len(self.frames)
        for frame, rectified_group, lidar_pc_group in idat:
            logging.info('[{} of {}] Processing Frame: {} '.format(
                cur_frame + 1,
                total_frames,
                frame
            ))
            cur_frame += 1
            # Get all the RGB and Thermal camera parmeters for all modules
            camera_params = {}
            for module in range(5):
                for lens in self.lenses:
                    camera_name = WorkDirectory.getCameraName(module, lens, self.robot)
                    camera_params[camera_name] = {
                        **self.intrinsics.get(camera_name),
                        **self.positions.get(frame, camera_name)
                    }

            temp = {
                'frame': frame,
                'rectified_image_group': rectified_group,
                'lidar_pointcloud_group': lidar_pc_group,
                'device_params': self.positions.get(frame, self.primary_cam_name),
                'camera_params': camera_params,
            }
            yield temp
