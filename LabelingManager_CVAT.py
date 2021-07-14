"""
 Copyright 2020 Carnegie Mellon University. All rights reserved.

 NREC Confidential and Proprietary

 This notice must appear in all copies of this file and its derivatives

"""

from datetime import datetime, date
import json
import logging
import os
import random
import traceback

from eurystheus.api.cvat import CvatWrapper
from eurystheus.datastructures.Dataset import Dataset
from eurystheus.utils.Filter import Filter
from eurystheus.utils import helperfuns, LabelingProfileGenerator
from eurystheus.utils.WorkDirectory import WorkDirectory

from ouroboros.db.utils.labeling_manager import LabelingManager as LM
from ouroboros.db.connection import create_session
from ouroboros.db.schemas import Status, LabelType, Robot
from ouroboros.integrations.jira import JiraWrapper, Issue, Stage


class LabelingManager:
    """ A controller class for managing the labeling tool.

        Intended to be run as a Jenkins Project to automatically upload new
        dataset to the labeling tool, assign tasks to labelers, download
        completed labels, and delete completed datasets.

        Can be manually driven via the console command `rys`. After pip
        intsalling this repo type `rys --help` for more info

        Uses CVAT currently, implementation is built off of previous LabelingManager() which used Deepen,
        the encapsulated APIs made for an easy transition to CVAT.
    """

    @staticmethod
    def _setupLogging(verbose):
        now = datetime.now().strftime('%Y%m%dT%H%M%S')
        log_path = "/var/log/eurystheus"
        if not os.path.exists(log_path):
            try:
                os.mkdir(log_path)
            except PermissionError:
                log_path = "/tmp/eurystheus"
                if not os.path.exists(log_path):
                    os.mkdir(log_path)

        if helperfuns.debugMode():
            logging.basicConfig(level=logging.DEBUG)
        elif verbose:
            logging.basicConfig(level=logging.INFO)
        else:
            logging.basicConfig(level=logging.INFO,
                                filename="{}/{}.log".format(log_path, now))

    def startup(self, credentials, config, free_space_buffer=1.25e+8, verbose=False, seed=42069):
        """ Creates all connections required to interface with the Labeling
            Tool api and Ouroboros

            Args:
                credentials (easy.dict): user specified credentials
                config (easy.dict): general project configuration info
                free_space_buffer (float): how much space to leave open on the
                    labeling tool
                verbose (bool): print logging output to the console.
                seed (int): The random seed.
        """
        self._setupLogging(verbose)

        random.seed(seed)

        # Are we running in the production or test database
        self.prod = config.cfg.DB_NAME != 'test'
        if not self.prod:
            logging.warning('RUNNING IN TEST DATABASE.')

        self.free_space_buffer = free_space_buffer
        self.jira_project = config.cfg.JIRA_PROJECT

        # Startup CVAT's API Wrapper
        self.api = CvatWrapper(
            username=credentials.cred.CVAT_USER,
            password=credentials.cred.CVAT_PASS,
            url=config.cfg.CVAT_URL
        )

        self.USE_JIRA = config.cfg.USE_JIRA

        # Startup JIRA's API Wrapper
        if self.USE_JIRA:
            self.jira = JiraWrapper(
                url=config.cfg.JIRA_URL,
                username=credentials.cred.JIRA_USERNAME,
                password=credentials.cred.JIRA_PASSWORD
            )

        # Create a new session with the Postgres database
        driver = "postgresql+psycopg2"
        username = credentials.cred.DB_USERNAME
        password = credentials.cred.DB_PASSWORD
        host = config.cfg.DB_HOST
        database = config.cfg.DB_NAME
        database_uri = driver + "://" + username + ":" + password + "@" + host + "/" + database
        self.session = create_session(database_uri)

        self.labeling_manager = LM(
            session=self.session
        )

        self.class_mapping = config.cfg.get('CLASS_MAPPING', {})

    def createProject(self, project, definition, type=None, active=True, **kwargs):
        """ Given a project name and path to configuration file, creates a new
            project on the Labeling Tool and uploads it to the database.

            Args:
                project [str]: the name of the project (must be unique)
                definition [str]: a path to the JSON configuration file of the
                    project
                type (ouroboros.db.schema.LabelType): LabelType for this project.
                active [bool]: sets the project as active (the one to upload
                    datasets to)

            Returns:
                success [bool]: if the api request completed successfully
        """
        logging.info('Creating new project `{}` from file `{}`'.format(project, definition))

        # Read the project's configuration file into memory as a dictionary
        with open(definition, 'r') as config_file:
            config_dict = json.load(config_file)
        
        success, project_id = self.api.createProject(
            project_name=project,
            config=definition
        )

        # If the project was successfully created on the labeling tool, add it
        # to the DB
        if success:
            success = self.labeling_manager.create_project(
                name=project,
                config=config_dict,
                active=active,
                type=type,
                profile=project_id
            )
        
        return success, project_id

    def getProject(self, project, **kwargs):
        """ Gets a project from the labeling tool.

            Args:]
                project [str]: the name of the project
            Returns:
                success [bool]: if the api request completed successfully
                project [dict]: the json representation of the project
        """
        logging.info('Getting project `{}`'.format(project))

        success, project = self.api.getProjects(project_name=project)

        return success, project

    def uploadDataset(self, project, dataset, filepath, issue_priority='Medium', **kwargs):
        """ Uploads a dataset to a project on the Labeling Tool.
            Updates its status in the database.

            Args:
                project [str]: the name of the project
                dataset [eurystheus.datastructures.Dataset]: the dataset
                filepath [str]: the path to the mp4 file containing the data
                    of the dataset

            Returns:
                success [bool]: if the api request completed successfully
        """
        logging.info('Uploading dataset `{}` (local path {}) to project `{}`'.format(
            str(dataset.name), filepath, project
        ))

        metadata = None
        today = date.today()
        
        success, profile_id = self.createProject(today.strftime("%d/%m/%Y"), 'eurystheus/labeling_profiles/imageSequence_Template_CVAT.json')
        dataset.profile = profile_id
         
        success, dataset_id, project_id = self.api.uploadDataset(
            project=today.strftime("%d/%m/%Y"),
            dataset=dataset,
            file_path=filepath
        )
         
        # If successful upload, update the status in the database
        if success:
            logging.info('Dataset {}: '.format(dataset))

            status = Status.UPLOADED


            projects = [project for project
                        in self.labeling_manager.get_projects()]

            for p in projects:
                if p.Name == today.strftime("%d/%m/%Y"):
                    project_id = p.ID

            values = {
                'Status': status,
                'DeepenID': dataset_id,
                'Project': project_id
            }

            success = self.labeling_manager.update_dataset(
                ID=dataset.name,
                values=values
            )

            # If still no success, create a new entry into the datasets table in the database
            # if not success:
            #     success = self.labeling_manager.create_dataset(
            #         project_id=project_id,
            #         dataset_id=dataset_id,
            #         status=status,
            #         filepath=filepath
            #     )

        # If successful upload, create a jira task and update the database.
        if success and self.USE_JIRA:
            logging.info('Dataset {}: Deepen ID = "{}"'.format(
                dataset.name, dataset.deepen_id
            ))

            status = Status.UPLOADED
            stage = self.jira.status_map[status]

            issue = Issue.factory(
                stage=stage,
                project=self.jira_project,
                dataset_name=dataset.name,
                dataset_type=dataset.type,
                dataset_link=self.api.getDatasetUiUrl(dataset.deepen_id, dataset.name),
                directions=dataset.directions,
                comments=dataset.comments,
                metadata=metadata,
                priority=issue_priority,
                num_frames=len(dataset.frames)
            )

            # Create a Jira task, and put it in the "Uploaded" Stage
            _, issue.key, _ = self.jira.create_issue(issue=issue)
            self.jira.set_issue_stage(issue=issue, stage=stage)

            # Update the database with the jira ticket information and new dataset status
            values = {
                'Frames': dataset.frames,
                'JIRAIssue': issue.key,
                'Status': status,
                'DeepenID': dataset.deepen_id
            }
            success = self.labeling_manager.update_dataset(
                ID=dataset.name,
                values=values
            )
        else:
            logging.error('Failed to upload dataset {} to the API tool. '
                          'has not been updated.'.format(dataset))

        return success

    def deleteDataset(self, dataset, dataset_id, error_msg=None, project=None, **kwargs):
        """ Deletes a dataset from the Labeling Tool and updates the database.

            Args:
                project [str]: the name of the project
                dataset [str]: the name of the dataset
                error_msg [str]: a traceback for any runtime errors or exceptions
                    that occur

            Returns:
                success [bool]: if the api request completed successfully
        """
        logging.info('Deleting dataset {}'.format(
            dataset
        ))

        success, response = self.api.deleteDataset(
            dataset=dataset
        )

        logging.info(response)

        # If successful, update the database
        if success:
            values = {
                'Status': Status.DELETED
            }
            if error_msg:
                values['ErrorMessage'] = str(error_msg)

            success = self.labeling_manager.update_dataset(
                ID=dataset_id,
                values=values
            )

            # If successful, update Jira
            if success and self.USE_JIRA:
                try:
                    status = Status.DELETED

                    issue = Issue.factory(
                        stage=self.jira.status_map[status],
                        key=self._getJiraIssueKey(dataset)
                    )

                    self.jira.set_issue_stage(
                        issue=issue,
                        stage=issue.stage
                    )
                except Exception as e:
                    logging.error('Unable to find Jira Issue associated with dataset.')
                    logging.error(e)

        return success

    def getDatasetStatus(self, dataset, project=None, **kwargs):
        """ Get a dataset's labeling status on the Labeling Tool.

            Args:
                project [str]: the name of the project
                dataset [str]: the name of the dataset

            Returns:
                success [bool]: if the api request completed successfully
                status [ouroboros.db.schemas.Status]: the status of the dataset

        """
        logging.info('Getting dataset status from Labeling Tool.')

        # Get the datasets status on the labeling tool
        success, status = self.api.getStatus(
            project=project,
            dataset=dataset
        )

        logging.info('Dataset {}: Status On Labeling Tool {}'.format(
            dataset, status
        ))

        # update jira and the db status for the dataset if the status is not
        # DELETED as that means there is an issue with the API
        stage = None

        if success and status != Status.DELETED and self.USE_JIRA:
            # Get the datasets status on Jira
            issue = Issue.factory(
                stage=self.jira.status_map[status],
                key=self._getJiraIssueKey(dataset)
            )
            stage = self.jira.get_issue_stage(issue)

            logging.info('Dataset {}: Status On Jira {}'.format(
                dataset, stage
            ))

            # Jira stage of "CANNOT COMPLETE" overrides labeling tool's status.
            if stage == Stage.CANNOTCOMPLETE.value:
                status = Status.CANNOTCOMPLETE
            # # Jira stage of "FAILED" overrides labeling tool's status.
            # LEAVING THIS HERE BECAUSE IT MAY BECOME NECESSARY AGAIN.
            # if stage == Stage.FAILED.value:
            #     status = Status.FAILED

        return success, status, stage

    def storeDatasetStatus(self, dataset_id, status, stage=None, project=None):
        """ Given a status for a dataset update the database

            Args:
                dataset_id [int]: the ID of the dataset
                status [ouroboros.db.schemas.Status]: the dataset's status

            Returns:
                None

        """
        # A dataset of status DELETED means there was an error in the labeling
        # API, we don't want to store this data.
        if status == Status.DELETED:
            logging.info('Task has been Deleted')
            return False

        # Get the Dataset's Status stored in the database
        try:
            stored_status = self.labeling_manager.get_dataset(dataset_id).Status
        except:
            return False

        logging.info('Dataset Status: {} -> {}'.format(stored_status, status))

        if stored_status == status:
            logging.info('Task has not changed status.')
            return True
        else:
            # update the database
            values = {
                'Status': status
            }
            _ = self.labeling_manager.update_dataset(
                ID=dataset_id,
                values=values
            )

        if self.USE_JIRA:
            try:
                # The labeling tool's stage overides the Jira task in all cases but
                # for manual columns (QAing)
                lab_tool_stage = self.jira.status_map[status]
                if lab_tool_stage.value != stage and stage != Stage.QAING:
                    logging.info('Transitioning Jira Task: {} -> {}.'.format(
                        stage, lab_tool_stage.value
                    ))
                    issue = Issue.factory(
                        stage=lab_tool_stage,
                        key=self._getJiraIssueKey(dataset)
                    )
                    self.jira.set_issue_stage(
                        issue=issue,
                        stage=issue.stage
                    )
            except Exception as e:
                logging.error('Unable to find Jira Issue associated with dataset.')
                logging.error(e)

    def getLabels(self, dataset_id=None, dataset_name=None, no_shape=False,
                  no_paint=False, frames=None, frame_points=None, raw=False,
                  project=None, dataset=None, **kwargs):
        """ Downloads the labels for a dataset

            Args:
                project [str]: the name of the project
                dataset [str]: the name of the dataset
                no_shape [bool]: Optional, if present no bounding boxes or
                    polygon labels will be retrieved
                no_paint [bool]: Optional, if present no paint labels will be
                    retrieved
                frames [list[str]]: the names of the frames in the dataset
                frame_points (list[int]): the number of points in each frame's
                    point cloud(s).
                raw [bool]: get the raw labels, do not parse the json

            Returns:
                success [bool]: if the api request completed successfully
                labels [list[eurystheus.datastructures.Label]]: A list of
                    shape labels in the dataset.
                paint_labels [eurystheus.datastructures.Label]: a list of
                    paint labels in the dataset.
                paint_categories [list[str]]: a list mapping the integers in
                    paint labels to label classes based on index of the class
                    in the list
        """

        logging.info('Downloading labels for dataset {}'.format(dataset))

        success, response = self.api.getLabels(
            project=project,
            dataset=dataset,
            format="PASCAL VOC 1.1"
        )

        return success, response

    def storeLabels(self, labels):
        """ Stores the labels in the database (for shape labels) and on disk
            (for paint labels)

            Args:
                labels [list[eurystheus.datastructures.Label]]: A list of
                    shape labels in the dataset.
                paint_labels [eurystheus.datastructures.Label]: a list of
                    paint labels in the dataset.
                paint_categories [list[str]]: a list mapping the integers in
                    paint labels to label classes based on index of the class
                    in the list

            Returns:
                None
        """

        bbox_labels = [json.loads(labels.txt)]

        try:
            bbox_2d_success = self.labeling_manager.upload_2D_box_labels(bbox_labels)
        except Exception as e:
            bbox_2d_success = False
            logging.error('Unable to store 2d bounding box labels')
            logging.error(e)

        return bbox_2d_success

    def getUnlabeledDatasets(self, lens_override=None,
                             type_override=None, status_override="unlabeled",
                             project_override=None, filter=None, **kwargs):
        """ Gets all the datasets that are ready to upload from the db

            Args:
                lens_override (list[int]): Optional, if present will only
                    get unlabeled datasets from the specified lenses
                type_override (list[str]): Optional, if present will only
                    get unlabeled datasets of the specified types
                status_override (ouroboros.db.schema.Status(UNLABELED|REDO)):
                    Optional, if present will only get unlabeled datasets of
                    the specified status (Redo or Unlabeled)
                filter (str): A filter string of format key=val. If provided
                    will be used to filter the unlabeled datasets.
                    Note: val can contain * as a wildcard at the begining or
                        end of the value or both (eg startswith, endswith,
                        contains).

            Returns:
                datasets (list[eurystheus.datastructures.Dataset]): A list of
                    unlabeled datasets
                invalid_datasets(list[tuple]): A list of datasets with invalid
                    data that were not created.

        """
        logging.info('Retrieving unlabeled datasets from Ouroboros.')

        if type_override is not None:
            enum_map = {
                '2dbbox': LabelType.BOXLABEL,
                '2dseg': LabelType.SEGLABEL,
                '3dbbox': LabelType.POINTCLOUDBOXLABEL,
                '3dseg': LabelType.POINTCLOUDSEGLABEL
            }
            type_override = [enum_map[_type] for _type in type_override]

        status_map = {
            'unlabeled': (Status.UNLABELED, None),  # unlabeled does not need the invalidate arg
            'redo': (Status.REDO, True),  # redo invalidates the datasets that are being relabeled
            'relabel': (Status.REDO, False)  # relabel does not invalidate the datasets that are being relabeled
        }

        status_override, invalidate = status_map[status_override]

        unlabeled_datasets = []
        if status_override == Status.UNLABELED:
            unlabeled_datasets += self.labeling_manager.get_unlabeled(
                lenses=lens_override,
                types=type_override
            )
        
        if status_override == Status.REDO:
            unlabeled_datasets += self.labeling_manager.get_redo(
                lenses=lens_override,
                types=type_override,
                project=project_override,
                invalidate=invalidate
            )

        logging.info('Found {} TOTAL dataset(s).'.format(
            len(unlabeled_datasets)
        ))

        if filter:
            unlabeled_datasets = Filter.factory(filter).apply(unlabeled_datasets)

        logging.info('Found {} FILTERED dataset(s).'.format(
            len(unlabeled_datasets)
        ))

        datasets = []
        invalid_datasets = []
        for unlabeled_dataset in unlabeled_datasets:
            try:
                dataset = Dataset.factory(
                    dataset_type=unlabeled_dataset['type'],
                    **unlabeled_dataset)
                datasets.append(dataset)
            except Exception as e:
                logging.error('Invalid Dataset.')
                tb = traceback.format_exc()
                logging.error(tb)
                invalid_datasets.append((unlabeled_dataset, e))
                self._setDatasetStatus(
                    unlabeled_dataset['ID'], Status.INVALID, tb
                )
                continue

        return datasets, invalid_datasets

    def handleCompletedDatasets(self, project_override, **kwargs):
        """ FIRST STAGE OF CHRON JOB:
                i) Check the status for all datasets on the labeling tool that are
                    in active projects
                ii) Get labels from completed datasets
                iii) Delete the completed datasets whose labels have been downloaded

            Note: Other projects can be on the labling tool and managed manually.
                The only projects that this tool will check are ones that are
                in the database and marked as active

            Args:
                project_override (str): Optional, if present will only download
                    labels and upload datasets to the specified project.

            Returns:
                full_success [bool]: True if the entire job encountered no
                    errors.
        """
        full_success = True

        if project_override is None:
            projects = [project for project
                        in self.labeling_manager.get_projects()
                        if project.Active]
        else:
            projects = [project for project
                        in self.labeling_manager.get_projects()
                        if project.ID in project_override]

        # Check all active projects on the labeling tool
        logging.info('Checking for completed datasets in projects')
        datasets = self.labeling_manager.get_datasets_by_project(
            [project.ID for project in projects],
            [Status.UPLOADED, Status.LABELING, Status.QA, Status.LABELED, Status.FAILED, Status.CANNOTCOMPLETE]
        )

        d_len = len(datasets)
        if datasets:
            # Get the status of each dataset
            for d_index, dataset in enumerate(datasets):
                logging.info('\nPROCESSING DATASET {} of {}\n'.format(
                    d_index, d_len
                ))

                dataset_id = dataset.ID
                print(dataset_id)
                cvat_id = dataset.DeepenID
                project_id = dataset.Project
                project = [project.Name for project in projects
                           if project.ID == dataset.Project][0]

                success, status, stage = self.getDatasetStatus(
                    dataset=cvat_id
                )

                type_status = ["completed", "annotation"]
                if status == type_status[0]:
                    status = Status.LABELED
                if status == type_status[1]:
                    status = Status.UPLOADED

                _ = self.storeDatasetStatus(
                    dataset_id=dataset_id,
                    status=status,
                    stage=stage,
                    project=project
                )

                full_success = full_success & success
                if success:
                    # If the project has been labeled. Download the labels
                    # and delete the dataset
                    if status == Status.LABELED:
                        no_paint = dataset.Type != LabelType.POINTCLOUDSEGLABEL
                        success, labels = self.getLabels(
                                dataset=cvat_id,
                                project=project_id
                            )
                        full_success = full_success & success
                        
                        '''
                        if success:
                            success = self.storeLabels(
                                labels=labels
                            )

                            full_success = full_success & success
                        '''
                        if success:
                            success, response = self.deleteDataset(
                                dataset=cvat_id,
                                dataset_id=dataset_id
                            )
                            full_success = full_success & success

        return full_success

    def handleUnlabeledDataset(self, issue_priority, project_override,
                               lens_override, type_override, status_override,
                               labeler_override, max_points, stereo_pointcloud,
                               no_imu, filter=None, limit=None,
                               max_uploaded=None, **kwargs):
        """  SECOND STAGE OF run():
                i) Get all unlabeled datasets from Ourobros
                ---- While there is more storage space on the labeling tool
                ii) Prepare unlabeled datasets
                iii) Upload the new datasets to the labeling tool
                iv) Assign the dataset to a labeler

            Args:
                issue_priority (str): Optional, if present will set the jira
                    issue to the defined priority.
                project_override (str): Optional, if present will only download
                    labels and upload datasets to the specified project.
                lens_override (list[int]): Optional, if present will only
                    upload datasets of the specified lens.
                type_override (list[str]): Optional, if present will only
                    upload datasets of the specified type.
                status_override (str): Optional, if present will only get
                    unlabeled datasets of the specified status (redo or unlabeled)
                labeler_override (list[int]): Optional, if present will only
                    assign tasks to the specified users.
                max_points (int): Option, if present will limit the number of
                    points in a pointcloud to upload in a dataset.
                stereo_pointcloud (bool): Option, if present also export and
                    upload the stereo point cloud (along with the LiDAR point
                    cloud) to the labeling tool.
                no_imu (bool): If True, position data will not be included
                    or used in the extracted dataset.
                filter (str): A filter string of format key=val. If provided
                    will be used to filter the unlabeled datasets.
                    Note: val can contain * as a wildcard at the begining or
                        end of the value or both (eg startswith, endswith,
                        contains).
                limit (int): The maximum number of datasets to attempt to
                    upload. Use this arg to upload a specific number of
                    datasets.
                max_uploaded (int): The maximum number of datasets to have on
                    the labeling tool in status "NEEDS LABELING". Use this arg
                    to keep the number of available datasets for the labelers
                    at a given number.

            Returns:
                full_success [bool]: True if the entire job encountered no
                    errors.
        """
        logging.info('Uploading new datasets to labeling tool')

        full_success = True

        unlabeled_datasets, invalid_datasets = self.getUnlabeledDatasets(
            lens_override=lens_override,
            type_override=type_override,
            status_override=status_override,
            project_override=project_override,
            filter=filter
        )

        if max_uploaded is not None:
            existing_datasets = self.labeling_manager.get_datasets_of_status(
                Status.UPLOADED
            )
            num_existing_datasets = len(existing_datasets)

            deficit = max_uploaded - num_existing_datasets

            if deficit <= 0:
                logging.info(f'{num_existing_datasets} datasets exist in '
                             f'status "{Status.UPLOADED}" on the labeling '
                             'tool, equaling or exeeding the '
                             f'maximum limit of {max_uploaded}')
                return True

            unlabeled_datasets = random.sample(
                unlabeled_datasets,
                min(deficit, len(unlabeled_datasets))
            )

        if limit is not None:
            # Sample randomly so a diverse set of datasets is selected, this is
            # important if uploading different types of datasets (seg, bbox)
            unlabeled_datasets = random.sample(
                unlabeled_datasets,
                min(limit, len(unlabeled_datasets))
            )

        # Used to confirm that the datasets to upload are appropriate
        if kwargs.get('confirm', False):
            import pdb
            pdb.set_trace()

        unexportable_datasets = []
        d_len = len(unlabeled_datasets)

        for count, unlabeled_dataset in enumerate(unlabeled_datasets):
            logging.info('\n\nUploading dataset {} [{} of {}]'.format(
                unlabeled_dataset.name, count, d_len
            ))

            with WorkDirectory(unlabeled_dataset.name) as work_directory:
                try:
                    '''
                    dataset_path = self.api.exportDataset(
                        work_directory=work_directory,
                        dataset=unlabeled_dataset,
                        no_imu=no_imu,
                        max_points_to_write=max_points,
                        export_stereo_pointcloud=stereo_pointcloud
                    )
                    '''
                    dataset_path = '/tmp/eurystheus/data/test/'
                    
                  
                except Exception as e:
                    logging.error('Unable to export dataset.')
                    tb = traceback.format_exc()
                    logging.error(tb)
                    unexportable_datasets.append((
                        unlabeled_dataset,
                        work_directory.getWorkDirectoryPath(),
                        e
                    ))
                    self._setDatasetStatus(
                        unlabeled_dataset.name, Status.INVALID, tb
                    )
                    continue
                
                success = self.uploadDataset(
                    project=unlabeled_dataset.project_name,
                    dataset=unlabeled_dataset,
                    filepath=dataset_path,
                    issue_priority=issue_priority
                )
                full_success = full_success & success

        if len(invalid_datasets) > 0:
            logging.warning("Datasets with invalid metadata:")
        for invalid_dataset in invalid_datasets:
            logging.warning(invalid_dataset)

        if len(unexportable_datasets) > 0:
            logging.warning("Datasets that were unable to be exported:")
        for unexported_dataset in unexportable_datasets:
            logging.warning(unexported_dataset)

        return full_success


    def run(self, **kwargs):
        """ The main process of Eurystheus. Handles, the creation and deletion
            of datasets as well as assigning labeling tasks and downloading the
            labels of completed datasets.

            Args:
                See: handleCompletedDatasets and handleUnlabeledDataset

            Returns:
                full_success [bool]: True if the entire job encountered no
                    errors.
        """
        completed_datasets_success = self.handleCompletedDatasets(**kwargs)

        unlabeled_datasets_success = self.handleUnlabeledDataset(**kwargs)

        return completed_datasets_success and unlabeled_datasets_success

    def _getJiraIssueKey(self, dataset):
        """ Given a dataset id (name in labeling tool), return the
            corresponding Jira Issue Key

            Args:
                dataset [str]: the name of the dataset

            Returns:
                jira_issue [str]: the key to the Jira Issue associated with the
                dataset
        """
        return self.labeling_manager.get_dataset(
            ID=dataset
        ).JIRAIssue

    def _setDatasetStatus(self, dataset, status, error_msg=None):
        """ Updates the dataset status in the database.

            Args:
                dataset [str]: the name of the dataset
                status [ouroboros.db.schemas.Status]: the dataset's status
                error_msg [str]: Optional, traceback of any errors or
                    exceptions that ocurred

            Returns:
                success [bool]: True if the database was successfully updated
        """
        values = {'Status': status}
        if error_msg:
            values['ErrorMessage'] = str(error_msg)
        return self.labeling_manager.update_dataset(
            ID=dataset,
            values=values
        )
