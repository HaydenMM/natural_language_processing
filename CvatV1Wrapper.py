"""
 Copyright 2021 Carnegie Mellon University. All rights reserved.

 This notice must appear in all copies of this file and its derivatives

"""

import json
import logging
import os

import requests

from eurystheus.api.cvat.CvatBase import CvatBase
from eurystheus.utils import helperfuns


class CvatV1Wrapper(CvatBase):
    """ Wrapper for CVAT's version 1 API.
    """

    def __init__(self, username, password, url):
        """
           Args:
               username (str): Username for CVAT api.
               password (str): Password for CVAT api.
               url (str): URL to CVAT server.

        """
        super().__init__(username, password, url)

    def createProject(self,
                      project_name,
                      config):
        """ Creates a project in CVAT based on NREC's labeling_profiles/project_definitions

            Args:
                project_name [str]: the name of the project (must be unique)
                labels [dict]: a dictionary defining the labels configuration of the project

            Returns:
                success [bool]: if the api request completed successfully response
                [Response]: the HTML response from CVAT
        """
        logging.info('Creating project "{}"'.format(project_name))

        url = self._URL + '/projects/{}'.format(project_name)
        logging.info('POST: {}'.format(url))
        with open(config, 'r') as config_file:
            config_dict = json.load(config_file)
        config = json.loads(json.dumps(config_dict))
        response = requests.post(self._URL + "/projects",
                                 json=config,
                                 cookies=self._COOKIES,
                                 headers=self._HEADERS)

        success, response = self._handleResponse(response)
        project_id = json.loads(response.text)['id']

        return success, project_id

    def getProjects(self,
                    project_name=None):
        """ Gets existing projects from CVAT.

            Args:
                project_name (str): If specified will only return the specified
                    project.

            Returns:
                success (bool): if the api request completed successfully.
                projects (list[dict]): A list of dictionaries each representing a project on CVAT.
        """
        logging.info('GET: {}'.format(self._URL + "/projects"))

        response = requests.get(self._URL + "/projects",
                                cookies=self._COOKIES)
        success, response = self._handleResponse(response)
        projects = response.json()['results']

        return success, projects

    def _createDataset(self,
                       project_name,
                       dataset_name,
                       file_path):
        """ Creates a project in CVAT.

            Args:
                project_name [str]: the name of the project (must be unique)
                dataset_name [str]: the name of the dataset
                filepath [str]: the path to the zip file containing the data of the dataset

            Returns:
                success [bool]: if the api request completed successfully
                response [Response]: the HTML response from CVAT
        """
        size = helperfuns.getBytes(file_path)

        success, id = self._getProjectID(project_name)

        if success:
            # On success (project name found) create an empty Task on a specific Project
            task_json = {
                'name': dataset_name,
                'project_id': id,
                'size': size
            }

            logging.info('POST: {}'.format(self._URL + "/tasks"))

            response = requests.post(self._URL + "/tasks",
                                     json=task_json,
                                     cookies=self._COOKIES,
                                     headers=self._HEADERS)

            success, response = self._handleResponse(response)
            return success, response
        else:
            return success, None

    def uploadDataset(self,
                      project,
                      dataset,
                      file_path):
        """ Creates a new dataset within a project and uploads data to it.

            Args:
                project_name [str]: the name of the project the dataset is part of.
                dataset [eurystheus.datastructures.Dataset]: the dataset
                file_path [str]: the path to the file containing the data of the dataset

            Returns:
                success [bool]: if the api request completed successfully
                response [Response]: the HTML response from CVAT
        """

        success, response = self._createDataset(project_name=project,
                                                dataset_name=dataset.name,
                                                file_path=file_path)

        if success:
            # On success, post the dataset to the new task
            task_id = response.json()['id']

            logging.info('POST: {}'.format(self._URL + "/tasks/data"))
            files = [file_path + f for f in os.listdir(file_path) if os.path.isfile(file_path + f)]
            #files = [file_path]
            client_files = {'client_files[{}]'.format(i): open(f, 'rb') for i, f in enumerate(files)}

            data_json = {
                "image_quality": 50
            }

            response = requests.post(self._URL + '/tasks/' + str(task_id) + '/data',
                                     data=data_json,
                                     files=client_files,
                                     cookies=self._COOKIES,
                                     headers=self._HEADERS)

            success, response = self._handleResponse(response)
            success, project_id = self._getProjectID(project)

        return success, task_id, project_id

    def getStatus(self,
                  project,
                  dataset):
        """ Get the status of a specifc task

                    Args:
                        project_name [str]: the name of the project the dataset is part of.
                        dataset_id [str]: the ID of the dataset

                    Returns:
                        success [bool]: if the api request completed successfully
                        response [Response]: the HTML response from CVAT
        """

        logging.info('GET: {}'.format(self._URL + "/tasks"))

        response = requests.get(self._URL + "/tasks/" + str(dataset),
                                    cookies=self._COOKIES,
                                    headers=self._HEADERS)

        success, response = self._handleResponse(response)

        if success:
            return success, response.json()['status']
        else:
            return success, None


    def getLabels(self,
                  project,
                  dataset,
                  format="PASCAL VOC 1.1"):
        """ Get the completed labels from a specific task

            Args:
                project_name [str]: the name of the project the dataset is part of.
                dataset_name [str]: the name of the dataset
                format [str]: the format to export the labels

            Returns:
                success [bool]: if the api request completed successfully
                response [Response]: the HTML response from CVAT
        """

        logging.info('GET: {}'.format(self._URL + "/task/" + str(dataset) + "/annotations"))

        json_data = {
            "filename": str(project) + "_" + str(dataset),
            "format": format,
            "action": "download"
        }

        response = requests.get(self._URL + "/tasks/" + str(dataset) + "/annotations",
                                json=json_data,
                                cookies=self._COOKIES)

        print(response.content)
        success, response = self._handleResponse(response)

        # NEED TO ADD CVAT_PARSER

        return success, response

    def deleteDataset(self,
                      dataset):
        """ Delete a dataset from a project.

            Args:
                project_name [str]: the name of the project the dataset is part of.
                dataset_name [str]: the name of the dataset

            Returns:
                success [bool]: if the api request completed successfully
                response [Response]: the HTML response from CVAT
        """



        response = requests.delete(self._URL + "/tasks/" + str(dataset),
                                   cookies=self._COOKIES,
                                   headers=self._HEADERS)
        print(response)
        success, response = self._handleResponse(response)
        print(response)
        if success:
            return success, dataset
        else:
            return success, None

    def getFreeSpace(self):
        """ Get the details of the storage for CVAT

                   Returns:
                       success [bool]: if the api request completed successfully
                       size [int]: the HTML response from CVAT
        """
        success = True

        size = sum(d.stat().st_size for d in os.scandir("/var/lib/docker/volumes/cvat_cvat_data/_data/") if d.is_file())
        free_space = self._STORAGE_CAPACITY - size

        if free_space < 0:
            success = False
            free_space = 0

        return success, free_space

    def _getProjectID(self,
                      project_name):
        """ gets a specific project's ID:

            Args:
                project_name [str]: the the name of the project

            Returns:
                (str): the project id
        """

        success, projects = self.getProjects()

        if success:
            try:
                for p in projects:
                    if p['name'] == project_name:
                        id = p['id']
                        return True, id
            except Exception as e:
                logging.error(e)
                return False, None

    def _getTaskID(self,
                   project,
                   dataset):
        """ gets a specific task's ID:

            Args:
                project [str]: the name of the project
                dataset [str]: the name of the task

            Returns:
                (str): the project/task id
        """

        success, projects = self.getProjects()

        if success:
            try:
                for p in projects:
                    if p['name'] == project:
                        for t in p['tasks']:
                            if t['name'] == dataset:
                                id = t['id']
                                return True, id
            except Exception as e:
                logging.error(e)
                return False, None

    def _buildUrl(self,
                  task_id):
        """ Builds the basic url for a task in CVAT:

            Args:
                task_id [str]: the id for the specific task that was created

            Returns:
                (str): the URL to the specific task
        """
        return self._URL + '/tasks/' + task_id
