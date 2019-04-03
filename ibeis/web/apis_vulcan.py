# -*- coding: utf-8 -*-
"""Dependencies: flask, tornado."""
from __future__ import absolute_import, division, print_function
from ibeis.control import controller_inject
from flask_swagger import swagger
from flask import current_app
from flask import jsonify
import utool as ut
import uuid


CLASS_INJECT_KEY, register_ibs_method = (
    controller_inject.make_ibs_register_decorator(__name__))

PREFIX         = controller_inject.VULCAN_API_PREFIX
register_api   = controller_inject.get_ibeis_flask_api(__name__)
register_route = controller_inject.get_ibeis_flask_route(__name__)


def _prefix(route=''):
    rule = '/%s/%s/' % (PREFIX, route, )
    while '//' in rule:
        rule = rule.replace('//', '/')
    return rule


def _image(ibs, gid):
    return {
        'uuid': str(ibs.get_image_uuids(gid)),
    }


def _task(ibs, taskid):
    return {
        'uuid': taskid,
    }


@register_route(_prefix('swagger'), methods=['GET'])
def vulcan_core_specification_swagger(*args, **kwargs):
    r"""
    Returns the API specification in the Swagger 2.0 (OpenAPI) JSON format.

    The Swagger API specification (https://github.com/OAI/OpenAPI-Specification/blob/master/versions/2.0.md) provides a standardized method to export REST API documentation and examples.  Our documentation is built on-demand with the help of the Python package flask-swagger (https://github.com/gangverk/flask-swagger).

    The API specification includes GET, POST, PUT, and DELETE methods and Model definitions.
    ---
    definitions:
    - schema:
        id: Image
        description: An Image is a semantic construct that represents an uploaded image.  Images can be uploaded for later processing or be used immediately for inference with the pipeline.
        required:
          - uuid
        properties:
          uuid:
            description: a deterministically-derived UUID based on the image pixels, which can be used to identify duplicate Images.
            type: string
            format: uuid
    - schema:
        id: Task
        description: A Task is a semantic construct that represents a background task in an asynchronous call.  A Task has an optional callback on completion or the status (and result) can be checked via the API
        required:
          - uuid
        properties:
          uuid:
            description: a random UUID to identify a given asynchronous call, used to check status and results of a background task
            type: string
            format: uuid
    produces:
    - application/json
    responses:
        200:
          description: Returns the Swagger 2.0 JSON format
    """
    swag = swagger(current_app)
    swag['info']['title'] = 'Wild Me - Vulcan Project'
    swag['info']['description'] = 'Documentation for all REST API endpoints provided by Wild Me for the Vulcan collaboration'
    swag['info']['version'] = 'v0.1'
    swag['info']['contact'] = {
        'name':  'Wild Me Developers',
        'url':   'http://wildme.org',
        'email': 'dev@wildme.org',
    }
    swag['info']['license'] = {
        'name': 'Apache 2.0',
        'url':  'http://www.apache.org/licenses/LICENSE-2.0.html'
    }
    swag['host'] = 'kaiju.dyn.wildme.io:5000'
    swag['schemes'] = [
        'http',
    ]

    response = jsonify(swag)
    return response


@register_api(_prefix('status'), methods=['GET'], __api_plural_check__=False)
def vulcan_core_status(*args, **kwargs):
    r"""
    Returns the health status of the API back-end; optionally can be used as a service heatbeat.
    ---
    produces:
    - application/json
    responses:
      200:
        description: Returns the status of the server
        schema:
          type: object
          properties:
            status:
              type: string
              enum:
              - healthy
              - warning
              - critical
        examples:
        - application/json:
            status: healthy
    """
    status = 'healthy'
    return {'status': status}


@register_api(_prefix('image'), methods=['POST'])
def vulcan_image_upload(ibs, return_time=False, *args, **kwargs):
    r"""
    Upload an image for future processing.

    ---
    parameters:
    - name: image
      in: formData
      description: The image to upload.
      required: true
      type: file
      enum:
      - image/png
      - image/jpg
      - image/tiff
    produces:
    - application/json
    responses:
      200:
        description: Returns an Image model with an ID
        schema:
          $ref: '#/definitions/Image'
      400:
        description: Invalid input parameter
      415:
        description: Unsupported media type in the request body. Currently only image/png, image/jpeg, image/tiff are supported.
    """
    with ut.Timer('Uploading') as time_upload:
        from ibeis.web.apis import image_upload
        try:
            gid = image_upload(cleanup=True, **kwargs)
            assert gid is not None
        except controller_inject.WebException:
            raise
        except:
            raise controller_inject.WebInvalidInput('Uploaded image is corrupted or is an unsupported file format (supported: image/png, image/jpeg, image/tiff)', 'image', image=True)
        image = _image(ibs, gid)
    if return_time:
        return image, time_upload
    else:
        return image


@register_ibs_method
def vulcan_pipeline(ibs, images,
                    testing=False,
                    quick=True,
                    __jobid__=None,
                    time_upload=None,
                    *args, **kwargs):

    def _timer(*args):
        time = 0.0
        for timer in args:
            if timer is not None:
                time += timer.ellapsed
        return time

    try:
        with ut.Timer('Config') as time_config:
            include_grid2 = not quick

            wic_classifier_algo     = 'densenet'
            loc_classifier_algo     = '%s+lightnet' % (wic_classifier_algo, )
            loc_all_classifier_algo = '%s+lightnet!' % (wic_classifier_algo, )
            agg_classifier_algo     = 'tile_aggregation_quick' if quick else 'tile_aggregation'
            # wic_model_tag           = 'vulcan-d3e8bf43-boost2'
            wic_model_tag           = 'vulcan-d3e8bf43-boost2:3'
            loc_model_tag           = 'vulcan_v0'
            wic_sensitivity         = 0.347
            loc_sensitivity         = 0.151
            loc_nms                 = 0.5

        with ut.Timer('UUIDs') as time_uuid:
            uuid_list = [
                uuid.UUID(image['uuid'])
                for image in images
            ]
            gid_list = ibs.get_image_gids_from_uuid(uuid_list)

        with ut.Timer('Test Deleting') as time_test:
            if testing:
                print('TESTING')
                tile_list = ibs.vulcan_get_valid_tile_rowids(gid_list=gid_list)
                flag_list = [tile for tile in tile_list if tile is not None]
                tile_list = ut.compress(tile_list, flag_list)
                ibs.depc_image.delete_property_all('tiles', gid_list)
                ibs.depc_image.delete_root(gid_list)
                ibs.delete_images(tile_list, trash_images=False)
            else:
                print('NOT TESTING')

        with ut.Timer('Tiling') as time_tile:
            # Pre-compute tiles
            tile_list = ibs.vulcan_get_valid_tile_rowids(gid_list=gid_list, include_grid2=include_grid2)
            ancestor_gid_list = ibs.get_vulcan_image_tile_ancestor_gids(tile_list)

        with ut.Timer('WIC') as time_wic:
            model_tag           = wic_model_tag
            wic_confidence_list = ibs.vulcan_wic_test(tile_list, classifier_algo=wic_classifier_algo, model_tag=model_tag)
            wic_flag_list       = [wic_confidence >= wic_sensitivity for wic_confidence in wic_confidence_list]  # NOQA

        with ut.Timer('LOC All') as time_loc_all:
            model_tag           = '%s,%0.03f,%s,%0.02f' % (wic_model_tag, wic_sensitivity, loc_model_tag, loc_nms, )
            loc_confidence_list = ibs.vulcan_wic_test(tile_list, classifier_algo=loc_all_classifier_algo, model_tag=model_tag)
            loc_flag_list       = [loc_confidence >= loc_sensitivity for loc_confidence in loc_confidence_list]  # NOQA

        with ut.Timer('LOC Filtered') as time_loc_filtered:
            model_tag           = '%s,%0.03f,%s,%0.02f' % (wic_model_tag, wic_sensitivity, loc_model_tag, loc_nms, )
            loc_confidence_list = ibs.vulcan_wic_test(tile_list, classifier_algo=loc_classifier_algo, model_tag=model_tag)
            loc_flag_list       = [loc_confidence >= loc_sensitivity for loc_confidence in loc_confidence_list]  # NOQA

            location_dict = {}
            for ancestor_gid, tile, loc_flag in zip(ancestor_gid_list, tile_list, loc_flag_list):
                if ancestor_gid not in location_dict:
                    location_dict[ancestor_gid] = []
                if loc_flag:
                    location_dict[ancestor_gid].append(tile)
            locations_list = [
                ibs.get_vulcan_image_tile_bboxes(location_dict.get(gid, []))
                for gid in gid_list
            ]

        with ut.Timer('Aggregate') as time_agg:
            model_tag           = '%s;%s,%0.03f,%s,%0.02f' % (loc_classifier_algo, wic_model_tag, wic_sensitivity, loc_model_tag, loc_nms)
            agg_confidence_list = ibs.vulcan_wic_test(gid_list, classifier_algo=agg_classifier_algo, model_tag=model_tag)
            agg_flag_list       = [agg_confidence >= loc_sensitivity for agg_confidence in agg_confidence_list]
    except:
        raise controller_inject.WebException('The Vulcan pipeline process has failed for an unknown reason')

    response = {
        'results': [
            {
                'score': confidence,
                'flag':  flag,
                'tiles': location_list,
            }
            for confidence, flag, location_list in zip(agg_confidence_list, agg_flag_list, locations_list)
        ],
        'times': {
            '_test'            : _timer(time_test),
            '_loc_all'         : _timer(time_loc_all),
            'step_0upload'     : _timer(time_upload),
            'step_1_uuid'      : _timer(time_uuid),
            'step_2_tile'      : _timer(time_tile),
            'step_3_wic'       : _timer(time_wic),
            'step_4_loc'       : _timer(time_loc_filtered),
            'step_5_aggregate' : _timer(time_agg),
            'inference'        : _timer(time_wic, time_loc_filtered),
            'overhead'         : _timer(time_upload, time_config, time_uuid, time_tile, time_agg),
            'total'            : _timer(time_upload, time_config, time_uuid, time_tile, time_wic, time_loc_filtered, time_agg),
        },
    }

    return response


@register_api(_prefix('pipeline'), methods=['POST'])
def vulcan_pipeline_upload(ibs, *_args, **kwargs):
    r"""
    Returns the results for an uploaded image and a provided model configuration.
    ---
    parameters:
    - name: image
      in: formData
      description: The image to process with the pipeline.
      required: true
      type: file
      enum:
      - image/png
      - image/jpg
      - image/tiff
    responses:
      200:
        description: Returns an array of results on the uploaded image
      400:
        description: Invalid input parameter
      415:
        description: Unsupported media type in the request body. Currently only image/png, image/jpeg, image/tiff are supported.
    """
    ibs = current_app.ibs

    # Input argument validation
    image, time_upload = vulcan_image_upload(ibs, return_time=True)
    images = [image]
    args = (images, )
    response = vulcan_pipeline(ibs, *args, time_upload=time_upload, **kwargs)
    return response


@register_api(_prefix('pipeline/batch'), methods=['POST'])
def vulcan_pipeline_batch(ibs, images, async=True,
                           callback_url=None, callback_method=None,
                           *_args, **kwargs):
    r"""
    The asynchronous variant of POST 'pipeline' that takes in a list of Image models and returns a task ID.

    It may be more ideal for a particular application to upload many images at one time and perform processing later in a large batch.  This type of batch API call is more efficient because the pipeline on GPU can process more images in parallel.  However, if you intend to run the pipeline on an upload as quickly as possible, please use the on-demand, non-batched API.
    ---
    parameters:
    - name: images
      in: body
      description: A JSON list of Image models to process with the pipeline.
      required: true
      type: array
      items:
        $ref: '#/definitions/Image'
    - name: callback_url
      in: body
      description: The URL of where to callback when the task is completed, must be a fully resolvable address and accessible.  The callback will include a 'body' parameter called `task` which will provide a Task model
      required: false
      type: string
      format: url
    - name: callback_method
      in: body
      description: The HTTP method for which to make the callback
      required: false
      default: post
      type: string
      enum:
      - get
      - post
      - put
      - delete
    responses:
      200:
        description: Returns a Task model
        schema:
          $ref: "#/definitions/Task"
      400:
        description: Invalid input parameter
      x-task-response:
        description: The task returns an array of arrays of results, in parallel lists with the provided Image models
    """
    ibs = current_app.ibs

    # Input argument validation
    for index, image in enumerate(images):
        try:
            parameter = 'images:%d' % (index, )
            assert 'uuid' in image, 'Image Model provided is invalid, missing UUID key'
        except AssertionError as ex:
            raise controller_inject.WebInvalidInput(str(ex), parameter)

    try:
        parameter = 'async'
        assert isinstance(async, bool), 'Asynchronous flag must be a boolean'

        parameter = 'callback_url'
        assert callback_url is None or isinstance(callback_url, str), 'Callback URL must be a string'
        if callback_url is not None:
            assert callback_url.startswith('http://') or callback_url.startswith('https://'), 'Callback URL must start with http:// or https://'

        parameter = 'callback_method'
        assert callback_method is None or isinstance(callback_method, str), 'Callback URL must be a string'
        if callback_method is not None:
            callback_method = callback_method.lower()
            assert callback_method in ['get', 'post', 'put', 'delete'], 'Unsupported callback method, must be one of ("get", "post", "put", "delete")'
    except AssertionError as ex:
        raise controller_inject.WebInvalidInput(str(ex), parameter)

    args = (images, )
    if async:
        taskid = ibs.job_manager.jobiface.queue_job('vulcan_pipeline',
                                                    callback_url, callback_method,
                                                    *args, **kwargs)
        response = _task(ibs, taskid)
    else:
        response = ibs.vulcan_pipeline(*args, **kwargs)

    return response


@register_api(_prefix('task'), methods=['GET'])
def vulcan_task_status(ibs, task):
    r"""
    Check the status of an asynchronous Task.

    A Task is an asynchronous task that was launched as a background process with an optional callback.  The status of a given Task with a UUID can be checked with this call.  The status of the call depends on where in the execution queue the Task is currently, which will be processed in a first-come-first-serve list and only one Task at a time to present atomicity of the API.

    The status can be one of the following:
    - received   -> The Task request was received but has not passed any input validation.
    - accepted   -> The Task request has passed basic input validation and will be queued soon for execution.
    - queued     -> The Task is queued in the execution list and will be processed in order and one at a time.
    - working    -> The Task is being processed, awaiting completed results or an error exception
    - publishing -> The Task is publishing the results of the background API call.
    - completed  -> One of two end states: the Task is complete, completed results available for downloading with the REST API.
    - exception  -> One of two end states: the Task has encountered an error, an error message can be received using the results REST API.
    - unknown    -> The Task you asked for is not known, indicating that the either UUID is not recognized (i.e. a Task with that ID was never currently created) or the server has been restarted.

    **Important: when the API server is restarted, all queued and running background Tasks are killed and all Task requests and cached results are deleted.**
    ---
    parameters:
    - name: task
      in: body
      description: A Task model
      required: true
      schema:
        $ref: "#/definitions/Task"
    responses:
      200:
        description: Returns the status of the provided Task
        schema:
          type: object
          properties:
            status:
              type: string
              enum:
              - received
              - accepted
              - queued
              - working
              - publishing
              - completed
              - exception
              - unknown
      400:
        description: Invalid input parameter
    """
    ibs = current_app.ibs

    # Input argument validation
    try:
        parameter = 'task'
        assert 'uuid' in task, 'Task Model provided is invalid, missing UUID key'
    except AssertionError as ex:
        raise controller_inject.WebInvalidInput(str(ex), parameter)

    uuid_ = task.get('uuid', None)
    assert uuid_ is not None
    status = ibs.get_job_status(uuid_)
    status = status.get('jobstatus', None)
    return {'status': status}


@register_api(_prefix('task'), methods=['POST'])
def vulcan_task_result(ibs, task):
    r"""
    Retrieve the result of a completed asynchronous Task
    ---
    parameters:
    - name: task
      in: body
      description: A Task model
      required: true
      schema:
        $ref: "#/definitions/Task"
    responses:
      200:
        description: Returns the result of the provided Task
      400:
        description: Invalid input parameter
    """
    ibs = current_app.ibs

    # Input argument validation
    try:
        parameter = 'task'
        assert 'uuid' in task, 'Task Model provided is invalid, missing UUID key'
    except AssertionError as ex:
        raise controller_inject.WebInvalidInput(str(ex), parameter)

    uuid_ = task.get('uuid', None)
    assert uuid_ is not None
    result = ibs.get_job_result(uuid_)
    result = result.get('json_result', None)
    return result


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.web.app
        python -m ibeis.web.app --allexamples
        python -m ibeis.web.app --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
