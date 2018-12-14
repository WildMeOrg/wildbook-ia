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

PREFIX         = controller_inject.MICROSOFT_API_PREFIX
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


def _annotation(ibs, aid):
    return {
        'uuid': str(ibs.get_annot_uuids(aid)),
    }


def _detection(ibs, aid):
    gid = ibs.get_annot_gids(aid)
    bbox = ibs.get_annot_bboxes(aid)
    return {
        '_image'      : _image(ibs, gid),
        '_annotation' : _annotation(ibs, aid),
        'xtl'         : bbox[0],
        'ytl'         : bbox[1],
        'width'       : bbox[2],
        'height'      : bbox[3],
        'theta'       : ibs.get_annot_thetas(aid),
        'class'       : ibs.get_annot_species_texts(aid),
        'score'       : ibs.get_annot_detect_confidence(aid),
    }


def _task(ibs, taskid):
    return {
        'uuid': taskid,
    }


@register_route(_prefix('swagger'), methods=['GET'])
def microsoft_core_specification_swagger(*args, **kwargs):
    r"""
    Returns the API specification in the Swagger 2.0 (OpenAPI) JSON format.
    ---
    definitions:
    - schema:
        id: Image
        required:
          - uuid
        properties:
          uuid:
            description: a deterministically-derived UUID based on the image pixels, which can be used to identify duplicate Images.
            type: string
            format: uuid
    - schema:
        id: Annotation
        required:
          - uuid
        properties:
          uuid:
            description: a deterministically-derived UUID based on the parent image's UUID and the bounding box coordinate (xtl, ytl, width, height) and orientation (theta), which can be used to identify duplicate Annotations.
            type: string
            format: uuid
    - schema:
        id: Detection
        required:
          - _image
          - _annotation
          - score
          - xtl
          - ytl
          - width
          - height
          - theta
          - class
        properties:
          _image:
            description: The Image that this Detection was found in
            $ref: '#/definitions/Image'
          _annotation:
            description: The Annotation that is the permanent record for this Detection
            $ref: '#/definitions/Annotation'
          score:
            description: The detection's classification score
            type: integer
            format: int32
          xtl:
            description: The pixel coordinate for the top-left corner along the x-axis (xtl = x-axis top left) for the bounding box
            type: integer
            format: int32
          ytl:
            description: The pixel coordinate for the top-left corner along the y-axis (ytl = y-axis top left) for the bounding box
            type: integer
            format: int32
          width:
            description: The pixel width for the bounding box
            type: integer
            format: int32
          height:
            description: The pixel height for the bounding box
            type: integer
            format: int32
          class:
            description: The rotation of the bounding box around its center, represented in radians
            type: number
            format: float
          class:
            description: The semantic classification (class label) of the bounding box
            type: string
    - schema:
        id: Task
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
    swag['info']['title'] = 'Wild Me - IA (Image Analysis)'
    swag['info']['description'] = 'Documentation for all classification, detection, and identification calls provided by Wild Me for the AI for Earth (AI4E) collaboration'
    swag['info']['version'] = 'v0.1'
    swag['info']['contact'] = {
        'name':  'Wild Me Developers (AI4E)',
        'url':   'http://wildme.org',
        'email': 'dev@wildme.org',
    }
    swag['info']['license'] = {
        'name': 'Apache 2.0',
        'url':  'http://www.apache.org/licenses/LICENSE-2.0.html'
    }
    swag['host'] = 'demo.wildme.org:6000'
    swag['schemes'] = [
        'http',
    ]
    # swag['basePath'] = PREFIX

    # "securityDefinitions": {
    #   "apiKeyHeader": {
    #     "type": "apiKey",
    #     "name": "Ocp-Apim-Subscription-Key",
    #     "in": "header"
    #   },
    #   "apiKeyQuery": {
    #     "type": "apiKey",
    #     "name": "subscription-key",
    #     "in": "query"
    #   }
    # },
    # "security": [
    #   {
    #     "apiKeyHeader": []
    #   },
    #   {
    #     "apiKeyQuery": []
    #   }
    # ],

    response = jsonify(swag)
    return response


@register_api(_prefix('status'), methods=['GET'], __api_plural_check__=False)
def microsoft_core_status(*args, **kwargs):
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
        example:
        - application/json:
            status: healthy
    """
    status = 'healthy'
    return {'status': status}


@register_api(_prefix('image'), methods=['POST'])
def microsoft_image_upload(ibs, *args, **kwargs):
    r"""
    Returns the available detection models and their supported species
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
    """
    from ibeis.web.apis import image_upload
    try:
        gid = image_upload(cleanup=True, **kwargs)
        assert gid is not None
    except controller_inject.WebException:
        raise
    except:
        raise controller_inject.WebInvalidInput('Uploaded image is corrupted or is an unsupported file format (supported: mage/png, image/jpeg, image/tiff)', 'image')
    return _image(ibs, gid)


@register_api(_prefix('detect'), methods=['GET'])
def microsoft_detect_model(ibs, *args, **kwargs):
    r"""
    Returns the available models and their supported species for detection.
    ---
    produces:
    - application/json
    responses:
      200:
        description: Returns an object of models (keys) and their supported species (values) listed as an array of strings.
        schema:
          type: object
    """
    return ibs.models_cnn_lightnet()


def microsoft_detect_input_validation(model, score_threshold, use_nms, nms_threshold):
    from ibeis.algo.detect.lightnet import CONFIG_URL_DICT
    try:
        parameter = 'model'
        assert model in CONFIG_URL_DICT, 'Specified model is not supported'

        parameter = 'score_threshold'
        assert isinstance(score_threshold, float), 'Score threshold must be a float'
        assert 0.0 <= score_threshold and score_threshold <= 1.0, 'Score threshold is invalid, must be in range [0.0, 1.0]'

        parameter = 'use_nms'
        assert isinstance(use_nms, bool), 'NMS flag must be a boolean'

        parameter = 'nms_threshold'
        assert isinstance(nms_threshold, float), 'NMS threshold must be a float'
        assert 0.0 <= nms_threshold and nms_threshold <= 1.0, 'NMS threshold is invalid, must be in range [0.0, 1.0]'
    except AssertionError as ex:
        raise controller_inject.WebInvalidInput(str(ex), parameter)


@register_ibs_method
def microsoft_detect(ibs, images, model, score_threshold=0.0, use_nms=True,
                     nms_threshold=0.4, __jobid__=None, *args, **kwargs):
    depc = ibs.depc_image

    uuid_list = [
        uuid.UUID(image['uuid'])
        for image in images
    ]
    gid_list = ibs.get_image_gids_from_uuid(uuid_list)

    try:
        config = {
            'algo'            : 'lightnet',
            'config_filepath' : model,
            'weight_filepath' : model,
            'sensitivity'     : score_threshold,
            'nms'             : use_nms,
            'nms_thresh'      : nms_threshold,
        }
        results_list = depc.get_property('localizations', gid_list, None, config=config)
        aids_list = ibs.commit_localization_results(gid_list, results_list)
    except:
        raise controller_inject.WebException('Detection process failed for an unknown reason')

    detections_list = {
        'detections_list': [
            {
                'detections' : [
                    _detection(ibs, aid)
                    for aid in aid_list
                ],
            }
            for aid_list in aids_list
        ]
    }

    return detections_list


@register_api(_prefix('detect'), methods=['POST'])
def microsoft_detect_upload(ibs, model, score_threshold=0.0, use_nms=True,
                            nms_threshold=0.4, *args, **kwargs):
    r"""
    Returns the detection results for an uploaded image and a provided model configuration.
    ---
    parameters:
    - name: image
      in: formData
      description: The image for which detection will be processed.
      required: true
      type: file
      enum:
      - image/png
      - image/jpg
      - image/tiff
    - name: model
      in: body
      description: The model name to use with detection.  Available models can be queried using the API $PREFIX/detect/model/
      required: true
      type: string
    - name: score_threshold
      in: body
      description: A score threshold to filter the detection results
      required: false
      type: number
      format: float
      default: 0.0
      minimum: 0.0
      maximum: 1.0
    - name: use_nms
      in: body
      description: Flag to turn on/off Non-Maximum Suppression (NMS)
      required: false
      type: boolean
      default: true
    - name: nms_threshold
      in: body
      description: A Intersection-over-Union (IoU) threshold to suppress detection results with NMS
      required: false
      type: number
      format: float
      default: 0.0
      minimum: 0.0
      maximum: 1.0
    responses:
      200:
        description: Returns an array of Detection models on the uploaded image
        schema:
          type: array
          items:
            $ref: "#/definitions/Detection"
      400:
        description: Invalid input parameter
    """
    ibs = current_app.ibs

    # Input argument validation
    image = microsoft_image_upload(ibs)

    microsoft_detect_input_validation(model, score_threshold, use_nms, nms_threshold)

    try:
        images = [image]
        detections_list = microsoft_detect(ibs, images, model, score_threshold, use_nms, nms_threshold)
    except:
        raise controller_inject.WebException('Detection process failed for an unknown reason')

    detections_list = detections_list.get('detections_list')
    assert len(detections_list) == 1
    detections = detections_list[0]

    print(detections)
    return detections


@register_api(_prefix('detect/batch'), methods=['POST'])
def microsoft_detect_batch(ibs, images, model, score_threshold=0.0, use_nms=True,
                           nms_threshold=0.4, async=True,
                           callback_url=None, callback_method=None,
                           *args, **kwargs):
    r"""
    The asynchronous variant of POST 'detect' that takes in a list of Image models and returns a task ID
    ---
    parameters:
    - name: images
      in: body
      description: A JSON list of Image models for which detection will be processed.
      required: true
      type: array
      items:
        $ref: '#/definitions/Image'
    - name: model
      in: body
      description: The model name to use with detection.  Available models can be queried using the API $PREFIX/detect/model/
      required: true
      type: string
    - name: score_threshold
      in: body
      description: A score threshold to filter the detection results
      required: false
      type: number
      format: float
      default: 0.0
      minimum: 0.0
      maximum: 1.0
    - name: use_nms
      in: body
      description: Flag to turn on/off Non-Maximum Suppression (NMS)
      required: false
      type: boolean
      default: true
    - name: nms_threshold
      in: body
      description: A Intersection-over-Union (IoU) threshold to suppress detection results with NMS
      required: false
      type: number
      format: float
      default: 0.0
      minimum: 0.0
      maximum: 1.0
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
        description: The task returns an array of arrays of Detection models, in parallel lists with the provided Image models
        schema:
          type: array
          items:
            schema:
              type: array
              items:
                $ref: "#/definitions/Detection"

    """
    ibs = current_app.ibs

    # Input argument validation
    for index, image in enumerate(images):
        try:
            parameter = 'images:%d' % (index, )
            assert 'uuid' in image, 'Image Model provided is invalid, missing UUID key'
        except AssertionError as ex:
            raise controller_inject.WebInvalidInput(str(ex), parameter)

    microsoft_detect_input_validation(model, score_threshold, use_nms, nms_threshold)

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

    args = (images, model, )
    kwargs = {
        'score_threshold': score_threshold,
        'use_nms'        : use_nms,
        'nms_threshold'  : nms_threshold,
    }

    if async:
        taskid = ibs.job_manager.jobiface.queue_job('microsoft_detect',
                                                    callback_url, callback_method,
                                                    *args, **kwargs)
        response = _task(ibs, taskid)
    else:
        response = ibs.microsoft_detect(*args, **kwargs)

    return response


@register_api(_prefix('task'), methods=['GET'])
def microsoft_task_status(ibs, task):
    r"""
    Check the status of an asynchronous Task
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
def microsoft_task_result(ibs, task):
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
