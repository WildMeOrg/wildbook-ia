# -*- coding: utf-8 -*-
"""Dependencies: flask, tornado."""
from __future__ import absolute_import, division, print_function
from ibeis.control import controller_inject
from flask_swagger import swagger
from flask import current_app
from flask import jsonify
import utool as ut
import uuid


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


@register_route(_prefix('core/specification'), methods=['GET'])
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


@register_api(_prefix('core/status'), methods=['GET'], __api_plural_check__=False)
def microsoft_core_status(*args, **kwargs):
    r"""
    Returns the health status of the API back-end, functioning as well as a heat beat.
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


@register_api(_prefix('image/upload'), methods=['POST'])
def microsoft_image_upload(ibs, *args, **kwargs):
    r"""
    Returns the available models and their supported species for detection.
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
    """
    from ibeis.web.apis import image_upload
    try:
        gid = image_upload(cleanup=True, **kwargs)
        assert gid is not None
    except controller_inject.WebException:
        raise
    except:
        raise controller_inject.WebInvalidInput('Uploaded image is corrupted or is an unsupported file format (supported:image/png, image/jpeg, image/tiff)', 'image')
    return _image(ibs, gid)


@register_api(_prefix('detect/model'), methods=['GET'])
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


def microsoft_detect(ibs, images, model, score_threshold, use_nms, nms_threshold):
    from ibeis.algo.detect.lightnet import CONFIG_URL_DICT

    depc = ibs.depc_image

    # Input argument validation
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

    uuid_list = [
        uuid.UUID(image['uuid'])
        for image in images
    ]
    gid_list = ibs.get_image_gids_from_uuid(uuid_list)

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

    return aids_list


@register_api(_prefix('detect/upload'), methods=['POST'])
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
    """
    ibs = current_app.ibs

    # Input validation is done in the next two functions
    image = microsoft_image_upload(ibs)
    images = [image]

    try:
        aids_list = microsoft_detect(ibs, images, model, score_threshold, use_nms, nms_threshold)
    except controller_inject.WebException:
        raise

    assert len(aids_list) == 1
    aid_list = aids_list[0]

    detections = {
        'detections' : [
            _detection(ibs, aid)
            for aid in aid_list
        ],
    }
    return detections


@register_api(_prefix('detect/image'), methods=['POST'])
def microsoft_detect_image(ibs, images, model, score_threshold=0.0, use_nms=True,
                            nms_threshold=0.4, *args, **kwargs):
    r"""
    Returns batched detection results for a list of uploaded Image models and a provided model configuration.
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
    responses:
      200:
        description: Returns an array of arrays of Detection models, in parallel lists with the provided Image models
        schema:
          type: array
          items:
            schema:
              type: array
              items:
                $ref: "#/definitions/Detection"
    """
    ibs = current_app.ibs

    for index, image in enumerate(images):
        try:
            parameter = 'images:%d' % (index, )
            assert 'uuid' in image, 'Image Model provided is invalid, missing key "uuid"'
        except AssertionError as ex:
            raise controller_inject.WebInvalidInput(str(ex), parameter)

    try:
        aids_list = microsoft_detect(ibs, images, model, score_threshold, use_nms, nms_threshold)
    except controller_inject.WebException:
        raise
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
